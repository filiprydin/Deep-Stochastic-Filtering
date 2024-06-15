import sys
import os
import time

import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from dataset_generator import DatasetGenerator
from SDEs.diffusion_sde import DiffusionSDE
from Filters.neural_network import SimpleNN
from Filters.bayesian_filter import BayesianFilter
from Filters.extended_kalman_filter import ExtendedKalmanFilter

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
torch.set_default_dtype(dtype)
print("Running on device:", device, flush=True)
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0), flush=True)

class EBDSFilter(BayesianFilter):
    def __init__(self, sde, times, measurement_indices, parameters, verbose = True):
        self.sde = sde
        self.state_dim = sde.state_dim
        self.meas_dim = sde.meas_dim

        self.parameters = parameters

        self.times = times
        self.n_times = len(times)
        self.measurement_indices = measurement_indices
        self.measurement_times = self.times[measurement_indices]
        self.times = np.round(self.times, 10) # To avoid error when comparing floats
        self.measurement_times = np.round(self.measurement_times, 10)
        self.train_proportion = parameters["Train proportion"]
        self.constant_network_size = parameters["Constant network size"]
        self.re_norm_method = parameters["Renormalisation method"]
        self.validation_partition = parameters["Validation partition"]
        self.remove_extremes = parameters["Remove extremes"]

        self.norm_method = parameters["Normalisation method"]
        self.min_value_integral = parameters["Normalisation range"][0]
        self.max_value_integral = parameters["Normalisation range"][1]
        self.points_per_dim = parameters["Normalisation points"]
        self.mc_n_points = parameters["Normalisation samples"]

        self.parameters["State dim"] = self.sde.state_dim

        self.verbose = verbose

        # For timing report
        self.norm_time = 0
        self.format_time = 0
        self.train_time = 0
        self.sim_time = 0
        self.train_time_tot = 0

        # Number of datapoints between each observation
        self.N = int((self.n_times - 1) / (len(self.measurement_indices) - 1)) 

        self.models = []
        self.input_sizes = []

        # Variables to keep track of whether the models in all timepoints have been trained
        self.all_times_trained = False
        self.filter_times_trained = False

    def _load_training_data(self, data_folder_name):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        data_folder_path = os.path.join(parent_dir, "Data", data_folder_name)

        decoupled_states = np.load(os.path.join(data_folder_path, "shuffled_states.npy"))
        observations = np.load(os.path.join(data_folder_path, "observations.npy"))
        
        training_size = int(decoupled_states.shape[0] * self.train_proportion)
        Validation_size = decoupled_states.shape[0] - training_size

        self.Z_train = np.zeros((self.state_dim, self.n_times, training_size))
        self.Y_train = np.zeros((self.meas_dim, len(self.measurement_indices), training_size))
        self.Z_val = np.zeros((self.state_dim, self.n_times, Validation_size))
        self.Y_val = np.zeros((self.meas_dim, len(self.measurement_indices), Validation_size))

        for i in range(training_size):
            self.Z_train[:,:,i] = np.reshape(decoupled_states[i,:], (-1, self.state_dim)).T
            self.Y_train[:,:,i] = np.reshape(observations[i,:], (-1, self.meas_dim)).T
        
        for i in range(Validation_size):
            self.Z_val[:,:,i] = np.reshape(decoupled_states[training_size + i,:], (-1, self.state_dim)).T
            self.Y_val[:,:,i] = np.reshape(observations[training_size + i,:], (-1, self.meas_dim)).T

    def _format_input(self, input_timestep, n_observations, validation = False):
        if validation:
            Z = self.Z_val
            Y = self.Y_val
        else:
            Z = self.Z_train
            Y = self.Y_train

        n_samples = Z.shape[2]
        z_input = torch.tensor(Z[:, input_timestep, :], dtype = dtype, device = device).view(-1, n_samples).t()
        y_input = torch.tensor(Y[:, :n_observations, :], dtype = dtype, device = device).reshape(n_observations, -1, n_samples).permute(2, 1, 0).reshape(Z.shape[2], -1)

        input = torch.cat((z_input, y_input), dim=1)

        if self.constant_network_size:
            zero_obs = torch.zeros((input.size(dim=0), self.network_size - input.size(dim=1)), device=device)
            input = torch.cat((input, zero_obs), dim=1)

        return input
    
    def _calculate_and_format_target(self, n, timestep, input_timestep, n_observations, deltat, validation = False):

        start_time = time.time()
        previous_input = self._format_input(input_timestep + 1, n_observations, validation)

        Z_input = previous_input[:, 0:self.state_dim]
        Y_input = previous_input[:, self.state_dim:]
        a1, a2, a3, a4, H = self.sde.get_ebds_filter_params(Z_input)

        # Latest available observation
        start_idx = (n_observations - 1) * self.meas_dim 
        end_idx = n_observations * self.meas_dim
        Yk = Y_input[:, start_idx:end_idx].clone()

        if n == 0 and n_observations == 1:
            # Use only prior and update
            likelihood = self._observation_likelihood_oto(Z_input, Yk).view(-1, 1)
            prior_grad, prior_likelihood = self.sde.prior_gradient_torch(Z_input)
            
            expanded_measurement = (Yk - self.sde.measurement_batch_torch(Z_input)).unsqueeze(2)   
            Ht = torch.transpose(H, 1, 2)
            likelihood_grad = (likelihood.t() * torch.einsum('ijk,ikm->ijm', Ht, expanded_measurement).transpose(2, 0)[0]).t() / self.sde.measurement_noise_std**2
            
            total_grad = (prior_grad * likelihood + likelihood_grad * prior_likelihood)
            t1 = prior_likelihood * likelihood
        elif n == 0:
            # Include update step and previous network
            previous_network = self.models[timestep - 1]
            if self.constant_network_size:
                start_idx = (n_observations - 1) * self.meas_dim 
                end_idx = n_observations * self.meas_dim
                Y_input[:, start_idx:end_idx] = torch.zeros(end_idx - start_idx, device=device)
                input_grad = torch.hstack((Z_input, Y_input)).requires_grad_(True)
            else:
                Y_input = Y_input[:, :-self.meas_dim]
                input_grad = torch.hstack((Z_input, Y_input)).requires_grad_(True)

            network_output_grad = previous_network(input_grad)
            network_output = network_output_grad.detach().view(-1, 1)
            likelihood = self._observation_likelihood_oto(Z_input, Yk).view(-1, 1)
            network_grad = torch.autograd.grad(outputs=network_output_grad, inputs=input_grad, 
                                               grad_outputs=torch.ones_like(network_output_grad))[0].detach()[:, :self.state_dim]

            expanded_measurement = (Yk - self.sde.measurement_batch_torch(Z_input)).unsqueeze(2)    
            Ht = torch.transpose(H, 1, 2)
            likelihood_grad = (likelihood.t() * torch.einsum('ijk,ikm->ijm', Ht, expanded_measurement).transpose(2, 0)[0]).t() / self.sde.measurement_noise_std**2

            total_grad = (network_grad * likelihood + likelihood_grad * network_output)
            t1 = network_output * likelihood
        else:
            # Propagate as normal
            previous_network = self.models[timestep - 1]
            input_grad = torch.hstack((Z_input, Y_input)).requires_grad_(True)
            network_output_grad = previous_network(input_grad)

            t1 = network_output_grad.detach().view(-1, 1)
            total_grad = torch.autograd.grad(outputs=network_output_grad, inputs=input_grad, 
                                             grad_outputs=torch.ones_like(network_output_grad))[0].detach()[:, :self.state_dim]

        expanded_grad = total_grad.unsqueeze(2)
        b1 = torch.einsum('ijk,ikm->ijm', a1, expanded_grad).permute(2, 0, 1)[0]
        b1 = torch.sum(b1, dim=1).view(-1, 1)
        b2 = torch.sum(a2, dim=(1, 2)).view(-1, 1) * t1.view(-1, 1) * 0.5
        b3 = -torch.sum(a3, dim=1).view(-1, 1) * t1.view(-1, 1)
        b4 = -2 * torch.einsum('ij,ij->i', (total_grad, a4)).view(-1, 1)

        t2 = (b1 + b2 + b3 + b4) * deltat
        targets = t1 + t2
        self.format_time += time.time() - start_time

        start_time = time.time()
        targets = self._renormalise_targets(targets, n, timestep, Y_input, Yk)
        self.norm_time += time.time() - start_time

        return targets
    
    def _renormalise_targets(self, targets, n, timestep, Y_input, Yk):
        """
        Applies one of five re-normalisation schemes to the targets in a timestep
        The purpose is to avoid the loss of target magnitude over time
        """

        if self.re_norm_method == 0:
            pass
        elif self.re_norm_method == 1:
            # Simple re-normalization after filtering step 
            if n == 0:
                avg_target = torch.sum(targets, dim=0) / targets.size(dim=0)
                targets = targets / avg_target
        elif self.re_norm_method == 2:
            # Simple re-normalization in every step
            avg_target = torch.sum(targets, dim=0) / targets.size(dim=0)
            targets = targets / avg_target
        elif self.re_norm_method == 3:
            # More complex re-normalisation
            if n == 0:
                n_norm_samples = Y_input.size(dim=0) // 10
                if self.norm_method == "Normal":
                    integrals = self._get_filter_integral_normal(n, timestep, Y_input[:n_norm_samples, :], Yk)
                elif self.norm_method == "Importance":
                    integrals = self._get_filter_integral_importance(n, timestep, Y_input[:n_norm_samples, :], Yk)
                else:
                    integrals = self._get_filter_integral(n, timestep, Y_input[:n_norm_samples, :], Yk)
                integral_avg = torch.mean(integrals)
                targets = targets / integral_avg
        elif self.re_norm_method == 4:
            # Target-wise re-normalisation
            if n == 0:
                if self.norm_method == "Normal":
                    integrals = self._get_filter_integral_normal(n, timestep, Y_input, Yk).view(-1,1)
                elif self.norm_method == "Importance":
                    integrals = self._get_filter_integral_importance(n, timestep, Y_input, Yk).view(-1,1)
                elif self.norm_method == "Quadrature":
                    integrals = self._get_filter_integral(n, timestep, Y_input, Yk).view(-1,1)
                targets = torch.div(targets, integrals)
            
        return targets
    
    def _get_filter_integral_importance(self, n, timestep, Y_input, Yk):
        """
        Returns the filter integrals in the given timepoint with the observations. 
        These are needed in re-normalisation schemes 3 and 4
        Importance sampling version.
        Optimized for internal use via _renormalize_targets, for external use see functions below
        Returns:
        integrals, vector with length = number of observation chains given
        """

        # TODO: remove this, should not be needed
        if n != 0:
            raise ValueError("Can only use Monte-carlo normalisation in time with observation")
        
        integrals = torch.zeros(Y_input.size(dim=0), device=device)
        for i in range(Y_input.size(dim=0)):
            observations = Y_input[i,:]
            yk = Yk[i,:]
            repeated_obs = observations.unsqueeze(0).repeat(self.mc_n_points, 1)

            # For fatter tails
            variance_factor = 5

            # Sample
            if self.meas_dim == self.state_dim:
                mean = yk
                cov = variance_factor * torch.eye(self.meas_dim, device=device) * self.sde.measurement_noise_std ** 2
                x = torch.distributions.MultivariateNormal(mean, cov).sample((self.mc_n_points,))
            else:
                ekf = ExtendedKalmanFilter(self.sde, self.times, self.measurement_indices)
                np_observations = observations.detach().cpu().numpy()
                np_yk = yk.detach().cpu().numpy()
                observations_formatted = np.reshape(np_observations, (self.meas_dim, len(np_observations) // self.meas_dim), order='F')
                n_obs = np.sum(self.measurement_indices < timestep)
                if self.constant_network_size:
                    observations_formatted = np.hstack((observations_formatted, np.zeros((self.meas_dim, 1))))
                    observations_formatted[:,n_obs] = np_yk
                else:
                    pad_zeros = len(self.measurement_indices) - n_obs
                    observations_formatted = np.hstack((observations_formatted, np.zeros((self.meas_dim, pad_zeros))))
                    observations_formatted[:,n_obs] = np_yk
                ekf.filter(observations_formatted)
                m = ekf.get_filter_mean(self.times[timestep], observations_formatted)
                P = ekf.get_filter_cov(self.times[timestep], observations_formatted)
                mean = torch.tensor(m, dtype = dtype, device = device)
                cov = 2 * torch.tensor(P, dtype = dtype, device = device)
                x = torch.distributions.MultivariateNormal(mean, cov).sample((self.mc_n_points,))

            update_factor = torch.log(self._observation_likelihood_mto(x, yk))

            # Importance factor
            mvn = torch.distributions.multivariate_normal.MultivariateNormal(loc=mean, covariance_matrix = cov)
            max_batch_size = 10 ** 5
            if x.size(dim = 0) <= max_batch_size: # Encountered Cuda error with too many samples
                importance_correction = mvn.log_prob(x)
            else:
                n_points = x.size(dim=0)
                importance_correction = torch.zeros((n_points, 1), device=device)
                n_iter = n_points // max_batch_size
                for k in range(n_iter):
                    start_idx = max_batch_size * k
                    end_idx = min(max_batch_size * (k + 1), n_points)
                    importance_correction[start_idx:end_idx, :] = mvn.log_prob(x[start_idx:end_idx, :]).view(-1, 1)
                importance_correction = importance_correction.flatten()

            if timestep == 0:
                prior_log_likelihoods = torch.log(self.sde.prior_likelihood_torch(x))
                predictions = torch.exp(prior_log_likelihoods + update_factor - importance_correction)
            elif n == 0:
                # If at a time with an observation, update model output with observation likelihood
                model = self.models[timestep - 1]

                input = torch.cat((x, repeated_obs), dim=1)
                model_log_predictions = torch.log(model(input).detach().flatten())
                predictions = torch.exp(model_log_predictions + update_factor - importance_correction)

            integrals[i] = torch.mean(predictions)
        print(f"Minimum integral: {min(integrals)}")
        return integrals
    
    def _get_filter_integral_normal(self, n, timestep, Y_input, Yk):
        """
        Returns the filter integrals in the given timepoint with the observations. 
        These are needed in re-normalisation schemes 3 and 4
        Monte-Carlo version. Only works when n=0 and measurement is invertible
        Optimized for internal use via _renormalize_targets, for external use see functions below
        Returns:
        integrals, vector with length = number of observation chains given
        """

        if n != 0:
            raise ValueError("Can only use Monte-carlo normalisation in time with observation")
        
        if not self.sde.invertible_meas:
            raise ValueError("Can only use Monte-carlo normalisation when measurement function is invertible")
        
        integrals = torch.zeros(Y_input.size(dim=0), device=device)
        for i in range(Y_input.size(dim=0)):
            observations = Y_input[i,:]
            yk = Yk[i,:]
            repeated_obs = observations.unsqueeze(0).repeat(self.mc_n_points, 1)

            # Sample
            mean = yk
            cov = torch.eye(self.meas_dim, device=device) * self.sde.measurement_noise_std ** 2
            z = torch.distributions.MultivariateNormal(mean, cov).sample((self.mc_n_points,))
            x = self.sde.measurement_inverse_batch(z)

            if timestep == 0:
                prior_likelihoods = self.sde.prior_likelihood_torch(x)
                predictions = prior_likelihoods
            elif n == 0:
                # If at a time with an observation, update model output with observation likelihood
                model = self.models[timestep - 1]

                input = torch.cat((x, repeated_obs), dim=1)
                model_predictions = model(input).detach().flatten()
                predictions = model_predictions

            integrals[i] = torch.mean(predictions)
        return integrals

    def _get_filter_integral(self, n, timestep, Y_input, Yk):
        """
        Returns the filter integrals in the given timepoint with the observations. 
        Quadrature version
        These are needed in re-normalisation schemes 3 and 4
        Optimized for internal use via _renormalize_targets, for external use see functions below
        Returns:
        integrals, vector with length = number of observation chains given
        """

        points = [torch.linspace(self.min_value_integral, self.max_value_integral, self.points_per_dim, device=device) for _ in range(self.state_dim)]
        mesh = torch.meshgrid(*points, indexing='ij')
        x = torch.vstack([dimension.flatten() for dimension in mesh]).T
        element_size = ((self.max_value_integral - self.min_value_integral) / (self.points_per_dim - 1))**self.state_dim
        
        integrals = torch.zeros(Y_input.size(dim=0), device=device)
        for i in range(Y_input.size(dim=0)):
            observations = Y_input[i,:]
            yk = Yk[i,:]

            if timestep == 0:
                # If at first step, form likelihood by prior and observation likelihood
                observation_likelihoods = self._observation_likelihood_mto(x, yk)
                prior_likelihoods = self.sde.prior_likelihood_torch(x)
                predictions = prior_likelihoods * observation_likelihoods
            elif n == 0:
                # If at a time with an observation, update model output with observation likelihood
                model = self.models[timestep - 1]
                repeated_obs = observations.unsqueeze(0).repeat(x.size(0), 1)
                
                input = torch.cat((x, repeated_obs), dim=1)
                model_predictions = model(input).detach().flatten()
                observation_likelihoods = self._observation_likelihood_mto(x, yk)
                predictions = model_predictions * observation_likelihoods
            else:
                # If at a time between observations, simply evaluate the model
                model = self.models[timestep - 1]
                repeated_obs = observations.unsqueeze(0).repeat(x.size(0), 1)
                input = torch.cat((x, repeated_obs), dim=1)
                predictions = model(input).detach().flatten()

            integrals[i] = torch.sum(predictions) * element_size
        return integrals

    def _observation_likelihood_oto(self, x, y):
        """
        Evaluates the multivariate normal pdf with mean h(x) in y
        one-to-one version -> one x point for one y point
        Input:
        x: (num_means, state_dim)
        y: (num_means, meas_dim)
        Output:
        likelihoods (num_means)
        """
        variance = self.sde.measurement_noise_std**2
        #R = np.identity(self.meas_dim) * variance
        means = self.sde.measurement_batch_torch(x)
        sqrt_det_cov = variance**(self.meas_dim/2)
        inv_cov = torch.eye(self.meas_dim, device=device) / variance
        diff = y - means
        exponent = -0.5 * torch.sum(torch.matmul(diff, inv_cov) * diff, dim=1)
        prefactor = 1 / ((2 * np.pi) ** (self.meas_dim / 2) * sqrt_det_cov)
        likelihoods = prefactor * torch.exp(exponent)

        return likelihoods
    
    def _observation_likelihood_mto(self, x, y):
        """
        Evaluates the multivariate normal pdf with mean h(x) in y
        Many-to-one version -> one y point many x points
        Input:
        x: (num_means, state_dim)
        y: (meas_dim)
        Output:
        likelihoods (num_means)
        """
        R = torch.eye(self.meas_dim, device=device) * self.sde.measurement_noise_std**2
        
        # Use symmetry of normal distribution to evaluate one pdf in many points instead of many pdfs in one point
        points = self.sde.measurement_batch_torch(x)
        mean = y
        mvn = torch.distributions.multivariate_normal.MultivariateNormal(loc=mean, covariance_matrix = R)

        # Encountered cuda error with too many samples
        max_batch_size = 10 ** 5
        if x.size(dim = 0) <= max_batch_size:
            likelihood = torch.exp(mvn.log_prob(points))
        else:
            n_points = x.size(dim=0)
            likelihood = torch.zeros((n_points, 1), device=device)
            n_iter = n_points // max_batch_size
            for i in range(n_iter):
                start_idx = max_batch_size * i
                end_idx = min(max_batch_size * (i + 1), n_points)
                likelihood[start_idx:end_idx, :] = torch.exp(mvn.log_prob(points[start_idx:end_idx, :])).view(-1, 1)
            likelihood = likelihood.flatten()

        return likelihood

    def define_and_train_models(self, data_folder_name, n_train_samples):
        start_time_tot = time.time()
        start_time_sim = time.time()
        generator = DatasetGenerator(self.sde, self.times, self.measurement_indices)
        generator.generate_and_save_dataset(n_train_samples, data_folder_name)
        self.sim_time = time.time() - start_time_sim
        self._load_training_data(data_folder_name)

        if self.constant_network_size:
            self.network_size = (len(self.measurement_indices) - 1) * self.meas_dim + self.state_dim

        # Define models
        for j in range(len(self.measurement_indices) - 1):
            for i in range(self.N):
                input_size = (j + 1) * self.meas_dim + self.state_dim
                self.input_sizes.append(input_size)
                if self.constant_network_size:
                    Network = SimpleNN(self.network_size, self.parameters, self.verbose).to(device)
                else:
                    Network = SimpleNN(input_size, self.parameters, self.verbose).to(device)
                self.models.append(Network)

        for k in range(len(self.measurement_indices) - 1):
            for n in range(self.N):
                timestep = k * self.N + n
                input_timestep = timestep # N - (n + 1) or n in derivation
                input_train = self._format_input(input_timestep, k + 1, validation = False)
                input_val = self._format_input(input_timestep, k + 1, validation = True)
                deltat = self.times[timestep + 1] - self.times[timestep]
                target_train = self._calculate_and_format_target(n, timestep, input_timestep, k + 1, deltat, validation = False)
                target_val = self._calculate_and_format_target(n, timestep, input_timestep, k + 1, deltat, validation = True)

                # Partition validation set
                if self.validation_partition:
                    val_per_step = int(target_val.size(dim=0) / self.n_times)
                    first_index = (k*self.N + n) * val_per_step
                    final_index = (k*self.N + n + 1) * val_per_step
                    target_val = target_val[first_index:final_index, :]
                    input_val = input_val[first_index:final_index, :]

                if self.verbose:
                    print("Training model", timestep + 1, "of", self.N * (len(self.measurement_indices) - 1), flush=True)
                elif (timestep + 1) % 10 == 0: 
                    print("Training model", timestep + 1, "of", self.N * (len(self.measurement_indices) - 1), flush=True)

                network = self.models[timestep]
                if self.constant_network_size:
                    if n == 0 and k == 0:
                        pass
                    else: 
                        previous_network = self.models[timestep - 1]
                        network.load_state_dict(previous_network.state_dict())
                else: 
                    if n > 0:
                        previous_network = self.models[timestep - 1]
                        network.load_state_dict(previous_network.state_dict())
                    else:
                        pass

                # Remove 1st and 99th percentile data
                if self.remove_extremes:
                    lower_percentile = torch.kthvalue(target_train, int(0.01 * target_train.size(0)), dim=0).values
                    upper_percentile = torch.kthvalue(target_train, int(0.99 * target_train.size(0)), dim=0).values

                    # Filter the data and labels
                    mask = (target_train >= lower_percentile) & (target_train <= upper_percentile)
                    input_train = input_train[mask.any(dim=1),:]
                    target_train = target_train[mask.any(dim=1)]

                    lower_percentile = torch.kthvalue(target_val, int(0.01 * target_val.size(0)), dim=0).values
                    upper_percentile = torch.kthvalue(target_val, int(0.99 * target_val.size(0)), dim=0).values

                    # Filter the data and labels
                    mask = (target_val >= lower_percentile) & (target_val <= upper_percentile)
                    input_val = input_val[mask.any(dim=1),:]
                    target_val = target_val[mask.any(dim=1)]

                # Remove any inf values that could emerge from failed normalisation (not needed in many cases)
                if torch.any(torch.isinf(target_train)):
                    print("Infinite targets removed")
                    mask1 = torch.isfinite(target_train).all(dim=1)
                    target_train = target_train[mask1]
                    input_train = input_train[mask1,:]
                else:
                    pass
                if torch.any(torch.isinf(target_val)):
                    print("Infinite targets removed")
                    mask2 = torch.isfinite(target_val).all(dim=1)
                    target_val = target_val[mask2]
                    input_val = input_val[mask2,:]
                else:
                    pass

                start_time = time.time()
                network.train_model(input_train, target_train, input_val, target_val)
                self.train_time += time.time() - start_time
                network.eval()

        self.train_time_tot = time.time() - start_time_tot
        
        self.all_times_trained = True
        self.filter_times_trained = True

    def load_saved_models(self, folder_name, only_filter_times = False):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        folder_name = os.path.join(parent_dir, "Models", folder_name)
        if not os.path.exists(folder_name):
            raise ValueError("The specified directory does not exist. Cannot load models")

        specifications_file_path = os.path.join(folder_name, "model_specifications.csv")
        specifications = pd.read_csv(specifications_file_path)

        if self.constant_network_size:
            self.network_size = (len(self.measurement_indices) - 1) * self.meas_dim + self.state_dim

        input_observations = []
        for j in range(len(self.measurement_indices) - 1):
            for i in range(self.N):
                input_size = (j + 1) * self.meas_dim + self.state_dim
                observations = j + 1
                input_observations.append(observations)
                self.input_sizes.append(input_size)

        timepoints = specifications["Timepoint"].to_numpy()
        state_dim = specifications["State dimension"].to_numpy()[0]
        obs_dim = specifications["Observation dimension"].to_numpy()[0]
        n_observations = specifications["Number of observations"].to_numpy()
        constant_size = specifications["Constant network sizes"].to_numpy()
        
        try: 
            model_constants = specifications["Normalisation factor"].to_numpy()
        except:
            model_constants = np.ones(self.n_times - 1)

        tolerance_decimals = 7
        time_consistent = np.all(np.round(timepoints, tolerance_decimals) == np.round(self.times[1:], tolerance_decimals))
        dimensions_consistent = (state_dim == self.state_dim and obs_dim == self.meas_dim)
        observations_consistent = np.all(input_observations == n_observations)
        constant_size_consistent = np.all((constant_size == self.constant_network_size))
        if not (time_consistent and dimensions_consistent and observations_consistent and constant_size_consistent):
            raise ValueError("Saved models are not consistent with setting")

        # Define and load models
        for j in range(self.n_times - 1):
            norm_factor = model_constants[j]
            
            if self.constant_network_size:
                model = SimpleNN(self.network_size, self.parameters, self.verbose, norm_factor).to(device)
            else:
                model = SimpleNN(self.input_sizes[j], self.parameters, self.verbose, norm_factor).to(device)
            
            if not only_filter_times:
                model_path = os.path.join(folder_name, f"Model_{j + 1}.pt")
                model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
            else:
                if (j+1) in self.measurement_indices:
                    model_path = os.path.join(folder_name, f"Model_{j + 1}.pt")
                    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
                else: 
                    pass

            model.eval()
            self.models.append(model)
        
        if only_filter_times:
            self.filter_times_trained = True
        else:
            self.filter_times_trained = True
            self.all_times_trained = True

    def get_filter_pdf(self, timepoint, observations, min_value, max_value, points_per_dim):
        """
        Returns the filter pdf in the grid defined by min_value, max_value and points_per_dim
        Also returns the points flattened to the format (n points, state space dimension)
        TODO: Adapt for non-quadrature methods
        """
        points = [torch.linspace(min_value, max_value, points_per_dim, device=device) for _ in range(self.state_dim)]
        mesh = torch.meshgrid(*points, indexing='ij')
        x = torch.vstack([dimension.flatten() for dimension in mesh]).T
        observations = torch.tensor(observations, dtype=dtype, device=device)
        predictions = self._evaluate_filter_nn(x, timepoint, observations)
        element_size = ((max_value - min_value) / (points_per_dim - 1))**self.state_dim
        integral = torch.sum(predictions) * element_size

        return (predictions / integral).detach().cpu().numpy(), x.detach().cpu().numpy()
    
    def get_filter_pdfs(self, timepoints, observations, min_value, max_value, points_per_dim):
        """
        Returns the filter pdfs in the grid defined by min_value, max_value and points_per_dim
        Also returns the points flattened to the format (n points, state space dimension)
        TODO: Adapt for non-quadrature methods
        """
        points = [torch.linspace(min_value, max_value, points_per_dim, device=device) for _ in range(self.state_dim)]
        mesh = torch.meshgrid(*points, indexing='ij')
        x = torch.vstack([dimension.flatten() for dimension in mesh]).T
        observations = torch.tensor(observations, dtype=dtype, device=device)
        element_size = ((max_value - min_value) / (points_per_dim - 1))**self.state_dim

        values = torch.zeros((x.shape[0], len(timepoints)), device=device)
        for i in range(len(timepoints)):
            predictions = self._evaluate_filter_nn(x, timepoints[i], observations)
            integral = torch.sum(predictions) * element_size
            values[:,i] = predictions / integral

        return values.detach().cpu().numpy(), x.detach().cpu().numpy() 
    
    def get_filter_pdf_and_integral(self, timepoint, observations, min_value, max_value, points_per_dim):
        """
        Returns the filter pdf in the grid defined by min_value, max_value and points_per_dim
        Also returns the points flattened to the format (n points, state space dimension) and the integral
        TODO: Adapt for non-quadrature methods
        """
        points = [torch.linspace(min_value, max_value, points_per_dim, device=device) for _ in range(self.state_dim)]
        mesh = torch.meshgrid(*points, indexing='ij')
        x = torch.vstack([dimension.flatten() for dimension in mesh]).T
        observations = torch.tensor(observations, dtype=dtype, device=device)
        predictions = self._evaluate_filter_nn(x, timepoint, observations)
        element_size = ((max_value - min_value) / (points_per_dim - 1))**self.state_dim
        integral = torch.sum(predictions) * element_size

        return (predictions / integral).detach().cpu().numpy(), x.detach().cpu().numpy(), integral.detach().cpu().numpy() 
    
    def get_all_filter_pdfs(self, observations, min_value, max_value, points_per_dim):
        """
        TODO: Adapt for non-quadrature methods
        """

        points = [torch.linspace(min_value, max_value, points_per_dim, device=device) for _ in range(self.state_dim)]
        mesh = torch.meshgrid(*points, indexing='ij')
        x = torch.vstack([dimension.flatten() for dimension in mesh]).T
        observations = torch.tensor(observations, dtype=dtype, device=device)
        element_size = ((max_value - min_value) / (points_per_dim - 1))**self.state_dim

        values = torch.zeros((x.shape[0], self.n_times), device=device)
        for i in range(self.n_times):
            predictions = self._evaluate_filter_nn(x, self.times[i], observations)
            integral = torch.sum(predictions) * element_size
            values[:,i] = predictions / integral

        return values.detach().cpu().numpy(), x.detach().cpu().numpy() 

    def evaluate_filter_pdf(self, x, timepoint, observations):
        """
        Evaluates the filter pdf at chosen timepoint in chosen point. 
        Input: 
        timepoint: must be among sampled timepoints
        x: matrix with dimensions (number of points, state space dimension)
        observations: must contain as many or more observations as the input requires at the timestep
        with dimensions (number of observations, observation dimensions)
        TODO: Adapt for non-quadrature methods
        """
        observations = torch.tensor(observations, dtype=dtype, device=device)
        x = torch.tensor(x, dtype=dtype, device=device)

        predictions = self._evaluate_filter_nn(x, timepoint, observations)

        # Get constant
        points = [torch.linspace(self.min_value_integral, self.max_value_integral, self.points_per_dim, device=device) for _ in range(self.state_dim)]
        mesh = torch.meshgrid(*points, indexing='ij')
        x = torch.vstack([dimension.flatten() for dimension in mesh]).T
        constant_predictions = self.evaluate_filter_nn(x, timepoint, observations)
        element_size = ((self.max_value_integral - self.min_value_integral) / (self.points_per_dim - 1))**self.state_dim
        integral = torch.sum(constant_predictions) * element_size

        return (predictions / integral).detach().cpu().numpy()
    
    def evaluate_filter_pdf_unnorm(self, x, timepoint, observations):
        """
        Evaluates the unnormalised filter pdf at chosen timepoint in chosen point. 
        Input: 
        timepoint: must be among sampled timepoints
        x: matrix with dimensions (number of points, state space dimension)
        observations: must contain as many or more observations as the input requires at the timestep
        with dimensions (number of observations, observation dimensions)
        """
        observations = torch.tensor(observations, dtype=dtype, device=device)
        x = torch.tensor(x, dtype=dtype, device=device)
        predictions = self._evaluate_filter_nn(x, timepoint, observations)
        return predictions.detach().cpu().numpy()
    
    def get_filter_mean(self, timepoint, observations):
        """
        TODO: This should probably be runable without quadrature
        Returns the mean of the distribution at chosen timepoint given the observations
        """
        points = [torch.linspace(self.min_value_integral, self.max_value_integral, self.points_per_dim, device=device) for _ in range(self.state_dim)]
        mesh = torch.meshgrid(*points, indexing='ij')
        x = torch.vstack([dimension.flatten() for dimension in mesh]).T
        observations = torch.tensor(observations, dtype=dtype, device=device)
        predictions = self._evaluate_filter_nn(x, timepoint, observations).detach().cpu().numpy()
        mean = np.average(x, axis=0, weights=predictions)
        return mean
    
    def get_filter_means(self, timepoints, observations):
        """
        TODO: Adapt for non-quadrature methods
        Returns the means of the distributions at chosen timepoints given the observations
        """

        points = [torch.linspace(self.min_value_integral, self.max_value_integral, self.points_per_dim, device=device) for _ in range(self.state_dim)]
        mesh = torch.meshgrid(*points, indexing='ij')
        x = torch.vstack([dimension.flatten() for dimension in mesh]).T
        observations = torch.tensor(observations, dtype=dtype, device=device)
        means = torch.zeros((self.state_dim, len(timepoints)), device=device)
        for i in range(len(timepoints)):
            predictions = self._evaluate_filter_nn(x, timepoints[i], observations)
            means[:,i] = torch.sum(x * predictions.unsqueeze(1), dim=0) / torch.sum(predictions)
        return means.detach().cpu().numpy()
    
    def get_filter_means_and_integrals(self, timepoints, observations):
        """
        TODO: Should probably be runable with quadrature
        Returns the means of the distributions at chosen timepoints given the observations
        Also returns the normalising constants
        """

        torch_observations = torch.tensor(observations, dtype=dtype, device=device)
        means = torch.zeros((self.state_dim, len(timepoints)), device=device)
        integrals = torch.zeros(len(timepoints), device=device)
        timepoints = np.round(timepoints, 10)
        for i in range(len(timepoints)):
            timestep = np.where(self.times == timepoints[i])[0][0]
            if timepoints[i] not in self.measurement_times:
                raise ValueError("Can only use Monte-carlo normalisation in time with observation")
            
            n_observations = timestep // self.N + 1
            observations_formatted = torch_observations[:(n_observations-1), :].flatten().repeat(self.mc_n_points, 1)
            if self.constant_network_size:
                n_zeros = self.network_size - self.state_dim - (n_observations - 1) * self.meas_dim
                observations_formatted = torch.hstack((observations_formatted, torch.zeros(self.mc_n_points, n_zeros, device=device)))

            yk = torch_observations[n_observations - 1, :].clone()

            if self.norm_method == "Normal":
                if not self.sde.invertible_meas:
                    raise ValueError("Can only use Monte-carlo normalisation when measurement function is invertible")
                
                # Sample
                mean = yk
                cov = torch.eye(self.meas_dim, device=device) * self.sde.measurement_noise_std ** 2
                z = torch.distributions.MultivariateNormal(mean, cov).sample((self.mc_n_points,))
                x = self.sde.measurement_inverse_batch(z)

                if timestep == 0:
                    prior_likelihoods = self.sde.prior_likelihood_torch(x)
                    predictions = prior_likelihoods
                else:
                    model = self.models[timestep - 1]
                    input = torch.cat((x, observations_formatted), dim=1)
                    model_predictions = model(input).detach().flatten()
                    predictions = model_predictions

            elif self.norm_method == "Importance":
                # For fatter tails
                variance_factor = 5

                # Sample
                if self.meas_dim == self.state_dim:
                    mean = yk
                    cov = variance_factor * torch.eye(self.meas_dim, device=device) * self.sde.measurement_noise_std ** 2
                    x = torch.distributions.MultivariateNormal(mean, cov).sample((self.mc_n_points,))
                else:
                    ekf = ExtendedKalmanFilter(self.sde, self.times, self.measurement_indices)
                    ekf.filter(observations.T)
                    m = ekf.get_filter_mean(timepoints[i], observations)
                    P = ekf.get_filter_cov(timepoints[i], observations)
                    mean = torch.tensor(m, dtype = dtype, device = device)
                    cov = 2 * torch.tensor(P, dtype = dtype, device = device)
                    x = torch.distributions.MultivariateNormal(mean, cov).sample((self.mc_n_points,))

                update_factor = torch.log(self._observation_likelihood_mto(x, yk))

                # Importance factor
                mvn = torch.distributions.multivariate_normal.MultivariateNormal(loc=mean, covariance_matrix = cov)
                max_batch_size = 10 ** 5
                if x.size(dim = 0) <= max_batch_size: # Encountered Cuda error with too many samples
                    importance_correction = mvn.log_prob(x)
                else:
                    n_points = x.size(dim=0)
                    importance_correction = torch.zeros((n_points, 1), device=device)
                    n_iter = n_points // max_batch_size
                    for k in range(n_iter):
                        start_idx = max_batch_size * k
                        end_idx = min(max_batch_size * (k + 1), n_points)
                        importance_correction[start_idx:end_idx, :] = mvn.log_prob(x[start_idx:end_idx, :]).view(-1, 1)
                    importance_correction = importance_correction.flatten()

                if timestep == 0:
                    prior_log_likelihoods = torch.log(self.sde.prior_likelihood_torch(x))
                    predictions = torch.exp(prior_log_likelihoods + update_factor - importance_correction)
                elif timestep in self.measurement_indices:
                    model = self.models[timestep - 1]
                    input = torch.cat((x, observations_formatted), dim=1)
                    model_log_predictions = torch.log(model(input).detach().flatten())
                    predictions = torch.exp(model_log_predictions + update_factor - importance_correction)

            integrals[i] = torch.mean(predictions)
            means[:, i] = torch.sum(x * predictions.unsqueeze(1).repeat(1, self.state_dim), dim=0) / (integrals[i] * self.mc_n_points)

        return means.detach().cpu().numpy(), integrals.detach().cpu().numpy()

    def get_all_filter_means(self, observations):
        """
        TODO: Adapt for non-quadrature methods
        """
        
        points = [torch.linspace(self.min_value_integral, self.max_value_integral, self.points_per_dim, device=device) for _ in range(self.state_dim)]
        mesh = torch.meshgrid(*points, indexing='ij')
        x = torch.vstack([dimension.flatten() for dimension in mesh]).T
        observations = torch.tensor(observations, dtype=dtype, device=device)
        means = torch.zeros((self.state_dim, self.n_times), device=device)
        for i in range(self.n_times):
            predictions = self._evaluate_filter_nn(x, self.times[i], observations)
            means[:,i] = torch.sum(x * predictions.unsqueeze(1), dim=0) / torch.sum(predictions)
        return means.detach().cpu().numpy()

    def _evaluate_filter_nn(self, x, timepoint, observations):
        """
        Evaluates the Neural Network at chosen timepoint in the chosen points. Note: This is the pdf up to a constant.
        Input: 
        timepoint: must be among sampled timepoints
        x: matrix with dimensions (number of points, state space dimension)
        observations: must contain as many or more observations as the input requires at the timestep
        with dimensions (number of observations, observation dimensions)
        """

        tolerance = 0.00001
        idx = (np.abs(self.times - timepoint)).argmin()
        if np.abs(self.times[idx] - timepoint) > tolerance:
            raise ValueError("Timepoint not among sampled times")

        if idx != 0:
            input_observations = int((self.input_sizes[idx - 1] - self.state_dim) / self.meas_dim)
            if idx in self.measurement_indices:
                input_observations += 1
        else:
            input_observations = 1

        if observations.shape[0] < input_observations:
            raise ValueError(f"{input_observations} observations needed, only {observations.shape[0]} given")
        
        if idx in self.measurement_indices:
            if not self.filter_times_trained:
                raise ValueError(f"Filter not properly initialised for evaluation in index {idx}")
        else: 
            if not self.all_times_trained:
                raise ValueError(f"Filter not properly initialised for evaluation in index {idx}")

        if idx == 0:
            # If at first step, form likelihood by prior and observation likelihood
            y0 = observations[0, :]
            observation_likelihoods = self._observation_likelihood_mto(x, y0)
            prior_likelihoods = self.sde.prior_likelihood_torch(x)
            predictions = prior_likelihoods * observation_likelihoods
        elif idx in self.measurement_indices:
            # If at a time with an observation, update model output with observation likelihood
            observations_formatted = observations[:(input_observations-1), :].flatten().repeat(x.size(dim=0), 1)
            if self.constant_network_size:
                n_zeros = self.network_size - self.state_dim - (input_observations - 1) * self.meas_dim
                observations_formatted = torch.hstack((observations_formatted, torch.zeros(x.size(dim=0), n_zeros, device=device)))
            model = self.models[idx - 1]
            input = torch.hstack((x, observations_formatted))
            if input.dim() == 1:
                input = input.unsqueeze(0)
            model_predictions = model(input).detach().flatten()
            yk = observations[input_observations - 1, :]
            observation_likelihoods = self._observation_likelihood_mto(x, yk)
            predictions = model_predictions * observation_likelihoods
        else:
            # If at a time between observations, simply evaluate the model
            observations_formatted = observations[:input_observations, :].flatten().repeat(x.size(dim=0), 1)
            if self.constant_network_size:
                n_zeros = self.network_size - self.state_dim - input_observations * self.meas_dim
                observations_formatted = torch.hstack((observations_formatted, torch.zeros(x.size(dim=0), n_zeros, device=device)))
            model = self.models[idx - 1]
            input = torch.hstack((x, observations_formatted))
            if input.dim() == 1:
                input = input.unsqueeze(0)
            predictions = model(input).detach().flatten()

        return predictions
    
    def sample_distributions(self, timepoints, observations, n_samples):
        # TODO implement

        return 0, 0

    def save_models(self, folder_name, only_filter_times = False):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        folder_name = os.path.join(parent_dir, "Models", folder_name)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        specifications_file_path = os.path.join(folder_name, "model_specifications.csv")
        specifications = pd.DataFrame(columns = ["Model id", "Timepoint", "State dimension", "Observation dimension", "Number of observations", "Constant network sizes", "Normalisation factor"])

        timing_report_path = os.path.join(folder_name, "timing_report.txt")
        with open(timing_report_path, 'w') as file:
            file.write(f"Total training time: {self.train_time_tot}\n")
            file.write(f"Renormalisation time: {self.norm_time}\n")
            file.write(f"Network training time: {self.train_time}\n")
            file.write(f"Target calculation time: {self.format_time}\n")
            file.write(f"Data simulation time: {self.sim_time}")

        for i in range(len(self.models)):
            if not only_filter_times:
                model_path = os.path.join(folder_name, f"Model_{i + 1}.pt")
                torch.save(self.models[i].state_dict(), model_path)
            else:
                if (i+1) in self.measurement_indices:
                    model_path = os.path.join(folder_name, f"Model_{i + 1}.pt")
                    torch.save(self.models[i].state_dict(), model_path)
                else:
                    pass

            specifications_row = {
                "Model id": i + 1, 
                "Timepoint": self.times[i + 1], 
                "State dimension": self.state_dim, 
                "Observation dimension": self.meas_dim, 
                "Number of observations": int((self.input_sizes[i] - self.state_dim) / self.meas_dim),
                "Constant network sizes": self.constant_network_size,
                "Normalisation factor": self.models[i].norm_factor.item()
            }
            specifications.loc[len(specifications)] = specifications_row

        specifications.to_csv(specifications_file_path, index=False)
