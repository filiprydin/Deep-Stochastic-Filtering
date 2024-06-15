import math

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats import gaussian_kde

from Filters.bayesian_filter import BayesianFilter

# Defines a bootstrap particle filter, which allows for dynamic filtering of nonlinear models

class ParticleFilter(BayesianFilter):
    def __init__(self, sde, times, measurement_indices, n_particles):
        self.sde = sde
        self.state_dim = sde.state_dim
        self.meas_dim = sde.meas_dim
        self.times = times
        self.n_times = len(times)
        self.measurement_indices = measurement_indices
        self.n_particles = n_particles

    def filter(self, observations):
        self.states = np.zeros((self.n_particles, self.state_dim, self.n_times), dtype=np.float32)
        self.weights = np.zeros((self.n_particles, self.n_times), dtype=np.float32)

        # At time t_0
        self.states[:,:,0] = self.sde.simulate_starting_state(self.n_particles)
        self.weights[:,0] = self._calculate_weights(self.states[:,:,0], observations[:,0])

        for k in range(self.n_times - 1):
            if k in self.measurement_indices:
                x = self._resample(self.states[:,:,k], self.weights[:,k])
            else:
                x = self.states[:,:,k]

            deltat = self.times[k+1] - self.times[k]
            deltaW = np.random.normal(0, math.sqrt(deltat), (self.n_particles, self.state_dim))
            drift_term = self.sde.drift_batch(x) * deltat
            diffusion_term = np.einsum('ijk,ik->ij', self.sde.diffusion_batch(x), deltaW)
            self.states[:,:,k+1] = x + drift_term + diffusion_term

            if (k + 1) in self.measurement_indices:
                idx = np.where(self.measurement_indices == (k+1))[0][0]
                self.weights[:,k+1] = self._calculate_weights(self.states[:,:,k+1], observations[:,idx])
            else:
                self.weights[:,k+1] = np.ones(self.n_particles)/self.n_particles 

        return self.states, self.weights
    
    def _resample(self, states, weights):
        sampled_indices = np.random.choice(states.shape[0], size=states.shape[0], p=weights)
        sampled_states = states[sampled_indices, :]
        return sampled_states
    
    def _calculate_weights(self, states, observations):
        log_likelihoods = self._observation_loglikelihood_mto(states, observations)
        weights = np.exp(log_likelihoods - np.max(log_likelihoods))
        return weights / np.sum(weights)
    
    def _observation_loglikelihood_mto(self, x, y):
        """
        Evaluates the multivariate normal pdf with mean h(x) in y
        Many-to-one version -> one y point many x points
        Input:
        x: (num_means, state_dim)
        y: (meas_dim)
        Output:
        likelihoods (num_means)
        """
        R = np.identity(self.meas_dim) * self.sde.measurement_noise_std**2
        
        # Use symmetry of normal distribution to evaluate one pdf in many points instead of many pdfs in one point
        points = self.sde.measurement_batch(x)
        mean = y
        mvn = multivariate_normal(mean=mean, cov=R)
        log_likelihoods = mvn.logpdf(points)

        return log_likelihoods
    
    def build_kdes(self, timepoints):
        """
        To improve performance downstream, this function prebuilds kde objects in desired timepoints
        """
        self.kdes = {}

        for time in timepoints:
            idx = np.where(self.times == time)[0][0]
            kde = gaussian_kde(self.states[:,:,idx].T, weights = self.weights[:,idx])
            self.kdes[time] = kde
    
    def get_filter_pdf(self, timepoint, observations, min_value, max_value, points_per_dim):
        """
        Returns the filter pdf in the grid defined by min_value, max_value and points_per_dim
        Also returns the points flattened to the format (n points, state space dimension)
        """
        points = [np.linspace(min_value, max_value, points_per_dim) for _ in range(self.state_dim)]
        mesh = np.meshgrid(*points, indexing='ij')
        x = np.vstack([dimension.ravel() for dimension in mesh]).T

        predictions = self.evaluate_filter_pdf(x, timepoint, observations)

        return predictions, x
    
    def get_filter_pdfs(self, timepoints, observations, min_value, max_value, points_per_dim):
        """
        Returns the filter pdfs in the grid defined by min_value, max_value and points_per_dim
        Also returns the points flattened to the format (n points, state space dimension)
        """
        points = [np.linspace(min_value, max_value, points_per_dim) for _ in range(self.state_dim)]
        mesh = np.meshgrid(*points, indexing='ij')
        x = np.vstack([dimension.ravel() for dimension in mesh]).T

        values = np.zeros((x.shape[0], len(timepoints)))
        for i in range(len(timepoints)):
            values[:,i] = self.evaluate_filter_pdf(x, timepoints[i], observations)

        return values, x

    def get_filter_pdfs2(self, timepoints, observations, min_value, max_value, points_per_dim):
        """
        Returns the filter pdfs in the grid defined by min_value, max_value and points_per_dim
        Also returns the points flattened to the format (n points, state space dimension)
        """
        from multiprocessing import Pool

        points = [np.linspace(min_value, max_value, points_per_dim) for _ in range(self.state_dim)]
        mesh = np.meshgrid(*points, indexing='ij')
        x = np.vstack([dimension.ravel() for dimension in mesh]).T

        def evaluate_timepoint(i):
            return self.evaluate_filter_pdf(x, timepoints[i], observations)

        with Pool(processes=len(timepoints)) as pool:
            values = np.array(pool.map(evaluate_timepoint, range(len(timepoints))))

        return values.T, x
    
    def get_all_filter_pdfs(self, observations, min_value, max_value, points_per_dim):
        points = [np.linspace(min_value, max_value, points_per_dim) for _ in range(self.state_dim)]
        mesh = np.meshgrid(*points, indexing='ij')
        x = np.vstack([dimension.ravel() for dimension in mesh]).T

        values = np.zeros((x.shape[0], self.n_times))
        for i in range(self.n_times):
            values[:,i] = self.evaluate_filter_pdf(x, self.times[i], observations)

        return values, x
    
    def sample_distributions(self, timepoints, observations, n_samples):
        samples = np.zeros((self.state_dim, n_samples, len(timepoints)))
        values = np.zeros((n_samples, len(timepoints)))
        for i in range(len(timepoints)):
            idx = np.where(self.times == timepoints[i])[0][0]
            states = self.states[:,:,idx]
            weights = self.weights[:,idx]

            sampled_indices = np.random.choice(states.shape[0], size=n_samples, p=weights)
            samples[:,:,i] = states[sampled_indices, :].T

            values[:,i] = self.evaluate_filter_pdf(samples[:,:,i].T, timepoints[i], observations)
        return samples, values

    def get_filter_mean(self, timepoint, observations):
        """
        Returns the mean of the distribution at chosen timepoint
        """
        tolerance = 0.00001
        idx = (np.abs(self.times - timepoint)).argmin()
        if np.abs(self.times[idx] - timepoint) > tolerance:
            raise ValueError("Timepoint not among sampled times")

        states = self.states[:,:,idx]
        weights = self.weights[:,idx]
        mean = np.average(states, axis=0, weights=weights)
        return mean
    
    def get_filter_means(self, timepoints, observations):
        """
        Returns the means of the distribution at chosen timepoints
        """
        tolerance = 0.00001

        times = np.repeat(self.times[np.newaxis, :], len(timepoints), axis=0)
        idxs = np.abs(times - timepoints.reshape(-1,1)).argmin(axis=1)
        values = np.abs(times - timepoints.reshape(-1,1)).min(axis=1)
        if np.any(values) > tolerance:
            raise ValueError("Timepoint not among sampled times")

        states = self.states[:,:,idxs]
        weights = self.weights[:,idxs]
        weights_new = np.repeat(weights[:, np.newaxis, :], self.state_dim, axis=1)
        means = np.average(states, axis=0, weights=weights_new)
        return means
    
    def get_all_filter_means(self, observations):
        weights_new = np.repeat(self.weights[:, np.newaxis, :], self.state_dim, axis=1)
        means = np.average(self.states, axis=0, weights=weights_new)
        return means

    def evaluate_filter_pdf(self, x, timepoint, observations):
        """
        Evaluates the pdf of the filter at chosen timepoint in the chosen points.
        Input: 
        timepoint: must be among sampled timepoints
        x: matrix with dimensions (number of points, state space dimension)
        """
        tolerance = 0.00001
        idx = (np.abs(self.times - timepoint)).argmin()
        if np.abs(self.times[idx] - timepoint) > tolerance:
            raise ValueError("Timepoint not among sampled times")
        
        if timepoint in self.kdes:
            # Use pre-built kde if available
            return self.kdes[timepoint].pdf(x.T)
        else:
            kde = gaussian_kde(self.states[:,:,idx].T, weights = self.weights[:,idx])
            predictions = kde.pdf(x.T)
            return predictions
    
    def plot_states_and_estimates(self, state_trajectories):
        plt.figure(figsize=(8, 6))
        color_map = plt.get_cmap('tab10')

        # Estimate means
        means = np.zeros((self.state_dim, self.n_times))
        for i in range(self.n_times):
            means[:,i] = self.get_filter_mean(self.times[i], np.zeros((len(self.measurement_indices), self.meas_dim)))

        for i in range(self.state_dim):
            color = color_map(i % 10)
            plt.plot(self.times, state_trajectories[i, :], label=f'State {i + 1}', color = color)
            plt.scatter(self.times[self.measurement_indices], means[i, self.measurement_indices], s = 20, color = color, edgecolors="black")

        plt.title('Simulated paths and kalman filter estimates')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()