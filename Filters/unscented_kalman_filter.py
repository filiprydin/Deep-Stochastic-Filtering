import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
import scipy.linalg as linalg

from Filters.bayesian_filter import BayesianFilter

# Defines an unscented kalman filter, which allows for dynamic filtering of nonlinear models

class UnscentedKalmanFilter(BayesianFilter):
    def __init__(self, sde, times, measurement_indices):
        self.sde = sde
        self.state_dim = sde.state_dim
        self.meas_dim = sde.meas_dim
        self.times = times
        self.n_times = len(times)
        self.measurement_indices = measurement_indices
        self.m0, self.P0 = sde.get_prior_moments()

        # Parameters
        self.nprime = 2 * self.state_dim
        self.alpha = 1e-3 # See https://groups.seas.harvard.edu/courses/cs281/papers/unscented.pdf
        self.kappa = 0
        self.beta = 2
        self.lambda_param = (self.alpha**2) * (self.nprime + self.kappa) - self.nprime
        self.nbis = self.state_dim + self.meas_dim
        self.lambda_param_m = (self.alpha**2) * (self.nbis + self.kappa) - self.nbis

    def filter(self, observations):
        self.observations = observations
        self.m = np.zeros((self.state_dim, self.n_times))
        self.P = np.zeros((self.state_dim, self.state_dim, self.n_times))

        R = self.sde.get_unscented_kalman_filter_update_params()

        # At time t_0
        m, P = self._unscented_transform_update(self.m0, self.P0, observations[:,0], R)

        self.m[:,0] = m
        self.P[:,:,0] = P

        for k in range(0, self.n_times-1):
            # Prediction step
            deltat = self.times[k + 1] - self.times[k]
            Qk = self.sde.get_unscented_kalman_filter_pred_params(deltat)
            mk = self.m[:,k]
            Pk = self.P[:,:,k]
            
            mm, Pm = self._unscented_transform_prediction(mk, Pk, deltat, Qk)

            # Update step
            if (k + 1) in self.measurement_indices:
                idx = np.where(self.measurement_indices == k+1)[0][0]
                m, P = self._unscented_transform_update(mm, Pm, observations[:,idx], R)

                self.m[:,k+1] = m
                self.P[:,:,k+1] = P
            else:
                self.m[:,k+1] = mm
                self.P[:,:,k+1] = Pm

    def _unscented_transform_update(self, mm, Pm, observation, R):
        # Form sigma points
        mm_aug = np.concatenate((mm, np.zeros(self.meas_dim)))
        Pm_aug = np.vstack((np.hstack((Pm, np.zeros((self.state_dim, self.meas_dim)))), np.hstack((np.zeros((self.meas_dim, self.state_dim)), R))))
        sqrt_Pm_aug = linalg.sqrtm(Pm_aug)
        sigma_points_m = np.zeros((self.nbis, 2*self.nbis + 1))
        sigma_points_m[:,0] = mm_aug
        for i in range(self.nbis):
            sigma_points_m[:,i+1] = mm_aug + np.sqrt(self.nbis + self.lambda_param_m)*sqrt_Pm_aug[:,i]
            sigma_points_m[:,self.nbis+i+1] = mm_aug - np.sqrt(self.nbis + self.lambda_param_m)*sqrt_Pm_aug[:,i]
        # Propagate
        prop_sigma_points_m = np.zeros((self.meas_dim, 2*self.nbis + 1))
        for i in range(2*self.nbis + 1):
            x = sigma_points_m[:self.state_dim,i]
            r = sigma_points_m[self.state_dim:,i]
            prop_sigma_points_m[:,i] = self.sde.measurement(x) + r
        # Predicted mean, covariance and cross-covariance
        m_measure = np.zeros(self.meas_dim)
        W_mean = self.lambda_param_m/(self.nbis + self.lambda_param_m)
        m_measure += W_mean * prop_sigma_points_m[:,0]
        for i in range(1, 2*self.nbis + 1):
            W_mean = 1/(2*(self.nbis + self.lambda_param_m))
            m_measure += W_mean * prop_sigma_points_m[:,i]
        S = np.zeros((self.meas_dim, self.meas_dim))
        C = np.zeros((self.state_dim, self.meas_dim))
        W_cov = self.lambda_param_m/(self.nbis + self.lambda_param_m) + (1 - self.alpha**2 + self.beta)
        diff1 = prop_sigma_points_m[:,0] - m_measure
        diff2 = sigma_points_m[:self.state_dim,0] - mm
        S += W_cov * np.outer(diff1, diff1)
        C += W_cov * np.outer(diff2, diff1)
        for i in range(1, 2*self.nbis + 1):
            W_cov = 1/(2*(self.nbis + self.lambda_param_m))
            diff1 = prop_sigma_points_m[:,i] - m_measure
            diff2 = sigma_points_m[:self.state_dim,i] - mm
            S += W_cov * np.outer(diff1, diff1)
            C += W_cov * np.outer(diff2, diff1)
        # Conditional distribution of state
        K = np.matmul(C, np.linalg.inv(S))
        m = mm + np.matmul(K, observation - m_measure)
        P = Pm - np.matmul(K, np.matmul(S, np.transpose(K)))

        return m, P

    def _unscented_transform_prediction(self, mk, Pk, deltat, Qk):
        # Form sigma points
        mk_aug = np.concatenate((mk, np.zeros(self.state_dim)))
        Pk_aug = np.vstack((np.hstack((Pk, np.zeros_like(Pk))), np.hstack((np.zeros_like(Pk), Qk))))
        sqrt_Pk_aug = linalg.sqrtm(Pk_aug)
        sigma_points = np.zeros((self.nprime, 2*self.nprime + 1))
        sigma_points[:,0] = mk_aug
        for i in range(self.nprime):
            sigma_points[:,i+1] = mk_aug + np.sqrt(self.nprime + self.lambda_param)*sqrt_Pk_aug[:,i]
            sigma_points[:,self.nprime+i+1] = mk_aug - np.sqrt(self.nprime + self.lambda_param)*sqrt_Pk_aug[:,i]
        # Propagate
        prop_sigma_points = np.zeros((self.state_dim, 2*self.nprime + 1))
        for i in range(2*self.nprime + 1):
            x = sigma_points[:self.state_dim,i]
            q = sigma_points[self.state_dim:,i]
            prop_sigma_points[:,i] = x + self.sde.drift(x) * deltat + np.matmul(self.sde.diffusion(x), q)
        # Predicted mean and covariance
        mm = np.zeros(self.state_dim)
        W_mean = self.lambda_param/(self.nprime + self.lambda_param)
        mm += W_mean * prop_sigma_points[:,0]
        for i in range(1, 2*self.nprime + 1):
            W_mean = 1/(2*(self.nprime + self.lambda_param))
            mm += W_mean * prop_sigma_points[:,i]
        Pm = np.zeros((self.state_dim, self.state_dim))
        W_cov = self.lambda_param/(self.nprime + self.lambda_param) + (1 - self.alpha**2 + self.beta)
        diff = prop_sigma_points[:,0] - mm
        Pm += W_cov * np.outer(diff, diff)
        for i in range(1, 2*self.nprime + 1):
            W_cov = 1/(2*(self.nprime + self.lambda_param))
            diff = prop_sigma_points[:,i] - mm
            Pm += W_cov * np.outer(diff, diff)
        
        return mm, Pm

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
    
    def get_all_filter_pdfs(self, observations, min_value, max_value, points_per_dim):
        points = [np.linspace(min_value, max_value, points_per_dim) for _ in range(self.state_dim)]
        mesh = np.meshgrid(*points, indexing='ij')
        x = np.vstack([dimension.ravel() for dimension in mesh]).T

        values = np.zeros((x.shape[0], self.n_times))
        for i in range(self.n_times):
            values[:,i] = self.evaluate_filter_pdf(x, self.times[i], observations)

        return values, x

    def get_filter_mean(self, timepoint, observations):
        """
        Returns the mean of the distribution at chosen timepoint
        """
        tolerance = 0.00001
        idx = (np.abs(self.times - timepoint)).argmin()
        if np.abs(self.times[idx] - timepoint) > tolerance:
            raise ValueError("Timepoint not among sampled times")

        m = self.m[:, idx]
        return m
    
    def get_filter_means(self, timepoints, observations):
        """
        Returns the means of the distributions at chosen timepoints
        """
        tolerance = 0.00001

        times = np.repeat(self.times[np.newaxis, :], len(timepoints), axis=0)
        idxs = np.abs(times - timepoints.reshape(-1,1)).argmin(axis=1)
        values = np.abs(times - timepoints.reshape(-1,1)).min(axis=1)
        if np.any(values) > tolerance:
            raise ValueError("Timepoint not among sampled times")

        m = self.m[:, idxs]
        return m
    
    def sample_distributions(self, timepoints, observations, n_samples):
        samples = np.zeros((self.state_dim, n_samples, len(timepoints)))
        values = np.zeros((n_samples, len(timepoints)))
        for i in range(len(timepoints)):
            idx = np.where(self.times == timepoints[i])[0][0]
            samples[:,:,i] = np.random.multivariate_normal(self.m[:,idx], self.P[:, :,idx], size = n_samples).T
            mvn = multivariate_normal(self.m[:,idx], self.P[:, :,idx])
            values[:,i] = mvn.pdf(samples[:,:,i].T)
        return samples, values
    
    def get_all_filter_means(self, observations):
        return self.m
    
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

        m = self.m[:, idx]
        P = self.P[:, :, idx]
        mvn = multivariate_normal(mean=m, cov=P)
        predictions = mvn.pdf(x)
        return predictions
    
    def plot_states_and_estimates(self, state_trajectories):
        plt.figure(figsize=(8, 6))
        color_map = plt.get_cmap('tab10')

        for i in range(self.state_dim):
            color = color_map(i % 10)
            plt.plot(self.times, state_trajectories[i, :], label=f'State {i + 1}', color = color)
            plt.scatter(self.times[self.measurement_indices], self.m[i, self.measurement_indices], s = 20, color = color, edgecolors="black")

        plt.title('Simulated paths and kalman filter estimates')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()