import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal

from Filters.bayesian_filter import BayesianFilter

# Defines a kalman filter, which allows for dynamic filtering of linear models

class KalmanFilter(BayesianFilter):
    def __init__(self, sde, times, measurement_indices):
        self.sde = sde
        self.state_dim = sde.state_dim
        self.meas_dim = sde.meas_dim
        self.times = times
        self.n_times = len(times)
        self.measurement_indices = measurement_indices
        self.m0, self.P0 = sde.get_prior_moments()

    def filter(self, observations):
        self.observations = observations
        self.m = np.zeros((self.state_dim, self.n_times))
        self.P = np.zeros((self.state_dim, self.state_dim, self.n_times))

        # At time t_0
        A, b, Q, R, H = self.sde.get_kalman_filter_params(0)

        v = observations[:,0] -  np.matmul(H, self.m0)
        S = np.matmul(np.matmul(H, self.P0), np.transpose(H)) + R
        K = np.matmul(np.matmul(self.P0, np.transpose(H)), np.linalg.inv(S))

        self.m[:,0] = self.m0 + np.matmul(K, v)
        self.P[:,:,0] = self.P0 - np.matmul(np.matmul(K, S), np.transpose(K))

        for k in range(0, self.n_times-1):
            deltat = self.times[k + 1] - self.times[k]
            Ak, bk, Qk, R, H = self.sde.get_kalman_filter_params(deltat)

            # Prediction step
            mm = np.matmul(Ak, self.m[:,k]) + bk
            Pm = np.matmul(np.matmul(Ak, self.P[:,:,k]), np.transpose(Ak)) + Qk

            # Update step
            if (k + 1) in self.measurement_indices:
                idx = np.where(self.measurement_indices == k+1)[0][0]
                v = observations[:,idx] -  np.matmul(H, mm)
                S = np.matmul(np.matmul(H, Pm), np.transpose(H)) + R
                K = np.matmul(np.matmul(Pm, np.transpose(H)), np.linalg.inv(S))

                self.m[:,k+1] = mm + np.matmul(K, v)
                self.P[:,:,k+1] = Pm - np.matmul(np.matmul(K, S), np.transpose(K))
            else:
                self.m[:,k+1] = mm
                self.P[:,:,k+1] = Pm 
    
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
    
    def get_all_filter_means(self, observations):
        return self.m
    
    def sample_distributions(self, timepoints, observations, n_samples):
        samples = np.zeros((self.state_dim, n_samples, len(timepoints)))
        values = np.zeros((n_samples, len(timepoints)))
        for i in range(len(timepoints)):
            idx = np.where(self.times == timepoints[i])[0][0]
            samples[:,:,i] = np.random.multivariate_normal(self.m[:,idx], self.P[:, :,idx], size = n_samples).T
            mvn = multivariate_normal(self.m[:,idx], self.P[:, :,idx])
            values[:,i] = mvn.pdf(samples[:,:,i].T)
        return samples, values

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