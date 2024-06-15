import numpy as np
from scipy.stats import multivariate_normal

# Parent class of all bayesian filters

class BayesianFilter():
    def __init__(self, sde, times, measurement_indices):
        self.sde = sde
        self.state_dim = sde.state_dim
        self.meas_dim = sde.meas_dim
        self.times = times
        self.n_times = len(times)
        self.measurement_indices = measurement_indices

    def filter(self, observations):
        pass

    def get_filter_pdf(self, timepoint, observations, min_value, max_value, points_per_dim):
        """
        Returns the filter pdf in the grid defined by min_value, max_value and points_per_dim
        Also returns the points flattened to the format (n points, state space dimension)
        """
        points = [np.linspace(min_value, max_value, points_per_dim) for _ in range(self.state_dim)]
        mesh = np.meshgrid(*points, indexing='ij')
        x = np.vstack([dimension.ravel() for dimension in mesh]).T

        mvn = multivariate_normal(mean=np.zeros(self.state_dim), cov=np.identity(self.state_dim))
        predictions = mvn.pdf(x)

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

        return np.zeros(self.state_dim)
    
    def get_filter_means(self, timepoints, observations):
        """
        Returns the means of the distribution at chosen timepoints
        """
        tolerance = 0.00001
        for timepoint in timepoints:
            idx = (np.abs(self.times - timepoint)).argmin()
            if np.abs(self.times[idx] - timepoint) > tolerance:
                raise ValueError("Timepoint not among sampled times")

        return np.zeros((self.state_dim, len(timepoints)))

    def get_all_filter_means(self, observations):
        return np.zeros((self.state_dim, self.n_times))
    
    def sample_distributions(self, timepoints, observations, n_samples):
        return np.zeros((self.state_dim, n_samples, len(timepoints))), np.zeros((n_samples, len(timepoints)))
    
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

        mvn = multivariate_normal(mean=np.zeros(self.state_dim), cov=np.identity(self.state_dim))
        predictions = mvn.pdf(x)
        return predictions