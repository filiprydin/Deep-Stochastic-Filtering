import numpy as np

from scipy.stats import multivariate_normal

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Parent class of diffusion-type SDEs with coupled measurements
# More specifically, defines uni-dimensional brownian motion with linear measurements and normal prior
# dXt = dWt
# h(x) = x

class DiffusionSDE:
    def __init__(self):
        self.identifier = "BM"

        self.measurement_noise_std = 1
        self.state_dim = 1
        self.meas_dim = 1

        self.invertible_meas = True

    def drift(self, x):
        mu = np.zeros(self.state_dim)
        return mu
    
    def drift_batch(self, x):
        mu = np.zeros_like(x)
        return mu
    
    def drift_batch_torch(self, x):
        mu = torch.zeros_like(x, device=device)
        return mu

    def diffusion(self, x):
        sigma = np.identity(self.state_dim)
        return sigma
    
    def diffusion_batch(self, x):
        sigma = np.identity(self.state_dim)
        n_sim = x.shape[0]
        sigma = np.repeat(sigma[np.newaxis,:,:], repeats=n_sim, axis=0)
        return sigma

    def measurement(self, x):
        h = [x[0]]
        return h
    
    def measurement_batch(self, x):
        h = x[:, 0].reshape(-1, 1)
        return h
    
    def measurement_batch_torch(self, x):
        h = x[:, 0].view(-1, 1)
        return h
    
    def measurement_inverse(self, y):
        return y

    def measurement_inverse_batch(self, y):
        return y
    
    def get_kalman_filter_params(self, deltat):
        """
        Returns parameters needed for the kalman filter
        Based on model: 
        x_k = A_(k-1) x_(k-1) + q_(k-1) + b_(k-1)
        y_k = H_k x_k + r_k
        q_(k-1) ~ N(0, Q_(k-1))
        r_k ~ R_k 
        """
        A = np.identity(self.state_dim)
        b = np.zeros(self.state_dim)
        Q = np.matmul(self.diffusion(np.zeros(self.state_dim)), np.transpose(self.diffusion(np.zeros(self.state_dim)))) * deltat
        R = np.identity(self.meas_dim) * self.measurement_noise_std**2
        H = np.identity(self.state_dim)
            
        return A, b, Q, R, H
    
    def get_extended_kalman_filter_update_params(self, x, r):
        """
        Returns parameters needed for the extended kalman filter update step
        Based on model: 
        y_k = h(x_k, r)
        r ~ N(0, R)
        Hx jacobian of h wrt x
        Hr jacobian of h wrt r
        """
        R = np.identity(self.meas_dim) * self.measurement_noise_std**2
        Hx = np.identity(self.meas_dim)
        Hr = np.identity(self.meas_dim)
            
        return R, Hx, Hr
    
    def get_extended_kalman_filter_pred_params(self, x, q, deltat):
        """
        Returns parameters needed for the extended kalman filter prediction step
        Based on model: 
        x_k = f(x_k-1, q_k-1)
        q_(k-1) ~ N(0, Q)
        Fx jacobian of f wrt x
        Fq jacobian of f wrt q
        """
        f = np.matmul(np.identity(self.state_dim), x)
        f += self.drift(x)*deltat
        f += np.matmul(self.diffusion(x), q)
        Fx = np.identity(self.state_dim)
        Fq = self.diffusion(x)
        Q = np.identity(self.state_dim) * deltat
            
        return f, Q, Fx, Fq
    
    def get_unscented_kalman_filter_pred_params(self, deltat):
        """
        Returns parameters needed for the unscented kalman filter prediction step
        Based on model: 
        x_k = f(x_k-1, q_k-1)
        q_(k-1) ~ N(0, Q)
        """
        Q = np.identity(self.state_dim) * deltat
            
        return Q
    
    def get_unscented_kalman_filter_update_params(self):
        """
        Returns parameters needed for the unscented kalman filter update step
        Based on model: 
        y_k = h(x_k, r)
        r ~ N(0, R)
        """
        R = np.identity(self.meas_dim) * self.measurement_noise_std**2
            
        return R

    def get_ebds_filter_params(self, x):
        """
        Returns parameters needed for the energy-based deep splitting filter
        Input: 
        x: (n_points, state_dim) a collection of states
        Let a = sigma*sigma^T
        returns 
        a1: da_ij/dx_i matrix
        a2: d^2 a_ij/dx_i*dx_j matrix
        a3: dmu_i/dx_i vector
        a4: mu vector
        H: Jacobian of the measurement function 
        """
        n_points = x.size(dim=0)

        a1 = torch.zeros((self.state_dim, self.state_dim), dtype=x.dtype, device=device)
        a1 = a1.unsqueeze(0).repeat(n_points, 1, 1)

        a2 = torch.zeros((self.state_dim, self.state_dim), dtype=x.dtype, device=device)
        a2 = a2.unsqueeze(0).repeat(n_points, 1, 1)

        a3 = torch.zeros(self.state_dim, dtype=x.dtype, device=device)
        a3 = a3.unsqueeze(0).repeat(n_points, 1)

        a4 = self.drift_batch_torch(x)

        H = torch.eye(self.state_dim, dtype=x.dtype, device=device)
        H = H.unsqueeze(0).repeat(n_points, 1, 1)

        return a1, a2, a3,  a4, H
    
    def simulate_starting_state(self, N_simulations):
        X_0 = np.random.normal(0, 1, (N_simulations, self.state_dim))
        return X_0
    
    def get_prior_moments(self):
        """ 
        Returns mean and covariance for the prior distribution
        Needed for all kalman filters
        """
        P0 = np.identity(self.state_dim)
        m0 = np.zeros(self.state_dim)
        return m0, P0
    
    def prior_likelihood(self, x):
        """
        Returns the likelihood of a given starting state (pdf of the prior)
        Input:
        x: (num_points, state_dim) or (state_dim)
        """
        P0 = np.identity(self.state_dim)
        m0 = np.zeros(self.state_dim)

        mvn = multivariate_normal(mean=m0, cov=P0)
        likelihood = mvn.pdf(x)
        return likelihood
    
    def prior_likelihood_torch(self, x):
        """
        Returns the likelihood of a given starting state (pdf of the prior)
        Input:
        x: (num_points, state_dim) or (state_dim)
        """
        P0 = torch.eye(self.state_dim, dtype=x.dtype, device=device)
        m0 = torch.zeros(self.state_dim, dtype=x.dtype, device=device)

        mvn = torch.distributions.multivariate_normal.MultivariateNormal(loc=m0, scale_tril = P0)

        # Encountered cuda error with too many samples
        max_batch_size = 10 ** 5
        if x.size(dim = 0) <= max_batch_size:
            likelihood = torch.exp(mvn.log_prob(x))
        else:
            n_points = x.size(dim=0)
            likelihood = torch.zeros((n_points, 1), device=device)
            n_iter = n_points // max_batch_size
            for i in range(n_iter):
                start_idx = max_batch_size * i
                end_idx = min(max_batch_size * (i + 1), n_points)
                likelihood[start_idx:end_idx, :] = torch.exp(mvn.log_prob(x[start_idx:end_idx, :])).view(-1, 1)
            likelihood = likelihood.flatten()

        return likelihood
    
    def prior_gradient_torch(self, x):
        """
        Returns the gradient of the prior in given points.
        Also returns the likelihood, which allows for more efficient code down the line
        Needed for energy-based deep splitting
        """
        P0 = torch.eye(self.state_dim, dtype=x.dtype, device=device)
        m0 = torch.zeros(self.state_dim, dtype=x.dtype, device=device)
        m0 = m0.unsqueeze(0).repeat(x.shape[0], 1)

        likelihood = self.prior_likelihood_torch(x).view(-1, 1)
        gradient = (-torch.matmul(torch.linalg.inv(P0), (x - m0).t()) * likelihood.t()).t()
        return gradient, likelihood

    
