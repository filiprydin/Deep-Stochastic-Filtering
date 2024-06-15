import numpy as np
import math
import matplotlib.pyplot as plt

# Class implementing methods for simulating trajectories for diffusion-type SDEs

class SDESimulator:
    def __init__(self, sde, times, measurement_indices):
        self.sde = sde
        self.state_dim = sde.state_dim
        self.meas_dim = sde.meas_dim
        self.times = times
        self.n_times = len(times)
        self.measurement_indices = measurement_indices
        self.n_measurements = len(measurement_indices)

    def _simulate_state_em(self):
        # Simulate state using euler-maruyama method
        self.X = np.zeros((self.N_simulations, self.state_dim, self.n_times), dtype=np.float32)
        self.X[:,:,0] = self.sde.simulate_starting_state(self.N_simulations)

        for i in range(1, self.n_times):
            deltat = self.times[i] - self.times[i-1]
            deltaW = np.random.normal(0, math.sqrt(deltat), (self.N_simulations, self.state_dim))
            drift_term = self.sde.drift_batch(self.X[:,:,i-1]) * deltat
            diffusion_term = np.einsum('ijk,ik->ij', self.sde.diffusion_batch(self.X[:,:,i-1]), deltaW)
            self.X[:,:,i] = self.X[:,:,i-1] + drift_term + diffusion_term

    def _simulate_state_ms(self):
        # Simulate state using milstein method
        # TODO
        pass

    def _measure(self):
        self.Y = np.zeros((self.N_simulations, self.meas_dim, self.n_measurements), dtype=np.float32)

        for i in range(self.n_measurements):
            m_index = self.measurement_indices[i]
            self.Y[:,:,i] = self.sde.measurement_batch(self.X[:,:,m_index]) + np.random.normal(0, self.sde.measurement_noise_std, (self.N_simulations, self.meas_dim))
    
    def simulate_state_and_measurement(self, N_simulations = 1):
        self.N_simulations = N_simulations
        self._simulate_state_em()
        self._measure()
        if self.N_simulations == 1:
            self.X = self.X[0, :, :]
            self.Y = self.Y[0, :, :]
        return self.X, self.Y
    
    def plot_trajectories(self, dimensions):
        plt.figure(figsize=(8, 6))
        color_map = plt.get_cmap('tab10')

        if self.X.ndim == 3:
            plot_index = 0 # If multiple simulations, choose one
            X = self.X[plot_index, :, :]
            Y = self.Y[plot_index, :, :]
        else:
            X = self.X
            Y = self.Y

        for i in dimensions:
            color = color_map(i % 10)
            plt.plot(self.times, X[i, :], label=f'State {i + 1}', color = color)
        
        for j in dimensions:
            plt.scatter(self.times[self.measurement_indices], Y[j, :], s = 20, color = "000000", label='Observations')

        plt.title('Simulated paths')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()