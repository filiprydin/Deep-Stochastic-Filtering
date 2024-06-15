import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from SDEs.diffusion_sde import DiffusionSDE
from sde_simulator import SDESimulator

# Class implementing methods for simulating a dataset for a given SDE

class DatasetGenerator:
    def __init__(self, sde, times, measurement_indices):
        self.sde = sde

        self.times = times
        self.measurement_indices = measurement_indices
        self.n_measurements = len(measurement_indices)

        self.simulator = SDESimulator(self.sde, self.times, self.measurement_indices)

    def _simulate_dataset(self, N_simulations):
        X, Y =  self.simulator.simulate_state_and_measurement(N_simulations)
        X_formatted = np.transpose(X, axes=(0, 2, 1)).reshape(N_simulations, X.shape[1] * X.shape[2])
        self.observations = np.transpose(Y, axes=(0, 2, 1)).reshape(N_simulations, Y.shape[1] * Y.shape[2])
        self.shuffled_states = X_formatted.copy()
        np.random.default_rng().shuffle(self.shuffled_states, axis = 0)

    def generate_and_save_dataset(self, N_simulations, folder_name):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        folder_path = os.path.join(current_dir, "Data", folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
  
        observations_file_path = os.path.join(folder_path, "observations.npy")
        shuffled_states_file_path = os.path.join(folder_path, "shuffled_states.npy")
        self._simulate_dataset(N_simulations)
        np.save(observations_file_path, self.observations)
        np.save(shuffled_states_file_path, self.shuffled_states) 

if __name__ == "__main__":
    sde = DiffusionSDE()

    T_0 = 0
    T_N = 1
    n_measurements = 11
    n_steps = 32
    timepoints_measurement = np.linspace(T_0, T_N, n_measurements)
    timepoints = np.linspace(T_0, T_N, (n_measurements-1)*n_steps + 1)
    m_indices = np.arange(0, len(timepoints), n_steps)

    generator = DatasetGenerator(sde, timepoints, m_indices)
    generator.generate_and_save_dataset(10 ** 3, "Test")