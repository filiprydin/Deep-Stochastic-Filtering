
import time

import numpy as np
from matplotlib import pyplot as plt
import tikzplotlib

import torch

from sde_simulator import SDESimulator

from Filters.ebds_filter import EBDSFilter

from SDEs.diffusion_sde import DiffusionSDE
from SDEs.ornstein_uhlenbeck import OrnsteinUhlenbeck
from SDEs.bimodal import BiModal

# Python script for evaluating ebds filter average integral over time
# This can be useful for evaluating different normalisation schemes

start_time = time.time()

sde = OrnsteinUhlenbeck()

n_test_samples = 1000

T_0 = 0
T_N = 1
n_measurements = 11
n_steps = 32

ebds_parameters = {
        "Constant network size": True,
        "Train proportion": 0.9,
        "Renormalisation method": 4, # Choose 0-4
        "Hidden size": 128,
        "Learning rate": 0.001, 
        "Max epochs": 100,
        "Early stopping patience": 5,
        "Batch size": 512,
        "Batch normalisation": False, # Does not seem to improve results
        "Validation partition": False, # Does not seem to improve results
        "Remove extremes": False,
        "Normalise targets": True, 
        "Normalisation constant targets": 1, # Only relevant if normalise targets is True
        "Tail terms": False,
        "Only positive": False,
        "Gradient clipping": False,
        "Gradient clip value": 10, # Only relevant if gradient clipping is true
        "Normalisation method": "Quadrature", # One of "Quadrature", "Normal" or "Importance"
        "Normalisation range": [-5, 5], # Only relevant if method is quadrature
        "Normalisation points": 2000, # Only relevant if method is quadrature
        "Normalisation samples": 10 ** 4 # Only relevant if method is "Normal" or "Importance"
    }

timepoints_measurement = np.linspace(T_0, T_N, n_measurements)
timepoints = np.linspace(T_0, T_N, (n_measurements-1)*n_steps + 1)
m_indices = np.arange(0, len(timepoints), n_steps)

folder_name = "233/OU_233_32_0"
nn_filter = EBDSFilter(sde, timepoints, m_indices, ebds_parameters)
nn_filter.load_saved_models(folder_name)

simulator = SDESimulator(sde, timepoints, m_indices)
models = nn_filter.models

x_min = -10
x_max = 10
n_points = 2000
points = np.linspace(x_min, x_max, n_points)
element_size = ((x_max - x_min) / (n_points - 1))

integrals = np.zeros(len(models))
X, Y = simulator.simulate_state_and_measurement(N_simulations=n_test_samples)
for j in range(n_test_samples):

    if (j + 1) % 1 == 0:
        print("Evaluating sample", j + 1)

    for i in range(len(models)):
        model = models[i]
        n_observations = i // n_steps + 1
        n_zeros = n_measurements - 1 - n_observations
        y = np.hstack((Y[j,0,0:n_observations], np.zeros(n_zeros)))
        y_formatted = np.repeat(y.reshape(1,len(y)), n_points, axis = 0)
        input = torch.tensor(np.hstack((points.reshape(len(points),1), y_formatted)), dtype = torch.float32)
        outputs = model(input).detach().numpy().flatten()

        integrals[i] += np.sum(outputs) * element_size
integrals = integrals / n_test_samples

plt.plot(timepoints[1:], integrals, label = "Average integral")
#plt.yscale("log")
plt.axvline(x=timepoints[m_indices][0], color='red', linestyle='--', label = "Observation time")
for i in range(1, len(timepoints[m_indices])):
    plt.axvline(x=timepoints[m_indices][i], color='red', linestyle='--')
plt.ylabel(r"$\overline{I}_t$")
plt.xlabel("Time")
plt.legend()
print(tikzplotlib.get_tikz_code())
plt.show()