
import os
import time
import csv
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import tikzplotlib

from sde_simulator import SDESimulator

from Filters.kalman_filter import KalmanFilter
from Filters.extended_kalman_filter import ExtendedKalmanFilter
from Filters.ebds_filter import EBDSFilter
from Filters.ebds_LSTM_filter import EBDSLSTMFilter
from Filters.particle_filter import ParticleFilter
from Filters.unscented_kalman_filter import UnscentedKalmanFilter

from SDEs.diffusion_sde import DiffusionSDE
from SDEs.ornstein_uhlenbeck import OrnsteinUhlenbeck
from SDEs.bimodal import BiModal
from SDEs.geometric import GeometricBrownianMotion
from SDEs.ornstein_uhlenbeck_2d import OrnsteinUhlenbeck2d
from SDEs.ornstein_uhlenbeck_10d import OrnsteinUhlenbeck10d

from dataset_generator import DatasetGenerator

# Python script for plotting an animation of filter predictions over time

sde = GeometricBrownianMotion()

n_particles = 10000

folder_name = "1003/GBM_1003_16_0"

dimension = 0

T_0 = 0
T_N = 1
n_measurements = 11
n_steps = 16

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
        "Normalisation constant targets": 0.1, # Only relevant if normalise targets is True
        "Tail terms": False,
        "Only positive": True,
        "Gradient clipping": False,
        "Gradient clip value": 10, # Only relevant if gradient clipping is true
        "Normalisation method": "Quadrature", # One of "Quadrature", "Normal" or "Importance"
        "Normalisation range": [-2, 10], # Only relevant if method is quadrature
        "Normalisation points": 2000, # Only relevant if method is quadrature
        "Normalisation samples": 10 ** 4 # Only relevant if method is "Normal" or "Importance"
    }

timepoints_measurement = np.linspace(T_0, T_N, n_measurements)
timepoints = np.linspace(T_0, T_N, (n_measurements-1)*n_steps + 1)
m_indices = np.arange(0, len(timepoints), n_steps)

ebds_filter = EBDSFilter(sde, timepoints, m_indices, ebds_parameters)
ebds_filter.load_saved_models(folder_name)

# Simulate test sde
simulator = SDESimulator(sde, timepoints, m_indices)
X, Y = simulator.simulate_state_and_measurement(N_simulations=2)
X = X[0, :, :]
Y = Y[0, :, :] 

#kf = KalmanFilter(sde, timepoints, m_indices)
#kf.filter(Y)
pf = ParticleFilter(sde, timepoints, m_indices, n_particles)
pf.filter(Y)
ekf = ExtendedKalmanFilter(sde, timepoints, m_indices)
ekf.filter(Y)
ukf = UnscentedKalmanFilter(sde, timepoints, m_indices)
ukf.filter(Y)

points_per_dim = 2000
x_min = -2
x_max = 10
element_size = ((x_max - x_min) / (points_per_dim - 1))**(sde.state_dim - 1)

kf_predictions = np.zeros((points_per_dim, (n_measurements-1)*n_steps + 1))
ekf_predictions = np.zeros((points_per_dim, (n_measurements-1)*n_steps + 1))
ukf_predictions = np.zeros((points_per_dim, (n_measurements-1)*n_steps + 1))
ebds_predictions = np.zeros((points_per_dim, (n_measurements-1)*n_steps + 1))
pf_predictions = np.zeros((points_per_dim, (n_measurements-1)*n_steps + 1))
pf.build_kdes(timepoints=timepoints)

x_min_plot = -2
x_max_plot = 10

linspace_array = np.linspace(x_min, x_max, points_per_dim)
plot_low_index = np.searchsorted(linspace_array, x_min_plot)
plot_high_index = np.searchsorted(linspace_array, x_max_plot)

def calculate_marginal_distribution(predictions, points):
    tolerance = (linspace_array[1] - linspace_array[0]) / 3
    marginal_distribution = np.zeros(len(linspace_array))
    
    for j in range(len(linspace_array)):
        marginal_distribution[j] = np.sum(predictions[np.abs(points[:, dimension] - linspace_array[j]) < tolerance])
    
    return marginal_distribution * element_size

for i in range(len(timepoints)):
    plot_time = timepoints[i]
    print(f"Evaluating filters at {np.round(plot_time, 4)}")

    # Kalman filter distribution
    #kf_predictions[:,i], x = kf.get_filter_pdf(plot_time, Y.T, x_min, x_max, points_per_dim=points)

    # ebds distribution
    ebds_grid_predictions, x = ebds_filter.get_filter_pdf(plot_time, Y.T, x_min, x_max, points_per_dim)
    ebds_predictions[:,i] = calculate_marginal_distribution(ebds_grid_predictions, x)

    # Particle filter distribution
    #pf_grid_predictions, x = pf.get_filter_pdf(plot_time, Y.T, x_min, x_max, points_per_dim)
    #pf_predictions[:,i] = calculate_marginal_distribution(pf_grid_predictions, x)

    # Extended kalman filter distribution
    ekf_grid_predictions, x = ekf.get_filter_pdf(plot_time, Y.T, x_min, x_max, points_per_dim)
    ekf_predictions[:,i] = calculate_marginal_distribution(ekf_grid_predictions, x)

    # Unscented kalman filter distribution
    ukf_grid_predictions, x = ukf.get_filter_pdf(plot_time, Y.T, x_min, x_max, points_per_dim)
    ukf_predictions[:,i] = calculate_marginal_distribution(ukf_grid_predictions, x)

def update(frame):
    plt.cla() 
    particles = pf.states[:, dimension, frame]
    weights = pf.weights[:, frame]

    plt.vlines(X[dimension, frame], ymin=0, ymax=np.max(ekf_predictions), label="State", color="blue")
    plt.plot(linspace_array[plot_low_index:plot_high_index], ebds_predictions[plot_low_index:plot_high_index,frame], label="EBDS", color = "black")
    #plt.plot(x[plot_low_index:plot_high_index,0], kf_predictions[plot_low_index:plot_high_index,frame], label="kf", color = "black") 
    plt.plot(linspace_array[plot_low_index:plot_high_index], ekf_predictions[plot_low_index:plot_high_index,frame], label="EKF", color = "green") 
    #plt.plot(x[plot_low_index:plot_high_index,0], ukf_predictions[plot_low_index:plot_high_index,frame], label="ukf", color = "orange") 
    #plt.plot(linspace_array[plot_low_index:plot_high_index], pf_predictions[plot_low_index:plot_high_index,frame], label=f"pf {n_particles}", color = "blue")
    hist_values, bins, _ = plt.hist(particles, weights=weights, bins=50, alpha = 0.5, density=True, color='orange', label=f'PF {n_particles}')

    plt.title(f'Filter/prediction density at time {np.round(timepoints[frame],2)}')
    plt.grid(True) 
    plt.legend()
    plt.tight_layout()

fig, ax = plt.subplots()
ani = FuncAnimation(fig, update, frames=ebds_predictions.shape[1], interval=50)

# Show the plot
plt.show(block = False)
input("Press Enter to continue...")

plt.cla()
plt.plot(timepoints, X[0,:])
plt.scatter(timepoints[m_indices], Y[0,:])
print(tikzplotlib.get_tikz_code())
tikzplotlib.save(f"{sde.identifier}_state.txt")
plt.show()

for frame in m_indices:
    plt.cla()
    particles = pf.states[:, dimension, frame]
    weights = pf.weights[:, frame]

    plt.plot(x[plot_low_index:plot_high_index, 0], ebds_predictions[plot_low_index:plot_high_index, frame], label="ebds", color="red")
    # plt.plot(x[plot_low_index:plot_high_index, 0], kf_predictions[plot_low_index:plot_high_index, frame], label="kf", color="black")
    plt.plot(x[plot_low_index:plot_high_index, 0], ekf_predictions[plot_low_index:plot_high_index, frame], label="ekf", color="purple")
    # plt.plot(x[plot_low_index:plot_high_index, 0], ukf_predictions[plot_low_index:plot_high_index, frame], label="ukf", color="orange")
    #plt.plot(x[plot_low_index:plot_high_index, 0], pf_predictions[plot_low_index:plot_high_index, frame], label=f"pf {n_particles}", color="blue")

    plt.vlines(X[dimension, frame], ymin=0, ymax=np.max(ekf_predictions), label="State")

    hist_values, bins, _ = plt.hist(particles, weights=weights, bins=50, alpha = 0.5, density=True, color='grey', label=f'PF {n_particles}')

    plt.title(f'Filtering/prediction densities at time {np.round(timepoints[frame], 2)}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    print(tikzplotlib.get_tikz_code())
    tikzplotlib.save(f"{sde.identifier}_{frame}.txt")
    plt.show()