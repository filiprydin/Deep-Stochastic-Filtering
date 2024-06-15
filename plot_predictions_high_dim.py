
import os
import time
import csv
from datetime import datetime

import numpy as np
import scipy
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
from SDEs.spring_mass_6d import SpringMass6d
from SDEs.spring_mass_8d4d import SpringMass8d4d
from SDEs.sin_vol import SinVol
from SDEs.lv3 import LotkaVolterra3

from dataset_generator import DatasetGenerator

plt.style.use('metropolis')

# Python script for plotting an animation of filter predictions over time
# High-dimensional version -> uses Monte-Carlo to calculate marginal densities

sde = SpringMass8d4d()

n_particles = 10000

folder_name = "1007/SM8D4D_1007_16_0"

dimensions = [0, 2, 4, 6]

T_0 = 0
T_N = 0.5
n_measurements = 6
n_steps = 16

ebds_parameters = {
        "Constant network size": True,
        "Train proportion": 0.9,
        "Renormalisation method": 4, # Choose 0-4
        "Hidden size": 256,
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
        "Only positive": False,
        "Gradient clipping": False,
        "Gradient clip value": 10, # Only relevant if gradient clipping is true
        "Normalisation method": "Importance", # One of "Quadrature", "Normal" or "Importance"
        "Normalisation range": [-5, 5], # Only relevant if method is quadrature
        "Normalisation points": 1000, # Only relevant if method is quadrature
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

evaluation_points = 200
sample_points = 5000
x_min = -5
x_max = 5

element_size = ((x_max - x_min) / (evaluation_points - 1))

kf_predictions = np.zeros((len(dimensions), evaluation_points, (n_measurements-1)*n_steps + 1))
ekf_predictions = np.zeros((len(dimensions), evaluation_points, (n_measurements-1)*n_steps + 1))
ukf_predictions = np.zeros((len(dimensions), evaluation_points, (n_measurements-1)*n_steps + 1))
ebds_predictions = np.zeros((len(dimensions), evaluation_points, (n_measurements-1)*n_steps + 1))
pf_predictions = np.zeros((len(dimensions), evaluation_points, (n_measurements-1)*n_steps + 1))
pf.build_kdes(timepoints=timepoints)

x_min_plot = -5
x_max_plot = 5

linspace_array = np.linspace(x_min, x_max, evaluation_points)
plot_low_index = np.searchsorted(linspace_array, x_min_plot)
plot_high_index = np.searchsorted(linspace_array, x_max_plot)

def calculate_marginal_distribution(timepoint, filter, samples, likelihood, Y, dimension):
    marginal_distribution = np.zeros(len(linspace_array))
    
    for j in range(len(linspace_array)):
        points = np.insert(samples, dimension, linspace_array[j], axis=1)
        values = filter.evaluate_filter_pdf(points, timepoint, Y.T)
        marginal_distribution[j] = np.mean(values / likelihood)

    return marginal_distribution / (np.sum(marginal_distribution) * element_size)

def calculate_marginal_distribution_ebds(timepoint, filter, samples, likelihood, Y, dimension):
    marginal_distribution = np.zeros(len(linspace_array))
    
    for j in range(len(linspace_array)):
        points = np.insert(samples, dimension, linspace_array[j], axis=1)
        values = filter.evaluate_filter_pdf_unnorm(points, timepoint, Y.T)
        marginal_distribution[j] = np.mean(values / likelihood)

    return marginal_distribution / (np.sum(marginal_distribution) * element_size)

for i in range(len(timepoints)):
    for j in range(len(dimensions)):
        plot_time = timepoints[i]
        print(f"Evaluating filters at {np.round(plot_time, 4)} in dimension {dimensions[j]}")
        m, P = sde.get_prior_moments()
        m_new = np.delete(m, dimensions[j])
        P_new = np.delete(P, dimensions[j], axis=0)  # Remove row
        P_new = np.delete(P_new, dimensions[j], axis=1)  # Remove column
        sample_mean = m_new
        sample_cov = 5 * P_new

        samples = np.random.multivariate_normal(mean=sample_mean, cov=sample_cov, size=sample_points)
        mvn = scipy.stats.multivariate_normal(sample_mean, sample_cov)
        likelihood_factor = mvn.pdf(samples)

        # ebds distribution
        ebds_predictions[j,:,i] = calculate_marginal_distribution_ebds(plot_time, ebds_filter, samples, likelihood_factor, Y, dimensions[j])

        # Particle filter distribution
        #pf_predictions[j,:,i] = calculate_marginal_distribution(plot_time, pf, samples, likelihood_factor, Y, dimensions[j])

        # Extended kalman filter distribution
        ekf_predictions[j,:,i] = calculate_marginal_distribution(plot_time, ekf, samples, likelihood_factor, Y, dimensions[j])

        # Unscented kalman filter distribution
        #ukf_predictions[j,:,i] = calculate_marginal_distribution(plot_time, ukf, samples, likelihood_factor, Y, dimensions[j])

def update(frame):
    plt.cla() 
    particles = pf.states[:, dimensions[-1], frame]
    weights = pf.weights[:, frame]

    plt.vlines(X[dimensions[-1], frame], ymin=0, ymax=np.max(ekf_predictions[-1,:,:]), label="State", color="blue")
    plt.plot(linspace_array[plot_low_index:plot_high_index], ebds_predictions[-1,plot_low_index:plot_high_index,frame], label="EBDS", color = "black")
    #plt.plot(x[plot_low_index:plot_high_index,0], kf_predictions[plot_low_index:plot_high_index,frame], label="kf", color = "black") 
    plt.plot(linspace_array[plot_low_index:plot_high_index], ekf_predictions[-1,plot_low_index:plot_high_index,frame], label="EKF", color = "green") 
    #plt.plot(x[plot_low_index:plot_high_index,0], ukf_predictions[plot_low_index:plot_high_index,frame], label="ukf", color = "orange") 
    #plt.plot(linspace_array[plot_low_index:plot_high_index], pf_predictions[plot_low_index:plot_high_index,frame], label=f"pf {n_particles}", color = "blue")
    hist_values, bins, _ = plt.hist(particles, weights=weights, bins=200, alpha = 0.5, density=True, color='orange', label=f'PF {n_particles}')

    plt.title(f'Filter/prediction density at time {np.round(timepoints[frame],2)}')
    plt.grid(True) 
    plt.legend()
    plt.tight_layout()

fig, ax = plt.subplots()
ani = FuncAnimation(fig, update, frames=ebds_predictions.shape[1], interval=1000)

# Show the plot
plt.show(block = False)
input("Press Enter to continue...")

for j in range(len(dimensions)):
    for frame in m_indices:
        dim = dimensions[j]
        plt.cla() 
        particles = pf.states[:, dim, frame]
        weights = pf.weights[:, frame]

        plt.vlines(X[dim, frame], ymin=0, ymax=np.max(ekf_predictions[j,:,:]), label="State", color="blue")
        plt.plot(linspace_array[plot_low_index:plot_high_index], ebds_predictions[j,plot_low_index:plot_high_index,frame], label="EBDS", color = "black")
        #plt.plot(x[plot_low_index:plot_high_index,0], kf_predictions[dim,plot_low_index:plot_high_index,frame], label="kf", color = "black") 
        plt.plot(linspace_array[plot_low_index:plot_high_index], ekf_predictions[j,plot_low_index:plot_high_index,frame], label="EKF", color = "green") 
        #plt.plot(x[plot_low_index:plot_high_index,0], ukf_predictions[dim,plot_low_index:plot_high_index,frame], label="ukf", color = "orange") 
        #plt.plot(linspace_array[plot_low_index:plot_high_index], pf_predictions[dim,plot_low_index:plot_high_index,frame], label=f"pf {n_particles}", color = "blue")
        hist_values, bins, _ = plt.hist(particles, weights=weights, bins=200, alpha = 0.5, density=True, color='orange', label=f'PF {n_particles}')

        plt.title(f'Filter/prediction density at time {np.round(timepoints[frame],2)}')
        plt.grid(True) 
        plt.legend()
        plt.tight_layout()
        print(tikzplotlib.get_tikz_code())
        tikzplotlib.save(f"{sde.identifier}_{frame}_{dim}.txt")
        plt.show()
