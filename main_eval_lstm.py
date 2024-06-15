
import sys
import os
import time
import csv
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt

from SDEs.diffusion_sde import DiffusionSDE
from SDEs.ornstein_uhlenbeck import OrnsteinUhlenbeck
from SDEs.bimodal import BiModal
from SDEs.ornstein_uhlenbeck_2d import OrnsteinUhlenbeck2d
from SDEs.ornstein_uhlenbeck_5d import OrnsteinUhlenbeck5d
from SDEs.ornstein_uhlenbeck_10d import OrnsteinUhlenbeck10d
from SDEs.spring_mass_6d import SpringMass6d

from Filters.kalman_filter import KalmanFilter
from Filters.extended_kalman_filter import ExtendedKalmanFilter
from Filters.unscented_kalman_filter import UnscentedKalmanFilter
from Filters.ebds_LSTM_filter import EBDSLSTMFilter
from Filters.particle_filter import ParticleFilter

from metric_evaluator import MetricEvaluator

def main():
    if len(sys.argv) != 3:
        raise ValueError("Usage: python script.py <integer> <integer>")
    try:
        experiment_identifier = int(sys.argv[1])
        eval_identifier = int(sys.argv[2])
    except:
        raise ValueError("Usage: python script.py <integer> <integer>")

    start_time = time.time()

    # Define important parameters below
    # _____________________________________________________________________
    sde = OrnsteinUhlenbeck()
    plot_results = True

    n_test_samples = 10 ** 2

    n_steps_reference = 128
    n_steps_benchmarks = 32

    evaluation_parameters = {
        "Grid-based": True,
        "Integration range": [-10, 10], # Only relevant for grid-based evaluation
        "Integration points": 2 * 10 ** 3, # Only relevant for grid-based evaluation
        "MC samples": 10 ** 3 # Only relevant for grid-free evaluation
    }

    # ____________________________________________________________________

    # Below should be same as training (# TODO read automatically)
    # ____________________________________________________________________

    T_0 = 0
    T_N = 1
    n_measurements = 11

    ebds_parameters = {
        "Train proportion": 0.9,
        "Renormalisation method": 4, # Choose 0-4
        "Hidden size": 64,
        "Hidden size DNN": 128,
        "LSTM layers": 1,
        "Learning rate": 0.001, 
        "Max epochs": 100,
        "Early stopping patience": 5,
        "Batch size": 512,
        "Validation partition": False, # Does not seem to improve results
        "Remove extremes": False,
        "Normalise targets": True, 
        "Normalisation constant targets": 0.1, # Only relevant if normalise targets is True
        "Tail terms": False,
        "Only positive": False,
        "Gradient clipping": True,
        "Gradient clip value": 10, # Only relevant if gradient clipping is true
        "Normalisation method": "Quadrature", # One of "Quadrature", "Normal" or "Importance"
        "Normalisation range": [-10, 10], # Only relevant if method is quadrature
        "Normalisation points": 2000, # Only relevant if method is quadrature
        "Normalisation samples": 10 ** 3 # Only relevant if method is "Normal" or "Importance"
    }
    # ______________________________________________________________________

    timepoints_reference = np.linspace(T_0, T_N, (n_measurements-1)*n_steps_reference + 1)
    m_indices_reference = np.arange(0, len(timepoints_reference), n_steps_reference)
    timepoints_benchmarks = np.linspace(T_0, T_N, (n_measurements-1)*n_steps_benchmarks + 1)
    m_indices_benchmarks = np.arange(0, len(timepoints_benchmarks), n_steps_benchmarks)

    # Define filters below
    # _________________________________________________________________________

    # Reference filter
    reference = ExtendedKalmanFilter(sde, timepoints_reference, m_indices_reference)
    reference_name = "kf"
    #reference = ParticleFilter(sde, timepoints_reference, m_indices_reference, n_particles=1000)
    #reference_name = "pf 1000"

    # Benchmark filters
    pf = ParticleFilter(sde, timepoints_benchmarks, m_indices_benchmarks, n_particles = 100)
    pf2 = ParticleFilter(sde, timepoints_benchmarks, m_indices_benchmarks, n_particles = 1000)
    #pf3 = ParticleFilter(sde, timepoints_benchmarks, m_indices_benchmarks, n_particles = 10000)

    # Input all benchmark filters
    benchmark_filters = {"pf 100": pf,
            "pf 1000": pf2
            #"pf 10000": pf3    
            }
    
    # __________________________________________________________________________

    identifier = sde.identifier
    log_name = os.path.join("Logs", str(experiment_identifier) ,f"eval_{experiment_identifier}_{eval_identifier}.out")
    with open(log_name, 'w') as file:
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            f"Current date and time: {current_datetime}\n"
            f"Experiment number: {experiment_identifier}\n",
            f"Evaluation identifier: {eval_identifier}\n",
            f"Number of intermediate steps reference filter: {n_steps_reference}\n",
            f"Number of intermediate steps benchmark filters: {n_steps_benchmarks}\n",
            f"SDE identifier: {identifier}\n",
            f"Number of test samples: {n_test_samples}\n\n",
            f"T0: {T_0}\n", 
            f"TN: {T_N}\n",
            f"Number of measurements: {n_measurements} \n\n"
        ]
        file.writelines(lines)
        file.flush()
        for line in lines:
            print(line, end="", flush=True)
        print(f"Reference filter name: {reference_name}", flush=True)
        file.write(f"Reference filter name: {reference_name}\n")
        print("Benchmark filters: ", end="", flush=True)
        file.write(f"Benchmark filters: ")
        idx = 0
        for key, value in benchmark_filters.items():
            idx += 1
            if idx < len(benchmark_filters):
                file.write(f"{key}, ")
                print(f"{key}, ", end="", flush=True)
            else:
                file.write(f"{key}\n\n")
                print(f"{key}\n", flush=True)
        file.flush()
        for key, value in evaluation_parameters.items():
            file.write(f"{key}: {value}\n")
            print(f"{key}: {value}", flush=True)
        print("", flush=True)
        file.flush()
        for key, value in ebds_parameters.items():
            file.write(f"{key}: {value}\n")
            print(f"{key}: {value}", flush=True)
        print("", flush=True)
        file.flush()

    # Loop through all saved EBDS models for the experiment and add to filters dict
    ebds_filters = {}
    folder_name = os.path.join("Models", str(experiment_identifier))
    for subdir, _, _ in os.walk(folder_name):
        subfolder_name = os.path.basename(subdir)
        parts = subfolder_name.split('_')
        if len(parts) != 4:
            continue
        n_steps = int(parts[2])

        timepoints = np.linspace(T_0, T_N, (n_measurements-1)*n_steps + 1)
        m_indices = np.arange(0, len(timepoints), n_steps)
        
        ebds_path = os.path.join(str(experiment_identifier), subfolder_name)
        ebds_filter = EBDSLSTMFilter(sde, timepoints, m_indices, ebds_parameters, verbose=True)
        ebds_filter.load_saved_models(ebds_path, only_filter_times=True)
        
        filter_name = f"EBDS {n_steps} {parts[3]}" # Construct name from steps and run idx
        ebds_filters[filter_name] = ebds_filter

    metric_evaluator = MetricEvaluator(sde, evaluation_parameters, timepoints_reference, m_indices_reference, ebds_filters, benchmark_filters, reference)
    MAEs, MAEs_reference, FMEs, L2L2s, L2Linfs, KLDs = metric_evaluator.evaluate_metrics(n_test_samples)
    metric_folder_name = os.path.join(str(experiment_identifier), str(eval_identifier))
    metric_evaluator.save_metric_results(metric_folder_name)

    end_time = time.time()
    execution_time = end_time - start_time
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("", flush=True)
    print(f"Evaluation done at date and time: {current_datetime}", flush=True)
    print(f"Evaluation time: {np.round(execution_time,2)} s", flush=True)
    print("", flush=True)
    with open(log_name, 'a') as file:
        file.write(f"\nEvaluation done at date and time: {current_datetime} \n")
        file.write(f"Evaluation time: {np.round(execution_time,2)} s\n")
        file.flush()

    if plot_results:
        plt.style.use('metropolis')
        plt.rcParams['figure.figsize'] = (8, 6)
        
        timepoints_measurement = timepoints_reference[m_indices_reference]

        # Plot MAE
        for i, (filter_name, filter) in enumerate(MAEs.items()):
            plt.plot(timepoints_measurement, MAEs[filter_name], label=filter_name)
        plt.plot(timepoints_measurement, MAEs_reference, label="reference")
        plt.title("Mean Absolute Error over time")
        plt.grid(True) 
        plt.ylabel("MAE")
        plt.xlabel("Time")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Plot FME
        for i, (filter_name, filter) in enumerate(FMEs.items()):
            plt.plot(timepoints_measurement, FMEs[filter_name], label=filter_name)
        plt.title("First Moment Error over time")
        plt.grid(True)
        plt.ylabel("FME")
        plt.xlabel("Time") 
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Plot L2L2 DEN
        for i, (filter_name, filter) in enumerate(L2L2s.items()):
            plt.plot(timepoints_measurement, L2L2s[filter_name], label=filter_name)
        plt.title(r"$L^2(\Omega;L^2(\mathbb{R}^d;\mathbb{R}))$-norm over time")
        plt.grid(True) 
        plt.ylabel(r"$L^2(\Omega;L^2(\mathbb{R}^d;\mathbb{R}))$-norm")
        plt.xlabel("Time")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Plot L2Linf DEN
        for i, (filter_name, filter) in enumerate(L2Linfs.items()):
            plt.plot(timepoints_measurement, L2Linfs[filter_name], label=filter_name)
        plt.title(r"$L^2(\Omega;L^\infty(\mathbb{R}^d;\mathbb{R}))$-norm over time")
        plt.grid(True) 
        plt.ylabel(r"$L^2(\Omega;L^\infty(\mathbb{R}^d;\mathbb{R}))$-norm")
        plt.xlabel("Time")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Plot KLD
        for i, (filter_name, filter) in enumerate(KLDs.items()):
            plt.plot(timepoints_measurement, KLDs[filter_name], label=filter_name)
        plt.title("Kullback-Leibler divergence over time")
        plt.grid(True) 
        plt.ylabel("KLD")
        plt.xlabel("Time")
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
