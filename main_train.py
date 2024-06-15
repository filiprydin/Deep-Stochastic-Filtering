
import sys
import os
import time
import csv
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt

from SDEs.diffusion_sde import DiffusionSDE
from SDEs.ornstein_uhlenbeck import OrnsteinUhlenbeck
from SDEs.sin_vol import SinVol
from SDEs.bimodal import BiModal
from SDEs.spring_mass_8d4d import SpringMass8d4d
from SDEs.lv3 import LotkaVolterra3

from dataset_generator import DatasetGenerator

from Filters.ebds_filter import EBDSFilter

def main():
    if len(sys.argv) != 4:
        raise ValueError("Usage: python script.py <integer> <integer> <integer>")
    try:
        n_steps = int(sys.argv[1])
        run_idx = int(sys.argv[2])
        experiment_identifier = int(sys.argv[3])
    except:
        raise ValueError("Usage: python script.py <integer> <integer> <integer>")

    start_time = time.time()

    # Define important parameters start
    # _____________________________________________________________________
    sde = OrnsteinUhlenbeck()

    n_train_samples = 1 * 10 ** 4

    T_0 = 0
    T_N = 1
    n_measurements = 11

    ebds_parameters = {
        "Constant network size": True,
        "Train proportion": 0.9,
        "Renormalisation method": 4, # Choose 0-4
        "Hidden size": 128,
        "Learning rate": 0.0001, 
        "Max epochs": 100,
        "Early stopping patience": 5,
        "Batch size": 512,
        "Batch normalisation": False, # Does not seem to improve results
        "Validation partition": False, # Does not seem to improve results
        "Remove extremes": False,
        "Normalise targets": False, 
        "Normalisation constant targets": 1, # Only relevant if normalise targets is True
        "Tail terms": False,
        "Only positive": False,
        "Gradient clipping": False,
        "Gradient clip value": 10, # Only relevant if gradient clipping is true
        "Normalisation method": "Quadrature", # One of "Quadrature", "Normal" or "Importance"
        "Normalisation range": [-10, 10], # Only relevant if method is quadrature
        "Normalisation points": 2000, # Only relevant if method is quadrature
        "Normalisation samples": 10 ** 3 # Only relevant if method is "Normal" or "Importance"
    }
    # Define parameters end
    # ____________________________________________________________________

    timepoints = np.linspace(T_0, T_N, (n_measurements-1)*n_steps + 1)
    m_indices = np.arange(0, len(timepoints), n_steps)

    identifier = sde.identifier
    folder_name = f"{identifier}_{experiment_identifier}_{n_steps}_{run_idx}"

    log_name = os.path.join("Logs", str(experiment_identifier) ,f"train_{experiment_identifier}_{n_steps}_{run_idx}.out")
    with open(log_name, 'w') as file:
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            f"Current date and time: {current_datetime}\n"
            f"Experiment number: {experiment_identifier}\n",
            f"Number of intermediate steps: {n_steps}\n",
            f"Run index: {run_idx}\n",
            f"SDE identifier: {identifier}\n\n",
            f"Number of training samples: {n_train_samples}\n",
            f"T0: {T_0}\n", 
            f"TN: {T_N}\n",
            f"Number of measurements: {n_measurements} \n\n"
        ]
        file.writelines(lines)
        for line in lines:
            print(line, flush=True, end="")
        for key, value in ebds_parameters.items():
                file.write(f"{key}: {value}\n")
                print(f"{key}: {value}", flush=True)
        print("", flush=True)
        file.flush()

    folder_name_model = os.path.join(str(experiment_identifier), folder_name)
    folder_name_experiment = os.path.join("Models", str(experiment_identifier))
    if not os.path.exists(folder_name_experiment):
        os.makedirs(folder_name_experiment)

    ebds_filter = EBDSFilter(sde, timepoints, m_indices, ebds_parameters, verbose=True)
    ebds_filter.define_and_train_models(folder_name, n_train_samples)
    if run_idx == 0:
        ebds_filter.save_models(folder_name_model, only_filter_times = False)
    else: 
        ebds_filter.save_models(folder_name_model, only_filter_times = True)

    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("", flush=True)
    print(f"Training done at date and time: {current_datetime}", flush=True)
    train_end_time = time.time()
    train_time = train_end_time - start_time
    print(f"Training time: {np.round(train_time,2)} s", flush=True)
    print("", flush=True)
    with open(log_name, 'a') as file:
        file.write(f"\nTraining done at date and time: {current_datetime} \n")
        file.write(f"Training time: {np.round(train_time,2)} s\n")
        file.flush()

if __name__ == "__main__":
    main()
