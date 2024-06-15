#!/bin/bash
#SBATCH -A C3SE2024-1-2 -p vera
#SBATCH --job-name=ebds
#SBATCH --time=168:00:00
#SBATCH --gpus-per-node=A40:1

# Experiment number
experiment_number=1007

# Parameters
n_discretisations=1
min_n_steps_exponent=4

# Calculate run index and number of steps
n_steps_exponent=$(($min_n_steps_exponent + $SLURM_ARRAY_TASK_ID % $n_discretisations))
n_steps=$((2 ** $n_steps_exponent))
run_idx=$(($SLURM_ARRAY_TASK_ID / $n_discretisations))

# Make model and log directories if they do not exist
mkdir -p Models/${experiment_number}
mkdir -p Logs/${experiment_number}

# Copy all required files
cp sde_simulator.py $TMPDIR
cp -r Filters/ $TMPDIR
cp -r SDEs/ $TMPDIR
cp dataset_generator.py $TMPDIR
cp main_train_sm3.py $TMPDIR

cd $TMPDIR

mkdir Data/
mkdir Models/
mkdir Models/${experiment_number}
mkdir Logs/
mkdir Logs/${experiment_number}

# Load required modules
module load SciPy-bundle/2023.07-gfbf-2023a
module load matplotlib/3.7.2-gfbf-2023a
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# Run the Python script
python main_train_sm3.py $n_steps $run_idx $experiment_number

# Copy back models and logs
cp -r Models/${experiment_number}/* $SLURM_SUBMIT_DIR/Models/${experiment_number}
cp -r Logs/${experiment_number}/* $SLURM_SUBMIT_DIR/Logs/${experiment_number}
