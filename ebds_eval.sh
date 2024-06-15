#!/bin/bash
#SBATCH -A C3SE2024-1-2 -p vera
#SBATCH --job-name=ebds
#SBATCH --time=96:00:00
#SBATCH --gpus-per-node=A40:1

# Experiment number
experiment_number=1006

# Evaluation number
eval_number=2

# Make metric directory
mkdir -p Metrics/${experiment_number}

# Make model directory in temporary folder
mkdir $TMPDIR/Models
mkdir $TMPDIR/Models/${experiment_number}

# Copy all required files
cp sde_simulator.py $TMPDIR
cp dataset_generator.py $TMPDIR
cp -r Filters/ $TMPDIR
cp -r SDEs/ $TMPDIR
cp -r Models/${experiment_number}/* $TMPDIR/Models/${experiment_number}
cp metric_evaluator.py $TMPDIR
cp main_eval_lv3.py $TMPDIR

cd $TMPDIR

mkdir Metrics
mkdir Metrics/${experiment_number}
mkdir Logs
mkdir Logs/${experiment_number}

# Load required modules
module load SciPy-bundle/2023.07-gfbf-2023a
module load matplotlib/3.7.2-gfbf-2023a
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# Run the Python script
python main_eval_lv3.py $experiment_number $eval_number

# Copy back metrics and logs
cp -r Metrics/${experiment_number}/* $SLURM_SUBMIT_DIR/Metrics/${experiment_number}
cp -r Logs/${experiment_number}/* $SLURM_SUBMIT_DIR/Logs/${experiment_number}
