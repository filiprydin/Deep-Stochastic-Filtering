# Deep-Stochastic-Filtering
Source code for the master's thesis "A Deep Learning Method for Nonlinear Stochastic Filtering: Energy-Based Deep Splitting for Fast and Accurate Estimation of Filtering Densities" at Chalmers University of Technology.

Link: <http://hdl.handle.net/20.500.12380/308090>

## Usage
The "Filters" folder contains implementations of the EBDS and EBDS-LSTM methods, as well as benchmark filters. The "SDEs" folder contains examples of filtering problems. The user trains EBDS models with the main_train files and evalutes them with the main_eval files. This will automatically create folders containing neural network binary files, metric csv files and log txt files. The file metric_visualizer.py provides simple assistance with visualising the metrics. 

The code is designed to easily run many large-scale experiments on a multitude of example problems. To change problem, simply input the correct SDE file in the main files. The three shell scripts were used to run experiments on Chalmers VERA cluster with Slurm. 
