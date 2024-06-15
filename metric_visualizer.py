import os
import csv

import pandas as pd
import matplotlib.pyplot as plt
import tikzplotlib
import numpy as np

class MetricVisualizer:
    def __init__(self, experiment_number, evaluation_number, only_best = False, average_runs = True, include_benchmarks = True):
        self.experiment_number = experiment_number
        self.evaluation_number = evaluation_number
        self.experiment_folder = os.path.join("Metrics", str(experiment_number), str(evaluation_number))

        self.only_best = only_best
        self.average_runs = average_runs
        self.include_benchmarks = include_benchmarks

        self.available_metrics = []
        for file in os.listdir(self.experiment_folder):
            self.available_metrics.append(file[:-1])        

    def plot_metric(self, metric, long_display, short_display, get_tikz_code = False):
        if metric not in self.available_metrics:
            raise ValueError("Metric not available")

        metric_file_path = os.path.join(self.experiment_folder, metric + "s")
        
        dataframe = pd.read_csv(metric_file_path)
    
        filter_names = list(dataframe.columns)[1:]
        timepoints = dataframe["Timepoint"]

        # Do different things depending on plot mode
        filters = {}
        if not self.only_best and not self.average_runs:
            for filter_name in filter_names:
                filters[filter_name] = dataframe[filter_name]
        elif self.only_best and not self.average_runs:
            max_steps = 0

            # Get max steps
            for filter_name in filter_names:
                parts = filter_name.split()
                if len(parts) == 3 and parts[0] == "EBDS":
                    steps = int(parts[1])
                    if steps > max_steps:
                        max_steps = steps

            # Add filters to list
            for filter_name in filter_names:
                parts = filter_name.split()
                if len(parts) == 3 and parts[0] == "EBDS" and parts[1] == str(max_steps):
                    values = dataframe[filter_name]
                    filters[f"{parts[0]} {parts[2]}"] = values
                elif parts[0] != "EBDS":
                    filters[filter_name] = dataframe[filter_name]
        elif self.only_best and self.average_runs:
            max_steps = 0

            # Get max steps
            for filter_name in filter_names:
                parts = filter_name.split()
                if len(parts) == 3 and parts[0] == "EBDS":
                    steps = int(parts[1])
                    if steps > max_steps:
                        max_steps = steps
            
            # Add filters to list and average ebds
            ebds_values = np.zeros(len(timepoints))
            n_ebds = 0
            for filter_name in filter_names:
                parts = filter_name.split()
                if len(parts) == 3 and parts[0] == "EBDS" and parts[1] == str(max_steps):
                    ebds_values += dataframe[filter_name]
                    n_ebds += 1
                elif parts[0] != "EBDS":
                    filters[filter_name] = dataframe[filter_name]
            filters["EBDS"] = ebds_values / n_ebds
        else:
            # Add filters to list and average ebds
            ebds_values = {}
            n_ebds = {}
            for filter_name in filter_names:
                parts = filter_name.split()
                if len(parts) == 3 and parts[0] == "EBDS":
                    if parts[1] not in ebds_values:
                        ebds_values[parts[1]] = dataframe[filter_name]
                        n_ebds[parts[1]] = 1
                    else:
                        ebds_values[parts[1]] += dataframe[filter_name]
                        n_ebds[parts[1]] += 1
                elif parts[0] != "EBDS":
                    filters[filter_name] = dataframe[filter_name]

            # Add ebds filters to list
            for n_steps, values in ebds_values.items():
                filters[f"EBDS {n_steps}"] = values / n_ebds[n_steps]

        # Remove benchmarks if they are to not be included
        if not self.include_benchmarks:
            for filter_name in list(filters.keys()):
                parts = filter_name.split()
                if parts[0] != "EBDS":
                    filters.pop(filter_name)

        #plt.style.use('metropolis')
        plt.rcParams['figure.figsize'] = (8, 6)

        # Plot metrics
        for filter_name, values in filters.items():
            plt.plot(timepoints, values, label=filter_name)
        plt.title(f"{long_display} over time")
        plt.grid(True) 
        plt.legend()
        plt.ylabel(short_display)
        plt.xlabel("Time")
        plt.tight_layout()
        if get_tikz_code:
            print(tikzplotlib.get_tikz_code())
        plt.show()

    def plot_convergence(self, time, metric, short_display, reference_order = 1/2, get_tikz_code = False):
        if metric not in self.available_metrics:
            raise ValueError("Metric not available")

        metric_file_path = os.path.join(self.experiment_folder, metric + "s")
        
        dataframe = pd.read_csv(metric_file_path)
    
        filter_names = list(dataframe.columns)[1:]
        timepoints = dataframe["Timepoint"]

        # add benchmarks
        if self.include_benchmarks:
            benchmarks = {}
            for filter_name in filter_names:
                parts = filter_name.split()
                if parts[0] != "EBDS":
                    benchmarks[filter_name] = dataframe[filter_name].loc[dataframe['Timepoint'] == time].iloc[0]

        # calculate averages and store values
        ebds_averages = {}
        n_ebds = {}
        n_steps = []
        values = []
        for filter_name in filter_names:
            parts = filter_name.split()
            if len(parts) == 3 and parts[0] == "EBDS":
                n_steps.append(int(parts[1]))
                error = dataframe[filter_name].loc[dataframe['Timepoint'] == time].iloc[0]
                values.append(error)
                if int(parts[1]) in ebds_averages:
                    ebds_averages[int(parts[1])] += error
                    n_ebds[int(parts[1])] += 1
                else:
                    ebds_averages[int(parts[1])] = error
                    n_ebds[int(parts[1])] = 1
        
        # Divide by number of occurences
        for steps in ebds_averages.keys():
            ebds_averages[steps] = ebds_averages[steps] / n_ebds[steps]
        
        min_steps = min(ebds_averages.keys())
        max_steps = max(ebds_averages.keys())
        min_value = ebds_averages[min_steps]

        #plt.style.use('metropolis')
        plt.rcParams['figure.figsize'] = (8, 6)

        if self.include_benchmarks:
            for filter_name, value in benchmarks.items():
                plt.axhline(y = value, linestyle = '-', label=filter_name)

        plt.scatter(n_steps, values, s=10, color='red', label='Data')
        plt.scatter(ebds_averages.keys(), ebds_averages.values(), color='blue', label='Average')
        # Plot the regression line
        x_values = np.linspace(min_steps, max_steps, 100)
        y_values = 1.5 * min_value * x_values ** (-reference_order)
        plt.plot(x_values, y_values, color='black', label=r"$O(N^{-1/2})$")
        plt.xlabel('Number of steps')
        plt.ylabel(short_display)
        plt.title(f'Convergence plot at time {time}')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        if get_tikz_code:
            print(tikzplotlib.get_tikz_code())
        plt.show()

    def print_EOC_table(self, time, metric, short_display):
        if metric not in self.available_metrics:
            raise ValueError("Metric not available")

        metric_file_path = os.path.join(self.experiment_folder, metric + "s")
        
        dataframe = pd.read_csv(metric_file_path)
    
        filter_names = list(dataframe.columns)[1:]

        # calculate averages and store values
        ebds_averages = {}
        n_ebds = {}
        for filter_name in filter_names:
            parts = filter_name.split()
            if len(parts) == 3 and parts[0] == "EBDS":
                error = dataframe[filter_name].loc[dataframe['Timepoint'] == time].iloc[0]
                if int(parts[1]) in ebds_averages:
                    ebds_averages[int(parts[1])] += error
                    n_ebds[int(parts[1])] += 1
                else:
                    ebds_averages[int(parts[1])] = error
                    n_ebds[int(parts[1])] = 1
        
        # Divide by number of occurences
        for steps in ebds_averages.keys():
            ebds_averages[steps] = ebds_averages[steps] / n_ebds[steps]
        sorted_averages = dict(sorted(ebds_averages.items()))
        
        steps = list(sorted_averages.keys())
        values = list(sorted_averages.values())
        print("Latex formatted output")
        print(f"EOCs at time {time}")
        print(f"N & {short_display} & EOC \\\\")
        print("\\hline")
        for i in range(len(steps)):
            if i < len(sorted_averages) - 1:
                EOC = -(np.log(values[i+1]) - np.log(values[i])) / (np.log(steps[i+1]) - np.log(steps[i]))
                print(steps[i], "&", np.round(values[i],4), "&", np.round(EOC,4), "\\\\")
            else:
                print(steps[i], "&", np.round(values[i],4), "& \\\\")
        print("")

    def format_timing_data(self, n_observations):
        ftimes_file_path = os.path.join(self.experiment_folder, "Filter_times")
        mtimes_file_path = os.path.join(self.experiment_folder, "Mean_times")

        filter_times = {}
        with open(ftimes_file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader) 
            for row in reader:
                key = row[0]
                value = float(row[1]) 
                filter_times[key] = value

        mean_times = {}
        with open(mtimes_file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            for row in reader:
                key = row[0]
                value = float(row[1]) 
                mean_times[key] = value

        # TODO: settings should probably affect this too

        print("Latex formatted output")
        print("Average time for filtering and calculating mean")
        # TODO: One could argue that the filtering time should be divided by number of observations for a fair comparison
        print("Filter & M time (s) & F time (s) & Total time (s)")
        for filter_name, time in mean_times.items():
            print(filter_name, end=" & ")
            print(np.round(n_observations * time,4), end=" & ")
            if filter_name in list(filter_times.keys()):
                f_time = filter_times[filter_name]
                print(np.round(f_time, 4), end=" & ")
            else:
                f_time = 0
                print("-", end=" & ")
            print(np.round(f_time + n_observations * time, 4), end=" \\\\")
            print("")
        print("")
            
            

if __name__ == "__main__":
    experiment_number = 1
    evaluation_number = 0

    only_best = True
    average_runs = True
    include_benchmarks = True

    visualiser = MetricVisualizer(experiment_number, evaluation_number, only_best, average_runs, include_benchmarks)

    visualiser.plot_metric("MAE", "Mean Absolute Error", "MAE", get_tikz_code = True)
    visualiser.plot_metric("FME", "First Moment Error", "FME", get_tikz_code = True)
    visualiser.plot_metric("L2L2", r"$L^2(\Omega;L^2(\mathbb{R}^d;\mathbb{R}))$-error", r"$L^2(\Omega;L^2(\mathbb{R}^d;\mathbb{R}))$-error", get_tikz_code = True)
    visualiser.plot_metric("L2Linf", r"$L^2(\Omega;L^\infty(\mathbb{R}^d;\mathbb{R}))$-error", r"$L^2(\Omega;L^\infty(\mathbb{R}^d;\mathbb{R}))$-error", get_tikz_code = True)
    visualiser.plot_metric("KLD", "Kullback-Leibler Divergence", "KLD", get_tikz_code = True)
    #visualiser.plot_metric("L2var", r"$L^2(\mathbb{R}^d;\mathbb{R})$-error Variance", r"$L^2(\mathbb{R}^d;\mathbb{R})$-error var", get_tikz_code = True)
    #visualiser.plot_metric("Linfvar", r"$L^\infty(\mathbb{R}^d;\mathbb{R})$-error Variance", r"$L^\infty(\mathbb{R}^d;\mathbb{R})$-error var", get_tikz_code = True)

    #visualiser.plot_convergence(1, "L2L2", r"$L^2(\Omega;L^2(\mathbb{R}^d;\mathbb{R}))$-error", get_tikz_code = True)
    #visualiser.plot_convergence(1, "L2Linf", r"$L^2(\Omega;L^\infty(\mathbb{R}^d;\mathbb{R}))$-error", get_tikz_code = True)

    #visualiser.print_EOC_table(1, "L2L2", r"$L^2(\Omega;L^2(\mathbb{R}^d;\mathbb{R}))$-error")
    #visualiser.print_EOC_table(1, "L2Linf", r"$L^2(\Omega;L^\infty(\mathbb{R}^d;\mathbb{R}))$-error")

    #visualiser.format_timing_data(n_observations = 11)