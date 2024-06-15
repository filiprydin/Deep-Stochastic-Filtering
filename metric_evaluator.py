import os

import numpy as np
import pandas as pd
import time
import csv

from scipy.optimize import minimize
from scipy.stats import entropy

from sde_simulator import SDESimulator

from Filters.particle_filter import ParticleFilter

# Defines a metric evaluator class used to find Mean Absolut Error (MAE), Moment Errors (ME), Distribution Error Norms (DEN) and Kullback-Leibler divergence (KLD)

class MetricEvaluator:
    def __init__(self, sde, parameters, reference_times, reference_measurement_indices, ebds_filters, benchmark_filters, reference_filter):
        self.sde = sde
        self.state_dim = sde.state_dim
        self.meas_dim = sde.meas_dim
        self.reference_times = reference_times
        self.n_times_reference = len(reference_times)
        self.reference_measurement_indices = reference_measurement_indices
        self.measurement_times = self.reference_times[self.reference_measurement_indices]
        self.n_measurements = len(self.measurement_times)

        self.ebds_filters = ebds_filters
        self.benchmark_filters = benchmark_filters
        self.reference_filter = reference_filter

        self.grid_based = parameters["Grid-based"]
        self.min_value_integration = parameters["Integration range"][0]
        self.max_value_integration = parameters["Integration range"][1]
        self.points_per_dim = parameters["Integration points"]
        self.mc_samples = parameters["MC samples"]
        self.element_size = ((self.max_value_integration - self.min_value_integration)/(self.points_per_dim-1))**self.state_dim

        self.MAEs = {}
        self.MAEs_reference = np.zeros(self.n_measurements, dtype=np.float32)
        self.FMEs = {} # First moment error
        self.L2L2s = {} # L^2L^2 distribution error norm
        self.L2Linfs = {} # L^2L^infty distribution error norm
        self.KLDs = {}
        self.filter_times = {}
        self.filter_times["reference"] = 0
        self.mean_times = {}
        self.mean_times["reference"] = 0

        self.L2s2 = {}
        self.Linfs2 = {}
        self.L2svar = {} # L^2 distribution error norm variance
        self.Linfsvar = {} # L^infty distribution error norm variance

        for filter_name, filter in benchmark_filters.items():
            self.MAEs[filter_name] = np.zeros(self.n_measurements, dtype=np.float32)
            self.FMEs[filter_name] = np.zeros(self.n_measurements, dtype=np.float32) 
            self.L2Linfs[filter_name] = np.zeros(self.n_measurements, dtype=np.float32)
            self.L2L2s[filter_name] = np.zeros(self.n_measurements, dtype=np.float32)    
            self.KLDs[filter_name] = np.zeros(self.n_measurements, dtype=np.float32)  
            self.filter_times[filter_name] = 0
            self.mean_times[filter_name] = 0

            self.Linfs2[filter_name] = np.zeros(self.n_measurements, dtype=np.float32)
            self.L2s2[filter_name] = np.zeros(self.n_measurements, dtype=np.float32)
            self.Linfsvar[filter_name] = np.zeros(self.n_measurements, dtype=np.float32)
            self.L2svar[filter_name] = np.zeros(self.n_measurements, dtype=np.float32)  

        for filter_name, filter in ebds_filters.items():
            self.MAEs[filter_name] = np.zeros(self.n_measurements, dtype=np.float32)
            self.FMEs[filter_name] = np.zeros(self.n_measurements, dtype=np.float32) 
            self.L2Linfs[filter_name] = np.zeros(self.n_measurements, dtype=np.float32)
            self.L2L2s[filter_name] = np.zeros(self.n_measurements, dtype=np.float32)    
            self.KLDs[filter_name] = np.zeros(self.n_measurements, dtype=np.float32)
            self.mean_times[filter_name] = 0  

            self.Linfs2[filter_name] = np.zeros(self.n_measurements, dtype=np.float32)
            self.L2s2[filter_name] = np.zeros(self.n_measurements, dtype=np.float32)
            self.Linfsvar[filter_name] = np.zeros(self.n_measurements, dtype=np.float32)
            self.L2svar[filter_name] = np.zeros(self.n_measurements, dtype=np.float32)   

    def evaluate_metrics(self, n_samples): 
        if self.grid_based:
            return self.evaluate_metrics_grid_based(n_samples)
        else:
            return self.evaluate_metrics_grid_free(n_samples)   

    def evaluate_metrics_grid_based(self, n_samples):
        simulator = SDESimulator(self.sde, self.reference_times, self.reference_measurement_indices)
        X, Y = simulator.simulate_state_and_measurement(N_simulations=n_samples)

        for i in range(n_samples):
            Yi = Y[i, :, :] 

            if ((i + 1) % 1 == 0):
                print(f"Evaluating metric on sample {i + 1}", flush=True)

            for filter_name, filter in self.benchmark_filters.items():
                start_time = time.time()
                filter.filter(Yi)
                end_time = time.time()
                self.filter_times[filter_name] += end_time - start_time
                if isinstance(filter, ParticleFilter):
                    filter.build_kdes(self.measurement_times)
            start_time = time.time()
            self.reference_filter.filter(Yi)
            end_time = time.time()
            self.filter_times["reference"] += end_time - start_time
            if isinstance(self.reference_filter, ParticleFilter):
                self.reference_filter.build_kdes(self.measurement_times)

            start_time = time.time()
            reference_filter_means = self.reference_filter.get_filter_means(self.measurement_times, Yi.T)
            end_time = time.time()
            self.mean_times["reference"] += end_time - start_time

            pdfs_reference, points_reference = self.reference_filter.get_filter_pdfs(self.measurement_times, Yi.T, min_value=self.min_value_integration, max_value=self.max_value_integration, points_per_dim=self.points_per_dim)
            states = X[i, :, self.reference_measurement_indices].T
            self.MAEs_reference += np.linalg.norm(reference_filter_means - states, axis=0, ord=2)

            for filter_name, filter in self.benchmark_filters.items():
                # MAE
                start_time = time.time()
                filter_means = filter.get_filter_means(self.measurement_times, Yi.T)
                end_time = time.time()
                self.mean_times[filter_name] += end_time - start_time
                self.MAEs[filter_name] += np.linalg.norm(filter_means - states, axis=0, ord=2)

                # FME
                self.FMEs[filter_name] += np.linalg.norm(filter_means - reference_filter_means, axis=0, ord=2)
                if np.any(np.isnan(np.linalg.norm(filter_means - reference_filter_means, axis=0, ord=2))):
                    print("Normalisation failed")
                    print(Yi)

                # DENs
                pdfs_filter, points_filter = filter.get_filter_pdfs(self.measurement_times, Yi.T, self.min_value_integration, self.max_value_integration, self.points_per_dim)
                pdfs_diff = pdfs_filter - pdfs_reference
                L2s = np.sum(np.power(pdfs_diff,2), axis=0)*self.element_size
                Linfs = np.max(np.power(pdfs_diff,2), axis=0)
                self.L2L2s[filter_name] += L2s
                self.L2Linfs[filter_name] += Linfs
                self.L2s2[filter_name] += np.power(L2s,2)
                self.Linfs2[filter_name] += np.power(Linfs,2)

                # KLDs
                #self.KLDs[filter_name] += np.sum(np.multiply(pdfs_filter ,np.log(np.divide(pdfs_filter, pdfs_reference))), axis=0) * element_size
                pdfs_filter += 10 ** (-30) # Add a very small number to prevent pdf from being 0
                self.KLDs[filter_name] += entropy(pdfs_reference, pdfs_filter, axis=0)

            for filter_name, filter in self.ebds_filters.items():
                # MAE
                start_time = time.time()
                filter_means = filter.get_filter_means(self.measurement_times, Yi.T)
                end_time = time.time()
                self.mean_times[filter_name] += end_time - start_time
                self.MAEs[filter_name] += np.linalg.norm(filter_means - states, axis=0, ord=2)

                # FME
                self.FMEs[filter_name] += np.linalg.norm(filter_means - reference_filter_means, axis=0, ord=2)
                if np.any(np.isnan(np.linalg.norm(filter_means - reference_filter_means, axis=0, ord=2))):
                    print("Normalisation failed")
                    print(Yi)

                # DENs
                pdfs_filter, points_filter = filter.get_filter_pdfs(self.measurement_times, Yi.T, self.min_value_integration, self.max_value_integration, self.points_per_dim)
                pdfs_diff = pdfs_filter - pdfs_reference
                L2s = np.sum(np.power(pdfs_diff,2), axis=0)*self.element_size
                Linfs = np.max(np.power(pdfs_diff,2), axis=0)
                self.L2L2s[filter_name] += L2s
                self.L2Linfs[filter_name] += Linfs
                self.L2s2[filter_name] += np.power(L2s,2)
                self.Linfs2[filter_name] += np.power(Linfs,2)

                # KLDs
                #self.KLDs[filter_name] += np.sum(np.multiply(pdfs_filter ,np.log(np.divide(pdfs_filter, pdfs_reference))), axis=0) * element_size
                pdfs_filter += 10 ** (-30) # Add a very small number to prevent pdf from being 0
                self.KLDs[filter_name] += entropy(pdfs_reference, pdfs_filter, axis=0)

        self.MAEs_reference = self.MAEs_reference / n_samples
        self.filter_times["reference"] = self.filter_times["reference"] / n_samples
        self.mean_times["reference"] = self.mean_times["reference"] / (n_samples * self.n_measurements)
        for filter_name, filter in self.benchmark_filters.items():
            self.MAEs[filter_name] = self.MAEs[filter_name] / n_samples
            self.FMEs[filter_name] = self.FMEs[filter_name] / n_samples
            self.L2L2s[filter_name] = np.sqrt(self.L2L2s[filter_name] / n_samples)
            self.L2Linfs[filter_name] = np.sqrt(self.L2Linfs[filter_name] / n_samples)
            self.KLDs[filter_name] = self.KLDs[filter_name] / n_samples
            self.filter_times[filter_name] = self.filter_times[filter_name] / n_samples
            self.mean_times[filter_name] = self.mean_times[filter_name] / (n_samples * self.n_measurements)

            self.L2svar[filter_name] = self.L2s2[filter_name] / n_samples - np.power(self.L2L2s[filter_name] / n_samples, 2)
            self.Linfsvar[filter_name] = self.Linfs2[filter_name] / n_samples - np.power(self.L2Linfs[filter_name] / n_samples, 2)

        for filter_name, filter in self.ebds_filters.items():
            self.MAEs[filter_name] = self.MAEs[filter_name] / n_samples
            self.FMEs[filter_name] = self.FMEs[filter_name] / n_samples
            self.L2L2s[filter_name] = np.sqrt(self.L2L2s[filter_name] / n_samples)
            self.L2Linfs[filter_name] = np.sqrt(self.L2Linfs[filter_name] / n_samples)
            self.KLDs[filter_name] = self.KLDs[filter_name] / n_samples
            self.mean_times[filter_name] = self.mean_times[filter_name] / (n_samples * self.n_measurements)

            self.L2svar[filter_name] = self.L2s2[filter_name] / n_samples - np.power(self.L2L2s[filter_name] / n_samples, 2)
            self.Linfsvar[filter_name] = self.Linfs2[filter_name] / n_samples - np.power(self.L2Linfs[filter_name] / n_samples, 2)

        return self.MAEs, self.MAEs_reference, self.FMEs, self.L2L2s, self.L2Linfs, self.KLDs

    def evaluate_metrics_grid_free(self, n_samples):
        simulator = SDESimulator(self.sde, self.reference_times, self.reference_measurement_indices)
        X, Y = simulator.simulate_state_and_measurement(N_simulations=n_samples)
        
        for i in range(n_samples):
            Yi = Y[i, :, :] 

            if ((i + 1) % 1 == 0):
                print(f"Evaluating metric on sample {i + 1}", flush=True)

            for filter_name, filter in self.benchmark_filters.items():
                start_time = time.time()
                filter.filter(Yi)
                end_time = time.time()
                self.filter_times[filter_name] += end_time - start_time
                if isinstance(filter, ParticleFilter):
                    filter.build_kdes(self.measurement_times)
            start_time = time.time()
            self.reference_filter.filter(Yi)
            end_time = time.time()
            self.filter_times["reference"] += end_time - start_time
            if isinstance(self.reference_filter, ParticleFilter):
                self.reference_filter.build_kdes(self.measurement_times)

            start_time = time.time()
            reference_filter_means = self.reference_filter.get_filter_means(self.measurement_times, Yi.T)
            end_time = time.time()
            self.mean_times["reference"] += end_time - start_time
            states = X[i, :, self.reference_measurement_indices].T
            self.MAEs_reference += np.linalg.norm(reference_filter_means - states, axis=0, ord=2)

            reference_samples, sample_pdf_reference = self.reference_filter.sample_distributions(self.measurement_times, Yi.T, self.mc_samples)

            for filter_name, filter in self.benchmark_filters.items():
                # MAE
                start_time = time.time()
                filter_means = filter.get_filter_means(self.measurement_times, Yi.T)
                end_time = time.time()
                self.mean_times[filter_name] += end_time - start_time
                self.MAEs[filter_name] += np.linalg.norm(filter_means - states, axis=0, ord=2)

                # FME
                self.FMEs[filter_name] += np.linalg.norm(filter_means - reference_filter_means, axis=0, ord=2)
                if np.any(np.isnan(np.linalg.norm(filter_means - reference_filter_means, axis=0, ord=2))):
                    print("Normalisation failed")
                    print(Yi)

                # L2Linf DEN, L2L2 DEN and KLD
                L2s = np.zeros(len(self.measurement_times), dtype=np.float32)
                KLDs = np.zeros(len(self.measurement_times), dtype=np.float32)
                Linfs = np.zeros(len(self.measurement_times), dtype=np.float32)
                for i in range(len(self.measurement_times)):
                    x = reference_samples[:,:,i].T
                    sample_pdf_filter = filter.evaluate_filter_pdf(x, self.measurement_times[i], Yi.T)
                    diff = sample_pdf_reference[:,i] - sample_pdf_filter
                    L2s[i] = np.mean(np.power(diff, 2)/(sample_pdf_reference[:,i]))
                    sample_pdf_filter += 10 ** (-30) # Add a very small number to prevent pdf from being 0
                    KLDs[i] = np.mean(np.log(sample_pdf_reference[:,i] / sample_pdf_filter))

                    # To be minimized
                    def objective_function(x):
                        f_pdf = filter.evaluate_filter_pdf(x, self.measurement_times[i], Yi.T)
                        r_pdf = self.reference_filter.evaluate_filter_pdf(x, self.measurement_times[i], Yi.T)
                        return -(f_pdf - r_pdf) ** 2
                    
                    x0 = x[np.argmax(np.abs(diff)), :]
                    result = minimize(objective_function, x0, method='nelder-mead') # Leave method blank for gradient-based
                    Linfs[i] = -result.fun

                self.L2L2s[filter_name] += L2s
                self.L2Linfs[filter_name] += Linfs
                self.KLDs[filter_name] += KLDs
                self.L2s2[filter_name] += np.power(L2s,2)
                self.Linfs2[filter_name] += np.power(Linfs,2)

            for filter_name, filter in self.ebds_filters.items():
                # MAE
                start_time = time.time()
                filter_means, filter_constants = filter.get_filter_means_and_integrals(self.measurement_times, Yi.T)
                end_time = time.time()
                self.mean_times[filter_name] += end_time - start_time
                self.MAEs[filter_name] += np.linalg.norm(filter_means - states, axis=0, ord=2)

                # FME
                self.FMEs[filter_name] += np.linalg.norm(filter_means - reference_filter_means, axis=0, ord=2)
                if np.any(np.isnan(np.linalg.norm(filter_means - reference_filter_means, axis=0, ord=2))):
                    print("Normalisation failed")
                    print(Yi)

                # L2Linf DEN, L2L2 DEN and KLD
                L2s = np.zeros(len(self.measurement_times), dtype=np.float32)
                KLDs = np.zeros(len(self.measurement_times), dtype=np.float32)
                Linfs = np.zeros(len(self.measurement_times), dtype=np.float32)
                for i in range(len(self.measurement_times)):
                    x = reference_samples[:,:,i].T
                    sample_pdf_filter = filter.evaluate_filter_pdf_unnorm(x, self.measurement_times[i], Yi.T) / filter_constants[i]
                    diff = sample_pdf_reference[:,i] - sample_pdf_filter
                    L2s[i] = np.mean(np.power(diff, 2)/(sample_pdf_reference[:,i]))
                    sample_pdf_filter += 10 ** (-30) # Add a very small number to prevent pdf from being 0
                    KLDs[i] = np.mean(np.log(sample_pdf_reference[:,i] / sample_pdf_filter))

                    # To be minimized
                    def objective_function(x):
                        x = np.expand_dims(x, axis=0)
                        f_pdf = filter.evaluate_filter_pdf_unnorm(x, self.measurement_times[i], Yi.T) / filter_constants[i]
                        r_pdf = self.reference_filter.evaluate_filter_pdf(x, self.measurement_times[i], Yi.T)
                        return -(f_pdf - r_pdf) ** 2
                    
                    x0 = x[np.argmax(np.abs(diff)), :]
                    result = minimize(objective_function, x0, method='nelder-mead') # Leave method blank for gradient-based
                    Linfs[i] = -result.fun

                self.L2L2s[filter_name] += L2s
                self.L2Linfs[filter_name] += Linfs
                self.KLDs[filter_name] += KLDs
                self.L2s2[filter_name] += np.power(L2s,2)
                self.Linfs2[filter_name] += np.power(Linfs,2)

        self.MAEs_reference = self.MAEs_reference / n_samples
        self.filter_times["reference"] = self.filter_times["reference"] / n_samples
        self.mean_times["reference"] = self.mean_times["reference"] / (n_samples * self.n_measurements)
        for filter_name, filter in self.benchmark_filters.items():
            self.MAEs[filter_name] = self.MAEs[filter_name] / n_samples
            self.FMEs[filter_name] = self.FMEs[filter_name] / n_samples
            self.L2L2s[filter_name] = np.sqrt(self.L2L2s[filter_name] / n_samples)
            self.L2Linfs[filter_name] = np.sqrt(self.L2Linfs[filter_name] / n_samples)
            self.KLDs[filter_name] = self.KLDs[filter_name] / n_samples
            self.filter_times[filter_name] = self.filter_times[filter_name] / n_samples
            self.mean_times[filter_name] = self.mean_times[filter_name] / (n_samples * self.n_measurements)

            self.L2svar[filter_name] = self.L2s2[filter_name] / n_samples - np.power(self.L2L2s[filter_name] / n_samples, 2)
            self.Linfsvar[filter_name] = self.Linfs2[filter_name] / n_samples - np.power(self.L2Linfs[filter_name] / n_samples, 2)

        for filter_name, filter in self.ebds_filters.items():
            self.MAEs[filter_name] = self.MAEs[filter_name] / n_samples
            self.FMEs[filter_name] = self.FMEs[filter_name] / n_samples
            self.L2L2s[filter_name] = np.sqrt(self.L2L2s[filter_name] / n_samples)
            self.L2Linfs[filter_name] = np.sqrt(self.L2Linfs[filter_name] / n_samples)
            self.KLDs[filter_name] = self.KLDs[filter_name] / n_samples
            self.mean_times[filter_name] = self.mean_times[filter_name] / (n_samples * self.n_measurements)

            self.L2svar[filter_name] = self.L2s2[filter_name] / n_samples - np.power(self.L2L2s[filter_name] / n_samples, 2)
            self.Linfsvar[filter_name] = self.Linfs2[filter_name] / n_samples - np.power(self.L2Linfs[filter_name] / n_samples, 2)

        return self.MAEs, self.MAEs_reference, self.FMEs, self.L2L2s, self.L2Linfs, self.KLDs

    def save_metric_results(self, folder_name):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        folder_name = os.path.join(current_dir, "Metrics", folder_name)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        maes_path = os.path.join(folder_name, "MAEs")
        fmes_path = os.path.join(folder_name, "FMEs")
        l2l2s_path = os.path.join(folder_name, "L2L2s")
        l2linfs_path = os.path.join(folder_name, "L2Linfs")
        klds_path = os.path.join(folder_name, "KLDs")
        l2vars_path = os.path.join(folder_name, "L2vars")
        linfvars_path = os.path.join(folder_name, "Linfvars")
        filter_times_path = os.path.join(folder_name, "Filter_times")
        mean_times_path = os.path.join(folder_name, "Mean_times")

        with open(filter_times_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Filter', 'Average filter time (s)'])
            for key, value in self.filter_times.items():
                writer.writerow([key, value])
        
        with open(mean_times_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Filter', 'Average time to calculate mean (s)'])
            for key, value in self.mean_times.items():
                writer.writerow([key, value])

        columns = ["Timepoint"]
        for filter_name, filter in self.benchmark_filters.items():
            columns.append(filter_name)
        for filter_name, filter in self.ebds_filters.items():
            columns.append(filter_name)
        columns_mae = columns.copy()
        columns_mae.append("reference")
        
        maes = pd.DataFrame(columns = columns_mae)
        fmes = pd.DataFrame(columns = columns)
        l2l2s = pd.DataFrame(columns = columns)
        l2linfs = pd.DataFrame(columns = columns)
        klds = pd.DataFrame(columns = columns)
        l2vars = pd.DataFrame(columns = columns)
        linfvars = pd.DataFrame(columns = columns)

        row_mae = {}
        row_fme = {}
        row_l2l2 = {}
        row_l2linf = {}
        row_kld = {}
        row_l2var = {}
        row_linfvar = {}
        for i in range(self.n_measurements):
            row_mae["Timepoint"] = self.measurement_times[i]
            row_mae["reference"] = self.MAEs_reference[i]
            row_fme["Timepoint"] = self.measurement_times[i]
            row_l2l2["Timepoint"] = self.measurement_times[i]
            row_l2linf["Timepoint"] = self.measurement_times[i]
            row_kld["Timepoint"] = self.measurement_times[i]
            row_l2var["Timepoint"] = self.measurement_times[i]
            row_linfvar["Timepoint"] = self.measurement_times[i]

            for filter_name, filter in self.benchmark_filters.items():
                row_mae[filter_name] = self.MAEs[filter_name][i]
                row_fme[filter_name] = self.FMEs[filter_name][i]
                row_l2l2[filter_name] = self.L2L2s[filter_name][i]
                row_l2linf[filter_name] = self.L2Linfs[filter_name][i]
                row_kld[filter_name] = self.KLDs[filter_name][i]
                row_l2var[filter_name] = self.L2svar[filter_name][i]
                row_linfvar[filter_name] = self.Linfsvar[filter_name][i]

            for filter_name, filter in self.ebds_filters.items():
                row_mae[filter_name] = self.MAEs[filter_name][i]
                row_fme[filter_name] = self.FMEs[filter_name][i]
                row_l2l2[filter_name] = self.L2L2s[filter_name][i]
                row_l2linf[filter_name] = self.L2Linfs[filter_name][i]
                row_kld[filter_name] = self.KLDs[filter_name][i]
                row_l2var[filter_name] = self.L2svar[filter_name][i]
                row_linfvar[filter_name] = self.Linfsvar[filter_name][i]

            maes.loc[len(maes)] = row_mae
            fmes.loc[len(fmes)] = row_fme
            l2l2s.loc[len(l2l2s)] = row_l2l2
            l2linfs.loc[len(l2linfs)] = row_l2linf
            klds.loc[len(klds)] = row_kld
            l2vars.loc[len(l2vars)] = row_l2var
            linfvars.loc[len(linfvars)] = row_linfvar

        maes.to_csv(maes_path, index=False)
        fmes.to_csv(fmes_path, index=False)
        l2l2s.to_csv(l2l2s_path, index=False)
        l2linfs.to_csv(l2linfs_path, index=False)
        klds.to_csv(klds_path, index=False)
        l2vars.to_csv(l2vars_path, index=False)
        linfvars.to_csv(linfvars_path, index=False)
