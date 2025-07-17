import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

# Growth models
def logistic_model(t, A, mu, lag):
    return A / (1 + np.exp(-mu * (t - lag)))

def gompertz_model(t, A, mu, lag):
    return A * np.exp(-np.exp((mu * np.e / A) * (lag - t) + 1))

def baranyi_model(t, A, mu, lag):
    h0 = 1  # constant h0
    return A / (1 + np.exp(-mu * (t - lag) + np.log(1 + np.exp(-mu * lag + h0))))

# Fit multiple models
def fit_models(time, density):
    models = {
        'Logistic': logistic_model,
        'Gompertz': gompertz_model,
        'Baranyi': baranyi_model
    }

    fit_results = []

    for name, model in models.items():
        try:
            popt, _ = curve_fit(model, time, density, p0=[max(density), 0.2, 1.0], maxfev=10000)
            fitted = model(time, *popt)

            residuals = density - fitted
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((density - np.mean(density))**2)
            r2 = 1 - (ss_res / ss_tot)
            mse = mean_squared_error(density, fitted)
            aic = len(density) * np.log(ss_res / len(density)) + 2 * len(popt)

            fit_results.append({
                'model': name,
                'params': popt,
                'mse': mse,
                'r2': r2,
                'aic': aic
            })

        except:
            fit_results.append({
                'model': name,
                'params': None,
                'mse': np.inf,
                'r2': 0,
                'aic': np.inf
            })

    best_fit = min(fit_results, key=lambda x: x['mse'])
    return best_fit, fit_results

def fit_average_growth(df, label=""):
    """Fit growth models to the mean of all replicates (excluding 'Time')."""
    time = df['Time'].values
    densities = df.iloc[:, 1:].values  # all replicate columns
    mean_density = np.nanmean(densities, axis=1)

    best_fit, all_fits = fit_models(time, mean_density)

    # Plot average fit
    t_fit = np.linspace(min(time), max(time), 100)
    plt.figure()
    plt.plot(time, mean_density, 'o', label='Mean Observed')

    for fit in all_fits:
        if fit['params'] is not None:
            model_func = {
                'Logistic': logistic_model,
                'Gompertz': gompertz_model,
                'Baranyi': baranyi_model
            }[fit['model']]
            y_fit = model_func(t_fit, *fit['params'])
            plt.plot(t_fit, y_fit, label=f"{fit['model']} (R²={fit['r2']:.2f})")

    plt.title(f"{label} Average Growth – Best: {best_fit['model']}")
    plt.xlabel("Time (hr)")
    plt.ylabel("Cell Density (Mean)")
    plt.legend()
    plt.tight_layout()

    os.makedirs('static', exist_ok=True)
    plot_path = f'static/growth_avg_{label.lower()}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'label': label,
        'time': time,
        'mean_density': mean_density,
        'best_model': best_fit['model'],
        'params': best_fit['params'],
        'mse': best_fit['mse'],
        'r2': best_fit['r2'],
        'aic': best_fit['aic'],
        'plot_path': plot_path
    }

def analyze_growth(df, label="", model_choice="auto"):
    time = df['Time'].values
    results = []
    plot_paths = []

    for i, col in enumerate(df.columns[1:]):  # skip 'Time'
        density = df[col].values

        try:
            if model_choice != "auto":
                # Fit only the user-selected model
                model_func = {
                    'Logistic': logistic_model,
                    'Gompertz': gompertz_model,
                    'Baranyi': baranyi_model
                }[model_choice]

                popt, _ = curve_fit(model_func, time, density, p0=[max(density), 0.2, 1.0], maxfev=10000)
                fitted = model_func(time, *popt)
                residuals = density - fitted
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((density - np.mean(density)) ** 2)
                r2 = 1 - (ss_res / ss_tot)
                mse = mean_squared_error(density, fitted)
                aic = len(density) * np.log(ss_res / len(density)) + 2 * len(popt)

                A, mu, lag = popt
                doubling = np.log(2) / mu if mu != 0 else np.nan

                # Plot selected model
                t_fit = np.linspace(min(time), max(time), 100)
                y_fit = model_func(t_fit, *popt)

                plt.figure()
                plt.plot(time, density, 'o', label='Observed')
                plt.plot(t_fit, y_fit, label=f"{model_choice} (R²={r2:.2f})")
                plt.title(f"{label} Replicate {i+1} – Model: {model_choice}")
                plt.xlabel("Time (hr)")
                plt.ylabel("Cell Density")
                plt.legend()
                plt.tight_layout()

                filename = f'static/growth_{label.lower()}_{i+1}.png'
                os.makedirs('static', exist_ok=True)
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()

                results.append({
                    'mu': mu,
                    'lag': lag,
                    'doubling': doubling,
                    'model': model_choice,
                    'fit_metrics': [{
                        'model': model_choice,
                        'params': popt,
                        'r2': r2,
                        'mse': mse,
                        'aic': aic
                    }],
                    'plot_path': filename
                })

            else:
                # Fit all models and choose best by MSE
                best_fit, all_fits = fit_models(time, density)
                if best_fit['params'] is not None:
                    A, mu, lag = best_fit['params']
                    doubling = np.log(2) / mu if mu != 0 else np.nan

                    # Plot all models
                    t_fit = np.linspace(min(time), max(time), 100)
                    plt.figure()
                    plt.plot(time, density, 'o', label='Observed')

                    for fit in all_fits:
                        if fit['params'] is not None:
                            model_func = {
                                'Logistic': logistic_model,
                                'Gompertz': gompertz_model,
                                'Baranyi': baranyi_model
                            }[fit['model']]
                            y_fit = model_func(t_fit, *fit['params'])
                            plt.plot(t_fit, y_fit, label=f"{fit['model']} (R²={fit['r2']:.2f})")

                    plt.title(f"{label} Replicate {i+1} – Best: {best_fit['model']}")
                    plt.xlabel("Time (hr)")
                    plt.ylabel("Cell Density")
                    plt.legend()
                    plt.tight_layout()

                    filename = f'static/growth_{label.lower()}_{i+1}.png'
                    os.makedirs('static', exist_ok=True)
                    plt.savefig(filename, dpi=300, bbox_inches='tight')
                    plt.close()

                    results.append({
                        'mu': mu,
                        'lag': lag,
                        'doubling': doubling,
                        'model': best_fit['model'],
                        'fit_metrics': all_fits,
                        'plot_path': filename
                    })
        except Exception as e:
            results.append({
                'mu': np.nan,
                'lag': np.nan,
                'doubling': np.nan,
                'model': 'Fit Error',
                'fit_metrics': [],
                'plot_path': None
            })
            plot_paths.append(None)

    return results, plot_paths