import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, pearsonr
import os
import csv

def compute_cv(values):
    """Compute coefficient of variation in percent."""
    values = np.array(values)
    return np.nanstd(values) / np.nanmean(values) * 100

def compare_methods(micro_df, large_df, param='mu'):
    import numpy as np
    from scipy.stats import ttest_rel, pearsonr
    import os

    x = micro_df[param].to_numpy()
    y = large_df[param].to_numpy()

    # Clean old Bland-Altman plots
    for file in os.listdir('static'):
        if file.startswith('bland_altman_') and file.endswith('.png'):
            os.remove(os.path.join('static', file))

    # Stats
    t_p = ttest_rel(x, y).pvalue
    corr, _ = pearsonr(x, y)
    micro_cv = compute_cv(x)
    large_cv = compute_cv(y)
    micro_sd = np.std(x)
    large_sd = np.std(y)

    # Dynamic filename
    plot_filename = f'bland_altman_{param}.png'
    plot_path = os.path.join('static', plot_filename)

    # Bland-Altman
    ba_result = bland_altman_analysis(x, y, param, save_path=plot_path)

    return {
        'mu_pvalue': round(t_p, 4),
        'doubling_corr': round(corr, 4),
        'micro_cv': round(micro_cv, 2),
        'large_cv': round(large_cv, 2),
        'repeatability_micro': round(micro_sd, 4),
        'repeatability_large': round(large_sd, 4),
        'bias': round(ba_result['bias'], 4),
        'loa_upper': round(ba_result['loa_upper'], 4),
        'loa_lower': round(ba_result['loa_lower'], 4),
        'repeatability_index': round(ba_result['repeatability_index'], 4),
        'ba_plot_path': plot_path,
        'ba_param': param
    }

def bland_altman_analysis(x, y, param, save_path):
    """Create Bland-Altman plot and return key stats."""
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    x = np.array(x)
    y = np.array(y)
    mean = (x + y) / 2
    diff = x - y
    bias = np.mean(diff)
    sd = np.std(diff)
    loa_upper = bias + 1.96 * sd
    loa_lower = bias - 1.96 * sd
    ri = 1.96 * sd  # Repeatability index

    # 🧠 Dynamic label for clarity
    param_labels = {
        'mu': 'Specific Growth Rate (μ)',
        'lag': 'Lag Time (hr)',
        'doubling': 'Doubling Time (hr)'
    }
    label = param_labels.get(param, param)

    # 📈 Plotting
    plt.figure(figsize=(6, 4))
    plt.scatter(mean, diff, color='blue', label='Replicate differences')
    plt.axhline(bias, color='gray', linestyle='--', label=f"Bias = {bias:.4f}")
    plt.axhline(loa_upper, color='red', linestyle='--', label=f"Upper LoA = {loa_upper:.4f}")
    plt.axhline(loa_lower, color='red', linestyle='--', label=f"Lower LoA = {loa_lower:.4f}")
    plt.fill_between(mean, loa_lower, loa_upper, color='red', alpha=0.1)

    plt.xlabel(f"Mean of {label}")
    plt.ylabel(f"Difference in {label}")
    plt.title(f"Bland-Altman Plot: {label}")
    plt.legend(loc='upper right')
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 📊 Return key stats
    return {
        'bias': bias,
        'loa_upper': loa_upper,
        'loa_lower': loa_lower,
        'repeatability_index': ri
    }

def generate_csv_report(micro_df, large_df, stats, filename='static/report.csv'):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(["Microwell Replicates (μ, Lag, Doubling)"])
        writer.writerow(["Rep#", "μ", "Lag", "Doubling Time"])
        for i, row in micro_df.iterrows():
            writer.writerow([i + 1, round(row['mu'], 4), round(row['lag'], 2), round(row['doubling'], 2)])

        writer.writerow([])
        writer.writerow(["Large Volume Replicates (μ, Lag, Doubling)"])
        writer.writerow(["Rep#", "μ", "Lag", "Doubling Time"])
        for i, row in large_df.iterrows():
            writer.writerow([i + 1, round(row['mu'], 4), round(row['lag'], 2), round(row['doubling'], 2)])

        writer.writerow([])
        writer.writerow(["Bland-Altman Analysis"])
        writer.writerow(["Parameter", stats['ba_param']])
        writer.writerow(["Bias", stats['bias']])
        writer.writerow(["Upper LoA", stats['loa_upper']])
        writer.writerow(["Lower LoA", stats['loa_lower']])
        writer.writerow(["Repeatability Index", stats['repeatability_index']])

        writer.writerow([])
        writer.writerow(["Statistical Summary"])
        for key, value in stats.items():
            if not key.endswith('_path'):
                writer.writerow([key.replace('_', ' ').title(), value])

    return filename