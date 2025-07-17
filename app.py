from flask import Flask, render_template, request
import pandas as pd
from growth_analysis import analyze_growth, fit_average_growth
from statistics import compare_methods, generate_csv_report

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    micro_file = request.files['micro_data']
    large_file = request.files['large_data']

    micro_df = pd.read_csv(micro_file)
    large_df = pd.read_csv(large_file)

    ba_param = request.form.get('ba_param', 'mu')  # default to μ
    model_choice = request.form.get('model_choice', 'auto')  # user-selected model

    # Analyze replicate-level growth
    micro_params, micro_curves = analyze_growth(micro_df, label="Microwell", model_choice=model_choice)
    large_params, large_curves = analyze_growth(large_df, label="Large", model_choice=model_choice)

    # Convert to DataFrame
    micro_df_params = pd.DataFrame(micro_params)
    large_df_params = pd.DataFrame(large_params)

    # ✅ Compute average model fits (this was missing!)
    avg_micro = fit_average_growth(micro_df, label="Microwell")
    avg_large = fit_average_growth(large_df, label="Large")

    stats = compare_methods(micro_df_params, large_df_params, ba_param)
    report_path = generate_csv_report(micro_df_params, large_df_params, stats)

    return render_template(
        'result.html',
        micro=micro_params,
        large=large_params,
        stats=stats,
        report_path=report_path,
        micro_curves=micro_curves,
        large_curves=large_curves,
        avg_micro=avg_micro,          # ✅ now defined
        avg_large=avg_large,          # ✅ now defined
        zip=zip                       # ✅ allows use of zip in Jinja2
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
