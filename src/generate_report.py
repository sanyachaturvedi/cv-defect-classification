import os
import pandas as pd
import json

def generate_report():
    """
    Reads prediction results and generates a dynamic HTML results dashboard.
    """
    predictions_path = os.path.join("outputs", "predictions.csv")
    cm_path = "confusion_matrix.png" # Relative path for HTML
    output_html = os.path.join("outputs", "results.html")
    
    if not os.path.exists(predictions_path):
        print(f"Error: {predictions_path} not found. Please run src.predict first.")
        return

    # Load predictions
    df = pd.read_csv(predictions_path)
    total_preds = len(df)
    
    # Infer true labels from path if possible (e.g., test/class_name/image.png)
    # This assumes the second-to-last component of the path is the class label
    df['true_label'] = df['image_path'].apply(lambda x: x.split('/')[-2] if '/' in x else None)
    
    has_accuracy = df['true_label'].notnull().all()
    accuracy = None
    if has_accuracy:
        accuracy = (df['true_label'] == df['predicted_label']).mean() * 100

    # Get label distribution
    dist = df['predicted_label'].value_counts()
    dist_html = "".join([f"<li><strong>{label}:</strong> {count}</li>" for label, count in dist.items()])

    # Get 20 sample predictions
    samples = df.sample(n=min(20, total_preds), random_state=42)
    samples_table = ""
    for _, row in samples.iterrows():
        status_class = "status-match" if has_accuracy and row['true_label'] == row['predicted_label'] else "status-mismatch"
        true_label_cell = f"<td>{row['true_label']}</td>" if has_accuracy else ""
        samples_table += f"""
            <tr>
                <td><code>{row['image_path']}</code></td>
                {true_label_cell}
                <td class="{status_class}">{row['predicted_label']}</td>
            </tr>
        """

    # HTML Template
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Defect Classification Results</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6; background: #f4f7f9; color: #333; margin: 0; padding: 20px; }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        header {{ text-align: center; margin-bottom: 40px; }}
        h1 {{ color: #2c3e50; margin-bottom: 5px; }}
        .subtitle {{ color: #7f8c8d; font-size: 1.1em; }}
        .card {{ background: #fff; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); padding: 25px; margin-bottom: 30px; }}
        h2 {{ border-bottom: 2px solid #eee; padding-bottom: 10px; color: #34495e; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }}
        .metric-box {{ text-align: center; padding: 20px; background: #fcfcfc; border: 1px solid #eee; border-radius: 6px; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #3498db; display: block; }}
        .metric-label {{ color: #95a5a6; text-transform: uppercase; font-size: 0.85em; letter-spacing: 1px; }}
        img.confusion-matrix {{ display: block; margin: 20px auto; max-width: 100%; border-radius: 4px; border: 1px solid #ddd; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f9f9f9; font-weight: 600; color: #2c3e50; }}
        .status-match {{ color: #27ae60; font-weight: 600; }}
        .status-mismatch {{ color: #e74c3c; font-weight: 600; }}
        ul.dist-list {{ list-style: none; padding: 0; display: flex; flex-wrap: wrap; gap: 15px; }}
        ul.dist-list li {{ background: #ebf5fb; padding: 8px 15px; border-radius: 20px; color: #2980b9; font-size: 0.9em; }}
        .obs-list li {{ margin-bottom: 10px; }}
        footer {{ text-align: center; margin-top: 40px; color: #95a5a6; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Rigetti Defect Classification Results</h1>
            <p class="subtitle">Image Classification Performance Dashboard</p>
        </header>

        <section class="card">
            <h2>Model Summary</h2>
            <div class="metrics-grid">
                <div class="metric-box">
                    <span class="metric-value">ResNet18</span>
                    <span class="metric-label">Architecture</span>
                </div>
                <div class="metric-box">
                    <span class="metric-value">{total_preds}</span>
                    <span class="metric-label">Total Predictions</span>
                </div>
                {"<div class='metric-box'><span class='metric-value'>{:.2f}%</span><span class='metric-label'>Inferred Accuracy</span></div>".format(accuracy) if accuracy is not None else ""}
            </div>
        </section>

        <section class="card">
            <h2>Predicted Class Distribution</h2>
            <ul class="dist-list">
                {dist_html}
            </ul>
        </section>

        <section class="card">
            <h2>Error Analysis</h2>
            <p>Confusion matrix indicating the model's classification patterns across categories:</p>
            <img src="{cm_path}" alt="Confusion Matrix" class="confusion-matrix">
        </section>

        <section class="card">
            <h2>Sample Predictions</h2>
            <table>
                <thead>
                    <tr>
                        <th>Image Path</th>
                        {"<th>True Label</th>" if has_accuracy else ""}
                        <th>Predicted Label</th>
                    </tr>
                </thead>
                <tbody>
                    {samples_table}
                </tbody>
            </table>
        </section>

        <section class="card">
            <h2>Observations</h2>
            <ul class="obs-list">
                <li><strong>Class Consistency:</strong> Check the distribution above to ensure no single class is dominating predictions unexpectedly.</li>
                <li><strong>Common Mistakes:</strong> Review the confusion matrix for overlap between similar textures (e.g., rust vs. scratch).</li>
                <li><strong>Data Representative:</strong> These 20 samples were randomly selected from the test dataset.</li>
            </ul>
        </section>

        <section class="card">
            <h2>Project Deliverables</h2>
            <p>Full results and artifacts available in the <code>outputs/</code> directory:</p>
            <ul>
                <li><code>predictions.csv</code>: Raw inference data</li>
                <li><code>confusion_matrix.png</code>: Graphical performance analysis</li>
                <li><code>best_model.pt</code>: Optimized model weights</li>
            </ul>
        </section>

        <footer>
            &copy; 2026 Rigetti Defect Classification Project
        </footer>
    </div>
</body>
</html>
    """

    with open(output_html, "w") as f:
        f.write(html_content)
    
    print(f"Success: Report generated at {output_html}")

if __name__ == "__main__":
    generate_report()
