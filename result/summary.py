import os
import pandas as pd

root_dir = 'result'
summary_data = []

# Traverse model and variant directories
for model_name in os.listdir(root_dir):
    model_path = os.path.join(root_dir, model_name)
    if not os.path.isdir(model_path):
        continue

    for variant_name in os.listdir(model_path):
        variant_path = os.path.join(model_path, variant_name)
        metrics_file = os.path.join(variant_path, 'metrics.csv')

        if not os.path.exists(metrics_file):
            print(f"⛔ Missing: {metrics_file}")
            continue

        print(f"📂 Reading: {metrics_file}")
        df = pd.read_csv(metrics_file, header=None)
        df.columns = ['Matric', 'Score']

        # Pivot to single-row format
        row = df.pivot_table(index=None, columns='Matric', values='Score', aggfunc='first')
        row['Model'] = model_name
        row['Variant'] = variant_name
        summary_data.append(row)

# If nothing was read
if not summary_data:
    raise ValueError("🚨 No valid metrics.csv files found.")

# Combine all rows
summary_df = pd.concat(summary_data, ignore_index=True)

# Reorder columns if present
ordered_cols = ['Model', 'Variant', 'Precision', 'Recall', 'F1 Score', 'Accuracy']
summary_df = summary_df[[col for col in ordered_cols if col in summary_df.columns]]

# ✅ Pad all metric values to 4 decimal places (as strings)
for col in ['Precision', 'Recall', 'F1 Score', 'Accuracy']:
    if col in summary_df.columns:
        summary_df[col] = pd.to_numeric(summary_df[col], errors='coerce')
        summary_df[col] = summary_df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "")

# Save to CSV (no quoting, plain padded strings)
summary_df.to_csv('summary.csv', index=False)

print("✅ summary.csv created with all metric values padded to 4 decimal digits.")
