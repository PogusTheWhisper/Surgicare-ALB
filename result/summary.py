import os
import pandas as pd

root_dir = '.'
summary_data = []

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

        row = df.pivot_table(index=None, columns='Matric', values='Score', aggfunc='first')
        for col in row.columns:
            row[f"{model_name}_{col}"] = row[col]
            row.drop(columns=col, inplace=True)

        row['Variant'] = variant_name
        summary_data.append(row)

if not summary_data:
    raise ValueError("🚨 No valid metrics.csv files found.")

summary_df = pd.concat(summary_data, ignore_index=True)

summary_df = summary_df.set_index('Variant')

for col in summary_df.columns:
    summary_df[col] = pd.to_numeric(summary_df[col], errors='coerce')
    summary_df[col] = summary_df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "")

# Save to CSV
summary_df.to_csv("summary.csv")

print("✅ summary.csv")
