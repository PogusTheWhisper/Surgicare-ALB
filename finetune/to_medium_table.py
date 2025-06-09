import pandas as pd
import os
from glob import glob

input_folder = 'asr_results_raw'
output_folder = 'asr_results'

os.makedirs(output_folder, exist_ok=True)

csv_files = glob(os.path.join(input_folder, '*_evaluation_summary.csv'))

cer_rows = []
wer_rows = []
all_datasets = set()

for file in csv_files:
    df = pd.read_csv(file)
    all_datasets.update(df['dataset'].tolist())

all_datasets = sorted(all_datasets)

for file in csv_files:
    model_name = os.path.basename(file).replace('_asr_evaluation_summary.csv', '')
    df = pd.read_csv(file)

    samples = int(df['samples'].iloc[0]) if 'samples' in df.columns else None

    cer_row = {'model': model_name, 'samples': samples}
    wer_row = {'model': model_name, 'samples': samples}

    for dataset in all_datasets:
        cer_val = df[df['dataset'] == dataset]['CER'].values
        wer_val = df[df['dataset'] == dataset]['WER'].values
        cer_row[dataset] = cer_val[0] * 100 if len(cer_val) else None
        wer_row[dataset] = wer_val[0] * 100 if len(wer_val) else None

    cer_rows.append(cer_row)
    wer_rows.append(wer_row)

cer_df = pd.DataFrame(cer_rows).round(2)
wer_df = pd.DataFrame(wer_rows).round(2)

cer_txt_path = os.path.join(output_folder, 'cer_table.txt')
wer_txt_path = os.path.join(output_folder, 'wer_table.txt')
cer_csv_path = os.path.join(output_folder, 'cer_table.csv')
wer_csv_path = os.path.join(output_folder, 'wer_table.csv')

cer_df.to_csv(cer_csv_path, index=False)
wer_df.to_csv(wer_csv_path, index=False)

with open(cer_txt_path, 'w') as f:
    f.write(cer_df.to_markdown(index=False))

with open(wer_txt_path, 'w') as f:
    f.write(wer_df.to_markdown(index=False))

print("Saved to asr_results/:")
print("- cer_table.txt")
print("- wer_table.txt")
print("- cer_table.csv")
print("- wer_table.csv")
