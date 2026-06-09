import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Set styles and colors
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 120

AT_RIST_COLOR = '#ED7D31'
TYPICAL_COLOR = '#5B9BD5'
REAL_COLOR = '#54A24B'
SYNTHETIC_COLOR = '#E45756'

ROOT_DIR = Path('/home/caineirb/Documents/DysCalc/DysCalc-ML-Development')
DATASET_DIR = ROOT_DIR / 'datasets' / 'processed'
OUTPUT_DIR = ROOT_DIR / 'outputs' / 'dataset_analysis_figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

datasets = {
    'Train': DATASET_DIR / 'train.csv',
    # 'S-Train': DATASET_DIR / 's_train.csv',
    'Val': DATASET_DIR / 'val.csv',
    'Test': DATASET_DIR / 'test.csv'
}

data = []
for name, path in datasets.items():
    if path.exists():
        df = pd.read_csv(path)
        if 'Label' in df.columns:
            counts = df['Label'].value_counts().to_dict()
            for label, count in counts.items():
                is_synthetic = (name == 'S-Train')
                label_name = 'Typical (0)' if label == 0 else 'At-Risk (1)'
                data.append({
                    'Dataset': name,
                    'Label': label_name,
                    'Type': 'Synthetic' if is_synthetic else 'Real',
                    'Count': count
                })
        else:
            # Fallback if no label column
            data.append({
                'Dataset': name,
                'Label': 'Unknown',
                'Type': 'Synthetic' if name == 'S-Train' else 'Real',
                'Count': len(df)
            })
    else:
        print(f"Warning: {path} not found.")

if not data:
    print("No data found to plot.")
    exit()

df_plot = pd.DataFrame(data)

# -------------------------------------------------------------------
# Plot 1: Class Distribution (Typical vs Real At-Risk vs Synthetic At-Risk)
# -------------------------------------------------------------------
def get_category(row):
    # if row['Type'] == 'Synthetic' and row['Label'] == 'At-Risk (1)':
    #     return 'Synthetic At-Risk (1)'
    if row['Label'] == 'Typical (0)':
        return 'Typical (0)'

    if row['Label'] == 'At-Risk (1)':
        return 'At-Risk (1)'

    return 'Unknown'

df_plot['Category'] = df_plot.apply(get_category, axis=1)

palette_class = {
    'Typical (0)': TYPICAL_COLOR,
    'At-Risk (1)': AT_RIST_COLOR,
    # 'Synthetic At-Risk (1)': SYNTHETIC_COLOR,
    'Unknown': '#808080'
}

plt.figure(figsize=(10, 6))
sns.barplot(data=df_plot, x='Dataset', y='Count', hue='Category', palette=palette_class, dodge=True)

# plt.title('Class Distribution Across Dataset Splits')
plt.xlabel('Dataset Split')
plt.ylabel('Number of Samples')
plt.legend(title='Class & Source', loc='upper right')

# Add bar labels
for p in plt.gca().patches:
    if p.get_height() > 0:
        plt.gca().annotate(f'{int(p.get_height())}', 
                           (p.get_x() + p.get_width() / 2., p.get_height()), 
                           ha='center', va='center', 
                           xytext=(0, 5), textcoords='offset points')

plt.tight_layout()
output_path_class = OUTPUT_DIR / 'dataset_splits_distribution.png'
plt.savefig(output_path_class)
print(f"Diagram saved to {output_path_class}")

# -------------------------------------------------------------------
# Plot 2: Real vs Synthetic Overall Distribution
# -------------------------------------------------------------------
df_real_sync = df_plot.groupby(['Dataset', 'Type'])['Count'].sum().reset_index()

palette_rs = {
    'Real': REAL_COLOR,
    'Synthetic': SYNTHETIC_COLOR
}

plt.figure(figsize=(10, 6))
sns.barplot(data=df_real_sync, x='Dataset', y='Count', hue='Type', palette=palette_rs, dodge=True)

plt.title('Real vs Synthetic Samples Across Dataset Splits')
plt.xlabel('Dataset Split')
plt.ylabel('Number of Samples')
plt.legend(title='Data Source', loc='upper right')

for p in plt.gca().patches:
    if p.get_height() > 0:
        plt.gca().annotate(f'{int(p.get_height())}', 
                           (p.get_x() + p.get_width() / 2., p.get_height()), 
                           ha='center', va='center', 
                           xytext=(0, 5), textcoords='offset points')

plt.tight_layout()
output_path_rs = OUTPUT_DIR / 'dataset_splits_real_synthetic.png'
plt.savefig(output_path_rs)
print(f"Diagram saved to {output_path_rs}")
