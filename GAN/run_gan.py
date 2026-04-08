import pandas as pd
import numpy as np
import random
import torch
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import os
from scipy.spatial.distance import jensenshannon
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import train_test_split
from sdv.single_table import CopulaGANSynthesizer, CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("Loading data...")
df = pd.read_csv('dataset/complete_vector.csv') 
features = [col for col in df.columns if col != 'Label']

# 70% Train, 30% Temporary (Validation + Test)
df_train, df_temp = train_test_split(df, test_size=0.30, stratify=df['Label'], random_state=SEED)

df_val, df_test = train_test_split(df_temp, test_size=0.50, stratify=df_temp['Label'], random_state=SEED)

df_train_maj = df_train[df_train['Label'] == 0]
df_train_min = df_train[df_train['Label'] == 1]

n_synthetic_needed = len(df_train_maj) - len(df_train_min)
print(f"Data Split Complete (70/15/15):")
print(f" - Train: {len(df_train)} rows")
print(f" - Validation: {len(df_val)} rows")
print(f" - Test: {len(df_test)} rows")
print(f"\nGenerating {n_synthetic_needed} synthetic samples to balance the 70% Train set...\n")

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df_train_min)

n_copula = n_synthetic_needed // 2
n_ctgan = n_synthetic_needed - n_copula

print(f"--- [1/2] Training CopulaGAN (Generating {n_copula} samples) ---")
copula_synth = CopulaGANSynthesizer(metadata, enforce_rounding=False, epochs=1500, verbose=True)
copula_synth.fit(df_train_min)
df_syn_copula = copula_synth.sample(num_rows=n_copula)

print(f"\n--- [2/2] Training CTGAN (Generating {n_ctgan} samples) ---")
ctgan_synth = CTGANSynthesizer(metadata, enforce_rounding=False, epochs=1500, verbose=True)
ctgan_synth.fit(df_train_min)
df_syn_ctgan = ctgan_synth.sample(num_rows=n_ctgan)

df_syn_min = pd.concat([df_syn_copula, df_syn_ctgan], ignore_index=True)

print("\n--- Statistical Similarity & Summary Checks ---")
EPSILON = 1e-9

for f in features:
    real_f = df_train_min[f].values
    syn_f = df_syn_min[f].values
    
    # A. Kolmogorov-Smirnov
    ks_stat, p_value = stats.ks_2samp(real_f, syn_f)
    ks_pass = p_value > 0.05
    
    # B. Jensen-Shannon Divergence 
    bins = np.histogram_bin_edges(np.concatenate([real_f, syn_f]), bins=20)
    p_real, _ = np.histogram(real_f, bins=bins, density=True)
    p_syn, _ = np.histogram(syn_f, bins=bins, density=True)
    jsd_val = jensenshannon(p_real, p_syn) ** 2  
    jsd_pass = jsd_val < 0.1
    
    # C. Summary Statistics Differences 
    delta_mean = abs(np.mean(syn_f) - np.mean(real_f)) / (abs(np.mean(real_f)) + EPSILON)
    delta_std = abs(np.std(syn_f) - np.std(real_f)) / (np.std(real_f) + EPSILON)
    
    print(f"Feature: {f}")
    print(f"  KS p-value: {p_value:.4f} [{'PASS' if ks_pass else 'FAIL'}]")
    print(f"  JSD:        {jsd_val:.4f} [{'PASS' if jsd_pass else 'FAIL'}]")
    print(f"  Δ Mean:     {delta_mean:.4f} | Δ Std: {delta_std:.4f}")

print("\nGenerating Visual Assessments...")

plt.figure(figsize=(8, 5))
sns.histplot(df_train_min['NC'], color='blue', label='Real At-Risk', kde=True, stat="density", alpha=0.5)
sns.histplot(df_syn_min['NC'], color='red', label='Hybrid Synthetic At-Risk', kde=True, stat="density", alpha=0.5)
plt.title("Distribution Match: Number Comparison (NC)")
plt.legend()
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(16, 6))
sns.heatmap(df_train_min[features].corr(), annot=False, cmap='coolwarm', vmin=-1, vmax=1, ax=ax[0])
ax[0].set_title("Real Minority Correlation Matrix")
sns.heatmap(df_syn_min[features].corr(), annot=False, cmap='coolwarm', vmin=-1, vmax=1, ax=ax[1])
ax[1].set_title("Hybrid Synthetic Minority Correlation Matrix")
plt.show()

print("\n--- Model-Based Utility Evaluation (TSTR) ---")
X_test, y_test = df_test[features], df_test['Label']
X_train_real, y_train_real = df_train[features], df_train['Label']

df_train_tstr = pd.concat([df_train_maj, df_syn_min], ignore_index=True)
X_train_syn, y_train_syn = df_train_tstr[features], df_train_tstr['Label']

clf_trtr = DecisionTreeClassifier(random_state=SEED, max_depth=10)
clf_trtr.fit(X_train_real, y_train_real)
pred_trtr = clf_trtr.predict(X_test)
pm_r = f1_score(y_test, pred_trtr)

clf_tstr = DecisionTreeClassifier(random_state=SEED, max_depth=10)
clf_tstr.fit(X_train_syn, y_train_syn)
pred_tstr = clf_tstr.predict(X_test)
pm_s = f1_score(y_test, pred_tstr)

delta_pm = abs(pm_s - pm_r) / (pm_r + EPSILON)

print(f"TRTR Baseline F1-Score: {pm_r:.4f}")
print(f"TSTR Augmented F1-Score: {pm_s:.4f}")
print(f"Relative Difference (Δ_PM): {delta_pm:.4f} [{'PASS' if delta_pm < 0.1 else 'WARNING (Improvement!)'}]")

print("\n--- Exporting Files for Jupyter Notebook ---")

os.makedirs('dataset', exist_ok=True)

df_balanced_train = pd.concat([df_train_maj, df_train_min, df_syn_min], ignore_index=True)
df_balanced_train = df_balanced_train.sample(frac=1, random_state=SEED).reset_index(drop=True)

df_balanced_train.to_csv('dataset/FUNADB_balanced_TRAIN.csv', index=False)
df_val.to_csv('dataset/FUNADB_real_VAL.csv', index=False)
df_test.to_csv('dataset/FUNADB_real_TEST.csv', index=False)

print(f"TRAIN SET (Balanced 70%): {len(df_balanced_train)} rows -> 'dataset/FUNADB_balanced_TRAIN.csv'")
print(f"VALIDATION SET (Real 15%): {len(df_val)} rows -> 'dataset/FUNADB_real_VAL.csv'")
print(f"TEST SET (Real 15%): {len(df_test)} rows -> 'dataset/FUNADB_real_TEST.csv'")