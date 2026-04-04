import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sdv.single_table import CopulaGANSynthesizer 
from sdv.metadata import SingleTableMetadata
import random
import torch
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

df = pd.read_csv('dataset/complete_vector.csv')

# 85% Train, 15% Test TSTR
train_df, test_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df['Label'])

print(f"Original Training Set: {len(train_df)} students")
print(f"Locked Test Set (Purely Real): {len(test_df)} students")

minority_train_df = train_df[train_df['Label'] == 1]
majority_train_count = len(train_df[train_df['Label'] == 0])

synthetic_samples_needed = majority_train_count - len(minority_train_df)
print(f"\nGenerating {synthetic_samples_needed} synthetic At-Risk samples using CopulaGAN...")

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=minority_train_df)

synthesizer = CopulaGANSynthesizer(metadata, enforce_rounding=False, epochs=500, verbose=True)
synthesizer.fit(minority_train_df)

synthetic_minority_data = synthesizer.sample(num_rows=synthetic_samples_needed)

balanced_train_df = pd.concat([train_df, synthetic_minority_data], ignore_index=True)

print("\nNew Balanced TRAINING Distribution:")
print(balanced_train_df['Label'].value_counts())

folder_name = 'dataset'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

#balanced_train_df.to_csv(f"{folder_name}/balanced_complete_vector_train.csv", index=False)
#test_df.to_csv(f"{folder_name}/complete_vector_test.csv", index=False)

print("\nSuccess! Files saved in the 'dataset' folder:")
print("1. 'balanced_complete_vector_train.csv'") # 85% of the original students, contains synthetic data
print("2. 'complete_vector_test.csv'") # remaining 15% of the original, untouched students. to avoid data leakage