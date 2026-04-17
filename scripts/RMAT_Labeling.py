# Convert the raw FUNA_DB dataset to a dataset for supervised learning

import os
from pathlib import Path
import pandas as pd
import numpy as np

## Configs
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR.parent / 'datasets'
os.makedirs(DATASET_DIR, exist_ok=True)
RAW_DATASET = DATASET_DIR / 'FUNADB_rawdata_SUPPL.csv'
LABELED_DATASET = DATASET_DIR / 'FUNADB_labled.csv'
KEEP_COLUMNS = ["NC_t1", "DM_t1", "NS_t1", "ADD_t1", "SUB_t1", "CA_t1", "RMAT"]

## Read and Drop Unusable Columns
df = pd.read_csv(RAW_DATASET, index_col=False)
df = df[KEEP_COLUMNS]
df = df.rename(columns=lambda c: c.removesuffix("_t1") if c != "RMAT" else c)   # remove the _t1 in the column names

## Label with 1 ("At-Risk") or 0 ("Typical") based on RMAT score
RMAT_scores = df['RMAT'].to_numpy()

### Compute population mean and std for RMAT scores
RMAT_mean = np.mean(RMAT_scores)
RMAT_std = np.std(RMAT_scores)

### Normalize scores
RMAT_normalized = []
for score in RMAT_scores:
    RMAT_normalized.append((score - RMAT_mean) / RMAT_std)

### Labeling, with 35th percentile as threshold
RMAT_labels = []
threshold = np.percentile(RMAT_normalized, 35)
for normalized in RMAT_normalized:
    RMAT_labels.append(1 if normalized <= threshold else 0)

## Add the "Label" to the dataframe then drop RMAT since it will not be used 
df['Label'] = np.array(RMAT_labels)
df = df.drop(columns=['RMAT'])

df.to_csv(LABELED_DATASET, index=False)