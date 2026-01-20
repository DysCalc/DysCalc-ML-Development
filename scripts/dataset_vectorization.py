import pandas as pd
import numpy as np
import os
from pathlib import Path

class Vectorizer:
    def __init__(self, labeled_dataset: Path, normalize: bool = False):
        self.labeled_dataset = labeled_dataset
        self.normalize = normalize
        self.epsilon = 1e-9
    
    def complete_vector(self, save_df: Path | None = None):
        self.df = self._load_dataset()      # Load the labeled dataset
        self._clean_dataset()               # Clean the dataset, remove < 0 values

        # Derived Features
        self.df["NP"] = self._overall_processing_efficiency(self.df['NC'], self.df['DM'])
        self.df["SN"] = self._sym_vs_non_sym(self.df['NC'], self.df['DM'])
        self.df["AF"] = self._overall_arithmetic_fluency(self.df['NS'], self.df['ADD'], self.df['SUB'], self.df['CA'])
        self.df["BC"] = self._arithmetic_contrast(self.df['ADD'], self.df['SUB'], self.df['CA'])
        self.df["AS"] = self._add_sub_asym(self.df['ADD'], self.df['SUB'])
        self.df["PF"] = self._proc_flue_ratio(self.df['AF'], self.df['NP'])

        col_order = ['NC', 'DM', 'NS', 'ADD', 'SUB', 'CA', 'NP', 'SN', 'AF', 'BC', 'AS', 'PF', 'Label']
        self.df = self.df[col_order]    # reorder

        if self.normalize:
            for col in col_order:
                if col != 'Label':
                    self.df[col] = self._normalize(self.df[col])

        if save_df and save_df.parent.exists():
            self.df.to_csv(save_df, index=False)

        return self.df.to_numpy()

    def _load_dataset(self) -> pd.DataFrame:
        return pd.read_csv(self.labeled_dataset)

    def _clean_dataset(self):
        for col in self.df.columns:
            if col == "Label":
                continue

            # compute median excluding < 0
            valid = self.df[col][self.df[col] >= 0]
            median = valid.median()

            # replace < 0 with median
            self.df.loc[self.df[col] < 0, col] = median

    def _overall_processing_efficiency(self, NC, DM):
        return (NC + DM) / 2
    
    def _sym_vs_non_sym(self, NC, DM):
        return NC - DM
    
    def _overall_arithmetic_fluency(self, NS, ADD, SUB, CA):
        return (NS + ADD + SUB + CA) / 4
    
    def _arithmetic_contrast(self, ADD, SUB, CA):
        return ((ADD + SUB) / 2) - CA
    
    def _add_sub_asym(self, ADD, SUB):
        return ADD - SUB
    
    def _proc_flue_ratio(self, AF, NP):
        return AF / (NP + self.epsilon)
    
    def _normalize(self, col):
        numpy_col = col.to_numpy()

        normalized = []
        mean = np.mean(numpy_col)
        std = np.std(numpy_col)

        for row in numpy_col:
            normalized.append((row - mean) / std)

        return np.array(normalized)


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    DATASET_DIR = BASE_DIR.parent / 'dataset'
    os.makedirs(DATASET_DIR, exist_ok=True)
    LABELED_DATASET = DATASET_DIR / 'FUNADB_labled.csv'
    VECTOR_DATASET = DATASET_DIR / 'complete_vector.csv'

    vector = Vectorizer(LABELED_DATASET).complete_vector(VECTOR_DATASET)

    print(vector)
