# Training the Deployment Model

This script trains the deployment model using the specified TRSTR condition from the parameter search, and generates tree visualizations.

## Execution Output

```text
============================================================
FunaDB C4.5 Deployment Training
============================================================

[1/3] Loading datasets...
Loaded real train: (221, 19)  |  class dist: {0: 143, 1: 78}
Loaded validation: (47, 19)  |  class dist: {0: 31, 1: 16}
Loaded test: (48, 19)  |  class dist: {0: 31, 1: 17}
Loaded synthetic train: (94, 13)  |  class dist: {1: 94}
Mode         : Synthetic-Augmented (TRSTR) Full
Prediction   : Probability threshold at 0.40
Combined Train shape: (410, 19)

[2/3] Training C4.5 decision tree...
Tree depth  : 12
Tree leaves : 13

Global Feature Importances:
  NS   : 0.4825
  DM   : 0.2701
  SUB  : 0.1409
  ADD  : 0.0761
  CA   : 0.0233
  NC   : 0.0071

[3/3] Saving model and visualization...
Model successfully saved to ../models/funa_c45.pkl
Locked threshold: 0.4
Model saved → /home/caineirb/Documents/DysCalc/DysCalc-ML-Development/models/funa_c45.pkl
Tree visualizations saved → ../outputs/figures/deployment/funa_c45_full_tree.svg and ../outputs/figures/deployment/funa_c45_full_tree.png

Done.
```
