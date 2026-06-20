import pandas as pd
import numpy as np
import random, os, warnings
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import json
from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sdv.single_table import CopulaGANSynthesizer, CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from pathlib import Path

warnings.filterwarnings('ignore')

# Set Seed for reproducability
SEED = 42

EPSILON = 1e-9
RAW_FEATURES = ['NC', 'DM', 'NS', 'ADD', 'SUB', 'CA']
GAN_FEATURES = RAW_FEATURES + ['BC']   # BC included so GAN learns ADD/SUB/CA joint
DERIVED_FEATURES = ['NP', 'SN', 'AF', 'BC', 'AS', 'PF']
ROOT_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT_DIR / 'datasets' / 'processed'
os.makedirs(DATASETS_DIR, exist_ok=True)
OUTPUTS_DIR = ROOT_DIR / 'outputs' / 'logs_and_metrics' / 'deployment'
os.makedirs(OUTPUTS_DIR, exist_ok=True)

AT_RIST_COLOR = '#ED7D31'
TYPICAL_COLOR = '#5B9BD5'
REAL_COLOR = '#54A24B'
SYNTHETIC_COLOR = '#E45756'

# Real Train Data only (both labels 0 and 1)
print('Configuration loaded.')
print(f'  RAW_FEATURES : {RAW_FEATURES}')
print(f'  GAN_FEATURES : {GAN_FEATURES}')

# Load empirical missing rates from file (per label, per feature).
MISSING_RATES = {}
try:
    with open(OUTPUTS_DIR.parent / 'missing_rates.json', encoding='utf-8') as file:
        loaded_rates = json.load(file)

    # Use the 'all' split for full dataset rates if available.
    target_rates = loaded_rates.get('all', loaded_rates)

    # Normalize label keys to strings so downstream lookup is consistent.
    MISSING_RATES = {str(k): v for k, v in target_rates.items()}

    # Validate expected schema: labels -> RAW_FEATURES -> numeric rates in [0, 1].
    for lbl in ['0', '1']:
        if lbl not in MISSING_RATES:
            raise ValueError(f"Missing label key '{lbl}' in {OUTPUTS_DIR.parent / 'missing_rates.json'}")
        for f in RAW_FEATURES:
            if f not in MISSING_RATES[lbl]:
                raise ValueError(f"Missing feature '{f}' under label '{lbl}' in missing_rates.json")
            MISSING_RATES[lbl][f] = float(np.clip(MISSING_RATES[lbl][f], 0.0, 1.0))

    print(f"{MISSING_RATES=}")
except FileNotFoundError:
    print(f"File not found: {OUTPUTS_DIR.parent / 'missing_rates.json'}")
except json.JSONDecodeError:
    print(f"Invalid JSON format: {OUTPUTS_DIR.parent / 'missing_rates.json'}")
except ValueError as err:
    print(f"Invalid missing-rate schema: {err}")
    
def set_seed(seed):
    """Global seed for full reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

print("set_seed helper function loaded.")
def compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute all derived features from raw features deterministically."""
    df_der = df.copy()
    df_der['NP'] = (df_der['NC'] + df_der['DM']) / 2                           # Eq. 3.3
    df_der['SN'] = df_der['NC'] - df_der['DM']                                  # Eq. 3.4
    df_der['AF'] = (df_der['NS'] + df_der['ADD'] + df_der['SUB'] + df_der['CA']) / 4   # Eq. 3.5
    df_der['BC'] = (df_der['ADD'] + df_der['SUB']) / 2 - df_der['CA']              # Eq. 3.6
    df_der['AS'] = df_der['ADD'] - df_der['SUB']                                # Eq. 3.7
    df_der['PF'] = df_der['AF'] / (df_der['NP'] + EPSILON)                     # Eq. 3.8
    return df_der

print("compute_derived helper function loaded.")
def inject_synthetic_missing(df: pd.DataFrame, missing_rates: dict,
                               features: list, df_min_ref: pd.DataFrame,
                               label_val: int = 1,
                               random_state: int = SEED) -> pd.DataFrame:
    """
    Simulate missing data in SYNTHETIC rows:
      1. Mark rows as missing per empirical rates.
      2. Impute with real minority-class median.
    """
    print("\nSimulating synthetic missing data (imputed with medians)...")
    rng = np.random.default_rng(random_state)
    df_der = df.copy()

    # Support both integer and string label keys (e.g., 1 and '1').
    rates = missing_rates.get(str(label_val), missing_rates.get(label_val, {}))

    feature_medians = {f: df_min_ref[f].median() for f in features}
    for f in features:
        rate = float(np.clip(rates.get(f, 0.0), 0.0, 1.0))
        if rate <= 0.0:
            continue
        mask = rng.binomial(1, rate, size=len(df)).astype(bool)
        if mask.sum() == 0:
            continue
        df_der.loc[mask, f] = feature_medians[f]
    print("Finished simulating synthetic missing data.")
    return df_der


print("inject_synthetic_missing helper function loaded.")
def distribution_check(
    df_syn_min: pd.DataFrame,
    df_train_min_full: pd.DataFrame,
    features: list,
    ks_pval: float = 0.05,
    T_mu: float = 0.10,
    T_sigma: float = 0.15,
    T_med: float = 0.1,
    T_skew: float = 0.75,
    T_kurt: float = 1.5,
    JSD_THRESHOLD: float = 0.10,
    max_moment_fails: int = 2,
    use_soft_accept: bool = True,
) -> tuple:
    """
    Statistical similarity check between real and synthetic minority-class data.

    Returns:
        passed: bool
        results_df: pd.DataFrame
        summary: dict
    """

    hard_pass_global = True
    moment_fail_total = 0
    results = []

    hdr = (
        f"{'Feature':<10} "
        f"{f'KS (p>{ks_pval})':>22} "
        f"{'JSD':>18} "
        f"{'Δμ':>18} "
        f"{'Δσ':>18} "
        f"{'Δmed':>18} "
        f"{'Δskew':>18} "
        f"{'Δkurt':>18} "
        f"{'ALL':>6}"
    )

    print(hdr)
    print("-" * len(hdr))

    for f in features:
        real_f = df_train_min_full[f].dropna().values
        syn_f  = df_syn_min[f].dropna().values

        if len(real_f) < 2 or len(syn_f) < 2:
            raise ValueError(f"Feature {f} has too few valid values for distribution checking.")

        # KS test
        ks_stat, p_value = stats.ks_2samp(real_f, syn_f)
        ks_pass = p_value > ks_pval
        ks_str = f"D={ks_stat:.4f} p={p_value:.4f} {'✓' if ks_pass else '✗'}"

        # Summary statistics
        mean_r, mean_s = np.mean(real_f), np.mean(syn_f)
        std_r,  std_s  = np.std(real_f, ddof=1), np.std(syn_f, ddof=1)
        med_r,  med_s  = np.median(real_f), np.median(syn_f)
        skew_r, skew_s = stats.skew(real_f, bias=False), stats.skew(syn_f, bias=False)
        kurt_r, kurt_s = stats.kurtosis(real_f, bias=False), stats.kurtosis(syn_f, bias=False)

        delta_mu   = abs(mean_s - mean_r) / (abs(mean_r) + EPSILON)
        delta_std  = abs(std_s  - std_r)  / (std_r + EPSILON)
        delta_med  = abs(med_s  - med_r)  / (abs(med_r) + EPSILON)
        delta_skew = abs(skew_s - skew_r)
        delta_kurt = abs(kurt_s - kurt_r)

        mu_pass   = delta_mu   < T_mu
        std_pass  = delta_std  < T_sigma
        med_pass  = delta_med  < T_med
        skew_pass = delta_skew < T_skew
        kurt_pass = delta_kurt < T_kurt

        # JSD using probability counts, not density
        bins = np.histogram_bin_edges(np.concatenate([real_f, syn_f]), bins=20)

        p_real, _ = np.histogram(real_f, bins=bins, density=False)
        p_syn,  _ = np.histogram(syn_f,  bins=bins, density=False)

        p_real = p_real.astype(float) + EPSILON
        p_syn  = p_syn.astype(float) + EPSILON

        p_real = p_real / p_real.sum()
        p_syn  = p_syn / p_syn.sum()

        jsd_val = jensenshannon(p_real, p_syn) ** 2
        jsd_pass = jsd_val < JSD_THRESHOLD

        hard_pass = ks_pass and jsd_pass

        moment_passes = [mu_pass, std_pass, med_pass, skew_pass, kurt_pass]
        moment_fail_count = sum(not ok for ok in moment_passes)
        moment_fail_total += moment_fail_count

        hard_pass_global = hard_pass_global and hard_pass

        stats_pass = all(moment_passes)
        feature_pass = hard_pass and stats_pass

        jsd_s = f"{jsd_val:.4f}<{JSD_THRESHOLD} {'✓' if jsd_pass else '✗'}"
        mu_s  = f"{delta_mu:.4f}<{T_mu} {'✓' if mu_pass else '✗'}"
        sig_s = f"{delta_std:.4f}<{T_sigma} {'✓' if std_pass else '✗'}"
        med_s = f"{delta_med:.4f}<{T_med} {'✓' if med_pass else '✗'}"
        skw_s = f"{delta_skew:.4f}<{T_skew} {'✓' if skew_pass else '✗'}"
        krt_s = f"{delta_kurt:.4f}<{T_kurt} {'✓' if kurt_pass else '✗'}"

        print(
            f"{f:<10} "
            f"{ks_str:>22} "
            f"{jsd_s:>18} "
            f"{mu_s:>18} "
            f"{sig_s:>18} "
            f"{med_s:>18} "
            f"{skw_s:>18} "
            f"{krt_s:>18} "
            f"{'✓' if feature_pass else '✗':>6}"
        )

        results.append(dict(
            feature=f,
            KS_stat=ks_stat,
            KS_p=p_value,
            JSD=jsd_val,
            delta_mu=delta_mu,
            delta_std=delta_std,
            delta_med=delta_med,
            delta_skew=delta_skew,
            delta_kurt=delta_kurt,
            hard_pass=hard_pass,
            moment_failures=moment_fail_count,
            ALL_pass=feature_pass,
        ))

    results_df = pd.DataFrame(results)

    # Threshold-normalized score
    score = (
        (results_df["delta_mu"]   / T_mu).sum() +
        (results_df["delta_std"]  / T_sigma).sum() +
        (results_df["delta_med"]  / T_med).sum() +
        (results_df["delta_skew"] / T_skew).sum() +
        (results_df["delta_kurt"] / T_kurt).sum() +
        (results_df["JSD"]        / JSD_THRESHOLD).sum()
    )

    print("-" * len(hdr))
    print("Hard checks:", "PASS ✓" if hard_pass_global else "FAIL ✗")
    print(f"Moment failures total: {moment_fail_total} (max allowed {max_moment_fails})")
    print(f"Normalized soft score: {score:.4f}")

    if use_soft_accept:
        passed = hard_pass_global and (moment_fail_total <= max_moment_fails)
    else:
        passed = hard_pass_global and (moment_fail_total == 0)

    summary = {
        "hard_pass": hard_pass_global,
        "moment_failures": moment_fail_total,
        "score": score,
        "max_moment_fails": max_moment_fails,
        "soft_accept": passed,
    }

    return passed, results_df, summary


print("distribution_check helper function loaded.")

df     = pd.read_csv(DATASETS_DIR / 'cleaned_dataset.csv')
df_raw = df[RAW_FEATURES + ['Label']].copy()

# Keep the source split with its existing incomplete flags intact.
df_train, df_temp = train_test_split(df, test_size=0.30, stratify=df['Label'], random_state=SEED)
df_val,   df_test = train_test_split(df_temp, test_size=0.50, stratify=df_temp['Label'], random_state=SEED)

df_raw_maj = df_raw[df_raw['Label'] == 0]
df_raw_min = df_raw[df_raw['Label'] == 1]
n_synthetic_needed = len(df_raw_maj) - len(df_raw_min)

print('Data split (70 / 15 / 15):')
print(f'  Train      : {len(df_train):>4}')
print(f'  Validation : {len(df_val):>4}')
print(f'  Test       : {len(df_test):>4}')
print(f'  Full       : {len(df_raw):>4}  (maj={len(df_raw_maj)}, min={len(df_raw_min)})')
print(f'  Synthetic samples needed to balance FULL dataset: {n_synthetic_needed}')

# Minority set — Use ALL minority rows from the full dataset for GAN training
df_min_all = df_raw_min[RAW_FEATURES].copy()
df_min_all['BC'] = (df_min_all['ADD'] + df_min_all['SUB']) / 2 - df_min_all['CA']
df_min_ref = df_min_all.copy()   # reference used for all checks

print(f'\nMinority rows available for GAN training: {len(df_min_all)}')

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df_min_all)

# Hyperparameter search grid.
# Each entry: (MULTIPLIER, epochs, batch_size, gen_lr, disc_lr)
GAN_SEARCH_GRID = [
    ( 7, 2000, 500, 2e-4, 2e-4),   # baseline
    (10, 2000, 500, 2e-4, 2e-4),   # bigger pool
    (10, 3000, 500, 2e-4, 2e-4),   # more epochs
    (10, 3000, 100, 2e-4, 2e-4),   # smaller batch
    (15, 3000, 100, 1e-4, 1e-4),   # lower LR
    (15, 5000, 100, 1e-4, 1e-4),   # high epochs
    (20, 5000, 100, 1e-4, 1e-4),   # max effort
    (30, 5000, 100, 1e-4, 1e-4),   # very high effort
    (50, 5000, 100, 1e-4, 1e-4),   # extreme effort
]

# Default thresholds for the statistical measurements
DIST_THRESHOLDS = dict(
    ks_pval = 0.05, T_mu=0.10, T_sigma=0.175, T_med=0.125,
    T_skew=0.75, T_kurt=1.5, JSD_THRESHOLD=0.10
)

# Fit scaler once on real minority data
scaler_global = StandardScaler().fit(df_min_all[GAN_FEATURES])

def run_gan_pipeline(config, seed=SEED, thresholds=None):
    """Train GANs, post-process, filter, and check distribution for one config."""
    if thresholds is None:
        thresholds = DIST_THRESHOLDS
    set_seed(seed)
    MULTIPLIER, epochs, batch_size, gen_lr, disc_lr = config

    n_generate = n_synthetic_needed * MULTIPLIER
    n_copula   = n_generate // 2
    n_ctgan    = n_generate - n_copula

    # Scale training data
    df_scaled = df_min_all.copy()
    df_scaled[GAN_FEATURES] = scaler_global.transform(df_scaled[GAN_FEATURES])

    common_kwargs = dict(
        metadata          = metadata,
        enforce_rounding  = False,
        embedding_dim     = 32,
        generator_dim     = (128, 128),
        discriminator_dim = (128, 128),
        epochs            = epochs,
        batch_size        = min(batch_size, 64),
        generator_lr      = gen_lr / 2,
        discriminator_lr  = disc_lr / 2,
        discriminator_steps = 2,
        pac               = 2,
        verbose           = True,
        enable_gpu        = False,
    )

    copula_synth = CopulaGANSynthesizer(**common_kwargs)
    copula_synth.fit(df_scaled)
    df_copula = copula_synth.sample(num_rows=n_copula)
    df_copula['source'] = 'copula'

    ctgan_synth = CTGANSynthesizer(**common_kwargs)
    ctgan_synth.fit(df_scaled)
    df_ctgan = ctgan_synth.sample(num_rows=n_ctgan)
    df_ctgan['source'] = 'ctgan'

    df_syn_raw = pd.concat([df_copula, df_ctgan], ignore_index=True)
    # ── Inverse scale back ──
    df_syn_raw[GAN_FEATURES] = scaler_global.inverse_transform(df_syn_raw[GAN_FEATURES])

    # ── Clip to observed real range (no redistribution) ──
    df_syn_pp = df_syn_raw[GAN_FEATURES + ['source']].copy()
    for f in GAN_FEATURES:
        lo = df_min_all[f].min()
        hi = df_min_all[f].max()
        df_syn_pp[f] = df_syn_pp[f].clip(lo, hi)

    # ── Round and cast integer features back to int64 ──
    # Must happen before distribution check so checks run on the same
    # dtype that will actually be saved — KS/JSD on floats vs integers
    # of the same values can produce different results.
    INT_FEATURES = ['NS', 'ADD', 'SUB', 'CA']
    for f in INT_FEATURES:
        if f in df_syn_pp.columns:
            df_syn_pp[f] = df_syn_pp[f].round().astype('int64')

    # KNN filter — keep top-90th-percentile candidates
    sc      = StandardScaler().fit(df_min_ref[GAN_FEATURES])
    real_sc = sc.transform(df_min_ref[GAN_FEATURES])
    syn_sc  = sc.transform(df_syn_pp[GAN_FEATURES])
    dists, _ = NearestNeighbors(n_neighbors=5).fit(real_sc).kneighbors(syn_sc)
    df_syn_pp['_knn'] = dists.mean(axis=1)
    thr = np.percentile(df_syn_pp['_knn'], 90)
    df_eligible = df_syn_pp[df_syn_pp['_knn'] <= thr].copy()

    df_selected = (
        df_eligible
        .sample(n=min(n_synthetic_needed, len(df_eligible)), random_state=seed)
        .drop(columns=['_knn', 'source', 'BC'], errors="ignore")
        .reset_index(drop=True)
    )

    if len(df_selected) < n_synthetic_needed:
        print(
            f"Only {len(df_selected)} eligible synthetic rows available; "
            f"{n_synthetic_needed} required."
        )
        return False, df_selected, df_syn_pp, None, {
            "hard_pass": False,
            "moment_failures": np.inf,
            "score": np.inf,
            "soft_accept": False,
            "reason": "not_enough_eligible_rows",
        }

    passed, df_stats, summary = distribution_check(
        df_syn_min=df_selected,
        df_train_min_full=df_min_ref,
        features=RAW_FEATURES,
        **thresholds,
    )

    return passed, df_selected, df_syn_pp, df_stats, summary

print('GAN pipeline function defined.')

search_results = []

best_soft_score = np.inf
best_hard_score = np.inf

best_soft_bundle = None
best_hard_bundle = None

for attempt, config in enumerate(GAN_SEARCH_GRID, 1):
    MULTIPLIER, epochs, batch_size, gen_lr, disc_lr = config

    print(
        f"\n[{attempt}/{len(GAN_SEARCH_GRID)}] "
        f"MULTIPLIER={MULTIPLIER}, epochs={epochs}, "
        f"batch={batch_size}, effective_batch={min(batch_size, 64)}, "
        f"lr=({gen_lr:.0e},{disc_lr:.0e})"
    )
    print("-" * 70)

    passed, df_sel_tmp, df_pp_tmp, df_stats_tmp, summary_tmp = run_gan_pipeline(
        config=config,
        seed=SEED,
        thresholds=DIST_THRESHOLDS
    )

    hard_pass = summary_tmp["hard_pass"]
    score = summary_tmp["score"]
    moment_fails = summary_tmp["moment_failures"]

    search_results.append((config, passed, hard_pass, score, moment_fails))

    print(
        f"\n  → soft_pass={passed} | "
        f"hard_pass={hard_pass} | "
        f"moment_fails={moment_fails} | "
        f"score={score:.4f}"
    )

    # Best fully accepted candidate
    if passed and score < best_soft_score:
        best_soft_score = score
        best_soft_bundle = {
            "config": config,
            "df_selected": df_sel_tmp.copy(),
            "df_syn_pp": df_pp_tmp.copy(),
            "df_dist_results": df_stats_tmp.copy() if df_stats_tmp is not None else None,
            "summary": summary_tmp,
        }

    # Best hard-passing fallback candidate
    if hard_pass and score < best_hard_score:
        best_hard_score = score
        best_hard_bundle = {
            "config": config,
            "df_selected": df_sel_tmp.copy(),
            "df_syn_pp": df_pp_tmp.copy(),
            "df_dist_results": df_stats_tmp.copy() if df_stats_tmp is not None else None,
            "summary": summary_tmp,
        }

print("\n" + "=" * 70)
print("Search complete. Summary:")
print("=" * 70)

for cfg, soft_ok, hard_ok, sc, mf in search_results:
    mult, ep, bs, glr, dlr = cfg

    marker = ""
    if best_soft_bundle is not None and cfg == best_soft_bundle["config"]:
        marker = " ← best soft-accepted"
    elif best_soft_bundle is None and best_hard_bundle is not None and cfg == best_hard_bundle["config"]:
        marker = " ← best hard-pass fallback"

    print(
        f"  MULT={mult:>2} ep={ep:>4} bs={bs:>3} "
        f"lr=({glr:.0e},{dlr:.0e}) → "
        f"soft={'✓' if soft_ok else '✗'} "
        f"hard={'✓' if hard_ok else '✗'} "
        f"moment_fails={mf} "
        f"score={sc:.4f}{marker}"
    )

# Prefer soft-accepted model. If none, use best hard-passing fallback.
if best_soft_bundle is not None:
    selected_bundle = best_soft_bundle
    selection_type = "soft-accepted"
elif best_hard_bundle is not None:
    selected_bundle = best_hard_bundle
    selection_type = "hard-pass fallback"
else:
    raise AssertionError(
        "All configs failed hard checks — do not patch the data. "
        "Extend the grid or investigate GAN quality."
    )

best_config = selected_bundle["config"]
best_score = selected_bundle["summary"]["score"]
df_selected = selected_bundle["df_selected"]
df_syn_pp = selected_bundle["df_syn_pp"]
df_dist_results = selected_bundle["df_dist_results"]

df_selected["Label"] = 1

print(f"\n✓ Selection type : {selection_type}")
print(f"✓ Best config    : {best_config}")
print(f"  Best score     : {best_score:.4f}")
print(f"  Moment failures: {selected_bundle['summary']['moment_failures']}")
print(f"  Selected rows  : {df_selected.shape[0]}")
# Recompute all derived features deterministically from raw features
df_selected_full = compute_derived(df_selected)

# Inject missing flags into synthetic rows
df_selected_full = inject_synthetic_missing(
    df            = df_selected_full,
    missing_rates = MISSING_RATES,
    features      = RAW_FEATURES,
    df_min_ref    = df_min_ref,
    label_val     = 1,
    random_state  = SEED,
)

df_selected_full['is_synthetic'] = 1

inc_cols = []
print('Derived features recomputed and missing values injected.')
print(f'  Final columns ({len(df_selected_full.columns)}): {list(df_selected_full.columns)}')
print(df_selected_full.head())

df_train_full = compute_derived(df_train)
df_train_full['is_synthetic'] = 0

df_full_derived = compute_derived(df_raw)
df_full_derived['is_synthetic'] = 0

# Concatenate full real dataset and full synthetic dataset into one balanced dataset.
df_full_balanced = pd.concat([df_full_derived, df_selected_full], ignore_index=True)
counts = df_full_balanced['Label'].value_counts().sort_index()
source_counts = df_full_balanced['is_synthetic'].value_counts().sort_index()
total_rows = len(df_full_balanced)


print('Balanced FULL dataset:')
print(f'  Total      : {len(df_full_balanced)}')
print(f'  Label = 0  : {counts.get(0, 0)}')
print(f'  Label = 1  : {counts.get(1, 0)}')
print(f'  Synthetic  : {source_counts.get(1, 0)}')

print(df_full_balanced.head())

df_val_full  = compute_derived(df_val)
df_test_full = compute_derived(df_test)

DATASETS_FILENAMES = {
    'train': DATASETS_DIR / 'deployment' / 'train_deployment.csv',
    's_train': DATASETS_DIR / 'deployment' / 's_full_deployment.csv',
    'val': DATASETS_DIR / 'deployment' / 'val_deployment.csv',
    'test': DATASETS_DIR / 'deployment' / 'test_deployment.csv'
}

# Real Train Data only (both labels 0 and 1)
df_train_full.drop(columns=["is_synthetic"]).to_csv(DATASETS_FILENAMES['train'], index=False)

# Synthetic Train Data (label 1 only)
df_selected_full.drop(columns=["is_synthetic"]).to_csv(DATASETS_FILENAMES['s_train'], index=False)

# Validation and Test Data
df_val_full.to_csv(DATASETS_FILENAMES['val'],   index=False)
df_test_full.to_csv(DATASETS_FILENAMES['test'], index=False)

print(f'Files saved to {DATASETS_DIR}:')
print(f'  {DATASETS_FILENAMES['train']}    — {len(df_train_full)} rows (real only, labels 0 and 1)')
print(f'  {DATASETS_FILENAMES['s_train']}  — {len(df_selected_full[df_selected_full["Label"] == 1])} rows (synthetic only, label 1)')
print(f'  {DATASETS_FILENAMES['val']}      — {len(df_val_full)} rows (real only)')
print(f'  {DATASETS_FILENAMES['test']}     — {len(df_test_full)} rows (real only)')

