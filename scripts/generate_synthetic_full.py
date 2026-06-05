import pandas as pd
import numpy as np
import random, os, warnings
import torch
import scipy.stats as stats
import json
import logging
import sys
from scipy.spatial.distance import jensenshannon
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sdv.single_table import CopulaGANSynthesizer, CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from pathlib import Path

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

# Set Seed for reproducability
SEED = 42

EPSILON = 1e-9
RAW_FEATURES = ['NC', 'DM', 'NS', 'ADD', 'SUB', 'CA']
GAN_FEATURES = RAW_FEATURES + ['BC']   # BC included so GAN learns ADD/SUB/CA joint
DERIVED_FEATURES = ['NP', 'SN', 'AF', 'BC', 'AS', 'PF']
ROOT_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
DATASETS_DIR = ROOT_DIR / 'datasets' / 'processed'
OUTPUTS_DIR = ROOT_DIR / 'outputs' / 'logs_and_metrics'

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

def compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute all derived features from raw features deterministically."""
    df_der = df.copy()
    df_der['NP'] = (df_der['NC'] + df_der['DM']) / 2
    df_der['SN'] = df_der['NC'] - df_der['DM']
    df_der['AF'] = (df_der['NS'] + df_der['ADD'] + df_der['SUB'] + df_der['CA']) / 4
    df_der['BC'] = (df_der['ADD'] + df_der['SUB']) / 2 - df_der['CA']
    df_der['AS'] = df_der['ADD'] - df_der['SUB']
    df_der['PF'] = df_der['AF'] / (df_der['NP'] + EPSILON)
    return df_der

def inject_synthetic_missing(df: pd.DataFrame, missing_rates: dict,
                               features: list, df_min_ref: pd.DataFrame,
                               label_val: int = 1,
                               random_state: int = SEED) -> pd.DataFrame:
    log.info("Generating synthetic incomplete flags...")
    rng = np.random.default_rng(random_state)
    df_der = df.copy()

    # Use the "all" split since we are using the full dataset
    split_key = "all"
    rates = missing_rates.get(split_key, {}).get(str(label_val), {})

    feature_medians = {f: df_min_ref[f].median() for f in features}
    for f in features:
        col = f + '_incomplete'
        df_der[col] = 0
        rate = float(np.clip(rates.get(f, 0.0), 0.0, 1.0))
        if rate <= 0.0:
            continue
        mask = rng.binomial(1, rate, size=len(df)).astype(bool)
        if mask.sum() == 0:
            continue
        df_der.loc[mask, f] = feature_medians[f]
        df_der.loc[mask, col] = 1
    log.info("Finished generating synthetic flags.")
    return df_der

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
) -> tuple:
    all_pass_global = True
    results = []
    
    for f in features:
        real_f = df_train_min_full[f].values
        syn_f  = df_syn_min[f].values

        ks_stat, p_value = stats.ks_2samp(real_f, syn_f)
        ks_pass = p_value > ks_pval

        mean_r, mean_s = np.mean(real_f),              np.mean(syn_f)
        std_r,  std_s  = np.std(real_f, ddof=1),       np.std(syn_f, ddof=1)
        med_r,  med_s  = np.median(real_f),            np.median(syn_f)
        skew_r, skew_s = stats.skew(real_f, bias=True), stats.skew(syn_f, bias=True)
        kurt_r, kurt_s = stats.kurtosis(real_f, bias=True), stats.kurtosis(syn_f, bias=True)

        delta_mu   = abs(mean_s - mean_r) / (abs(mean_r) + EPSILON)
        delta_std  = abs(std_s  - std_r)  / (std_r       + EPSILON)
        delta_med  = abs(med_s  - med_r)  / (abs(med_r)  + EPSILON)
        delta_skew = abs(skew_s - skew_r)
        delta_kurt = abs(kurt_s - kurt_r)

        mu_pass   = delta_mu   < T_mu
        std_pass  = delta_std  < T_sigma
        med_pass  = delta_med  < T_med
        skew_pass = delta_skew < T_skew
        kurt_pass = delta_kurt < T_kurt

        bins         = np.histogram_bin_edges(np.concatenate([real_f, syn_f]), bins=20)
        p_real, _    = np.histogram(real_f, bins=bins, density=True)
        p_syn,  _    = np.histogram(syn_f,  bins=bins, density=True)
        jsd_val      = jensenshannon(p_real + EPSILON, p_syn + EPSILON) ** 2
        jsd_pass     = jsd_val < JSD_THRESHOLD

        stats_pass   = mu_pass and std_pass and med_pass and skew_pass and kurt_pass
        feature_pass = ks_pass and jsd_pass and stats_pass
        all_pass_global = all_pass_global and feature_pass

        results.append(dict(
            feature=f, KS_stat=ks_stat, KS_p=p_value,
            JSD=jsd_val, delta_mu=delta_mu, delta_std=delta_std,
            delta_med=delta_med, delta_skew=delta_skew, delta_kurt=delta_kurt,
            ALL_pass=feature_pass,
        ))

    return all_pass_global, pd.DataFrame(results)

def main():
    log.info("=" * 60)
    log.info("Full Deployment Synthetic Data Generation")
    log.info("=" * 60)

    # 1. Load missing rates
    MISSING_RATES = {}
    try:
        with open(OUTPUTS_DIR / 'missing_rates.json', encoding='utf-8') as file:
            MISSING_RATES = json.load(file)
        log.info("Loaded missing_rates.json")
    except Exception as e:
        log.warning(f"Could not load missing_rates.json: {e}")

    # 2. Load all real datasets and combine
    train_df = pd.read_csv(DATASETS_DIR / 'train_deployment.csv')
    val_df = pd.read_csv(DATASETS_DIR / 'val_deployment.csv')
    test_df = pd.read_csv(DATASETS_DIR / 'test_deployment.csv')
    
    df_full = pd.concat([train_df, val_df, test_df], ignore_index=True)
    df_full_raw = df_full[RAW_FEATURES + ['Label']].copy()
    
    df_full_maj = df_full_raw[df_full_raw['Label'] == 0]
    df_full_min = df_full_raw[df_full_raw['Label'] == 1]
    
    n_synthetic_needed = len(df_full_maj) - len(df_full_min)
    log.info(f"Full Data Shape: {df_full.shape}")
    log.info(f"Majority (0) count: {len(df_full_maj)}")
    log.info(f"Minority (1) count: {len(df_full_min)}")
    log.info(f"Synthetic samples needed: {n_synthetic_needed}")
    
    if n_synthetic_needed <= 0:
        log.info("Dataset is balanced or minority is larger. No synthetic data needed.")
        sys.exit(0)

    df_min_all = df_full_min[RAW_FEATURES].copy()
    df_min_all['BC'] = (df_min_all['ADD'] + df_min_all['SUB']) / 2 - df_min_all['CA']
    df_min_ref = df_min_all.copy()

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df_min_all)

    GAN_SEARCH_GRID = [
        ( 7, 2000, 500, 2e-4, 2e-4),
        (10, 2000, 500, 2e-4, 2e-4),
        (10, 3000, 500, 2e-4, 2e-4),
        (10, 3000, 100, 2e-4, 2e-4),
        (15, 3000, 100, 1e-4, 1e-4),
        (15, 5000, 100, 1e-4, 1e-4),
        (20, 5000, 100, 1e-4, 1e-4),
        (30, 5000, 100, 1e-4, 1e-4),
        (50, 5000, 100, 1e-4, 1e-4),
    ]

    DIST_THRESHOLDS = dict(
        ks_pval = 0.05, T_mu=0.10, T_sigma=0.175, T_med=0.125,
        T_skew=0.75, T_kurt=1.5, JSD_THRESHOLD=0.13
    )

    scaler_global = StandardScaler().fit(df_min_all[GAN_FEATURES])

    def run_gan_pipeline(config, seed=SEED, thresholds=DIST_THRESHOLDS):
        set_seed(seed)
        MULTIPLIER, epochs, batch_size, gen_lr, disc_lr = config

        n_generate = n_synthetic_needed * MULTIPLIER
        n_copula   = n_generate // 2
        n_ctgan    = n_generate - n_copula

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
            verbose           = False,
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
        df_syn_raw[GAN_FEATURES] = scaler_global.inverse_transform(df_syn_raw[GAN_FEATURES])

        df_syn_pp = df_syn_raw[GAN_FEATURES + ['source']].copy()
        for f in GAN_FEATURES:
            lo = df_min_all[f].min()
            hi = df_min_all[f].max()
            df_syn_pp[f] = df_syn_pp[f].clip(lo, hi)

        INT_FEATURES = ['NS', 'ADD', 'SUB', 'CA']
        for f in INT_FEATURES:
            if f in df_syn_pp.columns:
                df_syn_pp[f] = df_syn_pp[f].round().astype('int64')

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
            .drop(columns=['_knn', 'source', 'BC'])
            .reset_index(drop=True)
        )

        passed, df_stats = distribution_check(
            df_syn_min        = df_selected,
            df_train_min_full = df_min_ref,
            features          = RAW_FEATURES,
            **thresholds,
        )
        return passed, df_selected, df_stats

    best_score = np.inf
    df_selected = None
    best_config = None

    for attempt, config in enumerate(GAN_SEARCH_GRID, 1):
        log.info(f"Trying config {attempt}/{len(GAN_SEARCH_GRID)}: {config}")
        passed, df_sel_tmp, df_stats_tmp = run_gan_pipeline(config)

        if passed:
            score = df_stats_tmp[['delta_mu', 'delta_std', 'delta_med', 'delta_skew', 'delta_kurt', 'JSD']].sum().sum()
            log.info(f"  → PASS (score={score:.4f})")
            if score < best_score:
                best_score = score
                df_selected = df_sel_tmp
                best_config = config
        else:
            log.info("  → FAIL")

    if df_selected is None:
        log.error("All GAN configs failed to produce valid synthetic data. Try expanding grid or checking input data.")
        sys.exit(1)

    log.info(f"Best config found: {best_config} with score {best_score:.4f}")
    
    df_selected['Label'] = 1
    df_selected_full = compute_derived(df_selected)
    
    df_selected_full = inject_synthetic_missing(
        df            = df_selected_full,
        missing_rates = MISSING_RATES,
        features      = RAW_FEATURES,
        df_min_ref    = df_min_ref,
        label_val     = 1,
        random_state  = SEED,
    )
    
    out_path = DATASETS_DIR / 's_full_deployment.csv'
    df_selected_full.to_csv(out_path, index=False)
    log.info(f"Synthetic data saved to {out_path} ({len(df_selected_full)} rows).")

if __name__ == "__main__":
    main()
