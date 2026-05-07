import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

def load_data(base_dir):
    grid_dir = base_dir / 'outputs' / 'grid_search'
    
    print(f"Loading data from: {grid_dir}")
    trtr_grid = pd.read_csv(grid_dir / 'trtr_grid_search_results.csv')
    tstr_grid = pd.read_csv(grid_dir / 'tstr_grid_search_results.csv')
    
    trtr_tie_path = grid_dir / 'trtr_validation_tie_cv_results.csv'
    tstr_tie_path = grid_dir / 'tstr_validation_tie_cv_results.csv'
    
    trtr_tie = pd.read_csv(trtr_tie_path) if trtr_tie_path.exists() else pd.DataFrame()
    tstr_tie = pd.read_csv(tstr_tie_path) if tstr_tie_path.exists() else pd.DataFrame()
    
    return trtr_grid, tstr_grid, trtr_tie, tstr_tie

def analyze_and_plot(trtr_grid, tstr_grid, trtr_tie, tstr_tie, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Best scores overall (validation)
    best_trtr_idx = trtr_grid['fbeta'].idxmax()
    best_tstr_idx = tstr_grid['fbeta'].idxmax()
    
    best_trtr_f2 = trtr_grid.loc[best_trtr_idx, 'fbeta']
    best_tstr_f2 = tstr_grid.loc[best_tstr_idx, 'fbeta']
    
    best_trtr_recall = trtr_grid.loc[best_trtr_idx, 'recall']
    best_tstr_recall = tstr_grid.loc[best_tstr_idx, 'recall']
    
    best_trtr_prec = trtr_grid.loc[best_trtr_idx, 'precision']
    best_tstr_prec = tstr_grid.loc[best_tstr_idx, 'precision']
    
    with open(output_dir / 'analysis_report.txt', 'w') as f:
        f.write("=== TSTR vs TRTR Analysis Report ===\n\n")
        f.write(f"TRTR Best F2: {best_trtr_f2:.4f} (Recall: {best_trtr_recall:.4f}, Precision: {best_trtr_prec:.4f})\n")
        f.write(f"TSTR Best F2: {best_tstr_f2:.4f} (Recall: {best_tstr_recall:.4f}, Precision: {best_tstr_prec:.4f})\n")
        f.write(f"\nTSTR Improvement in F2 over TRTR: {best_tstr_f2 - best_trtr_f2:.4f}\n")
        f.write(f"TSTR Improvement in Recall over TRTR: {best_tstr_recall - best_trtr_recall:.4f}\n")
        f.write(f"TSTR Improvement in Precision over TRTR: {best_tstr_prec - best_trtr_prec:.4f}\n")
        
        f.write("\n=== Tie Candidates CV Performance ===\n")
        if not trtr_tie.empty and not tstr_tie.empty:
            f.write(f"TRTR - Tie CV Mean F2: {trtr_tie['cv_mean_fbeta'].mean():.4f} +/- {trtr_tie['cv_mean_fbeta'].std():.4f}\n")
            f.write(f"TSTR - Tie CV Mean F2: {tstr_tie['cv_mean_fbeta'].mean():.4f} +/- {tstr_tie['cv_mean_fbeta'].std():.4f}\n")
            f.write(f"TRTR - Tie CV Mean Recall: {trtr_tie['cv_mean_recall'].mean():.4f} +/- {trtr_tie['cv_mean_recall'].std():.4f}\n")
            f.write(f"TSTR - Tie CV Mean Recall: {tstr_tie['cv_mean_recall'].mean():.4f} +/- {tstr_tie['cv_mean_recall'].std():.4f}\n")
        else:
            f.write("Tie CV results are missing for TRTR or TSTR.\n")
            
    print(open(output_dir / 'analysis_report.txt').read())
    
    # 1. Bar Plot of Best F2 and Recall
    fig, ax = plt.subplots(figsize=(8, 6))
    metrics = ['Best F2 Score', 'Corresponding Recall', 'Corresponding Precision']
    trtr_vals = [best_trtr_f2, best_trtr_recall, best_trtr_prec]
    tstr_vals = [best_tstr_f2, best_tstr_recall, best_tstr_prec]
    
    x = range(len(metrics))
    width = 0.35
    bars1 = ax.bar([i - width/2 for i in x], trtr_vals, width, label='TRTR (Real Only)', color='skyblue')
    bars2 = ax.bar([i + width/2 for i in x], tstr_vals, width, label='TSTR (Real + Synthetic)', color='salmon')
    
    ax.set_ylabel('Score')
    ax.set_title('TRTR vs TSTR: Best Validation Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(loc='lower left')
    ax.set_ylim(0, 1.1)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.3f}", ha='center')
        
    plt.tight_layout()
    plt.savefig(output_dir / 'best_performance_comparison.png', dpi=300)
    plt.close()
    
    # 2. Precision-Recall Tradeoff across thresholds
    plt.figure(figsize=(10, 6))
    # We will sample to avoid plotting too many points and making it unreadable
    trtr_sample = trtr_grid.sample(n=min(5000, len(trtr_grid)), random_state=42)
    tstr_sample = tstr_grid.sample(n=min(5000, len(tstr_grid)), random_state=42)
    
    plt.scatter(trtr_sample['recall'], trtr_sample['precision'], c='skyblue', alpha=0.5, label='TRTR', marker='o', s=15)
    plt.scatter(tstr_sample['recall'], tstr_sample['precision'], c='salmon', alpha=0.5, label='TSTR', marker='x', s=15)
    
    # Highlight the best points
    plt.scatter([best_trtr_recall], [best_trtr_prec], color='blue', edgecolors='black', s=150, label='TRTR Best F2', marker='*')
    plt.scatter([best_tstr_recall], [best_tstr_prec], color='red', edgecolors='black', s=150, label='TSTR Best F2', marker='*')

    plt.title('Precision vs Recall Tradeoff across Hyperparameters & Thresholds')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / 'precision_recall_tradeoff.png', dpi=300)
    plt.close()
    
    # 3. CV Results Distribution (if tie CV results are populated)
    if not trtr_tie.empty and not tstr_tie.empty:
        # Prepare data for plotting
        trtr_tie_plt = trtr_tie.copy()
        trtr_tie_plt['Model'] = 'TRTR'
        
        tstr_tie_plt = tstr_tie.copy()
        tstr_tie_plt['Model'] = 'TSTR'
        
        combined_cv = pd.concat([trtr_tie_plt, tstr_tie_plt], ignore_index=True)
        
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=combined_cv, x='Model', y='cv_mean_fbeta', palette='Set2', hue='Model', legend=False)
        sns.stripplot(data=combined_cv, x='Model', y='cv_mean_fbeta', color=".3", size=4, jitter=True)
        plt.title('Distribution of CV Mean F2 Scores among Tie Candidates')
        plt.ylabel('CV Mean F2 Score')
        plt.tight_layout()
        plt.savefig(output_dir / 'cv_f2_distribution.png', dpi=300)
        plt.close()
        
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=combined_cv, x='Model', y='cv_mean_recall', palette='Set3', hue='Model', legend=False)
        sns.stripplot(data=combined_cv, x='Model', y='cv_mean_recall', color=".3", size=4, jitter=True)
        plt.title('Distribution of CV Mean Recall among Tie Candidates')
        plt.ylabel('CV Mean Recall')
        plt.tight_layout()
        plt.savefig(output_dir / 'cv_recall_distribution.png', dpi=300)
        plt.close()
        
    # 4. Trend Charts for Hyperparameters
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    configs = ['max_depth', 'min_samples_leaf', 'conf_fact', 'threshold']
    
    for i, config in enumerate(configs):
        trtr_trend = trtr_grid.groupby(config)[['fbeta', 'recall']].max().reset_index()
        tstr_trend = tstr_grid.groupby(config)[['fbeta', 'recall']].max().reset_index()
        
        ax = axes[i]
        
        # Plot F2
        ax.plot(trtr_trend[config], trtr_trend['fbeta'], label='TRTR F2', color='skyblue', marker='o', linestyle='-', linewidth=5, alpha=0.6)
        ax.plot(tstr_trend[config], tstr_trend['fbeta'], label='TSTR F2', color='salmon', marker='o', linestyle='-', linewidth=2)
        
        # Plot Recall (dashed)
        ax.plot(trtr_trend[config], trtr_trend['recall'], label='TRTR Recall', color='blue', marker='s', linestyle='--', linewidth=5, alpha=0.4)
        ax.plot(tstr_trend[config], tstr_trend['recall'], label='TSTR Recall', color='red', marker='x', linestyle=':', linewidth=2)
        
        ax.set_title(f'Max F2 & Recall vs {config}')
        ax.set_xlabel(config)
        ax.set_ylabel('Score')
        ax.grid(True, linestyle=':', alpha=0.6)
        if i == 0:
            ax.legend(loc='best')
            
    plt.tight_layout()
    plt.savefig(output_dir / 'hyperparameter_trends.png', dpi=300)
    plt.close()
        
    print(f"\nFigures and analysis report saved to: {output_dir}")

def main():
    base_dir = Path(__file__).resolve().parent.parent
    trtr_grid, tstr_grid, trtr_tie, tstr_tie = load_data(base_dir)
    
    output_dir = base_dir / 'outputs' / 'figures'
    analyze_and_plot(trtr_grid, tstr_grid, trtr_tie, tstr_tie, output_dir)

if __name__ == '__main__':
    main()
