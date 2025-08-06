import pandas as pd
import json
import time
import shutil
from pathlib import Path

# --- Plotting Imports ---
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

# Assuming the pysisso package is in the current directory or installed
from pysisso import SISSORegressor

# --- Configuration for the Comparison Test ---

# List of search strategies to test.
# 'brute_force' can be very slow, so its parameters will be adjusted.
SEARCH_STRATEGIES = [
    'greedy',
    'sisso++',
    'omp',
    'brute_force',
    # 'miqp' will be added automatically if Gurobi is available.
]

# List of model selection methods to test.
SELECTION_METHODS = [
    'cv',
    'bootstrap',
    'aic',
    'bic'
]

BASE_CONFIG_FILE = "config_mohsen.json"
RESULTS_DIR = Path("comparison_runs")
SUMMARY_FILE = RESULTS_DIR / "comparison_summary.csv"

# --- Plotting Function ---

def plot_summary_results(results_df, output_dir):
    """
    Generates and saves summary plots from the comparison results DataFrame.
    """
    if not PLOTTING_AVAILABLE:
        print("\nWARNING: Matplotlib or Seaborn not found. Skipping plot generation.")
        print("         Please run 'pip install matplotlib seaborn' to enable plotting.")
        return

    print("\n--- Generating Summary Plots ---")
    
    # Filter out failed runs for plotting
    plot_df = results_df[results_df['status'] == 'Success'].copy()
    if plot_df.empty:
        print("No successful runs to plot.")
        return

    # Plot 1: R² Score Comparison (Grouped Bar Chart)
    plt.figure(figsize=(14, 7))
    sns.barplot(data=plot_df, x='search_strategy', y='final_r2_score', hue='selection_method', palette='viridis')
    plt.title('Final R² Score Comparison', fontsize=16)
    plt.ylabel('R² Score (Higher is Better)', fontsize=12)
    plt.xlabel('Search Strategy', fontsize=12)
    plt.ylim(0, max(1.0, plot_df['final_r2_score'].max() * 1.1)) # Ensure y-axis goes to at least 1.0
    plt.xticks(rotation=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Selection Method')
    plt.tight_layout()
    plot_path_r2 = output_dir / "r2_score_comparison.png"
    plt.savefig(plot_path_r2)
    print(f"  - Saved R² score comparison plot to: {plot_path_r2}")
    plt.show()

    # Plot 2: Execution Time Comparison (Grouped Bar Chart)
    plt.figure(figsize=(14, 7))
    sns.barplot(data=plot_df, x='search_strategy', y='time_seconds', hue='selection_method', palette='plasma')
    plt.title('Execution Time Comparison', fontsize=16)
    plt.ylabel('Time (seconds, Lower is Better)', fontsize=12)
    plt.xlabel('Search Strategy', fontsize=12)
    # Using a log scale can be helpful if times vary widely
    # plt.yscale('log')
    plt.xticks(rotation=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Selection Method')
    plt.tight_layout()
    plot_path_time = output_dir / "execution_time_comparison.png"
    plt.savefig(plot_path_time)
    print(f"  - Saved execution time comparison plot to: {plot_path_time}")
    plt.show()

    # Plot 3: Time vs. Accuracy Trade-off (Scatter Plot)
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=plot_df,
        x='time_seconds',
        y='final_r2_score',
        hue='search_strategy',
        style='selection_method',
        s=150, # size of markers
        palette='deep'
    )
    plt.title('Time vs. Accuracy Trade-off', fontsize=16)
    plt.xlabel('Execution Time (seconds) -> Faster', fontsize=12)
    plt.ylabel('Final R² Score -> More Accurate', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    # Add annotations for clarity
    for i, row in plot_df.iterrows():
        plt.text(row['time_seconds'] * 1.02, row['final_r2_score'], 
                 f"{row['search_strategy']}/{row['selection_method']}", 
                 fontsize=8, alpha=0.8, rotation=0, verticalalignment='bottom')

    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make room for legend
    plot_path_tradeoff = output_dir / "time_vs_accuracy_tradeoff.png"
    plt.savefig(plot_path_tradeoff)
    print(f"  - Saved trade-off scatter plot to: {plot_path_tradeoff}")
    plt.show()


# --- Main Test Logic ---

def run_comparison_test():
    """
    Runs a systematic comparison of SISSO search strategies and selection methods.
    """
    # 1. Setup and Pre-flight Checks
    print("="*60)
    print("--- Starting pySISSO Comparison Test ---")
    print("="*60)

    # Check for Gurobi and add 'miqp' strategy if available
    try:
        import gurobipy
        SEARCH_STRATEGIES.append('miqp')
        print("INFO: Gurobi found. 'miqp' search strategy will be tested.")
    except ImportError:
        print("INFO: Gurobi not found. Skipping 'miqp' search strategy.")

    if RESULTS_DIR.exists():
        if input(f"Results directory '{RESULTS_DIR}' already exists. Overwrite? (y/N): ").lower() != 'y':
            print("Aborting."); return
        shutil.rmtree(RESULTS_DIR)
    RESULTS_DIR.mkdir(parents=True)

    # 2. Load Base Configuration and Data
    print(f"\n--- Loading base configuration from '{BASE_CONFIG_FILE}' ---")
    with open(BASE_CONFIG_FILE, 'r') as f:
        lines = [line for line in f if not line.strip().startswith('//')]
        base_config = json.loads("".join(lines))

    print(f"--- Loading and preparing data from '{base_config['data_file']}' ---")
    data_file = base_config.get('data_file')
    data = pd.read_csv(data_file)
    
    prop_key = base_config.get('property_key')
    non_feature_cols = base_config.get('non_feature_cols', [])
    feature_cols = [c for c in data.columns if c not in [prop_key] + non_feature_cols]
    
    used_cols = [prop_key] + feature_cols
    data_subset = data[used_cols].dropna()

    y = data_subset[prop_key]
    X_raw = data_subset[feature_cols]
    X = pd.get_dummies(X_raw, dummy_na=False, drop_first=False)
    
    print(f"Data prepared. Using {X.shape[0]} samples and {X.shape[1]} primary features.")

    # 3. Run the Comparison Loop
    results_list = []
    
    for strategy in SEARCH_STRATEGIES:
        for method in SELECTION_METHODS:
            print("\n" + "-"*60)
            print(f"Testing Combination: Strategy='{strategy}', Selection='{method}'")
            print("-"*60)
            
            # Prepare configuration for this specific run
            run_config = base_config.copy()
            run_config['search_strategy'] = strategy
            run_config['selection_method'] = method
            run_config['workdir'] = str(RESULTS_DIR / f"run_{strategy}_{method}")
            
            if strategy == 'brute_force':
                run_config['max_D'] = min(base_config['max_D'], 2)
                run_config['sis_sizes'] = [min(base_config['sis_sizes'][0], 15)]
                print(f"INFO: For 'brute_force', reducing max_D to {run_config['max_D']} and sis_size to {run_config['sis_sizes'][0]} to ensure timely completion.")

            start_time = time.monotonic()
            
            try:
                sisso = SISSORegressor(**run_config)
                sisso.fit(X, y)
                
                end_time = time.monotonic()
                duration = end_time - start_time
                
                if sisso.best_D_ is not None and sisso.best_model_ is not None:
                    final_score = sisso.score(X, y)
                    best_dim = sisso.best_D_
                    formula = sisso.best_model_summary()
                else:
                    final_score, best_dim, formula = float('nan'), -1, "No model found"

                results_list.append({
                    'search_strategy': strategy, 'selection_method': method,
                    'time_seconds': duration, 'final_r2_score': final_score,
                    'best_dimension': best_dim, 'status': 'Success',
                    'best_model_formula': formula.replace('\n', ' ')
                })
                print(f"--- SUCCESS: Completed in {duration:.2f}s. Final R²: {final_score:.4f} ---")

            except Exception as e:
                end_time = time.monotonic()
                duration = end_time - start_time
                print(f"--- FAILED: Run for '{strategy}'/'{method}' failed after {duration:.2f}s ---")
                print(f"Error: {e}")
                
                results_list.append({
                    'search_strategy': strategy, 'selection_method': method,
                    'time_seconds': duration, 'final_r2_score': float('nan'),
                    'best_dimension': -1, 'status': 'Failed',
                    'best_model_formula': f"Error: {e}"
                })
                
    # 4. Final Summary
    print("\n\n" + "="*60)
    print("--- Comparison Test Summary ---")
    print("="*60)
    
    if not results_list:
        print("No results were generated."); return

    results_df = pd.DataFrame(results_list)
    results_df.to_csv(SUMMARY_FILE, index=False)
    print(f"\nFull results saved to: {SUMMARY_FILE}")

    print("\n--- Final R² Score Summary ---")
    r2_pivot = results_df.pivot_table(index='search_strategy', columns='selection_method', values='final_r2_score')
    print(r2_pivot.to_string(float_format="%.4f"))

    print("\n--- Execution Time Summary (seconds) ---")
    time_pivot = results_df.pivot_table(index='search_strategy', columns='selection_method', values='time_seconds')
    print(time_pivot.to_string(float_format="%.2f"))
    
    # 5. Generate and save plots
    plot_summary_results(results_df, RESULTS_DIR)


if __name__ == '__main__':
    run_comparison_test()