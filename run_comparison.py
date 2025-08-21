import pandas as pd
import numpy as np
import json
import time
import subprocess
import os
import shutil
import re
from pathlib import Path

# --- 1. Setup: Verify Datasets Exist ---

def check_for_datasets():
    """Checks if the required CSV files exist in the current directory."""
    required_files = ['mohsen_binary.csv', 'manuel_data.csv', 'train.csv']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("ERROR: The following required dataset files are missing:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease make sure all datasets are in the same directory as this script.")
        exit() # Terminate the script
    print("--- All required datasets found. Starting extended benchmark. ---")
    print("\n" + "!"*80)
    print("! WARNING: This extended benchmark is designed to run for a long time (potentially")
    print("! several hours) to thoroughly test and differentiate the search strategies.")
    print("!"*80 + "\n")


# --- 2. Helper Function to Parse Results ---

def parse_results_from_output(output_dir="discover_output"):
    """Parses the final R2 score and selected dimension from the SISSO.out file."""
    r2_score = np.nan
    best_dim = np.nan
    try:
        summary_file = Path(output_dir) / "SISSO.out"
        with open(summary_file, 'r') as f:
            content = f.read()
            # Parse R2 score
            r2_match = re.search(r'R2/Accuracy on full data:\s*(-?\d+\.\d+)', content)
            if r2_match:
                r2_score = float(r2_match.group(1))
            
            # Parse selected dimension
            dim_match = re.search(r'SELECTED_DIMENSION:\s*(\d+)', content)
            if dim_match:
                best_dim = int(dim_match.group(1))

    except (FileNotFoundError, IndexError, ValueError):
        return np.nan, np.nan 
    
    return r2_score, best_dim


# --- 3. Main Benchmark Orchestrator ---

def run_benchmark():
    """
    Runs the DISCOVER algorithm with different search strategies on multiple
    datasets and collects performance metrics.
    """
    check_for_datasets()

    # --- More Aggressive Configurations to Differentiate Methods ---
    base_config = {
        "workdir": "discover_output",
        "task_type": "regression",
        "selection_method": "cv",
        "cv": 5,
        "n_jobs": -1,
        "random_state": 42,
        "op_rules": [{"op": op} for op in 
                     ["add", "sub", "mul", "div", "sq", "cb", "sqrt", "cbrt", 
                      "inv", "abs", "log", "exp", "abs_diff", "harmonic_mean"]],
    }

    dataset_configs = {
        "Mohsen (Small)": {
            "data_file": "mohsen_binary.csv",
            "property_key": "property",
            "non_feature_cols": ["materials"],
            "primary_units": {
                "property": "electron_volt", "n_atom": "dimensionless", "mass_A": "dalton",
                "mass_B": "dalton", "oxid_A": "dimensionless", "oxid_B": "dimensionless",
                "IR_A": "angstrom", "IR_B": "angstrom", "EN_A": "dimensionless",
                "EN_B": "dimensionless", "bond": "dimensionless", "volume": "angstrom**3"
            },
            "depth": 3,
            "max_D": 3,
            "sis_sizes": [100], # Can be larger now that depth is smaller
        },
        "Manuel (Medium)": {
            "data_file": "manuel_data.csv",
            "property_key": "E_a",
            "non_feature_cols": ["Name", "Delta_E", "E_Ts", "E_KRA", "A-site", "B-site", "X-site"],
            "primary_units": {
                "E_a": "electron_volt", "A_valence": "dimensionless", "tolerance_factor": "dimensionless",
                "R_A": "angstrom", "R_B": "angstrom", "R_X": "angstrom", "a": "angstrom",
                "ELN_A": "dimensionless", "ELN_B": "dimensionless", "ELN_X": "dimensionless",
                "DELNAX": "dimensionless", "DELNBX": "dimensionless", "R_AX": "angstrom",
                "R_BX": "angstrom", "R_tet": "angstrom", "R_oct": "angstrom",
                "alpha": "dimensionless", "k64_m": "dimensionless", "U": "electron_volt",
                "E_hull": "electron_volt", "k64_emp": "dimensionless"
            },
            "depth": 3,
            "max_D": 3,
            "sis_sizes": [100],
        },
        "Train (Large)": {
            "data_file": "train.csv",
            "property_key": "property",
            "non_feature_cols": ["materials"],
            "primary_units": {
                "E": "electron_volt", "E_tet": "electron_volt", "E_oct": "electron_volt",
                "A": "angstrom", "B": "angstrom", "C": "angstrom", "Cation": "angstrom",
                "Metal": "angstrom", "Anion": "electron_volt", "U": "angstrom",
                "ELN_A": "dimensionless", "ELN_B": "dimensionless", "ELN_X": "dimensionless",
                "DELNAX": "dimensionless", "DELNBX": "dimensionless", "R_AX": "angstrom",
                "R_BX": "angstrom", "U_emp": "electron_volt", "E_site": "electron_volt",
                "oxidation_state": "dimensionless", "E_metal": "electron_volt",
                "OCV": "volt", "property": "electron_volt"
            },
            "depth": 3,
            "max_D": 3, # Search for more complex models
            "sis_sizes": [100], # Provide a large pool for the search
        }
    }

    # Define search strategies and method-specific overrides
    strategies = ['greedy', 'brute_force', 'sisso++', 'omp', 'rmhc', 'sa']
    strategy_overrides = {
        "brute_force": {"sis_sizes": [100], "max_D": 3},
        # Tuned down slightly to ensure completion in reasonable time
        "rmhc": {"rmhc_iterations": 1000, "rmhc_restarts": 8},
        "sa": {"sa_cooling_rate": 0.98, "sa_iterations": 1000}
    }

    try:
        import gurobipy
        strategies.append('miqp')
        print("\nFound Gurobi, will include 'miqp' in the benchmark.")
    except ImportError:
        print("\nGurobi not found, skipping 'miqp' search strategy.")

    results = []
    config_filename = "config_benchmark.json"

    # Main benchmark loop
    for dataset_name, dataset_config in dataset_configs.items():
        for strategy in strategies:
            
            # Skip brute_force for datasets where it's not the designated test case
            if strategy == 'brute_force' and dataset_name != 'Manuel (Medium)':
                print(f"\n>>> Skipping '{strategy}' for '{dataset_name}' (only testing on medium dataset).")
                continue

            print(f"\n>>> Running '{strategy}' on '{dataset_name}'...")

            # Build the final config for this run
            config = base_config.copy()
            config.update(dataset_config)
            config["search_strategy"] = strategy
            if strategy in strategy_overrides:
                config.update(strategy_overrides[strategy])
                if strategy == 'brute_force':
                    print(f"    (Using special config for brute_force: sis_sizes={config['sis_sizes']})")

            # Write the temporary config file
            with open(config_filename, 'w') as f:
                json.dump(config, f, indent=2)

            # Clean up previous run's output directory
            if os.path.exists("discover_output"):
                shutil.rmtree("discover_output")

            start_time = time.perf_counter()
            
            process = subprocess.run(
                ['python', 'run_discover.py', config_filename],
                input='y\n', text=True, capture_output=True, check=False
            )
            
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            
            run_status = "Success"
            r2_score, best_dim = np.nan, np.nan

            if process.returncode == 0:
                r2_score, best_dim = parse_results_from_output()
                if pd.isna(r2_score):
                    run_status = "Parse Error"
            else:
                run_status = "Runtime Error"
                print(f"--- ERROR running {strategy} on {dataset_name} ---")
                print(process.stderr[-1000:]) # Print last 1000 chars of error
                
            results.append({
                "Dataset": dataset_name,
                "Strategy": strategy,
                "R2 Score": r2_score,
                "Time (s)": elapsed_time,
                "Final Model Dim.": best_dim,
                "Status": run_status
            })
            
            print(f"    Done in {elapsed_time:.2f}s. R2: {r2_score:.4f}, D_best: {best_dim}, Status: {run_status}")

    # --- 4. Final Reporting ---
    if os.path.exists(config_filename): os.remove(config_filename)
    if os.path.exists("discover_output"): shutil.rmtree("discover_output")
    
    df_results = pd.DataFrame(results)
    
    print("\n\n" + "="*70)
    print(" " * 22 + "EXTENDED BENCHMARK SUMMARY")
    print("="*70)
    
    pivot_r2 = df_results.pivot(index='Dataset', columns='Strategy', values='R2 Score').reindex(columns=strategies)
    pivot_time = df_results.pivot(index='Dataset', columns='Strategy', values='Time (s)').reindex(columns=strategies)
    pivot_dim = df_results.pivot(index='Dataset', columns='Strategy', values='Final Model Dim.').reindex(columns=strategies)

    print("\n--- R2 Scores ---")
    print(pivot_r2.to_string(float_format="%.4f"))
    
    print("\n--- Final Model Dimension (D_best) ---")
    print(pivot_dim.to_string(float_format="%.0f"))

    print("\n--- Execution Time (seconds) ---")
    print(pivot_time.to_string(float_format="%.2f"))
    print("\n" + "="*70)

if __name__ == "__main__":
    run_benchmark()
