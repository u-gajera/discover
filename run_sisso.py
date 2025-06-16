"""
================================================================================
  Universal Runner for the Python Sure Independence Screening and
                       Sparsifying Operator (pySISSO)
================================================================================

This script serves as the main entry point for running pySISSO. It is designed
to be generic and controlled entirely by a configuration file.

HOW TO RUN:

1. With your own configuration file:
   ---------------------------------
   Create a config.json file and run:
   python run_sisso.py your_config.json

2. As a self-contained demo (no arguments needed):
   ----------------------------------------------
   Run the script without any arguments:
   python run_sisso.py

   This will create 'demo_data.csv' and 'demo_config.json' (if they don't exist)
   and run a sample analysis using the robust OMP solver.

"""
import pandas as pd
import json
import argparse
import shutil
from pathlib import Path
import sys
import numpy as np

# This setup assumes the script is run from the project directory
# and the package is in `src/pysisso`.
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from pysisso import (
    SISSORegressor,
    SISSOClassifier,
    SISSOLogRegressor,
    SISSOCHClassifier
)

def run_analysis(config_path):
    """Core logic to run SISSO based on a config file."""
    # --- 1. Load Configuration ---
    print(f"--- Loading configuration from '{config_path}' ---")
    try:
        with open(config_path, 'r') as f:
            # Strip comments (lines starting with //) before parsing JSON
            lines = [line for line in f if not line.strip().startswith('//')]
            config = json.loads("".join(lines))
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error reading or parsing config file '{config_path}': {e}")
        return

    # --- 2. Prepare Directories ---
    workdir = config.get('workdir', 'sisso_output')
    if Path(workdir).exists():
        if input(f"Workdir '{workdir}' already exists. Overwrite? (y/N): ").lower() != 'y':
            print("Aborting."); return
        shutil.rmtree(workdir)

    # --- 3. Load and Preprocess Data ---
    data_file = config.get('data_file')
    if not data_file or not Path(data_file).exists():
        raise FileNotFoundError(f"Data file '{data_file}' specified in config not found.")
    
    print(f"--- Loading data from '{data_file}' ---")
    data = pd.read_csv(data_file)
    
    initial_rows = len(data)
    data.dropna(inplace=True)
    if len(data) < initial_rows:
        print(f"--- Dropped {initial_rows - len(data)} rows with missing values. ---")
    
    # --- 4. Define Target and Features based on Config ---
    prop_key = config.get('property_key')
    if not prop_key or prop_key not in data.columns:
        raise ValueError(f"Config 'property_key' '{prop_key}' not found in data columns.")
    
    non_feature_cols = config.get('non_feature_cols', [])
    
    y = data[prop_key]
    feature_cols = [c for c in data.columns if c not in [prop_key] + non_feature_cols]
    X_raw = data[feature_cols]
    
    # One-hot encode categorical features (if any)
    X = pd.get_dummies(X_raw, dummy_na=False, drop_first=False)
    if X.shape[1] > X_raw.shape[1]:
        print(f"--- Performed one-hot encoding. Feature set expanded from {X_raw.shape[1]} to {X.shape[1]} features. ---")

    print(f"\nTarget property: '{prop_key}'")
    print(f"Final primary features ({len(X.columns)}) being used for this run:\n  {', '.join(X.columns)}")

    # --- 5. Initialize and Run the Correct SISSO Model ---
    task_map = {
        'regression': SISSORegressor, 'multitask': SISSORegressor,
        'classification_svm': SISSOClassifier, 'classification_logreg': SISSOLogRegressor,
        'ch_classification': SISSOCHClassifier, 'convex_hull': SISSOCHClassifier,
        'classification': SISSOClassifier,
    }
    # Use 'task_type' or the legacy 'calc_type' key for backward compatibility
    task_key = config.get('task_type', config.get('calc_type', 'regression')).lower()
    SissoClass = task_map.get(task_key)
    if SissoClass is None:
        raise ValueError(f"Unknown 'task_type' or 'calc_type' in config: {task_key}")

    print("\n--- Initializing and running SISSO ---")
    sisso = SissoClass(**config)
    
    try:
        sisso.fit(X, y)
    except (ImportError, ValueError, RuntimeError) as e:
        print(f"\nFATAL ERROR during SISSO fit: {e}")
        print("Please check your configuration and dependencies (e.g., gurobipy for MIQP).")
        return

    # --- 6. Print Final Summary ---
    print("\n" + "="*25 + " FINAL MODEL REPORT " + "="*25)
    print(sisso.summary_report(X, y, sample_weight=None))

    print("\n" + "="*25 + " LATEX FORMULA " + "="*25)
    try:
        latex_formula = sisso.best_model_latex(target_name="P")
        print("Copy and paste the following into your LaTeX document:")
        print(latex_formula)
    except Exception as e:
        print(f"Could not generate LaTeX formula: {e}")
    
    print("\n" + "="*70)
    print(f"\nAnalysis complete. All detailed results and plots are saved in '{workdir}'.")


def create_demo_files():
    """Creates a sample dataset and config file for demonstration."""
    print("--- Creating demo files (demo_config.json, demo_data.csv) ---")
    
    # Use the provided SISSO_Sample_Dataset.csv if it exists, otherwise create a dummy
    sample_data_path = Path("SISSO_Sample_Dataset.csv")
    demo_data_path = Path("demo_data.csv")
    if sample_data_path.exists():
        data_to_use = sample_data_path
        property_to_use = "Target_U (eV)"
    else:
        np.random.seed(42)
        f1 = np.random.rand(100) * 10
        f2 = np.random.rand(100) * 5 + 2
        y = 2.5 * (f1 / f2) + 5 + np.random.randn(100) * 0.5
        df = pd.DataFrame({'feature1': f1, 'feature2': f2, 'target_property': y})
        df.to_csv(demo_data_path, index=False)
        data_to_use = demo_data_path
        property_to_use = "target_property"
    
    # Create a corresponding demo config file
    demo_config = {
        "data_file": str(data_to_use),
        "property_key": property_to_use,
        "workdir": "sisso_demo_omp_output",
        "depth": 2,
        "max_D": 2,
        "sis_sizes": [50],
        "search_strategy": "omp", 
        "task_type": "regression",
        "selection_method": "cv",
        "cv": 5,
        "n_jobs": -1,
        "random_state": 42
    }
    with open('demo_config.json', 'w') as f:
        json.dump(demo_config, f, indent=2)
    return 'demo_config.json'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Python-native Sure Independence Screening and Sparsifying Operator (pySISSO)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('config_file', nargs='?', default=None, help="Path to the JSON configuration file.\nIf not provided, a demo will be run.")
    args = parser.parse_args()

    if args.config_file:
        run_analysis(args.config_file)
    else:
        print("No config file provided. Running a demonstration.")
        config_to_run = create_demo_files()
        run_analysis(config_to_run)