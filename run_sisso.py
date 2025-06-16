"""
================================================================================
  Universal Runner for the Python Sure Independence Screening and
                       Sparsifying Operator (pySISSO)
================================================================================

This script serves as the main entry point for running pySISSO. It is designed
to be generic and controlled entirely by a configuration file.

To run an analysis, execute this script from the command line and pass the
path to your configuration file as an argument:

    python run_new_analysis.py your_config.json

The script will automatically handle:
  - Loading and parsing the configuration (including advanced rules).
  - Preprocessing the data.
  - Selecting and initializing the correct SISSO model.
  - Running the complete SISSO workflow.
  - Printing a final summary report, including a LaTeX formula.
  - Saving all detailed outputs and new interpretation plots.

"""
import pandas as pd
import json
import argparse
import shutil
from pathlib import Path
import sys
import os

# Add the 'src' directory to the Python path to allow importing the pysisso package
# This assumes the script is in the parent directory of 'src/'
sys.path.append(str(Path(__file__).parent / 'src'))

from pysisso import (
    SISSORegressor,
    SISSOClassifier,
    SISSOLogRegressor,
    SISSOCHClassifier
)

def main():
    """Handles the Command Line Interface execution."""
    parser = argparse.ArgumentParser(
        description="Python-native Sure Independence Screening and Sparsifying Operator (pySISSO)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('config_file', type=str, help="Path to the JSON configuration file.")
    args = parser.parse_args()

    # --- 1. Load Configuration ---
    print(f"--- Loading configuration from '{args.config_file}' ---")
    with open(args.config_file, 'r') as f:
        lines = [line for line in f if not line.strip().startswith('//')]
        config = json.loads("".join(lines))

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
    if not prop_key:
        raise ValueError("Config must contain 'property_key'.")
    
    non_feature_cols = config.get('non_feature_cols', [])
    
    y = data[prop_key]
    X_raw = data.drop(columns=[prop_key] + non_feature_cols)
    X = pd.get_dummies(X_raw, dummy_na=False, drop_first=False)
    if X.shape[1] > X_raw.shape[1]:
        print(f"--- Performed one-hot encoding. Feature set expanded to {X.shape[1]} features. ---")

    print(f"\nTarget property: '{prop_key}'")
    print(f"Final primary features ({len(X.columns)}) being used for this run:\n  {', '.join(X.columns)}")

    # --- 5. Initialize and Run the Correct SISSO Model ---
    task_map = {
        'regression': SISSORegressor, 'multitask': SISSORegressor,
        'classification_svm': SISSOClassifier, 'classification_logreg': SISSOLogRegressor,
        'ch_classification': SISSOCHClassifier, 'convex_hull': SISSOCHClassifier,
        'classification': SISSOClassifier,
    }
    task_key = config.get('task_type', config.get('calc_type', 'regression')).lower()
    SissoClass = task_map.get(task_key)
    if SissoClass is None:
        raise ValueError(f"Unknown 'task_type' or 'calc_type' in config: {task_key}")

    print("\n--- Initializing and running SISSO ---")
    sisso = SissoClass(**config)
    sisso.fit(X, y)

    # --- 6. Print Final Summary and Demonstrate New Features ---
    print("\n" + "="*25 + " FINAL MODEL REPORT " + "="*25)
    print(sisso.summary_report(X, y, sample_weight=None))

    # --- NEW: DEMONSTRATE LATEX OUTPUT ---
    print("\n" + "="*25 + " LATEX FORMULA " + "="*25)
    try:
        latex_formula = sisso.best_model_latex(target_name="P")
        print("Copy and paste the following into your LaTeX document:")
        print(latex_formula)
    except Exception as e:
        print(f"Could not generate LaTeX formula: {e}")
    
    print("\n" + "="*68)
    print(f"\nAnalysis complete. All detailed results, models, and plots are saved in '{workdir}'.")
    print("New interpretation plots like 'feature_importance.png' and 'partial_dependence.png' are now available in the plots subdirectory.")


if __name__ == '__main__':
    # This setup allows running from CLI with a config file
    # or running a built-in demo if no args are given.
    if len(sys.argv) > 1:
        main()
    else:
        print("Usage: python run_new_analysis.py <path_to_config.json>")
        print("\nPlease provide a configuration file to run an analysis.")