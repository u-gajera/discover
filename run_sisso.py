"""
CLI driver that glues everything together.

Typical session:
    $ python run_sisso.py config_mohsen.json
    ├── Builds feature pool
    ├── Runs SIS + SO loops up to D_max
    └── Writes results:  final_models_summary.json, top_sis_candidates.csv, …

Example config entry:
    "max_dimension": 3    # search descriptors up to 3 terms
"""

import pandas as pd
import json
import argparse
import shutil
from pathlib import Path
import sys
import numpy as np
import sympy

# This setup assumes the script is run from the project directory
# and the package is in `src/pysisso`.
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from pysisso import (
    SISSORegressor,
    SISSOClassifier,
    SISSOLogRegressor,
    SISSOCHClassifier,
    print_descriptor_formula # Import the formatter
)
from pysisso.features import generate_features_iteratively # Need to import this for the return type
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


def run_analysis(config_path):
    """Core logic to run SISSO based on a config file."""
    # --- 1. Load Configuration ---
    print(f"--- Loading configuration from '{config_path}' ---")
    try:
        with open(config_path, 'r') as f:
            lines = [line for line in f if not line.strip().startswith('//')]
            config = json.loads("".join(lines))
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error reading or parsing config file '{config_path}': {e}")
        return

    # --- 2. Prepare Directories ---
    workdir = Path(config.get('workdir', 'sisso_output'))
    if workdir.exists():
        if input(f"Workdir '{workdir}' already exists. Overwrite? (y/N): ").lower() != 'y':
            print("Aborting."); return
        shutil.rmtree(workdir)
    
    workdir.mkdir(parents=True, exist_ok=True) # Ensure workdir is created

    # --- 3. Load and Preprocess Data ---
    data_file = config.get('data_file')
    if not data_file or not Path(data_file).exists():
        raise FileNotFoundError(f"Data file '{data_file}' specified in config not found.")
    
    print(f"--- Loading data from '{data_file}' ---")
    
    try:
        data_path = Path(data_file)
        if data_path.suffix == '.csv':
            data = pd.read_csv(data_path)
        else:
            raise ValueError(f"Unsupported data file format: '{data_path.suffix}'. Please use .csv.")
    except Exception as e:
        print(f"Pandas failed to read the data file. Error: {e}")
        return

    print("\n" + "="*20 + " DEBUGGING INFO: STAGE 1 (Raw Data) " + "="*20)
    print("Shape of data immediately after loading:", data.shape)
    print("First 5 rows of loaded data:\n", data.head())
    print("\nChecking for ALL missing values (NaNs) in each column of raw data:")
    print(data.isnull().sum())
    print("="*68 + "\n")

    prop_key = config.get('property_key')
    if not prop_key or prop_key not in data.columns:
        raise ValueError(f"Config 'property_key' '{prop_key}' not found in data columns. Available columns: {data.columns.tolist()}")
    
    non_feature_cols = config.get('non_feature_cols', [])
    feature_cols = [c for c in data.columns if c not in [prop_key] + non_feature_cols]
    
    used_cols = [prop_key] + feature_cols
    data_subset = data[used_cols].dropna()

    if data_subset.empty:
        print("\nFATAL: DataFrame is empty after dropping rows with missing values.")
        return

    y = data_subset[prop_key]
    X_raw = data_subset[feature_cols]
    X = pd.get_dummies(X_raw, dummy_na=False, drop_first=False)

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
    
    print("\n--- Initializing and running SISSO ---")
    sisso = SissoClass(**config)
    
    try:
        sisso.fit(X, y)
    except (ImportError, ValueError, RuntimeError) as e:
        print(f"\nFATAL ERROR during SISSO fit: {e}")
        return

    # --- 6. Save Additional, Plot-Friendly Results ---
    print("\n--- Saving additional results for plotting ---")

    # --- A. Save Top SIS Candidates ---
    sis_candidates_data = []
    feature_space_df = sisso.feature_space_df_
    
    print(f"  Evaluating top {min(20, feature_space_df.shape[1])} features from final feature space...")
    correlations = feature_space_df.corrwith(y).abs().sort_values(ascending=False)
    top_n_to_eval = min(20, len(correlations))
    
    for i in range(top_n_to_eval):
        feat_name = correlations.index[i]
        corr_score = correlations.iloc[i]
        X_feat = feature_space_df[[feat_name]]
        
        try:
            model = LinearRegression().fit(X_feat, y)
            y_pred = model.predict(X_feat)
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            
            sym_expr = sisso.feature_space_sym_map_.get(feat_name)
            formula = feat_name
            if sym_expr:
                full_formula_str = print_descriptor_formula(
                    [sym_expr], 
                    None,
                    'regression', 
                    True,
                    clean_to_original_map=sisso.sym_clean_to_original_map_
                )
                formula = full_formula_str.split("=")[-1].strip()

            sis_candidates_data.append({
                'Rank': i + 1, 'Feature': formula,
                'Internal_Name': feat_name,
                'Internal_Sym_Expr': str(sym_expr),
                'Correlation': corr_score, 'R2_Score': r2, 'RMSE': rmse
            })
        except Exception:
            continue
            
    sis_df = pd.DataFrame(sis_candidates_data)
    sis_candidates_path = workdir / "top_sis_candidates.csv"
    sis_df.to_csv(sis_candidates_path, index=False, float_format="%.5f")
    print(f"  Saved top SIS candidates to '{sis_candidates_path}'")

    # --- B. Save Final SISSO Models Summary ---
    models_summary = []
    fix_intercept = config.get('fix_intercept', False)
    
    for D, model_data in sisso.models_by_dim_.items():
        is_plottable = (model_data.get('coef') is not None) and (not model_data.get('is_parametric', False))
        
        coef_list = None
        if is_plottable and model_data['coef'] is not None:
             coef_list = model_data['coef'].flatten().tolist()
        
        sym_features_srepr = [sympy.srepr(sf) for sf in model_data.get('sym_features', [])]

        model_info = {
            'Dimension': D, 'is_best': D == sisso.best_D_,
            'Score_Train': model_data['score'], 'Score_CV': sisso.cv_results_.get(D, (None, None))[0],
            'is_plottable': is_plottable, 'fix_intercept': fix_intercept,
            'features': model_data.get('features', []),
            'coefficients': coef_list,
            'sym_features': sym_features_srepr,
            'Model_Type': 'Parametric' if model_data.get('is_parametric') else 'Linear' if is_plottable else 'Other'
        }
        models_summary.append(model_info)

    models_summary_path = workdir / "final_models_summary.json"
    with open(models_summary_path, 'w') as f:
        json.dump(models_summary, f, indent=2)
    print(f"  Saved final SISSO models summary to '{models_summary_path}'")

    # --- C. Save Symbol Map ---
    clean_map_str = {str(k): str(v) for k, v in sisso.sym_clean_to_original_map_.items()}
    symbol_map_path = workdir / "symbol_map.json"
    with open(symbol_map_path, 'w') as f:
        json.dump(clean_map_str, f, indent=2)
    print(f"  Saved symbol map to '{symbol_map_path}'")

    # --- 7. Print Final Standard Summary ---
    print("\n" + "="*25 + " FINAL MODEL REPORT " + "="*25)
    print(sisso.summary_report(X, y, sample_weight=None))
    print("\n" + "="*70)
    print(f"\nAnalysis complete. All results are saved in '{workdir}'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Python-native Sure Independence Screening and Sparsifying Operator (pySISSO)")
    parser.add_argument('config_file', help="Path to the JSON configuration file.")
    args = parser.parse_args()
    run_analysis(args.config_file)