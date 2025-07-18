"""
==========================================================================
    pySISSO Definitive Test Matrix & Diagnostic Script
==========================================================================

This script runs an exhaustive matrix of tests to verify that all major
code paths and combinations of parameters in pySISSO are functioning.
"""

import pandas as pd
import numpy as np
import sys
import shutil
from pathlib import Path
import time
import warnings
import itertools

# This setup assumes the script is run from the project directory
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
except NameError:
    sys.path.insert(0, 'src')

from pysisso import (
    SISSORegressor,
    SISSOClassifier,
    SISSOCHClassifier
)
from pysisso.search import GUROBI_AVAILABLE
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# --- Test Data Generation ---

### MODIFICATION: Updated data generation to use 4 features ###
def create_regression_data():
    """Creates a 100-sample, 4-feature dataset for regression."""
    np.random.seed(42)
    n_samples = 100
    X1 = np.linspace(1, 10, n_samples)
    X2 = np.cos(X1) + np.random.randn(n_samples) * 0.1
    X3 = np.random.rand(n_samples) * 5
    X4 = X1**2
    # True model: y = 2*X1 + 5*X2 - 0.5*X3^2 + noise
    y = 2 * X1 + 5 * X2 - 0.5 * X3**2 + np.random.randn(n_samples) * 0.2
    X_df = pd.DataFrame({
        'FeatureA': X1, 
        'FeatureB': X2,
        'FeatureC': X3,
        'FeatureD': X4
    })
    return y, X_df

def create_multitask_data():
    """Creates a 100-sample, 4-feature dataset for multitask regression."""
    y_reg, X = create_regression_data()
    # Second task: y2 = 0.1*X4 - 3*X2*X3 + noise
    y2 = 0.1 * X['FeatureD'] - 3 * X['FeatureB'] * X['FeatureC'] + np.random.randn(len(X)) * 0.2
    y_multi = pd.DataFrame({'Task1': y_reg, 'Task2': y2})
    return y_multi, X

def create_classification_data():
    """Creates a 100-sample, 4-feature dataset for classification."""
    np.random.seed(42)
    n_samples_per_class = 50
    # Class 0: Centered around (-1, -1)
    X1_c0 = np.random.randn(n_samples_per_class) - 1
    X2_c0 = np.random.randn(n_samples_per_class) - 1
    # Class 1: Centered around (1, 1)
    X1_c1 = np.random.randn(n_samples_per_class) + 1
    X2_c1 = np.random.randn(n_samples_per_class) + 1
    # Combine the informative features
    X1 = np.concatenate([X1_c0, X1_c1])
    X2 = np.concatenate([X2_c0, X2_c1])
    # Add two noise features
    X3 = np.random.randn(n_samples_per_class * 2)
    X4 = np.random.randn(n_samples_per_class * 2)
    
    y = np.concatenate([np.zeros(n_samples_per_class, dtype=int), np.ones(n_samples_per_class, dtype=int)])
    
    X_df = pd.DataFrame({
        'FeatureA': X1, 
        'FeatureB': X2,
        'FeatureC_noise': X3,
        'FeatureD_noise': X4
    })
    return pd.Series(y, name='Target'), X_df


def run_single_test(test_name, config, X, y):
    """Runs a single SISSO instance with a given configuration."""
    print("\n" + "="*70)
    print(f"  RUNNING TEST: {test_name}")
    print("="*70)
    
    workdir = config['workdir']
    if Path(workdir).exists():
        shutil.rmtree(workdir)

    start_time = time.time()
    try:
        task_type = config.get("calc_type", config.get("task_type"))
        model_class_map = {
            'regression': SISSORegressor,
            'multitask': SISSORegressor,
            'classification_svm': SISSOClassifier,
            'ch_classification': SISSOCHClassifier
        }
        ModelClass = model_class_map.get(task_type, SISSORegressor)

        model = ModelClass(**config)
        model.fit(X, y)
        
        if model.best_model_ is None or model.best_D_ is None:
            raise RuntimeError("Fit completed but no valid model was selected.")
        
        duration = time.time() - start_time
        print(f"\n[SUCCESS] Test '{test_name}' passed in {duration:.2f} seconds.")
        return True, test_name, duration

    except Exception as e:
        duration = time.time() - start_time
        print(f"\n[FAILURE] Test '{test_name}' failed after {duration:.2f} seconds.")
        print(f"  ERROR TYPE: {type(e).__name__}")
        print(f"  ERROR MSG:  {e}")
        # import traceback; traceback.print_exc()
        return False, test_name, duration

# --- Main Test Execution Logic ---
if __name__ == '__main__':
    results = []
    skipped_tests = []
    test_count = 0

    FEATURE_GENERATORS = ["iterative", "ga"]
    SEARCH_STRATEGIES = ["greedy", "omp", "brute_force", "sisso++", "miqp"]
    SELECTION_METHODS = ["cv", "bootstrap", "aic", "bic"]

    base_config = {
        "op_rules": [{"op": "add"}, {"op": "sub"}, {"op": "mul"}, {"op": "sq"}],
        "max_D": 2, "sis_sizes": [25], # Increased sis_size for 4 features
        "n_jobs": -1, "random_state": 42,
        "depth": 2, "cv": 3, "n_bootstrap": 5
    }
    
    fast_ga_params = {
        "ga_generations": 1, "ga_population_size": 10, # Slightly larger for more variety
        "ga_elitism_count": 2, "ga_tournament_size": 3
    }
    
    print("\nStarting pySISSO Definitive Test Suite...")

    for task_type, data_loader in [("regression", create_regression_data), 
                                   ("multitask", create_multitask_data)]:
        y_data, X_data = data_loader()
        for gen_strat, search_strat, sel_method in itertools.product(
            FEATURE_GENERATORS, SEARCH_STRATEGIES, SELECTION_METHODS):
            if search_strat == "miqp" and not GUROBI_AVAILABLE:
                if "miqp" not in "".join(skipped_tests): skipped_tests.append("Skipped: 'miqp' (Gurobi not found).")
                continue
            if sel_method in ["aic", "bic"] and task_type == "multitask":
                if "multitask" not in "".join(skipped_tests): skipped_tests.append(f"Skipped: '{sel_method}' is not for multitask.")
                continue
            if search_strat not in ["greedy", "omp", "brute_force", "sisso++", "miqp"]: continue
            test_count += 1
            test_name = f"{task_type}_{gen_strat}_{search_strat}_{sel_method}"
            config = base_config.copy()
            config.update({ "task_type": task_type, "feature_generation_strategy": gen_strat,
                            "search_strategy": search_strat, "selection_method": sel_method,
                            "workdir": f"test_output/{test_name}"})
            if gen_strat == "ga": config.update(fast_ga_params)
            if search_strat == "brute_force": config["depth"], config["sis_sizes"] = 1, [10]
            success, name, dur = run_single_test(test_name, config, X_data, y_data)
            results.append((success, name, dur))

    y_data_class, X_data_class = create_classification_data()
    
    for gen_strat, search_strat in itertools.product(
        FEATURE_GENERATORS, ["greedy", "brute_force", "sisso++"]):
        test_count += 1
        test_name = f"classification_svm_{gen_strat}_{search_strat}"
        config = base_config.copy()
        config.update({ "task_type": "classification_svm", "feature_generation_strategy": gen_strat,
                        "search_strategy": search_strat, "selection_method": "cv",
                        "workdir": f"test_output/{test_name}"})
        if gen_strat == "ga": config.update(fast_ga_params)
        if search_strat == "brute_force": config["depth"], config["sis_sizes"] = 1, [10]
        success, name, dur = run_single_test(test_name, config, X_data_class, y_data_class)
        results.append((success, name, dur))
        
    test_count += 1
    test_name = "ch_classification_geometric_search"
    config = base_config.copy()
    config.update({
        "task_type": "ch_classification", "workdir": f"test_output/{test_name}",
        "max_D": 1, "depth": 1, "sis_sizes": [20]
    })
    success, name, dur = run_single_test(test_name, config, X_data_class, y_data_class)
    results.append((success, name, dur))

    print("\n\n" + "="*80)
    print("                      DEFINITIVE TEST SUITE SUMMARY")
    print("="*80)
    passed_count = 0
    if results:
        print("--- TEST RESULTS ---")
        for success, name, duration in sorted(results, key=lambda x: x[1]):
            status = "PASSED" if success else "FAILED"
            print(f"  - {name:<50} | Status: {status:<7} | Time: {duration:5.2f}s")
            if success:
                passed_count += 1
    if skipped_tests:
        print("\n--- SKIPPED TESTS (Correctly Avoided Invalid Combinations) ---")
        for reason in sorted(list(set(skipped_tests))):
            print(f"  - {reason}")
    total_failed = len(results) - passed_count
    print("-" * 80)
    print(f"  SUMMARY: {passed_count} PASSED, {total_failed} FAILED, out of {len(results)} executed tests.")
    print("="*80)
    if total_failed > 0:
        print("\n!!! One or more tests failed. Please review the output above for errors. !!!")
        sys.exit(1)
    else:
        print("\n*** All executed tests passed successfully! ***")