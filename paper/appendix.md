# DISCOVER User Guide: Configuration and Execution

## A. Configuration with `config.json`

The DISCOVER workflow is controlled by a single JSON configuration file. This file allows users to define the dataset, specify the feature generation grammar, control the search and validation strategies, and set computational parameters. Below is a detailed explanation of each parameter.

### 1. Data & I/O Settings

This section defines the input data and output locations.

* **`data_file`** (string): Path to the input data file. The current implementation supports CSV format.
* **`property_key`** (string): The column name in the data file corresponding to the target property (the dependent variable, $y$). For multi-task regression, this can be a list of column names.
* **`non_feature_cols`** (list of strings): A list of columns to exclude from being used as primary features. This is useful for identifiers or metadata columns.
* **`workdir`** (string): The name of the directory where all output files, models, and plots will be saved. If the directory exists, the user will be prompted before overwriting.
* **`save_feature_space`** (boolean): If true, the entire generated feature space (after iterative screening) is saved to a compressed CSV file (`feature_space.csv.gz`) in the workdir. This can be useful for debugging or external analysis but may consume significant disk space.
* **`use_cache`** (boolean): If true, enables caching for feature generation. This dramatically speeds up repeated runs with the same feature generation settings by storing intermediate results in the workdir.

### 2. Feature-Space Construction

This section governs the symbolic feature engineering process.

* **`depth`** (integer): The maximum depth of the operator tree for feature generation. A depth of 1 applies operators to primary features. A depth of 2 applies operators to the results of depth 1, and so on. This is the primary control for the complexity of generated features.
* **`op_rules`** (list of dicts): Defines the set of mathematical operators used to build new features. The simplest form is a list of operator names, e.g., `[{"op": "add"}, {"op": "sub"}]`. Built-in operators include:
    * **Binary**: `add, sub, mul, div, abs_diff, harmonic_mean`.
    * **Unary**: `inv, abs, sq` (square), `cb` (cube), `sqrt, cbrt, log, exp, exp-` (exp(-x)), `sin, cos, sign`.
* **`parametric_ops`** (list of strings): A list of parametric operators to be used in a final non-linear refinement step after the main search. Current options include `'exp-'` (e.g., $e^{-p \cdot x}$) and `'pow'` (e.g., $x^p$), where $p$ is a free parameter optimized by the algorithm. This is only active for regression tasks.
* **`primary_units`** (dict): A dictionary mapping primary feature names (column headers) to their physical units. This enables `pint`-based dimensionality checking, ensuring that only dimensionally valid operations (e.g., addition of same-unit quantities, logarithm of dimensionless quantities) are performed. This is a critical feature for scientific applications.
* **`min_abs_feat_val`, `max_abs_feat_val`** (float): Minimum and maximum absolute values allowed for any generated feature. This helps prevent numerical instability (underflow/overflow).

### 3. Search & Selection Strategy

This section defines how the best descriptor model is identified from the vast feature space.

* **`max_D`** (integer): The maximum dimension (number of terms) of the final descriptor to search for. The algorithm will find the best 1-D, 2-D, ..., up to `max_D`-dimensional models.
* **`sis_sizes`** (list of integers): In the iterative framework, only the first element is used. It specifies the number of top features to retain after Sure Independence Screening (SIS) at each `depth` of feature generation.
* **`search_strategy`** (string): The algorithm used to find the best D-dimensional descriptor from the screened feature space. Options: `'greedy'`, `'brute_force'`, `'omp'`, `'sisso++'`, `'rmhc'`, `'sa'`, `'miqp'`. A detailed explanation of each strategy is provided in Section 6.
* **`max_feat_cross_correlation`** (float): A threshold (0.0 to 1.0) used to prune the feature space by removing highly correlated features before the main search begins. A value of 0.95 means that from any cluster of features correlated >0.95, only one is kept.

### 4. Model Construction & Validation

This section controls the final model fitting and selection process.

* **`task_type`** (string): Specifies the type of machine learning problem. This determines the model and scoring function. Options: `'regression'`, `'multitask'`, `'classification_svm'`, `'classification_logreg'`, `'ch_classification'`.
* **`selection_method`** (string): The method used to select the best dimension (`D`) from the models found.
    * `'cv'`: K-Fold Cross-validation.
    * `'bootstrap'`: Bootstrap with out-of-bag (OOB) error estimation.
    * `'aic'`/`'bic'`: Akaike/Bayesian Information Criterion (for regression only).
* **`cv`** (integer): The number of folds for cross-validation. Set to -1 for Leave-One-Out CV (LOOCV).
* **`fix_intercept`** (boolean): If true, the linear model is fitted without an intercept term ($\beta_0 = 0$).

### 5. Computational Settings

* **`n_jobs`** (integer): The number of CPU cores to use for parallelized tasks. Set to -1 to use all available cores.
* **`device`** (string): The computational device for accelerated calculations. Options: `'cpu'`, `'cuda'` (for NVIDIA GPUs, requires `cupy`), `'mps'` (for Apple Silicon GPUs, requires `pytorch`). GPU acceleration is available for feature generation and certain search/scoring functions.
* **`random_state`** (integer): A seed for the random number generator to ensure reproducibility of results.

---

### 6. Advanced Feature Selection and Search Strategies

DISCOVER includes several advanced methods for both screening candidate features and searching for the final descriptor. These methods offer trade-offs between computational cost, robustness, and the ability to capture complex relationships.

#### Leave-One-Out Cross-Validation (LOOCV)

* **Purpose:** LOOCV is an exhaustive form of cross-validation where the number of folds is equal to the number of data points. For a dataset of size $N$, it trains $N$ separate models, each on $N-1$ points, and validates on the single held-out point.
* **When to Use:** It is highly recommended for **small datasets** (e.g., $N < 100$) where K-Fold CV might yield volatile results due to the small size of the validation sets. It provides a nearly unbiased estimate of model performance but is computationally very expensive for large datasets.
* **Configuration:** To enable LOOCV, set the `cv` parameter to `-1`.
    ```json
    "selection_method": "cv",
    "cv": -1
    ```
* **Compatibility:** Can be used with any `task_type`.

#### Decision Tree Feature Screening

* **Purpose:** This method replaces the default correlation-based screening (SIS) with a more sophisticated approach. It scores features based on a combination of their importance in a trained decision tree model (the likelihood) and their symbolic complexity (the prior). The final score favors features that are both predictive and simple.
* **When to Use:** Use when you suspect that the relationship between features and the target is highly **non-linear or involves complex interactions**. Standard correlation might miss features that are only important in specific regimes. This method is slower than correlation-based screening but can lead to a higher-quality feature space.
* **Configuration:** Set the `sis_method` parameter.
    ```json
    "sis_method": "decision_tree"
    ```
* **Compatibility:** Works for both `regression` and `classification` tasks. For multi-task regression, it treats the problem as a standard multi-output regression task. It is a CPU-only feature.

#### Search Strategy: Greedy (`'greedy'`)

* **Purpose:** The default search strategy. This is a forward-selection algorithm that builds the descriptor one feature at a time. In the first step, it selects the single best feature. In the second step, it finds the next feature that, when added to the first, best explains the *residual* (the remaining error). It continues this process until the desired dimension is reached.
* **When to Use:** This is the fastest search algorithm and serves as an excellent baseline. It is a good first choice for exploring a new problem, but it can be susceptible to finding local optima.
* **Compatibility:** Can be used with any `task_type`.

#### Search Strategy: Brute-Force (`'brute_force'`)

* **Purpose:** This method exhaustively evaluates every possible combination of features for a given dimension. It guarantees that the returned model is the true optimum for that dimension from the given feature space.
* **When to Use:** Use when the number of candidate features and the maximum dimension are both small, and you want to be certain you have found the best possible linear model. The number of combinations grows factorially, so this becomes computationally infeasible very quickly.
* **Compatibility:** Can be used with any `task_type`.

#### Search Strategy: Orthogonal Matching Pursuit (`'omp'`)

* **Purpose:** OMP is a greedy algorithm that improves upon the standard greedy search. At each step, it selects the feature that is most correlated with the *current residual* of the model. After adding the new feature, it refits the model on the *entire set* of selected features using ordinary least squares. This re-fitting step makes it more robust than a simple greedy search.
* **When to Use:** OMP is a fast, robust, and reliable choice for most regression problems. It provides a good balance between speed and quality of the solution, often outperforming the standard greedy search.
* **Compatibility:** This method is designed for `regression` and `multitask` regression tasks only.

#### Search Strategy: SISSO++ (`'sisso++'`)

* **Purpose:** This is a highly efficient breadth-first search algorithm. It intelligently explores the search space by building up descriptors dimension by dimension, keeping a "beam" of the best candidates at each level. Its speed comes from using QR decomposition updates to efficiently calculate the residual sum of squares without refitting the model from scratch for each new combination.
* **When to Use:** SISSO++ is an excellent choice for large feature spaces where brute-force is infeasible. It is often much faster than other search methods for higher dimensions ($D > 2$) and can be GPU-accelerated for regression tasks.
* **Configuration:**
    ```json
    "search_strategy": "sisso++",
    "beam_width_decay": 1.0
    ```
    * `beam_width_decay`: A factor ($\le 1.0$) to shrink the search beam at each dimension. A value of 1.0 keeps the beam size constant, while a value like 0.8 speeds up the search for higher dimensions by focusing only on the most promising candidates.
* **Compatibility:** Can be used with any `task_type`. For classification and multitask problems, it uses an Ordinary Least Squares proxy for the search phase, with a final refit using the correct model.

#### Search Strategy: Random Mutation Hill Climbing (`'rmhc'`)

* **Purpose:** RMHC is a metaheuristic search algorithm that refines an existing solution. It starts with a good "seed" model (found via a fast greedy search) and then iteratively tries to improve it by making small, random changes ("mutations"). A mutation consists of swapping one feature in the current descriptor with a random feature from the wider pool. If the change improves the model's score, it is accepted.
* **When to Use:** RMHC is an excellent choice when you want a more thorough search than a simple greedy algorithm but cannot afford the computational cost of brute-force. It is very effective at finding small improvements and escaping the local optima that can trap a greedy search.
* **Configuration:**
    ```json
    "search_strategy": "rmhc",
    "rmhc_iterations": 200,
    "rmhc_restarts": 5
    ```
    * `rmhc_iterations`: The number of mutations to attempt for each restart.
    * `rmhc_restarts`: The number of times the hill-climbing process is restarted from the best-known solution. Multiple restarts help ensure the search is not permanently stuck on a suboptimal peak.
* **Compatibility:** Can be used with any `task_type`.

#### Search Strategy: Simulated Annealing (`'sa'`)

* **Purpose:** SA is a powerful global optimization technique inspired by annealing in metallurgy. It starts by exploring the feature space broadly (high "temperature") and gradually narrows its search to promising regions (low "temperature"). Unlike hill climbing, SA can accept a worse solution with a certain probability, allowing it to escape local optima and find a potentially better global optimum.
* **When to Use:** SA is one of the most robust search strategies. Use it when the feature space is likely rugged with many local minima and you need a high-confidence global search without resorting to brute-force. It is generally slower than RMHC but more likely to find the global optimum.
* **Configuration:**
    ```json
    "search_strategy": "sa",
    "sa_initial_temp": 1.0,
    "sa_final_temp": 1e-4,
    "sa_cooling_rate": 0.99,
    "sa_acceptance_rule": "metropolis"
    ```
    * `sa_initial_temp`: The starting temperature. Higher values encourage more exploration.
    * `sa_final_temp`: The temperature at which the search terminates.
    * `sa_cooling_rate`: The factor by which the temperature is multiplied at each step. Values closer to 1.0 mean slower, more thorough cooling.
    * `sa_acceptance_rule`: The criterion for accepting a worse move. `'metropolis'` uses $P = e^{-\Delta E / T}$, while `'glauber'` uses $P = 1 / (1 + e^{\Delta E / T})$. Metropolis is more common.
* **Compatibility:** Can be used with any `task_type`.

#### Search Strategy: Mixed-Integer Quadratic Programming (`'miqp'`)

* **Purpose:** This strategy formulates the descriptor selection problem as a formal mathematical optimization problem. It finds the **provably optimal** set of features that minimizes the L0-norm (i.e., the number of non-zero coefficients) for a given descriptor dimension.
* **When to Use:** Use MIQP when you require mathematical proof of optimality for the final model and the feature space is of a manageable size (typically a few hundred features). It is computationally intensive and relies on an external solver.
* **Configuration:**
    ```json
    "search_strategy": "miqp"
    ```
* **Compatibility:** This method is only available for `regression` tasks with an `'l2'` loss. It requires the `gurobipy` Python package and a valid Gurobi license.

### Example: `config.json`

```json
{
  // ====================================================================
  //                        1. DATA & I/O SETTINGS                        
  // ====================================================================
  "data_file":               "Sample_Dataset.csv",
  "property_key":             "Target_U (eV)",
  "non_feature_cols":        ["material_id"],
  "workdir":                 "discover_output",
  "save_feature_space":      false,
  
  // ====================================================================
  //                   2. FEATURE-SPACE CONSTRUCTION                    
  // ====================================================================
  "depth":                    2,
  "op_rules": [
    {"op": "add"}, {"op": "sub"}, {"op": "mul"}, {"op": "div"},
    {"op": "inv"}, {"op": "abs"}, {"op": "sq"}, {"op": "cb"}, 
    {"op": "sqrt"}, {"op": "cbrt"}, {"op": "log"}, {"op": "exp"}
  ],
  "primary_units": {
    "Activation_Energy (eV)": "electron_volt", 
    "Lattice_Constant_A (\AA)": "angstrom"
  },
  
  // ====================================================================
  //                    3. SEARCH & SELECTION STRATEGY                    
  // ====================================================================
  "max_D":                   3,
  "sis_sizes":               [100],
  "search_strategy":         "rmhc",
  "max_feat_cross_correlation": 0.95,

  // ====================================================================
  //                  4. MODEL CONSTRUCTION & VALIDATION                  
  // ====================================================================
  "task_type":               "regression",
  "selection_method":        "cv",
  "cv":                      10,
  "fix_intercept":           false,
  
  // ====================================================================
  //                       5. COMPUTATIONAL SETTINGS                     
  // ====================================================================
  "n_jobs":                  -1,
  "device":                  "cpu", 
  "random_state":             42
}

```

## B. Usage of `run_discover.py`

The primary entry point for running a DISCOVER analysis is the command-line script `run_discover.py`. It orchestrates the entire workflow, from data loading to model fitting and results serialization.

### Execution

To execute a run, a user needs two files:

1. A data file in CSV format.
2. A configuration file in JSON format (as described in Section A).

The script is invoked from the terminal, passing the path to the configuration file as the sole argument:

```bash
$ python run_discover.py path/to/your/config.json

```

### Workflow

Upon execution, the script performs the following steps:

1. **Load Configuration**: Parses the specified JSON file.
2. **Prepare Workspace**: Creates the output directory specified by `workdir`. It includes a safety prompt to prevent accidental overwriting of existing results.
3. **Load Data**: Reads the CSV file specified by `data_file` using pandas. It separates the target property (`property_key`) from the primary features and handles missing values by dropping the corresponding rows.
4. **Initialize Model**: Selects the appropriate model class (e.g., `DiscoverRegressor`, `DiscoverClassifier`) based on the `task_type` in the configuration.
5. **Run DISCOVER**: The `.fit(X, y)` method is called. This is the main computational step, which involves:
* Iterative feature generation and screening up to the specified `depth`.
* Search for the best D-dimensional models using the chosen `search_strategy`.
* Model selection via the specified `selection_method` (e.g., CV) to determine the optimal dimension `D`.
* Final model fitting and diagnostic calculations (VIF, coefficient errors).


6. **Save Results**: The script saves a comprehensive set of results to the `workdir`:
* `SISSO.out`: A human-readable summary report of the final model, including its formula, performance metrics, and diagnostics.
* `final_models_summary.json`: A structured JSON file containing detailed information for each dimensional model found, including features, coefficients, and scores. This is ideal for programmatic post-processing and plotting.
* `top_sis_candidates.csv`: A CSV file ranking the top individual features from the final screened space by their correlation with the target, along with their single-feature R$^2$ and RMSE.
* `symbol_map.json`: A JSON file mapping the internal symbolic names (`f0`, `f1`, ...) back to the original feature names from the input data file.
* `plots/`: A directory containing plots for model selection (`selection_scores.png`), parity, etc.
* `models/`: A directory containing detailed `.dat` files for each dimensional model, including its formula and coefficients.


7. **Print Summary**: The final model report is printed to the console.

### Illustrative Code Snippet

The core logic of `run_discover.py` demonstrates its role as a high-level driver.

```python
# ... (imports and argument parsing) ...
import json
import pandas as pd
from pathlib import Path
from discover import DiscoverRegressor, DiscoverClassifier

def run_analysis(config_path):
    # 1. Load Configuration
    with open(config_path, 'r') as f:
        # Note: Added logic to handle comments in the JSON file
        lines = [line for line in f if not line.strip().startswith('//')]
        config = json.loads("".join(lines))

    # 2. Prepare Directories and Data
    workdir = Path(config.get('workdir', 'discover_output'))
    # ... (directory setup logic) ...
    data = pd.read_csv(config.get('data_file'))
    y = data[config.get('property_key')]
    feature_cols = [c for c in data.columns if c not in [config.get('property_key')] + config.get('non_feature_cols', [])]
    X = data[feature_cols]

    # 3. Initialize and Run the Correct DISCOVER Model
    task_map = {
        'regression': DiscoverRegressor,
        'classification_svm': DiscoverClassifier,
        # ... other task types
    }
  
    task_key = config.get('task_type', 'regression').lower()
    DiscoverClass = task_map.get(task_key)
    
    print("\n--- Initializing and running DISCOVER ---")
    discover = DiscoverClass(**config)
    discover.fit(X, y)

    # 4. Save Additional Results and Print Final Report
    print("\n--- Saving additional results for plotting ---")
    # ... (logic to save top_sis_candidates.csv, final_models_summary.json)
    
    print("\n--- FINAL MODEL REPORT ---")
    print(discover.summary_report(X, y, sample_weight=None))

```
