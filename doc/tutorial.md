# Getting Started: A Step-by-Step Tutorial

This tutorial will guide you through your first analysis using DISCOVER. We will go through installation, data preparation, configuration, execution, and interpretation of the output. We will take `config.json` as an example.

### Step 1: Installation

Ensure you have Python 3.8+ installed. It is strongly advised to use a virtual environment.

1.  **Clone the Repository:**
```bash
git clone <your-repository-url>
cd discover-project
```

2.  **Activate a Virtual Environment:**
```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\\\\Scripts\\\\activate`
```

3.  **Install Core Dependencies:**
```bash
    pip install pandas numpy sympy scikit-learn matplotlib seaborn joblib
```


4.  **Install Optional Dependencies (Optional):**
    *   **For GPU Acceleration (NVIDIA):**
        ```bash
        # Put XX in place of your CUDA version (e.g., 118 for CUDA 11.8)
        pip install cupy-cudaXX
        ```
*   **For GPU Acceleration (Apple Silicon):**
        ```bash
        pip install torch torchvision torchaudio
        ```
    *   **For Unit-Awareness:**
        ```bash
        pip install pint
```
``` 

### Step 2: Prepare Your Data

DISCOVER expects your data in a single **CSV file**.
-   Each row should be an example (e.g., a material or a data point).
-   Each column should be a primary feature or the target property.

For our case, `manuel_data.csv` contains columns for main features like `R_A` (ionic radius), `ELN_A` (electronegativity), and target property `E_a` (migration barrier).

### Step 3: Prepare the Analysis

The workflow is managed by `config_manuel.json`. Let's break down the critical parts for an initial run.

```json
{
    // 1. DATA & I/O
    "data_file": "manuel_data.csv",
    "property_key": "E_a",
    "non_feature_cols": ["Name", "Delta_E", "."],
},

"workdir": "manuel_data",

    // 2. FEATURE CONSTRUCTION
    "depth": 3,
    "op_rules": [
      {"op": "add"}, {"op": "sub"}, {"op": "mul"}, {"op": "div"},
      {"op": "sqrt"}, {"op": "sq"}
    ],

    // 3. SEARCH & SELECTION
    "max_D": 2,
    "search_strategy": "sa",
"selection_method": "bic",

    // 4. MODEL & VALIDATION
    "task_type": "regression",
    "fix_intercept": false,

    // 5. COMPUTATIONAL
    "n_jobs": -1,
    "device": "cpu",
    "random_state": 42
}
```

-   **`data_file`**: Path to your input CSV.
-   **`property_key`**: The exact name of the target column in your CSV.
-   **`workdir`**: Current working directory where output files are written.
-   **`depth`**: Maximum complexity of the generated features. `2` or `3` is a good default.
-   **`op_rules`**: Mathematical operations with which features are merged.
-   **`max_D`**: Maximum number of features (dimension) in the final model.
-   **`search_strategy`**: The strategy used to find the best feature combination. `greedy` is fast, but `sisso++` or `sa` (Simulated Annealing) could be more thorough.
-   **`selection_method`**: The criterion to select the best dimension. `cv` (Cross-Validation) is stable, but `bic` (Bayesian Information Criterion) penalizes model complexity.
-   **`task_type`**: The machine learning task type. `regression` is used for continuous values.

### Step 4: Run the Analysis

With your configuration set, execute the `run_discover.py` script in your command line:

```bash
python run_discover.py config_manuel.json
```

DISCOVER will start, giving live output on how it's proceeding:
1.  Loading data and config.
2.  Creating features iteratively, reporting at each `depth`.
3.  Printing the top-screened features at each iteration.
4.  Executing the selected search strategy for each dimension `D`.
5.  Checking the models and selecting the best dimension.
6. Printing the final summary report to the console.

### Step 5: Understanding the Output

All output is saved to the directory specified by `workdir` (e.g., `manuel_data/`).

**Key Console Output:**
The final part of the console output is the summary report, the most important takeaway:
```
========================= FINAL MODEL REPORT =========================
DISCOVER Summary
=================
TASK_TYPE: regression
MODEL_SELECTION_METHOD: BIC
SELECTED_DIMENSION: 2
SEARCH_STRATEGY: SA
.

******* BEST MODEL (D=2) ********
Score BIC Score: -123.456
R2/Accuracy on full data: 0.9512

============================================================
Final Symbolic Model: regression
============================================================
P = -0.1234 + 0.5678*sqrt(R_A/R_B) + 0.9012*Abs(ELN_A - ELN_B)
```


**Important Files in the `workdir`:**
-   `discover.out`: A text file with the same exhaustive summary report to the console.
-   `final_models_summary.json`: A machine-readable JSON file providing exhaustive detail regarding the top discovered model for each dimension, including coefficients and feature expressions.
-   `top_sis_candidates.csv`: A CSV file ranking the best individual features (1D models) discovered, and worth examining which top features are most effective.
-   `plots/`: A folder of automatically generated plots, such as the model selection score plot (`selection_scores.png`).
-   `models/`: Contains detailed text files for each model dimension (`model_D01.dat`, `model_D02.dat`, etc.).

### What's Next?

Congratulations! You've successfully executed your first DISCOVER analysis. You now possess an interpretable, symbolic model that describes your data.

For exploring more sophisticated features, check out our **[How-To Guides](./howto.md)**, where you can observe how to:
-   Execute DISCOVER programmatically inside a Python program.
-   Employ GPU acceleration for massive speed improvements.
-   Employ sophisticated search strategies and operators.
-   Classify tasks.

### `doc/contribution.md`

```markdown
# Contributing to DISCOVER

Thank you for your interest in writing for DISCOVER! We welcome contributions that introduce new functionality, fix bugs, or improve documentation. This handbook should provide a good starting point.

### Code Structure Overview

The project is divided into several key modules:

-   `run_discover.py`: The level-zero command-line entry point.
-   `discover/models.py`: The central `DiscoverBase` class that executes the workflow.
-   `discover/features.py`: Contains all feature space generation functionality and the `UFeature` unit-aware class.
-   `discover/search.py`: Determines the different search methods (greedy, SISSO++, MIQP).
-   `discover/scoring.py`: Contains model evaluation functions (SIS, CV, regression/classification scores) and GPU-supported kernels.
-   `discover/utils.py`: Contains utility functions for plotting, result saving, and formula formatting.
-   `discover/constants.py`: Contains global constants (task types, etc.).

### Adding a New Mathematical Operator

Adding a custom operator is one of the easiest expansions for DISCOVER. Operators are defined in `discover/features.py`.

#### Adding a Custom Unary Operator

1.  Open `discover/features.py`.
2.  Navigate to the `CUSTOM_UNARY_OP_DEFS` dictionary.
3.  Add a new entry with a unique name.

The entry should be a dictionary with the following keys:
-   `func`: The mathematical function (e.g., `lambda x: np.tanh(x)`).
-   `sym_func`: Symbolic variant with `sympy` (e.g., `lambda s: sympy.tanh(s)`).
-   `unit_check` (optional): Function returning `True` if the input unit is acceptable. For example, `lambda u: u.dimensionless` for trig functions.
-   `domain_check` (optional): Function returning `True` if input values are acceptable (e.g., `lambda f: np.all(f.values > 0)` for `log`).
-   `name_func`: A function that produces the string name of the new feature (e.g., `lambda n: f"tanh({n})"`).

**Adding a `tanh` operator**
```python
# In discover/features.py
CUSTOM_UNARY_OP_DEFS = {
    #. other operators
    'tanh': {
        'func': np.tanh,
        'sym_func': lambda s: sympy.tanh(s),"}
```
'unit_check': lambda u: u.dimensionless,
        'domain_check': None,
        'name_func': lambda n: f"tanh({n})",
    }
}
```

#### Adding a Custom Binary Operator

1.  Open `discover/features.py`.
2.  Navigate to the `CUSTOM_BINARY_OP_DEFS` dictionary.
3.  Add a new entry.

The structure is similar to unary operators but takes two arguments.

**Example: Adding a `geometric_mean` operator**
```python
# In discover/features.py
CUSTOM_BINARY_OP_DEFS = {
    #. other operators
    'geometric_mean': {
        'func': lambda f1, f2: np.sqrt(f1 * f2),
        'sym_func': lambda s1, s2: sympy.sqrt(s1 * s2),
        'unit_op': lambda u1, u2: u1**0.5 * u2**0.5,
}}}
'unit_check': None, # Or a check for compatible units
        'domain_check': lambda f1, f2: np.all(f1.values >= 0) and np.all(f2.values >= 0),
        'op_name': "<G>"
    }
}
```

### Adding a New Search Strategy

1.  Open `discover/search.py`.
2.  Create a new function, typically prefixed with `_find_best_models_`. For an example, study `_find_best_models_greedy`.
3.  The new function should look like the following:
    ```python
def _find_best_models_new_strategy(sisso_instance, phi_sis_df, y, D_max, task_type, max_feat_cross_corr, sample_weight, device, torch_device, **kwargs):
    ```
4.  The function ought to return a dictionary with the keys being dimensions (e.g., `1`, `2`, `3`) and values being dictionaries containing the model information for each dimension. The dictionary containing model information ought to contain:
    -   `features`: A list of feature names (strings).
    -   `score`: Training score of the model.
-   `model`: The fitted model object (e.g., scikit-learn model).
    -   `coef`: The coefficients of the model (where it is applicable).
    -   `sym_features`: List of `sympy` expressions for features.
    -   `is_parametric`: Boolean (`False` for standard linear models).
5.  Open `discover/models.py`, locate the `fit` method within the `DiscoverBase` class, and include your new search strategy within the `if/elif` block that invokes the search functions.
    ```python
    # In discover/models.py in the fit method
    elif self.search_strategy == 'new_strategy':
        search_results = _find_best_models_new_strategy(**search_args)
    ```

### Coding Style and Conventions

-   **PEP 8:** Adhere to the PEP 8 Python coding style guide.
-   **Docstrings:** Provide brief, clear docstrings for modules, classes, and functions that explain their function, arguments, and return type.
-   **Type Hinting:** We suggest using type hints to make code more readable and maintainable.
-   **Readability:** Make the code readable and understandable. Add comments wherever the logic becomes too complicated.

### Making Changes

1.  **Fork the repository** on GitHub.
2.  **Create a new branch** for your feature or bugfix: `git checkout -b feature/my-new-operator`.
3.  **Make your changes** and commit them with a good commit message.
4.  **Push your branch** to your fork: `git push origin feature/my-new-operator`.
5.  **Submit a Pull Request** to the master repository, explaining the changes you did.
