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

### Step 2: Prepare Your Data

DISCOVER expects your data in a single **CSV file**.
-   Each row should be an example (e.g., a material or a data point).
-   Each column should be a primary feature or the target property.

For our case, `sample_dataset.csv` contains columns for main features like `R_A` (ionic radius), `ELN_A` (electronegativity), and target property `E_a` (migration barrier).

### Step 3: Prepare the Analysis

The workflow is managed by `config.json`. Let's break down the critical parts for an initial run.

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
python run_discover.py config.json
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