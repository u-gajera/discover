Of course. Here is the complete `README.md` file, fully formatted in Markdown.

````markdown
# pySISSO: A Python Implementation of Sure Independence Screening and Sparsifying Operator

**pySISSO** is a modern, Python-native implementation of the Sure Independence Screening and Sparsifying Operator (SISSO) method, a powerful machine learning framework for discovering predictive, interpretable models and descriptors from large feature spaces. This implementation is designed for ease of use, extensibility, and performance, incorporating modern best practices and algorithmic enhancements.

This tool is ideal for researchers and data scientists in materials science, chemistry, and other scientific domains who need to find physically meaningful relationships in their data.

***

## Features

* **Iterative Feature Generation:** Efficiently builds a high-quality feature space by iteratively generating and screening features, avoiding the combinatorial explosion of traditional methods.
* **Multiple Search Strategies:**
    * **Greedy Search:** A fast, iterative approach.
    * **Brute-Force:** Exhaustively finds the best model for low-dimensional problems.
    * **SISSO++:** A highly efficient, breadth-first search using QR decomposition.
    * **Orthogonal Matching Pursuit (OMP):** A robust greedy method for regression.
    * **MIQP (Gurobi):** Provably finds the optimal L0-norm model (requires Gurobi license).
* **Diverse Machine Learning Tasks:**
    * **Regression:** Linear (Ridge), Robust (Huber), and Multi-Task models.
    * **Classification:** Logistic Regression, Support Vector Machines (SVM), and Convex Hull-based classifiers.
* **GPU Acceleration:** Leverages NVIDIA (CUDA) and Apple Silicon (MPS) GPUs for significant speedups in feature screening and model fitting.
* **Advanced Analytics & Visualization:** Includes a suite of tools for model interpretation, such as feature importance, partial dependence plots, and a dedicated script for generating publication-quality parity plots.
* **Unit-Aware Feature Engineering:** Uses the `pint` library to prevent the creation of physically nonsensical features by enforcing unit consistency.

***

## Installation

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    A `requirements.txt` file would be ideal, but you can install the core packages manually:

    ```bash
    pip install pandas numpy sympy scikit-learn matplotlib seaborn joblib
    ```

4.  **Optional Installations:**
    * **For GPU Acceleration (NVIDIA):**
        ```bash
        # Replace XX with your CUDA version (e.g., 118 for CUDA 11.8)
        pip install cupy-cudaXX
        ```
    * **For MIQP Search:**
        ```bash
        pip install gurobipy
        ```
        *Note: Gurobi is a commercial solver and requires a license.*
    * **For Unit-Awareness:**
        ```bash
        pip install pint
        ```

***

## Quick Start: Running an Analysis

Running an analysis is straightforward. You modify a single JSON configuration file and then execute the main script.

### Step 1: Prepare Your Data

Ensure your data is in a single **CSV file**. The file should contain all primary features and the target property you want to predict in distinct columns.

### Step 2: Edit the Configuration File

Open `config.json`. This file controls every aspect of the SISSO run. For a quick start, you only need to change these key parameters:

* `"data_file"`: Set this to the path of your CSV data file (e.g., `"my_data.csv"`).
* `"property_key"`: The exact name of the column in your CSV that contains the target property (e.g., `"Target_U (eV)"`).
* `"workdir"`: The name of the directory where all results will be saved (e.g., `"my_first_run"`).
* `"task_type"`: The type of machine learning task. Common choices are:
    * `"regression"` (for predicting a continuous value).
    * `"classification_logreg"` (for predicting categories).
* `"max_D"`: The maximum dimension (number of features) for the final models you want to search for. A good starting point is `3`.

Here is a minimal example of `config.json`:
```json
{
  "data_file": "SISSO_Sample_Dataset.csv",
  "property_key": "Target_U (eV)",
  "workdir": "sisso_output",
  "task_type": "regression",
  "max_D": 3,
  "op_rules": [
    {"op": "add"}, {"op": "sub"}, {"op": "mul"}, {"op": "div"},
    {"op": "sqrt"}, {"op": "sq"}
  ],
  "search_strategy": "greedy",
  "selection_method": "cv",
  "cv": 5
}
````

### Step 3: Run the Analysis

Execute `run_sisso.py` from your terminal, passing the configuration file as the argument:

```bash
python run_sisso.py config.json
```

The script will start, display progress, and save all outputs—including models, plots, and summary reports—into the directory specified by `"workdir"`. The final, best model will be printed to the console.

-----

## Advanced Usage and Analytics

### Customizing the `config.json`

The `config.json` file offers deep control over the analysis. Here are some key sections to explore for advanced use:

  * **Feature-Space Construction (`op_rules`, `depth`):**

      * `"depth"`: Controls the recursion depth for creating new features. Higher values create more complex features but increase computation time.
      * `"op_rules"`: A list of mathematical operators to use. You can prune the list to enforce physical constraints or add advanced operators.

  * **Search and Selection (`search_strategy`, `selection_method`):**

      * `"search_strategy"`: Choose from `"greedy"`, `"sisso++"`, `"omp"`, or `"brute_force"` to balance speed and accuracy.
      * `"selection_method"`: How to pick the best model dimension. Use `"cv"` (cross-validation), `"bootstrap"`, `"aic"`, or `"bic"`.

  * **Model and Validation (`loss`, `alpha`, `cv`):**

      * `"loss"`: For regression, choose between `"l2"` (standard Ridge) or `"huber"` (robust to outliers).
      * `"alpha"`: The regularization strength for the linear model.
      * `"cv"`: The number of folds for cross-validation. Use `-1` for Leave-One-Out CV.

### Plotting and Interpreting Results with `plot_results.py`

After a run is complete, you can use the `plot_results.py` utility to generate publication-quality parity plots and other analyses.

**1. Plot the Best Model for All Dimensions:**
This creates a grid of parity plots, one for the best-found model at each dimension.

```bash
python plot_results.py <workdir> <data_file> --property-key "<your_target_name>"
```

  * **`<workdir>`**: The output directory from your `run_sisso.py` run.
  * **`<data_file>`**: The original data file used for the run.
  * **`<your_target_name>`**: The name of your target property column.

**Example:**

```bash
python plot_results.py sisso_output SISSO_Sample_Dataset.csv --property-key "Target_U (eV)"
```

This saves `parity_best_allD.png` in the `sisso_output` directory.

**2. Plot a Specific SISSO Model:**
Generate a parity plot for a specific model dimension (`-D` or `--dimension`).

```bash
python plot_results.py <workdir> <data_file> --property-key "<your_target_name>" --mode sisso -D 2
```

This command will generate a plot for the 2-dimensional model and save it as `parity_sisso_D2.png`.

**3. Plot Top SIS Candidates:**
Analyze the performance of the best individual features (1D models) found by Sure Independence Screening.

```bash
python plot_results.py <workdir> <data_file> --property-key "<your_target_name>" --mode sis --top 6
```

This command plots the top 6 features and saves the figure as `parity_sis_top6.png`.

-----

## Code Structure

The project is organized into several modules, each with a specific responsibility:

  * `run_sisso.py`: The main command-line interface to drive the analysis.
  * `config.json`: The central configuration file for setting up a run.
  * `pysisso/`: The main package directory.
      * `__init__.py`: Initializes the package and exports the main classes.
      * `models.py`: Contains the primary user-facing `SISSO` classes that orchestrate the workflow.
      * `features.py`: Handles the generation of the feature space from primary features and mathematical operators.
      * `search.py`: Implements the different search strategies (Greedy, SISSO++, OMP, etc.).
      * `scoring.py`: Contains functions for model evaluation, cross-validation, and Sure Independence Screening (SIS).
      * `utils.py`: Provides helper functions for plotting, saving results, and formatting formulas.
      * `constants.py`: Defines global constants used throughout the package.
  * `plot_results.py`: A standalone utility for visualizing the results after a run is complete.

This modular structure makes the code easier to understand, maintain, and extend.

```
```