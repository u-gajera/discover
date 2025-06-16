# pySISSO: A Python Implementation of SISSO

This project is a refactored, modular version of the original `pySISSO.py` script. It provides a flexible framework for feature engineering and sparse model discovery using Sure Independence Screening and Sparsifying Operator (SISSO) variants.

## Project Structure

- `run_sisso.py`: The main entry point for running the tool from the command line or executing the built-in demo.
- `src/pysisso/`: The core source code, structured as a Python package.
  - `models.py`: Contains the main `SISSOBase` class and its Scikit-learn API wrappers (`SISSORegressor`, `SISSOClassifier`).
  - `features.py`: Handles all aspects of feature generation, including the unit-aware `UFeature` class.
  - `search.py`: Implements the different model search strategies (`greedy`, `brute_force`, `sisso++`).
  - `scoring.py`: Contains functions for Sure Independence Screening (SIS), model evaluation (scoring), cross-validation, and GPU-accelerated kernels.
  - `utils.py`: Provides helper functions for plotting, saving results, and formula printing.
  - `constants.py`: Defines global constants used across the package.
- `requirements.txt`: A list of all Python dependencies.
- `demo_*.csv/json`: Data and configuration files for the built-in demo (these are created when you run the demo).

## How to Run

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    For optional GPU support, install `cupy` or `torch` according to their official documentation.

2.  **Run the Demo:**
    To see the tool in action, simply run the main script without any arguments. It will execute a pre-configured demo using the `sisso++` search strategy.

    ```bash
    python run_sisso.py
    ```
    This will create a `demo_sissopp_output` directory with all the results.

3.  **Run from Command Line with your own data:**
    To run a SISSO calculation, you need a JSON configuration file. Use the provided `run_sisso.py` script.

    ```bash
    python run_sisso.py /path/to/your/config.json
    ```

## Using in your own Python scripts

Because `pysisso` is now a package, you can easily import it into your own scripts or Jupyter notebooks:

```python
import pandas as pd
import numpy as np
import os
import sys

# Add the project's src directory to the Python path
# This is necessary so Python can find the 'pysisso' package
sys.path.append(os.path.join(os.getcwd(), 'src'))

from pysisso import SISSORegressor

# Create some dummy data
X = pd.DataFrame(np.random.rand(50, 4), columns=['f1', 'f2', 'f3', 'f4'])
y = 2.5 * X['f1'] / X['f2'] + 0.5

# Configure and run SISSO
sisso = SISSORegressor(
    max_D=2,
    opset=['add', 'sub', 'mul', 'div'],
    sis_sizes=[20],
    search_strategy='greedy'
)

sisso.fit(X, y)

# Print results using the built-in summary method
print(sisso.best_model_summary())