import pytest
import pandas as pd
import os
import shutil
from discover.models import DiscoverRegressor

@pytest.fixture
def sample_data():
    """Loading the sample dataset."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, 'sample_dataset.csv')
    df = pd.read_csv(csv_path)
    X = df[['feature1', 'feature2', 'feature3']]
    y = df['property']
    return X, y

def test_full_pipeline(sample_data):
    X, y = sample_data
    workdir = "test_run_output"

    if os.path.exists(workdir):
        shutil.rmtree(workdir)

    config = {
        "workdir": workdir,
        "primary_units": {
            "feature1": "angstrom",
            "feature2": "eV",
            "feature3": "dimensionless"
        },
        "task_type": "regression",
        "max_D": 2,
        "depth": 1,
        "n_sis_select": 5,
        "cv": 3,
        "random_state": 46
    }

    model = DiscoverRegressor(**config)
    model.fit(X, y)

    assert model.best_D_ is not None
    assert model.best_model_ is not None
    assert os.path.exists(os.path.join(workdir, "discover.out"))

    y_pred = model.predict(X)
    assert len(y_pred) == len(y)

    shutil.rmtree(workdir)