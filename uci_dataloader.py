import os
from pathlib import Path

import numpy as np
import pandas as pd

DATASETS = ["concrete_compression"]


def uci_dataset_loader(dataset):
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    path = Path(os.environ["CONCRETE_PATH"])
    table = pd.read_excel(path)
    values = table.to_numpy(dtype=float)
    if values.ndim != 2 or values.shape[1] != 9 or not np.isfinite(values).all():
        raise ValueError("Concrete requires the original nine-column numeric UCI spreadsheet")
    return values[:, :8], values[:, 8]
