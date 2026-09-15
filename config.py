# src/config.py
from dataclasses import dataclass

@dataclass
class ModelConfig:
    data_path: str = "data/raw/dataset.csv"
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 100