# src/data.py
import pandas as pd
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, config):
        self.config = config

    def load_and_split(self):
        df = pd.read_csv(self.config.data_path)

        # Apply cleaning and feature engineering...
        X = df.drop("target", axis=1)
        y = df["target"]

        return train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state
        )