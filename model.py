# src/model.py
from sklearn.ensemble import RandomForestClassifier

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.model = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            random_state=self.config.random_state
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self.model