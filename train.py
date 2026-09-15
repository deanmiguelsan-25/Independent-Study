# train.py
from src.config import ModelConfig
from src.data import DataLoader
from src.model import ModelTrainer

def main():
    # 1. Load configuration
    config = ModelConfig()

    # 2. Process data
    loader = DataLoader(config)
    X_train, X_test, y_train, y_test = loader.load_and_split()

    # 3. Train model
    trainer = ModelTrainer(config)
    trained_model = trainer.train(X_train, y_train)

    print("Training pipeline completed successfully.")

if __name__ == "__main__":
    main()