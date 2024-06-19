import sys
import os
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.model import get_models
from src.train_evaluate import train_and_evaluate

if __name__ == "__main__":
    data_dir = "data/raw/"
    results_file = "output/results.pkl"

    train_df, test_df = load_data(data_dir, "train.csv", "test.csv")
    
    X_train, y_train, X_test, y_test = preprocess_data(train_df, test_df, ["subject", "Activity"])
    
    classifiers = get_models()
    results = train_and_evaluate(classifiers, X_train, y_train, X_test, y_test)

    with open(results_file, 'rb') as f:
        loaded_results = pickle.load(f)

    for name, accuracy in loaded_results.items():
        print(f"{name} Accuracy: {accuracy:.4f}")

    
    