import sys
import optuna
import os
import pickle
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.model import get_models
from src.train_evaluate import train_and_evaluate
from src.hyperparameter_tuning import tune_hyperparameters, train_best_models

if __name__ == "__main__":
    data_dir = "data/raw/"
    results_file = "output/results.pkl"

    train_df, test_df = load_data(data_dir, "train.csv", "test.csv")
    
    X_train, y_train, X_test, y_test = preprocess_data(train_df, test_df, ["subject", "Activity"])
    #classifiers = get_models()

    
    best_params = tune_hyperparameters(X_train, y_train, n_trials=2)
    accuracies = train_best_models(X_train, y_train, best_params)
    print("Accuracies: ", accuracies)
    
    
    
    