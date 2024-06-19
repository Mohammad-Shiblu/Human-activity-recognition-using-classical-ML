import numpy as np
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_data
from src.preprocess import preprocess_data

if __name__ == "__main__":
    data_dir = "data/raw/"

    train_df, test_df = load_data(data_dir, "train.csv", "test.csv")
    
    X_train, y_train, X_test, y_test = preprocess_data(train_df, test_df, ["subject", "Activity"])
    
    classifiers = {
        "SVM": SVC(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "K-Neighbors": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier()
    }

    
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)  
        y_pred = clf.predict(X_test)  
        accuracy = accuracy_score(y_test, y_pred)  
        print(f"{name} Accuracy: {accuracy:.4f}")
    
    