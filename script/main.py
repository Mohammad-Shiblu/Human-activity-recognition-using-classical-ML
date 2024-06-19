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
if __name__ == "__main__":
    data_dir = "data/raw/"
    train_file_name = os.path.join(data_dir, "train.csv")
    test_file_name = os.path.join(data_dir, "test.csv" )
    train_df = pd.read_csv(train_file_name)
    test_df = pd.read_csv(test_file_name)

    X_train = train_df.drop(columns= ["subject", "Activity"])
    X_test = test_df.drop(columns= ["subject", "Activity"])

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["Activity"])
    y_test = label_encoder.transform(test_df["Activity"])
    
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
    
    