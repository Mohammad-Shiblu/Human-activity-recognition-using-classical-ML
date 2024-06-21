import sys
import optuna
import os
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.model import get_models
from src.train_evaluate import train_and_evaluate
from src.hyperparameter_tuning import tune_hyperparameters, train_best_models
from src.model import LSTMModel

def classical_ML():
    data_dir = "data/raw/"
    results_file = "output/results.pkl"

    train_df, test_df = load_data(data_dir, "train.csv", "test.csv")
    
    X_train, y_train, X_test, y_test = preprocess_data(train_df, test_df, ["subject", "Activity"])
    #classifiers = get_models()

    
    best_params = tune_hyperparameters(X_train, y_train, n_trials=2)
    accuracies = train_best_models(X_train, y_train, best_params)
    print("Accuracies: ", accuracies)

def lstm():
    data_dir = "data/raw/"
    # results_file = "output/results.pkl"

    train_df, test_df = load_data(data_dir, "train.csv", "test.csv")
    
    X_train, y_train, X_test, y_test = preprocess_data(train_df, test_df, ["subject", "Activity"])

    print(type(y_train))
    # X_train_np = X_train.values
    # y_train_np = y_train.values
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).unsqueeze(1)  
    y_train_tensor = torch.tensor(y_train, dtype=torch.long) 
    print(X_train_tensor.shape)
    print(y_train_tensor.shape)
    input_dim = 561  
    timesteps = 64  
    n_hidden = 50   
    n_classes = 6  
    pv = 0.5        
    learning_rate = 0.001
    num_epochs = 2
    batch_size = 32

    model = LSTMModel(input_dim, timesteps, n_hidden, n_classes, pv)
    
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = LSTMModel(input_dim, timesteps, n_hidden, n_classes, pv)
    criterion = nn.CrossEntropyLoss()  
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

   


    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if (i + 1) % 10 == 0:  # Print every 10 batches
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {running_loss / 10:.4f}')
                running_loss = 0.0

    print('Training finished.')



    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).unsqueeze(1)  
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)  

    
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss /= len(test_loader)
    accuracy = correct / total
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.4f}')  
    
if __name__ == "__main__":
    lstm()
