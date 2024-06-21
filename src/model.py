from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

import torch
import torch.nn as nn

def get_models():
    classifiers = {
        "SVM": SVC(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "K-Neighbors": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier()
    }
    return classifiers



class LSTMModel(nn.Module):
    def __init__(self, input_dim, timesteps, n_hidden, n_classes, pv):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, n_hidden, batch_first=True)
        self.dropout = nn.Dropout(pv)
        self.dense = nn.Linear(n_hidden, n_classes)
        
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]  
        x = self.dropout(x)
        x = self.dense(x)
        
        return x