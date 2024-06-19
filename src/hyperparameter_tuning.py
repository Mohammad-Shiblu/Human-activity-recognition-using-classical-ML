import optuna
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def objective_svm(trial, X, y):
    svc_c = trial.suggest_float('svc_c', 1e-5, 1e2, log=True)
    clf = SVC(C=svc_c)
    return train_and_evaluate(clf, X, y)

def objective_logistic(trial, X, y):
    solver = trial.suggest_categorical('solver', ['newton-cg', 'lbfgs', 'liblinear'])
    penalty = 'l2' if solver != 'liblinear' else trial.suggest_categorical('penalty', ['l2', 'l1'])
    C = trial.suggest_float('C', 1e-5, 100, log=True)
    clf = LogisticRegression(solver=solver, penalty=penalty, C=C, max_iter=1000)
    return train_and_evaluate(clf, X, y)

def objective_knn(trial, X, y):
    n_neighbors = trial.suggest_int('n_neighbors', 1, 50)
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    return train_and_evaluate(clf, X, y)

def objective_rf(trial, X, y):
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    max_depth = trial.suggest_int('max_depth', 1, 32)
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    return train_and_evaluate(clf, X, y)

def train_and_evaluate(clf, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    return accuracy

def tune_hyperparameters(X, y, n_trials=10):
    studies = {}
    
    study_svm = optuna.create_study(direction='maximize')
    study_svm.optimize(lambda trial: objective_svm(trial, X, y), n_trials=n_trials)
    studies['SVM'] = study_svm.best_params
    
    study_logistic = optuna.create_study(direction='maximize')
    study_logistic.optimize(lambda trial: objective_logistic(trial, X, y), n_trials=n_trials)
    studies['LogisticRegression'] = study_logistic.best_params
    
    study_knn = optuna.create_study(direction='maximize')
    study_knn.optimize(lambda trial: objective_knn(trial, X, y), n_trials=n_trials)
    studies['KNN'] = study_knn.best_params
    
    study_rf = optuna.create_study(direction='maximize')
    study_rf.optimize(lambda trial: objective_rf(trial, X, y), n_trials=n_trials)
    studies['RandomForest'] = study_rf.best_params
    
    return studies

def train_best_models(X, y, best_params):
    # Train SVM
    clf_svm = SVC(C=best_params['SVM']['svc_c'])
    svm_accuracy = train_and_evaluate(clf_svm, X, y)
    
    # Train Logistic Regression
    solver = best_params['LogisticRegression']['solver']
    penalty = 'l2' if solver != 'liblinear' else best_params['LogisticRegression']['penalty']
    C = best_params['LogisticRegression']['C']
    clf_logistic = LogisticRegression(solver=solver, penalty=penalty, C=C, max_iter=1000)
    logistic_accuracy = train_and_evaluate(clf_logistic, X, y)
    
    # Train KNN
    n_neighbors = best_params['KNN']['n_neighbors']
    clf_knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_accuracy = train_and_evaluate(clf_knn, X, y)
    
    # Train Random Forest
    n_estimators = best_params['RandomForest']['n_estimators']
    max_depth = best_params['RandomForest']['max_depth']
    clf_rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    rf_accuracy = train_and_evaluate(clf_rf, X, y)
    
    return {
        'SVM': svm_accuracy,
        'LogisticRegression': logistic_accuracy,
        'KNN': knn_accuracy,
        'RandomForest': rf_accuracy
    }

# Example usage:
# X, y = load_your_data()  # Load your data here
# best_params = tune_hyperparameters(X, y, n_trials=10)
# accuracies = train_best_models(X, y, best_params)
# print("Accuracies: ", accuracies)