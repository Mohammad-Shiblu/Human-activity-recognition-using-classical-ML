from sklearn.metrics import accuracy_score
import pickle

def train_and_evaluate(classifiers, X_train, y_train, X_test, y_test):
    results_file = "output/results.pkl"
    results = {}
    
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
    

    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    return results