from sklearn.preprocessing import LabelEncoder

def preprocess_data(train_df, test_df, drop_columns):
    X_train = train_df.drop(columns=drop_columns)
    X_test = test_df.drop(columns=drop_columns)
    
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["Activity"])
    y_test = label_encoder.transform(test_df["Activity"])
    
    return X_train, y_train, X_test, y_test
