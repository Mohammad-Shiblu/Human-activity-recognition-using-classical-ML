import pandas as pd
import os

def load_data(data_dir, train_file_name, test_file_name):
    train_df = pd.read_csv(os.path.join(data_dir, train_file_name))
    test_df = pd.read_csv(os.path.join(data_dir, test_file_name))
    return train_df, test_df