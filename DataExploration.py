import pandas as pd
import os

def explore_folder(folder): 
    for file in os.listdir(folder):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(folder, file))
            print(f"File: {file}")
            print(df.head())
            print(df.info())
            print(df.describe())
            print(df.columns)
            print(df.shape)
            print(df.dtypes)
            print(df.isnull().sum())
            print(df.isnull().sum().sum())
            print(df.isnull().sum().sum() / df.shape[0] * 100)
            print(df.isnull().sum().sum() / df.shape[0] * 100)

if __name__ == "__main__":
    explore_folder('healthcare_data_25countries/who')