import pandas as pd
import numpy as np

def feature_engineering(df):
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['weekday'] = df['date'].dt.weekday
    df['day'] = df['date'].dt.day
    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
    return df

def preprocess_data(train_file, test_file):
    # Load the data
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    # Handle missing values
    train.loc[:, 'num_sold'] = train['num_sold'].fillna(train['num_sold'].median())

    # Feature Engineering
    train = feature_engineering(train)
    test = feature_engineering(test)

    # One-hot encoding categorical columns
    train = pd.get_dummies(train, columns=['country', 'store', 'product'], drop_first=True)
    test = pd.get_dummies(test, columns=['country', 'store', 'product'], drop_first=True)

    # Drop 'id' and 'date' columns
    train.drop(['id', 'date'], axis=1, inplace=True)
    test.drop(['id', 'date'], axis=1, inplace=True)

    # Align the columns of test and train
    test = test.reindex(columns=train.columns, fill_value=0)

    # Log Transformation of Target Variable
    train['num_sold'] = np.log1p(train['num_sold'])

    return train, test
