import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

# Validation MAPE: 9.85137859645937

# Load the data
train = pd.read_csv('playground-series-s5e1/train.csv')
test = pd.read_csv('playground-series-s5e1/test.csv')
sample_submission = pd.read_csv('playground-series-s5e1/sample_submission.csv')

# Exploratory Data Analysis (EDA)
def EDA():
    print("Train Dataset Info:")
    print(train.info())

    print("\nMissing Values in Train Dataset:")
    print(train.isnull().sum())

    print("\nSummary Statistics for Train Dataset:")
    print(train.describe())

    # Analyze distribution of target variable
    plt.figure(figsize=(10, 6))
    sns.histplot(train['num_sold'], kde=True, bins=30)
    plt.title('Distribution of num_sold')
    plt.xlabel('num_sold')
    plt.ylabel('Frequency')
    plt.show()

    print("\nData for Distribution of num_sold:")
    print(train['num_sold'].describe())

    # Analyze categorical features
    for col in ['country', 'store', 'product']:
        counts = train[col].value_counts()
        print(f"\nDistribution of {col}:")
        print(counts)

        plt.figure(figsize=(8, 5))
        sns.countplot(data=train, x=col, order=counts.index)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.show()

# EDA()

# Handle missing values
train.loc[:, 'num_sold'] = train['num_sold'].fillna(train['num_sold'].median())

# Feature Engineering
def feature_engineering(df):
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['weekday'] = df['date'].dt.weekday
    df['day'] = df['date'].dt.day
    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
    return df

train = feature_engineering(train)
test = feature_engineering(test)

# Ensure the columns in the test set match the train set by reindexing after one-hot encoding
train = pd.get_dummies(train, columns=['country', 'store', 'product'], drop_first=True)
test = pd.get_dummies(test, columns=['country', 'store', 'product'], drop_first=True)

# Drop 'id' and 'date' columns from both train and test datasets
train.drop(['id', 'date'], axis=1, inplace=True)
test.drop(['id', 'date'], axis=1, inplace=True)

# Align the columns of test and train (ensure test has the same columns as train)
# Reindex test to match train columns
test = test.reindex(columns=train.columns, fill_value=0)

# Ensure that 'num_sold' is not present in the test dataset during prediction
test.drop('num_sold', axis=1, inplace=True)

# Check column alignment to confirm they match
print("Train Columns:", train.columns)
print("Test Columns:", test.columns)

# Log Transformation of Target Variable
train['num_sold'] = np.log1p(train['num_sold'])

# Split data
X = train.drop('num_sold', axis=1)
y = train['num_sold']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training with LightGBM
model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

# Train the model with evaluation set (without evals_result)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='l2'  # Optional, can be 'rmse', 'l2', or 'mae'
)

# Retrieve evaluation results from the training process
evals_result = model.evals_result_  # This will automatically store evaluation results during training

# Track best iteration manually based on validation score
best_iteration = np.argmin(evals_result['valid_0']['l2'])  # Find the iteration with the lowest l2 score
print(f"Best iteration: {best_iteration}")

# Evaluate Model using the best iteration
y_val_pred = model.predict(X_val, num_iteration=best_iteration)
mape = np.mean(np.abs((y_val - y_val_pred) / y_val)) * 100
print(f"Validation MAPE: {mape:.4f}%")

# Add a dummy column 'num_sold' to the test set to match training columns
test['num_sold'] = 0  # This is a dummy column, as num_sold is not present in the test data

# Align the columns between train and test
test = test.reindex(columns=train.columns, fill_value=0)

# Now drop 'num_sold' from the test set before predictions
test.drop('num_sold', axis=1, inplace=True)

# Predictions on Test Set
test_predictions = model.predict(test)
sample_submission['num_sold'] = np.expm1(test_predictions)
sample_submission.to_csv('submission1.csv', index=False)

print("Submission saved as 'submission1.csv'")