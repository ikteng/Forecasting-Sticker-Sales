import joblib
import numpy as np
import pandas as pd
from preprocessing import preprocess_data
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

def make_predictions(test_file):
    # Preprocess the data
    train, test = preprocess_data('playground-series-s5e1/train.csv', test_file)

    # Align the test columns with the train columns
    train_columns = train.drop('num_sold', axis=1).columns
    test = test[train_columns]  # Ensure the test set has the same columns as the training set

    # Split the training data for validation
    X = train.drop('num_sold', axis=1)
    y = train['num_sold']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load the trained models
    model = joblib.load('lightgbm_model1.pkl')
    # xgb_model = joblib.load('model/xgboost_model.pkl')
    # rf_model = joblib.load('model/random_forest_model.pkl')

    # Make predictions on the validation set
    preds_val = model.predict(X_val)

    # Evaluate using MAPE on validation set
    print(f"MAPE: {mean_absolute_percentage_error(y_val, preds_val):.4f}")

    # Make predictions on the test set
    preds_test = model.predict(test)

    # Prepare the submission file
    sample_submission = pd.read_csv('playground-series-s5e1/sample_submission.csv')
    sample_submission['num_sold'] = np.expm1(preds_test)
    sample_submission.to_csv('lgbm_submission.csv', index=False)

    print("Submission saved as 'lgbm_submission1.csv'")

if __name__ == "__main__":
    make_predictions('playground-series-s5e1/test.csv')
