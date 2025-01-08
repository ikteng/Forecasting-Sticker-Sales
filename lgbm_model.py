# lgbm_model.py
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from preprocessing import preprocess_data

def train_lgbm_model(train_file, test_file):
    # Preprocess the data
    train, test = preprocess_data(train_file, test_file)

    # Split the data
    X = train.drop('num_sold', axis=1)
    y = train['num_sold']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training LightGBM model...")
    # Train LightGBM model
    lgbm_model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )
    lgbm_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='l2'
        )

    # Save the model to disk
    joblib.dump(lgbm_model, 'lightgbm_model.pkl')
    print("LightGBM model saved")
    return lgbm_model

train_lgbm_model('playground-series-s5e1/train.csv', 'playground-series-s5e1/test.csv')
