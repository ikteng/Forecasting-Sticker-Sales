# Forecasting-Sticker-Sales

Link to Competition: https://www.kaggle.com/competitions/playground-series-s5e1/data

## Overview
The Sticker Sales Forecasting Project aims to predict the number of sticker sales (num_sold) in a variety of stores based on historical data. The project leverages machine learning models, including LightGBM, to forecast future sales. The dataset contains various features, such as date, country, store, product, and num_sold, which are used to build predictive models.

This repository contains the following key components:

Exploratory Data Analysis (EDA): Analysis of the dataset to understand its structure, identify missing values, and visualize key patterns.
Preprocessing: Data cleaning, feature engineering, and preparation for machine learning.
Model Training: Building and training a LightGBM model to predict sticker sales.
Prediction: Generating predictions for the test set and preparing the final submission.

## Project Structure
### EDA.py
This script performs an exploratory data analysis on the training dataset to understand its characteristics. 
It:
* Loads the training and test datasets.
* Handles missing values by identifying columns with null values.
* Provides summary statistics for numerical features and categorical distributions.
* Creates additional features from the date column (e.g., year, month, weekday, etc.).
* Visualizes key patterns, such as sales over time, sales by weekday, country, and product.

Key Visualizations:
* Sales over Time: A line plot showing sales trends across months and years.
* Sales by Weekday: A bar plot showing sales distribution by weekday.
Sales by Country: A bar plot comparing sales across different countries.
* Sales by Product: A bar plot comparing sales across different product types.
* Correlation Matrix: A heatmap showing correlations between numerical features.

Observation:
* Missing Values:
    * The num_sold column has 8,871 missing values.
    * Missingness is present in some categories (e.g., country: Kenya, store: Discount Stickers, and product: Holographic Goose).

* Target variable (num_sold):
    * Highly skewed with a maximum value of 5,939 and a mean of 752.5.
    * There are extreme outliers (e.g., max: 5939).

* Categorical Features:
    * Country: 6 unique values
    * Store: 3 unique values
    * Product: 5 unique values

* Temporal Trends:
    * Yearly total sales show growth over time.
    * Monthly total sales have seasonal variations.
    * Weekdays reveal that weekends (especially Sunday) have higher sales.
    
* Correlations:
    * Weak correlations among numerical features, though year shows a slight correlation with num_sold due to growth trends.

Fixes Needed:
1. Handle Missing Values:
    * For num_sold, consider imputation strategies based on temporal trends (e.g., median sales for the same country, store, or product).
    * Confirm missingness patterns in related columns.

2. Feature Engineering:
    * Create lagged features (e.g., sales_last_month) or rolling averages to capture temporal dependencies.
    * Include interaction terms between categorical features (e.g., store-product).

3. Outlier Treatment:
    * Handle extreme outliers using transformations (log or square root) or clipping based on business understanding.

### Preprocessing.py
This script handles the preprocessing of data, which includes:
* Feature Engineering: Extracting additional features from the date column, such as year, month, weekday, day, and whether the date is a weekend (is_weekend).
* Missing Value Handling: Filling missing values in the num_sold column with the median value of the column.
* One-Hot Encoding: Encoding categorical columns (country, store, product) using one-hot encoding to convert them into numeric features.
* Column Alignment: Ensuring that the training and test sets have the same columns.
* Log Transformation: Applying a log transformation (log1p) on the target variable (num_sold) to make the data more suitable for regression models.

### lgbm_model.py
This script builds and trains a LightGBM regression model:
* Data Loading and Preprocessing: Loads the dataset and applies preprocessing steps.
* Model Training: Trains a LightGBM model with hyperparameters like n_estimators, learning_rate, and max_depth.
* Model Evaluation: Splits the data into training and validation sets to evaluate the model's performance.
* Model Saving: Saves the trained model to disk using joblib.

### prediction.py
This script is used to make predictions using the trained model:
* Preprocessing: Loads and preprocesses the test data using the same preprocessing steps applied to the training data.
* Prediction: Loads the saved LightGBM model and uses it to predict the sales on the test dataset.
* Evaluation: Calculates the Mean Absolute Percentage Error (MAPE) on the validation set.
* Submission: Generates the final predictions for the test set and saves them as a CSV file in the required submission format.

### Data Files:
* train.csv: The training data containing sales data for various products across different stores and countries.
* test.csv: The test data on which the model will make predictions.
* sample_submission.csv: A template for submitting the predicted sales in the required format.