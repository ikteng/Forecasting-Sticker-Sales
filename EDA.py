import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
train = pd.read_csv('playground-series-s5e1/train.csv')
test = pd.read_csv('playground-series-s5e1/test.csv')

# Exploratory Data Analysis (EDA)
def EDA():
    print("Train Dataset Info:")
    print(train.info())
    
    # Missing values
    print("\nMissing Values in Train Dataset:")
    print(train.isnull().sum())

    print("\nSummary Statistics for Train Dataset:")
    print(train.describe())

    # Distribution of num_sold
    print("\nDistribution of num_sold:")
    print(train['num_sold'].describe())

    # Distribution of categorical features
    for col in ['country', 'store', 'product']:
        print(f"\nDistribution of {col}:")
        print(train[col].value_counts())

    # Date Feature Exploration (creating new features based on 'date')
    train['date'] = pd.to_datetime(train['date'])
    train['month'] = train['date'].dt.month
    train['year'] = train['date'].dt.year
    train['weekday'] = train['date'].dt.weekday
    train['day'] = train['date'].dt.day

    # Correlation Matrix for numerical features
    print("\nCorrelation Matrix:")
    numeric_columns = train.select_dtypes(include=[np.number]).columns
    correlation_matrix = train[numeric_columns].corr()
    print(correlation_matrix)

    # Missingness by Category (country, store, product)
    print("\nMissing Values by Category:")
    for col in ['country', 'store', 'product']:
        print(f"\nMissing Values by {col}:")
        print(train.groupby(col)['num_sold'].apply(lambda x: x.isnull().sum()))

# Visualization Functions
def plot_sales_over_time():
    monthly_sales = train.groupby(['year', 'month'])['num_sold'].sum().reset_index()
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_sales['year'].astype(str) + '-' + monthly_sales['month'].astype(str), 
             monthly_sales['num_sold'], marker='o')
    plt.xticks(rotation=90)
    plt.title('Sales Over Time')
    plt.xlabel('Year-Month')
    plt.ylabel('Total Sales')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_sales_by_weekday():
    weekday_sales = train.groupby('weekday')['num_sold'].sum().reset_index()
    plt.figure(figsize=(8, 6))
    sns.barplot(x='weekday', y='num_sold', data=weekday_sales, palette='viridis')
    plt.title('Sales by Weekday')
    plt.xlabel('Weekday')
    plt.ylabel('Total Sales')
    plt.xticks(ticks=np.arange(7), labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    plt.tight_layout()
    plt.show()

def plot_sales_by_country():
    country_sales = train.groupby('country')['num_sold'].sum().reset_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(x='num_sold', y='country', data=country_sales, palette='muted')
    plt.title('Sales by Country')
    plt.xlabel('Total Sales')
    plt.ylabel('Country')
    plt.tight_layout()
    plt.show()

def plot_sales_by_product():
    product_sales = train.groupby('product')['num_sold'].sum().reset_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(x='num_sold', y='product', data=product_sales, palette='cubehelix')
    plt.title('Sales by Product')
    plt.xlabel('Total Sales')
    plt.ylabel('Product')
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix():
    plt.figure(figsize=(12, 8))
    # Select only numeric columns for correlation matrix
    numeric_columns = train.select_dtypes(include=[np.number]).columns
    correlation_matrix = train[numeric_columns].corr()  # Apply corr() only to numeric columns
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()

# Call EDA function
if __name__ == "__main__":
    EDA() # Run EDA
    plot_sales_over_time()   # Plot sales over time
    plot_sales_by_weekday()  # Plot sales by weekday
    plot_sales_by_country()  # Plot sales by country
    plot_sales_by_product()  # Plot sales by product
    plot_correlation_matrix()  # Plot correlation matrix

"""
Output:
Train Dataset Info:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 230130 entries, 0 to 230129
Data columns (total 6 columns):
 #   Column    Non-Null Count   Dtype  
---  ------    --------------   -----  
 0   id        230130 non-null  int64  
 1   date      230130 non-null  object 
 2   country   230130 non-null  object 
 3   store     230130 non-null  object
 4   product   230130 non-null  object
 5   num_sold  221259 non-null  float64
dtypes: float64(1), int64(1), object(4)
memory usage: 10.5+ MB
None

Missing Values in Train Dataset:
id             0
date           0
country        0
store          0
product        0
num_sold    8871
dtype: int64

Summary Statistics for Train Dataset:
                  id       num_sold
count  230130.000000  221259.000000
mean   115064.500000     752.527382
std     66432.953062     690.165445
min         0.000000       5.000000
25%     57532.250000     219.000000
50%    115064.500000     605.000000
75%    172596.750000    1114.000000
max    230129.000000    5939.000000

Distribution of num_sold:
count    221259.000000
mean        752.527382
std         690.165445
min           5.000000
25%         219.000000
50%         605.000000
75%        1114.000000
max        5939.000000
Name: num_sold, dtype: float64

Distribution of country:
country
Canada       38355
Finland      38355
Italy        38355
Kenya        38355
Norway       38355
Singapore    38355
Name: count, dtype: int64

Distribution of store:
store
Discount Stickers       76710
Stickers for Less       76710
Premium Sticker Mart    76710
Name: count, dtype: int64

Distribution of product:
product
Holographic Goose     46026
Kaggle                46026
Kaggle Tiers          46026
Kerneler              46026
Kerneler Dark Mode    46026
Name: count, dtype: int64

Correlation Matrix:
                id  num_sold  ...   weekday       day
id        1.000000 -0.040866  ... -0.000338  0.013951
num_sold -0.040866  1.000000  ...  0.069613  0.001137
month     0.141850 -0.006255  ... -0.003208  0.010327
year      0.989743 -0.040462  ...  0.000098  0.000590
weekday  -0.000338  0.069613  ...  1.000000  0.001751
day       0.013951  0.001137  ...  0.001751  1.000000

[6 rows x 6 columns]

Missing Values by Category:

Missing Values by country:
country
Canada       4246
Finland         0
Italy           0
Kenya        4625
Norway          0
Singapore       0
Name: num_sold, dtype: int64

Missing Values by store:
store
Discount Stickers       5179
Premium Sticker Mart    1026
Stickers for Less       2666
Name: num_sold, dtype: int64

Missing Values by product:
product
Holographic Goose     8806
Kaggle                   0
Kaggle Tiers             0
Kerneler                64
Kerneler Dark Mode       1
Name: num_sold, dtype: int64
d:\Laptop\Github\Forecasting-Sticker-Sales\EDA.py:67: FutureWarning:

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.

  sns.barplot(x='weekday', y='num_sold', data=weekday_sales, palette='viridis')
d:\Laptop\Github\Forecasting-Sticker-Sales\EDA.py:78: FutureWarning:

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `y` variable to `hue` and set `legend=False` for the same effect.

  sns.barplot(x='num_sold', y='country', data=country_sales, palette='muted')
d:\Laptop\Github\Forecasting-Sticker-Sales\EDA.py:88: FutureWarning:

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `y` variable to `hue` and set `legend=False` for the same effect.

  sns.barplot(x='num_sold', y='product', data=product_sales, palette='cubehelix')
"""