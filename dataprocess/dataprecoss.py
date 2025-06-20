import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# === Step 1: Load Data ===
df = pd.read_csv("your_file.csv", parse_dates=["date"])

# === Step 2: Extract Time Features ===
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
df['day_of_month'] = df['date'].dt.day
df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)

# === Step 3: Transform Target (log) ===
df['sales_log'] = np.log1p(df['sales'])

# === Step 4: Define Feature Groups ===
numerical_features = ['cluster', 'dcoilwtico', 'sample_weight', 'onpromotion',
                      'month', 'day_of_week', 'day_of_month', 'week_of_year']
categorical_features = ['state', 'type']
embedding_features = ['store_nbr', 'family', 'city']
direct_features = ['is_working_day', 'quake_severe', 'quake_moderate', 'payday']

# === Step 5: Define Transformers ===
numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# === Step 6: Combine in ColumnTransformer ===
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ],
    remainder="passthrough"  # keep embedding and direct features
)

# === Step 7: Apply Transformation ===
processed_array = preprocessor.fit_transform(df)

# Get column names after one-hot
encoded_cat_columns = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
column_names = numerical_features + list(encoded_cat_columns) + embedding_features + direct_features

# === Step 8: To DataFrame ===
processed_df = pd.DataFrame(processed_array, columns=column_names)
processed_df['sales_log'] = df['sales_log'].values
processed_df['date'] = df['date']
processed_df['store_nbr'] = df['store_nbr']
processed_df['family'] = df['family']

# === Step 9: Save to CSV or Continue to Sequence Build ===
processed_df.to_csv("processed_lstm_data.csv", index=False)
