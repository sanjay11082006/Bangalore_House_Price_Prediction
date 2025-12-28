import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor

print("=" * 60)
print("FINAL MODEL TRAINING: RANDOM FOREST")
print("=" * 60)

# 1. LOAD DATA
print("\nLoading training data...")
df_train = pd.read_csv('cleaned_train.csv')

# Define features
features = ['location', 'bhk', 'bath', 'balcony', 'total_sqft_numeric']
X_train = df_train[features].copy()
y_train = df_train['price'].copy()

# One-hot encode locations
print("Preprocessing features...")
X_train = pd.get_dummies(X_train, columns=['location'], drop_first=True)

# 2. TRAIN MODEL (Random Forest)
print("Training Random Forest model (this might take a few seconds)...")
# n_estimators=100 means we build 100 decision trees
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
print("Model trained successfully!")

# 3. SAVE MODEL
print("Saving model to 'house_price_model_rf.pkl'...")
model_data = {
    'model': rf_model,
    'columns': X_train.columns.tolist()  # Save column names to match them later
}
joblib.dump(model_data, 'house_price_model_rf.pkl')
print(" Model saved.")

# 4. PREDICT ON TEST DATA
print("\nProcessing Test.csv for final predictions...")
df_test = pd.read_csv('Test.csv')

# Clean Test Data (Handle Missing Values)
df_test['bath'] = df_test['bath'].fillna(df_test['bath'].median())
df_test['balcony'] = df_test['balcony'].fillna(df_test['balcony'].median())
df_test['bhk'] = df_test['size'].str.extract(r'(\d+)').astype(float).fillna(2)

def convert_sqft(x):
    try:
        if '-' in str(x):
            parts = str(x).split('-')
            return (float(parts[0]) + float(parts[1])) / 2
        return float(x)
    except:
        return 1000 # Default if data is bad

df_test['total_sqft_numeric'] = df_test['total_sqft'].apply(convert_sqft)

# Encode Test Data
X_test = df_test[features].copy()
X_test = pd.get_dummies(X_test, columns=['location'], drop_first=True)

# Align Columns (CRITICAL STEP)
# Add missing columns from train set to test set with 0
for col in X_train.columns:
    if col not in X_test.columns:
        X_test[col] = 0
# Remove any extra columns in test not in train
X_test = X_test[X_train.columns]

# Predict
print("Generating predictions...")
predictions = rf_model.predict(X_test)
df_test['predicted_price'] = predictions

# Save Results
df_test.to_csv('test_predictions.csv', index=False)
print("Final predictions saved to 'test_predictions.csv'")
print("Project Complete.")