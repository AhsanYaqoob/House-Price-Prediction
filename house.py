import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load dataset
df = pd.read_csv('House_Rent_Dataset.csv')

# Assuming 'Rent' is the target column
target_column = 'Rent'

# Data Cleaning
df = df.dropna()

# Ensure 'Rent' is in the dataframe
if target_column not in df.columns:
    raise KeyError(f"'{target_column}' not found in dataframe columns")

# Numerical columns for feature scaling
numerical_columns = ['BHK', 'Size', 'Floor', 'Bathroom']

# Replace non-numeric values in numerical columns with NaN
df[numerical_columns] = df[numerical_columns].apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values after conversion
df = df.dropna()

# Feature scaling for numerical columns
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Split the data into features (X) and target variable (y)
X = df.drop(target_column, axis=1)
y = df[target_column]

# Model Training
rf = RandomForestRegressor(n_estimators=100, max_depth=None)  # Use default hyperparameters
rf.fit(X, y)

# Save the model
with open('model.pkl', 'wb') as file:
    pickle.dump(rf, file)

# Save the scaler for use during prediction
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

print("Model training and saving completed.")
