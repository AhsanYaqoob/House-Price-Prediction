from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load the trained model
model_path = 'House-price-prediction/model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Load the scaler
scaler_path = 'House-price-prediction/scaler.pkl'
with open(scaler_path, 'rb') as file:
    scaler = pickle.load(file)

# Load the train columns
train_columns_path = 'House-price-prediction/train_columns.pkl'
with open(train_columns_path, 'rb') as file:
    train_columns = pickle.load(file)

# Function to preprocess input data
def preprocess_input(data):
    # Create a DataFrame from input data
    input_df = pd.DataFrame([data])

    # Preprocess the input data similarly as done during training
    input_df['BHK'] = pd.to_numeric(input_df['BHK'], errors='coerce')
    input_df['Size'] = pd.to_numeric(input_df['Size'], errors='coerce')
    input_df['Floor'] = pd.to_numeric(input_df['Floor'], errors='coerce')
    input_df['Bathroom'] = pd.to_numeric(input_df['Bathroom'], errors='coerce')

    numerical_columns = ['BHK', 'Size', 'Floor', 'Bathroom']

    # Scale numerical columns
    input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])

    # Encode categorical variables
    input_df = pd.get_dummies(input_df, drop_first=True)

    # Ensure all columns used during training are present
    for col in train_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[train_columns]

    return input_df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get form data including the new field 'AreaLocality'
            data = {
                'BHK': request.form['BHK'],
                'Size': request.form['Size'],
                'Floor': request.form['Floor'],
                'Bathroom': request.form['Bathroom'],
                'Area Type': request.form['Area Type'],
                'City': request.form['City'],
                'Furnishing Status': request.form['Furnishing Status'],
                'Tenant Preferred': request.form['Tenant Preferred'],
                'Point of Contact': request.form['Point of Contact'],
                'Area Locality': request.form['AreaLocality']  # Include the new field
            }

            # Preprocess the input data
            input_df = preprocess_input(data)

            # Make prediction
            prediction = model.predict(input_df)

            return render_template('index.html', prediction=round(prediction[0], 2))
        except KeyError as e:
            return f"Missing form field: {str(e)}", 400
        except ValueError as e:
            return f"Value error: {str(e)}", 400

if __name__ == '__main__':
    # Ensure necessary files are present
    required_files = [model_path, scaler_path, train_columns_path]
    for file in required_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Required file '{file}' not found. Please ensure it exists in the directory.")

    app.run(debug=True)
