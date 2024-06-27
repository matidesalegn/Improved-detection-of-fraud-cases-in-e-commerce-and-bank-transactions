# serve_model.py

from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the fraud detection model
model_path = '../notebooks/models/fraud_data/random_forest_model.joblib'
model = joblib.load(model_path)

@app.route('/')
def home():
    return "Fraud Detection Model API"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json(force=True)
        # Convert data to DataFrame
        input_data = pd.DataFrame(data)
        # Ensure the data has the correct columns
        expected_columns = [...]  # Add the expected columns based on your model
        input_data = input_data.reindex(columns=expected_columns, fill_value=0)
        # Make predictions
        predictions = model.predict(input_data)
        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)