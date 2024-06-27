# Improved Detection of Fraud Cases in E-commerce and Bank Transactions

## Project Overview

This project aims to improve the detection of fraudulent activities in e-commerce and bank transactions by leveraging data preprocessing, exploratory data analysis (EDA), feature engineering techniques, model building, evaluation, explainability, and deployment. The project integrates data from various sources, including transaction data and geolocation data based on IP addresses, to enhance fraud detection models.

## Table of Contents

- [Project Overview](#project-overview)
- [Table of Contents](#table-of-contents)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preprocessing and Analysis](#data-preprocessing-and-analysis)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Geolocation Analysis](#geolocation-analysis)
- [Model Building and Training](#model-building-and-training)
- [Model Explainability](#model-explainability)
- [Model Deployment and API Development](#model-deployment-and-api-development)
- [Contributors](#contributors)
- [License](#license)

## Project Structure

```
fraud_detection/
├── .github/
│   ├── workflows/
│   │   └── ci.yml
├── data/
│   ├── raw/
│   │   ├── Fraud_Data.csv
│   │   ├── fraud_data_with_country.csv
│   │   ├── creditcard.csv
│   │   └── IpAddress_to_Country.csv
│   ├── processed/
│   │   ├── processed_credit_card_data.csv
│   │   └── processed_fraud_data_with_country.csv
├── mlruns/
├── notebooks/
│   ├── 01_data_analysis.ipynb
│   ├── model explainability.ipynb
│   └── model_building_and_training.ipynb
├── models/
│   ├── random_forest_model.joblib
├── src/
│   ├── __init__.py
│   ├── data_merging.py
│   ├── data_preprocessing.py
├── api/
│   ├── templates/index.html
│   ├── serve_model.py
│   ├── requirements.txt
│   ├── Dockerfile
├── tests/
│   ├── test_data_preprocessing.py
│   ├── test_data_merging.py
│   ├── test_model_training.py
│   ├── test_model_evaluation.py
│   ├── test_model_explainability.py
│   └── (other test files)
├── .gitignore
├── README.md
└── setup.py
```

## Requirements

- Python 3.8 or higher
- Required Python libraries:
  - pandas
  - matplotlib
  - seaborn
  - ipaddress
  - scikit-learn
  - imbalanced-learn
  - mlflow
  - shap
  - lime
  - flask
  - joblib

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/fraud-detection.git
   ```
2. Navigate to the project directory:
   ```sh
   cd fraud-detection
   ```
3. Create a virtual environment:
   ```sh
   python -m venv venv
   ```
4. Activate the virtual environment:
   - On Windows:
     ```sh
     .\venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```sh
     source venv/bin/activate
     ```
5. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

1. Place your raw data files (`Fraud_Data.csv`, `creditcard.csv`, `IpAddress_to_Country.csv`) in the `data/raw` directory.
2. Run the main notebook or script to preprocess data, perform EDA, merge datasets, and build models:
   ```sh
   jupyter notebook notebooks/01_data_analysis.ipynb
   ```

## Data Preprocessing and Analysis

### Handling Missing Values

- Drop rows with missing values from `Fraud_Data.csv`.
- Fill missing values in `creditcard.csv` with the mean of the respective columns.

### Data Cleaning

- Remove duplicate entries from both datasets.
- Correct data types for numerical and categorical columns.

### Normalization and Scaling

- Normalize and scale `purchase_value` in `Fraud_Data.csv`.
- Normalize and scale `Amount` in `creditcard.csv`.

### Save Processed Data

- Save the processed data into the `data/processed` directory.

## Exploratory Data Analysis (EDA)

### Univariate Analysis

- Histogram of `purchase_value` in `Fraud_Data.csv`.

### Bivariate Analysis

- Boxplot of `purchase_value` by `class` in `Fraud_Data.csv`.

## Geolocation Analysis

### Convert IP Addresses to Integer

- Convert IP addresses in `Fraud_Data.csv` and `IpAddress_to_Country.csv` to integer format for efficient merging.

### Merge Datasets

- Merge `Fraud_Data.csv` with `IpAddress_to_Country.csv` based on IP address ranges to enrich fraud data with geolocation information.

### Country Distribution

- Visualize the distribution of countries in `IpAddress_to_Country.csv`.

## Model Building and Training

### Data Preparation

1. Load Data:

   ```python
   fraud_data = pd.read_csv('processed_fraud_data_with_country.csv')
   credit_card_data = pd.read_csv('processed_credit_card_data.csv')
   ```

2. DateTime Parsing:

   ```python
   fraud_data['signup_time'] = pd.to_datetime(fraud_data['signup_time'])
   fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time'])
   ```

3. Feature Engineering:

   ```python
   # Extracting features from datetime columns
   fraud_data['signup_hour'] = fraud_data['signup_time'].dt.hour
   fraud_data['signup_day'] = fraud_data['signup_time'].dt.day
   fraud_data['signup_month'] = fraud_data['signup_time'].dt.month
   fraud_data['signup_year'] = fraud_data['signup_time'].dt.year
   fraud_data['purchase_hour'] = fraud_data['purchase_time'].dt.hour
   fraud_data['purchase_day'] = fraud_data['purchase_time'].dt.day
   fraud_data['purchase_month'] = fraud_data['purchase_time'].dt.month
   fraud_data['purchase_year'] = fraud_data['purchase_time'].dt.year
   ```

4. Drop Original Datetime Columns:

   ```python
   fraud_data = fraud_data.drop(columns=['signup_time', 'purchase_time'])
   ```

5. Encode Categorical Variables:

   ```python
   categorical_columns = fraud_data.select_dtypes(include=['object']).columns.tolist()
   fraud_data = pd.get_dummies(fraud_data, columns=categorical_columns, drop_first=True)
   ```

6. Separate Features and Targets:

   ```python
   X_fraud = fraud_data.drop(columns=['class'])
   y_fraud = fraud_data['class']
   X_credit = credit_card_data.drop(columns=['Class'])
   y_credit = credit_card_data['Class']
   ```

7. Handle Missing Values:

   ```python
   from sklearn.impute import SimpleImputer
   imputer = SimpleImputer(strategy='mean')
   X_fraud = imputer.fit_transform(X_fraud)
   X_credit = imputer.fit_transform(X_credit)
   y_fraud = y_fraud.fillna(y_fraud.mode()[0])
   y_credit = y_credit.fillna(y_credit.mode()[0])
   ```

8. Train-Test Split:
   ```python
   from sklearn.model_selection import train_test_split
   X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud = train_test_split(X_fraud, y_fraud, test_size=0.2, random_state=42)
   X_train_credit, X_test_credit, y_train_credit, y_test_credit = train_test_split(X_credit, y_credit, test_size=0.2, random_state=42)
   ```

### Model Selection

We experimented with various models including Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and Multi-layer Perceptron (MLP).

### Model Training and Evaluation

1. Initialize Models:

   ```python
   from sklearn.linear_model import LogisticRegression
   from sklearn.tree import DecisionTreeClassifier
   from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
   from sklearn.neural_network import MLPClassifier
   import mlflow
   import mlflow.sklearn
   models = {
       'Logistic Regression': LogisticRegression(max_iter=1000),
       'Decision Tree': DecisionTreeClassifier(),
       'Random Forest': RandomForestClassifier(),
       'Gradient Boosting': GradientBoostingClassifier(),
       'MLP': MLPClassifier(max_iter=1000)
   }
   ```

2. Train and Evaluate Models:

   ```python
   from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, roc_auc_score
   from imblearn.over_sampling import SMOTE

   # Handling imbalanced data
   smote = SMOTE(random_state=42)
   X_train_fraud, y_train_fraud = smote.fit_resample(X_train_fraud, y_train_fraud)

   for name, model in models.items():
       with mlflow.start_run(run_name=f'{name} on Fraud Data'):
           model.fit(X_train_fraud, y_train_f
   ```

raud)
y_pred = model.predict(X_test_fraud)
y_pred_proba = model.predict_proba(X_test_fraud)[:, 1] if hasattr(model, "predict_proba") else None

           accuracy = accuracy_score(y_test_fraud, y_pred)
           precision = precision_score(y_test_fraud, y_pred, zero_division=0)
           recall = recall_score(y_test_fraud, y_pred, zero_division=0)
           f1 = f1_score(y_test_fraud, y_pred, zero_division=0)
           roc_auc = roc_auc_score(y_test_fraud, y_pred_proba) if y_pred_proba is not None else None

           mlflow.log_params({
               'model': name,
               'dataset': 'Fraud Data',
               'test_size': 0.2,
               'random_state': 42
           })

           mlflow.log_metrics({
               'accuracy': accuracy,
               'precision': precision,
               'recall': recall,
               'f1_score': f1,
               'roc_auc': roc_auc
           })

           mlflow.sklearn.log_model(model, f'{name}_model')

           print(f"Model: {name}")
           print(f"Accuracy: {accuracy}")
           print(classification_report(y_test_fraud, y_pred))
           print("="*60)

````

3. Running the MLflow UI:
```sh
mlflow ui
````

### Model Performance Results

#### Fraud Data:

- **Logistic Regression**:

  - Accuracy: 0.91
  - Precision: 0.00 (for class 1)
  - Recall: 0.00 (for class 1)
  - F1-Score: 0.00 (for class 1)

- **Decision Tree**:

  - Accuracy: 0.92
  - Precision: 0.55 (for class 1)
  - Recall: 0.60 (for class 1)
  - F1-Score: 0.57 (for class 1)

- **Random Forest**:

  - Accuracy: 0.96
  - Precision: 1.00 (for class 1)
  - Recall: 0.54 (for class 1)
  - F1-Score: 0.70 (for class 1)

- **Gradient Boosting**:

  - Accuracy: 0.96
  - Precision: 1.00 (for class 1)
  - Recall: 0.54 (for class 1)
  - F1-Score: 0.70 (for class 1)

- **MLP**:
  - Accuracy: 0.72
  - Precision: 0.15 (for class 1)
  - Recall: 0.42 (for class 1)
  - F1-Score: 0.22 (for class 1)

#### Credit Card Data:

- **Logistic Regression**:

  - Accuracy: 0.999
  - Precision: 0.85 (for class 1)
  - Recall: 0.59 (for class 1)
  - F1-Score: 0.70 (for class 1)

- **Decision Tree**:

  - Accuracy: 0.999
  - Precision: 0.65 (for class 1)
  - Recall: 0.71 (for class 1)
  - F1-Score: 0.68 (for class 1)

- **Random Forest**:

  - Accuracy: 0.999
  - Precision: 0.99 (for class 1)
  - Recall: 0.73 (for class 1)
  - F1-Score: 0.84 (for class 1)

- **Gradient Boosting**:

  - Accuracy: 0.999
  - Precision: 0.89 (for class 1)
  - Recall: 0.63 (for class 1)
  - F1-Score: 0.74 (for class 1)

- **MLP**:
  - Accuracy: 0.998
  - Precision: 0.33 (for class 1)
  - Recall: 0.01 (for class 1)
  - F1-Score: 0.02 (for class 1)

### Enhancements and Future Work

1. **Handling Imbalanced Data**: Implementing SMOTE significantly improved the performance of the minority class.
2. **Add More Metrics**: Evaluate models on precision, recall, F1-score, and ROC AUC for a comprehensive view of performance.
3. **Experiment with Different Feature Engineering**: Explore additional features like day of the week, interaction terms, and domain-specific features.
4. **Hyperparameter Tuning**: Use GridSearchCV or RandomizedSearchCV to optimize model parameters.
5. **Model for Credit Card Data**: Repeat the same process for the credit card dataset.
6. **Use Pipelines**:
   ```python
   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler
   pipeline = Pipeline([
       ('scaler', StandardScaler()),
       ('model', RandomForestClassifier())
   ])
   ```

## Model Explainability

Model explainability is crucial in machine learning, particularly in sensitive applications like fraud detection and credit card default prediction. Understanding why a model makes a certain prediction helps in building trust, debugging issues, and ensuring regulatory compliance. In this task, we will use SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations) to explain the predictions of our Random Forest models.

### Key Files

- `notebooks/model_explainability.ipynb`

### Using SHAP for Model Explainability

SHAP values provide a unified measure of feature importance, explaining the contribution of each feature to the prediction.

#### Installing SHAP

First, install SHAP using the following command:

```sh
pip install shap
```

#### SHAP Plots

To understand our models, we will create three types of SHAP plots:

- **Summary Plot**: Provides an overview of the most important features.
- **Force Plot**: Visualizes the contribution of features for a single prediction.
- **Dependence Plot**: Shows the relationship between a feature and the model output.

#### Explaining the Fraud Detection Model

```python
import shap
import matplotlib.pyplot as plt
import joblib
import pandas as pd

# Load processed data and model
fraud_data = pd.read_csv('data/processed/processed_fraud_data_with_country.csv')
X_fraud = fraud_data.drop(columns=['class'])
y_fraud = fraud_data['class']
fraud_model = joblib.load('models/random_forest_model.joblib')

# Split the data into training and test sets
from sklearn.model_selection import train_test_split
X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud = train_test_split(X_fraud, y_fraud, test_size=0.2, random_state=42)

# Ensure test data is only numeric for SHAP explainability
X_test_fraud_numeric = X_test_fraud.select_dtypes(include=['number'])

# SHAP explainability for fraud model
explainer_shap_fraud = shap.TreeExplainer(fraud_model)
shap_values_fraud = explainer_shap_fraud.shap_values(X_test_fraud_numeric)

# SHAP summary plot for fraud model
shap.summary_plot(shap_values_fraud, X_test_fraud_numeric)
plt.show()

# SHAP force plot for the first instance in the fraud test set
shap.force_plot(explainer_shap_fraud.expected_value[1], shap_values_fraud[1][0], X_test_fraud_numeric.iloc[0])

# SHAP dependence plot for a specific feature, e.g., 'amount'
shap.dependence_plot('amount', shap_values_fraud[1], X_test_fraud_numeric)
plt.show()
```

#### Explaining the Credit Card Default Prediction Model

```python
# Load processed data and model
credit_card_data = pd.read_csv('data/processed/processed_credit_card_data.csv')
X_credit = credit_card_data.drop(columns=['Class'])
y_credit = credit_card_data['Class']
credit_model = joblib.load('models/random_forest_model.joblib')

# Split the data into training and test sets
X_train_credit, X_test_credit, y_train_credit, y_test_credit = train_test_split(X_credit, y_credit, test_size=0.2, random_state=42)

# Ensure test data is only numeric for SHAP explainability
X_test_credit_numeric = X_test_credit.select_dtypes(include=['number'])

# SHAP explainability for credit card model
explainer_shap_credit = shap.TreeExplainer(credit_model)
shap_values_credit = explainer_shap_credit.shap_values(X_test_credit_numeric)

# SHAP summary plot for credit card model
shap.summary_plot(shap_values_credit, X_test_credit_numeric)
plt.show()

# SHAP force plot for the first instance in the credit card test set
shap.force_plot(explainer_shap_credit.expected_value[1], shap_values_credit[1][0], X_test_credit_numeric.iloc[0])

# SHAP dependence plot for a specific feature, e.g., 'amount'
shap.dependence_plot('amount', shap_values_credit[1], X_test_credit_numeric)
plt.show()
```

### Using LIME for Model Explainability

LIME explains individual predictions by approximating the model locally with an interpretable model.

#### Installing LIME

First, install LIME using the following command:

```sh
pip install lime
```

#### Explaining a Model

with LIME

##### Explaining the Fraud Detection Model

```python
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

# LIME explainability for fraud model
explainer_lime_fraud = lime.lime_tabular.LimeTabularExplainer(X_train_fraud.values, feature_names=X_train_fraud.columns, class_names=['Not Fraud', 'Fraud'], discretize_continuous=True)

# Explain the prediction for the first instance in the fraud test set
i = 0
exp_fraud = explainer_lime_fraud.explain_instance(X_test_fraud_numeric.iloc[i].values, fraud_model.predict_proba, num_features=10)

# Show the explanation in a notebook
exp_fraud.show_in_notebook(show_all=False)

# LIME feature importance plot for fraud model
exp_fraud.as_pyplot_figure()
plt.show()
```

##### Explaining the Credit Card Default Prediction Model

```python
# LIME explainability for credit card model
explainer_lime_credit = lime.lime_tabular.LimeTabularExplainer(X_train_credit.values, feature_names=X_train_credit.columns, class_names=['Class 0', 'Class 1'], discretize_continuous=True)

# Explain the prediction for the first instance in the credit card test set
i = 0
exp_credit = explainer_lime_credit.explain_instance(X_test_credit_numeric.iloc[i].values, credit_model.predict_proba, num_features=10)

# Show the explanation in a notebook
exp_credit.show_in_notebook(show_all=False)

# LIME feature importance plot for credit card model
exp_credit.as_pyplot_figure()
plt.show()
```

### Conclusion

By using SHAP and LIME, we gain insights into our models’ decision-making processes. SHAP provides a global and local understanding of feature importance, while LIME offers a detailed look at individual predictions. This dual approach enhances our ability to trust and refine our models, ensuring they perform well and fairly.

For a detailed walkthrough and code execution, refer to the notebook `notebooks/model_explainability.ipynb`.

## Model Deployment and API Development

1. **Flask API**: Develop a Flask API to serve the model for real-time predictions.
2. **Dockerize the Application**: Create a Dockerfile to containerize the Flask application for deployment.

### Key Files

- `api/serve_model.py`
- `api/requirements.txt`
- `api/Dockerfile`

### Building a Fraud Detection Web Application with Flask and Random Forest

Fraud detection is a crucial task for financial institutions, e-commerce platforms, and various online services. Detecting fraudulent activities early can save significant amounts of money and prevent various forms of abuse. In this section, we’ll walk you through building a web application for fraud detection using Flask, a lightweight Python web framework, and a pre-trained Random Forest model.

### Prerequisites

Before we start, ensure you have the following:

1. Python installed on your system.
2. Flask installed (`pip install flask`).
3. Joblib installed (`pip install joblib`).
4. A pre-trained Random Forest model saved as `random_forest_model.joblib`.

### Step 1: Setting Up Flask

First, we’ll create a simple Flask application to serve our fraud detection model.

**serve_model.py**

```python
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the fraud detection model
model_path = 'models/random_forest_model.joblib'
model = joblib.load(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json(force=True)
        # Convert data to DataFrame
        input_data = pd.DataFrame(data, index=[0])
        # Ensure the data has the correct columns
        expected_columns = ['feature1', 'feature2', 'feature3'] # Replace with your actual column names
        input_data = input_data.reindex(columns=expected_columns, fill_value=0)
        # Make predictions
        predictions = model.predict(input_data)
        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

In this script, we define two routes:

- `/`: Renders the home page.
- `/predict`: Accepts POST requests, processes the input data, and returns fraud detection predictions.

### Step 2: Creating the HTML Interface

Next, we create an HTML file that will serve as the user interface for our web application.

**templates/index.html**

Create a `templates` directory in the same directory as `serve_model.py`, and inside it, create an `index.html` file:

```html
<!DOCTYPE html>
<html>
  <head>
    <title>Fraud Detection</title>
    <script>
      async function predictFraud() {
        // Get form data
        const formData = {
          feature1: document.getElementById("feature1").value,
          feature2: document.getElementById("feature2").value,
          feature3: document.getElementById("feature3").value,
          // Add more features as needed
        };

        // Send data to the server
        const response = await fetch("/predict", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(formData),
        });
        const result = await response.json();
        // Display results
        if (result.error) {
          document.getElementById(
            "result"
          ).innerText = `Error: ${result.error}`;
        } else {
          document.getElementById(
            "result"
          ).innerText = `Prediction: ${result.predictions[0]}`;
        }
      }
    </script>
  </head>
  <body>
    <h1>Fraud Detection</h1>
    <form onsubmit="event.preventDefault(); predictFraud();">
      <label for="feature1">Feature 1:</label>
      <input type="text" id="feature1" name="feature1" /><br />
      <label for="feature2">Feature 2:</label>
      <input type="text" id="feature2" name="feature2" /><br />
      <label for="feature3">Feature 3:</label>
      <input type="text" id="feature3" name="feature3" /><br />
      <!-- Add more fields as needed -->
      <input type="submit" value="Predict" />
    </form>
    <div id="result"></div>
  </body>
</html>
```

This HTML file contains a form where users can input data and a script that sends the input data to the `/predict` endpoint. The prediction result is then displayed on the page.

### Step 3: Running the Application

To run your Flask application, open a terminal and navigate to the directory containing `serve_model.py`. Run the following command:

```sh
python serve_model.py
```

This will start the Flask development server. Open a web browser and go to `http://localhost:5000/`. You should see the form where you can input data and get the fraud detection result.

## Contributors

- [Matiwos Desalegn](https://github.com/matidesalegn)
