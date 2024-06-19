# Improved Detection of Fraud Cases in E-commerce and Bank Transactions

## Project Overview

This project aims to improve the detection of fraudulent activities in e-commerce and bank transactions by leveraging data preprocessing, exploratory data analysis (EDA), and feature engineering techniques. The project integrates data from various sources, including transaction data and geolocation data based on IP addresses, to enhance fraud detection models.

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
- [Contributors](#contributors)
- [License](#license)

## Project Structure

```
.
├── data
│   ├── raw
│   │   ├── Fraud_Data.csv
│   │   ├── creditcard.csv
│   │   └── IpAddress_to_Country.csv
│   ├── processed
│   │   ├── processed_fraud_data.csv
│   │   ├── processed_credit_card_data.csv
│   │   └── processed_merged_data.csv
├── src
│   ├── data
│   │   ├── data_preprocessing.py
│   └── utils
│       └── utils.py
├── notebooks
│   └── main.ipynb
└── README.md
```

## Requirements

- Python 3.8 or higher
- Required Python libraries:
  - pandas
  - matplotlib
  - seaborn
  - ipaddress

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
2. Run the main notebook or script to preprocess data, perform EDA, and merge datasets:

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

- Save the processed data into `data/processed` directory.

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

## Contributors

- [Matiwos Desalegn](https://github.com/matidesalegn) - Project Lead
