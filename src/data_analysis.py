import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler

# Load the datasets
fraud_data = pd.read_csv('../data/raw/fraud_data_with_country.csv')
credit_card_data = pd.read_csv('../data/raw/creditcard.csv')
ip_address_data = pd.read_csv('../data/raw/IpAddress_to_Country.csv')

# Handle missing values
fraud_data.dropna(inplace=True)
credit_card_data.fillna(credit_card_data.mean(), inplace=True)
ip_address_data.dropna(inplace=True)

# Remove duplicates
fraud_data.drop_duplicates(inplace=True)
credit_card_data.drop_duplicates(inplace=True)
ip_address_data.drop_duplicates(inplace=True)

# Encode categorical features selectively
# Example: One-hot encode categorical columns in fraud_data
categorical_columns_fraud = fraud_data.select_dtypes(include=['object']).columns
for col in categorical_columns_fraud:
    if len(fraud_data[col].unique()) > 100:  # Example threshold for reducing cardinality
        fraud_data[col] = fraud_data[col].apply(lambda x: x if x in fraud_data[col].value_counts().index[:100] else 'other')
fraud_data = pd.get_dummies(fraud_data, columns=categorical_columns_fraud, drop_first=True)

# Example: One-hot encode categorical columns in credit_card_data
categorical_columns_credit = credit_card_data.select_dtypes(include=['object']).columns
credit_card_data = pd.get_dummies(credit_card_data, columns=categorical_columns_credit, drop_first=True)

# Example: One-hot encode categorical columns in ip_address_data
categorical_columns_ip = ip_address_data.select_dtypes(include=['object']).columns
ip_address_data = pd.get_dummies(ip_address_data, columns=categorical_columns_ip, drop_first=True)

# Normalize and scale numerical data
scaler_fraud = RobustScaler()
fraud_data['purchase_value_normalized'] = scaler_fraud.fit_transform(fraud_data[['purchase_value']])

scaler_credit = StandardScaler()
credit_card_data['Amount_scaled'] = scaler_credit.fit_transform(credit_card_data[['Amount']])

# Save the processed data
fraud_data.to_csv('../data/processed/processed_fraud_data.csv', index=False)
credit_card_data.to_csv('../data/processed/processed_credit_card_data.csv', index=False)
ip_address_data.to_csv('../data/processed/processed_ip_address_data.csv', index=False)

# Display the first few rows of the updated datasets
print("\nFirst few rows of processed fraud_data:")
print(fraud_data.head())

print("\nFirst few rows of processed credit_card_data:")
print(credit_card_data.head())

print("\nFirst few rows of processed ip_address_data:")
print(ip_address_data.head())