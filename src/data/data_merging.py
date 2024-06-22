import pandas as pd
import ipaddress
from intervaltree import Interval, IntervalTree

# Function to convert IP address to integer
def ip_to_int(ip):
    if pd.isna(ip):
        return None
    try:
        return int(ip)
    except ValueError:
        return int(ipaddress.ip_address(ip))

def main():
    # Load the datasets
    fraud_data = pd.read_csv('../data/processed/processed_fraud_data.csv')
    ip_address_data = pd.read_csv('../data/raw/IpAddress_to_Country.csv')

    # Print columns of ip_address_data to inspect
    print(ip_address_data.columns)

    # Handle NaN values in IP address columns and convert IP address ranges to integers
    ip_address_data.dropna(subset=['lower_bound_ip_address', 'upper_bound_ip_address'], inplace=True)
    ip_address_data['lower_bound_ip_address_int'] = ip_address_data['lower_bound_ip_address'].apply(ip_to_int)
    ip_address_data['upper_bound_ip_address_int'] = ip_address_data['upper_bound_ip_address'].apply(ip_to_int)

    # Create an interval tree for IP ranges
    ip_tree = IntervalTree()
    for _, row in ip_address_data.iterrows():
        ip_tree[row['lower_bound_ip_address_int']:row['upper_bound_ip_address_int'] + 1] = row['country']

    # Function to map IP address to country using the interval tree
    def map_ip_to_country(ip_int):
        if ip_int is None:
            return 'Unknown'
        interval = ip_tree[ip_int]
        if interval:
            return interval.pop().data
        else:
            return 'Unknown'

    # Handle NaN values in fraud_data IP addresses and apply function to get country for each IP address
    fraud_data.dropna(subset=['ip_address'], inplace=True)
    fraud_data['ip_address_int'] = fraud_data['ip_address'].apply(ip_to_int)
    fraud_data['country'] = fraud_data['ip_address_int'].apply(map_ip_to_country)

    # Save the processed fraud data with country information
    fraud_data.to_csv('../data/processed/processed_fraud_data_with_country.csv', index=False)

    # Display the first few rows of the updated fraud_data
    print(fraud_data.head())

if __name__ == "__main__":
    main()