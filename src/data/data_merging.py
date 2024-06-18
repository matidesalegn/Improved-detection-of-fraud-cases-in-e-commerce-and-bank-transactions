# src/data/data_merging.py

import pandas as pd
import ipaddress

class DataMerging:
    def __init__(self, fraud_data, ip_data):
        self.fraud_data = fraud_data
        self.ip_data = ip_data

    def is_valid_ip(self, ip):
        try:
            ipaddress.ip_address(ip)
            return True
        except ValueError:
            return False

    def convert_ip_to_int(self):
        print("Columns in fraud_data before conversion:", self.fraud_data.columns)
        print("Columns in ip_data before conversion:", self.ip_data.columns)
        # Filter out invalid IP addresses
        self.fraud_data = self.fraud_data[self.fraud_data['ip_address'].apply(self.is_valid_ip)]
        self.ip_data = self.ip_data[self.ip_data['lower_bound_ip_address'].apply(self.is_valid_ip)]
        self.ip_data = self.ip_data[self.ip_data['upper_bound_ip_address'].apply(self.is_valid_ip)]

        # Convert valid IP addresses to integers
        self.fraud_data['ip_address_int'] = self.fraud_data['ip_address'].apply(lambda x: int(ipaddress.ip_address(x)))
        self.ip_data['lower_bound_ip_address_int'] = self.ip_data['lower_bound_ip_address'].apply(lambda x: int(ipaddress.ip_address(x)))
        self.ip_data['upper_bound_ip_address_int'] = self.ip_data['upper_bound_ip_address'].apply(lambda x: int(ipaddress.ip_address(x)))
        
        return self.fraud_data, self.ip_data
    def merge_datasets(self):
        # Ensure the columns exist
        print("Columns in fraud_data before merging:", self.fraud_data.columns)
        print("Columns in ip_data before merging:", self.ip_data.columns)

        # Merge logic
        merged_data = pd.merge_asof(self.fraud_data.sort_values('ip_address_int'), 
                                    self.ip_data[['lower_bound_ip_address_int', 'upper_bound_ip_address_int', 'country']],
                                    left_on='ip_address_int', 
                                    right_on='lower_bound_ip_address_int', 
                                    direction='backward')
        return merged_data