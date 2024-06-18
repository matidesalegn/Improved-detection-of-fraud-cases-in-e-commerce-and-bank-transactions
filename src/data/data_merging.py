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
        
        # Filter out invalid IP addresses in fraud_data
        self.fraud_data = self.fraud_data[self.fraud_data['ip_address'].apply(self.is_valid_ip)]

        # Convert valid IP addresses to integers in fraud_data
        self.fraud_data['ip_address_int'] = self.fraud_data['ip_address'].apply(lambda x: int(ipaddress.ip_address(x)))

        # Convert IP addresses to integers in ip_data (already done in your notebook)
        self.ip_data['lower_bound_ip_address_int'] = self.ip_data['lower_bound_ip_address'].astype(int)
        self.ip_data['upper_bound_ip_address_int'] = self.ip_data['upper_bound_ip_address'].astype(int)

        return self.fraud_data, self.ip_data

    def ip_to_country(self, ip):
        left = 0
        right = len(self.ip_data) - 1
        while left <= right:
            mid = (left + right) // 2
            if self.ip_data.iloc[mid]['lower_bound_ip_address_int'] <= ip <= self.ip_data.iloc[mid]['upper_bound_ip_address_int']:
                return self.ip_data.iloc[mid]['country']
            elif ip < self.ip_data.iloc[mid]['lower_bound_ip_address_int']:
                right = mid - 1
            else:
                left = mid + 1
        return 'Unknown'

    def merge_datasets(self):
        # Ensure the columns exist
        if 'ip_address_int' not in self.fraud_data.columns:
            raise KeyError("Column 'ip_address_int' not found in self.fraud_data. Run convert_ip_to_int() first.")

        print("Columns in fraud_data before merging:", self.fraud_data.columns)
        print("Columns in ip_data before merging:", self.ip_data.columns)

        # Map IP addresses to countries using binary search
        self.fraud_data['country'] = self.fraud_data['ip_address_int'].apply(self.ip_to_country)

        # Merge logic
        merged_data = pd.merge_asof(self.fraud_data.sort_values('ip_address_int'),
                                    self.ip_data[['lower_bound_ip_address_int', 'upper_bound_ip_address_int', 'country']],
                                    left_on='ip_address_int',
                                    right_on='lower_bound_ip_address_int',
                                    direction='backward')

        return merged_data