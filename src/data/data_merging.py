import pandas as pd
import ipaddress

class DataMerging:
    def __init__(self, fraud_data, ip_data):
        self.fraud_data = fraud_data
        self.ip_data = ip_data

    def convert_ip_to_int(self):
        self.fraud_data['ip_address_int'] = self.fraud_data['ip_address'].apply(lambda x: int(ipaddress.ip_address(x)))
        self.ip_data['lower_bound_ip_address_int'] = self.ip_data['lower_bound_ip_address'].apply(lambda x: int(ipaddress.ip_address(x)))
        self.ip_data['upper_bound_ip_address_int'] = self.ip_data['upper_bound_ip_address'].apply(lambda x: int(ipaddress.ip_address(x)))
        return self.fraud_data, self.ip_data

    def merge_datasets(self):
        merged_data = pd.merge_asof(self.fraud_data.sort_values('ip_address_int'), 
                                    self.ip_data[['lower_bound_ip_address_int', 'upper_bound_ip_address_int', 'country']],
                                    left_on='ip_address_int', 
                                    right_on='lower_bound_ip_address_int', 
                                    direction='backward')
        return merged_data