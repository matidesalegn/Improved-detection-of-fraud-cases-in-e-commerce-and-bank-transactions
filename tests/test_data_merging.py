import unittest
import pandas as pd
import ipaddress
from intervaltree import IntervalTree

# Assuming the functions from data_merge.py are available here
def ip_to_int(ip):
    if pd.isna(ip):
        return None
    try:
        return int(ip)
    except ValueError:
        return int(ipaddress.ip_address(ip))

def map_ip_to_country(ip_int, ip_tree):
    if ip_int is None:
        return 'Unknown'
    interval = ip_tree[ip_int]
    if interval:
        return interval.pop().data
    else:
        return 'Unknown'

class TestDataMerging(unittest.TestCase):

    def setUp(self):
        # Create sample fraud data
        self.fraud_data = pd.DataFrame({
            'user_id': [1, 2, 3],
            'signup_time': ['2023-01-01 00:00:00', '2023-01-02 00:00:00', '2023-01-03 00:00:00'],
            'purchase_time': ['2023-01-01 01:00:00', '2023-01-02 01:00:00', '2023-01-03 01:00:00'],
            'purchase_value': [100.0, 150.0, 200.0],
            'device_id': ['A', 'B', 'C'],
            'source': ['SEO', 'Ads', 'Direct'],
            'browser': ['Chrome', 'Safari', 'Firefox'],
            'sex': ['M', 'F', 'M'],
            'age': [25, 30, 35],
            'ip_address': ['192.168.1.1', '192.168.1.2', '192.168.1.3'],
            'class': [0, 1, 0]
        })

        # Create sample IP to country data
        self.ip_address_data = pd.DataFrame({
            'lower_bound_ip_address': [3232235776, 3232235778],  # 192.168.1.0, 192.168.1.2
            'upper_bound_ip_address': [3232235777, 3232235779],  # 192.168.1.1, 192.168.1.3
            'country': ['Country1', 'Country2']
        })

        # Handle NaN values in IP address columns and convert IP address ranges to integers
        self.ip_address_data['lower_bound_ip_address_int'] = self.ip_address_data['lower_bound_ip_address'].apply(ip_to_int)
        self.ip_address_data['upper_bound_ip_address_int'] = self.ip_address_data['upper_bound_ip_address'].apply(ip_to_int)

        # Create an interval tree for IP ranges
        self.ip_tree = IntervalTree()
        for _, row in self.ip_address_data.iterrows():
            self.ip_tree[row['lower_bound_ip_address_int']:row['upper_bound_ip_address_int'] + 1] = row['country']

    def test_ip_to_int(self):
        self.assertEqual(ip_to_int('192.168.1.1'), 3232235777)
        self.assertEqual(ip_to_int('::1'), int(ipaddress.ip_address('::1')))
        self.assertEqual(ip_to_int(None), None)
        self.assertEqual(ip_to_int(float('nan')), None)

    def test_map_ip_to_country(self):
        ip_int = ip_to_int('192.168.1.1')
        self.assertEqual(map_ip_to_country(ip_int, self.ip_tree), 'Country1')

        ip_int = ip_to_int('192.168.1.2')
        self.assertEqual(map_ip_to_country(ip_int, self.ip_tree), 'Country2')

        ip_int = ip_to_int('192.168.1.4')
        self.assertEqual(map_ip_to_country(ip_int, self.ip_tree), 'Unknown')

    def test_data_merging(self):
        # Apply function to get country for each IP address
        self.fraud_data['ip_address_int'] = self.fraud_data['ip_address'].apply(ip_to_int)
        self.fraud_data['country'] = self.fraud_data['ip_address_int'].apply(lambda ip: map_ip_to_country(ip, self.ip_tree))

        # Check the merged results
        expected_countries = ['Country1', 'Country2', 'Unknown']
        self.assertListEqual(self.fraud_data['country'].tolist(), expected_countries)

if __name__ == "__main__":
    unittest.main()