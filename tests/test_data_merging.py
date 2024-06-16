import unittest
import pandas as pd
from src.data.data_merging import DataMerging

class TestDataMerging(unittest.TestCase):

    def setUp(self):
        self.fraud_data = pd.DataFrame({
            'ip_address': ['192.168.1.1', '192.168.1.2', '192.168.1.3'],
            'purchase_value': [100, 200, 300]
        })
        self.ip_data = pd.DataFrame({
            'lower_bound_ip_address': ['192.168.1.0', '192.168.1.2'],
            'upper_bound_ip_address': ['192.168.1.1', '192.168.1.3'],
            'country': ['US', 'CA']
        })
        self.dm = DataMerging(self.fraud_data, self.ip_data)

    def test_convert_ip_to_int(self):
        fraud_data_int, ip_data_int = self.dm.convert_ip_to_int()
        self.assertIsInstance(fraud_data_int['ip_address_int'][0], int)
        self.assertIsInstance(ip_data_int['lower_bound_ip_address_int'][0], int)

    def test_merge_datasets(self):
        self.dm.convert_ip_to_int()
        merged_data = self.dm.merge_datasets()
        self.assertIn('country', merged_data.columns)
        self.assertEqual(merged_data.iloc[0]['country'], 'US')
        self.assertEqual(merged_data.iloc[2]['country'], 'CA')

if __name__ == '__main__':
    unittest.main()