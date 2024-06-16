import unittest
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join('..', 'src')))

from data.data_merging import DataMerging

class TestDataMerging(unittest.TestCase):

    def setUp(self):
        self.fraud_data = pd.DataFrame({
            'ip_address': ['192.168.1.1', '192.168.1.2', '192.168.1.3', 'invalid_ip'],
            'purchase_value': [100, 200, 300, 400]
        })
        self.ip_data = pd.DataFrame({
            'lower_bound_ip_address': ['192.168.1.0', '192.168.1.2', 'invalid_ip'],
            'upper_bound_ip_address': ['192.168.1.1', '192.168.1.3', 'invalid_ip'],
            'country': ['US', 'CA', 'InvalidCountry']
        })
        self.dm = DataMerging(self.fraud_data, self.ip_data)

    def test_convert_ip_to_int(self):
        fraud_data_int, ip_data_int = self.dm.convert_ip_to_int()
        self.assertIsInstance(fraud_data_int['ip_address_int'][0], int)
        self.assertIsInstance(ip_data_int['lower_bound_ip_address_int'][0], int)
        self.assertNotIn('invalid_ip', fraud_data_int['ip_address'].values)
        self.assertNotIn('invalid_ip', ip_data_int['lower_bound_ip_address'].values)
        self.assertNotIn('invalid_ip', ip_data_int['upper_bound_ip_address'].values)

    def test_merge_datasets(self):
        self.dm.convert_ip_to_int()
        merged_data = self.dm.merge_datasets()
        self.assertIn('country', merged_data.columns)
        self.assertEqual(merged_data.iloc[0]['country'], 'US')
        self.assertEqual(merged_data.iloc[2]['country'], 'CA')

if __name__ == '__main__':
    unittest.main()