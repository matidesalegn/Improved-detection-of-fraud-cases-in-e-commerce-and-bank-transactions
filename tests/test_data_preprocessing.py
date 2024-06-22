import unittest
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join('..', 'src')))

from data_preprocessing import DataPreprocessing

class TestDataPreprocessing(unittest.TestCase):

    def setUp(self):
        self.data = pd.DataFrame({
            'A': [1, 2, 2, None],
            'B': [4, None, 6, 7],
            'C': ['2021-01-01', '2021-01-02', '2021-01-03', '2021-01-04']
        })
        self.dp = DataPreprocessing(self.data)

    def test_handle_missing_values(self):
        self.dp.handle_missing_values(method='drop')
        self.assertEqual(self.dp.data.shape[0], 2)

    def test_remove_duplicates(self):
        self.dp.remove_duplicates()
        self.assertEqual(self.dp.data.shape[0], 3)

    def test_correct_data_types(self):
        self.dp.correct_data_types()
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(self.dp.data['C']))

    def test_normalize_and_scale(self):
        self.dp.data = self.dp.data.fillna(0)  # Fill NA to avoid issues with scaler
        self.dp.normalize_and_scale(['A', 'B'])
        self.assertAlmostEqual(self.dp.data['A'].mean(), 0)
        self.assertAlmostEqual(self.dp.data['B'].mean(), 0)

if __name__ == '__main__':
    unittest.main()