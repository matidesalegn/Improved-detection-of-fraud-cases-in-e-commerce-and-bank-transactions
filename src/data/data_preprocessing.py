import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class DataPreprocessing:
    def __init__(self, data):
        self.data = data

    def handle_missing_values(self, method='drop', fill_value=None):
        if method == 'drop':
            self.data = self.data.dropna()
        elif method == 'fill':
            self.data = self.data.fillna(fill_value)
        return self.data

    def remove_duplicates(self):
        self.data = self.data.drop_duplicates()
        return self.data

    def correct_data_types(self):
        if 'signup_time' in self.data.columns:
            self.data['signup_time'] = pd.to_datetime(self.data['signup_time'])
        if 'purchase_time' in self.data.columns:
            self.data['purchase_time'] = pd.to_datetime(self.data['purchase_time'])
        return self.data

    def normalize_and_scale(self, columns):
        scaler = StandardScaler()
        self.data[columns] = scaler.fit_transform(self.data[columns])
        return self.data

    def encode_categorical(self, columns):
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        encoded_features = encoder.fit_transform(self.data[columns])
        encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(columns))
        self.data = pd.concat([self.data.drop(columns, axis=1), encoded_df], axis=1)
        return self.data