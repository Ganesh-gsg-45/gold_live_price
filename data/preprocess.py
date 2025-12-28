import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from api.config import SEQUENCE_LENGTH

class DataPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def prepare_data(self, df):
        if 'date' in df.columns:
            df = df.sort_values('date')
            df.set_index('date', inplace=True)
        
        prices = df['price_per_gram'].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(prices)
        
        return scaled_data
    
    def create_sequences(self, data, seq_length=SEQUENCE_LENGTH):
        X, y = [], []
        
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        
        return np.array(X), np.array(y)
    
    def inverse_transform(self, scaled_data):
        return self.scaler.inverse_transform(scaled_data)
    
    def split_data(self, X, y, train_ratio=0.8):
        split_idx = int(len(X) * train_ratio)
        
        X_train = X[:split_idx]
        y_train = y[:split_idx]
        X_test = X[split_idx:]
        y_test = y[split_idx:]
        
        return X_train, X_test, y_train, y_test
