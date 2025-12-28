import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from datetime import datetime, timedelta
from api.gold_api import GoldPriceAPI
from data.preprocess import DataPreprocessor
from models.lstm_train import LSTMModel
from models.gru_train import GRUModel
from api.config import MODEL_PATH, PREDICTION_DAYS, SEQUENCE_LENGTH

class ModelTrainer:
    def __init__(self, model_type='LSTM'):
        self.model_type = model_type
        self.api = GoldPriceAPI()
        self.preprocessor = DataPreprocessor()
        
        if model_type == 'LSTM':
            self.model = LSTMModel()
        elif model_type == 'GRU':
            self.model = GRUModel()
        else:
            raise ValueError("Model type must be 'LSTM' or 'GRU'")
        
        os.makedirs(MODEL_PATH, exist_ok=True)
    
    def fetch_training_data(self, currency='INR', days=180):
        df = self.api.get_historical_prices(currency, days)
        return df
    
    def prepare_training_data(self, df):
        scaled_data = self.preprocessor.prepare_data(df)
        X, y = self.preprocessor.create_sequences(scaled_data, SEQUENCE_LENGTH)
        X_train, X_test, y_train, y_test = self.preprocessor.split_data(X, y)
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, currency='INR', epochs=50):
        df = self.fetch_training_data(currency)
        
        if len(df) < SEQUENCE_LENGTH + 10:
            raise ValueError("Not enough historical data for training")
        
        X_train, X_test, y_train, y_test = self.prepare_training_data(df)
        
        self.model.train(X_train, y_train, X_test, y_test, epochs=epochs)
        
        metrics = self.model.evaluate(X_test, y_test)
        
        model_path = os.path.join(MODEL_PATH, f'{self.model_type}_{currency}.h5')
        self.model.save_model(model_path)
        
        return metrics
    
    def predict_future_prices(self, currency='INR', days=PREDICTION_DAYS):
        df = self.api.get_historical_prices(currency, days=SEQUENCE_LENGTH)
        
        if len(df) < SEQUENCE_LENGTH:
            raise ValueError("Not enough historical data for prediction")
        
        scaled_data = self.preprocessor.prepare_data(df)
        last_sequence = scaled_data[-SEQUENCE_LENGTH:]
        
        scaled_predictions = self.model.predict_future(last_sequence, days)
        predictions = self.preprocessor.inverse_transform(scaled_predictions)
        
        future_dates = [datetime.now() + timedelta(days=i+1) for i in range(days)]
        
        prediction_df = pd.DataFrame({
            'date': future_dates,
            'predicted_price': predictions.flatten()
        })
        
        return prediction_df
    
    def load_trained_model(self, currency='INR'):
        model_path = os.path.join(MODEL_PATH, f'{self.model_type}_{currency}.h5')
        
        if os.path.exists(model_path):
            self.model.load_model(model_path)
            return True
        return False