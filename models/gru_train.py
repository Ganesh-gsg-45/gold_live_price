import numpy as np
import sys 
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from api.config import SEQUENCE_LENGTH

class GRUModel:
    def __init__(self, seq_length=SEQUENCE_LENGTH):
        self.seq_length = seq_length
        self.model = None
        self.history = None
        
    def build_model(self, input_shape):
        model = Sequential([
            GRU(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            GRU(64, return_sequences=True),
            Dropout(0.2),
            GRU(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        if self.model is None:
            self.build_model((X_train.shape[1], X_train.shape[2]))
        
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=0
        )
        
        return self.history
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X, verbose=0)
    
    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        return {
            'RMSE': round(rmse, 4),
            'MAE': round(mae, 4),
            'R2': round(r2, 4)
        }
    
    def predict_future(self, last_sequence, days=5):
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(days):
            pred = self.model.predict(
                current_sequence.reshape(1, self.seq_length, 1),
                verbose=0
            )
            predictions.append(pred[0, 0])
            current_sequence = np.append(current_sequence[1:], pred)
        
        return np.array(predictions).reshape(-1, 1)
    
    def save_model(self, path):
        if self.model:
            self.model.save(path)
    
    def load_model(self, path):
        self.model = keras.models.load_model(path)
