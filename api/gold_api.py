import requests
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime, timedelta
from api.config import METALS_API_KEY, METALS_API_BASE_URL, GOLD_PURITIES

class GoldPriceAPI:
    def __init__(self):
        self.api_key = METALS_API_KEY
        self.base_url = METALS_API_BASE_URL
        self.headers = {
            'x-access-token': self.api_key,
            'Content-Type': 'application/json'
        }
    
    def get_current_price(self, currency='INR'):
        # Demo prices for different currencies (approximate values)
        demo_prices = {
            'USD': 65.0,
            'EUR': 60.0,
            'GBP': 52.0,
            'INR': 5400.0,
            'AUD': 95.0,
            'CAD': 85.0
        }

        demo_price_per_gram = demo_prices.get(currency, 65.0)  # Default to USD if currency not found

        return {
            'timestamp': datetime.now(),
            'price_per_oz': demo_price_per_gram * 31.1035,
            'price_per_gram': demo_price_per_gram,
            'currency': currency,
            '24K': demo_price_per_gram,
            '22K': demo_price_per_gram * GOLD_PURITIES['22K'],
            '18K': demo_price_per_gram * GOLD_PURITIES['18K']
        }
    
    def get_historical_prices(self, currency='INR', days=5):
        historical_data = []
        
        for i in range(days, 0, -1):
            date = (datetime.now() - timedelta(days=i)).strftime('%Y%m%d')
            
            try:
                endpoint = f"{self.base_url}/{currency}/{date}"
                response = requests.get(endpoint, headers=self.headers)
                
                if response.status_code == 200:
                    data = response.json()
                    price_per_oz = data.get('price', 0)
                    price_per_gram = price_per_oz / 31.1035
                    
                    historical_data.append({
                        'date': datetime.strptime(date, '%Y%m%d'),
                        'price_per_gram': price_per_gram,
                        '24K': price_per_gram,
                        '22K': price_per_gram * GOLD_PURITIES['22K'],
                        '18K': price_per_gram * GOLD_PURITIES['18K']
                    })
            except:
                continue
        
        return pd.DataFrame(historical_data)
    
    def calculate_price_by_weight(self, weight_grams, purity='24K', currency='INR'):
        current_data = self.get_current_price(currency)
        price_per_gram = current_data[purity]
        total_price = weight_grams * price_per_gram
        
        return {
            'weight_grams': weight_grams,
            'purity': purity,
            'price_per_gram': price_per_gram,
            'total_price': total_price,
            'currency': currency
        }

