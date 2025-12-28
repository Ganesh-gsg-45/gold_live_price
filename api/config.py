import os
from dotenv import load_dotenv

load_dotenv()

GOLD_API_KEY = os.getenv('GOLD_API_KEY', 'goldapi-demo-key')
GOLD_API_BASE_URL = 'https://www.goldapi.io/api'

METALS_API_KEY = os.getenv('METALS_API_KEY', '')
METALS_API_BASE_URL = 'https://metals-api.com/api'

SUPPORTED_CURRENCIES = ['USD', 'EUR', 'GBP', 'INR', 'AUD', 'CAD']
GOLD_PURITIES = {
    '24K': 1.0,
    '22K': 0.916,
    '18K': 0.75
}

SEQUENCE_LENGTH = 5
PREDICTION_DAYS = 5
MODEL_PATH = 'models/saved/'
DATA_PATH = 'data/'