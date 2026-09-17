from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"

DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
STATIC_DIR = BASE_DIR / "static"

# Model Feature Specifications
FEATURE_NAMES = [
    "account_length",
    "area_code",
    "international_plan",
    "voice_mail_plan",
    "total_day_calls",
    "total_day_charge",
    "total_eve_calls",
    "total_eve_charge",
    "total_night_calls",
    "total_night_charge",
    "total_intl_calls",
    "total_intl_charge",
    "number_customer_service_calls"
]

# Categorical Value Encodings (Fixes the scaler/model encoding mismatch)
# In training, LabelEncoder mapped 'area_code_408' -> 0, 'area_code_415' -> 1, 'area_code_510' -> 2
AREA_CODE_MAP = {
    408: 0,
    415: 1,
    510: 2,
    "408": 0,
    "415": 1,
    "510": 2,
    "area_code_408": 0,
    "area_code_415": 1,
    "area_code_510": 2,
    0: 0,
    1: 1,
    2: 2
}

BINARY_MAP = {
    "yes": 1,
    "no": 0,
    "y": 1,
    "n": 0,
    1: 1,
    0: 0,
    True: 1,
    False: 0,
    "true": 1,
    "false": 0,
    "Active": 1,
    "Active International Plan": 1,
    "Active Voicemail Plan": 1,
    "No International Plan": 0,
    "No Voicemail Plan": 0
}

# Default customer profile for initialization
DEFAULT_CUSTOMER = {
    "account_length": 36,
    "area_code": 415,
    "international_plan": 1,
    "voice_mail_plan": 0,
    "total_day_calls": 110,
    "total_day_charge": 45.0,
    "total_eve_calls": 100,
    "total_eve_charge": 20.0,
    "total_night_calls": 95,
    "total_night_charge": 10.0,
    "total_intl_calls": 3,
    "total_intl_charge": 4.5,
    "number_customer_service_calls": 4
}
