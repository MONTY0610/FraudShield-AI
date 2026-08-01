import pandas as pd
import joblib

# ==========================
# File Paths
# ==========================

DATASET_PATH = "data/creditcard.csv/creditcard.csv"
MODEL_PATH = "models/fraud_model.pkl"
METRICS_PATH = "reports/model_metrics.csv"
FEATURE_IMPORTANCE_PATH = "reports/feature_importance.csv"

# ==========================
# Load Functions
# ==========================

def load_dataset():
    return pd.read_csv("data/sample_creditcard.csv")


def load_model():
    return joblib.load(MODEL_PATH)


def load_metrics():
    return pd.read_csv(METRICS_PATH)


def load_feature_importance():
    return pd.read_csv(FEATURE_IMPORTANCE_PATH)