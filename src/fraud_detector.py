import os
import joblib
import pandas as pd
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'fraud_detector.joblib')

class FraudDetector:
    def __init__(self):
        saved = joblib.load(MODEL_PATH)
        self.model = saved['model']
        self.scaler = saved['scaler']
        self.ohe = saved['ohe']
        self.num_cols = saved['num_cols']
        self.cat_cols = saved['cat_cols']

    def predict_fraud_probability(self, df: pd.DataFrame) -> np.ndarray:
        df = df.copy()
        if 'claim_to_premium_ratio' not in df.columns:
            df['claim_to_premium_ratio'] = df['total_claim_amount'] / (df['policy_annual_premium'] + 1e-5)
        if 'vehicle_claim_ratio' not in df.columns:
            df['vehicle_claim_ratio'] = df['vehicle_claim'] / (df['total_claim_amount'] + 1e-5)
            
        for c in self.cat_cols:
            if c not in df.columns:
                df[c] = 'MISSING'
            df[c] = df[c].fillna('MISSING')
            
        for c in self.num_cols:
            if c not in df.columns:
                df[c] = 0.0
            df[c] = df[c].fillna(0.0)
            
        enc = np.hstack([df[self.num_cols].values, self.ohe.transform(df[self.cat_cols])])
        scaled = self.scaler.transform(enc)
        return self.model.predict_proba(scaled)[:, 1]
