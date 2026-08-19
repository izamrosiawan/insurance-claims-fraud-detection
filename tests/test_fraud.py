import pytest
import pandas as pd
import numpy as np
from src.fraud_detector import FraudDetector

def test_fraud_probability_range():
    detector = FraudDetector()
    sample = pd.read_csv('data/insurance_claims.csv', nrows=5)
    probs = detector.predict_fraud_probability(sample)
    
    assert len(probs) == 5
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
