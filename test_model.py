"""Quick test to check model type"""
import joblib
from pathlib import Path

model_path = Path("models/model.pkl")
if model_path.exists():
    model = joblib.load(model_path)
    print(f"✅ Model loaded successfully!")
    print(f"Model type: {type(model).__name__}")
    print(f"Is DummyRegressor: {type(model).__name__ == 'DummyRegressor'}")
    print(f"Has feature_names_in_: {hasattr(model, 'feature_names_in_')}")
    if hasattr(model, 'feature_names_in_'):
        print(f"Number of features: {len(model.feature_names_in_)}")
else:
    print("❌ Model file not found")

