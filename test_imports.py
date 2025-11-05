#!/usr/bin/env python3
"""Test script to verify all imports work correctly"""

print("Testing imports...")

try:
    import streamlit as st
    print("✅ streamlit")
except ImportError as e:
    print(f"❌ streamlit: {e}")

try:
    import pandas as pd
    print("✅ pandas")
except ImportError as e:
    print(f"❌ pandas: {e}")

try:
    import numpy as np
    print("✅ numpy")
except ImportError as e:
    print(f"❌ numpy: {e}")

try:
    import joblib
    print(f"✅ joblib (version: {joblib.__version__})")
except ImportError as e:
    print(f"❌ joblib: {e}")

try:
    from sklearn.dummy import DummyRegressor
    print("✅ scikit-learn")
except ImportError as e:
    print(f"❌ scikit-learn: {e}")

try:
    import altair as alt
    print("✅ altair")
except ImportError as e:
    print(f"❌ altair: {e}")

try:
    from src.preprocess import build_feature_frame, validate_inputs, format_currency
    print("✅ src.preprocess")
except ImportError as e:
    print(f"❌ src.preprocess: {e}")

print("\nAll imports tested!")

