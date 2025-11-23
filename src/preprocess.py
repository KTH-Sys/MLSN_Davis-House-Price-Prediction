"""
Preprocessing utilities for house price prediction model.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any


# Default feature order if model.feature_names_in_ is not available
FEATURE_ORDER = [
    'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
]


def build_feature_frame(
    user_inputs: Dict[str, Any],
    expected_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Build a single-row DataFrame matching the model's expected column order.
    
    Args:
        user_inputs: Dictionary with feature names as keys and values as inputs
        expected_cols: List of expected column names from model.feature_names_in_
                       If None, uses FEATURE_ORDER plus inferred zipcode columns
    
    Returns:
        DataFrame with a single row matching the expected column order
    """
    # Determine expected columns
    if expected_cols is not None:
        # Check if model expects simplified features (bed, bath, house_size, acre_lot)
        if len(expected_cols) == 4 and set(expected_cols) == {'bed', 'bath', 'house_size', 'acre_lot'}:
            # Model expects simplified 4-feature format
            sqft_lot = user_inputs.get('sqft_lot', 5000)
            acre_lot = sqft_lot / 43560.0  # Convert sqft to acres
            
            feature_dict = {
                'bed': float(user_inputs.get('bedrooms', 3)),
                'bath': float(user_inputs.get('bathrooms', 2.0)),
                'house_size': float(user_inputs.get('sqft_living', 1500)),
                'acre_lot': acre_lot
            }
        else:
            # Original multi-feature format
            base_features = {
                'bedrooms': user_inputs.get('bedrooms', 3),
                'bathrooms': user_inputs.get('bathrooms', 2.0),
                'sqft_living': user_inputs.get('sqft_living', 1500),
                'sqft_lot': user_inputs.get('sqft_lot', 5000),
            }
            
            # Use model's expected columns
            feature_dict = {col: 0.0 for col in expected_cols}
            
            # Fill in base features
            for feat, val in base_features.items():
                if feat in feature_dict:
                    feature_dict[feat] = float(val)
    else:
        # Fallback to FEATURE_ORDER
        base_features = {
            'bedrooms': user_inputs.get('bedrooms', 3),
            'bathrooms': user_inputs.get('bathrooms', 2.0),
            'sqft_living': user_inputs.get('sqft_living', 1500),
            'sqft_lot': user_inputs.get('sqft_lot', 5000),
        }
        
        feature_dict = base_features.copy()
    
    # Create DataFrame
    df = pd.DataFrame([feature_dict])
    
    # Ensure column order matches expected_cols if provided
    if expected_cols is not None:
        # Add missing columns with zeros
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0.0
        # Reorder columns
        df = df[expected_cols]
    
    return df


def validate_inputs(user_inputs: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Validate user inputs with sensible bounds.
    
    Returns:
        (is_valid, error_message)
    """
    # Bedrooms
    bedrooms = user_inputs.get('bedrooms')
    if bedrooms is None or not (1 <= bedrooms <= 10):
        return False, "Bedrooms must be between 1 and 10"
    
    # Bathrooms
    bathrooms = user_inputs.get('bathrooms')
    if bathrooms is None or not (0.5 <= bathrooms <= 10):
        return False, "Bathrooms must be between 0.5 and 10"
    
    # Square footage
    sqft_living = user_inputs.get('sqft_living')
    if sqft_living is None or not (100 <= sqft_living <= 20000):
        return False, "Square footage (living) must be between 100 and 20,000"
    
    sqft_lot = user_inputs.get('sqft_lot')
    if sqft_lot is None or not (100 <= sqft_lot <= 1000000):
        return False, "Square footage (lot) must be between 100 and 1,000,000"
    
    return True, None


def coerce_number(value: Any, default: float = 0.0) -> float:
    """
    Coerce a value to a float, returning default if conversion fails.
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def format_currency(amount: float) -> str:
    """
    Format a number as US currency.
    """
    return f"${amount:,.2f}"

