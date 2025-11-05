"""
Davis House Price Prediction - Streamlit App

A polished, modern Streamlit UI for predicting house prices using a trained
regression model with glassy design and purple accents.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.dummy import DummyRegressor
import warnings
import altair as alt

from src.preprocess import (
    build_feature_frame,
    validate_inputs,
    format_currency,
    FEATURE_ORDER
)

# Page configuration
st.set_page_config(
    page_title="Davis House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Suppress warnings
warnings.filterwarnings('ignore')


def inject_custom_css():
    """Inject custom CSS for modern glassy design with purple accents"""
    st.markdown("""
    <style>
    /* Import Google Fonts - Open Sans */
    @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap');
    
    /* Main background - Subtle gradient */
    .stApp {
        background: linear-gradient(180deg, #ffffff 0%, #f4f0ff 100%);
        background-attachment: fixed;
        font-family: 'Open Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Main container */
    .main .block-container {
        max-width: 900px;
        margin: 0 auto;
        padding: 2rem 2rem;
        background: transparent;
    }
    
    /* Glassy sidebar */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.85) !important;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-right: 1px solid rgba(108, 99, 255, 0.1);
        box-shadow: 2px 0 20px rgba(0, 0, 0, 0.05);
    }
    
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
        background: transparent !important;
    }
    
    /* Title styling */
    h1 {
        text-align: center;
        color: #6C63FF;
        font-weight: 700;
        margin-bottom: 0.5rem;
        font-size: 2.5rem;
        font-family: 'Open Sans', sans-serif;
    }
    
    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Glassy form container */
    [data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.9) !important;
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border: 1px solid rgba(108, 99, 255, 0.15);
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(108, 99, 255, 0.1);
        margin: 2rem 0;
    }
    
    /* Input fields - Glassy white */
    [data-testid="stNumberInput"] > div > div,
    [data-testid="stTextInput"] > div > div {
        background: rgba(255, 255, 255, 0.95) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(108, 99, 255, 0.2) !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 8px rgba(108, 99, 255, 0.05);
    }
    
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input {
        background: transparent !important;
        color: #1f2937 !important;
        font-family: 'Open Sans', sans-serif;
    }
    
    /* Input labels */
    label {
        font-weight: 600;
        color: #374151;
        font-size: 0.95rem;
        font-family: 'Open Sans', sans-serif;
    }
    
    /* Purple pill button */
    .stButton > button {
        background: #6C63FF !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        font-family: 'Open Sans', sans-serif;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(108, 99, 255, 0.3);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(108, 99, 255, 0.5);
        background: #5a52e6 !important;
    }
    
    /* Prediction metric card */
    .prediction-card {
        background: rgba(255, 255, 255, 0.9) !important;
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border: 1px solid rgba(108, 99, 255, 0.2);
        border-radius: 20px;
        padding: 2.5rem;
        box-shadow: 0 8px 30px rgba(108, 99, 255, 0.15);
        margin: 2rem 0;
        text-align: center;
    }
    
    .prediction-price {
        font-size: 3.5rem;
        font-weight: 700;
        margin: 1rem 0;
        color: #6C63FF;
        font-family: 'Open Sans', sans-serif;
    }
    
    /* Metric styling */
    [data-testid="stMetricContainer"] {
        background: rgba(255, 255, 255, 0.9) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(108, 99, 255, 0.15);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(108, 99, 255, 0.1);
    }
    
    [data-testid="stMetricValue"] {
        font-size: 3rem;
        color: #6C63FF;
        font-family: 'Open Sans', sans-serif;
        font-weight: 700;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1.1rem;
        color: #6b7280;
        font-family: 'Open Sans', sans-serif;
    }
    
    /* Sidebar info boxes */
    [data-testid="stInfo"] {
        background: rgba(108, 99, 255, 0.08) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(108, 99, 255, 0.2);
        border-left: 4px solid #6C63FF;
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(108, 99, 255, 0.1);
    }
    
    [data-testid="stSuccess"] {
        background: rgba(34, 197, 94, 0.1) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(34, 197, 94, 0.2);
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(34, 197, 94, 0.1);
    }
    
    [data-testid="stWarning"] {
        background: rgba(251, 191, 36, 0.1) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(251, 191, 36, 0.2);
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(251, 191, 36, 0.1);
    }
    
    /* Section headers */
    h2, h3 {
        color: #1f2937;
        font-weight: 600;
        font-family: 'Open Sans', sans-serif;
    }
    
    h3 {
        color: #6C63FF;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1.5rem 1rem;
        }
        
        h1 {
            font-size: 2rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource(ttl=0)
def load_model():
    """Load the trained model from models/model.pkl"""
    model_path = Path("models/model.pkl")
    
    if model_path.exists():
        try:
            model = joblib.load(model_path)
            return model, None
        except Exception as e:
            return None, f"Error loading model: {str(e)}"
    else:
        # Fallback to dummy model
        dummy_model = DummyRegressor(strategy="mean")
        dummy_X = np.random.rand(10, 8)
        dummy_y = np.random.rand(10) * 500000 + 300000
        dummy_model.fit(dummy_X, dummy_y)
        return dummy_model, "⚠️ Model file not found. Using dummy predictor for demonstration."


@st.cache_data(ttl=0)
def load_dataset():
    """Load the Davis housing dataset"""
    dataset_path = Path("data/davis_housing_clean_2.csv")
    
    if dataset_path.exists():
        try:
            df = pd.read_csv(dataset_path)
            return df, None
        except Exception as e:
            return None, f"Error loading dataset: {str(e)}"
    else:
        return None, "Dataset file not found."


@st.cache_data(ttl=0)
def get_model_info(_model):
    """Extract model information"""
    info = {
        'has_feature_names': hasattr(_model, 'feature_names_in_'),
        'feature_names': getattr(_model, 'feature_names_in_', None),
        'model_type': type(_model).__name__
    }
    return info


def reset_form():
    """Reset all form inputs"""
    for key in st.session_state:
        if key.startswith('input_'):
            del st.session_state[key]


def main():
    """Main application"""
    
    # Inject custom CSS
    inject_custom_css()
    
    # Load model
    model, model_warning = load_model()
    if model is None:
        st.error(model_warning)
        st.stop()
    
    # Load dataset
    df, dataset_warning = load_dataset()
    if df is None:
        st.warning(dataset_warning)
        dataset_median = 600000  # Fallback
        dataset_prices = None
    else:
        if 'price' in df.columns:
            dataset_median = df['price'].median()
            dataset_prices = df['price'].values
        else:
            dataset_median = 600000
            dataset_prices = None
    
    # Get model info
    model_info = get_model_info(model)
    expected_cols = model_info['feature_names']
    
    # Sidebar - Dynamic Insights & Recommendations
    with st.sidebar:
        has_prediction = 'predicted_price' in st.session_state and st.session_state.predicted_price is not None
        
        # Logo paths
        logo_path_png = Path("assets/mlsn_logo.png")
        logo_path_jpg = Path("assets/MLSN_logo.jpg")
        logo_path = logo_path_png if logo_path_png.exists() else (logo_path_jpg if logo_path_jpg.exists() else None)
        
        if not has_prediction:
            # Before prediction: Show logo and credit
            if logo_path:
                col1, col2, col3 = st.columns([1, 4, 1])
                with col2:
                    st.image(str(logo_path), width=150)
            
            st.markdown("""
            <div style="text-align: center; margin-top: 1.5rem; margin-bottom: 1rem; padding: 0 0.5rem;">
                <p style="font-size: 0.85rem; color: #6b7280; line-height: 1.5;">
                    Built by <strong style="color: #6C63FF;">Machine Learning Student Network – UC Davis</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.divider()
        else:
            # After prediction: Show Insights & Recommendations
            predicted_price = st.session_state.predicted_price
            user_inputs = st.session_state.get('user_inputs', {})
            
            st.markdown("""
            <div style="text-align: center; margin-bottom: 1.5rem;">
                <h3 style="color: #6C63FF; font-size: 1.4rem; margin-bottom: 0.5rem; font-weight: 700;">
                    📊 Insights & Recommendations
                </h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Calculate percentile
            if dataset_prices is not None:
                percentile_rank = (dataset_prices < predicted_price).sum() / len(dataset_prices) * 100
                percentile_rank = int(percentile_rank)
            else:
                price_ratio = predicted_price / dataset_median if dataset_median > 0 else 1.0
                if price_ratio >= 1.5:
                    percentile_rank = 75 + min(20, int((price_ratio - 1.5) * 10))
                elif price_ratio >= 1.2:
                    percentile_rank = 60 + int((price_ratio - 1.2) * 50)
                elif price_ratio >= 1.0:
                    percentile_rank = 50 + int((price_ratio - 1.0) * 50)
                elif price_ratio >= 0.8:
                    percentile_rank = 30 + int((price_ratio - 0.8) * 100)
                else:
                    percentile_rank = max(5, int(price_ratio * 37.5))
                percentile_rank = min(95, max(5, percentile_rank))
            
            # Insight 1: Percentile comparison
            if percentile_rank >= 70:
                st.success(f"✅ Your predicted home price is higher than **{percentile_rank}%** of Davis listings.")
            elif percentile_rank >= 50:
                st.info(f"💡 Your predicted home price is higher than **{percentile_rank}%** of Davis listings.")
            else:
                st.warning(f"⚠️ Your predicted home price is higher than **{percentile_rank}%** of Davis listings.")
            
            # Insight 2: Home size comparison
            sqft_living = user_inputs.get('sqft_living', 0)
            if df is not None and 'sqft_living' in df.columns:
                median_sqft = df['sqft_living'].median()
                if sqft_living < median_sqft:
                    st.warning(f"🏠 Homes under **{int(median_sqft)} sqft** are typically below the median price.")
            
            # Insight 3: What-if for adding bathroom
            bathrooms = user_inputs.get('bathrooms', 0)
            if bathrooms < 5:
                try:
                    what_if_inputs = user_inputs.copy()
                    what_if_inputs['bathrooms'] = bathrooms + 1.0
                    what_if_feature_df = build_feature_frame(what_if_inputs, expected_cols)
                    what_if_prediction = model.predict(what_if_feature_df)[0]
                    if what_if_prediction < 0:
                        what_if_prediction = abs(what_if_prediction)
                    price_increase = what_if_prediction - predicted_price
                    if price_increase > 0:
                        st.info(f"💡 Adding another bathroom could increase value by ≈ **{format_currency(price_increase)}**.")
                except:
                    pass
            
            # Altair comparison chart
            st.markdown("### 📊 Price Comparison")
            chart_data = pd.DataFrame({
                'Type': ['Predicted Price', 'Davis Median'],
                'Price': [predicted_price, dataset_median]
            })
            
            chart = alt.Chart(chart_data).mark_bar(
                cornerRadius=10
            ).encode(
                x=alt.X('Type', axis=alt.Axis(title='', labelFontSize=12)),
                y=alt.Y('Price', axis=alt.Axis(title='Price ($)', format='$,.0f')),
                color=alt.condition(
                    alt.datum.Type == 'Predicted Price',
                    alt.value('#6C63FF'),
                    alt.value('#9ca3af')
                )
            ).properties(
                width=300,
                height=200
            )
            
            text = chart.mark_text(
                align='center',
                baseline='bottom',
                dy=-5,
                fontSize=12,
                fontWeight='bold',
                color='white'
            ).encode(
                text=alt.Text('Price:Q', format='$,.0f')
            )
            
            st.altair_chart(chart + text, use_container_width=True)
            
            # Reset button
            if st.button("🔄 Reset Form", use_container_width=True):
                reset_form()
                if 'predicted_price' in st.session_state:
                    del st.session_state.predicted_price
                if 'user_inputs' in st.session_state:
                    del st.session_state.user_inputs
                st.rerun()
            
            st.divider()
            
            # Logo and credit at bottom
            if logo_path:
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.image(str(logo_path), width=80)
                with col2:
                    st.markdown("""
                    <div style="display: flex; align-items: center; height: 100%; padding-left: 0.5rem; padding-top: 1rem;">
                        <p style="font-size: 0.75rem; color: #6b7280; margin: 0;">
                            Built by <strong style="color: #6C63FF;">Machine Learning Student Network – UC Davis</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Model Information (collapsed)
        with st.expander("⚙️ Model Information"):
            is_dummy = model_info['model_type'] == 'DummyRegressor'
            if is_dummy:
                st.info("🧪 **Demo Mode: Using Dummy Predictor**")
            else:
                st.success(f"✅ **Model Type:** `{model_info['model_type']}`")
            if model_info['has_feature_names']:
                st.markdown(f"**Features:** {len(expected_cols)}")
    
    # Main content
    st.title("🏠 Davis House Price Predictor")
    st.markdown('<p class="subtitle">Estimate housing prices in Davis with machine learning.</p>', unsafe_allow_html=True)
    
    # Prediction form
    with st.form("prediction_form"):
        st.markdown("### 🏡 Property Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            bedrooms = st.number_input(
                "🛏 Bedrooms",
                min_value=1,
                max_value=10,
                value=3,
                step=1,
                key="input_bedrooms"
            )
            
            bathrooms = st.number_input(
                "🛁 Bathrooms",
                min_value=0.5,
                max_value=10.0,
                value=2.0,
                step=0.5,
                key="input_bathrooms"
            )
        
        with col2:
            sqft_living = st.number_input(
                "📏 House Size (sqft)",
                min_value=100,
                max_value=20000,
                value=1500,
                step=100,
                key="input_sqft_living"
            )
            
            sqft_lot = st.number_input(
                "🌳 Lot Size (sqft)",
                min_value=100,
                max_value=1000000,
                value=5000,
                step=100,
                key="input_sqft_lot"
            )
        
        # Additional inputs in second row
        col3, col4 = st.columns(2)
        
        with col3:
            year_built = st.number_input(
                "📅 Year Built",
                min_value=1800,
                max_value=2025,
                value=1990,
                step=1,
                key="input_year_built"
            )
            
            condition = st.number_input(
                "⭐ Condition (1-5)",
                min_value=1,
                max_value=5,
                value=3,
                step=1,
                key="input_condition"
            )
        
        with col4:
            grade = st.number_input(
                "🏆 Grade (1-13)",
                min_value=1,
                max_value=13,
                value=7,
                step=1,
                key="input_grade"
            )
            
            zipcode = st.text_input(
                "📍 Zipcode",
                value="95616",
                key="input_zipcode"
            )
        
        distance_to_ucd = st.number_input(
            "🎓 Distance to UC Davis (km)",
            min_value=0.0,
            max_value=50.0,
            value=5.0,
            step=0.5,
            key="input_distance_to_ucd"
        )
        
        submitted = st.form_submit_button("🔮 Predict Price", use_container_width=True, type="primary")
    
    # Prediction logic
    if submitted:
        user_inputs = {
            'bedrooms': int(bedrooms),
            'bathrooms': float(bathrooms),
            'sqft_living': int(sqft_living),
            'sqft_lot': int(sqft_lot),
            'year_built': int(year_built),
            'condition': int(condition),
            'grade': int(grade),
            'zipcode': str(zipcode),
            'distance_to_ucd': float(distance_to_ucd)
        }
        
        is_valid, error_msg = validate_inputs(user_inputs)
        
        if not is_valid:
            st.error(f"❌ Validation Error: {error_msg}")
        else:
            try:
                # Build feature frame
                feature_df = build_feature_frame(user_inputs, expected_cols)
                
                # Verify column alignment
                if expected_cols is not None:
                    missing_cols = set(expected_cols) - set(feature_df.columns)
                    extra_cols = set(feature_df.columns) - set(expected_cols)
                    
                    if missing_cols or extra_cols:
                        st.error(f"❌ Column mismatch detected!")
                        st.stop()
                
                # Debug: Show feature info
                with st.expander("🔍 Debug: Feature Information", expanded=False):
                    st.write(f"**Model Type:** {model_info['model_type']}")
                    st.write(f"**Expected Columns:** {len(expected_cols) if expected_cols is not None else 'N/A'}")
                    st.write(f"**Feature DataFrame Shape:** {feature_df.shape}")
                    st.write(f"**Feature DataFrame Columns:** {list(feature_df.columns)[:10]}")
                    st.write("**Feature Values:**")
                    st.dataframe(feature_df.T)
                    st.write("**Non-zero features:**")
                    non_zero = feature_df.loc[:, (feature_df != 0).any(axis=0)]
                    st.dataframe(non_zero)
                
                # Make prediction
                prediction_raw = model.predict(feature_df)[0]
                
                # Debug: Show raw prediction
                with st.expander("🔍 Debug: Prediction Info", expanded=False):
                    st.write(f"**Raw Prediction:** {prediction_raw:.2f}")
                    st.write(f"**Is Negative:** {prediction_raw < 0}")
                
                # Handle negative predictions
                if prediction_raw < 0:
                    prediction = abs(prediction_raw)
                    st.warning(f"⚠️ Model returned negative prediction ({prediction_raw:.2f}). This suggests a feature mismatch or model issue. Using absolute value.")
                else:
                    prediction = prediction_raw
                
                # Clamp to reasonable range
                prediction = max(10000, min(prediction, 10000000))
                
                # Store in session state
                st.session_state.predicted_price = float(prediction)
                st.session_state.user_inputs = user_inputs
                
                # Show success
                st.toast("🎉 Prediction Complete!", icon="✅")
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Prediction Error: {str(e)}")
                if st.checkbox("Show technical details"):
                    st.code(str(e))
    
    # Display predicted price in main section if available
    if 'predicted_price' in st.session_state and st.session_state.predicted_price is not None:
        predicted_price = st.session_state.predicted_price
        
        st.markdown("---")
        st.markdown("### 📊 Predicted Price")
        
        # Display in styled card
        st.markdown(f"""
        <div class="prediction-card">
            <div style="font-size: 1.2rem; margin-bottom: 1rem; color: #6b7280;">Estimated Property Value</div>
            <div class="prediction-price">{format_currency(predicted_price)}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show property details
        if 'user_inputs' in st.session_state:
            user_inputs = st.session_state.user_inputs
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🛏 Bedrooms", user_inputs.get('bedrooms', 'N/A'))
                st.metric("📏 House Size", f"{user_inputs.get('sqft_living', 0):,} sqft")
            
            with col2:
                st.metric("🛁 Bathrooms", user_inputs.get('bathrooms', 'N/A'))
                st.metric("🌳 Lot Size", f"{user_inputs.get('sqft_lot', 0):,} sqft")
            
            with col3:
                st.metric("⭐ Condition", user_inputs.get('condition', 'N/A'))
                st.metric("🏆 Grade", user_inputs.get('grade', 'N/A'))


if __name__ == "__main__":
    main()
