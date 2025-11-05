# MLSN Davis House Price Prediction

Machine Learning Student Network collaborative project for predicting house prices in Davis, CA using real estate and economic data.

## 🚀 Quick Start

### Local Development
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Streamlit Cloud Deployment
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository: `KTH-Sys/MLSN_Davis-House-Price-Prediction`
4. Main file: `app.py`
5. Deploy!

## 📋 Requirements
- Python 3.8+
- See `requirements.txt` for dependencies

## 🏗️ Project Structure
```
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── models/
│   └── model.pkl         # Trained XGBoost model
├── data/
│   └── davis_housing_clean_2.csv  # Dataset
├── assets/
│   └── MLSN_logo.jpg     # MLSN logo
└── src/
    └── preprocess.py     # Preprocessing utilities
```

## 🎨 Features
- Modern glassy UI with purple accents
- Dynamic insights & recommendations
- What-if scenario calculations
- Altair comparison charts
- MLSN UC Davis branding

## 📊 Model
- **Type**: XGBoost Regressor
- **Features**: bed, bath, house_size, acre_lot
- **Output**: Predicted house price in USD

## 👥 Built By
**Machine Learning Student Network – UC Davis**  
Advancing applied AI research and education.

## 📝 License
MIT License
