# MLSN Davis House Price Prediction

Machine Learning Student Network collaborative project for predicting house prices in Davis, CA using real estate and economic data.

### Local Development
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Streamlit Cloud Deployment
1. Push code to GitHub
2. Go to [https://share.streamlit.io](https://mlsndavis-house-price-prediction.streamlit.app/)
3. Connect repository: `KTH-Sys/MLSN_Davis-House-Price-Prediction`
4. Main file: `app.py`
5. Deploy!

- See `requirements.txt` for dependencies


### Model
- **Type**: XGBoost Regressor
- **Features**: bed, bath, house_size, acre_lot
- **Output**: Predicted house price in USD

## 👥 Built By
**Machine Learning Student Network – UC Davis**  

## 📝 License
MIT License
