# 🌾 Cameroon Food Price Forecasting

## 📌 Project Overview
A comprehensive machine learning system for forecasting food prices across different regions in Cameroon using time series analysis and multiple ML models.

### 🎯 Key Features
- **Data Processing**: Clean and standardize WFP food price data
- **Model Training**: XGBoost, Random Forest, CatBoost, and LSTM models
- **Forecasting**: 6-month price predictions with confidence intervals
- **Web Interface**: Interactive Streamlit dashboard
- **Risk Assessment**: Automated risk level classification

### 📊 Models Used
1. **XGBoost** - Gradient boosting with regularization
2. **Random Forest** - Ensemble decision trees
3. **CatBoost** - Categorical feature handling
4. **LSTM** - Deep learning for sequence prediction

### 🚀 Quick Start

#### 1. Clone Repository
```bash
git clone https://github.com/yourusername/cameroon-food-forecasting.git
cd cameroon-food-forecasting
### 2. Install Dependencies
pip install -r scripts/requirements.txt
### 3. Run Analysis Pipeline
# Open Jupyter Lab
bash
jupyter lab
# Run notebooks:
# 1. FootPredictionModel.ipynb
# 2. Run the codes inorer: 1-4
### 4. Launch Web App
bash
streamlit run scripts/PredictionApp.py
