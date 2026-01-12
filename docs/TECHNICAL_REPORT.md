Technical Report: Cameroon Food Price Forecasting

Executive Summary
This project develops a machine learning system to forecast food prices in Cameroon...

Methodology

1.  Data Collection & Preprocessing

- Source: World Food Programme (WFP) monthly price data
- Period: 2005-2023
- Regions: 10 administrative regions
- Commodities: 15+ food items

2.  Feature Engineering

- Time-based features (lags 1, 3, 6, 12 months)
- Rolling statistics (mean, std, volatility)
- Seasonal indicators (month, quarter)
- Price change metrics

3.  Model Architecture

    **XGBoost Parameters:**

```python
{
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}
```

4. Evaluation Metrics
   RMSE: Root Mean Square Error

MAE: Mean Absolute Error

SMAPE: Symmetric Mean Absolute Percentage Error

MAPE: Mean Absolute Percentage Error

Results & Analysis
Performance by Region
Region Best Model SMAPE Volatility
Extrême-Nord XGBoost 8.2% High
Nord CatBoost 9.5% Medium
Centre Random Forest 11.3% Low
Key Insights
Staple commodities show seasonal patterns

Import-dependent items have higher volatility

Urban markets are more predictable

Climate events significantly impact prices

Limitations & Future Work
Limitations:

Limited external data integration

Manual feature engineering

Model retraining frequency

Future Improvements:

Real-time data streaming

Ensemble methods

External factor integration

Automated retraining pipeline
