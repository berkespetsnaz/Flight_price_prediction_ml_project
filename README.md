# Flight Price Prediction

Machine learning models to predict domestic Indian flight prices. Includes data preprocessing pipeline and Flask web interface.

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

Train models by running `flight_price_prediction_final.ipynb`. This generates `rfr.pkl` and `dtr.pkl` files.

Run the web application:
```bash
python main.py
```

Access at `http://127.0.0.1:5000`

## Project Structure

```
├── flight_price_prediction_final.ipynb
├── main.py
├── templates/
├── static/
├── Data_Train.xlsx
└── requirements.txt
```

## Data Pipeline

1. Data cleaning (missing values, duplicates)
2. Feature extraction (date/time components, duration conversion)
3. Encoding (one-hot for source cities, label encoding for airline/destination)
4. Outlier removal (IQR method)
5. Model training (Random Forest, Decision Tree)

## Models

- Random Forest Regressor (100 estimators)
- Decision Tree Regressor

Evaluation metrics: R², MAE, MSE, RMSE, MAPE

## Tech Stack

Python 3.8+, pandas, numpy, scikit-learn, Flask, matplotlib, seaborn

## Routes

**Sources:** Bangalore, Chennai, Delhi, Kolkata, Mumbai  
**Destinations:** Bangalore, Cochin, Delhi, Hyderabad, Kolkata
