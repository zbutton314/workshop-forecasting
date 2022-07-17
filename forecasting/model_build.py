import json
import pickle
from fbprophet import Prophet
from fbprophet.serialize import model_to_json
import xgboost as xgb
import streamlit as st

from forecasting.utils import Clock


c = Clock()


def train_prophet(train_df, model_location_prophet):
    c.start()
    X = train_df.drop(['ds', 'y'], axis=1)

    model = Prophet(growth='linear', seasonality_mode='multiplicative', weekly_seasonality=True)
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.add_country_holidays(country_name='US')

    for col in X:
        model.add_regressor(col)

    # Fit and predict
    model.fit(train_df)
    with open(model_location_prophet, 'w') as file_out:
        json.dump(model_to_json(model), file_out)

    print(f'Prophet model fitted: {c.stop()} seconds')

    return model


def train_xgb(train_df, model_location_xgb):
    c.start()
    y = train_df['y']
    X = train_df.drop(['ds', 'y'], axis=1)

    model = xgb.XGBRegressor(n_estimators=100, reg_lambda=1, gamma=0, max_depth=3)

    # Fit and predict
    model.fit(X, y)
    with open(model_location_xgb, 'wb') as file_out:
        pickle.dump(model, file_out)

    print(f'XGBoost model fitted: {c.stop()} seconds')

    return model
