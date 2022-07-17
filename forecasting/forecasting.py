import pandas as pd
import json
import pickle
from fbprophet.serialize import model_from_json


def load_prophet(model_location):
    with open(model_location, 'r') as file_in:
        model = model_from_json(json.load(file_in))

    return model


def load_xgb(model_location):
    with open(model_location, 'rb') as file_in:
        model = pickle.load(file_in)

    return model


def forecast_prophet(fcst_df, model):
    pred_df = model.predict(fcst_df.drop('y', axis=1))
    pred_df['ds'] = pred_df['ds'].dt.date

    return pred_df


def forecast_xgb(fcst_df, model):
    pred = model.predict(fcst_df.drop(['ds', 'y'], axis=1))
    pred_df = pd.DataFrame({'yhat': pred})

    return pred_df
