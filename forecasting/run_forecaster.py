import pandas as pd

from forecasting.config import train_cutoff, model_location_prophet, model_location_xgb, date_string
from forecasting.utils import *
from forecasting.queries import get_ride_austin_data
from forecasting.preprocessing import preprocess_df, prepare_df_msda_workshop, add_endog_vars, train_test
from forecasting.model_build import train_prophet, train_xgb
from forecasting.forecasting import load_prophet, forecast_prophet, load_xgb, forecast_xgb


c = Clock()


if __name__ == '__main__':

    c.start()

    rides_df = get_ride_austin_data(filepath='msda_workshop')
    df = preprocess_df(rides_df)
    df = prepare_df_msda_workshop(df)

    # Build Prophet Model
    prophet_df = add_endog_vars(df)
    prophet_df_train, prophet_df_test = train_test(prophet_df, train_cutoff)
    model_prophet = train_prophet(prophet_df_train, model_location_prophet)
    prophet_df.to_csv(f'data/feature_space/features_prophet_{date_string}.csv', index=False)

    # Forecast with Prophet
    model_prophet = load_prophet(model_location_prophet)
    pred_df_prophet = forecast_prophet(prophet_df_test, model_prophet)
    pred_df_p = prophet_df_test[['ds', 'y']].merge(pred_df_prophet[['ds', 'yhat']], on='ds', how='inner')

    # Build XGBoost Model
    xgb_df = add_endog_vars(df, day_of_week=True)
    xgb_df_train, xgb_df_test = train_test(xgb_df, train_cutoff)
    model_xgb = train_xgb(xgb_df_train, model_location_xgb)
    xgb_df.to_csv(f'data/feature_space/features_xgb_{date_string}.csv', index=False)

    # Forecast with XGBoost
    model_xgb = load_xgb(model_location_xgb)
    pred_xgb = forecast_xgb(xgb_df_test, model_xgb)
    pred_df_x = pd.concat([xgb_df_test[['ds', 'y']].reset_index(drop=True), pred_xgb], axis=1)

    # Combine into single prediction output
    pred_df = pred_df_p.merge(pred_df_x[['ds', 'yhat']], on='ds', how='inner')\
        .rename(columns={'yhat_x': 'pred_prophet', 'yhat_y': 'pred_xgb'})
    pred_df.to_csv(f'data/forecasts/forecasts_{date_string}.csv', index=False)

    print(f'Total Time: {c.stop()} seconds')


    # print(pred_df)
    #
    # display_results(pred_df, 'pred_prophet')
    # display_results(pred_df, 'pred_xgb')
