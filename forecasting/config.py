import datetime

train_cutoff = datetime.date(2017, 3, 14)
date_string = datetime.date.today().strftime('%Y%m%d')
model_location_prophet = f'models/prophet_model_{date_string}.json'
model_location_xgb = f'models/xgb_model_{date_string}.pkl'
