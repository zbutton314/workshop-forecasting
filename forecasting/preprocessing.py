import pandas as pd


def celsius_to_fahrenheit(temp_c):
    temp_f = (temp_c * 1.8) + 32
    return temp_f


def clean_b_string(string):
    start = 2
    end = len(string) - 1
    clean_string = string[start:end]
    return clean_string


def preprocess_df(df_orig):
    df = df_orig.copy()
    df['date'] = pd.to_datetime(df['date']).dt.date
    df['car_category'] = df['car_category'].apply(clean_b_string)
    df['distance'] = df['distance'].replace('', '0').astype(float)
    df['charity_amount'] = df['charity_amount'].replace('', '0').astype(float)
    df['daily_temp_max'] = df['daily_temp_max'].astype(float)
    df['daily_temp_min'] = df['daily_temp_min'].astype(float)
    df['daily_temp_max_f'] = df['daily_temp_max'].apply(celsius_to_fahrenheit)
    df['daily_temp_min_f'] = df['daily_temp_min'].apply(celsius_to_fahrenheit)

    days_df = df.groupby(['date', 'car_category']).agg(
        rides=('date', 'count'),
        distance=('distance', 'sum'),
        charity_amount=('charity_amount', 'sum'),
        visibility=('daily_visibility', 'max'),
        temp_max=('daily_temp_max', 'max'),
        temp_min=('daily_temp_min', 'max'),
        humidity=('daily_humidity', 'max'),
        wind_speed=('daily_wind_speed', 'max'),
        precip=('daily_precip', 'max')
    ).reset_index()

    scaffold = pd.DataFrame(columns=['car_category', 'date'])

    for cat in days_df['car_category'].unique():
        start = days_df.loc[days_df['car_category'] == cat, 'date'].min()
        end = days_df.loc[days_df['car_category'] == cat, 'date'].max()
        scaffold_cat = pd.DataFrame({'car_category': cat, 'date': pd.date_range(start, end)})
        scaffold = pd.concat([scaffold, scaffold_cat]).reset_index(drop=True)

    scaffold['date'] = scaffold['date'].dt.date
    days_df = scaffold.merge(days_df, on=['car_category', 'date'], how='left')

    weather_df = days_df.groupby(['date']).agg(
        visibility=('visibility', 'max'),
        temp_max=('temp_max', 'max'),
        temp_min=('temp_min', 'max'),
        humidity=('humidity', 'max'),
        wind_speed=('wind_speed', 'max'),
        precip=('precip', 'max')
    ).reset_index()

    days_df_final = days_df[['car_category', 'date', 'rides', 'distance', 'charity_amount']]
    days_df_final = days_df_final.merge(weather_df, on='date', how='inner')
    fill_na_cols = ['rides', 'distance', 'charity_amount']
    days_df_final[fill_na_cols] = days_df_final[fill_na_cols].fillna(0)

    return days_df_final


### NOTE:
# This function reduces the data to our chosen target variable and car_category
# Modify/eliminate this function if you decide to experiment with other forecasts
def prepare_df_msda_workshop(df):
    drop_cols = ['car_category', 'distance', 'charity_amount']
    input_df = df[df['car_category'] == 'REGULAR'].rename(columns={'date': 'ds', 'rides': 'y'}).drop(drop_cols, axis=1)

    return input_df


def add_endog_vars(df, day_of_week=False):
    # Simple lag and difference for y
    df['y_lag_1'] = df['y'].shift(1)
    df['y_lag_diff_1'] = df['y_lag_1'].diff(1)

    # Add day of week, if needed
    # Day 0 = Monday
    if day_of_week:
        df['day_of_week'] = pd.to_datetime(df['ds']).dt.dayofweek
        df = pd.get_dummies(df, prefix='day_', columns=['day_of_week'])

    # Remove rows without full data
    df = df.dropna(0).reset_index(drop=True)

    return df


def train_test(df, train_cutoff):
    idx = df[df['ds'] == train_cutoff].index[0]
    train, test = df[0:idx], df[idx:]
    return train, test
