import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


st.set_option('deprecation.showPyplotGlobalUse', False)


@st.cache
def load_forecasts():
    path = 'data/forecasts/'
    data_file_list = os.listdir(path)
    filename = sorted(data_file_list, reverse=True)[0]
    latest_forecasts_location = path + filename
    df = pd.read_csv(latest_forecasts_location)
    return df


def root_mean_squared_error(y_true, y_pred):
    result = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return np.round(result, 3)


def mean_forecast_error(y_true, y_pred):
    result = np.mean(y_true - y_pred)
    return np.round(result, 20)


def mean_absolute_error(y_true, y_pred):
    result = np.mean(np.abs(y_true - y_pred))
    return np.round(result, 3)


def mean_absolute_percentage_error(y_true, y_pred, zero_method='adjust', adj=0.1):
    if zero_method == 'adjust':
        y_true_adj = y_true.copy()
        y_true_adj[y_true_adj == 0] = adj
        result = np.mean(np.abs((y_true_adj - y_pred) / y_true_adj)) * 100

    elif zero_method == 'error':
        if len(y_true[y_true == 0]) > 0:
            raise ValueError('Input y_true array contains a zero.')
        else:
            result = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    elif zero_method == 'ignore':
        y_true_ign = y_true.copy()
        y_true_ign = y_true_ign[y_true_ign != 0]
        result = np.mean(np.abs((y_true_ign - y_pred) / y_true_ign)) * 100

    else:
        raise ValueError("Invalid zero_method value - must be 'adjust', 'error', or 'ignore'.")

    return np.round(result, 3)


def plot_results(df, pred_col):
    # Plot results
    x = df['ds']
    y = df['y']
    y_pred = df[pred_col]

    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.plot(y_pred, 'r--', alpha=0.8)
    x_locs = range(0, df.shape[0], 7)
    x_labels = pd.to_datetime(x[x_locs]).dt.strftime('%b %d')
    ax.set_xticks(x_locs)
    ax.set_xticklabels(x_labels)
    ax.legend(['Actual', 'Predicted'])
    if pred_col == 'pred_prophet':
        title = 'Ride Austin - Prophet Forecast'
    elif pred_col == 'pred_xgb':
        title = 'Ride Austin - XGBoost Forecast'
    ax.set_title(title)
    fig_show = fig.show()

    return fig_show


def calculate_metrics(df, pred_col):
    mape = mean_absolute_percentage_error(df['y'], df[pred_col])
    rmse = root_mean_squared_error(df['y'], df[pred_col])
    mae = mean_absolute_error(df['y'], df[pred_col])
    mfe = mean_forecast_error(df['y'], df[pred_col])
    metrics_df = pd.DataFrame({'mape': mape, 'rmse': rmse, 'mae': mae, 'mfe': mfe}, index=[0])
    return metrics_df


# Run Streamlit app
pred_df = load_forecasts()
st.title('Ride Austin Forecast')

# Print Prophet Metrics
st.subheader('Prophet Forecast Accuracy')
st.write(calculate_metrics(pred_df, 'pred_prophet'))

# Plot Prophet Forecast
fig_prophet = plot_results(pred_df, 'pred_prophet')
st.pyplot(fig_prophet)

# Print XGBoost Metrics
st.subheader('XGBoost Forecast Accuracy')
st.write(calculate_metrics(pred_df, 'pred_xgb'))

# Plot XGBoost Forecast
fig_xgb = plot_results(pred_df, 'pred_xgb')
st.pyplot(fig_xgb)

# Optionally show raw data
if st.checkbox('Show raw forecast data'):
    st.subheader('Forecasts')
    st.dataframe(pred_df, width=900, height=900)
