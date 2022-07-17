import numpy as np
import time
import matplotlib.pyplot as plt


class Clock:
    """The clock object stores start times, allowing the tracking of compute time for multiple processes.
    """

    def __init__(self):
        """Constructor method
        """
        self.start_times = {'origin': time.time()}

    def elapsed_time(self):
        """Compute the elapsed time since Clock was initiated

        :return: Number of seconds elapsed since Clock object was instantiated
        :rtype: float
        """
        return round(time.time() - self.start_times['origin'], 5)

    def start(self, clock_id):
        """Start the clock for specified clock_id by recording current time in dictionary

        :param clock_id: Unique ID for each timing process, which will be tracked separately from other IDs
        :type clock_id: any
        """
        self.start_times[clock_id] = time.time()

    def stop(self, clock_id):
        """Stop the clock for specified clock_id by calculating elapsed time and printing

        :param clock_id: Unique ID for each timing process, which will be tracked separately from other IDs
        :type clock_id: any
        :return: Elapsed time since start time for clock_id (info message if clock_id has not been started)
        :rtype: float (str if clock_id has not been started)
        """
        start_time = self.start_times.get(clock_id)
        if start_time is None:
            return f'Clock {clock_id} not started'
        else:
            return round(time.time() - start_time, 5)


def disp(df, lines=5, type='head'):
    print(df.shape)
    if type == 'head':
        print(df.head(lines))
    elif type == 'tail':
        print(df.tail(lines))
    else:
        print("Error: Please enter either 'head' or 'tail'.")


def display_results(pred_df, pred_col):
    # Plot results
    plt.plot(pred_df['y'])
    plt.plot(pred_df[pred_col], 'r--', alpha=0.8)
    plt.legend(['Actual', 'Predicted'])
    plt.title('Ride Austin Forecast')
    plt.show()

    # Accuracy
    metrics_dict = {}
    metrics_dict['mape'] = mean_absolute_percentage_error(pred_df['y'], pred_df[pred_col])
    metrics_dict['rmse'] = root_mean_squared_error(pred_df['y'], pred_df[pred_col])
    metrics_dict['mae'] = mean_absolute_error(pred_df['y'], pred_df[pred_col])
    metrics_dict['mfe'] = mean_forecast_error(pred_df['y'], pred_df[pred_col])

    print(pred_col)
    print(metrics_dict)


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
