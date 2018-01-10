from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_json('sample.txt')

#parsing and feature extraction
def parse(data):
    '''
    Parses the DataFrame to extract relevant features. In this case, we use only the "valorToral" and "dhEmi$date"
      as features.
    :param data: a DataFrame
    :return: a groupedDataFrame
    '''
    dates = map(lambda x: datetime.strptime(str(x['dhEmi']['$date']), '%Y-%m-%dT%H:%M:%S.%fZ'), data.ide)
    total = map(lambda x: x['valorTotal'], data.complemento)
    e = pd.DataFrame(data=list(zip(dates, total)), columns=['date', 'total'])  # feature extraction
    e.date = pd.to_datetime(e.date)  # now dates is the index
    e.set_index(e.date, inplace=True)
    g = e.groupby(pd.TimeGrouper('D'))  # group by day
    return g.sum(dropna=True)



def derive_patterns(data, w):
    '''
    Derives patterns from observations. It uses the w param to produced shift series.
    :param data: the observations
    :param w: the number of observations to produce a single pattern.
    :return: an array of pattern with shifted observations.
    '''
    patterns = np.array([(data).shift(-x).values[:w] for x in range(len(data))]).reshape(len(data), -1)
    patterns = np.nan_to_num(np.array(patterns))
    return patterns.reshape(patterns.shape[0], -1)


def train_model(training_samples, training_labels):
    '''
    Trains a regular linear model.
    :param training_samples: the observations.
    :param training_labels: their labels.
    :return: a trained model.
    '''
    regr = LinearRegression()
    regr.fit(training_samples, training_labels)
    return regr


def next_estimations(regr, days, today):
    '''
    Returns the predictions for
    :param regr:
    :param days:
    :param today:
    :return:
    '''
    next_week_estimates = np.zeros((0, 0), dtype=np.float32)
    for i in range(days):
        tomorrow  = regr.predict(today)
        next_week_estimates  = np.append(next_week_estimates, tomorrow)
        today = np.append(tomorrow, today)[:-1].reshape(1, -1)
    return next_week_estimates


def plot_estimations(data, training_values, estimated):
    '''
    Plots the information.
    :param data: the original dataset.
    :param training_values: the prediction for the same values on the original dataset.
    :param estimated: the prediction for the future days.
    :return: None
    '''
    days = np.array(data.index.day)
    fig, ax = plt.subplots(figsize=(8, 3))

    ax.plot(days, np.array(data), 'C1', label='Original data')
    ax.plot(days[-len(training_values):], training_values, 'C2', label='Training data')
    ax.plot(range(days[-1], days[-1] + len(estimated)), estimated, 'C3', label='Estimated data')
    ax.plot(days, np.tile(np.mean(data), (len(days), 1)), 'C4--', label='Sample mean')
    ax.legend(loc='best')

    plt.show()



# parse data
data_parsed = parse(data)

# use 6 days to derive patterns
window_size = 7

training_data = derive_patterns(data_parsed, window_size)

# skip the first samples
training_samples = training_data[:-window_size] #
training_labels  = np.array(data_parsed[window_size:])

# train a model
regr_model = train_model(training_samples, training_labels)

# use the trained model to estimate the next week
last_day = training_samples[-1:].reshape(1, -1)
next_week = next_estimations(regr_model, 7, last_day)

# predicts the original dataset values to notice the error
estimate_from_training_samples = regr_model.predict(training_samples)

# show results
plot_estimations(data_parsed, estimate_from_training_samples, next_week)