# Time Series Notes

## Components of Time Series

```Python
series = trend + seasons + cycles + error
```

## Features

### Time Step Features

Derived from the time stamp.
```Python
# create a time dummy
df['Time'] = np.arange(len(df.index))

# better method: with DeterministicProcess method
from statsmodels.tsa.deterministic import DeterministicProcess
dp = DeterministicProcess(
    index=tunnel.index,  # dates from the training data
    constant=True,       # dummy feature for the bias (y_intercept)
    order=1,             # the time dummy (trend),
    drop=True,           # drop terms if necessary to avoid collinearity
)

# the fit intercept in linear Regression later with df should be turned off
model = LinearRegression(fit_intercept=False)
model.fit(X, y)

# `in_sample` creates features for the dates given in the `index` argument
X = dp.in_sample()
```

Let tou model time dependence.

### Lag Features

Shift the observations of the target series so that they appear to have occured later in time.

```Python
# create a lag1
lag_1 = df['sales'].shift(1)
df['lag_1'] = lag_1
```

Let you model serial dependence.

### Trend
For a change to be a part of the trend, it should occur over a longer period than any seasonal changes. 
To visualize a trend, therefore, we take an average over a period longer than any seasonal period in the series. 

We can use moving average plots to visualize the trend.

```Python
# use .rolling method to create a moving average
## tunnel is the df containing our data
moving_average = tunnel.rolling(
    window=365,       # 365-day window
    center=True,      # puts the average at the center of the window
    min_periods=183,  # choose about half the window size
).mean()              # compute the mean (could also do median, std, min, max, ...)

ax = tunnel.plot(style=".", color="0.5")
moving_average.plot(
    ax=ax, linewidth=3, title="Tunnel Traffic - 365-Day Moving Average", legend=False,
);
```

To make a prediction with Time series, we can use `DeterminisicProcess` out of sample method that creates time steps after 
observation window.

```Python
X = dp.out_of_sample(steps=30) # predict 30 time steps
y_fore = pd.Series(model.predict(X), index=X.index)
y_fore.head()
```

### Seasonality

Use **seasonal plot** to see seasonal patterns: A seasonal plot shows segments of the time series plotted against some common period, 
the period being the "season" you want to observe. 

We can use one-hot encoding to encode the seasonal indicators, serving as ON/OFF switches to add to the data starting from the first indicator.

In large observations where seasonal indicators are impossible, we have **Fourier features** (pairs of sine and cosine values), that tries to capture the overall shape of the seasonal curve with just a few features.

It is these frequencies within a season that we attempt to capture with Fourier features. 
The idea is to include in our training data periodic curves having the same frequencies as the season we are trying to model. 
The curves we use are those of the trigonometric functions sine and cosine.

We choose Fourier features with the Periodogram.

A sample of computing Fourier features.

```Python
import numpy as np


def fourier_features(index, freq, order):
    time = np.arange(len(index), dtype=np.float32)
    k = 2 * np.pi * (1 / freq) * time
    features = {}
    for i in range(1, order + 1):
        features.update({
            f"sin_{freq}_{i}": np.sin(i * k),
            f"cos_{freq}_{i}": np.cos(i * k),
        })
    return pd.DataFrame(features, index=index)


# Compute Fourier features to the 4th order (8 new features) for a
# series y with daily observations and annual seasonality:
#
# fourier_features(y, freq=365.25, order=4)
```

Use Fourier features to encode DP:
```Python
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

fourier = CalendarFourier(freq="A", order=10)  # 10 sin/cos pairs for "A"nnual seasonality

dp = DeterministicProcess(
    index=tunnel.index,
    constant=True,               # dummy feature for bias (y-intercept)
    order=1,                     # trend (order 1 means linear)
    seasonal=True,               # weekly seasonality (indicators)
    additional_terms=[fourier],  # annual seasonality (fourier)
    drop=True,                   # drop terms to avoid collinearity
)

X = dp.in_sample()  # create features for dates in tunnel.index

# build the model and train
y = tunnel["NumVehicles"]

model = LinearRegression(fit_intercept=False)
_ = model.fit(X, y)

y_pred = pd.Series(model.predict(X), index=y.index)
X_fore = dp.out_of_sample(steps=90)
y_fore = pd.Series(model.predict(X_fore), index=X_fore.index)

ax = y.plot(color='0.25', style='.', title="Tunnel Traffic - Seasonal Forecast")
ax = y_pred.plot(ax=ax, label="Seasonal")
ax = y_fore.plot(ax=ax, label="Seasonal Forecast", color='C3')
_ = ax.legend()
```

### Time Series as Features

Cycle is one of the ways to make serial dependence appear. Cycles are patterns of growth and decay in a time series associated 
with how the value in a series at one time depends on values at previous times, but not necessarily on the time step itself.

Difference of cycles from seasonality: cycle doesn't necessarily depend on time, it is more about what happened in the past.
Can be much more irregular than seasonality.

Create lagged copies of the time series.

**Auto-correlation**: measure used for serial dependence. The **partial autocorrelation** tells you the correlation of a lag accounting for all of the previous lags 
-- the amount of "new" correlation the lag contributes, so to speak. 

They are all measures of **linear** dependence. For more realistic data, it is better to look at lag plot.

```Python
from statsmodels.graphics.tsaplots import plot_pacf # plot partial auto correlation
# tool functions from kaggle
def lagplot(x, y=None, lag=1, standardize=False, ax=None, **kwargs):
    from matplotlib.offsetbox import AnchoredText
    x_ = x.shift(lag)
    if standardize:
        x_ = (x_ - x_.mean()) / x_.std()
    if y is not None:
        y_ = (y - y.mean()) / y.std() if standardize else y
    else:
        y_ = x
    corr = y_.corr(x_)
    if ax is None:
        fig, ax = plt.subplots()
    scatter_kws = dict(
        alpha=0.75,
        s=3,
    )
    line_kws = dict(color='C3', )
    ax = sns.regplot(x=x_,
                     y=y_,
                     scatter_kws=scatter_kws,
                     line_kws=line_kws,
                     lowess=True,
                     ax=ax,
                     **kwargs)
    at = AnchoredText(
        f"{corr:.2f}",
        prop=dict(size="large"),
        frameon=True,
        loc="upper left",
    )
    at.patch.set_boxstyle("square, pad=0.0")
    ax.add_artist(at)
    ax.set(title=f"Lag {lag}", xlabel=x_.name, ylabel=y_.name)
    return ax


def plot_lags(x, y=None, lags=6, nrows=1, lagplot_kwargs={}, **kwargs):
    import math
    kwargs.setdefault('nrows', nrows)
    kwargs.setdefault('ncols', math.ceil(lags / nrows))
    kwargs.setdefault('figsize', (kwargs['ncols'] * 2, nrows * 2 + 0.5))
    fig, axs = plt.subplots(sharex=True, sharey=True, squeeze=False, **kwargs)
    for ax, k in zip(fig.get_axes(), range(kwargs['nrows'] * kwargs['ncols'])):
        if k + 1 <= lags:
            ax = lagplot(x, y, lag=k + 1, ax=ax, **lagplot_kwargs)
            ax.set_title(f"Lag {k + 1}", fontdict=dict(fontsize=14))
            ax.set(xlabel="", ylabel="")
        else:
            ax.axis('off')
    plt.setp(axs[-1, :], xlabel=x.name)
    plt.setp(axs[:, 0], ylabel=y.name if y is not None else x.name)
    fig.tight_layout(w_pad=0.1, h_pad=0.1)
    return fig

# usage:
_ = plot_lags(flu_trends.FluVisits, lags=12, nrows=2)
_ = plot_pacf(flu_trends.FluVisits, lags=12)
```
Using lags to predict data:

```Python
# make lags as features
def make_lags(ts, lags):
    return pd.concat(
        {
            f'y_lag_{i}': ts.shift(i)
            for i in range(1, lags + 1)
        },
        axis=1)


X = make_lags(flu_trends.FluVisits, lags=4)
X = X.fillna(0.0)

# Create target series and data splits
y = flu_trends.FluVisits.copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=60, shuffle=False)

# Fit and predict
model = LinearRegression()  # `fit_intercept=True` since we didn't use DeterministicProcess
model.fit(X_train, y_train)
y_pred = pd.Series(model.predict(X_train), index=y_train.index)
y_fore = pd.Series(model.predict(X_test), index=y_test.index)
```

But usually with this model, it needs a time step to react to sudden changes in the target series. 
This is a common limitation of models using only lags of the target series as features.

We can solve this problem by having a "leading indicator". Such as using other data to forecast.

```Python
# using google search data
search_terms = ["FluContagious", "FluCough", "FluFever", "InfluenzaA", "TreatFlu", "IHaveTheFlu", "OverTheCounterFlu", "HowLongFlu"]

# Create three lags for each search term
X0 = make_lags(flu_trends[search_terms], lags=3)
X0.columns = [' '.join(col).strip() for col in X0.columns.values]

# Create four lags for the target, as before
X1 = make_lags(flu_trends['FluVisits'], lags=4)

# Combine to create the training data
X = pd.concat([X0, X1], axis=1).fillna(0.0)
```

Trend and seasonality will all show serial dependence. So we first need to deseasonalize the time series.

## Using hybrid model to predict time series

```Python
# 1. Train and predict with first model
model_1.fit(X_train_1, y_train)
y_pred_1 = model_1.predict(X_train)

# 2. Train and predict with second model on residuals
model_2.fit(X_train_2, y_train - y_pred_1)
y_pred_2 = model_2.predict(X_train_2)

# 3. Add to get overall predictions
y_pred = y_pred_1 + y_pred_2
```

The training features are usually different. For example, if the first model learns a trend, then 
the second training set does not need the trend feature.

- Feature transforming algorithms (can extrapolate target values): learn some mathematical function that takes features as an input and then combines and transforms them to produce an output that matches the target values in the training set. 
Linear regression and neural nets are of this kind.

- Target transforming algorithms: use the features to group the target values in the training set and make predictions by averaging values in a group; a set of feature just indicates which group to average. 
Decision trees and nearest neighbors are of this kind.

This difference is what motivates the hybrid design in this lesson: use linear regression to extrapolate the trend, 
transform the target to remove the trend, 
and apply XGBoost to the detrended residuals. 

To hybridize a neural net (a feature transformer), you could instead include the predictions of another model as a feature, 
which the neural net would then include as part of its own predictions.

## Forecasting with Machine Learning

- Forcast origin: time at which you are making a forecast

- Forcast horizon: the time for which you are making a forecast. We often describe a forecast by the number of time steps in its horizon: a "1-step" forecast or "5-step" forecast

- Lead time (latency): The time between the origin and the horizon

### Multistep Forecasting Strategies

- Multioutput model: use a model that can output multiple outputs, like linear regression and neural networks, but not xgboost

- Direct strategy:  one model forecasts 1-step ahead, another 2-steps ahead, and so on. Forecasting 1-step ahead is a different problem than 2-steps ahead (and so on).

- Recursive strategy: Train a single one-step model and use its forecasts to update the lag features for the next step.

- DirRec strategy: A combination of the direct and recursive strategies: train a model for each step and use forecasts from previous steps as new lag features.

Sample code of making 8 week prediction (horizon=8 weeks) with lead=1 (starting from the next week)
```Python
def make_lags(ts, lags, lead_time=1):
    return pd.concat(
        {
            f'y_lag_{i}': ts.shift(i)
            for i in range(lead_time, lags + lead_time)
        },
        axis=1)


# Four weeks of lag features
y = flu_trends.FluVisits.copy()
X = make_lags(y, lags=4).fillna(0.0)


def make_multistep_target(ts, steps):
    return pd.concat(
        {f'y_step_{i + 1}': ts.shift(-i)
         for i in range(steps)},
        axis=1)


# Eight-week forecast
y = make_multistep_target(y, steps=8).dropna()

# Shifting has created indexes that don't match. Only keep times for
# which we have both targets and features.
y, X = y.align(X, join='inner', axis=0)
```

## AR, MA, ARMA and ARIMA

AR(p): ACF tails off. Use PACF to determine the order p.
MA(q): PACF tails off. Use ACF to determine the order q.

### ARMA

**Ljung Box test**: The Ljung-Box test is a classical hypothesis test that is designed to test whether a set of autocorrelations of a fitted time series model differ significantly from zero.
The Ljung-Box test is a classical hypothesis test that is designed to test whether a set of autocorrelations of a fitted time series model differ significantly from zero.