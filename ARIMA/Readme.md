# ARIMA Model Implementation

Complete from-scratch ARIMA (AutoRegressive Integrated Moving Average) model for time series forecasting.

## What is ARIMA?

ARIMA(p,d,q) models combine three components:
- **AR(p)**: uses p past values to predict current value
- **I(d)**: applies d-order differencing for stationarity  
- **MA(q)**: uses q past forecast errors for predictions

Mathematical form: `φ(B)(1-B)^d X_t = θ(B)ε_t`

## Features

- parameter estimation via maximum likelihood
- automatic differencing for non-stationary data
- multi-step ahead forecasting
- model diagnostics (AIC, BIC, residual tests)
- stationarity/invertibility constraints
- handles missing values and edge cases

## Data Generation

The example creates synthetic time series:

**step 1: ar(1) process**
```
y_t = 0.7 * y_{t-1} + ε_t, where ε_t ~ N(0,1)
```

**step 2: add trend**
```
final_series = ar(1) process + linear_trend(0 to 5)
```

**characteristics:**
- 100 observations
- non-stationary due to trend (requires d=1)
- autocorrelated with deterministic trend
- reproducible (seed=42)

## Model Components

**differencing**: transforms non-stationary data to stationary, reverses for predictions

**optimization**: mle with l-bfgs-b, bounded parameters for stability

**forecasting**: recursive prediction using fitted ar/ma terms

**diagnostics**: information criteria, jarque-bera normality test

## Usage

```python
# fit arima(1,1,1) model
model = ARIMA(order=(1, 1, 1))
model.fit(time_series_data)

# view results
model.summary()

# forecast 5 steps ahead  
forecasts = model.predict(steps=5)
```

## Output

The model provides:
- fitted parameters (ar coefficients, ma coefficients, error variance)
- goodness of fit (aic, bic)
- residual diagnostics
- multi-step forecasts with proper scaling
