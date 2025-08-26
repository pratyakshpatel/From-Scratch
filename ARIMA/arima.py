import numpy as np
from scipy.optimize import minimize
from scipy.stats import jarque_bera
import warnings

class ARIMA:
    def __init__(self, order=(1, 1, 1)):
        """
        arima model implementation
        order: (p, d, q) where p=ar terms, d=differencing, q=ma terms
        """
        self.p, self.d, self.q = order
        self.params = None
        self.fitted_values = None
        self.residuals = None
        self.original_series = None
        self.differenced_series = None
        self.aic = None
        self.bic = None
        
    def difference(self, series, d):
        """apply differencing d times"""
        diff_series = series.copy()
        for _ in range(d):
            diff_series = np.diff(diff_series)
        return diff_series
    
    def undifference(self, diff_series, original_series, d):
        """reverse differencing to get back to original scale"""
        result = diff_series.copy()
        
        for i in range(d):
            # get the last d values from original series for integration
            last_vals = original_series[-(d-i):] if i < d-1 else [original_series[-1]]
            
            # integrate by cumulative sum
            integrated = np.zeros(len(result) + 1)
            integrated[0] = last_vals[-1]
            integrated[1:] = result
            result = np.cumsum(integrated)[1:]
            
        return result
    
    def likelihood(self, params):
        """compute negative log likelihood for optimization"""
        # extract parameters
        ar_params = params[:self.p] if self.p > 0 else []
        ma_params = params[self.p:self.p+self.q] if self.q > 0 else []
        sigma = params[-1]
        
        if sigma <= 0:
            return 1e10
        
        n = len(self.differenced_series)
        errors = np.zeros(n)
        fitted = np.zeros(n)
        
        # initialize with zeros for ar and ma terms
        ar_buffer = np.zeros(self.p) if self.p > 0 else []
        ma_buffer = np.zeros(self.q) if self.q > 0 else []
        
        for t in range(n):
            # ar component
            ar_component = 0
            if self.p > 0:
                for i in range(min(t, self.p)):
                    ar_component += ar_params[i] * self.differenced_series[t-1-i]
            
            # ma component  
            ma_component = 0
            if self.q > 0:
                for i in range(min(t, self.q)):
                    ma_component += ma_params[i] * errors[t-1-i]
            
            fitted[t] = ar_component + ma_component
            errors[t] = self.differenced_series[t] - fitted[t]
        
        # compute log likelihood
        log_likelihood = -0.5 * n * np.log(2 * np.pi * sigma**2)
        log_likelihood -= 0.5 * np.sum(errors**2) / sigma**2
        
        return -log_likelihood  # return negative for minimization
    
    def fit(self, series):
        """fit arima model to time series"""
        self.original_series = np.array(series)
        
        # apply differencing
        self.differenced_series = self.difference(series, self.d)
        n = len(self.differenced_series)
        
        # initialize parameters
        initial_params = []
        
        # ar parameters (small random values)
        if self.p > 0:
            initial_params.extend(np.random.normal(0, 0.1, self.p))
            
        # ma parameters (small random values)  
        if self.q > 0:
            initial_params.extend(np.random.normal(0, 0.1, self.q))
            
        # sigma parameter (std of differenced series)
        initial_params.append(np.std(self.differenced_series))
        
        # parameter bounds for stability
        bounds = []
        
        # ar bounds (stationary condition: sum < 1)
        if self.p > 0:
            bounds.extend([(-0.99, 0.99)] * self.p)
            
        # ma bounds (invertible condition)
        if self.q > 0:
            bounds.extend([(-0.99, 0.99)] * self.q)
            
        # sigma bound (positive)
        bounds.append((1e-6, None))
        
        # optimize parameters
        try:
            result = minimize(self.likelihood, initial_params, 
                            method='L-BFGS-B', bounds=bounds)
            
            if not result.success:
                warnings.warn("optimization did not converge")
                
            self.params = result.x
            
        except Exception as e:
            raise RuntimeError(f"fitting failed: {e}")
        
        # compute fitted values and residuals
        self._compute_fitted_values()
        self._compute_information_criteria()
        
        return self
    
    def _compute_fitted_values(self):
        """compute fitted values and residuals after fitting"""
        ar_params = self.params[:self.p] if self.p > 0 else []
        ma_params = self.params[self.p:self.p+self.q] if self.q > 0 else []
        
        n = len(self.differenced_series)
        fitted_diff = np.zeros(n)
        errors = np.zeros(n)
        
        for t in range(n):
            # ar component
            ar_component = 0
            if self.p > 0:
                for i in range(min(t, self.p)):
                    ar_component += ar_params[i] * self.differenced_series[t-1-i]
            
            # ma component
            ma_component = 0
            if self.q > 0:
                for i in range(min(t, self.q)):
                    ma_component += ma_params[i] * errors[t-1-i]
            
            fitted_diff[t] = ar_component + ma_component
            errors[t] = self.differenced_series[t] - fitted_diff[t]
        
        # undifference fitted values to original scale
        if self.d > 0:
            self.fitted_values = self.undifference(fitted_diff, self.original_series, self.d)
            # trim to match original series length
            start_idx = len(self.original_series) - len(self.fitted_values)
            if start_idx > 0:
                self.fitted_values = np.concatenate([
                    np.full(start_idx, np.nan), 
                    self.fitted_values
                ])
        else:
            self.fitted_values = fitted_diff
            
        self.residuals = self.original_series - self.fitted_values
        # set residuals to nan where fitted values are nan
        self.residuals = np.where(np.isnan(self.fitted_values), np.nan, self.residuals)
    
    def _compute_information_criteria(self):
        """compute aic and bic"""
        n = len(self.differenced_series)
        k = len(self.params)  # number of parameters
        log_likelihood = -self.likelihood(self.params)
        
        self.aic = 2 * k - 2 * log_likelihood
        self.bic = k * np.log(n) - 2 * log_likelihood
    
    def predict(self, steps=1):
        """forecast future values"""
        if self.params is None:
            raise ValueError("model must be fitted before prediction")
        
        ar_params = self.params[:self.p] if self.p > 0 else []
        ma_params = self.params[self.p:self.p+self.q] if self.q > 0 else []
        
        # use last values for forecasting
        last_values = self.differenced_series[-self.p:] if self.p > 0 else []
        last_errors = self.residuals[-self.q:] if self.q > 0 else []
        last_errors = last_errors[~np.isnan(last_errors)]  # remove nan values
        
        forecasts_diff = []
        
        for step in range(steps):
            # ar component
            ar_component = 0
            if self.p > 0:
                for i in range(min(len(last_values), self.p)):
                    ar_component += ar_params[i] * last_values[-(i+1)]
            
            # ma component (errors become 0 for multi-step ahead)
            ma_component = 0
            if self.q > 0 and step == 0:  # only for one-step ahead
                for i in range(min(len(last_errors), self.q)):
                    ma_component += ma_params[i] * last_errors[-(i+1)]
            
            forecast = ar_component + ma_component
            forecasts_diff.append(forecast)
            
            # update last values for next iteration
            if self.p > 0:
                last_values = np.append(last_values, forecast)[-self.p:]
        
        # undifference forecasts to original scale
        if self.d > 0:
            forecasts = self.undifference(np.array(forecasts_diff), 
                                        self.original_series, self.d)
        else:
            forecasts = np.array(forecasts_diff)
        
        return forecasts
    
    def summary(self):
        """print model summary"""
        if self.params is None:
            print("model not fitted")
            return
        
        print(f"arima({self.p},{self.d},{self.q}) model summary")
        print("=" * 40)
        
        param_names = []
        if self.p > 0:
            param_names.extend([f"ar{i+1}" for i in range(self.p)])
        if self.q > 0:
            param_names.extend([f"ma{i+1}" for i in range(self.q)])
        param_names.append("sigma")
        
        for name, param in zip(param_names, self.params):
            print(f"{name:>8}: {param:8.4f}")
        
        print(f"\naic: {self.aic:.4f}")
        print(f"bic: {self.bic:.4f}")
        
        # residual diagnostics
        valid_residuals = self.residuals[~np.isnan(self.residuals)]
        if len(valid_residuals) > 0:
            jb_stat, jb_pvalue = jarque_bera(valid_residuals)
            print(f"\njarque-bera test (normality): {jb_stat:.4f} (p-value: {jb_pvalue:.4f})")

# example usage
if __name__ == "__main__":
    # generate sample time series
    np.random.seed(42)
    n = 100
    
    # ar(1) process: y_t = 0.7*y_{t-1} + error_t
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.7 * y[t-1] + np.random.normal(0, 1)
    
    # add trend to make it non-stationary
    trend = np.linspace(0, 5, n)
    y_with_trend = y + trend
    
    print("fitting arima(1,1,1) model...")
    model = ARIMA(order=(1, 1, 1))
    model.fit(y_with_trend)
    
    # print summary
    model.summary()
    
    # make forecasts
    forecasts = model.predict(steps=5)
    print(f"\n5-step ahead forecasts: {forecasts}")
