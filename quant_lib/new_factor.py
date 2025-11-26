from factor import factor
import numpy as np
import pandas as pd
from analysis import *

def ts_to_wm(series, window):
    """
    Apply linear decay weighting to the past 'window' days - Optimized version
    """
    # Define a function to apply to each rolling window
    def weighted_max_div_mean(values):
        # Get the max value in the window
        max_val = np.max(values)
        # Calculate weighted average
        weights = np.arange(1, len(values)+1)
        weights = weights / weights.sum()
        weighted_avg = np.sum(values * weights)
        # Return max divided by weighted average
        return max_val / (weighted_avg + 1e-10)
    
    # print(series)
    
    # Group by InnerCode and apply the rolling calculation
    result = series.rolling(
            window=window, min_periods=int(0.75*window)
        ).apply(weighted_max_div_mean, raw=True)
    
    # Handle the MultiIndex after groupby.apply
    if isinstance(result.index, pd.MultiIndex) and len(result.index.names) > 2:
        result = result.droplevel(0)
    
    return result

class collective(factor):
    # correlation(div(vwap, high), high, 10)
    def set_factor_property(self):
        factor_property={'factor_name': 'my_f','data_needed': ['high','vwap','amount','volume'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        highs = self.data['high']
        amounts = self.data['amount']
        volumes = self.data['volume']
        vwaps = amounts / volumes.where(volumes > 0, np.nan)
        division = vwaps/highs.where(highs > 0, np.nan)
        corr = ts_corr1(division, highs, 10)
        self.factor_data = ts_mean(cs_rank(corr),10)

class another_f(factor):
    def set_factor_property(self):
        factor_property={'factor_name': 'another_f','data_needed': ['adjhigh','turnover_volume'], 'factor_type': 'market'}
        return factor_property
    def calc_factor(self):
        highs = self.data['adjhigh']
        turnover_volumes = self.data['turnover_volume']
        ts_wm_result = ts_to_wm(highs, 20)
        lnto = np.log(turnover_volumes + 1)
        mul = cs_rank(ts_wm_result) * cs_rank(lnto)

        self.factor_data = mul