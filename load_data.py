import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import scipy.stats as stats
import math
from statsmodels.tsa.stattools import grangercausalitytests

def get_data(tickers, start_date, end_date):

    """
    This function takes a list of stock tickers and return a data frame with features:
    - start date: yyyy-mm-dd
    - end date: yyyy-mm-dd
    - tickers: list of tickers

    Features:
    - Adj Closing Price and lagged Adj Closing Price to get Return (in percent)
    - High and Low because traders might set Sell or Buy commands
    - Rolling VaR(5) of a rolling week (5 trading days)
    - target: summed Rolling VaR(5) of a rolling week (5 trading days)

    We may change the features later
    """
    # initialise list to store ticker data farmes
    dfs = []

    for ticker in tickers:
        # Download the stock price data with yfinance
        data = yf.download(ticker, start=start_date, end=end_date)
    
        # Create a new data frame with the necessary columns
        df = pd.DataFrame(index=data.index)
        df["ticker"] = ticker
        df["close"] = data["Adj Close"]
        df["close_lag"] = data["Adj Close"].shift(1)
        df["return"] = ((df["close"] / df["close_lag"]) - 1)*100
        df["volume"] = data["Volume"]
        df["high"] = data["High"]
        df["low"] = data["Low"]
        df["rolling_var_5"] = df["return"].rolling(5, min_periods=1).quantile(0.05)
        # target is the cummulative 
        df["target"] = df["rolling_var_5"].rolling(5, min_periods=5).sum()
    
        # Append the data frame to the list
        dfs.append(df)

    # Concatenate the data frames vertically
    result = pd.concat(dfs)
    result = result.dropna()

    return result





def granger_causality(df, target):
     """
     Computes Granger causality between a target series and multiple control series in a data frame
     Goal: Identify good predictor series for a Ticker
    
     Parameters:
        df: data frame with variables of interest 
        target: name of the target series in the data frame
        
        
     Returns:
        dict: a dictionary with the results of the Granger causality tests
     """

     gc_result = {}

     covariates = df.columns.tolist()
     covariates.pop(0)

     for i in covariates:
        # Drop any missing values
        data_subset = df[[str(target), i]]
        # Perform the Granger causality test
        result = grangercausalitytests(data_subset, maxlag=10, verbose=True)
        gc_result[i] = result
        #gc_result[value] = result
        
     return result

