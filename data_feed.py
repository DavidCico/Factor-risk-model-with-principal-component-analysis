import yfinance as yf
import pandas as pd


def get_securities_adjusted_prices_from_yahoo(symbol_list, start_date, end_date, interval="1d", to_csv=False):
    """
    Uses the yfinance package (https://pypi.org/project/yfinance/) to fetch the historical prices
    of a list of securities.

    :param symbol_list:
        List of strings containing all securities + factors to fetch

    :param start_date:
        The start date for the historical data in format "%Y-%m-%d"

    :param end_date:
        The ending date for the historical data in format "%Y-%m-%d"

    :param interval:
        Interval between each data point. By default it is set to 1 day.

    :param to_csv:
        Boolean to save or not the resulting dataframe to csv format.

    :return: A dataframe containing the price time series of the closing prices for each security
    """

    securities_df = yf.download(tickers=symbol_list, start=start_date, end=end_date, interval=interval)
    securities_df = securities_df.drop(["Close", "High", "Low", "Open", "Volume"], axis=1)
    securities_df.index = pd.to_datetime(securities_df.index, format='%Y%m%d')
    securities_df.columns = securities_df.columns.get_level_values(1)
    securities_df = securities_df.dropna(axis=1, thresh=0.9 * len(securities_df.index))
    securities_df = securities_df.interpolate(axis=0, method="linear")
    securities_df = securities_df.fillna(method="bfill")

    if to_csv:
        securities_df.to_csv("securities_and_factors.csv")

    return securities_df


def create_sec_and_risk_df(securities_df, factors_symbols):
    """
        Creates 2 dataframes containing respectively securities and factor risk prices

        :param securities_df: pandas.Dataframe
            Dataframe containing prices of all assets

        :param factors_symbols: List of str
            List of strings containing the risk factors symbols

        :return: 2 dataframes
    """

    risk_df = securities_df[factors_symbols]
    risk_df.index = pd.to_datetime(risk_df.index, format='%Y-%m-%d')
    sec_df = securities_df.drop(factors_symbols, axis=1)
    sec_df.index = pd.to_datetime(sec_df.index, format='%Y-%m-%d')

    return sec_df, risk_df


def keep_random_columns_in_df(dataframe, n_to_keep, random_seed=None):
    """
    Function that takes a dataframe of securities as parameter and returns
    a new dataframe after removing n securities

    :param dataframe: pandas.Dataframe
        dataframe containing the prices of different securities

    :param n_to_keep: int
        integer to specify the number of securities to keep from the dataframe

    :param random_seed:
        optional seed for rng

    :return: dataframe
    """

    new_df = dataframe.sample(n=n_to_keep, axis=1, random_state=random_seed)

    return new_df
