import pandas as pd
from data_feed import get_securities_adjusted_prices_from_yahoo
from data_feed import create_sec_and_risk_df
from mappings import map_multiple_securities
from os import path

"""
FACTORS_SAMPLE_BUCKETS = {
    'VTI': 'US Equities',
    '^NDX': Nasdaq 100 Index'
    'IVV': 'S&P 500 Equities'
    'VGK': 'EU Equities',
    '^STOXX50E': 'Eurostoxx 50 Index'
    'VPL': 'APAC Equities',
    '^HSI': 'Hang Seng Index' 
    'IEF': 'US 10Y Treasuries',
    'SHY': 'Short Term US Treasuries',
    'VIXY': 'Short Term Futures Vol Index'
    'LQD': 'US High Grade Credit',
    'HYG': 'US High Yield Credit',
    'IBND': 'International Corporate Bonds',
    'TIP': 'US Treasury Inflation-Protected Securities'
    }
"""

factors_symbols = ['^NDX', '^STOXX50E', '^HSI', 'IEF', 'SHY', 'VIXY', 'LQD', 'HYG', 'IBND', 'TIP']
securities_symbols = ['AAPL', 'MSFT', 'AMZN', 'BRK-A', 'AC.PA', 'AIR.PA', 'DAL', 'MAR', 'DIS', 'SIX', 'VOW3.DE',
                      '0700.HK', '2318.HK', 'BABA', '0005.HK', '1299.HK', 'ENI.MI', 'ALV.DE']

securities_symbols.extend(factors_symbols)
start_date = "2015-01-01"
end_date = "2020-01-01"
read_csv = True

if __name__ == "__main__":
    if read_csv and path.exists("securities_and_factors.csv"):
        securities_df = pd.read_csv("securities_and_factors.csv", index_col=0)
    else:
        securities_df = get_securities_adjusted_prices_from_yahoo(securities_symbols, start_date, end_date, to_csv=True)

    sec_df, risk_df = create_sec_and_risk_df(securities_df, factors_symbols)

    pca_mappings = map_multiple_securities(sec_df, risk_df)
    pd.DataFrame(pca_mappings).to_csv("pcaMappingsResults.csv")
