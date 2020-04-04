import random
import pandas as pd
from data_feed import get_securities_adjusted_prices_from_yahoo
from data_feed import keep_random_columns_in_df
from main import start_date, end_date

n_to_keep = 100
random_seed = 1

"""
with open("snp_500_symbols_list.txt") as f:
    snp_500_symbols_list = [word.strip(",'") for line in f for word in line.split()]

snp_securities_df = get_securities_prices_from_yahoo(snp_500_symbols_list, start_date, end_date, to_csv=False)
snp_securities_df.to_csv("snp_500_securities.csv")
"""

snp_securities_df = pd.read_csv("snp_500_securities.csv", index_col=0)
snp_securities_df_smaller = keep_random_columns_in_df(snp_securities_df, n_to_keep, 1)

