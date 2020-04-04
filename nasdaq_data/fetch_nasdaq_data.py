import random
import pandas as pd
from data_feed import get_securities_adjusted_prices_from_yahoo
from data_feed import keep_random_columns_in_df
from main import start_date, end_date

n_to_keep = 100
random_seed = 1


with open("nasdaq_securities.txt") as f:
    next(f)  # to skip header row
    nasdaq_symbols_list = [line.split("|")[0] for line in f]

snp_securities_df = get_securities_adjusted_prices_from_yahoo(nasdaq_symbols_list, start_date, end_date, to_csv=False)
snp_securities_df.to_csv("nasdaq_securities.csv")


# nasdaq_securities_df = pd.read_csv("nasdaq_securities.csv", index_col=0)
# nasdaq_securities_df_smaller = keep_random_columns_in_df(nasdaq_securities_df, n_to_keep, 1)
