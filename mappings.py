import pandas as pd
from portfolio_decomposition import map_series_to_factors


def map_security(security_prices, risk_factor_prices):
    """
    Given a dataframe of security prices, this function pulls down the necessary
    data and maps the data series risk to the selected factors.

    :param security_prices: pandas.Dataframe
        dataframe of prices over time utilized to map risk,
        should utilize split-adjusted prices or total return series if possible.

    :param risk_factor_prices: pandas.Dataframe
        dataframe of risk factor prices

    :return: dict
        Additive component mappings of the specified security to
        mapping factors.
    """

    assert type(security_prices) == pd.DataFrame, 'security_prices must be a pandas DataFrame'

    # Extract key name from dataframe
    key_name = security_prices.keys()[0]

    # Pull down factor total return series.
    risk_factor_prices = pd.DataFrame(risk_factor_prices).dropna()

    # Run mapping
    mapping_res = map_series_to_factors(risk_factor_prices, security_prices, {})
    if type(mapping_res) == str:
        return {"mapping result": mapping_res}
    else:
        m_mapping, map_stats = mapping_res

    # assemble mappings object
    mappings = {key_name: m_mapping.loc[key_name].to_dict()}
    mappings[key_name]['systematic_ratio'] = map_stats.loc[key_name, 'pct_systematic']
    mappings[key_name]['correlation_to_pca'] = map_stats.loc[key_name, 'correl_to_pca']
    mappings[key_name]['correl_to_generic'] = map_stats.loc[key_name, 'correl_to_generic']

    return mappings


def map_multiple_securities(securities_prices, risk_factor_prices):
    """
    Given a dataframe of securities prices, this function pulls down the necessary
    data and maps the data series risk to the selected factors.

    :param securities_prices: pandas.Dataframe
        dataframe of prices over time utilized to map risk,
        should utilize split-adjusted prices or total return series if possible.

    :param risk_factor_prices: pandas.Dataframe
        dataframe of risk factor prices

    :return: dict
        Additive component mappings of the specified security to
        mapping factors.
    """

    # Pull down factor total return series.
    risk_factor_prices = pd.DataFrame(risk_factor_prices)

    mappings = {}
    for i in securities_prices:
        try:
            security = pd.DataFrame(securities_prices[i])
            mappings[i] = map_security(security, risk_factor_prices)[i]
        except RuntimeError:
            mappings[i] = 'Error in mapping process.'

    return mappings
