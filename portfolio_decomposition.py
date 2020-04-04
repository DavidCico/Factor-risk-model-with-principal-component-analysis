import pandas as pd
import numpy as np
import datetime

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale
from statsmodels.formula.api import ols

from var_exp_weighted import var_expw

# half life for exponentially weighted var
VAR_HALF_LIFE = 1.4


def map_series_to_factors(risk_factor_prices, security_prices, sec_other=None):
    """
    Uses PCA to decompose the risk underlying the position price series
    into those of the risk factor price series.

    :param risk_factor_prices: pandas.Dataframe
        Price series of known risk factors.

    :param security_prices: pandas.Dataframe
        Price series of securities that will be mapped
        to the risk factor series.

    :param sec_other: dict or pandas.Series
        dict specifying if a security is a Money Market fund.
        {'security_1': 'Money Market', 'security_2': 'Money Market'}

    :return: tuple
        First object contains decomposition of security risk
        into factor risk. Second object contains statistics on those mappings,
        including how much of the security risk is described by the mapped
        factors.
    """

    if sec_other is None:
        sec_other = {}

    assert type(risk_factor_prices) == pd.DataFrame, 'risk_factor_prices must be a pandas DataFrame'
    assert type(security_prices) == pd.DataFrame, 'security_prices must be a pandas DataFrame'
    assert type(sec_other) in [dict, pd.Series], 'sec_other must either be a dict or a series'

    # Create the PCA vectors and mappings to generics
    scale_pca = False
    pca_prices, generic_pca_mappings, scalars = create_pca_components(
        risk_factor_prices,
        days=1,
        scale_pca=scale_pca)

    # Map instruments
    days = 1
    mappings = pd.DataFrame(columns=pca_prices.columns)
    map_stats = pd.DataFrame()
    drop_correl = pd.DataFrame()

    for t in security_prices.columns:
        mapping, correl, d_correl, pct_systematic = map_security(security_prices[t],
                                                                 pca_prices,
                                                                 days=days,
                                                                 drop_outliers=True)
        mappings.loc[t] = mapping
        drop_correl[t] = d_correl.values()
        map_stats.loc[t, 'correl_to_pca'] = correl
        map_stats.loc[t, 'pct_systematic'] = pct_systematic

    asset_std = security_prices.std(axis=0).astype(np.double).round(5)

    for asset in asset_std.keys():
        if (asset_std[asset] == 0) or (sec_other.get(asset, '') == 'Money Market'):
            mappings.loc[asset] = 0.
            map_stats.loc[asset, ['pct_systematic', 'correl_to_pca']] = 1.

    failed_mappings = list(mappings[mappings.iloc[:, 1].isnull()].index)
    if len(failed_mappings) > 0:
        return str('Failure:  - Mappings failed for ' + ', '.join(failed_mappings) + '.')

    m_inst = mappings.dot(generic_pca_mappings.T) / scalars.T
    m_inst = pd.DataFrame(np.array(m_inst).astype(np.double).round(4), columns=m_inst.columns, index=m_inst.index)
    m_mapping = m_inst  # (position_value * m_inst.T).T

    # Calculate statistics for each security & mapping (volatility, VaR, goodness of fit)
    map_stats = calc_map_stats(map_stats, m_mapping, mappings, security_prices,
                               risk_factor_prices, pca_prices, days)

    m_inst.loc['CUR:USD'] = 0.
    for t in security_prices.columns:
        if sec_other.get(t, '') == 'Money Market' and 'Money Market' not in m_inst.columns:
            m_inst['Money Market'] = 0.

    return m_mapping, map_stats


def map_security(position_prices, pca_prices, days=1, drop_outliers=False):
    """
    Maps the risk underlying the provided position prices to the input set
    of PCA component vectors. The mapping function will automatically drop
    exposures to PCA components that do not contribute to the position.

    :param position_prices: pandas.Series
        A price series for the security that is to
        be mapped.

    :param pca_prices: pandas.Dataframe
        A dataframe of PCA component price series. Percent
        return components must be converted to total return series before being
        passed into the function.

    :param days: int
        Step size for generating returns from price series.
        Default is 1 day.

    :param drop_outliers:
        Boolean to drop outliers from ordinary least square regression

    :return: Tuple of
        coefs - mapping of PCA components to input series risk
        correl - correlation of PCA risk series to actual series
        drop_correl - stats on how much dropping some PCA components affects
            final mapping correlation
        pct_systematic - how much of position risk is accounted for by the
            factor mappings.
    """

    date_set = position_prices.dropna().index.intersection(pca_prices.index)
    position_prices = position_prices.loc[date_set]
    pca_prices = pca_prices.loc[date_set]
    y_unadj = position_prices.pct_change(periods=days)[days:]
    x = pca_prices.pct_change(periods=days)[days:]

    y = np.array(y_unadj)
    x = np.array(x)

    if drop_outliers:
        outlier_regression = ols("data ~ x", data=dict(data=y, x=x)).fit()
        test_outliers = outlier_regression.outlier_test()
        non_outliers = [i for i, r in test_outliers.iterrows() if r['bonf(p)'] > 0.5]
        y = y[non_outliers]
        x = x[non_outliers]

    clf = LinearRegression()
    clf.fit(x, y)
    coefs = pd.Series(clf.coef_, index=pca_prices.columns)
    res_series = (np.array(coefs) * x).sum(axis=1)
    correl = np.corrcoef(res_series, y)[1, 0]

    drop = 1
    correl_giveup = 0.
    drop_correl = {0: correl}

    while (correl_giveup <= 0.03) and (drop < x.shape[1]):
        clf.fit(x[:, :-drop], y)
        coefs = pd.Series(clf.coef_, index=pca_prices.columns[:-drop])
        res_series = (np.array(coefs) * x[:, :-drop]).sum(axis=1)
        correl_giveup = correl - np.corrcoef(res_series, y)[1, 0]
        drop += 1

    drop -= 2

    if (drop == 0) or (correl > 0.99):
        clf.fit(x, y)
        coefs = pd.Series(clf.coef_, index=pca_prices.columns)
    else:
        clf.fit(x[:, :-drop], y)
        coefs = pd.Series(clf.coef_, index=pca_prices.columns[:-drop])
        coefs = coefs.append(pd.Series(index=pca_prices.columns[-drop:]).fillna(0))

    res_series = (np.array(coefs) * x).sum(axis=1)
    pct_systematic = min(res_series.std() / y_unadj.std(), 1.)
    # coefs = coefs * (y.std() / res_series.std())
    # res_series = (np.array(coefs)*x).sum(axis=1)
    correl = np.corrcoef(res_series, y)[1, 0]

    # For testing, display how much of fit is given up by dropping PCA components
    for i in range(1, len(coefs) - 1):
        drop = i
        clf2 = LinearRegression()
        clf2.fit(x[:, :-drop], y)
        coefs2 = pd.Series(clf2.coef_, index=pca_prices.columns[:-drop])
        res_series2 = (np.array(coefs2) * x[:, :-drop]).sum(axis=1)
        coefs2 = coefs2 * (y.std() / res_series2.std())
        res_series2 = (np.array(coefs2) * x[:, :-drop]).sum(axis=1)
        correl2 = np.corrcoef(res_series2, y)[1, 0]
        drop_correl[i] = correl2

    return coefs, correl, drop_correl, pct_systematic


def calc_map_stats(map_stats, m_mapping, mappings, position_prices, generic_prices, pca_prices, days):
    """
    Calculates the risk and quality of mapping measures for the PCA decomposition series.
    :param map_stats:
        Stats already calculated during the mapping

    :param m_mapping:
    :param mappings:
    :param position_prices:
    :param generic_prices:
    :param pca_prices:
    :param days:

    :return: map_stats:
        dictionary updated
    """

    var_scalar = np.sqrt(252.0 / days) * -0.5

    for s in m_mapping.index:
        inst_pnl = position_prices[s].dropna()
        d = inst_pnl.index.intersection(generic_prices.index)
        inst_pnl = inst_pnl.loc[d].pct_change(periods=days)[days:]
        pca_prices_d = pca_prices.reindex(d).pct_change(periods=days)[days:]
        pca_pnl_d = (pca_prices_d * mappings.loc[s]).sum(axis=1)
        map_pnl = generic_prices.loc[d].pct_change(periods=days)[days:]
        map_pnl = (map_pnl * m_mapping.loc[s]).sum(axis=1)
        map_stats.loc[s, 'correl_to_generic'] = 0 if inst_pnl.std() == 0 else inst_pnl.corr(map_pnl)
        map_stats.loc[s, 'pca_to_act_vol_ratio'] = 0 if inst_pnl.std() == 0 else pca_pnl_d.std() / inst_pnl.std()
        map_stats.loc[s, 'mapped_to_act_vol_ratio'] = 0 if inst_pnl.std() == 0 else map_pnl.std() / inst_pnl.std()

        # Save vol measures
        map_stats.loc[s, 'inst_var_risk'] = var_expw(inst_pnl, VAR_HALF_LIFE, 0.02) * var_scalar
        map_stats.loc[s, 'mapped_var_risk'] = var_expw(map_pnl, VAR_HALF_LIFE, 0.02) * var_scalar

    return map_stats


def create_pca_components(generic_prices, days=1, scale_pca=True):
    generic_pnl = generic_prices.pct_change(periods=days).iloc[days:]

    X = np.array(generic_pnl)
    scalar = pd.Series(1, index=generic_pnl.columns)
    if scale_pca:
        scalar = generic_pnl.std(axis=0) * 100.
        X = scale(X) / 100.

    # Create mappings of generics to PCA components
    pca = PCA()
    fit = pca.fit(X)
    generic_pca_mappings = pd.DataFrame(data=fit.components_,
                                        columns=generic_pnl.columns,
                                        index=['PCA_' + str(i) for i in range(1, len(generic_pnl.columns) + 1)]).T

    # Create PNL series for PCA components (to account for missing days in user security data)
    pca_reg = pca.fit_transform(X)
    pca_df = pd.DataFrame(pca_reg, index=generic_pnl.index)
    pca_df += 1

    pca_df.loc[pca_df.index[0] + datetime.timedelta(-1)] = [100] * len(pca_df.columns)
    pca_df = pca_df.sort_index(ascending=True)

    pca_prices = pca_df.cumprod(axis=0)

    pca_prices.columns = ['PCA_' + str(i) for i in range(1, len(generic_pnl.columns) + 1)]

    return pca_prices, generic_pca_mappings, scalar
