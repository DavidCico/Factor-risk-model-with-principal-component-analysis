import numpy as np
from pandas import Series


def _expw(index, half_life):
    """
    Creates exponential weighting for a given timeseries using half life.
    """

    last = max(index[0], index[-1])
    # time since last day in pnls as a percentage of a year
    t = (last - index.to_series()).astype('timedelta64[D]').apply(float) / 365.25
    base = np.exp2(1.0 / half_life)
    weights = np.power(base, -t)
    return weights


def var_expw(pnls, half_life, percentile=0.02):
    """
    Calculates the exponentially weighted VaR of a portfolio
    using the given half-life.

    :param pnls:
        A timeseries containing PNL of the portfolio

    :param half_life: float
        The half-life of the exponential weighting. A
        shorter half-life means more precedence is given
        to recent data.

    :param percentile: float
        Percentile of VaR, e.g. percentile=0.01 means
        1% VaR

    :return var : float
        The VaR of the portfolio at a given percentile and half-life
    """

    weights = _expw(pnls.index, half_life)

    # sort by PNL
    wpnls, pnl_weights = zip(*sorted(zip(pnls, weights)))
    wpnls = Series(list(wpnls))
    pnl_weights = Series(list(pnl_weights))

    cum_prob = pnl_weights.cumsum() / pnl_weights.sum()

    # consistency map (the 50% percentile for 1,2,3,...,10 is 5.5)
    percentile = (percentile * (len(pnls) - 1.0) + 1.0) / float(len(pnls))
    rank = wpnls.index[cum_prob >= percentile][0]
    if rank == 0:
        return wpnls[0]

    correction = (wpnls[rank] - wpnls[rank - 1]) * max(0.0, percentile - cum_prob[rank - 1]) \
        / (cum_prob[rank] - cum_prob[rank - 1])

    return wpnls[rank - 1] + correction
