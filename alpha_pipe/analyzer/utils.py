from numpy import sqrt, mean
from collections import Iterable
from functools import wraps
import six
import pandas as pd
import numpy as np
import re
import warnings

from IPython.display import display
from pandas.tseries.offsets import CustomBusinessDay, Day, BusinessDay


class NonMatchingTimezoneError(Exception):
    pass


class MaxLossExceededError(Exception):
    pass


def rethrow(exception, additional_message):
    """
    Re-raise the last exception that was active in the current scope
    without losing the stacktrace but adding an additional message.
    This is hacky because it has to be compatible with both python 2/3
    """
    e = exception
    m = additional_message
    if not e.args:
        e.args = (m,)
    else:
        e.args = (e.args[0] + m,) + e.args[1:]
    raise e


def non_unique_bin_edges_error(func):
    """
    Give user a more informative error in case it is not possible
    to properly calculate quantiles on the input dataframe (factor)
    """
    message = """

    An error occurred while computing bins/quantiles on the input provided.
    This usually happens when the input contains too many identical
    values and they span more than one quantile. The quantiles are choosen
    to have the same number of records each, but the same value cannot span
    multiple quantiles. Possible workarounds are:
    1 - Decrease the number of quantiles
    2 - Specify a custom quantiles range, e.g. [0, .50, .75, 1.] to get unequal
        number of records per quantile
    3 - Use 'bins' option instead of 'quantiles', 'bins' chooses the
        buckets to be evenly spaced according to the values themselves, while
        'quantiles' forces the buckets to have the same number of records.
    4 - for factors with discrete values use the 'bins' option with custom
        ranges and create a range for each discrete value
    Please see utils.get_clean_factor_and_forward_returns documentation for
    full documentation of 'bins' and 'quantiles' options.

"""

    def dec(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as e:
            if 'Bin edges must be unique' in str(e):
                rethrow(e, message)
            raise
    return dec


@non_unique_bin_edges_error
def quantize_factor(factor_data,
                    quantiles=5,
                    bins=None,
                    by_group=False,
                    no_raise=False,
                    zero_aware=False):
    """
    Computes period wise factor quantiles.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.

        - See full explanation in utils.get_clean_factor_and_forward_returns

    quantiles : int or sequence[float]
        Number of equal-sized quantile buckets to use in factor bucketing.
        Alternately sequence of quantiles, allowing non-equal-sized buckets
        e.g. [0, .10, .5, .90, 1.] or [.05, .5, .95]
        Only one of 'quantiles' or 'bins' can be not-None
    bins : int or sequence[float]
        Number of equal-width (valuewise) bins to use in factor bucketing.
        Alternately sequence of bin edges allowing for non-uniform bin width
        e.g. [-4, -2, -0.5, 0, 10]
        Only one of 'quantiles' or 'bins' can be not-None
    by_group : bool, optional
        If True, compute quantile buckets separately for each group.
    no_raise: bool, optional
        If True, no exceptions are thrown and the values for which the
        exception would have been thrown are set to np.NaN
    zero_aware : bool, optional
        If True, compute quantile buckets separately for positive and negative
        signal values. This is useful if your signal is centered and zero is
        the separation between long and short signals, respectively.

    Returns
    -------
    factor_quantile : pd.Series
        Factor quantiles indexed by date and asset.
    """
    if not ((quantiles is not None and bins is None) or
            (quantiles is None and bins is not None)):
        raise ValueError('Either quantiles or bins should be provided')

    if zero_aware and not (isinstance(quantiles, int)
                           or isinstance(bins, int)):
        msg = ("zero_aware should only be True when quantiles or bins is an"
               " integer")
        raise ValueError(msg)

    def quantile_calc(x, _quantiles, _bins, _zero_aware, _no_raise):
        try:
            x = x.rank(method='first')
            if _quantiles is not None and _bins is None and not _zero_aware:
                return pd.qcut(x, _quantiles, labels=False) + 1
            elif _quantiles is not None and _bins is None and _zero_aware:
                pos_quantiles = pd.qcut(x[x >= 0], _quantiles // 2,
                                        labels=False) + _quantiles // 2 + 1
                neg_quantiles = pd.qcut(x[x < 0], _quantiles // 2,
                                        labels=False) + 1
                return pd.concat([pos_quantiles, neg_quantiles]).sort_index()
            elif _bins is not None and _quantiles is None and not _zero_aware:
                return pd.cut(x, _bins, labels=False) + 1
            elif _bins is not None and _quantiles is None and _zero_aware:
                pos_bins = pd.cut(x[x >= 0], _bins // 2,
                                  labels=False) + _bins // 2 + 1
                neg_bins = pd.cut(x[x < 0], _bins // 2,
                                  labels=False) + 1
                return pd.concat([pos_bins, neg_bins]).sort_index()
        except Exception as e:
            if _no_raise:
                return pd.Series(index=x.index)
            raise e

    grouper = [factor_data.index.get_level_values('date')]
    if by_group:
        grouper.append('group')

    factor_quantile = factor_data.groupby(grouper)['factor'] \
        .apply(quantile_calc, quantiles, bins, zero_aware, no_raise)
    factor_quantile.name = 'factor_quantile'

    return factor_quantile.dropna()


def demean_forward_returns(factor_data: pd.DataFrame, grouper=None):
    """
    根据相关分组为因子远期收益去均值.
    分组去均值包含了投资组合分组中性化约束的假设，因此允许跨组评估因子
    Parameters
    ----------
    factor_data : pd.DataFrame
        标准因子数据, index 为 date (level 0) 和 asset (level 1) 
        factor-因子值 return(xxx)-远期收益 factor_quantile-因子值分位数
        group-因子分组(可选) weight-因子权重(可选)
    grouper : list, optional
        如果为 None, 则只根据日期去均值, 否则则根据列表中提供的组分组去均值, by default None

    Returns
    -------
    和 factor_data 相同形状的 DataFrame, 但每个收益都被分组去均值了
    """

    factor_data = factor_data.copy()

    if not grouper:
        grouper = factor_data.index.get_level_values('date')

    cols = get_forward_returns_columns(factor_data.columns)

    # factor_data[cols] = factor_data.groupby(grouper)[cols] \
    #     .transform(lambda x: x - x.mean())
    
    # 如果有weights计算均值的时候会利用weights进行加权
    factor_data[cols] = factor_data.groupby(grouper, as_index=False)[cols.append(pd.Index(['weights']))].apply(
        lambda x: x[cols].subtract(np.average(x[cols], axis=0, weights=x['weights'].fillna(0.0).values), axis=1))

    return factor_data


def print_table(table, name=None, fmt=None):
    """
    Pretty print a pandas DataFrame.

    Uses HTML output if running inside Jupyter Notebook, otherwise
    formatted text output.

    Parameters
    ----------
    table : pd.Series or pd.DataFrame
        Table to pretty-print.
    name : str, optional
        Table name to display in upper left corner.
    fmt : str, optional
        Formatter to use for displaying table elements.
        E.g. '{0:.2f}%' for displaying 100 as '100.00%'.
        Restores original setting after displaying.
    """
    if isinstance(table, pd.Series):
        table = pd.DataFrame(table)

    if isinstance(table, pd.DataFrame):
        table.columns.name = name

    prev_option = pd.get_option('display.float_format')
    if fmt is not None:
        pd.set_option('display.float_format', lambda x: fmt.format(x))

    display(table)

    if fmt is not None:
        pd.set_option('display.float_format', prev_option)


def rate_of_returns(ret, period):
    """转换收益率为每天的收益率

    Parameters
    ----------
    ret : 
        原始收益率序列
    period : 
        原始收益计算周期

    Returns
    -------
        转换后的每日收益率
    """
    return ((np.nansum(ret) + 1)**(1. / period)) - 1


def get_forward_returns_columns(columns):
    """
    返回远期收益的序列
    """
    pattern = re.compile(r"^(return\(.+\))$", re.IGNORECASE)
    valid_columns = [(pattern.match(col) is not None) for col in columns]
    return columns[valid_columns]


def timedelta_strings_to_integers(sequence):
    """
    Converts pandas string representations of timedeltas into integers of days.

    Parameters
    ----------
    sequence : iterable
        List or array of timedelta string representations, e.g. ['1D', '5D'].

    Returns
    -------
    sequence : list
        Integer days corresponding to the input sequence, e.g. [1, 5].
    """
    return list(map(lambda x: pd.Timedelta(x).days, sequence))


def add_custom_calendar_timedelta(input, timedelta, freq):
    """
    Add timedelta to 'input' taking into consideration custom frequency, which
    is used to deal with custom calendars, such as a trading calendar

    Parameters
    ----------
    input : pd.DatetimeIndex or pd.Timestamp
    timedelta : pd.Timedelta
    freq : pd.DataOffset (CustomBusinessDay, Day or BusinessDay)

    Returns
    -------
    pd.DatetimeIndex or pd.Timestamp
        input + timedelta
    """
    if not isinstance(freq, (Day, BusinessDay, CustomBusinessDay)):
        raise ValueError("freq must be Day, BDay or CustomBusinessDay")
    days = timedelta.components.days
    offset = timedelta - pd.Timedelta(days=days)
    return input + freq * days + offset


def ignore_warning(message='', category=Warning, module='', lineno=0, append=False):
    """过滤 warnings"""
    def decorator(func):
        @wraps(func)
        def func_wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message=message, category=category,
                                        module=module, lineno=lineno, append=append)
                return func(*args, **kwargs)
        return func_wrapper

    return decorator


def ensure_tuple(x):
    if isinstance(x, six.string_types) or not isinstance(x, Iterable):
        return (x,)
    else:
        return tuple(x)


def Indicators(value):

    columns = ['总收益', '年化收益', '波动率', '夏普比', '最大回撤', '卡玛比率', '日胜率', '盈亏比']

    def MaxBack(value):

        drawback = []

        for i, v in enumerate(value):

            drawback.append(max(1-value[i:]/v))

        MaxBack = max(drawback)

        return MaxBack

    value = [i/value[0] for i in value]

    AllRtn = round(value[-1]*100-100, 2)

    AulRtn = round(pow(value[-1], 250/len(value))*100-100, 2)

    value = pd.Series(value)
    Rtns = value.pct_change(1).dropna()

    Volity = round(sqrt(Rtns.var()*250)*100, 2)
    SpRatio = round((AulRtn-4)/Volity, 2)
    MaxBack = round(MaxBack(value)*100, 2)
    CmRatio = round(AulRtn/MaxBack, 2)

    R1 = [i for i in Rtns.values if i > 0]
    R2 = [i for i in Rtns.values if i < 0]

    WinRate = round(len(R1)/(len(R1)+len(R2))*100, 3)
    BidRatio = round(-mean(R1)/mean(R2), 3)

    data = [AllRtn, AulRtn, Volity, SpRatio,
            MaxBack, CmRatio, WinRate, BidRatio]
    result = pd.Series(index=columns, data=data)

    return result
