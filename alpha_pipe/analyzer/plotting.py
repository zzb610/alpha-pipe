import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from functools import wraps

from . import utils
from . import performance as perf

from .plot_utils import (
    print_table, customize, ICTS, ICHIST, ICQQ, QRETURNBAR, QRETURNVIOLIN,
    QRETURNTS, ICGROUP, AUTOCORR, TBTURNOVER, ICHEATMAP, CUMRET, TDCUMRET,
    CUMRETQ, AVGCUMRET, EVENTSDIST, MISSIINGEVENTSDIST
)

DECIMAL_TO_BPS = 10000


def plotting_context(context='notebook', font_scale=1.5, rc=None):
    """
    Create alphalens default plotting style context.

    Under the hood, calls and returns seaborn.plotting_context() with
    some custom settings. Usually you would use in a with-context.

    Parameters
    ----------
    context : str, optional
        Name of seaborn context.
    font_scale : float, optional
        Scale font by factor font_scale.
    rc : dict, optional
        Config flags.
        By default, {'lines.linewidth': 1.5}
        is being used and will be added to any
        rc passed in, unless explicitly overriden.

    Returns
    -------
    seaborn plotting context

    Example
    -------
    with alphalens.plotting.plotting_context(font_scale=2):
        alphalens.create_full_tear_sheet(..., set_context=False)

    See also
    --------
    For more information, see seaborn.plotting_context().
    """
    if rc is None:
        rc = {}

    rc_default = {'lines.linewidth': 1.5}

    # Add defaults if they do not exist
    for name, val in rc_default.items():
        rc.setdefault(name, val)

    return sns.plotting_context(context=context, font_scale=font_scale, rc=rc)


def axes_style(style='darkgrid', rc=None):
    """Create alphalens default axes style context.

    Under the hood, calls and returns seaborn.axes_style() with
    some custom settings. Usually you would use in a with-context.

    Parameters
    ----------
    style : str, optional
        Name of seaborn style.
    rc : dict, optional
        Config flags.

    Returns
    -------
    seaborn plotting context

    Example
    -------
    with alphalens.plotting.axes_style(style='whitegrid'):
        alphalens.create_full_tear_sheet(..., set_context=False)

    See also
    --------
    For more information, see seaborn.plotting_context().

    """
    if rc is None:
        rc = {}

    rc_default = {}

    # Add defaults if they do not exist
    for name, val in rc_default.items():
        rc.setdefault(name, val)

    return sns.axes_style(style=style, rc=rc)


def plot_returns_table(alpha_beta,
                       mean_ret_quantile,
                       mean_ret_spread_quantile):
    returns_table = pd.DataFrame()
    returns_table = returns_table.append(alpha_beta)
    returns_table.loc["Mean Period Wise Return Top Quantile (bps)"] = \
        mean_ret_quantile.iloc[-1] * DECIMAL_TO_BPS
    returns_table.loc["Mean Period Wise Return Bottom Quantile (bps)"] = \
        mean_ret_quantile.iloc[0] * DECIMAL_TO_BPS
    returns_table.loc["Mean Period Wise Spread (bps)"] = \
        mean_ret_spread_quantile.mean() * DECIMAL_TO_BPS

    print("收益分析")
    print_table(returns_table.apply(lambda x: x.round(3)))


def plot_turnover_table(autocorrelation_data, quantile_turnover):
    turnover_table = pd.DataFrame()
  
    for ret_name in quantile_turnover.keys():
        for quantile, p_data in quantile_turnover[ret_name].iteritems():
            turnover_table.loc["Quantile {} Mean Turnover ".format(quantile),
                               "{}".format(ret_name)] = p_data.mean()
        
    auto_corr = pd.DataFrame()
    for ret_name, p_data in autocorrelation_data.items():
        auto_corr.loc["Mean Factor Rank Autocorrelation",
                      "{}".format(ret_name)] = p_data.mean()

    print("换手率分析")
    print_table(turnover_table.apply(lambda x: x.round(3)))
    print_table(auto_corr.apply(lambda x: x.round(3)))


def plot_information_table(ic_data):
    ic_summary_table = pd.DataFrame()
    ic_summary_table["IC Mean"] = ic_data.mean()
    ic_summary_table["IC Std."] = ic_data.std()
    ic_summary_table["Risk-Adjusted IC"] = \
        ic_data.mean() / ic_data.std()
    t_stat, p_value = stats.ttest_1samp(ic_data, 0)
    ic_summary_table["t-stat(IC)"] = t_stat
    ic_summary_table["p-value(IC)"] = p_value
    ic_summary_table["IC Skew"] = stats.skew(ic_data)
    ic_summary_table["IC Kurtosis"] = stats.kurtosis(ic_data)

    print("IC 分析")
    print_table(ic_summary_table.apply(lambda x: x.round(3)).T)


def plot_quantile_statistics_table(factor_data):
    quantile_stats = factor_data.groupby('factor_quantile') \
        .agg(['min', 'max', 'mean', 'std', 'count'])['factor']
    quantile_stats['count %'] = quantile_stats['count'] \
        / quantile_stats['count'].sum() * 100.

    print("分位数统计")
    utils.print_table(quantile_stats)


def plot_ic_ts(ic, ax=None):
    """
    Plots Spearman Rank Information Coefficient and IC moving
    average for a given factor.

    Parameters
    ----------
    ic : pd.DataFrame
        DataFrame indexed by date, with IC for each forward return.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """
    ic = ic.copy()

    num_plots = len(ic.columns)
    if ax is None:
        f, ax = plt.subplots(num_plots, 1, figsize=(18, num_plots * 7))
        ax = np.asarray([ax]).flatten()

    ymin, ymax = (None, None)
    for a, (period_num, ic) in zip(ax, ic.iteritems()):
        ic.plot(alpha=0.7, ax=a, lw=0.7, color='steelblue')
        ic.rolling(window=22).mean().plot(
            ax=a,
            color='forestgreen',
            lw=2,
            alpha=0.8
        )

        a.axhline(0.0, linestyle='-', color='black', lw=1, alpha=0.8)
        a.set(ylabel='IC', xlabel="")
        a.set_title(ICTS.get("TITLE").format(period_num))
        a.legend(ICTS.get("LEGEND"), loc='upper right')
        a.text(
            .05,
            .95,
            ICTS.get("TEXT").format(ic.mean(), ic.std()),
            fontsize=16,
            bbox={
                'facecolor': 'white',
                'alpha': 1,
                'pad': 5
            },
            transform=a.transAxes,
            verticalalignment='top'
        )

        curr_ymin, curr_ymax = a.get_ylim()
        ymin = curr_ymin if ymin is None else min(ymin, curr_ymin)
        ymax = curr_ymax if ymax is None else max(ymax, curr_ymax)

    for a in ax:
        a.set_ylim([ymin, ymax])

    return ax

@customize
def plot_ic_hist(ic, ax=None):
    """
    Plots Spearman Rank Information Coefficient histogram for a given factor.

    Parameters
    ----------
    ic : pd.DataFrame
        DataFrame indexed by date, with IC for each forward return.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    ic = ic.copy()

    num_plots = len(ic.columns)

    v_spaces = ((num_plots - 1) // 3) + 1

    if ax is None:
        f, ax = plt.subplots(v_spaces, 3, figsize=(18, v_spaces * 6))
        ax = ax.flatten()

    for a, (period_num, ic) in zip(ax, ic.iteritems()):
        sns.distplot(ic.replace(np.nan, 0.), norm_hist=True, ax=a)
        a.set(title=ICHIST.get("TITLE") % period_num, xlabel='IC')
        a.text(
            .05,
            .95,
            ICHIST.get("LEGEND").format(ic.mean(), ic.std()),
            fontsize=16,
            bbox={
                'facecolor': 'white',
                'alpha': 1,
                'pad': 5
            },
            transform=a.transAxes,
            verticalalignment='top'
        )
        a.axvline(ic.mean(), color='w', linestyle='dashed', linewidth=2)

    if num_plots < len(ax):
        for a in ax[num_plots:]:
            a.set_visible(False)

    return ax

@customize
def plot_ic_qq(ic, theoretical_dist=stats.norm, ax=None):
    """
    Plots Spearman Rank Information Coefficient "Q-Q" plot relative to
    a theoretical distribution.

    Parameters
    ----------
    ic : pd.DataFrame
        DataFrame indexed by date, with IC for each forward return.
    theoretical_dist : scipy.stats._continuous_distns
        Continuous distribution generator. scipy.stats.norm and
        scipy.stats.t are popular options.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    ic = ic.copy()

    num_plots = len(ic.columns)
    v_spaces = ((num_plots - 1) // 3) + 1

    if ax is None:
        f, ax = plt.subplots(v_spaces, 3, figsize=(18, v_spaces * 6))
        ax = ax.flatten()

    if isinstance(theoretical_dist, stats.norm.__class__):
        dist_name = ICQQ.get("NORM")
    elif isinstance(theoretical_dist, stats.t.__class__):
        dist_name = ICQQ.get("T")
    else:
        dist_name = ICQQ.get("CUSTOM")

    for a, (period, ic) in zip(ax, ic.iteritems()):
        period_num = period.replace('period_', '')
        sm.qqplot(
            ic.replace(np.nan, 0.).values,
            theoretical_dist,
            fit=True,
            line='45',
            ax=a
        )
        a.set(
            title=ICQQ.get("TITLE").format(period_num, dist_name),
            xlabel=ICQQ.get("XLABEL").format(dist_name),
            ylabel=ICQQ.get("YLABEL"),
        )

    if num_plots < len(ax):
        for a in ax[num_plots:]:
            a.set_visible(False)

    return ax

@customize
def plot_quantile_returns_bar(mean_ret_by_q,
                              by_group=False,
                              ylim_percentiles=None,
                              ax=None):
    """
    Plots mean period wise returns for factor quantiles.

    Parameters
    ----------
    mean_ret_by_q : pd.DataFrame
        DataFrame with quantile, (group) and mean period wise return values.
    by_group : bool
        Disaggregated figures by group.
    ylim_percentiles : tuple of integers
        Percentiles of observed data to use as y limits for plot.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    mean_ret_by_q = mean_ret_by_q.copy()

    if ylim_percentiles is not None:
        ymin = (np.nanpercentile(mean_ret_by_q.values,
                                 ylim_percentiles[0]) * DECIMAL_TO_BPS)
        ymax = (np.nanpercentile(mean_ret_by_q.values,
                                 ylim_percentiles[1]) * DECIMAL_TO_BPS)
    else:
        ymin = None
        ymax = None

    if by_group:
        num_group = len(
            mean_ret_by_q.index.get_level_values('group').unique())

        if ax is None:
            v_spaces = ((num_group - 1) // 2) + 1
            f, ax = plt.subplots(v_spaces, 2, sharex=False,
                                 sharey=True, figsize=(18, 6 * v_spaces))
            ax = ax.flatten()

        for a, (sc, cor) in zip(ax, mean_ret_by_q.groupby(level='group')):
            (cor.xs(sc, level='group')
                .multiply(DECIMAL_TO_BPS)
                .plot(kind='bar', title=sc, ax=a))

            a.set(xlabel='', ylabel=QRETURNBAR.get("YLABEL"), ylim=(ymin, ymax))

        if num_group < len(ax):
            ax[-1].set_visible(False)

        return ax

    else:
        if ax is None:
            f, ax = plt.subplots(1, 1, figsize=(18, 6))

        mean_ret_by_q.multiply(DECIMAL_TO_BPS).plot(
            kind='bar', title=QRETURNBAR.get("TITLE"), ax=ax
        )
        ax.set(xlabel="", ylabel=QRETURNBAR.get("YLABEL"), ylim=(ymin, ymax))
        return ax

@customize
def plot_quantile_returns_violin(return_by_q,
                                 ylim_percentiles=None,
                                 ax=None):
    """
    Plots a violin box plot of period wise returns for factor quantiles.

    Parameters
    ----------
    return_by_q : pd.DataFrame - MultiIndex
        DataFrame with date and quantile as rows MultiIndex,
        forward return windows as columns, returns as values.
    ylim_percentiles : tuple of integers
        Percentiles of observed data to use as y limits for plot.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    return_by_q = return_by_q.copy()

    if ylim_percentiles is not None:
        ymin = (np.nanpercentile(return_by_q.values,
                                 ylim_percentiles[0]) * DECIMAL_TO_BPS)
        ymax = (np.nanpercentile(return_by_q.values,
                                 ylim_percentiles[1]) * DECIMAL_TO_BPS)
    else:
        ymin = None
        ymax = None

    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(18, 6))

    unstacked_dr = (return_by_q
                    .multiply(DECIMAL_TO_BPS))
    unstacked_dr.columns = unstacked_dr.columns.set_names('forward_periods')
    unstacked_dr = unstacked_dr.stack()
    unstacked_dr.name = 'return'
    unstacked_dr = unstacked_dr.reset_index()

    sns.violinplot(
        data=unstacked_dr,
        x='factor_quantile',
        hue='forward_periods',
        y='return',
        orient='v',
        cut=0,
        inner='quartile',
        ax=ax
    )

    ax.set(
        xlabel='',
        ylabel=QRETURNVIOLIN.get("YLABEL"),
        title=QRETURNVIOLIN.get("TITLE"),
        ylim=(ymin, ymax)
    )

    ax.axhline(0.0, linestyle='-', color='black', lw=0.7, alpha=0.6)

    return ax

 

@customize
def plot_mean_quantile_returns_spread_time_series(mean_returns_spread,
                                                  std_err=None,
                                                  bandwidth=1,
                                                  ax=None):
    """
    Plots mean period wise returns for factor quantiles.

    Parameters
    ----------
    mean_returns_spread : pd.Series
        Series with difference between quantile mean returns by period.
    std_err : pd.Series
        Series with standard error of difference between quantile
        mean returns each period.
    bandwidth : float
        Width of displayed error bands in standard deviations.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if isinstance(mean_returns_spread, pd.DataFrame):
        if ax is None:
            ax = [None for a in mean_returns_spread.columns]

        ymin, ymax = (None, None)
        for (i, a), (name, fr_column) in zip(enumerate(ax),
                                             mean_returns_spread.iteritems()):
            stdn = None if std_err is None else std_err[name]
            a = plot_mean_quantile_returns_spread_time_series(fr_column,
                                                              std_err=stdn,
                                                              ax=a)
            ax[i] = a
            curr_ymin, curr_ymax = a.get_ylim()
            ymin = curr_ymin if ymin is None else min(ymin, curr_ymin)
            ymax = curr_ymax if ymax is None else max(ymax, curr_ymax)

        for a in ax:
            a.set_ylim([ymin, ymax])

        return ax

    if mean_returns_spread.isnull().all():
        return ax

    periods = mean_returns_spread.name
 
    title = QRETURNTS.get(
        "TITLE"
    ).format(periods.replace('period_', '') if periods is not None else '')

    if ax is None:
        f, ax = plt.subplots(figsize=(18, 6))

    mean_returns_spread_bps = mean_returns_spread * DECIMAL_TO_BPS

    mean_returns_spread_bps.plot(alpha=0.4, ax=ax, lw=0.7, color='forestgreen')
    mean_returns_spread_bps.rolling(window=22).mean().plot(
        color='orangered',
        alpha=0.7,
        ax=ax
    )
    ax.legend(
        [QRETURNTS.get("LEGEND0").format(bandwidth),
         QRETURNTS.get("LEGEND1")],
        loc='upper right'
    )

    if std_err is not None:
        std_err_bps = std_err * DECIMAL_TO_BPS
        upper = mean_returns_spread_bps.values + (std_err_bps * bandwidth)
        lower = mean_returns_spread_bps.values - (std_err_bps * bandwidth)
        ax.fill_between(mean_returns_spread.index,
                        lower,
                        upper,
                        alpha=0.3,
                        color='steelblue')

    ylim = np.nanpercentile(abs(mean_returns_spread_bps.values), 95)
    ax.set(
        ylabel=QRETURNTS.get("YLABEL"),
        xlabel="",
        title=title,
        ylim=(-ylim, ylim)
    )
    ax.axhline(0.0, linestyle='-', color='black', lw=1, alpha=0.8)

    return ax

@customize
def plot_ic_by_group(ic_group, ax=None):
    """
    Plots Spearman Rank Information Coefficient for a given factor over
    provided forward returns. Separates by group.

    Parameters
    ----------
    ic_group : pd.DataFrame
        group-wise mean period wise returns.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(18, 6))
    ic_group.plot(kind='bar', ax=ax)

    ax.set(title=ICGROUP.get("TITLE"), xlabel="")
    ax.set_xticklabels(ic_group.index, rotation=45)

    return ax

@customize
def plot_factor_rank_auto_correlation(factor_autocorrelation,
                                      ret_type,
                                      ax=None):
    """
    Plots factor rank autocorrelation over time.
    See factor_rank_autocorrelation for more details.

    Parameters
    ----------
    factor_autocorrelation : pd.Series
        Rolling 1 period (defined by time_rule) autocorrelation
        of factor values.
    ret_type:
        return type
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(18, 6))

    factor_autocorrelation.plot(title=AUTOCORR.get("TITLE").format(ret_type), ax=ax)
    ax.set(ylabel=AUTOCORR.get("YLABEL").format(ret_type), xlabel="")
    ax.axhline(0.0, linestyle='-', color='black', lw=1)
    ax.text(
        .05,
        .95,
        AUTOCORR.get("TEXT").format(factor_autocorrelation.mean()),
        fontsize=16,
        bbox={
            'facecolor': 'white',
            'alpha': 1,
            'pad': 5
        },
        transform=ax.transAxes,
        verticalalignment='top'
    )

    return ax

@customize
def plot_top_bottom_quantile_turnover(quantile_turnover, ret_type, ax=None):
    """
    Plots period wise top and bottom quantile factor turnover.

    Parameters
    ----------
    quantile_turnover: pd.Dataframe
        Quantile turnover (each DataFrame column a quantile).
    period: int, optional
        Period over which to calculate the turnover.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(18, 6))

    max_quantile = quantile_turnover.columns.max()
    min_quantile = quantile_turnover.columns.min()
    from IPython.display import display
 
    turnover = pd.DataFrame()
    turnover[TBTURNOVER.get("TURNOVER").format(max_quantile)
             ] = quantile_turnover[max_quantile]
    turnover[TBTURNOVER.get("TURNOVER").format(min_quantile)
             ] = quantile_turnover[min_quantile]
    turnover.plot(
        title=TBTURNOVER.get("TITLE").format(ret_type), ax=ax, alpha=0.6, lw=0.8
    )

    ax.set(ylabel=TBTURNOVER.get("YLABEL"), xlabel="")

    return ax


@customize
def plot_monthly_ic_heatmap(mean_monthly_ic, ax=None):
    """
    Plots a heatmap of the information coefficient or returns by month.

    Parameters
    ----------
    mean_monthly_ic : pd.DataFrame
        The mean monthly IC for N periods forward.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    mean_monthly_ic = mean_monthly_ic.copy()

    num_plots = len(mean_monthly_ic.columns)

    v_spaces = ((num_plots - 1) // 3) + 1

    if ax is None:
        f, ax = plt.subplots(v_spaces, 3, figsize=(18, v_spaces * 6))
        ax = ax.flatten()

    new_index_year = []
    new_index_month = []
    for date in mean_monthly_ic.index:
        new_index_year.append(date.year)
        new_index_month.append(date.month)

    mean_monthly_ic.index = pd.MultiIndex.from_arrays(
        [new_index_year, new_index_month],
        names=["year", "month"])

    for a, (periods_num, ic) in zip(ax, mean_monthly_ic.iteritems()):

        sns.heatmap(
            ic.unstack(),
            annot=True,
            alpha=1.0,
            center=0.0,
            annot_kws={"size": 7},
            linewidths=0.01,
            linecolor='white',
            cmap=cm.coolwarm_r,
            cbar=False,
            ax=a)
        a.set(ylabel='', xlabel='')

        a.set_title(ICHEATMAP.get("TITLE").format(periods_num))

    if num_plots < len(ax):
        ax[-1].set_visible(False)

    return ax

@customize
def plot_cumulative_returns(factor_returns,
                            period,
                            freq=None,
                            title=None,
                            ax=None):
    """
    Plots the cumulative returns of the returns series passed in.

    Parameters
    ----------
    factor_returns : pd.Series
        Period wise returns of dollar neutral portfolio weighted by factor
        value.
    period : pandas.Timedelta or string
        Length of period for which the returns are computed (e.g. 1 day)
        if 'period' is a string it must follow pandas.Timedelta constructor
        format (e.g. '1 days', '1D', '30m', '3h', '1D1h', etc)
    freq : pandas DateOffset
        Used to specify a particular trading calendar e.g. BusinessDay or Day
        Usually this is inferred from utils.infer_trading_calendar, which is
        called by either get_clean_factor_and_forward_returns or
        compute_forward_returns
    title: string, optional
        Custom title
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(18, 6))

    factor_returns = factor_returns.resample(freq).first()
    factor_returns = perf.cumulative_returns(factor_returns)
    factor_returns.plot(ax=ax, lw=3, color='forestgreen', alpha=0.6)
    ax.set(
        ylabel=CUMRET.get("YLABEL"),
        title=CUMRET.get("TITLE").format(period),
        xlabel=""
    )
    ax.axhline(1.0, linestyle='-', color='black', lw=1)

    return ax

@customize
def plot_cumulative_returns_by_quantile(quantile_returns,
                                        period,
                                        freq=None,
                                        ax=None,
                                        title=None):
    """
    Plots the cumulative returns of various factor quantiles.

    Parameters
    ----------
    quantile_returns : pd.DataFrame
        Returns by factor quantile
    period : pandas.Timedelta or string
        Length of period for which the returns are computed (e.g. 1 day)
        if 'period' is a string it must follow pandas.Timedelta constructor
        format (e.g. '1 days', '1D', '30m', '3h', '1D1h', etc)
    freq : pandas DateOffset
        Used to specify a particular trading calendar e.g. BusinessDay or Day
        Usually this is inferred from utils.infer_trading_calendar, which is
        called by either get_clean_factor_and_forward_returns or
        compute_forward_returns
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
    """

    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(18, 6))

    ret_wide = quantile_returns.unstack('factor_quantile')

    quantiles = ret_wide.columns
    ret_wide['top_bottom'] = ret_wide[max(
        quantiles)] - ret_wide[min(quantiles)]
    if ret_wide['top_bottom'].sum() < 0:
        ret_wide['top_bottom'] = -ret_wide['top_bottom']

    # from IPython.display import display
    # display(ret_wide)
    ret_wide = ret_wide.reset_index().set_index('date')
    ret_wide = ret_wide.resample(freq).first()
    cum_ret = ret_wide.apply(perf.cumulative_returns)

    cum_ret = cum_ret.loc[:, ::-1]  # we want negative quantiles as 'red'

    cum_ret.plot(lw=2, ax=ax, cmap=cm.coolwarm)
    ax.legend()

    ax.set(
        ylabel=TDCUMRET.get("YLABEL"),
        title=TDCUMRET.get("TITLE").format(period),
        xlabel=""
    )

    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.axhline(1.0, linestyle='-', color='black', lw=1)

    return ax


@customize
def plot_quantile_average_cumulative_return(
    avg_cumulative_returns,
    by_quantile=False,
    std_bar=False,
    ax=None,
    periods_before='',
    periods_after=''
):

    avg_cumulative_returns = avg_cumulative_returns.multiply(DECIMAL_TO_BPS)
    quantiles = len(avg_cumulative_returns.index.levels[0].unique())
    palette = [cm.RdYlGn_r(i) for i in np.linspace(0, 1, quantiles)]

    if by_quantile:

        if ax is None:
            v_spaces = ((quantiles - 1) // 2) + 1
            f, ax = plt.subplots(
                v_spaces,
                2,
                sharex=False,
                sharey=False,
                figsize=(18, 6 * v_spaces)
            )
            ax = ax.flatten()

        for i, (quantile, q_ret) in enumerate(
            avg_cumulative_returns.groupby(level='factor_quantile')
        ):

            mean = q_ret.loc[(quantile, 'mean')]
            mean.name = AVGCUMRET.get("COLUMN").format(quantile)
            mean.plot(ax=ax[i], color=palette[i])
            ax[i].set_ylabel(AVGCUMRET.get("YLABEL"))

            if std_bar:
                std = q_ret.loc[(quantile, 'std')]
                ax[i].errorbar(
                    std.index,
                    mean,
                    yerr=std,
                    fmt='none',
                    ecolor=palette[i],
                    label=None
                )

            ax[i].axvline(x=0, color='k', linestyle='--')
            ax[i].legend()
            i += 1

    else:

        if ax is None:
            f, ax = plt.subplots(1, 1, figsize=(18, 6))

        for i, (quantile, q_ret) in enumerate(
            avg_cumulative_returns.groupby(level='factor_quantile')
        ):

            mean = q_ret.loc[(quantile, 'mean')]
            mean.name = AVGCUMRET.get("COLUMN").format(quantile)
            mean.plot(ax=ax, color=palette[i])

            if std_bar:
                std = q_ret.loc[(quantile, 'std')]
                ax.errorbar(
                    std.index,
                    mean,
                    yerr=std,
                    fmt='none',
                    ecolor=palette[i],
                    label=None
                )
            i += 1

        ax.axvline(x=0, color='k', linestyle='--')
        ax.legend()
        ax.set(
            title=AVGCUMRET.get("YLABEL").format(periods_before, periods_after),
            xlabel=AVGCUMRET.get("XLABEL"),
            ylabel=AVGCUMRET.get("YLABEL"),
        )

    return ax


@customize
def plot_events_distribution(events, num_days=5, full_dates=None, ax=None):

    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(18, 6))

    if full_dates is None:
        full_dates = events.index.get_level_values('date').unique()

    group = pd.Series(range(len(full_dates)), index=full_dates) // num_days
    grouper_label = group.drop_duplicates()
    grouper = group.reindex(events.index.get_level_values('date'))

    count = events.groupby(grouper.values).count()
    count = count.reindex(grouper_label.values, fill_value=0)
    count.index = grouper_label.index.map(lambda x: x.strftime('%Y-%m-%d'))
    count.plot(kind="bar", grid=False, ax=ax)

    def annotateBars(x, dt, ax=ax):
        color = 'black'
        vertalign = 'top'
        ax.text(
            x,
            count.loc[dt],
            "{:d}".format(count.loc[dt]),
            rotation=45,
            color=color,
            horizontalalignment='center',
            verticalalignment=vertalign,
            fontsize=15,
            weight='heavy'
        )

    [annotateBars(x, dt, ax=ax) for x, dt in enumerate(list(count.index))]
    ax.set(
        ylabel=EVENTSDIST.get("YLABEL"),
        title=EVENTSDIST.get("TITLE"),
        xlabel=EVENTSDIST.get("XLABEL"),
    )
    return ax


@customize
def plot_missing_events_distribution(
    events, num_days=5, full_dates=None, ax=None
):

    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(18, 6))

    if full_dates is None:
        full_dates = events.index.get_level_values('date').unique()

    daily_count = events.groupby(level='date').count()
    most_common_count = np.argmax(np.bincount(daily_count))
    daily_missing = daily_count / most_common_count - 1
    daily_missing = daily_missing.reindex(full_dates, fill_value=-1.0)

    grouper = pd.Series(range(len(full_dates)), index=full_dates) // num_days
    grouper_label = grouper.drop_duplicates()

    missing = daily_missing.groupby(grouper.values).mean()
    missing = missing.reindex(grouper_label.values, fill_value=-1.0)
    missing.index = grouper_label.index.map(lambda x: x.strftime('%Y-%m-%d'))
    missing.plot(kind="bar", grid=False, ax=ax)

    def annotateBars(x, dt, ax=ax):
        color = 'black'
        vertalign = 'top'
        ax.text(
            x,
            missing.loc[dt],
            "{:+.1f}%".format(missing.loc[dt] * 100),
            rotation=45,
            color=color,
            horizontalalignment='center',
            verticalalignment=vertalign,
            fontsize=15,
            weight='heavy'
        )

    [annotateBars(x, dt, ax=ax) for x, dt in enumerate(list(missing.index))]
    ax.set(
        ylabel=MISSIINGEVENTSDIST.get("YLABEL"),
        title=MISSIINGEVENTSDIST.get("TITLE"),
        xlabel=MISSIINGEVENTSDIST.get("XLABEL")
    )

    return ax
