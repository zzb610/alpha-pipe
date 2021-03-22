from typing import Tuple
from IPython.core.display import display
from numpy import exp, quantile
from .alphalens.utils import get_clean_factor_and_forward_returns, MaxLossExceededError

import pandas as pd

import matplotlib.pyplot as plt

import re
from .alphalens import plotting
from .alphalens import performance as perf
from .alphalens import utils
from .alphalens.tears import GridFigure

import abc

class BaseTest():

    def __init__(self):

        self._factor_data = None
        self._ret_types = None
        self._periods = None

        self._long_short = None
        self._group_neutral = None
        self._equal_weight = None
        self._by_group = None

        self._factor_returns = None
        self._alpha_beta  = None
        self._mean_quant_rateret = None
        self._mean_ret_spread_quant = None
        self._mean_quant_rateret_bydate = None
        self._std_spread_quant = None
        self._mean_quant_ret_bydate = None

        self._mean_return_quantile_group = None
        self._mean_return_quantile_group_std_err = None
        self._mean_quant_rateret_group = None
        self._num_groups = None

        self._ic = None
        self._mean_monthly_ic = None
        self._mean_group_ic = None

        self._quantile_turnover = None
        self._autocorrelation = None
        self._turnover_periods = None

        self._avg_cumulative_returns = None
        self._avg_cumret_by_group= None

    @abc.abstractmethod
    def factor_data(self):
        pass
        
    def return_analysis(self):
        """calculate return performance
        """
        
        factor_data = self.factor_data()

        long_short = self._long_short
        group_neutral = self._group_neutral
        equal_weight = self._equal_weight
         

        self._factor_returns = perf.factor_returns(
            factor_data, long_short, group_neutral, equal_weight
        )
         

        mean_quant_ret, std_quantile = perf.mean_return_by_quantile(
            factor_data,
            by_group=False,
            demeaned=long_short,
            group_adjust=group_neutral,
        )
        self._mean_quant_rateret = mean_quant_ret

        # self._mean_quant_rateret = mean_quant_ret.apply(
        #     utils.rate_of_return, axis=0, base_period=mean_quant_ret.columns[0]
        # )

        self._mean_quant_ret_bydate, std_quant_daily = perf.mean_return_by_quantile(
            factor_data,
            by_date=True,
            by_group=False,
            demeaned=long_short,
            group_adjust=group_neutral,
        )

        self._mean_quant_rateret_bydate = self._mean_quant_ret_bydate

        # self._mean_quant_rateret_bydate = self._mean_quant_ret_bydate.apply(
        #     utils.rate_of_return,
        #     axis=0,
        #     base_period=self._mean_quant_ret_bydate.columns[0],
        # )

        compstd_quant_daily = std_quant_daily

        # compstd_quant_daily = std_quant_daily.apply(
        #     utils.std_conversion, axis=0, base_period=std_quant_daily.columns[0]
        # )

        self._alpha_beta  = perf.factor_alpha_beta(
            factor_data,self._periods,self._factor_returns, long_short, group_neutral
        )

        self._mean_ret_spread_quant, self._std_spread_quant = perf.compute_mean_returns_spread(
            self._mean_quant_rateret_bydate,
            factor_data["factor_quantile"].max(),
            factor_data["factor_quantile"].min(),
            std_err=compstd_quant_daily,
        )

        if self._by_group:
            (
                self._mean_return_quantile_group,
                self._mean_return_quantile_group_std_err,
            ) = perf.mean_return_by_quantile(
                factor_data,
                by_date=False,
                by_group=True,
                demeaned=long_short,
                group_adjust=group_neutral,
            )
            
            self._mean_quant_rateret_group = self._mean_return_quantile_group

            # self._mean_quant_rateret_group = self._mean_return_quantile_group.apply(
            #     utils.rate_of_return,
            #     axis=0,
            #     base_period=self._mean_return_quantile_group.columns[0],
            # )

            self._num_groups = len(
                self._mean_quant_rateret_group.index.get_level_values(
                    "group").unique()
            )

    def information_analysis(self):
        """calculate information performance
        """
        self._ic = perf.factor_information_coefficient(
            self.factor_data(), self._group_neutral)

        self._mean_monthly_ic = perf.mean_information_coefficient(
            self.factor_data(),
            group_adjust=self._group_neutral,
            by_group=False,
            by_time="M",
        )

        if self._by_group:
            self._mean_group_ic = perf.mean_information_coefficient(
                self.factor_data(), group_adjust=self._group_neutral, by_group=True
            )

    def turnover_analysis(self):
        """calculate turnover performance
        """
        # if self._turnover_periods is None:
        #     input_periods = utils.get_forward_returns_columns(
        #         self.factor_data().columns, require_exact_day_multiple=True,).to_numpy()
        #     self._turnover_periods = utils.timedelta_strings_to_integers(
        #         input_periods)
        # else:
        #     self._turnover_periods = utils.timedelta_strings_to_integers(
        #         self._turnover_periods,)

        self._turnover_periods = self._periods

        quantile_factor = self.factor_data()["factor_quantile"]

        self._quantile_turnover = {
            self._ret_types[i]: pd.concat(
                [
                    perf.quantile_turnover(quantile_factor, q, self._ret_types[i],self._turnover_periods[i])
                    for q in quantile_factor.sort_values().unique().tolist()
                ],
                axis=1,
            )
            for i in range(len(self._turnover_periods))
        }

        self._autocorrelation = pd.concat(
            [
                perf.factor_rank_autocorrelation(self.factor_data(), self._ret_types[i],self._turnover_periods[i])
                for i in range(len(self._turnover_periods))
            ],
            axis=1,
        )

    def event_analysis(self, avgretplot=(5, 15)):
        before, after = avgretplot

        self._avg_cumulative_returns = perf.average_cumulative_return_by_quantile(
            self.factor_data(),
            self.prices(),
            periods_before=before,
            periods_after=after,
            demeaned=self._long_short,
            group_adjust=self._group_neutral,
        )

        if self._by_group:

            self._avg_cumret_by_group= perf.average_cumulative_return_by_quantile(
                self.factor_data(),
                self._factor_returns,
                periods_before=before,
                periods_after=after,
                demeaned=self._long_short,
                group_adjust=self._group_neutral,
                by_group=True,
            )

    def plot_returns_table(self):
        plotting.plot_returns_table(
            self._alpha_beta , self._mean_quant_rateret, self._mean_ret_spread_quant
        )

    def plot_turnover_table(self):
        plotting.plot_turnover_table(
            self._autocorrelation, self._quantile_turnover)

    def plot_information_table(self):
        plotting.plot_information_table(self._ic)

    def plot_ic_ts(self):
        """
        Plots Spearman Rank Information Coefficient and IC moving
        average for a given factor.
        """
        columns_wide = 2
        fr_cols = len(self._ic.columns)
        rows_when_wide = ((fr_cols - 1) // columns_wide) + 1
        vertical_sections = fr_cols + 3 * rows_when_wide + 2 * fr_cols
        gf = GridFigure(rows=vertical_sections, cols=columns_wide)

        ax_ic_ts = [gf.next_row() for _ in range(fr_cols)]
        plotting.plot_ic_ts(self._ic, ax=ax_ic_ts)
        # gf.close()

    def plot_ic_hist(self):
        """
        Plots Spearman Rank Information Coefficient histogram for a given factor.
        """
        columns_wide = 1
        fr_cols = len(self._ic.columns)
        rows_when_wide = ((fr_cols - 1) // columns_wide) + 1
        vertical_sections = fr_cols + 3 * rows_when_wide + 2 * fr_cols
        gf = GridFigure(rows=vertical_sections, cols=columns_wide)

        ax_ic_hqq = [gf.next_cell() for _ in range(fr_cols * 1)]
        plotting.plot_ic_hist(self._ic, ax=ax_ic_hqq[::1])

        # # gf.close()

    def plot_ic_qq(self):
        """
        Plots Spearman Rank Information Coefficient "Q-Q" plot relative to
        a theoretical distribution.
        """
        columns_wide = 1
        fr_cols = len(self._ic.columns)
        rows_when_wide = ((fr_cols - 1) // columns_wide) + 1
        vertical_sections = fr_cols + 3 * rows_when_wide + 2 * fr_cols
        gf = GridFigure(rows=vertical_sections, cols=columns_wide)

        ax_ic_hqq = [gf.next_cell() for _ in range(fr_cols * 1)]
        plotting.plot_ic_qq(self._ic, ax=ax_ic_hqq[::1])

    def plot_quantile_statistics_table(self):

        plotting.plot_quantile_statistics_table(self.factor_data())

    def plot_quantile_returns_bar(self):
        """
        Plots mean period wise returns for factor quantiles.
        """
        fr_cols = len(self._factor_returns.columns)
        vertical_sections = 2 + fr_cols * 3
        gf = GridFigure(rows=vertical_sections, cols=1)
        plotting.plot_quantile_returns_bar(
            self._mean_quant_rateret,
            by_group=False,
            ylim_percentiles=None,
            ax=gf.next_row(),
        )
        if self._by_group:

            vertical_sections = 1 + (((self._num_groups - 1) // 2) + 1)
            gf = GridFigure(rows=vertical_sections, cols=2)

            ax_quantile_returns_bar_by_group = [
                gf.next_cell() for _ in range(self._num_groups)
            ]
            plotting.plot_quantile_returns_bar(
                self._mean_quant_rateret_group,
                by_group=True,
                ylim_percentiles=(5, 95),
                ax=ax_quantile_returns_bar_by_group,
            )
            plt.show()
        # gf.close()

    def plot_quantile_returns_violin(self):
        """
        Plots a violin box plot of period wise returns for factor quantiles.
        """
        fr_cols = len(self._factor_returns.columns)
        vertical_sections = 2 + fr_cols * 3
        gf = GridFigure(rows=vertical_sections, cols=1)
        plotting.plot_quantile_returns_violin(
            self._mean_quant_rateret_bydate, ylim_percentiles=(1, 99), ax=gf.next_row()
        )
        # gf.close()

    def plot_cumulative_returns(self):
        """Plots the cumulative returns of the returns series passed in.

        """
        fr_cols = len(self._factor_returns.columns)
        vertical_sections = 2 + fr_cols * 3
        gf = GridFigure(rows=vertical_sections, cols=1)

        for i in range(len(self._ret_types)):
            ret_type = self._ret_types[i]
            title = (
                "Factor Weighted "
                + ("Group Neutral " if self._group_neutral else "")
                + ("Long/Short " if self._long_short else "")
                + "Portfolio Cumulative Return ({})".format(ret_type)
            )
            period = '{}D'.format(self._periods[i])
            plotting.plot_cumulative_returns(
                self._factor_returns['return({})'.format(ret_type)], period=period, title=title, ax=gf.next_row(
                ), freq=str(self._periods[i]) + 'B'
            )
        # gf.close()

    def plot_mean_quantile_returns_spread_time_series(self):
        """
        Plots mean period wise returns for factor quantiles.

        """
        fr_cols = len(self._factor_returns.columns)
        vertical_sections = 2 + fr_cols * 3
        gf = GridFigure(rows=vertical_sections, cols=1)
        ax_mean_quantile_returns_spread_ts = [
            gf.next_row() for x in range(fr_cols)
        ]
        plotting.plot_mean_quantile_returns_spread_time_series(
            self._mean_ret_spread_quant,
            std_err=self._std_spread_quant,
            bandwidth=0.5,
            ax=ax_mean_quantile_returns_spread_ts,
        )
        # gf.close()

    def plot_ic_by_group(self):
        """
        Plots Spearman Rank Information Coefficient for a given factor over
        provided forward returns. Separates by group.
        """
        columns_wide = 2
        fr_cols = len(self._ic.columns)
        rows_when_wide = ((fr_cols - 1) // columns_wide) + 1
        vertical_sections = fr_cols + 3 * rows_when_wide + 2 * fr_cols
        gf = GridFigure(rows=vertical_sections, cols=columns_wide)
        plotting.plot_ic_by_group(self._mean_group_ic, ax=gf.next_row())
        # gf.close()

    def plot_factor_rank_auto_correlation(self):
        """
        Plots factor rank autocorrelation over time.
        See factor_rank_autocorrelation for more details.
        """
        fr_cols = len(self._turnover_periods)
        columns_wide = 1
        rows_when_wide = ((fr_cols - 1) // 1) + 1
        vertical_sections = fr_cols + 3 * rows_when_wide + 2 * fr_cols
        gf = GridFigure(rows=vertical_sections, cols=columns_wide)
        for ret_type in self._ret_types:
            if self._autocorrelation[ret_type].isnull().all():
                continue
            plotting.plot_factor_rank_auto_correlation(
                self._autocorrelation[ret_type],ret_type=ret_type , ax=gf.next_row()
            )
        # gf.close()

    def plot_top_bottom_quantile_turnover(self):
        """
        Plots period wise top and bottom quantile factor turnover.
        """
        fr_cols = len(self._turnover_periods)
        columns_wide = 1
        rows_when_wide = ((fr_cols - 1) // 1) + 1
        vertical_sections = fr_cols + 3 * rows_when_wide + 2 * fr_cols
        gf = GridFigure(rows=vertical_sections, cols=columns_wide)

        for ret_type in self._ret_types:
            if self._quantile_turnover[ret_type].isnull().all().all():
                continue
            plotting.plot_top_bottom_quantile_turnover(
                self._quantile_turnover[ret_type], ret_type=ret_type, ax=gf.next_row(
                )
            )
        # gf.close()

    def plot_monthly_ic_heatmap(self):
        """
        Plots a heatmap of the information coefficient or returns by month.
        """
        columns_wide = 2
        fr_cols = len(self._ic.columns)
        rows_when_wide = ((fr_cols - 1) // columns_wide) + 1
        vertical_sections = fr_cols + 3 * rows_when_wide + 2 * fr_cols
        gf = GridFigure(rows=vertical_sections, cols=columns_wide)
        ax_monthly_ic_heatmap = [gf.next_cell() for x in range(fr_cols)]
        plotting.plot_monthly_ic_heatmap(
            self._mean_monthly_ic, ax=ax_monthly_ic_heatmap)
        # gf.close()

    def plot_cumulative_returns_by_quantile(self):
        """
        Plots the cumulative returns of various factor quantiles.
        """
        fr_cols = len(self._factor_returns.columns)
        vertical_sections = 2 + fr_cols * 3
        gf = GridFigure(rows=vertical_sections, cols=1)
        for i in range(len(self._ret_types)):
            ret_type = self._ret_types[i]
            title = (
                "Factor Weighted "
                + ("Group Neutral " if self._group_neutral else "")
                + ("Long/Short " if self._long_short else "")
                + "Portfolio Cumulative Return ({})".format(ret_type)
            )
            period = '{}D'.format(self._periods[i])
            plotting.plot_cumulative_returns_by_quantile(
                self._mean_quant_ret_bydate['return({})'.format(ret_type)], period=period, ax=gf.next_row(), freq=str(self._periods[i]) + 'B', title=title)
        # gf.close()

    def plot_quantile_average_cumulative_return(self, std_bar=True):
        """
        Plots sector-wise mean daily returns for factor quantiles
        across provided forward price movement columns.
        """
        num_quantiles = int(self.factor_data()["factor_quantile"].max())
        vertical_sections = 1
        if std_bar:
            vertical_sections += ((num_quantiles - 1) // 2) + 1
        cols = 2 if num_quantiles != 1 else 1
        gf = GridFigure(rows=vertical_sections, cols=cols)
        plotting.plot_quantile_average_cumulative_return(
            self._avg_cumulative_returns,
            by_quantile=False,
            std_bar=False,
            ax=gf.next_row(),
        )
        if std_bar:
            ax_avg_cumulative_returns_by_q = [
                gf.next_cell() for _ in range(num_quantiles)
            ]
            plotting.plot_quantile_average_cumulative_return(
                self._avg_cumulative_returns,
                by_quantile=True,
                std_bar=True,
                ax=ax_avg_cumulative_returns_by_q,
            )

        if self._by_group:
            groups = self.factor_data()["group"].unique()
            num_groups = len(groups)
            vertical_sections = ((num_groups - 1) // 2) + 1
            gf = GridFigure(rows=vertical_sections, cols=2)
            for group, avg_cumret in self.__avg_cumret_by_group.groupby(level="group"):
                avg_cumret.index = avg_cumret.index.droplevel("group")
                plotting.plot_quantile_average_cumulative_return(
                    avg_cumret,
                    by_quantile=False,
                    std_bar=False,
                    title=group,
                    ax=gf.next_cell(),
                )
        # gf.close()

    def plot_events_distribution(self, num_bars=50, ax=None):
        """
        Plots the distribution of events in time.
        """
        gf = GridFigure(rows=1, cols=1)
        plotting.plot_events_distribution(
            events=self.factor_data()["factor"], num_bars=num_bars, ax=gf.next_row())
        plt.show()
        # gf.close()

import alpha_pipe.qlib as qlib
from alpha_pipe.qlib.data import D


def qcut_by_date(group, q, duplicates):
    group['factor_quantile'] = pd.qcut(group['factor'], q, labels=False, duplicates=duplicates) + 1
    return group

def qlib_to_alphalens(qlib_factor_data, quantile, duplicates):
    
    factor_data = qlib_factor_data.reset_index().rename(columns={'datetime':'date','instrument':'asset'})
    factor_data = factor_data.groupby('date').apply(qcut_by_date, q=quantile, duplicates=duplicates)
    factor_data = factor_data.sort_values(by=['date','asset']).set_index(['date','asset'])
    factor_data = factor_data.dropna()
    factor_data['group'] = factor_data['group'].apply(int).apply(str)
    
    return factor_data

class FormulaTest(BaseTest):
        
    def __init__(self, market, start_time, end_time, periods: Tuple, quantile, factor_exp, ret_exps: Tuple,ret_types: Tuple,provider_uri, region, equal_weighted=True, long_short=False, group_neutral = False, by_group=False):
        
        super().__init__()

        assert len(periods) == len(ret_exps), "len(periods) must equal len(ret_exps) !"

        self._market = market
        self._start_time = start_time
        self._end_time = end_time

        self._periods = periods
        self._ret_types = ret_types

        self._quantile = quantile
        self._factor_exp = factor_exp
        self._ret_exps = ret_exps 
        self._equal_weighted = equal_weighted
        self._long_short = long_short
        self._group_neutral = group_neutral
        self._by_group = by_group

        qlib.init(provider_uri=provider_uri, region=region)
        

    def factor_data(self):
        if self._factor_data is None:
            instruments = D.instruments(market=self._market)
            factors_ret = D.features(instruments=instruments,fields=([self._factor_exp,'$group'] + [ret_exp for ret_exp in self._ret_exps]),start_time=self._start_time, end_time=self._end_time)
            factors_ret.columns = ['factor','group'] + ['return({})'.format(ret_type) for ret_type in self._ret_types]
            factors_ret = factors_ret.dropna()
            try_num = 0
            while try_num < 10:
                try:
                    self._factor_data = qlib_to_alphalens(factors_ret, self._quantile, duplicates='raise')
                    break
                except ValueError as e:
                    # print(e)
                    self._quantile -= 1
                    print('分层数过多!!!尝试减少分层数到{}, 尝试第{}次'.format(self._quantile, try_num + 1))
                try_num += 1

        return self._factor_data


def df_to_alphalens(df_factor_data, q, duplicates):
    factor_data = df_factor_data.groupby('date').apply(qcut_by_date, q, duplicates)
    factor_data = factor_data.sort_values(by=['date','asset']).set_index(['date','asset'])
    factor_data = factor_data.dropna()
    factor_data['group'] = factor_data['group'].apply(int).apply(str)
    return factor_data

class DFTest(BaseTest):

    def __init__(self, factor_data_df: pd.DataFrame, quantile, periods, ret_types, equal_weighted=True, long_short=False, group_neutral = False, by_group=False):
        super().__init__()

        self._factor_data_df = factor_data_df
        self._quantile = quantile
        self._periods = periods
        self._ret_types = ret_types

        self._equal_weighted = equal_weighted
        self._long_short = long_short
        self._group_neutral = group_neutral
        self._by_group = by_group
    
    def factor_data(self):
        if self._factor_data is None:
            try_num = 0
            while try_num < 10:
                try:
                    self._factor_data = df_to_alphalens(self._factor_data_df, self._quantile, duplicates='raise')
                    break
                except ValueError as e:
                    # print(e)
                    self._quantile -= 1
                    print('分层数过多!!!尝试减少分层数到{}, 尝试第{}次'.format(self._quantile, try_num + 1))
                    try_num += 1

        return self._factor_data
