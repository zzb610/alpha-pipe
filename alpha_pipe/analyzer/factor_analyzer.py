# -*- coding: utf-8 -*-

from __future__ import division, print_function

from collections import Iterable
from IPython.core.display import display

import numpy as np
import pandas as pd
from fastcache import lru_cache
from scipy.stats import spearmanr, pearsonr, morestats

from . import performance as perf, plotting as pl
from .plot_utils import _use_chinese, customize
from .utils import Indicators, convert_to_forward_returns_columns, ensure_tuple, ignore_warning, quantize_factor, rate_of_returns
from .tears import GridFigure

import matplotlib.pyplot as plt


class FactorAnalyzer(object):

    def __init__(self, ret_names, quantiles, periods, binning_by_group, zero_aware):

        self._ret_names = ensure_tuple(ret_names)
        self._quantiles = quantiles
        self._periods = ensure_tuple(periods)
        self._binning_by_group = binning_by_group
        self._zero_aware = zero_aware

    def set_factor_data(self, factor_data):
        self._clean_factor_data = factor_data.copy().dropna()
        self._clean_factor_data['factor_quantile'] = quantize_factor(
            self._clean_factor_data, quantiles=self._quantiles, by_group=self._binning_by_group)
        self._clean_factor_data.dropna(inplace=True)

    @property
    def clean_factor_data(self):
        return self._clean_factor_data

    @property
    def _factor_quantile(self):
        data = self.clean_factor_data
        if not data.empty:
            return max(data.factor_quantile)
        else:
            _quantiles = self._quantiles
            _bins = self._bins
            _zero_aware = self._zero_aware
            def get_len(x): return len(x) - \
                1 if isinstance(x, Iterable) else int(x)
            if _quantiles is not None and _bins is None and not _zero_aware:
                return get_len(_quantiles)
            elif _quantiles is not None and _bins is None and _zero_aware:
                return int(_quantiles) // 2 * 2
            elif _bins is not None and _quantiles is None and not _zero_aware:
                return get_len(_bins)
            elif _bins is not None and _quantiles is None and _zero_aware:
                return int(_bins) // 2 * 2

    @lru_cache(16)
    def calc_mean_return_by_quantile(self, by_date, by_group, demeaned, group_adjust):
        """计算按分位数分组因子收益和标准差

        因子收益为收益按照 weight 列中权重的加权平均值

        参数:
        by_date:
        - True: 按天计算收益
        - False: 不按天计算收益
        by_group:
        - True: 按行业计算收益
        - False: 不按行业计算收益
        demeaned:
        - True: 使用超额收益计算各分位数收益，超额收益=收益-基准收益
                (基准收益被认为是每日所有股票收益按照weight列中权重的加权的均值)
        - False: 不使用超额收益
        group_adjust:
        - True: 使用行业中性收益计算各分位数收益，行业中性收益=收益-行业收益
                (行业收益被认为是每日各个行业股票收益按照weight列中权重的加权的均值)
        - False: 不使用行业中性收益
        """
        return perf.mean_return_by_quantile(self._clean_factor_data,
                                            by_date=by_date,
                                            by_group=by_group,
                                            demeaned=demeaned,
                                            group_adjust=group_adjust)

    @lru_cache(4)
    def calc_factor_returns(self, demeaned, group_adjust):
        """计算按因子值加权组合每日收益

        权重 = 每日因子值 / 每日因子值的绝对值的和
        正的权重代表买入, 负的权重代表卖出

        参数:
        demeaned:
        - True: 对权重去均值 (每日权重 = 每日权重 - 每日权重的均值), 使组合转换为 cash-neutral 多空组合
        - False: 不对权重去均值
        group_adjust:
        - True: 对权重分行业去均值 (每日权重 = 每日权重 - 每日各行业权重的均值)，
                使组合转换为 industry-neutral 多空组合
        - False: 不对权重分行业去均值
        """
        return perf.factor_returns(self._clean_factor_data,
                                   demeaned=demeaned,
                                   group_adjust=group_adjust)

    def compute_mean_returns_spread(self, upper_quant, lower_quant,
                                    by_date, by_group,
                                    demeaned, group_adjust):
        """计算两个分位数相减的因子收益和标准差

        参数:
        upper_quant: 用 upper_quant 选择的分位数减去 lower_quant 选择的分位数
        lower_quant: 用 upper_quant 选择的分位数减去 lower_quant 选择的分位数
        by_date:
        - True: 按天计算两个分位数相减的因子收益和标准差
        - False: 不按天计算两个分位数相减的因子收益和标准差
        by_group:
        - True: 分行业计算两个分位数相减的因子收益和标准差
        - False: 不分行业计算两个分位数相减的因子收益和标准差
        demeaned:
        - True: 使用超额收益 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
        - False: 不使用超额收益
        group_adjust:
        - True: 使用行业中性收益 (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False: 不使用行业中性收益
        """
        upper_quant = upper_quant if upper_quant is not None else self._factor_quantile
        lower_quant = lower_quant if lower_quant is not None else 1
        if ((not 1 <= upper_quant <= self._factor_quantile) or
                (not 1 <= lower_quant <= self._factor_quantile)):
            raise ValueError("upper_quant 和 low_quant 的取值范围为 1 - %s 的整数"
                             % self._factor_quantile)
        mean, std = self.calc_mean_return_by_quantile(by_date=by_date, by_group=by_group,
                                                      demeaned=demeaned, group_adjust=group_adjust,)

        return perf.compute_mean_returns_spread(mean_returns=mean,
                                                upper_quant=upper_quant,
                                                lower_quant=lower_quant,
                                                std_err=std)

    @lru_cache(4)
    def calc_factor_alpha_beta(self, demeaned, group_adjust):
        """计算因子的 alpha 和 beta

        因子值加权组合每日收益 = beta * 市场组合每日收益 + alpha

        因子值加权组合每日收益计算方法见 calc_factor_returns 函数
        市场组合每日收益是每日所有股票收益按照weight列中权重加权的均值
        结果中的 alpha 是年化 alpha

        参数:
        demeaned:
        详见 calc_factor_returns 中 demeaned 参数
        - True: 对因子值加权组合每日收益的权重去均值 (每日权重 = 每日权重 - 每日权重的均值),
                使组合转换为cash-neutral多空组合
        - False: 不对权重去均值
        group_adjust:
        详见 calc_factor_returns 中 group_adjust 参数
        - True: 对权重分行业去均值 (每日权重 = 每日权重 - 每日各行业权重的均值)，
                使组合转换为 industry-neutral 多空组合
        - False: 不对权重分行业去均值
        """
        return perf.factor_alpha_beta(self._clean_factor_data, self._periods,
                                      demeaned=demeaned,
                                      group_adjust=group_adjust)

    @lru_cache(8)
    def calc_factor_information_coefficient(self, group_adjust, by_group, method):
        """计算每日因子信息比率 (IC值)

        参数:
        group_adjust:
        - True: 使用行业中性收益计算 IC (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False: 不使用行业中性收益
        by_group:
        - True: 分行业计算 IC
        - False: 不分行业计算 IC
        method:
        - 'rank': 用秩相关系数计算IC值
        - 'normal': 用普通相关系数计算IC值
        """
        if method is None:
            method = 'rank'
        if method not in ('rank', 'normal'):
            raise ValueError(
                "`method` should be chosen from ('rank' | 'normal')")

        if method == 'rank':
            method = spearmanr
        elif method == 'normal':
            method = pearsonr
        return perf.factor_information_coefficient(self._clean_factor_data,
                                                   group_adjust=group_adjust,
                                                   by_group=by_group,
                                                   method=method)

    @lru_cache(16)
    def calc_mean_information_coefficient(self, group_adjust, by_group, by_time, method):
        """计算因子信息比率均值 (IC值均值)

        参数:
        group_adjust:
        - True: 使用行业中性收益计算 IC (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False: 不使用行业中性收益
        by_group:
        - True: 分行业计算 IC
        - False: 不分行业计算 IC
        by_time:
        - 'Y': 按年求均值
        - 'M': 按月求均值
        - None: 对所有日期求均值
        method:
        - 'rank': 用秩相关系数计算IC值
        - 'normal': 用普通相关系数计算IC值
        """
        if method is None:
            method = 'rank'
        if method not in ('rank', 'normal'):
            raise ValueError(
                "`method` should be chosen from ('rank' | 'normal')")

        if method == 'rank':
            method = spearmanr
        elif method == 'normal':
            method = pearsonr
        return perf.mean_information_coefficient(
            self._clean_factor_data,
            group_adjust=group_adjust,
            by_group=by_group,
            by_time=by_time,
            method=method
        )

    @lru_cache(16)
    def calc_average_cumulative_return_by_quantile(self, ret_name, periods,periods_before, periods_after, demeaned, group_adjust):
        """按照当天的分位数算分位数未来和过去的收益均值和标准差

        参数:
        periods_before: 计算过去的天数
        periods_after: 计算未来的天数
        demeaned:
        - True: 使用超额收益计算累积收益 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
        - False: 不使用超额收益
        group_adjust:
        - True: 使用行业中性化后的收益计算累积收益
                (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False: 不使用行业中性化后的收益
        """
        returns = self._clean_factor_data[[
            'return({})'.format(ret_name)]].unstack()
        cols = tuple(returns.columns.get_level_values(1))
        returns = pd.DataFrame(returns)
        returns.columns = cols

        return perf.average_cumulative_return_by_quantile(
            self._clean_factor_data,
            returns=returns,
            periods=periods,
            periods_before=periods_before,
            periods_after=periods_after,
            demeaned=demeaned,
            group_adjust=group_adjust
        )

    @lru_cache(2)
    def calc_autocorrelation(self, rank):
        """根据调仓周期确定滞后期的每天计算因子自相关性

        当日因子值和滞后period天的因子值的自相关性

        参数:
        rank:
        - True: 秩相关系数
        - False: 普通相关系数
        """
        return {name: perf.factor_autocorrelation(self._clean_factor_data, p, rank=rank) for name, p in zip(self._ret_names, self._periods)}

    @lru_cache(None)
    def calc_quantile_turnover_mean_n_days_lag(self, n):
        """各分位数滞后1天到n天的换手率均值

        参数:
        n: 滞后 1 天到 n 天的换手率
        """
        quantile_factor = self._clean_factor_data['factor_quantile']

        quantile_turnover_rate = pd.concat(
            [pd.Series([perf.quantile_turnover(quantile_factor, q, p).mean()
                        for q in range(1, int(quantile_factor.max()) + 1)],
                       index=range(1, int(quantile_factor.max()) + 1),
                       name=p)
             for p in range(1, n + 1)],
            axis=1, keys='lag_' + pd.Index(range(1, n + 1)).astype(str)
        ).T
        quantile_turnover_rate.columns.name = 'factor_quantile'

        return quantile_turnover_rate

    @lru_cache(None)
    def calc_autocorrelation_n_days_lag(self, n, rank):
        """滞后1-n天因子值自相关性

        参数:
        n: 滞后1天到n天的因子值自相关性
        rank:
        - True: 秩相关系数
        - False: 普通相关系数
        """
        return pd.Series(
            [
                perf.factor_autocorrelation(
                    self._clean_factor_data, p, rank=rank).mean()
                for p in range(1, n + 1)
            ],
            index='lag_' + pd.Index(range(1, n + 1)).astype(str)
        )

    @lru_cache(None)
    def _calc_ic_mean_n_day_lag(self, n, group_adjust, by_group, method):
        if method is None:
            method = 'rank'
        if method not in ('rank', 'normal'):
            raise ValueError(
                "`method` should be chosen from ('rank' | 'normal')")

        if method == 'rank':
            method = spearmanr
        elif method == 'normal':
            method = pearsonr

        factor_data = self._clean_factor_data.copy()
        factor_value = factor_data['factor'].unstack('asset')

        factor_data['factor'] = factor_value.shift(n).stack(dropna=True)
        if factor_data.dropna().empty:
            return pd.Series(np.nan, index=perf.get_forward_returns_columns(factor_data.columns))
        ac = perf.factor_information_coefficient(
            factor_data.dropna(),
            group_adjust=group_adjust, by_group=by_group,
            method=method
        )
        return ac.mean(level=('group' if by_group else None))

    def calc_ic_mean_n_days_lag(self, n, group_adjust, by_group, method):
        """滞后 0 - n 天因子收益信息比率(IC)的均值

        滞后 n 天 IC 表示使用当日因子值和滞后 n 天的因子收益计算 IC

        参数:
        n: 滞后0-n天因子收益的信息比率(IC)的均值
        group_adjust:
        - True: 使用行业中性收益计算 IC (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False: 不使用行业中性收益
        by_group:
        - True: 分行业计算 IC
        - False: 不分行业计算 IC
        method:
        - 'rank': 用秩相关系数计算IC值
        - 'normal': 用普通相关系数计算IC值
        """
        ic_mean = [self.calc_factor_information_coefficient(
            group_adjust=group_adjust, by_group=by_group, method=method,
        ).mean(level=('group' if by_group else None))]

        for lag in range(1, n + 1):
            ic_mean.append(self._calc_ic_mean_n_day_lag(
                n=lag,
                group_adjust=group_adjust,
                by_group=by_group,
                method=method
            ))
        if not by_group:
            ic_mean = pd.concat(ic_mean, keys='lag_' +
                                pd.Index(range(n + 1)).astype(str), axis=1)
            ic_mean = ic_mean.T
        else:
            ic_mean = pd.concat(ic_mean, keys='lag_' +
                                pd.Index(range(n + 1)).astype(str), axis=0)
        return ic_mean

    @lru_cache(5)
    def calc_cumulative_return_by_quantile(self, ret_name, period, demeaned, group_adjust):
        """计算指定调仓周期的各分位数每日累积收益

        参数:
        period: 指定调仓周期
        demeaned:
        详见 calc_mean_return_by_quantile 中 demeaned 参数
        - True: 使用超额收益计算累积收益 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
        - False: 不使用超额收益
        group_adjust:
        详见 calc_mean_return_by_quantile 中 group_adjust 参数
        - True: 使用行业中性化后的收益计算累积收益
                (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False: 不使用行业中性化后的收益
        """

        ret_name = 'return({})'.format(ret_name)
        quantile_returns, _ = self.calc_mean_return_by_quantile(
            by_date=True, by_group=False, demeaned=demeaned, group_adjust=group_adjust
        )
        quantile_returns = quantile_returns[ret_name]
        ret_wide = quantile_returns.unstack('factor_quantile')

        quantiles = ret_wide.columns
        ret_wide['top_bottom'] = ret_wide[max(
            quantiles)] - ret_wide[min(quantiles)]
        if ret_wide['top_bottom'].sum() < 0:
            ret_wide['top_bottom'] = -ret_wide['top_bottom']

        # from IPython.display import display
        # display(ret_wide)

        # freq = '{}B'.format(period)
        # ret_wide = ret_wide.reset_index().set_index('date')
        # ret_wide = ret_wide.resample(freq).first()

        cum_ret = ret_wide.apply(perf.cumulative_returns, period=period, axis=0)
 
 
        return cum_ret

    @lru_cache(20)
    def calc_cumulative_returns(self, ret_name, period, demeaned, group_adjust):
        """计算指定调仓周期的按因子值加权组合每日累积收益

        当 period > 1 时，组合的累积收益计算方法为：
        组合每日收益 = （从第0天开始每period天一调仓的组合每日收益 +
                        从第1天开始每period天一调仓的组合每日收益 + ... +
                        从第period-1天开始每period天一调仓的组合每日收益) / period
        组合累积收益 = 组合每日收益的累积

        参数:
        period: 指定调仓周期
        demeaned:
        详见 calc_factor_returns 中 demeaned 参数
        - True: 对权重去均值 (每日权重 = 每日权重 - 每日权重的均值), 使组合转换为 cash-neutral 多空组合
        - False: 不对权重去均值
        group_adjust:
        详见 calc_factor_returns 中 group_adjust 参数
        - True: 对权重分行业去均值 (每日权重 = 每日权重 - 每日各行业权重的均值)，
                使组合转换为 industry-neutral 多空组合
        - False: 不对权重分行业去均值
        """
        ret_name = 'return({})'.format(ret_name)
        factor_returns = self.calc_factor_returns(
            demeaned=demeaned, group_adjust=group_adjust)[ret_name]

        # freq = '{}B'.format(period)
        # factor_returns = factor_returns.reset_index().set_index('date')
        # factor_returns = factor_returns.resample(freq).first()
        # from IPython.display import display
        # display(factor_returns)
        cum_ret = perf.cumulative_returns(factor_returns, period=period)
        return cum_ret

    @lru_cache(20)
    def calc_top_down_cumulative_returns(self, ret_name, period, demeaned, group_adjust):
        """计算做多最大分位，做空最小分位组合每日累积收益

        参数:
        period: 指定调仓周期
        demeaned:
        详见 calc_mean_return_by_quantile 中 demeaned 参数
        - True: 使用超额收益计算累积收益 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
        - False: 不使用超额收益
        group_adjust:
        详见 calc_mean_return_by_quantile 中 group_adjust 参数
        - True: 使用行业中性化后的收益计算累积收益
                (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False: 不使用行业中性化后的收益
        """

        ret_name = 'return({})'.format(ret_name)
        quantile_returns, _ = self.calc_mean_return_by_quantile(
            by_date=True, by_group=False, demeaned=demeaned, group_adjust=group_adjust
        )
        quantile_returns = quantile_returns[ret_name]
        ret_wide = quantile_returns.unstack('factor_quantile')

        quantiles = ret_wide.columns
        ret_wide = ret_wide[max(
            quantiles)] - ret_wide[min(quantiles)]
        if ret_wide.sum() < 0:
            ret_wide = -ret_wide

        # freq = '{}B'.format(period)
        # ret_wide = ret_wide.reset_index().set_index('date')
        # ret_wide = ret_wide.resample(freq).first()
        cum_ret = perf.cumulative_returns(ret_wide, period=period)
        cum_ret.columns = [ret_name]
        return cum_ret

    def plot_returns_table(self, demeaned, group_adjust):
        """打印因子收益表

        参数:
        demeaned:
        - True: 使用超额收益计算 (基准收益被认为是每日所有股票收益按照weight列中权重的加权的均值)
        - False: 不使用超额收益
        group_adjust:
        - True: 使用行业中性收益 (行业收益被认为是每日各个行业股票收益按照weight列中权重的加权的均值)
        - False: 不使用行业中性收益
        """

        mean_return_by_quantile, _ = self.calc_mean_return_by_quantile(
            by_date=False, by_group=False,
            demeaned=demeaned, group_adjust=group_adjust,
        )

        mean_returns_spread, _ = self.compute_mean_returns_spread(
            upper_quant=self._factor_quantile,
            lower_quant=1,
            by_date=True,
            by_group=False,
            demeaned=demeaned,
            group_adjust=group_adjust,
        )

        pl.plot_returns_table(
            self.calc_factor_alpha_beta(
                demeaned=demeaned, group_adjust=group_adjust),
            mean_return_by_quantile,
            mean_returns_spread
        )

    def plot_turnover_table(self, rank):
        """打印换手率表"""
        quantile_factor = self._clean_factor_data['factor_quantile']
        quantile_turnover_rate = {
            name:
            pd.concat([perf.quantile_turnover(quantile_factor, q, p)
                       for q in range(1, int(quantile_factor.max()) + 1)],
                      axis=1)
            for name, p in zip(self._ret_names, self._periods)
        }
        pl.plot_turnover_table(
            self.calc_autocorrelation(rank),
            quantile_turnover_rate
        )

    def plot_information_table(self, group_adjust, method):
        """打印信息比率 (IC)相关表

        参数:
        group_adjust:
        - True：使用行业中性收益 (行业收益被认为是每日各个行业股票收益按照weight列中权重的加权的均值)
        - False：不使用行业中性收益
        method：
        - 'rank'：用秩相关系数计算IC值
        - 'normal':用相关系数计算IC值
        """
        ic = self.calc_factor_information_coefficient(
            group_adjust=group_adjust,
            by_group=False,
            method=method
        )
        pl.plot_information_table(ic)

    def plot_quantile_statistics_table(self):
        """打印各分位数统计表"""
        pl.plot_quantile_statistics_table(self._clean_factor_data)

    def plot_ic_ts(self, group_adjust, method):
        """画信息比率(IC)时间序列图

        参数:
        group_adjust:
        - True: 使用行业中性收益 (行业收益被认为是每日各个行业股票收益按照weight列中权重的加权的均值)
        - False: 不使用行业中性收益
        method:
        - 'rank': 用秩相关系数计算IC值
        - 'normal':用相关系数计算IC值
        """
        ic = self.calc_factor_information_coefficient(
            group_adjust=group_adjust, by_group=False, method=method
        )
        pl.plot_ic_ts(ic)

    def plot_ic_hist(self, group_adjust, method):
        """画信息比率分布直方图

        参数:
        group_adjust:
        - True: 使用行业中性收益 (行业收益被认为是每日各个行业股票收益按照weight列中权重的加权的均值)
        - False: 不使用行业中性收益
        method:
        - 'rank': 用秩相关系数计算IC值
        - 'normal': 用相关系数计算IC值
        """
        ic = self.calc_factor_information_coefficient(
            group_adjust=group_adjust,
            by_group=False,
            method=method
        )
        pl.plot_ic_hist(ic)

    def plot_ic_qq(self, group_adjust, method, theoretical_dist):
        """画信息比率 qq 图

        参数:
        group_adjust:
        - True: 使用行业中性收益 (行业收益被认为是每日各个行业股票收益按照weight列中权重的加权的均值)
        - False: 不使用行业中性收益
        method:
        - 'rank': 用秩相关系数计算IC值
        - 'normal': 用相关系数计算IC值
        theoretical_dist:
        - 'norm': 正态分布
        - 't': t 分布
        """
        theoretical_dist = 'norm' if theoretical_dist is None else theoretical_dist
        theoretical_dist = morestats._parse_dist_kw(theoretical_dist)
        ic = self.calc_factor_information_coefficient(
            group_adjust=group_adjust,
            by_group=False,
            method=method,
        )

        columns_wide = 1
        fr_cols = len(ic.columns)
        rows_when_wide = ((fr_cols - 1) // columns_wide) + 1
        vertical_sections = fr_cols + 3 * rows_when_wide + 2 * fr_cols
        gf = GridFigure(rows=vertical_sections, cols=columns_wide)
        ax_ic_hqq = [gf.next_cell() for _ in range(fr_cols * 1)]

        pl.plot_ic_qq(ic, theoretical_dist=theoretical_dist, ax=ax_ic_hqq[::1])

    def plot_quantile_returns_bar(self, by_group, demeaned, group_adjust):
        """画各分位数平均收益图

        参数:
        by_group:
        - True: 各行业的各分位数平均收益图
        - False: 各分位数平均收益图
        demeaned:
        - True: 使用超额收益计算累积收益 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
        - False: 不使用超额收益
        group_adjust:
        - True: 使用行业中性化后的收益计算累积收益
                (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False: 不使用行业中性化后的收益
        """

        mean_return_by_quantile, _ = self.calc_mean_return_by_quantile(
            by_date=False, by_group=by_group,
            demeaned=demeaned, group_adjust=group_adjust,
        )
        
        # from IPython.display import display
        # display(mean_return_by_quantile)

        for name, period in zip(self._ret_names, self._periods):
            mean_return_by_quantile['return({})'.format(name)] = mean_return_by_quantile['return({})'.format(name)].apply(rate_of_returns, period=period)

        # from IPython.display import display
        # display(mean_return_by_quantile)

        pl.plot_quantile_returns_bar(
            mean_return_by_quantile, by_group=by_group, ylim_percentiles=None
        )

    def plot_quantile_returns_violin(self, demeaned, group_adjust,
                                     ylim_percentiles=(1, 99)):
        """画各分位数收益分布图

        参数:
        demeaned:
        - True: 使用超额收益计算累积收益 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
        - False: 不使用超额收益
        group_adjust:
        - True: 使用行业中性化后的收益计算累积收益
                (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False: 不使用行业中性化后的收益
        plot_quantile_returns_violin: 有效收益分位数(单位为百分之). 画图时y轴的范围为有效收益的最大/最小值.
                                      例如 (1, 99) 代表收益的从小到大排列的 1% 分位到 99% 分位为有效收益.
        """
        mean_return_by_quantile, _ = self.calc_mean_return_by_quantile(
            by_date=True, by_group=False,
            demeaned=demeaned, group_adjust=group_adjust
        )
        for name, period in zip(self._ret_names, self._periods):
            mean_return_by_quantile['return({})'.format(name)] = mean_return_by_quantile['return({})'.format(name)].apply(rate_of_returns, period=period)
        pl.plot_quantile_returns_violin(mean_return_by_quantile,
                                        ylim_percentiles=ylim_percentiles)
                                        

    def plot_mean_quantile_returns_spread_time_series(
        self, demeaned, group_adjust, bandwidth=1
    ):
        """画最高分位减最低分位收益图

        参数:
        demeaned:
        - True: 使用超额收益计算累积收益 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
        - False: 不使用超额收益
        group_adjust:
        - True: 使用行业中性化后的收益计算累积收益
                (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False: 不使用行业中性化后的收益
        bandwidth: n, 加减 n 倍当日标准差
        """
        mean_returns_spread, mean_returns_spread_std = self.compute_mean_returns_spread(
            upper_quant=self._factor_quantile,
            lower_quant=1,
            by_date=True,
            by_group=False,
            demeaned=demeaned,
            group_adjust=group_adjust,
        )

        pl.plot_mean_quantile_returns_spread_time_series(
            mean_returns_spread, std_err=mean_returns_spread_std,
            bandwidth=bandwidth
        )

    def plot_ic_by_group(self, group_adjust, method):
        """画按行业分组信息比率(IC)图

        参数:
        group_adjust:
        - True: 使用行业中性收益 (行业收益被认为是每日各个行业股票收益按照weight列中权重的加权的均值)
        - False: 不使用行业中性收益
        method:
        - 'rank': 用秩相关系数计算IC值
        - 'normal': 用相关系数计算IC值
        """
        ic_by_group = self.calc_mean_information_coefficient(
            group_adjust=group_adjust,
            by_group=True,
            by_time=None,
            method=method
        )
        pl.plot_ic_by_group(ic_by_group)

    def plot_factor_auto_correlation(self, rank):
        """画因子自相关图

        参数:
        periods: 滞后周期
        rank:
        - True: 用秩相关系数
        - False: 用相关系数
        """

        for name in self._ret_names:
            pl.plot_factor_rank_auto_correlation(
                self.calc_autocorrelation(rank=rank)[name], name)

    def plot_top_bottom_quantile_turnover(self):
        """画最高最低分位换手率图

        参数:
        periods: 调仓周期
        """
        quantile_factor = self._clean_factor_data['factor_quantile']
        quantile_turnover_rate = {
            name:
            pd.concat([perf.quantile_turnover(quantile_factor, q, p)
                       for q in range(1, int(quantile_factor.max()) + 1)],
                      axis=1)
            for name, p in zip(self._ret_names, self._periods)
        }
        for name in self._ret_names:
            pl.plot_top_bottom_quantile_turnover(
                quantile_turnover_rate[name], name)

   
    def plot_monthly_ic_heatmap(self, group_adjust, ic_method):
        """画月度信息比率(IC)图

        参数:
        group_adjust:
        - True: 使用行业中性收益 (行业收益被认为是每日各个行业股票收益按照weight列中权重的加权的均值)
        - False: 不使用行业中性收益
        """
        ic_monthly = self.calc_mean_information_coefficient(
            group_adjust=group_adjust, by_group=False, by_time="M", method=ic_method
        )
        pl.plot_monthly_ic_heatmap(ic_monthly)

    @customize
    def plot_cumulative_returns(self, demeaned, group_adjust):
        """画按因子值加权组合每日累积收益图

        参数:
        periods: 调仓周期
        demeaned:
        详见 calc_factor_returns 中 demeaned 参数
        - True: 对因子值加权组合每日收益的权重去均值 (每日权重 = 每日权重 - 每日权重的均值),
                使组合转换为cash-neutral多空组合
        - False: 不对权重去均值
        group_adjust:
        详见 calc_factor_returns 中 group_adjust 参数
        - True: 对权重分行业去均值 (每日权重 = 每日权重 - 每日各行业权重的均值)，
                使组合转换为 industry-neutral 多空组合
        - False: 不对权重分行业去均值
        """
        f, ax = plt.subplots(1, 1, figsize=(18, 6))
        for name, period in zip(self._ret_names, self._periods):
            cum_ret = self.calc_cumulative_returns(
                name, period, demeaned, group_adjust)

            # cum_ret.plot(ax=ax, lw=3, alpha=0.6, label=name)
            ax.plot(cum_ret, lw=3, alpha=0.6, label=name)
        ax.set(title='因子值加权累积收益')
        ax.legend()
        ax.axhline(1.0, linestyle='-', color='black', lw=1)

    @customize
    def plot_top_down_cumulative_returns(self, demeaned, group_adjust):
        """画做多最大分位数做空最小分位数组合每日累积收益图

        period: 指定调仓周期
        demeaned:
        详见 calc_mean_return_by_quantile 中 demeaned 参数
        - True: 使用超额收益计算累积收益 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
        - False: 不使用超额收益
        group_adjust:
        详见 calc_mean_return_by_quantile 中 group_adjust 参数
        - True: 使用行业中性化后的收益计算累积收益
                (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False: 不使用行业中性化后的收益
        """
        f, ax = plt.subplots(1, 1, figsize=(18, 6))
        for name, period in zip(self._ret_names, self._periods):
            cum_ret = self.calc_top_down_cumulative_returns(
                name, period, demeaned, group_adjust)
            cum_ret.plot(ax=ax, lw=3, alpha=0.6, label=name)
        ax.set(title='最高层最低层多空对冲收益')
        ax.axhline(1.0, linestyle='-', color='black', lw=1)
        ax.legend()

    @customize
    def plot_cumulative_returns_by_quantile(self, demeaned, group_adjust):
        """画各分位数每日累积收益图

        参数:
        demeaned:
        详见 calc_mean_return_by_quantile 中 demeaned 参数
        - True: 使用超额收益计算累积收益 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
        - False: 不使用超额收益
        group_adjust:
        详见 calc_mean_return_by_quantile 中 group_adjust 参数
        - True: 使用行业中性化后的收益计算累积收益
                (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False: 不使用行业中性化后的收益
        """
        for name, period in zip(self._ret_names, self._periods):
            f, ax = plt.subplots(1, 1, figsize=(18, 6))
            cum_ret = self.calc_cumulative_return_by_quantile(
                name, period, demeaned, group_adjust)
            cum_ret.plot(ax=ax, lw=2, alpha=0.6)

            ax.axhline(1.0, linestyle='-', color='black', lw=1)
            ax.set(title='分层累积收益 {} '.format(name))
    
    def plot_cumulative_returns_indicators(self, demeaned, group_adjust):
        """
        夏普比等收益指标
        """
        for name, period in zip(self._ret_names, self._periods):
            cum_ret = self.calc_cumulative_return_by_quantile(
                name, period, demeaned, group_adjust)
            quantiles = set(pd.DataFrame(cum_ret).columns)
            quantiles.remove('top_bottom')
            ret = cum_ret[['top_bottom',min(quantiles), max(quantiles)]]
            indicators = pd.DataFrame()
            print('{} 收益指标 {} 天调仓'.format(name, period))
            for col in ret.columns:
                indicator = Indicators(ret[col])
                indicator.name = col
                indicators = indicators.append(indicator)
            display(indicators)

      

    def plot_quantile_average_cumulative_return(self, periods,periods_before, periods_after,
                                                by_quantile, std_bar,
                                                demeaned, group_adjust):
        """因子预测能力平均累计收益图

        参数:
        periods_before: 计算过去的天数
        periods_after: 计算未来的天数
        by_quantile: 是否各分位数分别显示因子预测能力平均累计收益图
        std_bar:
        - True: 显示标准差
        - False: 不显示标准差
        demeaned:
        详见 calc_mean_return_by_quantile 中 demeaned 参数
        - True: 使用超额收益计算累积收益 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
        - False: 不使用超额收益
        group_adjust:
        详见 calc_mean_return_by_quantile 中 group_adjust 参数
        - True: 使用行业中性化后的收益计算累积收益
                (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False: 不使用行业中性化后的收益
        """
        for name in self._ret_names:
            average_cumulative_return_by_q = self.calc_average_cumulative_return_by_quantile(
                name,
                periods,
                periods_before=periods_before, periods_after=periods_after,
                demeaned=demeaned, group_adjust=group_adjust
            )
            pl.plot_quantile_average_cumulative_return(average_cumulative_return_by_q,
                                                       by_quantile=by_quantile,
                                                       std_bar=std_bar,
                                                       periods_before=periods_before,
                                                       periods_after=periods_after)

    def plot_events_distribution(self, num_days):
        """画有效因子数量统计图

        参数:
        num_days: 统计间隔天数
        """
        pl.plot_events_distribution(
            events=self._clean_factor_data['factor'],
            num_days=num_days,
            full_dates=pd.to_datetime(
                self.factor.index.get_level_values('date').unique())
        )

    def create_summary_tear_sheet(self, demeaned, group_adjust):
        """因子值特征分析

        参数:
        demeaned:
        - True: 对每日因子收益去均值求得因子收益表
        - False: 因子收益表
        group_adjust:
        - True: 按行业对因子收益去均值后求得因子收益表
        - False: 因子收益表
        """
        self.plot_quantile_statistics_table()
        self.plot_returns_table(demeaned=demeaned, group_adjust=group_adjust)
        self.plot_quantile_returns_bar(
            by_group=False, demeaned=demeaned, group_adjust=group_adjust)
        pl.plt.show()
        self.plot_information_table(group_adjust=group_adjust)
        self.plot_turnover_table()

    def create_returns_tear_sheet(self, demeaned, group_adjust, by_group):
        """因子值特征分析

        参数:
        demeaned:
        详见 calc_mean_return_by_quantile 中 demeaned 参数
        - True: 使用超额收益计算 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
        - False: 不使用超额收益
        group_adjust:
        详见 calc_mean_return_by_quantile 中 group_adjust 参数
        - True: 使用行业中性化后的收益计算 (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False: 不使用行业中性化后的收益
        by_group:
        - True: 画各行业的各分位数平均收益图
        - False: 不画各行业的各分位数平均收益图
        """
        self.plot_returns_table(demeaned=demeaned, group_adjust=group_adjust)
        self.plot_quantile_returns_bar(by_group=False,
                                       demeaned=demeaned,
                                       group_adjust=group_adjust)
        pl.plt.show()
        self.plot_cumulative_returns(
            demeaned=demeaned, group_adjust=group_adjust)
        pl.plt.show()
        self.plot_cumulative_returns_by_quantile(
            demeaned=demeaned, group_adjust=group_adjust)
        self.plot_cumulative_returns_indicators(demeaned=demeaned, group_adjust=group_adjust)
        self.plot_top_down_cumulative_returns(
            demeaned=demeaned, group_adjust=group_adjust)
        pl.plt.show()
        self.plot_mean_quantile_returns_spread_time_series(
            demeaned=demeaned, group_adjust=group_adjust
        )
        pl.plt.show()
        if by_group:
            self.plot_quantile_returns_bar(by_group=True,
                                           demeaned=demeaned,
                                           group_adjust=group_adjust)
            pl.plt.show()

        self.plot_quantile_returns_violin(demeaned=demeaned,
                                          group_adjust=group_adjust)
        pl.plt.show()

    def create_information_tear_sheet(self, group_adjust, by_group, ic_method, theoretical_dist):
        """因子 IC 分析

        参数:
        group_adjust:
        - True: 使用行业中性收益 (行业收益被认为是每日各个行业股票收益按照weight列中权重的加权的均值)
        - False: 不使用行业中性收益
        by_group:
        - True: 画按行业分组信息比率(IC)图
        - False: 画月度信息比率(IC)图
        """
        self.plot_ic_ts(group_adjust=group_adjust, method=ic_method)
        pl.plt.show()
        self.plot_ic_qq(group_adjust=group_adjust,
                        method=ic_method, theoretical_dist=theoretical_dist)
        pl.plt.show()
        if by_group:
            self.plot_ic_by_group(group_adjust=group_adjust, method=ic_method)
        else:
            self.plot_monthly_ic_heatmap(
                group_adjust=group_adjust, ic_method=ic_method)
        pl.plt.show()

    def create_turnover_tear_sheet(self, rank):
        """因子换手率分析

        参数:
        turnover_periods: 调仓周期
        """
        self.plot_turnover_table(rank)
        self.plot_top_bottom_quantile_turnover()
        pl.plt.show()
        self.plot_factor_auto_correlation(rank)
        pl.plt.show()

    def create_event_returns_tear_sheet(self, avgretplot,
                                        demeaned, group_adjust,
                                        std_bar):
        """因子预测能力分析

        参数:
        avgretplot: tuple 因子预测的天数
        -(计算过去的天数, 计算未来的天数)
        demeaned:
        详见 calc_mean_return_by_quantile 中 demeaned 参数
        - True: 使用超额收益计算累积收益 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
        - False: 不使用超额收益
        group_adjust:
        详见 calc_mean_return_by_quantile 中 group_adjust 参数
        - True: 使用行业中性化后的收益计算累积收益
                (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False: 不使用行业中性化后的收益
        std_bar:
        - True: 显示标准差
        - False: 不显示标准差
        """
        before, after = avgretplot
        self.plot_quantile_average_cumulative_return(
            periods_before=before, periods_after=after,
            by_quantile=False, std_bar=False,
            demeaned=demeaned, group_adjust=group_adjust
        )
        pl.plt.show()
        if std_bar:
            self.plot_quantile_average_cumulative_return(
                periods_before=before, periods_after=after,
                by_quantile=True, std_bar=True,
                demeaned=demeaned, group_adjust=group_adjust
            )
            pl.plt.show()

    def create_full_tear_sheet(self, demeaned, group_adjust, by_group, avgretplot, ic_method, ic_theoretical_dist, turnover_rank, std_bar):
        """全部分析

        参数:
        demeaned:
        - True：使用超额收益计算 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
        - False：不使用超额收益
        group_adjust:
        - True：使用行业中性化后的收益计算
                (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False：不使用行业中性化后的收益
        by_group:
        - True: 按行业展示
        - False: 不按行业展示
        turnover_periods: 调仓周期
        avgretplot: tuple 因子预测的天数
        -(计算过去的天数, 计算未来的天数)
        std_bar:
        - True: 显示标准差
        - False: 不显示标准差
        """
        self.plot_quantile_statistics_table()
        print("\n-------------------------\n")
        self.plot_returns_table(demeaned=demeaned, group_adjust=group_adjust)
        self.plot_quantile_returns_bar(by_group=False,
                                       demeaned=demeaned,
                                       group_adjust=group_adjust)
        pl.plt.show()
        self.plot_cumulative_returns(
            demeaned=demeaned, group_adjust=group_adjust)
        pl.plt.show()
        self.plot_top_down_cumulative_returns(
            demeaned=demeaned,
            group_adjust=group_adjust)
        pl.plt.show()
        self.plot_cumulative_returns_by_quantile(
            demeaned=demeaned,
            group_adjust=group_adjust)
        self.plot_cumulative_returns_indicators(demeaned=demeaned, group_adjust=group_adjust)
        self.plot_mean_quantile_returns_spread_time_series(demeaned=demeaned,
                                                           group_adjust=group_adjust)
        pl.plt.show()
        if by_group:
            self.plot_quantile_returns_bar(by_group=True,
                                           demeaned=demeaned,
                                           group_adjust=group_adjust)
            pl.plt.show()
        self.plot_quantile_returns_violin(demeaned=demeaned,
                                          group_adjust=group_adjust)
        pl.plt.show()
        print("\n-------------------------\n")
        self.plot_information_table(
            group_adjust=group_adjust, method=ic_method)
        self.plot_ic_ts(group_adjust=group_adjust, method=ic_method)
        pl.plt.show()
        self.plot_ic_qq(group_adjust=group_adjust, method=ic_method,
                        theoretical_dist=ic_theoretical_dist)
        pl.plt.show()
        if by_group:
            self.plot_ic_by_group(group_adjust=group_adjust, method=ic_method)
        else:
            self.plot_monthly_ic_heatmap(
                group_adjust=group_adjust, ic_method=ic_method)
        pl.plt.show()
        print("\n-------------------------\n")
        self.plot_turnover_table(turnover_rank)
        self.plot_top_bottom_quantile_turnover()
        pl.plt.show()
        self.plot_factor_auto_correlation(turnover_rank)
        pl.plt.show()
        print("\n-------------------------\n")
        # before, after = avgretplot
        # self.plot_quantile_average_cumulative_return(
        #     periods_before=before, periods_after=after,periods=self._periods
        #     by_quantile=False, std_bar=False,
        #     demeaned=demeaned, group_adjust=group_adjust
        # )
        # pl.plt.show()
        # if std_bar:
        #     self.plot_quantile_average_cumulative_return(
        #         periods_before=before, periods_after=after,
        #         by_quantile=True, std_bar=True,
        #         demeaned=demeaned, group_adjust=group_adjust
        #     )
        #     pl.plt.show()

    def plot_disable_chinese_label(self):
        """关闭中文图例显示

        画图时默认会从系统中查找中文字体显示以中文图例
        如果找不到中文字体则默认使用英文图例
        当找到中文字体但中文显示乱码时, 可调用此 API 关闭中文图例显示而使用英文
        """
        _use_chinese(False)
