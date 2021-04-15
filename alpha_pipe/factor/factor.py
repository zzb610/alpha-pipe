import abc

from IPython.core.display import display
from alpha_pipe.qlib_data.data import LocalProvider, LocalInstrumentProvider
from alpha_pipe.qlib_data.config import QlibConfig, _default_config

import datetime as dt
import gc
import pandas as pd


class BaseFactor(metaclass=abc.ABCMeta):

    def __init__(self):
        self._factor_data = None

    def _clear_factor(self):
        self._factor_data = None
        gc.collect()

    @abc.abstractmethod
    def _load_factor_data(self):
        pass

    def factor_data(self):
        if self._factor_data is None:
            self._factor_data = self._load_factor_data()
        return self._factor_data

    def to_csv(self, target_dir):

        self._factor_data.to_csv(target_dir)


class ExpressionFactor(BaseFactor):

    def __init__(self, market, start_time, end_time, freq, factor_exp, ret_exps, ret_names, provider_uri):
        super().__init__()

        self._market = market
        self._start_time = start_time
        self._end_time = end_time
        self._freq = freq

        self._factor_exp = factor_exp
        self._ret_exps = ret_exps
        self._ret_names = ret_names

        self.raw_data = None

        _config = _default_config
        _config['provider_uri'] = provider_uri
        config = QlibConfig(_config)
        config.register()

        self._data_provider = LocalProvider()
        self._inst_provider = LocalInstrumentProvider()

    def _load_factor_data(self):

        instruments = self._inst_provider.instruments(market=self._market)

        if self._freq == 'min':

            print('开始计算因子...')
            start_time = dt.datetime.now()
            factors_values = self._data_provider.features(instruments=instruments, fields=[
                self._factor_exp, '$group'], start_time=self._start_time, end_time=self._end_time, freq=self._freq)
            ret_values = self._data_provider.features(instruments=instruments, fields=self._ret_exps,
                                                      start_time=self._start_time, end_time=self._end_time, freq='day')
            
            factors_values = factors_values.reset_index()
            ret_values = ret_values.reset_index().drop(columns=['date'])

            print('因子计算完成! 耗时 {}'.format(dt.datetime.now() - start_time))
            self.raw_data = factors_values

            print('开始拼接收益...')
            start_time = dt.datetime.now()
            factors_ret = pd.merge(factors_values, ret_values, how='inner', on=[
                                   'asset', 'day'])
      
            factors_ret = factors_ret.set_index(['date', 'asset']).sort_index()
            print('收益拼接完成! 耗时 {}'.format(dt.datetime.now() - start_time))

        elif self._freq == 'day':
            factors_ret = self._data_provider.features(instruments=instruments, fields=([self._factor_exp, '$group'] + [
                ret_exp for ret_exp in self._ret_exps]), start_time=self._start_time, end_time=self._end_time, freq=self._freq)

        factors_ret = factors_ret.rename(columns={self._factor_exp: 'factor', '$group':'group'})
        factors_ret = factors_ret.rename(columns={exp: 'return({})'.format(
            name) for exp, name in zip(self._ret_exps, self._ret_names)})

        return factors_ret


class CSVFactor(BaseFactor):

    def __init__(self, factor_dir):
        super().__init__()
        self._factor_dir = factor_dir

    def _load_factor_data(self):
        factor_data = pd.read_csv(self._factor_dir, parse_dates=True)
        if factor_data is not None:
            factor_data = factor_data.set_index(['date', 'asset']).sort_index()
        return factor_data
