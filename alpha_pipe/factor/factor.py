import abc
from alpha_pipe.qlib.data import D
import alpha_pipe.qlib as qlib
import datetime as dt
from .utils import qlib_to_alphalens
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

    def __init__(self, market, start_time, end_time, freq, factor_exp, ret_exps, ret_names, provider_uri, region, to_day_func=None):
        super().__init__()

        self._market = market
        self._start_time = start_time
        self._end_time = end_time
        self._freq = freq

        self._factor_exp = factor_exp
        self._ret_exps = ret_exps
        self._ret_names = ret_names

        self._to_day_func = to_day_func

        self.raw_data = None

        print(provider_uri, region)
        qlib.init(provider_uri=provider_uri, region=region)

  
    def _load_factor_data(self):
        
        instruments = D.instruments(market=self._market)

        if self._freq == 'min':

            factors_values = D.features(instruments=instruments, fields=[
                                        self._factor_exp, '$group'], start_time=self._start_time, end_time=self._end_time, freq=self._freq)
            ret_values = D.features(instruments=instruments, fields=self._ret_exps,
                                    start_time=self._start_time, end_time=self._end_time, freq='day')
            print('因子计算完成')
            factors_values.reset_index(inplace=True)
            ret_values.reset_index(inplace=True)
            factors_values['date'] = factors_values.datetime.apply(
                dt.datetime.strftime, format='%Y-%m-%d')
            ret_values['date'] = ret_values.datetime.apply(
                dt.datetime.strftime, format='%Y-%m-%d')
            ret_values.drop(columns=['datetime'], inplace=True)

            self.raw_data = factors_values
            factors_values = self._to_day_func(factors_values).reset_index()
        
            factors_ret = pd.merge(factors_values, ret_values, how='inner', on=[
                                   'instrument', 'date']).drop(columns=['date'])

            factors_ret = factors_ret.set_index(['instrument', 'datetime']).sort_index()
        elif self._freq == 'day':
            factors_ret = D.features(instruments=instruments, fields=([self._factor_exp, '$group'] + [
                                        ret_exp for ret_exp in self._ret_exps]), start_time=self._start_time, end_time=self._end_time, freq=self._freq)
        
        factors_ret.columns = ['factor', 'group'] + \
            ['return({})'.format(ret_type) for ret_type in self._ret_names]
        factors_ret = qlib_to_alphalens(factors_ret.dropna())
        return factors_ret


class CSVFactor(BaseFactor):

    def __init__(self, factor_dir):
        super().__init__()
        self._factor_dir = factor_dir

    def _load_factor_data(self):
        factor_data = pd.read_csv(self._factor_dir, parse_dates=True)
        if factor_data is not None:
            factor_data = factor_data.set_index(['date','asset']).sort_index()
        return factor_data
