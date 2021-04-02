import abc
from alpha_pipe.qlib.data import D
import alpha_pipe.qlib as qlib

class BaseFactor(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def _load_factor_data(self):
        pass    

class ExpressionFactor(BaseFactor):

    def __init__(self, market, start_time, end_time, factor_exp, ret_exps, ret_names, provider_uri, region):
        super().__init__()

        self._market = market
        self._start_time = start_time
        self._end_time = end_time
        
        self._factor_exp = factor_exp
        self._ret_exps = ret_exps
        self._ret_names = ret_names
        self._factor_data = None
        print(provider_uri, region)
        qlib.init(provider_uri=provider_uri, region=region)
        
    
    def _load_factor_data(self):

        
        instruments = D.instruments(market=self._market)
        factors_ret = D.features(instruments=instruments,fields=([self._factor_exp,'$group'] + [ret_exp for ret_exp in self._ret_exps]),start_time=self._start_time, end_time=self._end_time)
        factors_ret.columns = ['factor','group'] + ['return({})'.format(ret_type) for ret_type in self._ret_names]
        factors_ret = factors_ret.dropna()
        return factors_ret


    def factor_data(self):

        if self._factor_data is None:
            self._factor_data = self._load_factor_data()

        return self._factor_data            
