# %%
import traceback
try:
    import os
    import sys
    cur_path = os.path.abspath(os.path.join(
        os.path.dirname('__file__'), os.path.pardir))
    sys.path.append(cur_path)

    from alpha_pipe.mtdata.api.fetch import DataFetcher
except Exception as e:
    traceback.print_exc()
from alpha_pipe.factor.factor import ExpressionFactor
from alpha_pipe.qlib.config import REG_CN
# %%
data_dir = './data/bin_data'
ret_names = ['隔夜收益','五天收益']
periods = (1,5)

# 分钟滚动因子
factor_config = dict(
    market = 'zz800', # 股票池 中证800
    start_time = '2017-01-01', # 开始时间
    end_time = '2018-06-30', # 结束时间
    freq='min', # 滚动频率 1分钟
    factor_exp = 'Corr($open, $volume, 240)', # 过去240分钟的open与volume的相关系数
    ret_exps = ['Ref($close_10, -1)/$close_230 - 1', 'Ref($open, -5)/$close - 1'], # 收益计算公式，注意目前仍然是日频滚动的
    ret_names = ret_names, # 收益的名称
    provider_uri =  data_dir, # 数据库路径
    region = REG_CN # A股
)

min_factor = ExpressionFactor(**factor_config) ## 创建ExpressionFactor对象
# min_factor.factor_data() # 获取因子值的接口 DataFrame

## 高频因子低频化
import pandas as pd
day_factor_data = min_factor.factor_data().groupby('asset').resample('1d', level=0).first().reset_index().set_index(['date','asset']).sort_index()
day_factor_data

# # 日滚动因子
 
# factor_config = dict(
#     market = 'zz800',
#     start_time = '2017-01-01',
#     end_time = '2018-06-30',
#     freq='day', # 滚动频率 1天
#     factor_exp = '$close + $volume / 2',
#     ret_exps = ['Ref($close_10, -1)/$close_230 - 1', 'Ref($open, -5)/$close - 1'], # 收益计算公式
#     ret_names = ret_names,
#     provider_uri = data_dir,
#     region = REG_CN
# )

# day_factor = ExpressionFactor(**factor_config)
# day_factor_data = day_factor.factor_data()

# %%
from alpha_pipe.analyzer.factor_analyzer import FactorAnalyzer

# 创建分析对象
analyzer_config = dict(
    quantiles = 5, # 分层数
    periods = periods, # 收益计算周期、调仓周期
    ret_names = ret_names, # 收益名称
    binning_by_group =  False, # 是否对每个组分别计算分位数
    zero_aware = False # 是否分别为正负因子值计算分位数
)
far = FactorAnalyzer(**analyzer_config) 

# 设置因子值 注意不能是ExpressionFactor等Factor对象, 可以是他们factor_data() 接口返回的Dataframe因子值
far.set_factor_data(day_factor_data) 

## 生成所有图表
test_config = dict(
    
    ## 这两项需要weight字段 没有应设为False
    demeaned=False, # 是否使用超额收益计算 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
    group_adjust=False, # 是否使用使用行业中性化后的收益计算


    by_group=False, # 是否按行业展示
    ic_method='rank', # 'rank': 用秩相关系数计算IC值 'normal': 用普通相关系数计算IC值
    ic_theoretical_dist='norm', # ic的理论分布 'norm': 正态分布 't': t 分布

    turnover_rank=True, # 因子自相关性是否使用秩

    ## 因子预测能力 事件分析 暂时不用 使用以下值即可
    avgretplot=(5, 15), 
    std_bar=False
)
far.create_full_tear_sheet(**test_config)
# %%

# 保存因子
day_factor_data.to_csv('factors/factor1.csv')
# 加载本地因子
from alpha_pipe.factor.factor import CSVFactor
local_factors = CSVFactor('factors/factor1.csv')
local_factors.factor_data()
# %%