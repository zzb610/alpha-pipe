# %%
import traceback
try:
    import os
    import sys
    cur_path = os.path.abspath(os.path.join(
        os.path.dirname('__file__'), os.path.pardir))
    sys.path.append(cur_path)
except Exception as e:
    traceback.print_exc()
    
from alpha_pipe.factest.factor_test import FormulaTest
from alpha_pipe.qlib import config
from alpha_pipe.qlib.config import REG_CN
#  def __init__(self, market, start_time, end_time, peroids: Tuple, quantile, factor_exp, ret_exps: Tuple,ret_types: Tuple,provider_uri, region, equal_weighted=True, long_short=False, group_neutral = False, by_group=False):
config = {
    'market':'all',
    'start_time':'2018-01-01',
    'end_time':'2020-01-01',
    'peroids':(1,1),
    'quantile':10,
    # 'factor_exp':'Ref(Mean(Ref($close_10, -1) / $close_230 - 1, 20), 1)',
    'factor_exp':'Ref(Mean(Ref($close_10, -1) / $close_230 - 1, 20), 1)',
    'ret_exps':['Ref($close_10, -1) / $close_230 - 1', 'Ref($close_20, -1) / $close_230 - 1'],
    'ret_types':['ret1', 'ret2'],
    'provider_uri':'./data/bin',
    'region':REG_CN
}
tester = FormulaTest(**config)
# %%
tester.return_analysis()
tester.information_analysis()
tester.turnover_analysis() 
# %%

# 收益分析表
tester.plot_returns_table()

# %%

# 因子加权(目前只支持等权)累积收益
tester.plot_cumulative_returns()

# %%

# 因子分层回测
tester.plot_cumulative_returns_by_quantile()

# %%

# 因子分层平均收益直方图
tester.plot_quantile_returns_bar()
# %%

# 因子分层收益小提琴图
tester.plot_quantile_returns_violin()

# %%

# 收益差(最高层 - 最低层)
tester.plot_mean_quantile_returns_spread_time_series()

# %%

# ic分析表
tester.plot_information_table()

# %%

# 时序IC
tester.plot_ic_ts()

# %%

# IC 直方图
tester.plot_ic_hist()

# %%

# IC Q-Q 图
tester.plot_ic_qq()
# %%

# 月度IC热力图
tester.plot_monthly_ic_heatmap()

# %%

# 因子换手率表
tester.plot_turnover_table()

# %%

# 最高层换手率-最低层换手率
tester.plot_top_bottom_quantile_turnover()

# %%

# 因子排序的自相关
tester.plot_factor_rank_auto_correlation()

# %%

# 因子值
tester.factor_data()
# %%

# %%
