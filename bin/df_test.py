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
    
from alpha_pipe.factest.factor_test import DFTest
from alpha_pipe.qlib import config, data
from alpha_pipe.qlib.config import REG_CN
# %%
factor_data = ('算好的因子数据')
config = {
    'factor_data_df': factor_data,
    'periods': (1,),
    'quantile':5,
    'ret_types':['close to close'],
    'equal_weighted': True,
    'long_short': False,
    'group_neutral': False,
    'by_group': True
}
tester = DFTest(**config)
# %%

# 收益分析
tester.return_analysis()
# ic分析
tester.information_analysis()
# 换手分析
tester.turnover_analysis() 
# %%

# 收益分析表
tester.plot_returns_table()

# %%

# 因子加权累积收益
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
factor_data
# %%
