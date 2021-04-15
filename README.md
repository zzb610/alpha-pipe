# alpha-pipe

## 源码级用户使用说明

### 注意

代码必须在 bin 目录下运行

### 下载

直接从 github 下载源码包即可(**zzb 分支!!!!!!!**)

### 安装环境

#### 创建 Python 虚拟环境(推荐)

创建环境

```shell
python -m venv venv
```

激活环境

linux

```shell
source venv/bin/activate
```

windows

```
cd venv/Scripts
./activate
```

#### 安装所需库

```shell
pip install -r requirements.txt
```

#### 编译 pyx

```shell
python setup.py build_ext --inplace
```

### 配置数据库

将 data.zip 解压即可

#### 因子测试

建议习惯用 jupyter 的用户,将 test.py 中的代码**按块**复制到.ipynb 文件中运行

#### 基础因子

##### 日级别滚动因子（算子在日 K 线滑动）

- '$open','$close','$low','$high','$volume','$money','$factor' 日频

- '$open_x', '$close_x', '$low_x', '$high_x', '$volume_x', '$money_x', 其中 x ~ (1, 240) 1 分钟频

##### 分钟级别滚动因子（算子在 1 分钟 K 线滑动）

- '$open','$close','$low','$high','$volume','$money','$factor' 分钟频

#### 基础算子

<!-- ![](docs/op1.jpg)
![](docs/op2.jpg)
![](docs/op3.jpg)
![](docs/op4.jpg) -->

- "> >= < <= == !="
- "+ - \* /"
- "& |": and or
- Not: Not Operator
- Abs: Feature Absolute Value
- Sign: Feature Sign
- Log: Feature Log
- Power: Feature Power
- If: If($close>$open, $close, $open)
- Ref: Feature Reference, 类似 pandas shift Ref($close, -1) 下一天的收盘价
- Mean: Rolling Mean (MA) Mean($close, 5)
- Sum: Rolling Sum Sum($close, 5)
- Std: Rolling Std. Std($close, 5)
- Var: Rolling Variance. Var($close, 5)
- Skew: Rolling Skewness. Skew($close, 10)
- Kurt: Rolling Kurtosis($close 10)
- Max: Rolling Max. Max($close, 10)
- IdxMax: Rolling Max Index. IdxMax($close, 10)
- IdxMin: Rolling Min Index. IdxMin($close, 10)
- Min: Rolling Min. Min($close, 10)
- Quantile: Rolling Quantile. Quantile($close, 10, 0.5) 过去十天 大于 50% 收盘价的分位数
- Med: Rolling Median. Med($close, 10)
- Mad: Rolling Mean Absolute Deviation. Mad($close, 10)\
- Rank: Rolling Rank (Percentile). Rank($close, 10)
- Count: rolling count of number of non-NaN elements
- Delta: end minus start in rolling window. Delta($close, 10)
- Slope: linear regression slope of given window. Slope($close, 10)
- Resi: regression residuals of given window. Resi($close, 10)
- WMA: weighted moving average. WMA($close,10)
- EMA: Exponential Mean. EMA($close, 10)
- Corr: rolling correlation of two input features. Corr($close, $open, 10)
- Cov: Rolling Covariance. Cov($close, $open, 10)

### 使用例子

配置好环境以后 因子测试分为两步

- 第一步 计算因子值 alpha_pipe.factor

```python
from alpha_pipe.factor.factor import ExpressionFactor
```

ExpressionFactor 通过公式计算因子值， CSVFactor 通过 csv 文件加载因子值

因子值计算分为日频滑动与 1 分钟频滑动两种

- 日频滑动对于每个标的每天计算一个因子值
- 1 分钟频滑动对于每个标的每分钟计算一个因子值，通常还需要用户手动进行高频因子低频化

```python
data_dir = './data/bin_data'

# 分钟滚动因子
factor_config = dict(
    market = 'small_sample', # 股票池 中证800
    start_time = '2017-01-01', # 开始时间
    end_time = '2017-01-05', # 结束时间
    freq='min', # 滚动频率 1分钟
    factor_exp = 'Corr($open, $volume, 240)', # 过去240分钟的open与volume的相关系数
    ret_exps = ['Ref($close_10, -1)/$close_230 - 1', 'Ref($open, -5)/$close - 1'], # 收益计算公式，注意目前仍然是日频滚动的
    ret_names = ret_names, # 收益的名称
    provider_uri =  data_dir, # 数据库路径
)

min_factor = ExpressionFactor(**factor_config) ## 创建ExpressionFactor对象
min_factor.factor_data() # 获取因子值的接口 DataFrame

# 日滚动因子
 
factor_config = dict(
    market = 'zz800',
    start_time = '2017-01-01',
    end_time = '2018-06-30',
    freq='day', # 滚动频率 1天
    factor_exp = '$close + $volume / 2',
    ret_exps = ['Ref($close_10, -1)/$close_230 - 1', 'Ref($open, -5)/$close - 1'], # 收益计算公式
    ret_names = ret_names,
    provider_uri = data_dir,
)


day_factor = ExpressionFactor(**factor_config)
day_factor.factor_data()
```

- 第二步 因子测试 alpha_pipe.analyzer

```python
from alpha_pipe.analyzer.factor_analyzer import FactorAnalyzer
```

```python
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
# 注意高频因子需要低频化

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

```
