# alpha-pipe

## 源码级用户使用说明

### 下载

直接从github下载源码包即可(**zzb分支!!!!!!!**)

### 安装环境

#### 创建Python虚拟环境(推荐)

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

#### 编译pyx

```shell
python setup.py build_ext --inplace    
```

### 配置数据库

将data.zip解压即可


#### 因子测试

建议习惯用jupyter的用户,将test.py中的代码**按块**复制到.ipynb文件中运行

#### 基础因子

 - '$open','$close','$low','$high','$volume','$money','$factor' 日级别因子

 - '$open_x', '$close_x', '$low_x', '$high_x', '$volume_x', '$money_x', 其中 x ~ (1, 240) 分钟级别因子

#### 基础算子

![](docs/op1.jpg)
![](docs/op2.jpg)
![](docs/op3.jpg)
![](docs/op4.jpg)