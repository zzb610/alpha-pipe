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

```shell
source venv/bin/activate
```

#### 安装所需库

```shell
pip install -r requirements.txt -i  https://pypi.tuna.tsinghua.edu.cn/simple
```

#### 编译pyx

```shell
python setup.py build_ext --inplace    
```

### 配置数据库

将data.zip解压即可
