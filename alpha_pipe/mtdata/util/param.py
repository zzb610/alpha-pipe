from enum import Enum
import json

# MongoDB
MONGO_SETTINGS_PATH = './settings/mongo.json'
MONGO_DB_PATH = None
MONGO_LOG_PATH = None
MONGO_DOCKER_NAME = None
DB_NAME = None
LOG_DIR = None

# JQData
JQ_LOGIN_PATH = './settings/jqdata.json'

# RUN
RUN_SETTINGS_PATH = './settings/run.json'
LOG_DIR = None

with open(MONGO_SETTINGS_PATH) as f:
    mongo_info = json.load(f)
    MONGO_URI = mongo_info['MONGO_URI']
    MONGO_DB_PATH = mongo_info['MONGO_DB_PATH']
    MONGO_LOG_PATH = mongo_info['MONGO_LOG_PATH']
    MONGO_DOCKER_NAME = mongo_info['MONGO_DOCKER_NAME']
    DB_NAME = mongo_info['DB_NAME']

with open(RUN_SETTINGS_PATH) as f:
    run_info = json.load(f)
    LOG_DIR = run_info['LOG_DIR']

# 时间频率
class Freq(Enum):

    MINUTE = 'min'
    DAY = 'day'
    WEEK = 'week'
    MONTH = 'month'
    QUARTER = 'quarter'
    YEAR = 'year'

# 期货 期权 合约类型
CONTRACT_TYPES = [
    'IC', 'IF', 'IH', 'T', 'TF', 'TS', 'SC', 'NR', 'LU', 'BC',
    'AG', 'AL', 'AU', 'BU', 'CU', 'FU', 'HC', 'NI', 'PB', 'RB', 'RU', 'SN', 'WR', 'ZN', 'SP', 'SS',
    'AP', 'CF', 'CY', 'FG', 'GN', 'JR', 'LR', 'MA', 'ME', 'PM', 'ER', 'RI', 'RM', 'RO', 'OI', 'RS', 'SF', 'SM', 'SR', 'TA', 'TC', 'ZC', 'WH', 'WS', 'WT', 'CJ', 'UR', 'SA', 'PF',
    'A', 'B', 'BB', 'C', 'CS', 'FB', 'I', 'J', 'JD', 'JM', 'L', 'M', 'P', 'PP', 'V', 'Y', 'EG', 'EB', 'PG', 'LH'
]
