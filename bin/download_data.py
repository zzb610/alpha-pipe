# %%
import pymongo
import json
import sys
import traceback
import pymongo
try:
    import os
    import sys
    cur_path = os.path.abspath(os.path.join(
        os.path.dirname('__file__'), os.path.pardir))
    sys.path.append(cur_path)

    from alpha_pipe.mtdata.api.fetch import DataFetcher
except Exception as e:
    traceback.print_exc()


with open('./settings/mongo.json') as f:
    mongo_info = json.load(f)
    MONGO_URI = mongo_info['MONGO_URI']
    MONGO_DB_PATH = mongo_info['MONGO_DB_PATH']
    MONGO_LOG_PATH = mongo_info['MONGO_LOG_PATH']
    MONGO_DOCKER_NAME = mongo_info['MONGO_DOCKER_NAME']
    DB_NAME = mongo_info['DB_NAME']

DB = pymongo.MongoClient(MONGO_URI)[DB_NAME]
fetcher = DataFetcher(DB,'jqdata')
# %%
from IPython.display import display
import pandas as pd
import datetime as dt
from tqdm import tqdm
import numpy as np
start_time = '2018-01-01'
end_time = '2021-01-01'
DATA_DIR = './data/csv'
stock_pool = fetcher.get_index_stocks('000906.XSHG', date=start_time)
day_fields = ['open','close','low','high','volume','money','factor']
min_fields = ['open','close','low','high','volume','money']

def flatten_min_data(group):
    
    columns =  ['{}_{}'.format(field, i+1) for i in range(240) for field in min_fields] 
    flatten_values = group[min_fields].values.flatten()
    flatten_values =  flatten_values.reshape(1, len(columns))
    flatten_data = pd.DataFrame(flatten_values,columns=columns)
    return flatten_data
    
for stock in tqdm(stock_pool[:]):
    day_price = fetcher.get_price(stock, start_time, end_time, '1d', day_fields, fq='pre')
    min_price = fetcher.get_price(stock, start_time, end_time, '1m', min_fields, fq='pre')
 
    day_price.rename(columns={'time':'date'}, inplace=True)
    min_price['date'] = pd.to_datetime(min_price['time']).apply(dt.datetime.strftime,format='%Y-%m-%d')
    min_price = min_price.groupby('date').apply(flatten_min_data)
    min_price = min_price.set_index(min_price.index.get_level_values(0)).reset_index()
    min_price['code'] = stock   
    # # 申万一级行业分类
    industry_group = fetcher.get_industry(stock, date=start_time)[stock]['sw_l1']['industry_code']
    data = pd.concat([day_price, min_price], axis=1).dropna()
    data['group'] = industry_group
    data.to_csv('{}/{}.csv'.format(DATA_DIR, stock))
# %%
