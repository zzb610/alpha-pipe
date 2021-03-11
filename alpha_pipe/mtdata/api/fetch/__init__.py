from IPython.core.display import display
from alpha_pipe.mtdata.util import get_security_type, Freq
import pandas as pd

import datetime
from itertools import repeat
import numpy as np

def str_date(s, begin, end):
    return s[begin:end]


class DataFetcher(object):

    def __init__(self, database, ds_name) -> None:
        super().__init__()
        self.__database = database
        self.__ds_name = ds_name

    def recent_trade_day(self, date):
        # last_trade_day = self.get_trade_days('2004-01-01', date)[-1]
        last_trade_day = self.get_trade_days(date, '2999-01-01')[0]
        last_trade_day = last_trade_day.strftime("%Y-%m-%d")
        return last_trade_day

    def get_price(self, security, start_date, end_date, frequency='1d', fields=['open', 'close', 'high', 'low', 'volume', 'money'], fq='pre'):

        if 'factor' in fields and fq is None:
            raise Exception(' 请设置复权方法! 前复权 pre 后复权 post')

        freq: Freq = None
        if frequency in ['d', '1d', '1D', 'daily']:
            freq = Freq.DAY
        elif frequency in ['m', '1m', 'min', 'minute']:
            freq = Freq.MINUTE
        else:
            raise Exception('不支持的频率 {} '.format(frequency))

        if isinstance(security, str):
            security = [security]

        data = self.fetch_price(security, start_date, end_date, freq, fields)

        if fq is not None:
            fq_factor = None
            if fq == 'pre':
                fq_factor = self.fetch_fq_factor(
                    security, start_date, end_date, 'factor')
            elif fq == 'post':
                fq_factor = self.fetch_fq_factor(
                    security, start_date, end_date, 'post_factor')

            if freq == Freq.MINUTE:
                str_d = data.time.apply(str_date, begin=0, end=10)

                fq_factor.rename(columns={'time': 'date'}, inplace=True)
                data = data.assign(date=str_d)

                data = data.set_index(['date', 'code'])
                fq_factor = fq_factor.set_index(['date', 'code'])

            # 对齐复权因子
            data['factor'] = fq_factor[fq_factor.columns[-1]]
            cols = data.columns
            for col in cols:
                if col in ['open', 'close', 'low', 'high']:
                    data[col] = data[col] * data['factor']
                    data[col] = data[col].round(2)

                elif col in ['volume']:
                    data[col] = data[col] / data['factor']
                    data[col] = data[col].round(1)
            
            data.reset_index(inplace=True)
        
        fields = ['time', 'code'] + fields
        if data is not None and data.shape[0] > 0:
            return data[fields]
        return data

    def get_all_securities(self, types=['stock'], date=None):
        if isinstance(types, str):
            types = [types]
        return self.fetch_all_securities(types, date)

    def get_all_trade_days(self):
        data = self.fetch_all_trade_days()
        data = list(map(datetime.datetime.strptime, data, repeat("%Y-%m-%d")))
        data = np.array(list(map(datetime.datetime.date, data)))
        return data

    def get_trade_days(self, start_date, end_date):
        data = self.fetch_trade_days(start_date, end_date)
        data = list(map(datetime.datetime.strptime, data, repeat("%Y-%m-%d")))
        data = np.array(list(map(datetime.datetime.date, data)))
        return data

    def get_billboard_list(self, stock_list, start_date, end_date):
        return self.fetch_billboard_list(stock_list, start_date, end_date)

    def get_index_stocks(self, index_symbol, date):
        return self.fetch_index_stocks(index_symbol, date)

    def get_industry_stocks(self, industry_code, date):

        return self.fetch_industry_stocks(industry_code, date)

    def get_concept_stocks(self, concept_code, date):
        return self.fetch_concept_stocks(concept_code, date)

    def get_industries(self, name, date):

        return self.fetch_industries(name, date)

    def get_concepts(self):
        return self.fetch_concepts()

    def get_security_info(self, code, date):
        return self.fetch_security_info(code, date)

    def get_money_flow(self, security_list, start_date, end_date, fields):
        if fields is None:
            fields = ['date', 'sec_code', 'change_pct', 'net_amount_main', 'net_pct_main', 'net_amount_xl',
                      'net_pct_xl', 'net_amount_l', 'net_pct_l', 'net_amount_m', 'net_pct_m', 'net_amount_s', 'net_pct_s']
        return self.fetch_money_flow(security_list, start_date, end_date, fields)

    def get_mtss(self, security_list, start_date, end_date, fields):
        if fields is None:
            fields = ['date', 'sec_code', 'fin_value', 'fin_buy_value', 'fin_refund_value',
                      'sec_value', 'sec_sell_value', 'sec_refund_value', 'fin_sec_value']
        return self.fetch_mtss(security_list, start_date, end_date, fields)

    def get_margincash_stocks(self, date):
        return self.fetch_margincash_stocks(date)

    def get_marginsec_stocks(self, date):
        return self.fetch_marginsec_stocks(date)

    def get_future_contracts(self, underlying_symbol, date):
        return self.fetch_future_contracts(underlying_symbol, date)

    def get_dominant_future(self, underlying_symbol, date):
        return self.fetch_dominant_future(underlying_symbol, date)

    def get_index_weights(self, index_id, date):
        return self.fetch_index_weights(index_id, date)

    def get_industry(self, security, date):
        return self.fetch_industry(security, date)

    def get_concept(self, security, date):
        return self.fetch_concept(security, date)

    def get_locked_shares(self, stock_list, start_date, end_date):
        return self.fetch_locked_shares(stock_list, start_date, end_date)

    def fetch_locked_shares(self, stock_list, start_date, end_date):
        coll_name = 'locked_shares_{}'.format(self.__ds_name)
        coll_locked_shares = self.__database[coll_name]
        cursor = coll_locked_shares.find({'code': {'$in': stock_list}, 'day': {
                                         '$gte': start_date, '$lte': end_date}})
        data = pd.DataFrame([item for item in cursor])
        if data is not None and data.shape[0] > 0:
            fields = ['day', 'code', 'num', 'rate1', 'rate2']
            data = data[fields]
        return data

    def fetch_industry(self, security, date):
        coll_stocks_name = 'industry_stocks_{}'.format(self.__ds_name)
        coll_stocks = self.__database[coll_stocks_name]
        last_trade_day = self.recent_trade_day(date)
        if isinstance(security, str):
            security = [security]
        res = {}
        for sec in security:
            cursor = coll_stocks.find(
                {'stocks': sec, 'date': last_trade_day})
            industry_infos = {}
            for item in cursor:
                industry_code = item['code']
                industry_type, industry_info = self.fetch_industry_info(
                    industry_code, date)
                industry_infos[industry_type] = industry_info
            res[sec] = industry_infos
        return res

    def fetch_industry_info(self, industry_code, date):
        coll_name = 'industry_{}'.format(self.__ds_name)
        coll_industry = self.__database[coll_name]
        cursor = coll_industry.find(
            {'code': industry_code, 'start_date': {'$lte': date}})
        industry_info = {}
        industry_type = None
        for item in cursor:
            industry_info['industry_code'] = item['code']
            industry_info['industry_name'] = item['name']
            industry_type = item['type']
        return industry_type, industry_info

    def fetch_concept(self, security, date):
        coll_stocks_name = 'concept_stocks_{}'.format(self.__ds_name)
        coll_stocks = self.__database[coll_stocks_name]

        last_trade_day = self.recent_trade_day(date)
        if isinstance(security, str):
            security = [security]
        res = {}
        for sec in security:
            sec_res = {}
            cursor = coll_stocks.find({'stocks': sec, 'date': last_trade_day})
            jq_concept = []
            for item in cursor:
                concept = {}
                name = self.fetch_concepts_name(item['code'])
                concept['concpet_name'] = None if (
                    name is None or len(name) < 1) else name[0]
                concept['concept_code'] = item['code']
                jq_concept.append(concept)
            sec_res['jq_concept'] = jq_concept
            res[sec] = sec_res
        return res

    def fetch_index_weights(self, index_id, date):
        coll_name = 'index_weights_{}'.format(self.__ds_name)
        coll = self.__database[coll_name]
        last_trade_day = self.recent_trade_day(date)
        cursor = coll.find(
            {'index_code': index_id, 'date': last_trade_day})
        data = pd.DataFrame([item for item in cursor])
        fields = ['code', 'date', 'weight', 'display_name']
        if data is not None and data.shape[0] > 0:
            data = data[fields]
        return data

    def fetch_dominant_future(self, underlying_symbol, date):
        coll_name = 'dominant_future_{}'.format(self.__ds_name)
        coll = self.__database[coll_name]
        last_trade_day = self.recent_trade_day(date)
        cursor = coll.find(
            {'underlying_symbol': underlying_symbol, 'date': last_trade_day})
        data = []
        for item in cursor:
            data.append(item['contracts'])
        if len(data) > 0:
            data = data[0]
        return data

    def fetch_margincash_stocks(self, date):

        coll_name = 'margincash_stocks_{}'.format(self.__ds_name)
        coll = self.__database[coll_name]
        last_trade_day = self.recent_trade_day(date)
        cursor = coll.find({'date': last_trade_day})
        data = []
        for item in cursor:
            data.extend(item['stocks'])
        return data

    def fetch_marginsec_stocks(self, date):

        coll_name = 'marginsec_stocks_{}'.format(self.__ds_name)
        coll = self.__database[coll_name]
        last_trade_day = self.recent_trade_day(date)
        cursor = coll.find({'date': last_trade_day})
        data = []
        for item in cursor:
            data.extend(item['stocks'])
        return data

    def fetch_future_contracts(self, underlying_symbol, date):
        coll_name = 'future_contracts_{}'.format(self.__ds_name)
        coll = self.__database[coll_name]
        last_trade_day = self.recent_trade_day(date)
        cursor = coll.find(
            {'date': last_trade_day, 'underlying_symbol': underlying_symbol})
        data = []
        for item in cursor:
            data.extend(item['contracts'])
        return data

    def fetch_mtss(self, security_list, start_date, end_date, fields):
        if isinstance(security_list, str):
            security_list = [security_list]
        coll_name = 'mtss_{}'.format(self.__ds_name)
        coll = self.__database[coll_name]

        cursor = coll.find({'sec_code': {'$in': security_list}, 'date': {
            '$gte': start_date, '$lte': end_date}})

        data = pd.DataFrame([item for item in cursor])
        if data is not None and data.shape[0] > 0:
            data = data[fields].sort_values(
                by=['sec_code', 'date'], ascending=[True, True])
        return data

    def fetch_money_flow(self, security_list, start_date, end_date, fields):
        if isinstance(security_list, str):
            security_list = [security_list]
        coll_name = 'money_flow_{}'.format(self.__ds_name)
        coll = self.__database[coll_name]

        cursor = coll.find({'sec_code': {'$in': security_list}, 'date': {
            '$gte': start_date, '$lte': end_date}})

        data = pd.DataFrame([item for item in cursor])
        if data is not None and data.shape[0] > 0:
            data = data[fields].sort_values(
                by=['sec_code', 'date'], ascending=[True, True])
        return data

    def fetch_security_info(self, code, date):
        if isinstance(code, str):
            code = [code]

        coll_name = 'security_type_{}'.format(self.__ds_name)

        coll = self.__database[coll_name]
        if date is None:
            cursor = coll.find({'code': {'$in': code}})
        else:
            cursor = coll.find(
                {'code': {'$in': code}, 'start_date': {'$lte': date}, 'end_date': {'$gt': date}})
        data = pd.DataFrame([item for item in cursor])
        fields = ['code', 'display_name', 'name',
                  'start_date', 'end_date', 'type']
        if data is not None and data.shape[0] > 0:
            data = data[fields]
            data = data.sort_values(by='code').reset_index().drop('index', 1)

        return data

    def fetch_concepts_name(self, concept_code):
        coll_concepts_name = 'concept_{}'.format(self.__ds_name)
        coll_concepts = self.__database[coll_concepts_name]

        cursor = coll_concepts.find({'code': concept_code})
        data = [item['name'] for item in cursor]
        return data

    def fetch_concept(self, security, date):
        coll_stocks_name = 'concept_stocks_{}'.format(self.__ds_name)
        coll_stocks = self.__database[coll_stocks_name]

        last_trade_day = self.recent_trade_day(date)
        if isinstance(security, str):
            security = [security]
        res = {}
        for sec in security:
            sec_res = {}
            cursor = coll_stocks.find({'stocks': sec, 'date': last_trade_day})
            jq_concept = []
            for item in cursor:
                concept = {}
                name = self.fetch_concepts_name(item['code'])
                concept['concpet_name'] = None if (
                    name is None or len(name) < 1) else name[0]
                concept['concept_code'] = item['code']
                jq_concept.append(concept)
            sec_res['jq_concept'] = jq_concept
            res[sec] = sec_res
        return res

    def fetch_concepts(self):
        coll_name = 'concept_{}'.format(self.__ds_name)
        coll = self.__database[coll_name]
        cursor = coll.find()
        data = pd.DataFrame([item for item in cursor])
        fields = ['code', 'name', 'start_date']
        return data[fields]

    def fetch_industries(self, name, date):

        coll_name = 'industry_{}'.format(self.__ds_name)
        coll = self.__database[coll_name]
        cursor = coll.find({'type': name, 'start_date': {'$lte': date}})
        data = pd.DataFrame([item for item in cursor])
        fields = ['code', 'name', 'start_date']
        if data is not None and data.shape[0] > 0:
            return data[fields]
        return data

    def fetch_concept_stocks(self, concept_code, date):

        coll_name = 'concept_stocks_{}'.format(self.__ds_name)
        coll = self.__database[coll_name]
        last_trade_day = self.recent_trade_day(date)
        cursor = coll.find({'code': concept_code, 'date': last_trade_day})
        data = []
        for item in cursor:
            data.extend(item['stocks'])
        return data

    def fetch_industry_stocks(self, industry_code, date):
        coll_name = 'industry_stocks_{}'.format(self.__ds_name)
        coll = self.__database[coll_name]
        last_trade_day = self.recent_trade_day(date)
        cursor = coll.find({'code': industry_code, 'date': last_trade_day})
        data = []
        for item in cursor:
            data.extend(item['stocks'])
        return data

    def fetch_index_stocks(self, index_symbol, date):

        coll_name = 'index_stocks_{}'.format(self.__ds_name)
        coll = self.__database[coll_name]

        last_trade_day = self.recent_trade_day(date)
        cursor = coll.find({'code': index_symbol, 'date': last_trade_day})
        data = []
        for item in cursor:
            data.extend(item['stocks'])
        return data

    def fetch_fq_factor(self, securities, start_date, end_date, fq):
        coll_name = None

        sec_type = None

        if isinstance(securities, str):
            securities = [securities]

        if securities is not None and len(securities) > 0:
            sec_type = get_security_type(securities[0])

        if sec_type in ['stock', 'fund', 'index']:
            sec_type = 'stock'
        elif sec_type in ['future', 'option']:
            sec_type = 'future'

        coll_name = 'fq_factor_{}'.format(self.__ds_name)

        coll = self.__database[coll_name]
        cursor = coll.find({'code': {'$in': securities}, 'time': {
            '$gte': start_date, '$lte': end_date}})
        data = pd.DataFrame([item for item in cursor])
        if isinstance(fq, str):
            fq = [fq]
        fields = ['time', 'code'] + fq
        if data is not None and data.shape[0] > 0:
            return data[fields]
        return data

    def fetch_trade_days(self, start_date, end_date):
        coll_name = 'trade_days_{}'.format(self.__ds_name)
        coll = self.__database[coll_name]
        cursor = coll.find(
            {'trade_date': {'$gte': start_date, '$lte': end_date}})
        data = [item['trade_date'] for item in cursor]
        return data

    def fetch_all_trade_days(self):

        coll_name = 'trade_days_{}'.format(self.__ds_name)
        coll = self.__database[coll_name]
        cursor = coll.find()
        data = [item['trade_date'] for item in cursor]
        return data

    def fetch_all_securities(self, types, date):

        coll_name = 'security_type_{}'.format(self.__ds_name)

        coll = self.__database[coll_name]
        if date is None:
            cursor = coll.find({'type': {'$in': types}})
        else:
            cursor = coll.find(
                {'type': {'$in': types}, 'start_date': {'$lte': date}, 'end_date': {'$gt': date}})
        data = pd.DataFrame([item for item in cursor])
        fields = ['code', 'display_name', 'name',
                  'start_date', 'end_date', 'type']
        if data is not None and data.shape[0] > 0:
            data = data[fields]
            data = data.sort_values(by='code').reset_index().drop('index', 1)

        return data

    def fetch_price(self, securities, start_date, end_date, freq: Freq, fields):

        coll_name = None

        sec_type = None
        if securities is not None and len(securities) > 0:
            sec_type = get_security_type(securities[0])

        if sec_type in ['stock', 'fund', 'index']:
            sec_type = 'stock'
        elif sec_type in ['future', 'option']:
            sec_type = 'future'

        if freq == Freq.DAY:
            coll_name = '{}_price_day_{}'.format(sec_type, self.__ds_name)
        elif freq == Freq.MINUTE:
            coll_name = '{}_price_minute_{}'.format(sec_type, self.__ds_name)

        coll = self.__database[coll_name]
        cursor = coll.find({'code': {'$in': securities}, 'time': {
            '$gte': start_date, '$lte': end_date}})
        data = pd.DataFrame([item for item in cursor])
        fields = ['time', 'code'] + fields
        fields = [f for f in fields if f != 'factor']
        if data is not None and data.shape[0] > 0:
            return data[fields]
        return data

    def fetch_billboard_list(self, stock_list, start_date, end_date):

        coll_name = 'billboard_list_{}'.format(self.__ds_name)
        coll = self.__database[coll_name]
        cursor = None
        if stock_list is None:
            cursor = coll.find(
                {'day': {'$gte': start_date, '$lte': end_date}})
        else:
            cursor = coll.find({'code': {'$in': stock_list}, 'day': {
                '$gte': start_date, '$lte': end_date}})
        data = pd.DataFrame([item for item in cursor])
        fields = ['code', 'day', 'direction', 'rank', 'abnormal_code', 'abnormal_name', 'sales_depart_name',
                  'buy_value', 'buy_rate', 'sell_value', 'sell_rate', 'total_value', 'net_value', 'amount']
        if data is not None and data.shape[0] > 0:
            data = data[fields]
        return data
