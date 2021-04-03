# %%
import traceback
try:
    import os
    import sys
    cur_path = os.path.abspath(os.path.join(
        os.path.dirname('__file__'), os.path.pardir))
    sys.path.append(cur_path)
    from alpha_pipe.scripts.dump_bin import DumpDataAll
except Exception as e:
    traceback.print_exc()
from alpha_pipe.scripts.dump_bin import DumpDataAll

# %%
def csv_to_bin(csv_path, bin_path, freq, instruments_name, fields, date_field_name):
    if not os.path.exists(bin_path):
        os.makedirs(bin_path)
    dumper = DumpDataAll(csv_path=csv_path, qlib_dir=bin_path, freq=freq, include_fields=fields, instruments_name=instruments_name, date_field_name=date_field_name, symbol_field_name='code')
    dumper.dump()
# %%
# dump daily data
CSV_PATH = './data/day_csv_data/'
BIN_PATH = './data/bin_data/'
day_fields = ['open','close','low','high','volume','money','factor','group']
min_fields = ['open','close','low','high','volume','money']
day_min_fields = ['{}_{}'.format(field, i+1) for i in range(240) for field in min_fields]
fields = day_fields + day_min_fields
csv_to_bin(CSV_PATH, BIN_PATH, freq='day', instruments_name='zz800', fields=fields, date_field_name='date')
# %%
# dump minute data
CSV_PATH = './data/min_csv_data/'
BIN_PATH = './data/bin_data/'
min_fields = ['open','close','low','high','volume','money','factor','group']
csv_to_bin(CSV_PATH, BIN_PATH,freq='min', instruments_name='zz800', fields=min_fields, date_field_name='time')
# %%
