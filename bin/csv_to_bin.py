# %%
import traceback
import pymongo
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
CSV_PATH = './data/csv/'
BIN_PATH = './data/bin/'

# %%
day_fields = ['open','close','low','high','volume','money','factor','group']
min_fields = ['open','close','low','high','volume','money']
day_min_fields = ['{}_{}'.format(field, i+1) for i in range(240) for field in min_fields]
fields = day_fields + day_min_fields

dumper = DumpDataAll(csv_path=CSV_PATH, qlib_dir=BIN_PATH, freq='day',date_field_name='date', symbol_field_name='code',include_fields=fields)
dumper.dump()
# %%
