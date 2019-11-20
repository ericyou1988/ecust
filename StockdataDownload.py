import numpy as np
import pandas as pd
import tushare as ts
import pymysql
import datetime
from sqlalchemy import create_engine

ts.set_token('c59597bb23c5a81d7537178e40e0307bfcb9f9b1ede382ff50623ac6')
pro = ts.pro_api()
ts_code = '000001.SZ'
start_dt = '20100101'
time_temp = datetime.datetime.now() - datetime.timedelta(days=1)
end_dt = time_temp.strftime('%Y%m%d')
df = pro.daily(ts_code=ts_code, start_date=start_dt, end_date=end_dt)

engine = create_engine('mysql+pymysql://root:Chilugan6xiang@localhost:3306/test')
clo_mapping = {'trade_date':'state_dt','ts_code':'stock_code','change':'amt_change','pct_chg':'pct_change'}
df.rename(columns=clo_mapping,inplace=True)
order = ['state_dt', 'stock_code', 'open', 'close', 'high', 'low', 'vol', 'amount', 'pre_close', 'amt_change', 'pct_change']
df = df[order]
df.to_sql('stock'+ts_code[:6], engine, index = False)