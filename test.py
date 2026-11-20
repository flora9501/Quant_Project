import tushare as ts
from config import TUSHARE_TOKEN
pro = ts.pro_api(TUSHARE_TOKEN)

df = pro.fina_indicator(ts_code="000001.SZ", start_date="20230101", end_date="20231231")
print(df.shape)
print(list(df.columns))
print(df[["ann_date", "end_date"]].head())
