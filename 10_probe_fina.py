# 10_probe_fina.py
import os, tushare as ts
from dotenv import load_dotenv
load_dotenv()
pro = ts.pro_api(os.getenv("TUSHARE_TOKEN"))

FIELDS = {
    "income":       "ts_code,ann_date,f_ann_date,end_date,report_type,update_flag,n_income,n_income_attr_p",
    "cashflow":     "ts_code,ann_date,f_ann_date,end_date,report_type,update_flag,n_cashflow_act",
    "balancesheet": "ts_code,ann_date,f_ann_date,end_date,report_type,update_flag,comp_type,total_assets",
}

for api, f in FIELDS.items():
    df = getattr(pro, api)(ts_code="000001.SZ", start_date="20240101",
                           end_date="20251231", report_type="1", fields=f)
    print(f"\n=== {api}  shape={df.shape}")
    print(df.head(6).to_string())
    if not df.empty:
        print("f_ann_date 空值率:", df.f_ann_date.isna().mean())
        print("f_ann_date < ann_date 的条数:", (df.f_ann_date < df.ann_date).sum())
