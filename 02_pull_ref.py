"""
第 1 天补充：拉股票列表和名称变更历史（ST 判定用）
"""
import pandas as pd
import tushare as ts

from config import TUSHARE_TOKEN, DATA_DIR

pro = ts.pro_api(TUSHARE_TOKEN)

REF = DATA_DIR / "ref"
REF.mkdir(parents=True, exist_ok=True)

FIELDS = "ts_code,symbol,name,area,industry,market,list_date,delist_date"


# ---------- 1. 在市股票 ----------
print("拉取在市股票列表...")
basic = pro.stock_basic(exchange="", list_status="L", fields=FIELDS)
basic.to_parquet(REF / "stock_basic.parquet", index=False)
print(f"  ✅ {len(basic):,} 只")
print(f"  list_date 缺失：{basic['list_date'].isna().sum()}")
print(f"  上市日期范围：{basic['list_date'].min()} ~ {basic['list_date'].max()}")


# ---------- 2. 已退市股票（防生存者偏差） ----------
print("\n拉取退市股票列表...")
try:
    dead = pro.stock_basic(exchange="", list_status="D", fields=FIELDS)
    dead.to_parquet(REF / "stock_delisted.parquet", index=False)
    print(f"  ✅ {len(dead):,} 只")
    print(f"  退市日期范围：{dead['delist_date'].min()} ~ {dead['delist_date'].max()}")
except Exception as e:
    print(f"  ⚠️ 失败：{e}")


# ---------- 3. 名称变更历史（ST 判定） ----------
print("\n拉取名称变更历史...")
try:
    nc = pro.namechange(fields="ts_code,name,start_date,end_date,ann_date,change_reason")
    nc.to_parquet(REF / "namechange.parquet", index=False)
    print(f"  ✅ {len(nc):,} 条")
    print(f"  涉及股票：{nc['ts_code'].nunique():,} 只")
    print(f"  日期范围：{nc['start_date'].min()} ~ {nc['start_date'].max()}")
    st = nc[nc["name"].str.contains("ST", na=False)]
    print(f"  含 ST 的记录：{len(st):,} 条")
except Exception as e:
    print(f"  ⚠️ 失败：{e}")
    print("    备用方案：用 stock_basic 的当前 name 判断是否 ST（有轻微偏差）")


# ---------- 4. 汇总 ----------
print("\n===== 参考表汇总 =====")
for f in sorted(REF.glob("*.parquet")):
    df = pd.read_parquet(f)
    print(f"  {f.name:<26} {len(df):>8,} 行  {df.shape[1]} 列")
    print(f"    列：{list(df.columns)}")
