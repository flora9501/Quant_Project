# 21_fix_industry.py
"""
从 stock_basic 的 L / D / P 三个状态补全行业映射。
不剔除任何股票——剔除会重新引入生存者偏差。
输出 data/ref/industry_map.parquet
"""
import os
from pathlib import Path

import pandas as pd
import tushare as ts
from dotenv import load_dotenv

from config import DATA_DIR

load_dotenv()
pro = ts.pro_api(os.getenv("TUSHARE_TOKEN"))
REF = DATA_DIR / "ref"
FIELDS = "ts_code,name,industry,list_date,delist_date"

frames = []
for status in ("L", "D", "P"):
    df = pro.stock_basic(exchange="", list_status=status, fields=FIELDS)
    df["list_status"] = status
    frames.append(df)
    print(f"list_status={status}: {len(df)} 行, industry 空值 {df.industry.isna().sum()}")
    import time; time.sleep(1)

allb = pd.concat(frames, ignore_index=True).drop_duplicates("ts_code")

# 优先级：L > P > D（在市股票的行业最新）
prio = {"L": 0, "P": 1, "D": 2}
allb["_p"] = allb.list_status.map(prio)
allb = (allb.sort_values(["ts_code", "_p"])
            .drop_duplicates("ts_code")
            .drop(columns="_p"))

ind = allb.set_index("ts_code")["industry"]

cols = pd.read_parquet(DATA_DIR / "panel" / "tradable.parquet").columns
mapped = ind.reindex(cols)
print("\n股票池", len(cols), "只")
print("有行业 :", mapped.notna().sum())
print("仍为空 :", mapped.isna().sum())
if mapped.isna().any():
    print("剩余示例:", mapped[mapped.isna()].index.tolist()[:15])

out = pd.DataFrame({"industry": mapped.fillna("UNKNOWN")})
out.index.name = "ts_code"
out.to_parquet(REF / "industry_map.parquet")
print(f"\n✅ {len(out)} 行 -> {REF/'industry_map.parquet'}")
print(out.industry.value_counts().head(12).to_string())
