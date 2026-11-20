"""
按年分段重拉 namechange，绕过单次 10000 行上限。
"""
import time

import pandas as pd
import tushare as ts

from config import TUSHARE_TOKEN, DATA_DIR

pro = ts.pro_api(TUSHARE_TOKEN)
REF = DATA_DIR / "ref"
REF.mkdir(parents=True, exist_ok=True)

FIELDS = "ts_code,name,start_date,end_date,ann_date,change_reason"


def retry(fn, tries=3):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            if i == tries - 1:
                raise
            print(f"  重试 {i+1}/{tries}: {e}")
            time.sleep(2 * (i + 1))


frames = []
for y in range(1990, 2027):
    s, e = f"{y}0101", f"{y}1231"
    try:
        df = retry(lambda: pro.namechange(start_date=s, end_date=e, fields=FIELDS))
    except Exception as ex:
        print(f"{y} 失败：{ex}")
        continue

    if df is not None and len(df):
        n = len(df)
        flag = "  ⚠️ 可能仍被截断" if n >= 9000 else ""
        print(f"{y}: {n:>5,} 条{flag}")
        frames.append(df)
    else:
        print(f"{y}:     0 条")
    time.sleep(0.35)

if not frames:
    raise SystemExit("一条都没拉到")

all_df = (pd.concat(frames, ignore_index=True)
          .drop_duplicates(subset=["ts_code", "start_date", "name"]))

all_df.to_parquet(REF / "namechange.parquet", index=False)

print("\n===== 汇总 =====")
print(f"总条数    : {len(all_df):,}")
print(f"涉及股票  : {all_df['ts_code'].nunique():,}")
print(f"日期范围  : {all_df['start_date'].min()} ~ {all_df['start_date'].max()}")

st = all_df[all_df["name"].str.contains("ST", na=False)]
print(f"含 ST 记录: {len(st):,}")
print(f"涉及股票  : {st['ts_code'].nunique():,}")

print("\n前 5 条样例：")
print(all_df.head().to_string(index=False))
