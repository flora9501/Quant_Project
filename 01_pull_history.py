"""
第 1 天：拉取 2018-2025 全历史日频数据，按年存 parquet。
支持断点续传：某年的文件已存在就跳过，中断后重跑即可。

用法：
    source .venv/bin/activate
    python -u 01_pull_history.py
"""
import time
from collections import defaultdict

import pandas as pd
import tushare as ts

from config import TUSHARE_TOKEN, DATA_DIR

# ==================== 配置 ====================
pro = ts.pro_api(TUSHARE_TOKEN)

START = "20180101"
END = "20251231"
INDEX_CODE = "000852.SH"          # 中证 1000

RAW = DATA_DIR / "raw"
RAW.mkdir(parents=True, exist_ok=True)

SLEEP = 0.35                       # 200 次/分钟上限，留点余量
SINGLE_CALL_LIMIT = 5800           # 单次返回行数上限附近，超过则按交易所拆分
SPLIT_SUPPORTED = {"daily", "daily_basic"}
API_NAMES = {"daily": "日线行情", "adj_factor": "复权因子",
             "daily_basic": "每日指标"}

BASIC_FIELDS = (
    "ts_code,trade_date,close,turnover_rate,turnover_rate_f,volume_ratio,"
    "pe,pe_ttm,pb,ps_ttm,total_share,float_share,free_share,total_mv,circ_mv"
)


# ==================== 工具函数 ====================
def retry(fn, tries=3, wait=2):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            if i == tries - 1:
                raise
            print(f"    重试 {i + 1}/{tries}：{e}")
            time.sleep(wait * (i + 1))


def fetch_day(api_name, d, fields=None):
    """拉单日数据；若接近单次上限，按交易所拆分再拼。"""
    params = {"trade_date": d}
    if fields:
        params["fields"] = fields

    df = retry(lambda: pro.query(api_name, **params))

    if (df is not None and len(df) >= SINGLE_CALL_LIMIT
            and api_name in SPLIT_SUPPORTED):
        parts = []
        for ex in ("SSE", "SZSE", "BSE"):
            p = dict(params)
            p["exchange"] = ex
            sub = retry(lambda: pro.query(api_name, **p))
            if sub is not None and len(sub):
                parts.append(sub)
        if parts:
            df = pd.concat(parts, ignore_index=True)
    return df


# ==================== 交易日 ====================
cal = pro.trade_cal(exchange="SSE", start_date=START, end_date=END)
dates = cal.loc[cal["is_open"] == 1, "cal_date"].sort_values().tolist()
print(f"交易日数：{len(dates)}（{dates[0]} ~ {dates[-1]}）")

by_year = defaultdict(list)
for d in dates:
    by_year[d[:4]].append(d)


# ==================== 按日拉取（分年落盘） ====================
def pull(api_name, fields=None):
    print(f"\n{'=' * 60}")
    print(f"===== {api_name}（{API_NAMES.get(api_name, '')}） =====")
    print(f"{'=' * 60}")

    for year, ds in sorted(by_year.items()):
        out = RAW / f"{api_name}_{year}.parquet"
        if out.exists():
            print(f"  {year} 已存在，跳过")
            continue

        rows = []
        t0 = time.time()
        for i, d in enumerate(ds, 1):
            try:
                df = fetch_day(api_name, d, fields)
            except Exception as e:
                print(f"  ❌ {d} 失败：{e}")
                continue
            if df is not None and len(df):
                rows.append(df)
            time.sleep(SLEEP)

            if i % 20 == 0:
                n = sum(len(r) for r in rows)
                print(f"  {year} {i}/{len(ds)}  累计 {n:,} 行  "
                      f"{time.time() - t0:.0f}s")

        if rows:
            all_df = pd.concat(rows, ignore_index=True)
            all_df.to_parquet(out, index=False)
            print(f"  ✅ {year} 完成：{len(all_df):,} 行 → {out.name}")
        else:
            print(f"  ⚠️  {year} 无数据")


# ==================== 指数成分（按月） ====================
def pull_index_weight(index_code=INDEX_CODE, start=START, end=END):
    print(f"\n{'=' * 60}")
    print(f"===== index_weight（{index_code} 成分） =====")
    print(f"{'=' * 60}")

    out = RAW / f"index_weight_{index_code}.parquet"
    if out.exists():
        print("  已存在，跳过")
        return

    frames = []
    periods = pd.period_range(start=start, end=end, freq="M")
    for i, p in enumerate(periods, 1):
        s = p.start_time.strftime("%Y%m%d")
        e = p.end_time.strftime("%Y%m%d")
        try:
            df = retry(lambda: pro.index_weight(
                index_code=index_code, start_date=s, end_date=e))
            if df is not None and len(df):
                frames.append(df)
        except Exception as ex:
            print(f"  ❌ {p} 失败：{ex}")
        time.sleep(SLEEP)

        if i % 12 == 0:
            print(f"  {p.year} 完成  累计 {sum(len(f) for f in frames):,} 行")

    if frames:
        all_df = (pd.concat(frames, ignore_index=True)
                  .drop_duplicates(subset=["index_code", "con_code", "trade_date"]))
        all_df.to_parquet(out, index=False)
        print(f"  ✅ 完成：{len(all_df):,} 行，"
              f"{all_df['trade_date'].nunique()} 个日期 → {out.name}")


# ==================== 跑完自检 ====================
def verify():
    print(f"\n{'=' * 60}")
    print("===== 数据自检 =====")
    print(f"{'=' * 60}")

    for f in sorted(RAW.glob("*.parquet")):
        df = pd.read_parquet(f)
        size_mb = f.stat().st_size / 1e6
        n_date = df["trade_date"].nunique() if "trade_date" in df else -1
        n_code = df["ts_code"].nunique() if "ts_code" in df else -1
        print(f"  {f.name:<32} {len(df):>10,} 行  "
              f"{n_date:>4} 日  {n_code:>5} 只  {size_mb:>7.1f} MB")

    print("\n===== 检查重点 =====")
    print("  · daily_YYYY：每年行数应 ≈ 交易日数 × 5300")
    print("  · 若某年行数明显偏少 → 该年被截断，需要按交易所重拉")
    print("  · index_weight：应有 ~96 个日期，每日 1000 只")


# ==================== 主程序 ====================
if __name__ == "__main__":
    t_all = time.time()

    pull("daily")
    pull("adj_factor")
    pull("daily_basic", fields=BASIC_FIELDS)
    pull_index_weight(INDEX_CODE)

    verify()

    print(f"\n全部完成，总耗时 {(time.time() - t_all) / 60:.1f} 分钟")
