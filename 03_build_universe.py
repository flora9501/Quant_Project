"""
第 2 天：构造时点可交易股票池
输出 data/panel/tradable.parquet（date × code 布尔矩阵）

五个条件：
  ① 在中证 1000 当期成分内（≤当天的最近一期快照）
  ② 上市满 60 个交易日
  ③ 当日不是 ST
  ④ 当日有行情记录（= 没停牌）
  ⑤ 当日成交额 > 0
"""
import time

import numpy as np
import pandas as pd

from config import DATA_DIR

RAW = DATA_DIR / "raw"
REF = DATA_DIR / "ref"
PANEL = DATA_DIR / "panel"
PANEL.mkdir(parents=True, exist_ok=True)


def log(m):
    print(m, flush=True)


t0 = time.time()

# ==================== 1. 行情：交易日 + 交易标记 ====================
log("① 读取日线...")
frames = []
for f in sorted(RAW.glob("daily_[0-9][0-9][0-9][0-9].parquet")):
    frames.append(pd.read_parquet(f, columns=["ts_code", "trade_date", "amount"]))
daily = pd.concat(frames, ignore_index=True).drop_duplicates(["trade_date", "ts_code"])

dates = np.sort(daily["trade_date"].unique())
codes = np.sort(daily["ts_code"].unique())
log(f"   交易日 {len(dates)}   股票 {len(codes)}")

# ④ 有行情 = 没停牌
traded = (daily.assign(v=True)
          .pivot(index="trade_date", columns="ts_code", values="v")
          .reindex(index=dates, columns=codes)
          .fillna(False).astype(bool))

# ⑤ 成交额 > 0
amount_pos = (daily.assign(v=daily["amount"] > 0)
              .pivot(index="trade_date", columns="ts_code", values="v")
              .reindex(index=dates, columns=codes)
              .fillna(False).astype(bool))

del daily
log(f"   完成 {time.time() - t0:.0f}s")

# ==================== 2. 指数成分：时点对齐 ====================
log("② 构建成分矩阵（时点对齐）...")
w = pd.read_parquet(RAW / "index_weight_000852.SH.parquet")
w["trade_date"] = pd.to_datetime(w["trade_date"])

# 每个快照的成分集合
members_by_snap = w.groupby("trade_date")["con_code"].apply(set).to_dict()
snap_dates = np.sort(list(members_by_snap.keys()))

# 快照质量检查
cnt_snap = w.groupby("trade_date")["con_code"].nunique()
log(f"   快照 {len(snap_dates)} 期  成分数 min={cnt_snap.min()} "
    f"max={cnt_snap.max()}  mean={cnt_snap.mean():.0f}")
log(f"   首个快照 {pd.Timestamp(snap_dates.min()).date()}  "
    f"末个快照 {pd.Timestamp(snap_dates.max()).date()}")

# 每个交易日 → ≤ 该日的最近一期快照
d_arr = pd.to_datetime(dates)
snap_s = pd.Series(pd.to_datetime(snap_dates), index=pd.to_datetime(snap_dates))
mapped = snap_s.reindex(d_arr, method="ffill")
valid_dates = mapped.notna().values
log(f"   无快照的交易日 {int((~valid_dates).sum())} 个（将被剔除）")

code_pos = {c: i for i, c in enumerate(codes)}
in_index = np.zeros((len(dates), len(codes)), dtype=bool)

for i, (d, sd) in enumerate(zip(dates, mapped)):
    if pd.isna(sd):
        continue
    members = members_by_snap[sd]
    pos = [code_pos[c] for c in members if c in code_pos]
    in_index[i, pos] = True

log(f"   完成 {time.time() - t0:.0f}s")

# ==================== 3. ST 标记 ====================
log("③ 构建 ST 矩阵...")
nc = pd.read_parquet(REF / "namechange.parquet")
nc["start_date"] = pd.to_datetime(nc["start_date"])
nc["end_date"] = pd.to_datetime(nc["end_date"], errors="coerce")
nc["is_st"] = nc["name"].astype(str).str.upper().str.contains("ST")
st_rows = nc[nc["is_st"]]

d_np = pd.to_datetime(dates).values
st_flag = np.zeros((len(dates), len(codes)), dtype=bool)

for _, r in st_rows.iterrows():
    j = code_pos.get(r["ts_code"])
    if j is None:
        continue
    i0 = np.searchsorted(d_np, np.datetime64(r["start_date"]))
    i1 = (len(dates) if pd.isna(r["end_date"])
          else np.searchsorted(d_np, np.datetime64(r["end_date"])))
    if i1 > i0:
        st_flag[i0:i1, j] = True

log(f"   ST 区间 {len(st_rows):,} 条，完成 {time.time() - t0:.0f}s")

# ==================== 4. 上市满 60 个交易日 ====================
log("④ 构建上市满 60 日矩阵...")
basic = pd.read_parquet(REF / "stock_basic.parquet")
dead = pd.read_parquet(REF / "stock_delisted.parquet")
allb = pd.concat([basic, dead], ignore_index=True)
list_map = dict(zip(allb["ts_code"],
                    pd.to_datetime(allb["list_date"], errors="coerce")))

listed_ok = np.zeros((len(dates), len(codes)), dtype=bool)
miss = 0
for c, ld in list_map.items():
    j = code_pos.get(c)
    if j is None:
        continue
    if pd.isna(ld):
        miss += 1
        continue

    if ld < pd.Timestamp(dates[0]):
        i0 = 0                              # ← 数据开始前就上市，全程可交易
    else:
        i0 = np.searchsorted(d_np, np.datetime64(ld)) + 60
    
    if i0 < len(dates):
        listed_ok[i0:, j] = True

log(f"   上市日期缺失 {miss} 只，完成 {time.time() - t0:.0f}s")

# ==================== 5. 合并五条件 + 剔除无快照日期 ====================
log("⑤ 合并...")
tradable = (in_index & listed_ok & (~st_flag) & traded.values & amount_pos.values)

res = pd.DataFrame(tradable, index=dates, columns=codes)
res = res.loc[valid_dates]                    # 丢掉快照开始之前的日期
res.to_parquet(PANEL / "tradable.parquet")

log(f"   ✅ 已保存 data/panel/tradable.parquet")
log(f"      形状 {res.shape}   完成 {time.time() - t0:.0f}s")

# ==================== 6. 年度统计 ====================
log("\n===== 每年平均可交易股票数 =====")
year = pd.Series(res.index).str[:4].values
cnt = res.sum(axis=1).values
stat = (pd.DataFrame({"year": year, "n": cnt})
        .groupby("year")["n"].agg(["mean", "min", "max"]).round(0))
print(stat.to_string())

# ==================== 7. 过滤漏斗 ====================
def funnel(day, idx):
    i = idx
    a = in_index[i]
    b = listed_ok[i]
    c = ~st_flag[i]
    d_ = traded.values[i]
    e = amount_pos.values[i]
    print(f"\n[{day}]")
    print(f"  中证1000 成分        {a.sum():>5}")
    print(f"  ├─ 剔除次新后        {(a & b).sum():>5}   （剔除 {(a & ~b).sum()}）")
    print(f"  ├─ 剔除 ST 后        {(a & b & c).sum():>5}   （剔除 {(a & b & ~c).sum()}）")
    print(f"  ├─ 剔除停牌后        {(a & b & c & d_).sum():>5}   （剔除 {(a & b & c & ~d_).sum()}）")
    print(f"  └─ 剔除零成交后      {(a & b & c & d_ & e).sum():>5}   （剔除 {(a & b & c & d_ & ~e).sum()}）")

log("\n===== 过滤漏斗 =====")
date_list = list(dates)
for day in ["20190628", "20220628", "20240628"]:
    if day in date_list:
        funnel(day, date_list.index(day))
    else:
        log(f"  {day} 不在交易日列表，跳过")

log(f"\n总耗时 {(time.time() - t0) / 60:.1f} 分钟")
