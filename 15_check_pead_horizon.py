#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
15_check_pead_horizon.py
在 PEAD 假设的正确时间尺度（20–60 交易日）上检验一次，并按公告后天数拆解衰减。

预承诺：fwd_20 的 IC >= 0.02 且 ICIR >= 0.3 且分年同号 -> 进入组合测试；否则关闭。
"""
import glob

import numpy as np
import pandas as pd

HORIZONS = [5, 20, 60]
AGE_BUCKETS = [(0, 10), (11, 20), (21, 40), (41, 60), (61, 90)]


# ---------- 1. 因子 ----------
res = pd.read_parquet("data/panel/factor_pead.parquet")
res["age"] = (res.trade_date - res.f_ann_date).dt.days
print(f"因子 {res.shape} | age 范围 {res.age.min()}–{res.age.max()}")


# ---------- 2. 收益矩阵 ----------
frames = [pd.read_parquet(f, columns=["ts_code", "trade_date", "pct_chg"])
          for f in sorted(glob.glob("data/raw/daily_[0-9][0-9][0-9][0-9].parquet"))]
daily = pd.concat(frames, ignore_index=True).drop_duplicates(["trade_date", "ts_code"])
daily = daily[daily.ts_code.str.endswith((".SH", ".SZ"))]

all_dates = np.sort(daily.trade_date.unique())
all_codes = np.sort(daily.ts_code.unique())

ret = (daily.pivot(index="trade_date", columns="ts_code", values="pct_chg")
            .reindex(index=all_dates, columns=all_codes) / 100.0).fillna(0.0)
px = (1 + ret).cumprod()

# 股票池掩码，保证 stack 只产生 ~190 万行而不是 1100 万
mask = pd.read_parquet("data/panel/tradable.parquet")
mask.index = mask.index.astype(str)
mask = (mask.reindex(index=px.index, columns=px.columns)
            .fillna(False).astype(bool))


def fwd_long(h):
    f = px.shift(-h) / px - 1.0
    f = f.where(mask)                        # 非在册/无价格 -> NaN，stack 时丢弃
    out = f.stack().rename(f"fwd_{h}").reset_index()
    out.columns = ["trade_date", "ts_code", f"fwd_{h}"]
    lo, hi = out[f"fwd_{h}"].quantile([0.01, 0.99])
    out[f"fwd_{h}"] = out[f"fwd_{h}"].clip(lo, hi)
    out["trade_date"] = pd.to_datetime(out.trade_date, format="%Y%m%d")
    out["trade_date"] = out["trade_date"].astype(res.trade_date.dtype)
    print(f"  fwd_{h}: {len(out):,} 行  std={out[f'fwd_{h}'].std():.4f}")
    return out


# ---------- 3. 工具 ----------
def daily_rank_corr(frame, x, y, min_n=50):
    t = frame[["trade_date", x, y]].dropna()
    if t.empty:
        return pd.Series(dtype=float)
    t = t.assign(rx=t.groupby("trade_date")[x].rank(),
                 ry=t.groupby("trade_date")[y].rank())
    return (t.groupby("trade_date")
             .apply(lambda z: z.rx.corr(z.ry) if len(z) >= min_n else np.nan)
             .dropna())


def stat(r, name):
    if len(r) == 0:
        return {"": name, "IC": np.nan, "ICIR": np.nan, "t": np.nan, "days": 0}
    return {"": name, "IC": round(r.mean(), 4), "ICIR": round(r.mean() / r.std(), 3),
            "t": round(r.mean() / r.std() * np.sqrt(len(r)), 2), "days": len(r)}


# ---------- 4. 分期限 IC ----------
print("\n===== 分期限 Rank IC =====")
rows, fwd20 = [], None
for h in HORIZONS:
    fl = fwd_long(h)
    d = res.merge(fl, on=["trade_date", "ts_code"], how="left")
    r = daily_rank_corr(d, "factor_z", f"fwd_{h}")
    rows.append(stat(r, f"fwd_{h}"))
    if h == 20:
        fwd20 = d
print(pd.DataFrame(rows).to_string(index=False))
print("注：20/60 日窗口重叠，IC 序列自相关，t 值偏乐观，以 IC/ICIR 为准")

# ---------- 5. 按公告后天数拆解（fwd_20） ----------
print("\n===== 衰减曲线（fwd_20 × 公告后天数） =====")
rows = []
for lo, hi in AGE_BUCKETS:
    sub = fwd20[(fwd20.age >= lo) & (fwd20.age <= hi)]
    r = daily_rank_corr(sub, "factor_z", "fwd_20")
    rows.append(stat(r, f"D+{lo}–{hi}"))
print(pd.DataFrame(rows).to_string(index=False))

# ---------- 6. 五分组（fwd_20） ----------
print("\n===== 五分组（fwd_20） =====")
s = fwd20[["trade_date", "factor_z", "fwd_20"]].dropna().copy()
s["g"] = s.groupby("trade_date")["factor_z"].transform(
    lambda x: pd.qcut(x.rank(method="first"), 5, labels=False))
grp = s.groupby("g")["fwd_20"].agg(["mean", "count"])
grp["mean_bp"] = (grp["mean"] * 1e4).round(1)
print(grp[["mean_bp", "count"]].to_string())
spread_ann = (1 + grp.loc[4, "mean"]) ** (252 / 20) - 1 - ((1 + grp.loc[0, "mean"]) ** (252 / 20) - 1)
print(f"Q5-Q1 年化 = {spread_ann:+.2%}")

# ---------- 7. 分年 ----------
print("\n===== fwd_20 分年 IC =====")
fwd20["year"] = fwd20.trade_date.dt.year.astype(str)
yr = []
for y, chunk in fwd20.groupby("year"):
    r = daily_rank_corr(chunk, "factor_z", "fwd_20")
    yr.append(stat(r, y))
print(pd.DataFrame(yr).to_string(index=False))
