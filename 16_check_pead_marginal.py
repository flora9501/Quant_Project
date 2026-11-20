#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
16_check_pead_marginal.py
SUE 相对现有 4 因子的增量价值：
  1) 每日截面相关性
  2) 把 SUE 对 4 因子做截面回归取残差，看残差还剩多少 IC
  3) 4 因子等权复合 vs 5 因子等权复合的 IC 对比（fwd_20）

预承诺：残差 IC >= 0.015 且 max|corr| < 0.3 -> 进 07 做组合测试；否则关闭。
"""
import glob

import numpy as np
import pandas as pd

FACTORS = ["rev_5", "turn_20", "vol_20", "ep"]
H = 20


# ---------- 1. 数据 ----------
fac = (pd.read_parquet("data/panel/factors.parquet",
                       columns=["trade_date", "ts_code"] + FACTORS)
         .drop_duplicates(["trade_date", "ts_code"]))

pead = pd.read_parquet("data/panel/factor_pead.parquet",
                       columns=["trade_date", "ts_code", "factor_z"])
pead["trade_date"] = pd.to_datetime(pead.trade_date).dt.strftime("%Y%m%d")
pead = pead.rename(columns={"factor_z": "sue"}).drop_duplicates(["trade_date", "ts_code"])

df = fac.merge(pead, on=["trade_date", "ts_code"], how="inner")
print(f"合并 {df.shape} | sue 非空 {df.sue.notna().mean():.1%}")

# 截面标准化
for c in FACTORS + ["sue"]:
    df[c + "_z"] = df.groupby("trade_date")[c].transform(
        lambda x: (x - x.mean()) / x.std(ddof=0))


# ---------- 2. 前瞻收益 ----------
frames = [pd.read_parquet(f, columns=["ts_code", "trade_date", "pct_chg"])
          for f in sorted(glob.glob("data/raw/daily_[0-9][0-9][0-9][0-9].parquet"))]
daily = pd.concat(frames, ignore_index=True).drop_duplicates(["trade_date", "ts_code"])
daily = daily[daily.ts_code.str.endswith((".SH", ".SZ"))]

ret = (daily.pivot(index="trade_date", columns="ts_code", values="pct_chg")
            .reindex(index=np.sort(daily.trade_date.unique()),
                     columns=np.sort(daily.ts_code.unique())) / 100.0).fillna(0.0)
px = (1 + ret).cumprod()

f = px.shift(-H) / px - 1.0
fl = f.stack().rename("fwd").reset_index()
fl.columns = ["trade_date", "ts_code", "fwd"]
lo, hi = fl.fwd.quantile([0.01, 0.99])
fl["fwd"] = fl.fwd.clip(lo, hi)

df = df.merge(fl, on=["trade_date", "ts_code"], how="left")
print(f"加收益后 {df.shape} | fwd 非空 {df.fwd.notna().mean():.1%}")


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


def ic_line(r, name):
    return {"": name, "IC": round(r.mean(), 4),
            "ICIR": round(r.mean() / r.std(), 3),
            "t": round(r.mean() / r.std() * np.sqrt(len(r)), 2), "days": len(r)}


# ---------- 4. 相关性 ----------
print("\n===== sue 与现有因子的截面相关性 =====")
rows = []
for c in FACTORS:
    r = daily_rank_corr(df, "sue", c)
    rows.append({"vs": c, "corr": round(r.mean(), 3),
                 "abs_corr": round(r.abs().mean(), 3)})
corr_tab = pd.DataFrame(rows)
print(corr_tab.to_string(index=False))
max_corr = corr_tab.abs_corr.max()


# ---------- 5. 残差 IC ----------
print("\n===== 残差 IC（sue 对 4 因子截面回归后） =====")
cols = [c + "_z" for c in FACTORS]


def resid_of(g):
    X = g[cols].to_numpy(dtype=float)
    y = g["sue_z"].to_numpy(dtype=float)
    m = np.isfinite(X).all(axis=1) & np.isfinite(y)
    r = np.full(len(g), np.nan)
    if m.sum() > 60:
        Xm = np.c_[np.ones(m.sum()), X[m]]
        beta, *_ = np.linalg.lstsq(Xm, y[m], rcond=None)
        r[m] = y[m] - Xm @ beta
    return pd.Series(r, index=g.index)


df["sue_resid"] = df.groupby("trade_date", group_keys=False).apply(resid_of)
r_raw = daily_rank_corr(df, "sue_z", "fwd")
r_res = daily_rank_corr(df, "sue_resid", "fwd")
print(pd.DataFrame([ic_line(r_raw, "sue 原始"), ic_line(r_res, "sue 残差")]).to_string(index=False))


# ---------- 6. 复合 IC 对比 ----------
print("\n===== 4 因子 vs 5 因子等权复合 =====")
df["comp4"] = df[[c + "_z" for c in FACTORS]].mean(axis=1)
df["comp5"] = df[[c + "_z" for c in FACTORS] + ["sue_z"]].mean(axis=1)

rows = [ic_line(daily_rank_corr(df, "comp4", "fwd"), "comp4"),
        ic_line(daily_rank_corr(df, "comp5", "fwd"), "comp5")]
print(pd.DataFrame(rows).to_string(index=False))


# ---------- 7. 裁决 ----------
res_ic = r_res.mean()
print("\n" + "=" * 50)
print(f"max |corr|      = {max_corr:.3f}    门槛 < 0.30")
print(f"残差 IC         = {res_ic:+.4f}   门槛 >= 0.015")
print(f"原始 IC         = {r_raw.mean():+.4f}")
print("裁决:", "PASS -> 进 07 做组合测试" if (max_corr < 0.3 and res_ic >= 0.015)
      else "REJECT -> 关闭 PEAD，不再试变体")
print("=" * 50)
