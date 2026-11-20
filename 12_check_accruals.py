#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
12_check_accruals.py  (v2)
应计项因子准入检验。否决顺序：相关性 → IC/ICIR → 五分组。

用法：
    .venv/bin/python 12_check_accruals.py              # 原始因子（默认）
    .venv/bin/python 12_check_accruals.py --neutral    # 读 factors_neutral.parquet 的 acc

v2 修改
  - 收益矩阵只保留 .SH / .SZ：北交所 pct_chg 有垃圾行（max 200,400%），
    会把全样本收益标准差撑到 2.35（正常 0.06–0.09）
  - fwd_5 按 1% / 99% winsorize，防止极值污染分组均值
  - 新增 --neutral，用于中性化后的复测（预承诺：仍 <0.02 即永久关闭）
  - 新增分年 IC 表，判断信号是否只是某一年的产物
"""
import glob
import sys

import numpy as np
import pandas as pd

FACTOR_COLS = ["rev_5", "turn_20", "vol_20", "ep"]
NEW = "acc"
USE_NEUTRAL = "--neutral" in sys.argv


# ==================== 1. 因子 ====================
if USE_NEUTRAL:
    path = "data/panel/factors_neutral.parquet"
    fac = pd.read_parquet(path)
    need = ["trade_date", "ts_code"] + FACTOR_COLS + ["ln_mv", NEW]
    missing = [c for c in need if c not in fac.columns]
    if missing:
        raise KeyError(f"{path} 缺少 {missing}：先把 acc 加进 06 的因子列表并重跑 06")
    fac = fac[need].drop_duplicates(["trade_date", "ts_code"])
    print(f"[neutral] {path}  {fac.shape}")
else:
    fac = (pd.read_parquet("data/panel/factors.parquet",
                           columns=["trade_date", "ts_code"] + FACTOR_COLS + ["ln_mv"])
             .drop_duplicates(["trade_date", "ts_code"]))
    print(f"[raw] factors.parquet  {fac.shape}")

    acc = pd.read_parquet("data/panel/factor_accruals.parquet",
                          columns=["trade_date", "ts_code", "factor_z"])
    acc["trade_date"] = pd.to_datetime(acc.trade_date).dt.strftime("%Y%m%d")
    acc = (acc.rename(columns={"factor_z": NEW})
              .drop_duplicates(["trade_date", "ts_code"]))
    fac = fac.merge(acc, on=["trade_date", "ts_code"], how="left")
    print(f"      合并 accruals 后 {fac.shape} | acc 非空 {fac[NEW].notna().mean():.1%}")


# ==================== 2. 收益矩阵（复刻 04 口径） ====================
frames = [pd.read_parquet(f, columns=["ts_code", "trade_date", "pct_chg"])
          for f in sorted(glob.glob("data/raw/daily_[0-9][0-9][0-9][0-9].parquet"))]
daily = pd.concat(frames, ignore_index=True).drop_duplicates(["trade_date", "ts_code"])

n0 = len(daily)
daily = daily[daily.ts_code.str.endswith((".SH", ".SZ"))]
print(f"\ndaily {n0:,} -> {len(daily):,}  （剔除北交所 {n0 - len(daily):,} 行）")

all_dates = np.sort(daily.trade_date.unique())
all_codes = np.sort(daily.ts_code.unique())

ret = (daily.pivot(index="trade_date", columns="ts_code", values="pct_chg")
            .reindex(index=all_dates, columns=all_codes) / 100.0)
ret = ret.fillna(0.0)
px = (1 + ret).cumprod()

fwd = px.shift(-5) / px - 1.0
fwd_long = fwd.stack().rename("fwd_5").reset_index()
fwd_long.columns = ["trade_date", "ts_code", "fwd_5"]

print(f"fwd_5 原始 std = {fwd_long.fwd_5.std():.4f}   （期望 0.06–0.09）")
lo, hi = fwd_long.fwd_5.quantile([0.01, 0.99])
n_clip = int(((fwd_long.fwd_5 < lo) | (fwd_long.fwd_5 > hi)).sum())
fwd_long["fwd_5"] = fwd_long.fwd_5.clip(lo, hi)
print(f"winsorize 1%/99% = [{lo:.4f}, {hi:.4f}]，截断 {n_clip:,} 行")
print(f"winsorize 后 std = {fwd_long.fwd_5.std():.4f}")


# ==================== 3. 合并 ====================
df = (fac.merge(fwd_long, on=["trade_date", "ts_code"], how="left")
         .drop_duplicates(["trade_date", "ts_code"]))
print(f"\n合并 {df.shape} | 新因子非空 {df[NEW].notna().mean():.1%} "
      f"| fwd_5 非空 {df.fwd_5.notna().mean():.1%}")


# ==================== 4. 工具 ====================
def daily_rank_corr(frame, x, y, min_n=50):
    s = frame[["trade_date", x, y]].dropna()
    if s.empty:
        return pd.Series(dtype=float)
    s = s.assign(rx=s.groupby("trade_date")[x].rank(),
                 ry=s.groupby("trade_date")[y].rank())
    g = s.groupby("trade_date")
    return g.apply(lambda z: z.rx.corr(z.ry) if len(z) >= min_n else np.nan).dropna()


def ic_stats(series, name):
    m, sd, n = series.mean(), series.std(), len(series)
    if n == 0 or sd == 0:
        return {"factor": name, "IC": np.nan, "IC_std": np.nan,
                "ICIR": np.nan, "t": np.nan, "days": n}
    return {"factor": name, "IC": round(m, 4), "IC_std": round(sd, 4),
            "ICIR": round(m / sd, 3), "t": round(m / sd * np.sqrt(n), 2), "days": n}


# ==================== 5. 检验 1：相关性（否决项） ====================
print("\n===== 与现有因子的截面相关性 =====")
rows = []
for c in FACTOR_COLS + ["ln_mv"]:
    r = daily_rank_corr(df, NEW, c)
    rows.append({"vs": c,
                 "corr": round(r.mean(), 3),
                 "abs_corr_mean": round(r.abs().mean(), 3),
                 "abs_corr_p90": round(r.abs().quantile(0.9), 3)})
corr_tab = pd.DataFrame(rows)
print(corr_tab.to_string(index=False))

# 门槛只看信号因子；ln_mv 是中性化变量，06 会残差化掉
max_corr = corr_tab.loc[corr_tab.vs.isin(FACTOR_COLS), "abs_corr_mean"].max()


# ==================== 6. 检验 2：IC ====================
print("\n===== Rank IC（对 fwd_5） =====")
stats = [ic_stats(daily_rank_corr(df, NEW, "fwd_5"), "acc_ttm")]
for c in FACTOR_COLS:
    stats.append(ic_stats(daily_rank_corr(df, c, "fwd_5"), c))
ic_tab = pd.DataFrame(stats)
print(ic_tab.to_string(index=False))


# ==================== 7. 检验 3：五分组 ====================
print("\n===== 五分组（等权，5 日，winsorized） =====")
s = df[["trade_date", NEW, "fwd_5"]].dropna().copy()
s["g"] = s.groupby("trade_date")[NEW].transform(
    lambda x: pd.qcut(x.rank(method="first"), 5, labels=False))
grp = s.groupby("g")["fwd_5"].agg(["mean", "median", "count"])
grp["mean_bp"] = (grp["mean"] * 1e4).round(1)
grp["median_bp"] = (grp["median"] * 1e4).round(1)
grp["ann"] = (1 + grp["mean"]) ** (252 / 5) - 1
print(grp[["mean_bp", "median_bp", "ann", "count"]].to_string())
spread = grp.loc[4, "ann"] - grp.loc[0, "ann"]
monotone = bool((grp["mean"].diff().dropna() > 0).all())


# ==================== 8. 分年 IC（判稳定性） ====================
print("\n===== acc_ttm 分年 IC =====")
sy = df[["trade_date", NEW, "fwd_5"]].dropna().copy()
sy["year"] = sy.trade_date.str[:4]
yrows = []
for y, g in sy.groupby("year"):
    r = daily_rank_corr(g, NEW, "fwd_5")
    yrows.append({"year": y, "IC": round(r.mean(), 4) if len(r) else np.nan,
                  "days": len(r)})
print(pd.DataFrame(yrows).to_string(index=False))


# ==================== 9. 裁决 ====================
ic, icir = ic_tab.loc[0, "IC"], ic_tab.loc[0, "ICIR"]
tag = "中性化后" if USE_NEUTRAL else "原始"
print("\n" + "=" * 54)
print(f"[{tag}] 相关性(信号因子) |corr| = {max_corr:.3f}   门槛 < 0.30")
print(f"[{tag}] IC   = {ic:+.4f}                        门槛 |IC| >= 0.02")
print(f"[{tag}] ICIR = {icir:+.3f}                        门槛 |ICIR| >= 0.30")
print(f"[{tag}] 五分组单调 = {monotone}；Q5-Q1 年化 = {spread:+.2%}")
ok = (max_corr < 0.30) and (abs(ic) >= 0.02) and (abs(icir) >= 0.30)
print("裁决:", "PASS → 加入 06/07 做风格中性 alpha" if ok else "REJECT → 放弃这条线索，不必回测")
print("=" * 54)
