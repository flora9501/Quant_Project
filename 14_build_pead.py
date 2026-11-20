#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
14_build_pead.py  (v2)
PEAD / SUE 因子：单季净利润同比"超预期"，事件驱动。

构建
  单季净利润 ni_q（归母）
    -> 同比增速 g = (ni_q_t - ni_q_{t-4}) / |ni_q_{t-4}|
    -> SUE = (g_t - mean(g 过去8季)) / std(g 过去8季)
  事件日 = f_ann_date，信号在公告后 90 个自然日内有效（约 60 个交易日）

输出
  data/panel/factor_pead.parquet
  columns = ts_code, trade_date, f_ann_date, end_date, sue, factor, factor_z

v2 修改
  - 快检部分统一日期类型：res 是 datetime64[us]，pivot 出来是 str，直接 merge 会报
    "You are trying to merge on datetime64[us] and str columns"
  - 分年用 dt.year，不用 dt.str[:4]
  - 收益矩阵显式 reindex 到交易日历，保证 shift(-5) 是 5 个交易日
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw_fina")
EVENT_WINDOW_DAYS = 90        # 公告后信号有效期（自然日）
MIN_BASE_EARNINGS = 1e6       # 去年同期净利润绝对值下限（元）
SUE_CLIP = 5.0


# ==================== 1. 读 income ====================
def load_income():
    fs = sorted(RAW.glob("income_part*.parquet"))
    if not fs:
        raise FileNotFoundError("没有 income 数据，先跑 10_pull_fina.py")

    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    df["ts_code"] = df.ts_code.astype(str)
    df["end_date"] = df.end_date.astype(str)
    df["f_ann_date"] = df.f_ann_date.fillna(df.ann_date).astype(str)
    df["f_ann_date_dt"] = pd.to_datetime(df.f_ann_date, format="%Y%m%d", errors="coerce")
    df["update_flag"] = df.update_flag.fillna("0").astype(str)
    df = df.dropna(subset=["f_ann_date_dt", "end_date"])

    df = df.sort_values(["ts_code", "end_date", "f_ann_date_dt", "update_flag"],
                        ascending=[True, True, True, False])
    df = df.drop_duplicates(["ts_code", "end_date"], keep="first")
    print(f"income 去重后 {len(df):,} 行  codes={df.ts_code.nunique()}")
    return df


# ==================== 2. 剔除金融股 ====================
def load_comp_type():
    fs = sorted(RAW.glob("balancesheet_part*.parquet"))
    df = pd.concat([pd.read_parquet(f, columns=["ts_code", "comp_type"])
                    for f in fs], ignore_index=True)
    df["ts_code"] = df.ts_code.astype(str)
    df["comp_type"] = df.comp_type.fillna("9").astype(str)
    return (df.groupby("ts_code")["comp_type"]
              .agg(lambda s: s.value_counts().idxmax())
              .rename("comp_type"))


inc = load_income()
inc = inc.merge(load_comp_type(), on="ts_code", how="left")
inc["comp_type"] = inc.comp_type.fillna("9")

before = len(inc)
inc = inc[inc.comp_type == "1"].copy()
print(f"剔除金融/其他 {before - len(inc):,} 行 -> {len(inc):,} 行")


# ==================== 3. YTD 累计 -> 单季 ====================
inc["q"] = inc.end_date.str[4:6].map({"03": 1, "06": 2, "09": 3, "12": 4})
inc = inc[inc.q.notna()].copy()
inc["q"] = inc.q.astype(int)

inc = inc.sort_values(["ts_code", "end_date"])
g = inc.groupby("ts_code", sort=False)
prev = g["n_income_attr_p"].shift(1)
prevq = g["q"].shift(1)
inc["ni_q"] = inc["n_income_attr_p"].where(
    inc.q == 1, inc["n_income_attr_p"] - prev.where(prevq == inc.q - 1))

inc["qnum"] = inc.end_date.str[:4].astype(int) * 4 + inc.q
print(f"单季净利润可用 {inc.ni_q.notna().sum():,} 行")


# ==================== 4. 同比增速 + SUE ====================
inc = inc.sort_values(["ts_code", "end_date"])
g = inc.groupby("ts_code", sort=False)

lag4 = g["ni_q"].shift(4)
lag4_qnum = g["qnum"].shift(4)
ok = (inc["qnum"] - lag4_qnum == 4) & lag4.notna() & (lag4.abs() >= MIN_BASE_EARNINGS)
inc["g_yoy"] = np.where(ok, (inc["ni_q"] - lag4) / lag4.abs(), np.nan)
print(f"同比增速可用 {inc.g_yoy.notna().sum():,} 行")

gg = inc.groupby("ts_code")["g_yoy"]
inc["g_mu"] = gg.transform(lambda s: s.shift(1).rolling(8, min_periods=4).mean())
inc["g_sd"] = gg.transform(lambda s: s.shift(1).rolling(8, min_periods=4).std())

valid = inc.g_sd.notna() & (inc.g_sd > 1e-6)
inc["sue"] = np.where(valid, (inc.g_yoy - inc.g_mu) / inc.g_sd, np.nan)
inc["factor"] = inc["sue"].clip(-SUE_CLIP, SUE_CLIP)
print(f"SUE 可用 {inc.sue.notna().sum():,} 行")
print(inc.sue.describe(percentiles=[0.01, 0.5, 0.99]).round(3).to_string())

print("\n每年公告事件数（含 SUE 的）:")
print(inc.dropna(subset=["sue"]).groupby(inc.end_date.str[:4]).size().to_string())


# ==================== 5. 宽表 -> 长表 ====================
pan = pd.read_parquet("data/panel/tradable.parquet")
if not isinstance(pan.index, pd.DatetimeIndex):
    pan.index = pd.to_datetime(pan.index.astype(str), format="%Y%m%d")
pan.index.name = "trade_date"
pan = pan[~pan.index.duplicated(keep="first")]

s = pan.stack()
s = s[s.astype(bool)]
long = s.reset_index()
long.columns = ["trade_date", "ts_code", "in_universe"]
long = long[["trade_date", "ts_code"]].copy()
long["ts_code"] = long.ts_code.astype(str)
print(f"\n面板 {len(long):,} 行")


# ==================== 6. 事件对齐 ====================
long = long.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
ev = (inc.dropna(subset=["factor", "f_ann_date_dt"])
         .sort_values(["f_ann_date_dt", "ts_code"])
         .reset_index(drop=True))

assert long.trade_date.is_monotonic_increasing, "long 未按 trade_date 排序"
assert ev.f_ann_date_dt.is_monotonic_increasing, "ev 未按公告日排序"

out = pd.merge_asof(
    long,
    ev[["ts_code", "f_ann_date_dt", "end_date", "sue", "factor"]],
    left_on="trade_date", right_on="f_ann_date_dt", by="ts_code",
    direction="backward",
    allow_exact_matches=False,        # 公告次日才建仓
)
print(f"对齐后非空 {out.factor.notna().mean():.1%}")

# 只保留公告后 EVENT_WINDOW_DAYS 天内的信号
age = (out.trade_date - out.f_ann_date_dt).dt.days
out.loc[age > EVENT_WINDOW_DAYS, "factor"] = np.nan
print(f"限制在公告后 {EVENT_WINDOW_DAYS} 天内 -> 非空 {out.factor.notna().mean():.1%}")


# ==================== 7. 去极值 + 截面标准化 ====================
def mad_winsorize(x, n=3):
    med = x.median()
    mad = (x - med).abs().median() * 1.4826
    if not np.isfinite(mad) or mad == 0:
        return x
    return x.clip(med - n * mad, med + n * mad)


out["factor_w"] = out.groupby("trade_date")["factor"].transform(mad_winsorize)
out["factor_z"] = out.groupby("trade_date")["factor_w"].transform(
    lambda x: (x - x.mean()) / x.std(ddof=0))

res = (out.dropna(subset=["factor_z"])
          [["ts_code", "trade_date", "f_ann_date_dt", "end_date",
            "sue", "factor", "factor_z"]]
          .rename(columns={"f_ann_date_dt": "f_ann_date"})
          .sort_values(["trade_date", "ts_code"]))

res.to_parquet("data/panel/factor_pead.parquet", index=False)
print(f"\n已写出 data/panel/factor_pead.parquet  {res.shape}")


# ==================== 8. 快检：IC + 五分组 ====================
print("\n===== 快检（fwd_5，winsorized 1%/99%）=====")
frames = [pd.read_parquet(f, columns=["ts_code", "trade_date", "pct_chg"])
          for f in sorted(glob.glob("data/raw/daily_[0-9][0-9][0-9][0-9].parquet"))]
daily = pd.concat(frames, ignore_index=True).drop_duplicates(["trade_date", "ts_code"])
daily = daily[daily.ts_code.str.endswith((".SH", ".SZ"))]

all_dates = np.sort(daily.trade_date.unique())
all_codes = np.sort(daily.ts_code.unique())

ret = (daily.pivot(index="trade_date", columns="ts_code", values="pct_chg")
            .reindex(index=all_dates, columns=all_codes) / 100.0)
ret = ret.fillna(0.0)
px = (1 + ret).cumprod()

fwd = px.shift(-5) / px - 1.0
fwd_long = fwd.stack().rename("fwd_5").reset_index()
fwd_long.columns = ["trade_date", "ts_code", "fwd_5"]

lo, hi = fwd_long.fwd_5.quantile([0.01, 0.99])
fwd_long["fwd_5"] = fwd_long.fwd_5.clip(lo, hi)
print(f"fwd_5 std = {fwd_long.fwd_5.std():.4f}  （期望 0.06–0.09）")

# 统一日期类型：res 是 datetime64[us]，pivot 出来是 str
fwd_long["trade_date"] = pd.to_datetime(fwd_long["trade_date"], format="%Y%m%d")
fwd_long["trade_date"] = fwd_long["trade_date"].astype(res.trade_date.dtype)

df = (res.merge(fwd_long, on=["trade_date", "ts_code"], how="left")
         .rename(columns={"factor_z": "pead"}))


def daily_rank_corr(frame, x, y, min_n=50):
    t = frame[["trade_date", x, y]].dropna()
    if t.empty:
        return pd.Series(dtype=float)
    t = t.assign(rx=t.groupby("trade_date")[x].rank(),
                 ry=t.groupby("trade_date")[y].rank())
    return (t.groupby("trade_date")
             .apply(lambda z: z.rx.corr(z.ry) if len(z) >= min_n else np.nan)
             .dropna())


ic = daily_rank_corr(df, "pead", "fwd_5")
if len(ic):
    print(f"\n全样本 IC = {ic.mean():+.4f}  ICIR = {ic.mean() / ic.std():.3f}  "
          f"t = {ic.mean() / ic.std() * np.sqrt(len(ic)):.2f}  days = {len(ic)}")

    df["year"] = df.trade_date.dt.year.astype(str)
    print("\n分年 IC:")
    for y, chunk in df.groupby("year"):
        r = daily_rank_corr(chunk, "pead", "fwd_5")
        if len(r):
            print(f"  {y}  IC = {r.mean():+.4f}  days = {len(r)}")

    s = df[["trade_date", "pead", "fwd_5"]].dropna().copy()
    s["g"] = s.groupby("trade_date")["pead"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 5, labels=False))
    grp = s.groupby("g")["fwd_5"].agg(["mean", "median", "count"])
    grp["mean_bp"] = (grp["mean"] * 1e4).round(1)
    grp["median_bp"] = (grp["median"] * 1e4).round(1)
    print("\n五分组:")
    print(grp[["mean_bp", "median_bp", "count"]].to_string())
else:
    print("IC 无法计算：检查 fwd_5 与因子日期是否对齐")
