#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
11_build_accruals.py  (v4)
应计项因子：(净利润 − 经营现金流) / 平均总资产，TTM 口径，取负（低应计为优）。

前置：10_pull_fina.py 三张表全部跑完
输出：data/panel/factor_accruals.parquet
      columns = ts_code, trade_date, f_ann_date, end_date, acc_ttm, factor, factor_z

修订记录
  v2  修正 end_date 月份切片 [5:7] -> [4:6]（原写法让 q 全为 NaN，TTM 静默全空）
  v2  merge_asof 改用 datetime 键，避免 str/datetime 类型不匹配
  v2  update_flag 的 NaN 先填 '0'，避免排序时 'nan' > '1'
  v2  stack 先过滤再 reset_index，避免 1100 万行 object 列吃内存
  v3  新增陈旧报告检查：end_date 距交易日 >500 天置空
  v4  reset_index 后是 3 列（两个索引层 + 值列），按 3 列命名再取 2 列
  v4  merge_asof 的排序改为 on 键优先（要求全局单调，而非组内有序）
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw_fina")
STALE_DAYS = 500          # 报告期距交易日超过此天数视为失效


# ==================== 1. 读入 ====================
def load_parts(api):
    fs = sorted(RAW.glob(f"{api}_part*.parquet"))
    if not fs:
        raise FileNotFoundError(f"没有 {api} 数据，先跑 10_pull_fina.py")

    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    df["ts_code"] = df.ts_code.astype(str)
    df["end_date"] = df.end_date.astype(str)

    # 先 fillna 再 astype(str)，否则 NaN 会变成字符串 'nan'
    df["f_ann_date"] = df.f_ann_date.fillna(df.ann_date).astype(str)
    df["f_ann_date_dt"] = pd.to_datetime(df["f_ann_date"], format="%Y%m%d", errors="coerce")
    df["update_flag"] = df.update_flag.fillna("0").astype(str)

    n0 = len(df)
    df = df.dropna(subset=["f_ann_date_dt", "end_date"])
    if len(df) < n0:
        print(f"{api}: 丢弃 {n0 - len(df):,} 行无法解析公告日的记录")

    print(f"{api:13s} raw={len(df):>7,} 行  codes={df.ts_code.nunique()}")
    return df


def dedup(df):
    """同一报告期取最早公告（无前视），同日多行取 update_flag=1（已修正）。"""
    df = df.sort_values(["ts_code", "end_date", "f_ann_date_dt", "update_flag"],
                        ascending=[True, True, True, False])
    out = df.drop_duplicates(["ts_code", "end_date"], keep="first")
    print(f"{'':13s} 去重 {len(df):>7,} -> {len(out):>7,}")
    return out


inc = dedup(load_parts("income"))
cf = dedup(load_parts("cashflow"))
bs = dedup(load_parts("balancesheet"))


# ==================== 2. 三表对齐 ====================
fin = (
    inc[["ts_code", "end_date", "f_ann_date_dt", "n_income"]]
    .rename(columns={"f_ann_date_dt": "f_ann_inc"})
    .merge(
        cf[["ts_code", "end_date", "f_ann_date_dt", "n_cashflow_act"]]
        .rename(columns={"f_ann_date_dt": "f_ann_cf"}),
        on=["ts_code", "end_date"], how="inner")
    .merge(
        bs[["ts_code", "end_date", "f_ann_date_dt", "comp_type", "total_assets"]]
        .rename(columns={"f_ann_date_dt": "f_ann_bs"}),
        on=["ts_code", "end_date"], how="inner")
)

if fin.empty:
    raise RuntimeError("三表交集为空：检查三张表是否都拉完、end_date 格式是否一致")

# 三表齐全才算可用 -> 取最晚公告日
fin["f_ann_date_dt"] = fin[["f_ann_inc", "f_ann_cf", "f_ann_bs"]].max(axis=1)
print(f"\n三表交集 {len(fin):,} 行  codes={fin.ts_code.nunique()}")

fin["comp_type"] = fin.comp_type.astype(str)
print("comp_type 分布:", fin.comp_type.value_counts().to_dict())

fin = fin[(fin.comp_type == "1") & (fin.total_assets > 0)].copy()
print(f"剔除金融股后 {len(fin):,} 行  codes={fin.ts_code.nunique()}")
if fin.empty:
    raise RuntimeError("剔除金融股后样本为空：检查 comp_type 是否为字符串")


# ==================== 3. YTD 累计 -> 单季 -> TTM ====================
def to_single(df, col):
    df = df.sort_values(["ts_code", "end_date"]).copy()
    # '20250930' 的月份在 [4:6]
    df["q"] = df.end_date.str[4:6].map({"03": 1, "06": 2, "09": 3, "12": 4})
    bad = df["q"].isna()
    if bad.any():
        print(f"  警告：{bad.sum()} 行 end_date 月份异常，已丢弃: "
              f"{sorted(df.loc[bad, 'end_date'].unique())[:5]}")
        df = df[~bad].copy()
    df["q"] = df["q"].astype(int)

    g = df.groupby("ts_code", sort=False)
    prev = g[col].shift(1)
    prevq = g["q"].shift(1)
    # q=1 时累计值本身就是单季；否则减去上一季，且要求上一季确实相邻
    df[f"{col}_q"] = df[col].where(df.q == 1, df[col] - prev.where(prevq == df.q - 1))
    return df


def add_ttm(df, col):
    df = df.sort_values(["ts_code", "end_date"]).copy()
    df["qnum"] = df.end_date.str[:4].astype(int) * 4 + df.q
    g = df.groupby("ts_code", sort=False)

    ttm = g[f"{col}_q"].rolling(4, min_periods=4).sum()
    df[f"{col}_ttm"] = ttm.reset_index(level=0, drop=True).to_numpy()

    # 断季（上市不足一年 / 缺报）：qnum 跨度不足 3 则置 NaN
    gap = df["qnum"] - g["qnum"].shift(3)
    df.loc[gap.to_numpy() != 3, f"{col}_ttm"] = np.nan
    return df


fin = to_single(fin, "n_income")
fin = to_single(fin, "n_cashflow_act")
fin = add_ttm(fin, "n_income")
fin = add_ttm(fin, "n_cashflow_act")

fin = fin.sort_values(["ts_code", "end_date"])
fin["ta_avg"] = (fin.total_assets + fin.groupby("ts_code").total_assets.shift(4)) / 2

fin["acc_ttm"] = (fin.n_income_ttm - fin.n_cashflow_act_ttm) / fin.ta_avg
fin["factor"] = -fin.acc_ttm           # 低应计 = 高分

print("\nacc_ttm 描述:")
print(fin.acc_ttm.describe(percentiles=[0.01, 0.5, 0.99]).round(4).to_string())
print(f"累计可用样本 {fin.factor.notna().sum():,} 行")


# ==================== 4. 宽表 -> 长表（只保留在册区间） ====================
pan = pd.read_parquet("data/panel/tradable.parquet")
if not isinstance(pan.index, pd.DatetimeIndex):
    pan.index = pd.to_datetime(pan.index.astype(str), format="%Y%m%d")
pan.index.name = "trade_date"
pan = pan[~pan.index.duplicated(keep="first")]

s = pan.stack()                        # MultiIndex: (trade_date, ts_code)
s = s[s.astype(bool)]                  # 先过滤，10.9M -> 1.9M

long = s.reset_index()                 # 3 列：两个索引层 + 值列
long.columns = ["trade_date", "ts_code", "in_universe"]
long = long[["trade_date", "ts_code"]].copy()
long["ts_code"] = long.ts_code.astype(str)

print(f"\n面板 {len(long):,} 行，日均 {long.groupby('trade_date').size().mean():.0f} 只")


# ==================== 5. 按公告日对齐到交易日 ====================
# merge_asof 要求 on 键全局单调递增 -> on 键放在排序第一位
long = long.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
fin_s = (fin.dropna(subset=["factor", "f_ann_date_dt"])
            .sort_values(["f_ann_date_dt", "ts_code"])
            .reset_index(drop=True))

assert long.trade_date.is_monotonic_increasing, "long 未按 trade_date 排序"
assert fin_s.f_ann_date_dt.is_monotonic_increasing, "fin_s 未按公告日排序"

out = pd.merge_asof(
    long,
    fin_s[["ts_code", "f_ann_date_dt", "end_date", "acc_ttm", "factor"]],
    left_on="trade_date", right_on="f_ann_date_dt", by="ts_code",
    direction="backward",
    allow_exact_matches=False,          # 公告当天不出手，T+1 生效
)
print(f"对齐后覆盖率 {out.factor.notna().mean():.1%}")


# ==================== 6. 陈旧报告检查 ====================
out["end_dt"] = pd.to_datetime(out["end_date"], format="%Y%m%d", errors="coerce")
stale = ((out["trade_date"] - out["end_dt"]).dt.days > STALE_DAYS) & out.factor.notna()
n_valid = max(out.factor.notna().sum(), 1)
print(f"陈旧报告置空（报告期距今 >{STALE_DAYS} 天）: {stale.sum():,} 行 "
      f"({stale.sum() / n_valid:.2%} of 有值样本)")
out.loc[stale, "factor"] = np.nan
out = out.drop(columns=["end_dt"])

print(f"陈旧检查后覆盖率 {out.factor.notna().mean():.1%}")


# ==================== 7. 去极值 + 截面标准化 ====================
def mad_winsorize(x, n=3):
    med = x.median()
    mad = (x - med).abs().median() * 1.4826
    if not np.isfinite(mad) or mad == 0:
        return x
    return x.clip(med - n * mad, med + n * mad)


out["factor_w"] = out.groupby("trade_date")["factor"].transform(mad_winsorize)
out["factor_z"] = out.groupby("trade_date")["factor_w"].transform(
    lambda x: (x - x.mean()) / x.std(ddof=0)
)


# ==================== 8. 输出 + 诊断 ====================
res = (out.dropna(subset=["factor_z"])
          [["ts_code", "trade_date", "f_ann_date_dt", "end_date",
            "acc_ttm", "factor", "factor_z"]]
          .rename(columns={"f_ann_date_dt": "f_ann_date"})
          .sort_values(["trade_date", "ts_code"]))

res.to_parquet("data/panel/factor_accruals.parquet", index=False)
print(f"\n已写出 data/panel/factor_accruals.parquet  {res.shape}")

print("\n===== 分年覆盖率 =====")
yr = (out.assign(year=out.trade_date.dt.year)
         .groupby("year")
         .apply(lambda g: pd.Series({
             "覆盖": round(g.factor.notna().mean(), 3),
             "日均只数": int(g.groupby("trade_date").size().mean()),
             "日均有值": int(g.groupby("trade_date").factor.apply(
                 lambda s: s.notna().sum()).mean()),
         })))
print(yr.to_string())

print("\n前 3 行:")
print(res.head(3).to_string())
