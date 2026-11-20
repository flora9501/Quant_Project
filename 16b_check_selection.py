#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""16b_check_selection.py
区分两件事：
  (1) comp4 在 20 日 horizon 上本来就高  -> horizon 效应
  (2) 只在"有近期公告"的子样本上高        -> 选择效应
做法：同一个 comp4，分别在 全样本/子样本、5日/20日 上算 IC。
"""
import glob

import numpy as np
import pandas as pd

F = ["rev_5", "turn_20", "vol_20", "ep"]

# ---------- 1. 四因子复合 ----------
fac = pd.read_parquet("data/panel/factors.parquet",
                      columns=["trade_date", "ts_code"] + F)
for c in F:
    fac[c + "_z"] = fac.groupby("trade_date")[c].transform(
        lambda x: (x - x.mean()) / x.std(ddof=0))
fac["comp4"] = fac[[c + "_z" for c in F]].mean(axis=1)

pead = pd.read_parquet("data/panel/factor_pead.parquet",
                       columns=["trade_date", "ts_code", "factor_z"])
pead["trade_date"] = pd.to_datetime(pead.trade_date).dt.strftime("%Y%m%d")

fac = (fac.drop_duplicates(["trade_date", "ts_code"])
          .merge(pead.rename(columns={"factor_z": "sue"})
                     .drop_duplicates(["trade_date", "ts_code"]),
                 on=["trade_date", "ts_code"], how="left"))
fac["has_sue"] = fac.sue.notna()
print(f"全样本 {len(fac):,} | 有 SUE {fac.has_sue.sum():,} ({fac.has_sue.mean():.1%})")


# ---------- 2. 收益 ----------
frames = [pd.read_parquet(f, columns=["ts_code", "trade_date", "pct_chg"])
          for f in sorted(glob.glob("data/raw/daily_[0-9][0-9][0-9][0-9].parquet"))]
daily = pd.concat(frames, ignore_index=True).drop_duplicates(["trade_date", "ts_code"])
daily = daily[daily.ts_code.str.endswith((".SH", ".SZ"))]

px = (daily.pivot(index="trade_date", columns="ts_code", values="pct_chg")
           .reindex(index=np.sort(daily.trade_date.unique()),
                    columns=np.sort(daily.ts_code.unique())) / 100.0).fillna(0.0)
px = (1 + px).cumprod()


def daily_rank_corr(frame, x, y, min_n=50):
    t = frame[["trade_date", x, y]].dropna()
    if t.empty:
        return pd.Series(dtype=float)
    t = t.assign(rx=t.groupby("trade_date")[x].rank(),
                 ry=t.groupby("trade_date")[y].rank())
    return (t.groupby("trade_date")
             .apply(lambda z: z.rx.corr(z.ry) if len(z) >= min_n else np.nan)
             .dropna())


# ---------- 3. 五个口径 ----------
rows = []
for h in (5, 20):
    f = px.shift(-h) / px - 1.0
    fl = f.stack().rename("fwd").reset_index()
    fl.columns = ["trade_date", "ts_code", "fwd"]
    lo, hi = fl.fwd.quantile([0.01, 0.99])
    fl["fwd"] = fl.fwd.clip(lo, hi)

    d = fac.merge(fl, on=["trade_date", "ts_code"], how="left")
    d["year"] = d.trade_date.str[:4]

    for tag, sub in [("全样本", d), ("有SUE子样本", d[d.has_sue])]:
        for yr in ["ALL"] + sorted(d.year.unique()):
            s = sub if yr == "ALL" else sub[sub.year == yr]
            r = daily_rank_corr(s, "comp4", "fwd")
            if len(r) == 0:
                continue
            rows.append({"horizon": f"fwd_{h}", "样本": tag, "year": yr,
                         "IC": round(r.mean(), 4),
                         "ICIR": round(r.mean() / r.std(), 3),
                         "days": len(r)})


# ---------- 4. 只打关键对比 ----------
t = pd.DataFrame(rows)
print("\n===== 全样本 vs 有SUE子样本（comp4） =====")
key = t[(t.year == "ALL")][["horizon", "样本", "IC", "ICIR", "days"]]
piv = key.pivot(index="horizon", columns="样本", values="IC")
print(piv.to_string())
print()
print(key.to_string(index=False))

print("\n===== 分年（只看 20 日） =====")
print(t[(t.horizon == "fwd_20") & (t.year != "ALL")].to_string(index=False))
