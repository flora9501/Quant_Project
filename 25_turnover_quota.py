#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
25_turnover_quota.py
把 hysteresis 约束在行业配额内：每个行业单独设分数门槛，不跨行业保留。

问题（出自 24_diagnose_ab）
  无约束 hysteresis 让 Strat 的最大行业偏离从 1.35% 涨到 15.80%，
  组合漂移成化工原料 / 元器件 / 专用机械 的行业动量组合。

做法
  1) 先按基准占比算出每个行业的配额 k（与 select_stratified 完全一致）
  2) 行业内取第 k 名的分数作为 cut，门槛 floor = cut - Δ
  3) 已持有的本行业股票，分数 >= floor 就保留，不足 k 个再用最高分补
  -> 配额严格守恒，行业偏离回到基线水平；同时仍能减少替换

预承诺
  Δ = 0.25σ 不扫描。通过条件：
    最大行业偏离 <= 3%  且  net@30bp >= Baseline + 1.5pp

输出：data/output_turnover/quota_compare.csv
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_DIR

RAW = DATA_DIR / "raw"
REF = DATA_DIR / "ref"
PANEL = DATA_DIR / "panel"
OUT = DATA_DIR / "output_turnover"
OUT.mkdir(parents=True, exist_ok=True)

FACTORS = ["rev_5", "turn_20", "vol_20", "ep"]
TOP_N = 50
REBAL = 5
COSTS_BP = [12, 20, 30]
COSTS = [bp / 1e4 for bp in COSTS_BP]
THRESH = 0.25


def log(m):
    print(m, flush=True)


t0 = time.time()

# ==================== 1. 数据 ====================
log("① 读取数据...")
frames = [pd.read_parquet(f, columns=["ts_code", "trade_date", "pct_chg"])
          for f in sorted(RAW.glob("daily_[0-9][0-9][0-9][0-9].parquet"))]
daily = pd.concat(frames, ignore_index=True).drop_duplicates(["trade_date", "ts_code"])
ret = (daily.pivot(index="trade_date", columns="ts_code", values="pct_chg") / 100.0).sort_index()
del daily
fwd = ret.shift(-1)

fac = pd.read_parquet(PANEL / "factors_neutral.parquet")
wide = {n: fac.pivot(index="trade_date", columns="ts_code", values=n) for n in FACTORS}
idx = ret.index.intersection(wide["rev_5"].index).sort_values()
cols = ret.columns
for n in FACTORS:
    wide[n] = wide[n].reindex(index=idx, columns=cols)

b = [pd.read_parquet(f, columns=["ts_code", "trade_date", "circ_mv"])
     for f in sorted(RAW.glob("daily_basic_*.parquet"))]
bb = pd.concat(b, ignore_index=True).drop_duplicates(["trade_date", "ts_code"])
logmv = np.log(bb.pivot(index="trade_date", columns="ts_code", values="circ_mv")
                 .reindex(index=idx, columns=cols) * 1e4)

mask = (pd.read_parquet(PANEL / "tradable.parquet")
        .reindex(index=idx, columns=cols).fillna(False).astype(bool))
fwd = fwd.reindex(index=idx, columns=cols)

sb = pd.concat([pd.read_parquet(REF / "stock_basic.parquet"),
                pd.read_parquet(REF / "stock_delisted.parquet")],
               ignore_index=True).drop_duplicates("ts_code")
industry = sb.set_index("ts_code")["industry"].reindex(cols).fillna("UNKNOWN")
log(f"   {len(idx)} 天 × {len(cols)} 只")


# ==================== 2. 分数 ====================
def combine(dfs):
    return sum(d.fillna(0.0) for d in dfs).where(
        sum(d.notna().astype(int) for d in dfs) > 0)


score_raw = combine([wide[n] for n in FACTORS])

log("② 平滑分数...")
acc = score_raw.fillna(0.0).copy()
cnt = score_raw.notna().astype(np.int16).copy()
for k in (REBAL, 2 * REBAL):
    sh = score_raw.shift(k)
    acc = acc + sh.fillna(0.0)
    cnt = cnt + sh.notna().astype(np.int16)
    del sh
score_smooth = acc / cnt.replace(0, np.nan)
del acc, cnt


# ==================== 3. 选股 ====================
def industry_quota(df, top_n=TOP_N):
    share = df.groupby("ind").size() / len(df)
    raw = share * top_n
    base = np.floor(raw).astype(int)
    remain = top_n - int(base.sum())
    if remain > 0:
        for key in list((raw - base).sort_values(ascending=False).index)[:remain]:
            base.loc[key] += 1
    return base


def select_stratified(s, d):
    df = pd.DataFrame({"score": s, "ind": industry.reindex(s.index)}).dropna()
    if len(df) <= TOP_N:
        return list(df.nlargest(TOP_N, "score").index)

    base = industry_quota(df)
    picks = []
    for ind_, k in base.items():
        if k <= 0:
            continue
        picks.extend(df[df["ind"] == ind_].nlargest(int(k), "score").index.tolist())
    picks = list(dict.fromkeys(picks))[:TOP_N]
    if len(picks) < TOP_N:
        rest = df[~df.index.isin(set(picks))].nlargest(TOP_N - len(picks), "score")
        picks += rest.index.tolist()
    return list(dict.fromkeys(picks))[:TOP_N]


def select_strat_hysteresis(s, d, held, sigma_mult=THRESH):
    """配额守恒的 hysteresis：每个行业单独设门槛。"""
    df = pd.DataFrame({"score": s, "ind": industry.reindex(s.index)}).dropna()
    if df.empty:
        return []
    if len(df) <= TOP_N:
        return list(df.nlargest(TOP_N, "score").index)

    base = industry_quota(df)
    delta = sigma_mult * float(df["score"].std())
    picks = []

    for ind_, k in base.items():
        if k <= 0:
            continue
        pool = df[df["ind"] == ind_].sort_values("score", ascending=False)
        if pool.empty:
            continue

        cut = float(pool["score"].iloc[min(int(k), len(pool)) - 1])
        floor = cut - delta

        held_in = [c for c in held if c in pool.index]
        keep = [c for c in held_in if float(pool.loc[c, "score"]) >= floor]
        keep = sorted(keep, key=lambda c: -float(pool.loc[c, "score"]))[:int(k)]
        fill = [c for c in pool.index if c not in set(keep)][:int(k) - len(keep)]
        picks.extend(keep + fill)

    picks = list(dict.fromkeys(picks))[:TOP_N]
    if len(picks) < TOP_N:
        rest = df[~df.index.isin(set(picks))].nlargest(TOP_N - len(picks), "score")
        picks += rest.index.tolist()
    return list(dict.fromkeys(picks))[:TOP_N]


# ==================== 4. 回放 ====================
def run(score_df, method):
    w_live = pd.Series(0.0, index=cols)
    held = []
    gross_s = pd.Series(0.0, index=idx)
    to_s = pd.Series(0.0, index=idx)
    rec = []

    for i, d in enumerate(idx):
        if i % REBAL == 0:
            valid = mask.loc[d]
            s = score_df.loc[d].where(valid).dropna()
            if len(s) >= TOP_N:
                if method == "baseline":
                    picks = select_stratified(s, d)
                elif method == "hyst":
                    floor = s.reindex(select_stratified(s, d)).min() - THRESH * float(s.std())
                    keep = [c for c in held if c in s.index and s[c] >= floor][:TOP_N]
                    fill = [c for c in select_stratified(s, d) if c not in set(keep)][:TOP_N - len(keep)]
                    picks = keep + fill
                else:
                    picks = select_strat_hysteresis(s, d, held)

                picks = list(dict.fromkeys(picks))[:TOP_N]
                w_new = pd.Series(1.0 / len(picks), index=picks).reindex(cols).fillna(0.0)

                tot = w_live.sum()
                w_drift = (w_live / tot) if tot > 0 else w_live
                to_s.loc[d] = float((w_new - w_drift).abs().sum() / 2)

                before = set(w_drift[w_drift > 0].index)
                pw = (pd.Series(1.0 / len(picks), index=picks)
                        .groupby(industry.reindex(picks)).sum())
                bw = industry.reindex(valid[valid].index).value_counts(normalize=True)
                dev = pw.reindex(bw.index).fillna(0.0) - bw

                ref = logmv.loc[d].where(valid).dropna()
                rec.append({
                    "date": d,
                    "n_replaced": len(set(picks) - before),
                    "size_pct": ref.rank(pct=True).reindex(picks).dropna().mean() * 100,
                    "dev_mean": dev.abs().mean(),
                    "dev_max": dev.abs().max(),
                    "n_over2": int((dev.abs() > 0.02).sum()),
                })
                held = picks
                w_live = w_new

        r = fwd.loc[d].fillna(0.0)
        gross_s.loc[d] = float((w_live * r).sum())
        w_live = w_live * (1 + r)

    return pd.DataFrame(rec).set_index("date"), gross_s, to_s


def metrics(rec, gross, to, bench_nav):
    out = {"年化换手": to.replace(0, np.nan).mean() * (252 / REBAL) * 100,
           "替换/期": rec.n_replaced.mean(),
           "持有天数": TOP_N / max(rec.n_replaced.mean(), 0.1) * REBAL,
           "偏离均值": rec.dev_mean.mean() * 100,
           "偏离最大": rec.dev_max.max() * 100,
           ">2%行业": rec.n_over2.mean(),
           "规模分位": rec.size_pct.mean()}
    for bp, c in zip(COSTS_BP, COSTS):
        net = (gross - to * c).iloc[:-1]
        nav = (1 + net).cumprod()
        ann = nav.iloc[-1] ** (252 / len(net)) - 1
        ex_nav = nav / bench_nav.reindex(nav.index).ffill()
        ex_ann = ex_nav.iloc[-1] ** (252 / len(ex_nav)) - 1
        out[f"net@{bp}"] = ann * 100
        out[f"ex@{bp}"] = ex_ann * 100
        if bp == 30:
            ex_ret = ex_nav.pct_change().dropna()
            out["IR@30"] = ex_ann / (ex_ret.std() * np.sqrt(252))
    return out


bench = fwd.where(mask).mean(axis=1).reindex(idx).fillna(0.0)
bench_nav = (1 + bench).cumprod()

log("\n③ 回放三个配置（全部 Stratified 构造）...")
rows = []
for label, sc, method in [("Baseline      ", score_raw, "baseline"),
                          ("A+B 无配额   ", score_smooth, "hyst"),
                          ("A+B 配额约束 ", score_smooth, "quota")]:
    rec, g, t = run(sc, method)
    rows.append({"配置": label, **metrics(rec, g, t, bench_nav)})
    log(f"   {label} 完成")

tab = pd.DataFrame(rows).set_index("配置")

log("\n" + "=" * 96)
log("【结果】")
log("=" * 96)
log(tab.round(2).to_string())

log("\n【相对 Baseline 的增量】")
base = tab.loc["Baseline      "]
for lbl in ["A+B 无配额   ", "A+B 配额约束 "]:
    r = tab.loc[lbl]
    log(f"   {lbl.strip():<12} 换手 {r['年化换手'] - base['年化换手']:+.0f}pp   "
        f"偏离最大 {r['偏离最大'] - base['偏离最大']:+.2f}pp   "
        f"net@30 {r['net@30'] - base['net@30']:+.2f}pp   "
        f"ex@30 {r['ex@30'] - base['ex@30']:+.2f}pp")

# ==================== 5. 裁决 ====================
q = tab.loc["A+B 配额约束 "]
ok_dev = q["偏离最大"] <= 3.0
ok_net = q["net@30"] >= base["net@30"] + 1.5

log("\n" + "=" * 96)
log("【裁决】")
log("=" * 96)
log(f"最大行业偏离 {q['偏离最大']:.2f}%   门槛 <= 3.00%      {'✅' if ok_dev else '❌'}")
log(f"net@30bp 增量 {q['net@30'] - base['net@30']:+.2f}pp  门槛 >= +1.50pp   {'✅' if ok_net else '❌'}")
log("\n" + ("✅ PASS —— 配额约束版可作为 Phase 3 的候选构造"
            if (ok_dev and ok_net) else
            "❌ FAIL —— 需要缩小 Δ 或改用其他换手控制手段"))
log("=" * 96)

tab.to_csv(OUT / "quota_compare.csv", encoding="utf-8-sig")
log(f"\n✅ 输出到 {OUT}/quota_compare.csv")
log(f"总耗时 {(time.time() - t0) / 60:.1f} 分钟")
