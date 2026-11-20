#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
24_diagnose_ab.py
验证 A+B（分数平滑 + 分数差门槛）是否破坏了行业中性。

风险：hysteresis 只看分数、不看行业配额。上一期按配额选出的 50 只，
      5 天后只换 2 只，各行业权重比例可能已经偏离基准——
      而 Stratified 的全部价值就在把最大偏离压在 1.35%。

四个配置同口径对比（指标与 19_diagnose_holdings 完全一致）
  EW Baseline / EW A+B / Strat Baseline / Strat A+B

门槛（预承诺）
  Strat A+B 的最大行业偏离 > 3%  ->  hysteresis 不能单独用，必须加配额约束
  <= 3%                          ->  可以继续，进入冲击成本与行业上限

输出：data/output_diagnose/diag_ab.csv
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_DIR

RAW = DATA_DIR / "raw"
REF = DATA_DIR / "ref"
PANEL = DATA_DIR / "panel"
OUT = DATA_DIR / "output_diagnose"
OUT.mkdir(parents=True, exist_ok=True)

FACTORS = ["rev_5", "turn_20", "vol_20", "ep"]
TOP_N = 50
REBAL = 5
THRESH = 0.25
GATE = 0.03


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

log("② 构建平滑分数...")
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
def select_plain(s, valid):
    return list(s.nlargest(TOP_N).index)


def select_stratified(s, valid, d):
    df = pd.DataFrame({"score": s, "ind": industry.reindex(s.index)}).dropna()
    if len(df) <= TOP_N:
        return list(df.nlargest(TOP_N, "score").index)

    share = df.groupby("ind").size() / len(df)
    raw = share * TOP_N
    base = np.floor(raw).astype(int)
    remain = TOP_N - int(base.sum())
    if remain > 0:
        for key in list((raw - base).sort_values(ascending=False).index)[:remain]:
            base.loc[key] += 1

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


def apply_hysteresis(s, target, held, delta):
    if not target or not held:
        return target
    floor = s.reindex(target).min() - delta
    keep = [c for c in held if c in s.index and s[c] >= floor][:TOP_N]
    fill = [c for c in target if c not in set(keep)][:TOP_N - len(keep)]
    return keep + fill


# ==================== 4. 回放 ====================
def run(score_df, mode, use_thresh):
    w_live = pd.Series(0.0, index=cols)
    held = []
    gross_s = pd.Series(0.0, index=idx)
    to_s = pd.Series(0.0, index=idx)
    records = []

    for i, d in enumerate(idx):
        if i % REBAL == 0:
            valid = mask.loc[d]
            s = score_df.loc[d].where(valid).dropna()
            if len(s) >= TOP_N:
                target = (select_plain(s, valid) if mode == "plain"
                          else select_stratified(s, valid, d))
                picks = (apply_hysteresis(s, target, held, THRESH * float(s.std()))
                         if use_thresh else target)
                picks = list(dict.fromkeys(picks))[:TOP_N]
                w_new = pd.Series(1.0 / len(picks), index=picks).reindex(cols).fillna(0.0)

                tot = w_live.sum()
                w_drift = (w_live / tot) if tot > 0 else w_live
                turnover = float((w_new - w_drift).abs().sum() / 2)
                to_s.loc[d] = turnover

                before = set(w_drift[w_drift > 0].index)
                after = set(picks)

                # 规模分位
                ref = logmv.loc[d].where(valid).dropna()
                pick_pct = ref.rank(pct=True).reindex(picks).dropna()

                # 行业偏离
                pw = (pd.Series(1.0 / len(picks), index=picks)
                        .groupby(industry.reindex(picks)).sum())
                bw = industry.reindex(valid[valid].index).value_counts(normalize=True)
                dev = (pw.reindex(bw.index).fillna(0.0) - bw)

                records.append({
                    "date": d,
                    "turnover": turnover,
                    "n_replaced": len(after - before),
                    "size_pct": pick_pct.mean() * 100,
                    "ind_dev_mean": dev.abs().mean(),
                    "ind_dev_max": dev.abs().max(),
                    "ind_max_name": dev.abs().idxmax(),
                    "ind_max_signed": dev.loc[dev.abs().idxmax()],
                    "n_ind_over_2pct": int((dev.abs() > 0.02).sum()),
                })

                held = picks
                w_live = w_new

        r = fwd.loc[d].fillna(0.0)
        gross_s.loc[d] = float((w_live * r).sum())
        w_live = w_live * (1 + r)

    return pd.DataFrame(records).set_index("date")


CONFIGS = [
    ("EW Baseline",   score_raw,    "plain",      False),
    ("EW A+B",        score_smooth, "plain",      True),
    ("Strat Baseline", score_raw,   "stratified", False),
    ("Strat A+B",     score_smooth, "stratified", True),
]

log("\n③ 回放四个配置...")
recs = {}
for label, sc, mode, th in CONFIGS:
    recs[label] = run(sc, mode, th)
    log(f"   {label:<15} 完成")


# ==================== 5. 汇总 ====================
log("\n" + "=" * 96)
log("【行业中性检查】")
log("=" * 96)

rows = []
for label, rec in recs.items():
    ann_to = rec.turnover.replace(0, np.nan).mean() * (252 / REBAL)
    rows.append({
        "配置": label,
        "年化换手": ann_to,
        "替换/期": rec.n_replaced.mean(),
        "估算持有天数": TOP_N / max(rec.n_replaced.mean(), 0.1) * REBAL,
        "行业偏离均值": rec.ind_dev_mean.mean(),
        "行业偏离最大": rec.ind_dev_max.max(),
        ">2%行业数": rec.n_ind_over_2pct.mean(),
        "规模分位": rec.size_pct.mean(),
    })

tab = pd.DataFrame(rows).set_index("配置")
show = tab.copy()
show["年化换手"] = (show["年化换手"] * 100).round(0).astype(int).astype(str) + "%"
for c in ["行业偏离均值", "行业偏离最大", "规模分位"]:
    show[c] = (show[c] * 100).round(2)
show["替换/期"] = show["替换/期"].round(1)
show["估算持有天数"] = show["估算持有天数"].round(0).astype(int)
show[">2%行业数"] = show[">2%行业数"].round(1)
log(show.to_string())

log("\n【各配置的最强行业暴露（Top5）】")
for label, rec in recs.items():
    log(f"\n  -- {label} --")
    for k, v in rec.ind_max_name.value_counts().head(5).items():
        sub = rec[rec.ind_max_name == k]
        log(f"     {k:<12} {v:>3} 次   平均 {sub.ind_max_signed.mean():+.2%}")

log("\n【分年行业偏离（均值 / 最大）】")
for label, rec in recs.items():
    g = rec.groupby(rec.index.str[:4]).agg(
        dev_mean=("ind_dev_mean", "mean"), dev_max=("ind_dev_max", "max"))
    log(f"\n  -- {label} --")
    log((g * 100).round(2).to_string())


# ==================== 6. 裁决 ====================
strat_ab = recs["Strat A+B"]
worst = strat_ab.ind_dev_max.max()
log("\n" + "=" * 96)
log("【裁决】")
log("=" * 96)
log(f"Strat A+B 最大行业偏离 = {worst:.2%}   门槛 <= {GATE:.0%}")
log(f"Strat A+B 平均行业偏离 = {strat_ab.ind_dev_mean.mean():.2%}")
log(f"Strat A+B 单期替换     = {strat_ab.n_replaced.mean():.1f} 只")
if worst <= GATE:
    log("\n✅ PASS —— hysteresis 没有破坏行业中性，可以进入冲击成本与行业上限")
else:
    log("\n❌ FAIL —— 行业偏离超门槛，hysteresis 必须加配额约束后再用")
log("=" * 96)

tab.to_csv(OUT / "diag_ab.csv", encoding="utf-8-sig")
log(f"\n✅ 输出到 {OUT}/diag_ab.csv")
log(f"总耗时 {(time.time() - t0) / 60:.1f} 分钟")
