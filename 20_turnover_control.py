#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
20_turnover_control.py  (v2)
降低换手的两个方案对照实验。只改组合构建，不动因子。

背景（来自 19_diagnose_holdings）
  - 每 5 个交易日换掉 23.8/50 只；换手 98% 来自名单更换，权重漂移只占 2%
  - 12bp 下超额 4.38%；30bp 下 Stratified 已跑不过基准
  - 因此降换手是 Phase 3 唯一有量级意义的方向

两个方案
  A 分数平滑：score_t = mean(score_t, score_{t-5}, score_{t-10})
  B 分数差门槛：持有股分数 >= 目标集最弱成员 - Δ 才保留，Δ = 0.25 × 截面 std

预承诺
  方法选择只看 Equal Weight 构造；选定后套到 Stratified，不回头改。
  判据用 30bp 单边成本下的年化净收益。

v2 修正
  - COSTS 用整数 bp 定义，标签直接拼整数，不再 int(浮点) —— 0.0012*1e4 截断成 11
  - 替换只数统计跳过第一次建仓（否则 50 只全部算"替换"，抬高均值）
  - 最优方案名 strip 后再拼 Stratified 的标签
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
log(f"   {len(idx)} 天 × {len(cols)} 只，行业 {industry.nunique()} 个")


# ==================== 2. 复合分 ====================
def combine(dfs):
    return sum(d.fillna(0.0) for d in dfs).where(
        sum(d.notna().astype(int) for d in dfs) > 0)


score_raw = combine([wide[n] for n in FACTORS])

log("② 构建平滑分数（3 期等权平均）...")
acc = score_raw.fillna(0.0).copy()
cnt = score_raw.notna().astype(np.int16).copy()
for k in (REBAL, 2 * REBAL):
    sh = score_raw.shift(k)
    acc = acc + sh.fillna(0.0)
    cnt = cnt + sh.notna().astype(np.int16)
    del sh
score_smooth = acc / cnt.replace(0, np.nan)
del acc, cnt
log("   完成")


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
    """持有股分数 >= 目标集最弱成员 - delta 就保留，其余从目标集补齐。"""
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
    n_rep = []

    for i, d in enumerate(idx):
        if i % REBAL == 0:
            valid = mask.loc[d]
            s = score_df.loc[d].where(valid).dropna()
            if len(s) >= TOP_N:
                target = (select_plain(s, valid) if mode == "plain"
                          else select_stratified(s, valid, d))

                if use_thresh:
                    delta = THRESH * float(s.std())
                    picks = apply_hysteresis(s, target, held, delta)
                else:
                    picks = target

                picks = list(dict.fromkeys(picks))[:TOP_N]
                w_new = pd.Series(1.0 / len(picks), index=picks).reindex(cols).fillna(0.0)

                tot = w_live.sum()
                w_drift = (w_live / tot) if tot > 0 else w_live
                to_s.loc[d] = float((w_new - w_drift).abs().sum() / 2)

                if held:                       # 跳过第一次建仓
                    n_rep.append(len(set(picks) - set(held)))

                held = picks
                w_live = w_new

        r = fwd.loc[d].fillna(0.0)
        gross_s.loc[d] = float((w_live * r).sum())
        w_live = w_live * (1 + r)

    return gross_s, to_s, (np.mean(n_rep) if n_rep else np.nan)


def metrics(gross, to, bench):
    bench_nav = (1 + bench).cumprod()
    out = {"年化换手": to.mean() * (252 / REBAL)}
    for bp, c in zip(COSTS_BP, COSTS):
        net = (gross - to * c).iloc[:-1]
        nav = (1 + net).cumprod()
        ann = nav.iloc[-1] ** (252 / len(net)) - 1
        ex_nav = nav / bench_nav.reindex(nav.index).ffill()
        ex_ann = ex_nav.iloc[-1] ** (252 / len(ex_nav)) - 1
        ex_ret = ex_nav.pct_change().dropna()
        out[f"net@{bp}bp"] = ann
        out[f"ex@{bp}bp"] = ex_ann
        out[f"IR@{bp}bp"] = ex_ann / (ex_ret.std() * np.sqrt(252))
    return out


bench = fwd.where(mask).mean(axis=1).reindex(idx).fillna(0.0)

CONFIGS = [
    ("Baseline", score_raw,    False),
    ("A 平滑",   score_smooth, False),
    ("B 门槛",   score_raw,    True),
    ("A+B",      score_smooth, True),
]

log("\n③ 回放 Equal Weight 构造（用于选方法）...")
rows_ew = []
for name, sc, th in CONFIGS:
    g, t, rep = run(sc, "plain", th)
    m = metrics(g, t, bench)
    m["替换只数"] = rep
    rows_ew.append({"方案": name, **m})
    log(f"   {name:<9} 完成   换手 {m['年化换手']:.0%}   替换 {rep:.1f} 只")

tab_ew = pd.DataFrame(rows_ew).set_index("方案")

best = tab_ew["net@30bp"].idxmax()
log(f"\n>>> 30bp 下最优：{best}（净收益 {tab_ew.loc[best, 'net@30bp']:.2%}，"
    f"超额 {tab_ew.loc[best, 'ex@30bp']:.2%}）")
log(">>> 按预承诺，只把这个方法套到 Stratified，不再比较其他方案")

best_smooth = ("平滑" in best) or ("A+B" in best)
best_thresh = ("门槛" in best) or ("A+B" in best)

log("\n④ 回放 Stratified 构造...")
rows_st = []
for label, sc, th in [("Baseline", score_raw, False),
                      (f"{best}→Strat", score_smooth if best_smooth else score_raw, best_thresh)]:
    g, t, rep = run(sc, "stratified", th)
    m = metrics(g, t, bench)
    m["替换只数"] = rep
    rows_st.append({"方案": label, **m})
    log(f"   {label:<12} 完成   换手 {m['年化换手']:.0%}   替换 {rep:.1f} 只")

tab_st = pd.DataFrame(rows_st).set_index("方案")


# ==================== 5. 汇总 ====================
def show(tab, label):
    log("\n" + "=" * 82)
    log(f"【{label}】")
    log("=" * 82)
    cols_show = ["年化换手", "替换只数"] + \
                [f"net@{bp}bp" for bp in COSTS_BP] + \
                [f"ex@{bp}bp" for bp in COSTS_BP] + \
                ["IR@12bp", "IR@30bp"]
    log(tab[cols_show].round(4).to_string())

    base = tab.iloc[0]
    log("\n相对 Baseline 的增量：")
    for i in range(1, len(tab)):
        r = tab.iloc[i]
        log(f"   {tab.index[i]:<12} 换手 {r['年化换手'] - base['年化换手']:+.0%}   "
            f"净收益@30bp {r['net@30bp'] - base['net@30bp']:+.2%}   "
            f"超额@30bp {r['ex@30bp'] - base['ex@30bp']:+.2%}")


show(tab_ew, "Equal Weight 构造（方法选择）")
show(tab_st, "Stratified 构造（套用选中方法）")

tab_ew.to_csv(OUT / "turnover_ew.csv", encoding="utf-8-sig")
tab_st.to_csv(OUT / "turnover_stratified.csv", encoding="utf-8-sig")

log(f"\n✅ 输出到 {OUT}/")
log(f"总耗时 {(time.time() - t0) / 60:.1f} 分钟")
