#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
19_diagnose_holdings.py
量化当前 Top50 组合的特征，为 Phase 3 的约束设计提供基线。
不改因子、不调参、不重算因子值——只读现成的 factors_neutral + 行情。

四个诊断
  1) 规模分位：Top50 在全池 ln_mv 排名中的位置
  2) 行业偏离：均值/最大值的来源，哪些行业在系统性超配
  3) 换手拆解：新进榜 vs 权重回归，各占多少
  4) 成本敏感度：12 / 20 / 30 bp 下的净收益

输出：data/output_diagnose/*.csv + 控制台汇总
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
COSTS = [0.0012, 0.0020, 0.0030]


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
name_map = sb.set_index("ts_code")["name"] if "name" in sb.columns else None

log(f"   {len(idx)} 天 × {len(cols)} 只，行业 {industry.nunique()} 个")


# ==================== 2. 复合分（与 07 完全一致） ====================
def combine(dfs):
    parts = [d.fillna(0.0) for d in dfs]
    out = sum(parts)
    valid = sum(d.notna().astype(int) for d in dfs) > 0
    return out.where(valid)


score = combine([wide[n] for n in FACTORS])


# ==================== 3. 选股 ====================
def select_plain(s, valid):
    return list(s.nlargest(TOP_N).index)


def select_stratified(s, valid, d):
    lm = logmv.loc[d]
    df = pd.DataFrame({"score": s,
                       "ind": industry.reindex(s.index)}).dropna()
    if len(df) <= TOP_N:
        return list(df.nlargest(TOP_N, "score").index)

    share = df.groupby("ind").size() / len(df)
    raw = share * TOP_N
    base = np.floor(raw).astype(int)
    remain = TOP_N - int(base.sum())
    if remain > 0:
        order = (raw - base).sort_values(ascending=False).index
        for key in list(order)[:remain]:
            base.loc[key] += 1

    picks = []
    for ind_, k in base.items():
        if k <= 0:
            continue
        sub = df[df["ind"] == ind_].nlargest(int(k), "score")
        picks.extend(sub.index.tolist())

    picks = list(dict.fromkeys(picks))[:TOP_N]
    if len(picks) < TOP_N:
        taken = set(picks)
        rest = df[~df.index.isin(taken)].nlargest(TOP_N - len(picks), "score")
        picks += rest.index.tolist()
    return list(dict.fromkeys(picks))[:TOP_N]


# ==================== 4. 回放 + 诊断 ====================
def replay(mode):
    """mode: 'plain' / 'stratified'"""
    w_live = pd.Series(0.0, index=cols)     # 未归一化的市值权重
    records, holdings = [], {}
    gross_s = pd.Series(0.0, index=idx)
    to_s = pd.Series(0.0, index=idx)

    for i, d in enumerate(idx):
        # --- 先在重平衡日结算换手 ---
        if i % REBAL == 0:
            valid = mask.loc[d]
            s = score.loc[d].where(valid).dropna()
            if len(s) >= TOP_N:
                picks = select_plain(s, valid) if mode == "plain" else select_stratified(s, valid, d)
                w_new = pd.Series(1.0 / len(picks), index=picks).reindex(cols).fillna(0.0)

                # 漂移到今天的权重（归一化）
                tot = w_live.sum()
                w_drift = (w_live / tot) if tot > 0 else w_live

                held_before = set(w_drift[w_drift > 0].index)
                held_after = set(picks)

                entry = w_new[~w_new.index.isin(held_before)].sum()
                exit_ = w_drift[~w_drift.index.isin(held_after)].sum()
                topup = (w_new - w_drift).clip(lower=0).sum() - entry
                trim = (w_drift - w_new).clip(lower=0).sum() - exit_

                turnover = (w_new - w_drift).abs().sum() / 2

                # --- 诊断：规模分位 ---
                ref = logmv.loc[d].where(valid).dropna()
                pct_rank = ref.rank(pct=True)
                pick_pct = pct_rank.reindex(picks).dropna()

                # --- 诊断：行业 ---
                pw = (pd.Series(1.0 / len(picks), index=picks)
                        .groupby(industry.reindex(picks)).sum())
                bw = industry.reindex(valid[valid].index).value_counts(normalize=True)
                dev = (pw.reindex(bw.index).fillna(0.0) - bw).sort_values()

                records.append({
                    "date": d,
                    "turnover": turnover,
                    "entry": entry, "exit": exit_,
                    "topup": max(topup, 0.0), "trim": max(trim, 0.0),
                    "n_replaced": len(held_after - held_before),
                    "n_held_before": len(held_before),
                    "size_pct_mean": pick_pct.mean() * 100,
                    "size_pct_median": pick_pct.median() * 100,
                    "size_pct_min": pick_pct.min() * 100,
                    "size_pct_max": pick_pct.max() * 100,
                    "ind_dev_mean": dev.abs().mean(),
                    "ind_dev_max": dev.abs().max(),
                    "ind_max_name": dev.abs().idxmax(),
                    "ind_max_signed": dev.loc[dev.abs().idxmax()],
                    "n_ind_over_2pct": int((dev.abs() > 0.02).sum()),
                })
                holdings[d] = picks
                w_live = w_new
                to_s.loc[d] = turnover

        # --- 当日收益 ---
        r = fwd.loc[d].fillna(0.0)
        gross_s.loc[d] = float((w_live * r).sum())
        w_live = w_live * (1 + r)

    return pd.DataFrame(records).set_index("date"), gross_s, to_s, holdings


log("② 回放 Equal Weight...")
rec_ew, gross_ew, to_ew, hold_ew = replay("plain")
log("③ 回放 Stratified...")
rec_st, gross_st, to_st, hold_st = replay("stratified")


# ==================== 5. 汇总 ====================
def summarize(rec, gross, to, label):
    log("\n" + "=" * 62)
    log(f"【{label}】")
    log("=" * 62)

    log("\n-- 换手拆解（每 5 日一次重平衡）--")
    log(f"   总换手 / 次        : {rec.turnover.mean():7.2%}")
    log(f"   其中 新进榜        : {rec.entry.mean():7.2%}  ({rec.entry.mean()/rec.turnover.mean():.0%})")
    log(f"   其中 权重回归      : {rec.topup.mean():7.2%}  ({rec.topup.mean()/rec.turnover.mean():.0%})")
    log(f"   平均替换只数       : {rec.n_replaced.mean():7.1f} / {TOP_N}")
    log(f"   年化换手           : {rec.turnover.mean()*252/REBAL:7.0%}")

    log("\n-- 规模分位（100 = 全池最大）--")
    log(f"   均值 {rec.size_pct_mean.mean():5.1f}   中位 {rec.size_pct_median.mean():5.1f}   "
        f"区间 [{rec.size_pct_min.mean():.1f}, {rec.size_pct_max.mean():.1f}]   （基准 50）")

    log("\n-- 行业偏离 --")
    log(f"   均值 |dev|  {rec.ind_dev_mean.mean():6.2%}   最大 |dev|  {rec.ind_dev_max.max():6.2%}")
    log(f"   |dev| > 2% 的行业数（平均）: {rec.n_ind_over_2pct.mean():.1f}")
    log("   最常出现最大偏离的行业 Top5:")
    for k, v in rec.ind_max_name.value_counts().head(5).items():
        sub = rec[rec.ind_max_name == k]
        log(f"     {k:<14} 出现 {v:>3} 次   平均偏离 {sub.ind_max_signed.mean():+.2%}")

    log("\n-- 年度规模分位 / 换手 --")
    tab = pd.DataFrame({
        "size_pct": rec.size_pct_mean.groupby(rec.index.str[:4]).mean(),
        "turnover": rec.turnover.groupby(rec.index.str[:4]).mean(),
        "ind_dev": rec.ind_dev_mean.groupby(rec.index.str[:4]).mean(),
    })
    log(tab.round(3).to_string())

    log("\n-- 成本敏感度（年化净收益）--")
    rows = []
    for c in COSTS:
        net = gross - to * c
        nav = (1 + net.iloc[:-1]).cumprod()
        ann = nav.iloc[-1] ** (252 / len(net.iloc[:-1])) - 1
        rows.append({"单边成本": f"{c:.2%}", "年化净收益": ann,
                     "成本拖累": (to.mean() * c * 252)})
    log(pd.DataFrame(rows).to_string(index=False))
    return tab


tab_ew = summarize(rec_ew, gross_ew, to_ew, "Equal Weight（Top50 等权）")
tab_st = summarize(rec_st, gross_st, to_st, "EW + Stratified（行业配额）")


# ==================== 6. 落盘 ====================
rec_ew.to_csv(OUT / "diag_equal_weight.csv", encoding="utf-8-sig")
rec_st.to_csv(OUT / "diag_stratified.csv", encoding="utf-8-sig")
tab_ew.to_csv(OUT / "yearly_ew.csv", encoding="utf-8-sig")
tab_st.to_csv(OUT / "yearly_stratified.csv", encoding="utf-8-sig")

pd.DataFrame(hold_ew).to_parquet(OUT / "holdings_ew.parquet")
log(f"\n✅ 输出到 {OUT}/")
log(f"总耗时 {(time.time() - t0) / 60:.1f} 分钟")
