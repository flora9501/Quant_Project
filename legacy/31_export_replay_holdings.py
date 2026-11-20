#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
31_export_replay_holdings.py  (v2)
把全历史回测到最后一个调仓日的持仓导出，并据此写 paper/state.json。

用途
  1) 给 daily_run.py 播种，保证 hysteresis 有历史可依
  2) 提供"标准答案"，用于验证 daily_run 的选股逻辑

v2 修改
  ★ 改为使用 3 期平滑分数（与 28 的 "Strat + A+B" 配置一致）
    此前用原始分数，导致与 daily_run 的交集只有 36/50

输出
  paper/replay_holdings.parquet   所有期持仓（date, ts_code）
  paper/state.json                倒数第二期持仓，供 daily_run 继续
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_DIR

RAW = DATA_DIR / "raw"
REF = DATA_DIR / "ref"
PANEL = DATA_DIR / "panel"
PAPER = DATA_DIR / "paper"
PAPER.mkdir(parents=True, exist_ok=True)

FACTORS = ["rev_5", "turn_20", "vol_20", "ep"]
TOP_N, REBAL, THRESH = 50, 5, 0.25


def log(m):
    print(m, flush=True)


# ==================== 数据 ====================
fac = pd.read_parquet(PANEL / "factors_neutral.parquet")
wide = {n: fac.pivot(index="trade_date", columns="ts_code", values=n) for n in FACTORS}

idx = sorted(wide["rev_5"].index)
cols = wide["rev_5"].columns
for n in FACTORS:
    wide[n] = wide[n].reindex(index=idx, columns=cols)

tr = pd.read_parquet(PANEL / "tradable.parquet")
if isinstance(tr.index, pd.DatetimeIndex):
    tr.index = tr.index.strftime("%Y%m%d")
mask = tr.reindex(index=idx, columns=cols).fillna(False).astype(bool)

sb = pd.concat([pd.read_parquet(REF / "stock_basic.parquet"),
                pd.read_parquet(REF / "stock_delisted.parquet")],
               ignore_index=True).drop_duplicates("ts_code")
industry = sb.set_index("ts_code")["industry"].reindex(cols).fillna("UNKNOWN")

log(f"{len(idx)} 天 × {len(cols)} 只")
log(f"  日均在册 {mask.sum(axis=1).mean():.0f} 只")


# ==================== 分数（含平滑） ====================
score_raw = sum(wide[n].fillna(0) for n in FACTORS) / len(FACTORS)


def smooth_score(s):
    """3 期等权平均：mean(score_t, score_{t-5}, score_{t-10})"""
    acc = s.fillna(0.0).copy()
    cnt = s.notna().astype(int).copy()
    for k in (REBAL, 2 * REBAL):
        sh = s.shift(k)
        acc = acc + sh.fillna(0.0)
        cnt = cnt + sh.notna().astype(int)
    return acc / cnt.replace(0, np.nan)


score = smooth_score(score_raw)
log(f"分数完成；平滑后有效天数 {int(score.notna().any(axis=1).sum())}")


# ==================== 选股 ====================
def select(s, universe, held, sigma_mult=THRESH):
    s = s.reindex(universe).dropna()
    if len(s) < TOP_N:
        return []

    df = pd.DataFrame({"score": s,
                       "ind": industry.reindex(s.index).fillna("UNKNOWN")}).dropna()
    if df.empty:
        return []

    share = df.groupby("ind").size() / len(df)
    raw = share * TOP_N
    base = np.floor(raw).astype(int)
    remain = TOP_N - int(base.sum())
    if remain > 0:
        for key in list((raw - base).sort_values(ascending=False).index)[:remain]:
            base.loc[key] += 1

    delta = sigma_mult * float(df["score"].std())
    picks = []
    for ind_, k in base.items():
        if k <= 0:
            continue
        pool = df[df["ind"] == ind_].sort_values("score", ascending=False)
        if pool.empty:
            continue
        cut = float(pool["score"].iloc[min(int(k), len(pool)) - 1])
        kept = [c for c in held if c in pool.index and float(pool.loc[c, "score"]) >= cut - delta]
        kept = sorted(kept, key=lambda c: -float(pool.loc[c, "score"]))[:int(k)]
        fill = [c for c in pool.index if c not in set(kept)][:int(k) - len(kept)]
        picks.extend(kept + fill)

    picks = list(dict.fromkeys(picks))[:TOP_N]
    if len(picks) < TOP_N:
        rest = df[~df.index.isin(set(picks))].nlargest(TOP_N - len(picks), "score")
        picks += rest.index.tolist()
    return list(dict.fromkeys(picks))[:TOP_N]


# ==================== 回放 ====================
log("回放...")
held = []
records, history = [], []

for i, d in enumerate(idx):
    if i % REBAL != 0:
        continue
    universe = mask.loc[d]
    universe = universe[universe].index.tolist()
    s = score.loc[d].where(mask.loc[d]).dropna()
    if len(s) < TOP_N:
        continue

    before = list(held)
    held = select(s, universe, held)
    if not held:
        continue

    history.append((d, before, list(held)))
    for c in held:
        records.append({"date": d, "ts_code": c})

if len(history) < 2:
    raise SystemExit("有效调仓期不足 2 个，检查平滑窗口")

log(f"共 {len(history)} 个调仓期，最后一期 {history[-1][0]}")

pd.DataFrame(records).to_parquet(PAPER / "replay_holdings.parquet", index=False)
log("→ paper/replay_holdings.parquet")


# ==================== 播种 ====================
last_d, _, last_after = history[-1]
prev_d, _, prev_after = history[-2]

state = {
    "last_rebalance": prev_d,
    "holdings": prev_after,
    "nav": 1.0,
    "last_date": prev_d,
}
(PAPER / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2))

log("\n已播种 paper/state.json")
log(f"   last_rebalance = {prev_d}（{len(prev_after)} 只）")
log(f"   目标验证日     = {last_d}")
log(f"\n验证命令：")
log(f"   .venv/bin/python daily_run.py --date {last_d} --force")
log(f"   → 与 paper/replay_holdings.parquet 中 date == {last_d} 的集合比对")
log(f"\n验证完记得再跑一次本脚本复位 state.json（--force 会改写它）")
