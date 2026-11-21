#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
62_realistic.py
最现实口径：次日开盘成交 + 逐股冲击成本。

两个折价此前分别测过，但没有叠加：
  成本：12bp → 逐股（N=400 @1亿：5.15 → 3.51）
  执行：收盘 → 开盘（N=400：5.15 → 3.86）
本脚本给 2×2 矩阵，并外推「开盘口径」下的容量。

为什么重要
  对外报 5.15pp 会被问「成本呢」；报 3.51pp 会被问「怎么成交」。
  只有叠加之后的数，才经得起这两问。

判据
  叠加后 > 2.5pp → 产品在这个规模上仍成立
  1.5–2.5pp    → 勉强，靠规模和费率活着
  < 1.5pp      → 1 亿规模上已不具备商业价值

用法
  .venv/bin/python -u 62_realistic.py
输出
  data/output_zoo/realistic.csv
"""
import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from factorlab.core import (load_panel, composite, smooth_score, replay,
                            benchmark)
from factorlab.factors import load_price_data

OUT = Path("data/output_zoo")
OUT.mkdir(parents=True, exist_ok=True)

N = 400
CAP = 0.02
AUMS = [(0.5e8, "0.5亿"), (1e8, "1.0亿"), (2e8, "2.0亿")]


def log(m):
    print(m, flush=True)


def geom(s):
    s = s.dropna()
    return float((1 + s).cumprod().iloc[-1] ** (252 / len(s)) - 1) if len(s) else np.nan


def cost_matrix(p, daily, aum, k=0.7, floor=0.0003, part=1.0 / N):
    sig = p.ret.replace([np.inf, -np.inf], np.nan).rolling(20, min_periods=20).std()
    amt = (daily.pivot(index="trade_date", columns="ts_code", values="amount")
                .reindex(index=p.idx, columns=p.cols)) * 1000.0
    adv = amt.rolling(20, min_periods=20).mean()
    c = k * sig * np.sqrt((aum * part) / adv.where(adv > 0))
    return (c + floor).clip(upper=CAP)


def excess(r, b):
    net = r.net
    bb = b.reindex(net.index).fillna(0.0)
    return (geom(net) - geom(bb)) * 100


def main():
    t0 = time.time()
    log("① 加载...")
    p = load_panel()
    daily, _ = load_price_data(daily_cols=("pct_chg", "open", "amount"))
    op = (daily.pivot(index="trade_date", columns="ts_code", values="open")
              .reindex(index=p.idx, columns=p.cols).replace(0, np.nan))

    p_oo = copy.copy(p)
    p_oo.fwd = op.shift(-2) / op.shift(-1) - 1
    log(f"   {len(p.idx)} 天 × {len(p.cols)} 只")

    score = smooth_score(composite(p))
    b_cc, b_oo = benchmark(p), benchmark(p_oo)

    log(f"\n② 2×2 矩阵（N={N}）...")
    cms = {lab: cost_matrix(p, daily, aum) for aum, lab in AUMS}

    rows = []
    for conv, pp, bb in (("收盘", p, b_cc), ("开盘", p_oo, b_oo)):
        row = {"成交口径": conv, "12bp": round(excess(replay(score, pp, top_n=N), bb), 2)}
        for _, lab in AUMS:
            r = replay(score, pp, top_n=N, cost=cms[lab])
            row[lab] = round(excess(r, bb), 2)
        rows.append(row)
        log(f"   {conv}  " + "  ".join(f"{k}={v:+.2f}" for k, v in row.items()
                                       if k != "成交口径"))

    tab = pd.DataFrame(rows).set_index("成交口径")
    log("\n" + "=" * 88)
    log("【超额 pp：成交口径 × 规模】")
    log("=" * 88)
    log(tab.to_string())

    # ---------- 分解 ----------
    cc12 = float(tab.loc["收盘", "12bp"])
    oo12 = float(tab.loc["开盘", "12bp"])
    cc1 = float(tab.loc["收盘", "1.0亿"])
    oo1 = float(tab.loc["开盘", "1.0亿"])

    log("\n" + "=" * 88)
    log("【分解】")
    log("=" * 88)
    log(f"  12bp / 收盘          {cc12:+.2f}pp   ← 论文口径（最乐观）")
    log(f"  → 执行改开盘         {cc12 - oo12:+.2f}pp   （{oo12:+.2f}pp）")
    log(f"  → 成本改逐股@1亿     {cc12 - cc1:+.2f}pp   （{cc1:+.2f}pp）")
    log(f"  → 两者叠加           {cc12 - oo1:+.2f}pp   （{oo1:+.2f}pp）")
    log(f"\n  可加性检验：{oo12 - cc12:+.2f} + {cc1 - cc12:+.2f} = "
        f"{(oo12 - cc12) + (cc1 - cc12):+.2f}，实测 {oo1 - cc12:+.2f}")

    # ---------- 容量（开盘口径） ----------
    log("\n" + "=" * 88)
    log("【容量外推（开盘口径，线性插值）】")
    log("=" * 88)
    for conv in ("收盘", "开盘"):
        y1 = float(tab.loc[conv, "1.0亿"])
        y2 = float(tab.loc[conv, "2.0亿"])
        slope = y2 - y1                      # pp per 亿
        be = 2.0 + y2 / (-slope) if slope < 0 else np.nan
        log(f"  {conv}口径：1亿 {y1:+.2f}pp，2亿 {y2:+.2f}pp，"
            f"斜率 {slope:+.2f}pp/亿 → **平衡点约 {be:.1f} 亿**")

    log("\n" + "=" * 88)
    log("【判读】")
    log("=" * 88)
    if oo1 > 2.5:
        log(f"  1 亿、开盘成交、逐股成本下超额 {oo1:+.2f}pp → 产品仍成立")
    elif oo1 > 1.5:
        log(f"  1 亿、开盘成交、逐股成本下超额 {oo1:+.2f}pp → 勉强，靠规模与费率")
    else:
        log(f"  1 亿、开盘成交、逐股成本下超额 {oo1:+.2f}pp → 该规模不具备商业价值")
    log(f"\n  对外表述建议：")
    log(f"    论文口径 {cc12:.2f}pp；真实执行口径 {oo1:.2f}pp（1 亿），差 {cc12-oo1:.2f}pp。")

    tab.to_csv(OUT / "realistic.csv", encoding="utf-8-sig")
    log(f"\n✅ 输出 {OUT / 'realistic.csv'}")
    log(f"总耗时 {(time.time() - t0) / 60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    sys.exit(main())
