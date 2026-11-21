#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
50_rebuild_base.py
按「增量」而不是「单因子 IC」重选基础因子。

背景
  原始 4 因子是按单因子 IC 选的：vol_20 / turn_20 / rev_5 / ep。
  后来的九个外部因子全部卡在「残差 IC ≈ 0.01、组合边际 t < 2」——
  说明既不是因子弱，也不是选错了，是**筛选标准选错了**。

本脚本回答三件事
  A. 缩小基础：1 / 2 / 3 / 4 个因子，哪个最好？
  B. 跨数据源：量价 + 融资余额，是否比同源的四个强？
  C. 贪心前向选择：从最好的单因子出发，按「边际 t」逐个加，加不动就停。

权重（重要）
  加第 k+1 个因子时用 w = 1/(k+1)，即 (Σk + f)/(k+1)，
  与「等权 k+1 因子」完全等价，不是 80% 方差倾斜那个旧口径。

判据（预注册）
  边际 t >= 2 才允许加入；否则停。

用法
  .venv/bin/python -u 50_rebuild_base.py      # 约 10–12 分钟
输出
  data/output_zoo/rebuild_base.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from factorlab.core import load_panel, smooth_score, replay, benchmark, FACTORS
from factorlab.factors import load_cached

OUT = Path("data/output_zoo")
OUT.mkdir(parents=True, exist_ok=True)

EXTRA = ["margin_chg", "ovn_mom", "idr_rev", "q_ag", "max_ret", "amihud"]
POOL = list(FACTORS) + EXTRA


def log(m):
    print(m, flush=True)


def geom(s):
    s = s.dropna()
    return float((1 + s).cumprod().iloc[-1] ** (252 / len(s)) - 1) if len(s) else np.nan


def minfo(r, b):
    net = r.net
    bb = b.reindex(net.index).fillna(0.0)
    ex = (net - bb).dropna()
    te = float(ex.std() * np.sqrt(252))
    return {
        "年化": geom(net),
        "超额pp": (geom(net) - geom(bb)) * 100,
        "毛超额pp": (geom(net) - geom(bb)) * 100
                    + r.ann_turnover() * 0.0012 * 2 * 100,
        "TE": te,
        "IR": ((geom(net) - geom(bb)) / te) if te > 0 else np.nan,
        "换手": r.ann_turnover(),
    }


def make_composite(mats):
    """等权复合，NaN-aware，与 core.composite 同语义"""
    out = sum(m.fillna(0.0) for m in mats)
    valid = sum(m.notna().astype(int) for m in mats) > 0
    return (out / len(mats)).where(valid)


def main():
    t0 = time.time()
    log("① 加载面板与因子...")
    p = load_panel()
    b = benchmark(p)
    fac = {n: p.fac[n] for n in FACTORS}
    for n in EXTRA:
        f = load_cached(n)
        if f is None:
            log(f"   缺 {n}，跳过")
            continue
        fac[n] = f.reindex(index=p.idx, columns=p.cols)
    log(f"   {len(p.idx)} 天 × {len(p.cols)} 只；可用因子 {len(fac)} 个")

    rows = []

    def evaluate(names, tag):
        comp = make_composite([fac[n] for n in names])
        r = replay(smooth_score(comp), p)
        d = minfo(r, b)
        d["组合"] = tag
        d["n"] = len(names)
        rows.append(d)
        log(f"   {tag:<34} 年化 {d['年化']:+.2%}  超额 {d['超额pp']:+.2f}pp  "
            f"毛 {d['毛超额pp']:+.2f}pp  TE {d['TE']:.2%}  "
            f"IR {d['IR']:.2f}  换手 {d['换手']:.0%}")
        return d

    log("\n② A. 缩小基础...")
    evaluate(["vol_20"], "A1 单因子 vol_20")
    evaluate(["vol_20", "rev_5"], "A2 vol_20+rev_5")
    evaluate(["vol_20", "rev_5", "turn_20"], "A3 +turn_20")
    evaluate(list(FACTORS), "A4 现行 4 因子")

    log("\n③ A'. 其他单因子（找最好的起点）...")
    for n in ("rev_5", "turn_20", "ep", "margin_chg"):
        if n in fac:
            evaluate([n], f"A0 单因子 {n}")

    log("\n④ B. 跨数据源...")
    if "margin_chg" in fac:
        evaluate(["vol_20", "margin_chg"], "B1 vol_20 + margin_chg（量价+融资）")
        evaluate(["vol_20", "rev_5", "margin_chg"], "B2 三因子跨源")
    if "q_ag" in fac:
        evaluate(["vol_20", "q_ag"], "B3 vol_20 + q_ag（量价+财务）")

    log("\n⑤ C. 贪心前向选择（按边际 t）...")
    # 起点：单因子里净超额最高的
    singles = {r["组合"].split()[-1]: r["超额pp"] for r in rows
               if r["组合"].startswith("A0") or r["组合"].startswith("A1")}
    start = max(singles, key=singles.get)
    chosen = [start]
    log(f"   起点 {start}（单因子净超额 {singles[start]:+.2f}pp）")

    base_comp = make_composite([fac[start]])
    r_base = replay(smooth_score(base_comp), p)
    base_ann = r_base.ann()

    for step in range(3):
        k = len(chosen)
        w = 1.0 / (k + 1)
        cand = [n for n in POOL if n in fac and n not in chosen]
        best = None
        for n in cand:
            aug = smooth_score(base_comp * (1 - w) + fac[n].fillna(0.0) * w)
            r = replay(aug, p)
            d = (r.net - r_base.net).dropna()
            t = (float(d.mean() / d.std() * np.sqrt(len(d)))
                 if d.std() > 0 else np.nan)
            if best is None or (np.isfinite(t) and t > best[1]):
                best = (n, t, r, d)
        if best is None:
            break
        n, t, r, d = best
        log(f"   第 {k + 1} 个候选：{n:<10} w={w:.2f}  边际 t {t:+.2f}  "
            f"Δ {(r.ann() - base_ann) * 100:+.2f}pp")
        if not (np.isfinite(t) and t >= 2.0):
            log(f"   → t < 2，停止")
            break
        chosen.append(n)
        base_comp = make_composite([fac[x] for x in chosen])
        r_base = replay(smooth_score(base_comp), p)
        base_ann = r_base.ann()
        log(f"   → 加入 {n}，当前组合 {chosen}")

    log(f"\n   贪心结果：{chosen}")
    evaluate(chosen, "C 贪心结果")

    tab = pd.DataFrame(rows).set_index("组合")[
        ["n", "年化", "超额pp", "毛超额pp", "TE", "IR", "换手"]]
    for c in ("年化", "超额pp", "毛超额pp", "TE", "换手"):
        tab[c] = tab[c].round(4)
    tab["IR"] = tab["IR"].round(2)

    log("\n" + "=" * 110)
    log("【汇总】")
    log("=" * 110)
    log(tab.to_string())

    cur = float(tab.loc["A4 现行 4 因子", "超额pp"])
    best_row = tab["超额pp"].idxmax()
    log("\n【判读】")
    log(f"  现行 4 因子：{cur:+.2f}pp")
    log(f"  最优组合：  {best_row}  {tab.loc[best_row, '超额pp']:+.2f}pp  "
        f"（差 {tab.loc[best_row, '超额pp'] - cur:+.2f}pp）")
    log(f"  贪心结果：  {tab.loc['C 贪心结果', '超额pp']:+.2f}pp，"
        f"因子数 {int(tab.loc['C 贪心结果', 'n'])}")
    log("  若少而优 → 说明当年的「单因子 IC」标准确实选多了；"
        "若差别 <0.5pp → 说明结果对基础选择不敏感，这条也不重要")

    tab.to_csv(OUT / "rebuild_base.csv", encoding="utf-8-sig")
    log(f"\n✅ 输出 {OUT / 'rebuild_base.csv'}")
    log(f"总耗时 {(time.time() - t0) / 60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    sys.exit(main())
