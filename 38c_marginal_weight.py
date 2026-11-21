#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
38c_marginal_weight.py
修正组合边际测试的权重。

问题
  gate.evaluate 的边际测试是 composite + f。
  composite 是 4 个 z 的均值（std≈0.5），f 的 std≈1，
  所以这个小测试实际给新因子 80% 的方差权重 —— 它测的是"能不能替换"，不是"有没有补充"。

换算
  等权 5 因子 = (Σ4z + f)/5 = 0.8·composite + 0.2·f   → 新因子占 20% 方差
  半权        = 0.9·composite + 0.1·f                 → 占 10%

用法
  .venv/bin/python -u 38c_marginal_weight.py      # 约 8–9 分钟（12 次回放）

输出
  data/output_zoo/marginal_weight.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from factorlab.core import load_panel, composite, smooth_score, replay
from factorlab.factors import load_cached

OUT = Path("data/output_zoo")
OUT.mkdir(parents=True, exist_ok=True)

FACTORS = ["rev_20", "amihud", "max_ret", "high_52w", "turn_abn", "q_ag"]
CONFIGS = [("等权5 w=0.20", 0.20), ("半权 w=0.10", 0.10)]

# 旧口径（composite + f，即 w 对应 80% 方差）的已知结果，仅作对照打印
OLD = {"rev_20": (-1.70, -5.97), "amihud": (+0.31, +0.44),
       "max_ret": (-1.91, -2.91), "high_52w": (np.nan, np.nan),
       "turn_abn": (-1.77, -3.71), "q_ag": (-0.31, np.nan)}


def log(m):
    print(m, flush=True)


def main():
    t0 = time.time()
    log("① 加载面板与缓存因子...")
    p = load_panel()
    fac = {}
    for n in FACTORS:
        f = load_cached(n)
        if f is None:
            log(f"   ⚠️ 缺 {n} 的缓存，跳过")
            continue
        fac[n] = f.reindex(index=p.idx, columns=p.cols)
    log(f"   {len(p.idx)} 天 × {len(p.cols)} 只；可用因子 {list(fac)}")

    log("② 基准回放（composite 等权 4 因子）...")
    comp = composite(p)
    base = replay(smooth_score(comp), p)
    base_ann, base_to = base.ann(), base.ann_turnover()
    log(f"   年化 {base_ann:.2%}  换手 {base_to:.0%}")

    rows = []
    log("\n③ 逐因子 × 权重...")
    for name, f in fac.items():
        old_t, old_d = OLD.get(name, (np.nan, np.nan))
        log(f"\n   {name}  （旧口径 composite+f：t {old_t:+.2f}, Δ {old_d:+.2f}pp）")
        for label, w in CONFIGS:
            aug = smooth_score(comp * (1 - w) + f.fillna(0.0) * w)
            r = replay(aug, p)
            d = (r.net - base.net).dropna()
            t = (float(d.mean() / d.std() * np.sqrt(len(d)))
                 if d.std() > 0 else np.nan)
            delta = (r.ann() - base_ann) * 100
            rows.append({"因子": name, "权重": label, "w": w,
                         "年化": round(r.ann(), 4),
                         "Δpp": round(delta, 2),
                         "配对t": round(t, 2) if np.isfinite(t) else None,
                         "换手": round(r.ann_turnover(), 4)})
            log(f"     {label:<12} 年化 {r.ann():.2%}  Δ {delta:+.2f}pp  "
                f"t {t:+.2f}  换手 {r.ann_turnover():.0%}")

    tab = pd.DataFrame(rows)
    log("\n" + "=" * 92)
    log("【组合边际 · 权重敏感性】")
    log("=" * 92)
    log(tab.to_string(index=False))

    good = tab[(tab["配对t"].notna()) & (tab["配对t"] >= 1.0) & (tab["Δpp"] > 0)]
    if good.empty:
        log("\n结论：没有任何因子在正确权重下给出正的边际贡献。")
        log("      → 之前 0/4 的结论成立，不是权重问题。")
    else:
        log("\n以下组合在正确权重下边际为正，值得进一步：")
        log(good.to_string(index=False))
        log("      → 之前 0/4 部分源于 80% 权重倾斜，需要改写。")

    tab.to_csv(OUT / "marginal_weight.csv", index=False, encoding="utf-8-sig")
    log(f"\n✅ 输出 {OUT / 'marginal_weight.csv'}")
    log(f"总耗时 {(time.time() - t0) / 60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    sys.exit(main())
