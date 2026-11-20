#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
18_significance.py — 增量是否显著（配对检验）

对每一个"加 SUE"的方案，与对应基线逐日相减，检验日均差是否显著不为零。
两个组合共用同一个基准，所以净收益之差的均值 = 超额收益之差的均值，基准可约掉。
"""
import numpy as np
import pandas as pd

d = pd.read_parquet("data/output_backtest/daily_net.parquet")

PAIRS = [
    ("EW + SUE",              "Equal Weight"),
    ("EW + SUE (w=0.5)",      "Equal Weight"),
    ("EW + SUE + Stratified", "EW + Stratified"),
]

print(f"{'对比':<44}{'年均差':>9}{'年化波动':>10}{'t':>7}{'AC1':>7}{'n':>7}")
print("-" * 84)
rows = []
for a, b in PAIRS:
    diff = (d[a] - d[b]).dropna()
    ann = diff.mean() * 252
    sd = diff.std() * np.sqrt(252)
    t = diff.mean() / diff.std() * np.sqrt(len(diff))
    ac1 = diff.autocorr(1)
    rows.append({"pair": f"{a} − {b}", "ann": ann, "t": t})
    print(f"{a + ' − ' + b:<44}{ann:>+9.2%}{sd:>10.2%}{t:>7.2f}{ac1:>7.3f}{len(diff):>7}")

# 分年差异
print("\n===== 年均差（百分点）=====")
diff_tab = pd.DataFrame({f"{a} − {b}": (d[a] - d[b]) * 252 * 100 for a, b in PAIRS})
diff_tab["year"] = diff_tab.index.str[:4].values
print(diff_tab.groupby("year").mean().round(2).to_string())

# Bootstrap 置信区间（不依赖正态假设）
print("\n===== Bootstrap 95% CI（年化）=====")
rng = np.random.default_rng(42)
for a, b in PAIRS:
    diff = (d[a] - d[b]).dropna().to_numpy()
    idx = rng.integers(0, len(diff), (5000, len(diff)))
    boot = diff[idx].mean(axis=1) * 252
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"{a + ' − ' + b:<44}[{lo:+.2%}, {hi:+.2%}]  {'含 0' if lo < 0 < hi else '不含 0'}")

# 与 headline 的比较（能不能取代 EW）
print("\n===== 能不能取代 headline（EW 4 因子）=====")
for a in ["EW + SUE", "EW + SUE + Stratified"]:
    diff = (d[a] - d["Equal Weight"]).dropna()
    t = diff.mean() / diff.std() * np.sqrt(len(diff))
    print(f"{a:<26} vs Equal Weight : 年均差 {diff.mean() * 252:+.2%}  t={t:.2f}")
