#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""优化器烟雾测试：能不能解 + 约束松不松。约 1–2 分钟。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from factorlab.core import load_panel, composite, smooth_score
from factorlab.optimize import estimate_risk, exposures, solve_mv, diagnose

W_MAX, BAND_IND, BAND_STY, TAU, LAM, M_CAND = 0.02, 0.02, 0.25, 0.30, 5.0, 400

p = load_panel(with_index=True)
sc = smooth_score(composite(p))
rm = estimate_risk(p)                       # ← 约 30–60 秒

labels = list(rm.factors) + list(rm.gnames)
band = np.array([BAND_STY] * len(rm.factors) + [BAND_IND] * len(rm.gnames))

d = p.idx[-1]
names = p.mask.loc[d][p.mask.loc[d]].index
cand = sc.loc[d].where(p.mask.loc[d]).dropna().nlargest(M_CAND).index

Xall = exposures(p, rm.groups, rm.gnames, d, names)
Xc = Xall[np.isin(np.asarray(names), cand)]
wb_full = pd.Series(1.0 / len(names), index=names)
xb = Xall.T @ wb_full.to_numpy()
wb_c = wb_full.reindex(cand).fillna(0.0).to_numpy()
d_c = rm.spec_var.loc[d, cand].to_numpy()   # ← spec_var，不是 sv
d_c = np.where(np.isfinite(d_c), d_c, 1e-6)
w0_c = np.zeros(len(cand))
alpha = sc.loc[d].reindex(cand).to_numpy()

print(f"\n日期 {d}   候选 {len(cand)}   基准池 {len(names)}   K={Xall.shape[1]}")

# ① 约束诊断（在等权解上）
dg = diagnose(alpha, Xc, xb, rm.L, d_c, wb_c, wb_c, band, W_MAX, TAU,
              cand, labels)
print("\n【约束诊断 · 等权解】")
print(f"  TE(年化)      {dg['te_ann']*100:.2f}%")
print(f"  最紧的约束    {dg['worst_exposure']}  用掉 {dg['worst_ratio']:.0%} 预算")
print(f"  换手          {dg['turnover']:.3f} / {TAU}  → 用掉 {dg['turnover_ratio']:.0%}")
print(f"  单票权重      {dg['max_w']:.4f} / 上限 {W_MAX}")

# ② 真解一次
w = solve_mv(alpha, Xc, xb, rm.L, d_c, wb_c, w0_c, band, W_MAX, TAU, LAM)
if w is None:
    print("\n❌ solve 返回 None —— 把 solve_mv 的 solver 改成 cp.ECOS 再试")
else:
    nz = int((w > 1e-6).sum())
    print(f"\n【QP 解】λ={LAM:g}   持仓 {nz} 只   "
          f"最大权重 {w.max():.4f}   与等权 L1 距离 "
          f"{np.abs(w - 1.0/len(cand)).sum():.3f}")
