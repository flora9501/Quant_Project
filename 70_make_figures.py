#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
70_make_figures.py  (v3)
只生成两张图：
  figures/nav.png      净值：N=50 / N=400 / 基准
  figures/excess.png   超额净值：N=50 vs N=400

容量图（capacity.png）由 63_capacity.py 生成，本脚本不再重复。
v3 变更：删掉容量部分（避免与 63 冲突、省 6 分钟）；修正净值索引对齐。

用法
  .venv/bin/python -u 70_make_figures.py     # 约 2 分钟
然后跑
  .venv/bin/python -u 63_capacity.py         # 约 10 分钟，生成 capacity.png
"""
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from factorlab.core import load_panel, composite, smooth_score, replay, benchmark
from factorlab.factors import load_price_data

FIG = Path("figures")
FIG.mkdir(exist_ok=True)


def log(m):
    print(m, flush=True)


def nav(s):
    return (1 + s.dropna()).cumprod()


def main():
    log("① 加载与回放（2 次）...")
    p = load_panel()
    load_price_data(daily_cols=("pct_chg",))
    score = smooth_score(composite(p))
    b = benchmark(p)

    r50 = replay(score, p, top_n=50)
    r400 = replay(score, p, top_n=400)

    # ★ 统一索引：replay 净值 1920 天，基准必须对齐
    idx_net = r50.net.index
    b_net = b.reindex(idx_net).fillna(0.0)
    n50, n400, nbm = nav(r50.net), nav(r400.net), nav(b_net)
    idx = pd.to_datetime(idx_net, format="%Y%m%d")
    log(f"   净值天数：{len(n50)} / {len(n400)} / {len(nbm)}")

    # ---------- 图 1：净值 ----------
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(idx, nbm / nbm.iloc[0], label="Benchmark (EW universe)",
            lw=1.4, c="#888888")
    ax.plot(idx, n50 / n50.iloc[0], label="N=50 (original)",
            lw=1.4, c="#c0392b")
    ax.plot(idx, n400 / n400.iloc[0], label="N=400 (widened)",
            lw=1.9, c="#1a5276")
    ax.set_yscale("log")
    ax.set_ylabel("NAV (log scale)")
    ax.set_title("Net asset value, 2018–2025, net of 12 bp two-sided cost")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, ls="--")
    fig.tight_layout()
    fig.savefig(FIG / "nav.png", dpi=160)
    plt.close(fig)

    # ---------- 图 2：超额净值 ----------
    e50 = nav(r50.net) / nav(b_net)
    e400 = nav(r400.net) / nav(b_net)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(idx, e50 / e50.iloc[0], label="N=50", lw=1.4, c="#c0392b")
    ax.plot(idx, e400 / e400.iloc[0], label="N=400", lw=1.9, c="#1a5276")
    ax.axhline(1.0, ls="--", lw=1, c="#888888")
    ax.set_ylabel("Relative NAV vs benchmark")
    ax.set_title("Excess NAV (net of cost)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, ls="--")
    fig.tight_layout()
    fig.savefig(FIG / "excess.png", dpi=160)
    plt.close(fig)

    log(f"\n✅ nav.png / excess.png 写入 {FIG}/")
    log("   容量图请跑 63_capacity.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
