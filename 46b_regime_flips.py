#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
46b_regime_flips.py
数 regime overlay 的翻转频率与成本 —— 只诊断，不改任何策略、不碰 paper 目录。

判据
  >8 次/年  → 关闭（择时错误 + 换手，多于收益）
  <=4 次/年 → 值得做两小时的 paper 实验

用法
  .venv/bin/python -u 46b_regime_flips.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from factorlab.core import load_panel

W = 20
MINP = 250
COST = 0.0012          # 每单位成交额的成本
CUT = 0.5              # 低状态下把仓位打到 50%


def log(m):
    print(m, flush=True)


def main():
    p = load_panel()
    mkt = p.ret.where(p.mask).mean(axis=1)
    s = ((1 + mkt).rolling(W, min_periods=W).apply(np.prod) - 1).dropna()
    log(f"市场 20 日动量：{len(s)} 天  {s.index[0]} ~ {s.index[-1]}")
    yrs = len(s) / 252

    rows = []
    for q in (0.25, 1 / 3):
        for mode in ("expanding", "rolling"):
            thr = (s.expanding(MINP).quantile(q) if mode == "expanding"
                   else s.rolling(MINP).quantile(q))
            flag = (s < thr.to_numpy()).fillna(False).astype(bool)

            flips = int((flag != flag.shift(1)).sum())
            grp = (flag != flag.shift(1)).cumsum()
            runs = flag.groupby(grp).agg(["first", "size"])
            lows = runs[runs["first"]]
            mean_len = float(lows["size"].mean()) if len(lows) else np.nan

            # 每次翻转成交 50%，成本 0.5*12bp
            cost_pp = flips * CUT * COST * 100

            rows.append({
                "阈值": f"q={q:.2f}/{mode}",
                "翻转次数": flips,
                "次/年": round(flips / yrs, 1),
                "低状态占比": round(float(flag.mean()), 3),
                "平均持续天": round(mean_len, 1) if np.isfinite(mean_len) else None,
                "额外成本pp/年": round(cost_pp / yrs, 2),
            })

    tab = pd.DataFrame(rows)
    log("\n" + "=" * 90)
    log("【regime overlay 的翻转频率与成本】")
    log("=" * 90)
    log(tab.to_string(index=False))

    log("\n判读")
    log("  >8 次/年  → 关闭：择时错误 + 换手，收益盖不住")
    log("  <=4 次/年 → 值得做两小时 paper 实验（不改生产）")
    log("  平均持续天 <20 → 典型的 whipsaw 机器")

    log("\n注意：这里只算了成本，没算收益。收益端的天花板是")
    log("  低状态占 1/3 时间 × 超额 −3.70%/年 × 砍半 = 约 +0.6~1.8pp/年")
    log("  而且前提是这个 regime 对未来有预测力 —— 46 号只证明了");
    log("  「同期超额低」，没有证明「可预测」，两者不是一回事。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
