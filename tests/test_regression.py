# tests/test_regression.py
"""
factorlab 必须先复现现有结论，才允许替换旧脚本。
运行：.venv/bin/python tests/test_regression.py

与 v1 的差别
  ★ 区间从 9.4–10.3% 收紧到 ±0.10pp（原区间抓不到 composite 口径变化）
  ★ 失败时 sys.exit(1)：此前无论成败退出码都是 0，CI 无法阻断
  ★ 新增一条成本闭包：net + cost_amt == gross
  ★ 新增一条双边计费恒等式：年化成本 ≈ 换手 × 2 × COST
  ★ 50/50 一致性由 print 改为 assert
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from factorlab.core import (load_panel, composite, smooth_score, replay,
                            TOP_N, COST)

EXPECT = {"ann": 0.0965, "turnover": 7.14, "dev_max": 0.0135}
TOL = {"ann": 0.0010, "turnover": 0.06, "dev_max": 0.0005}


def main():
    print("=" * 60)
    print("factorlab 回归测试")
    print("=" * 60)

    p = load_panel()
    print(f"面板 {len(p.idx)} 天 × {len(p.cols)} 只")

    score = smooth_score(composite(p))
    r = replay(score, p, top_n=TOP_N)

    ann = r.ann()
    to = r.ann_turnover()
    dev_max = max(r.ind_dev.values())

    print(f"\n组合年化      {ann:.2%}   （期望 {EXPECT['ann']:.2%} ± {TOL['ann']*100:.2f}pp）")
    print(f"年化换手      {to:.0%}     （期望 {EXPECT['turnover']:.0%} ± {TOL['turnover']*100:.0f}pp）")
    print(f"最大行业偏离  {dev_max:.2%}  （上限 {EXPECT['dev_max']:.2%}）")
    print(f"净值天数      {len(r.net)}  最后一天 {r.net.index[-1]}")

    checks = [
        ("年化复现", abs(ann - EXPECT["ann"]) <= TOL["ann"]),
        ("换手复现", abs(to - EXPECT["turnover"]) <= TOL["turnover"]),
        ("行业偏离", dev_max <= EXPECT["dev_max"] + TOL["dev_max"]),
    ]

    # ① 成本闭包：core 里 net = gross − cost_amt（net 少最后一行）
    if r.cost_amt is not None:
        lhs = r.net + r.cost_amt.reindex(r.net.index).fillna(0.0)
        gap = float((lhs - r.gross.reindex(r.net.index)).abs().max())
        checks.append((f"net + cost_amt == gross（max|gap| = {gap:.1e}）", gap < 1e-10))

        # ② 双边计费恒等式
        c_pp = float(r.cost_amt.mean() * 252 * 100)
        t_pp = to * 2 * COST * 100
        checks.append((f"成本 {c_pp:.2f}pp ≈ 换手×2×{COST*1e4:.0f}bp = {t_pp:.2f}pp",
                       abs(c_pp - t_pp) < 0.05))

    # ③ 与纸面回放持仓一致性（原版只 print，不阻断）
    f = Path("data/paper/replay_holdings.parquet")
    if f.exists():
        stored = pd.read_parquet(f)
        last = sorted(stored.date.unique())[-1]
        a = set(stored[stored.date == last].ts_code)
        b = set(r.holdings.get(last, []))
        same = len(a & b)
        print(f"\n与 paper/replay_holdings.parquet 比对（{last}）：交集 {same}/{TOP_N}")
        if same < TOP_N:
            print(f"  只在存储: {sorted(a - b)}")
            print(f"  只在本次: {sorted(b - a)}")
        checks.append((f"live == replay（{same}/{TOP_N}）", same == TOP_N))
    else:
        print("\n未找到 replay_holdings.parquet，跳过 50/50 校验")

    print()
    ok = True
    for name, cond in checks:
        print(f"  {'✅' if cond else '❌'} {name}")
        ok = ok and cond

    print("\n" + "=" * 60)
    print("✅ 通过" if ok else "❌ 未通过")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
