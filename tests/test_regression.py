# tests/test_regression.py
"""
factorlab 必须先复现现有结论，才允许替换旧脚本。
运行：.venv/bin/python tests/test_regression.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from factorlab.core import (Panel, load_panel, composite, smooth_score,
                            replay, benchmark, TOP_N)


def main():
    print("=" * 60)
    print("factorlab 回归测试")
    print("=" * 60)

    p = load_panel()
    print(f"面板 {len(p.idx)} 天 × {len(p.cols)} 只")

    score = smooth_score(composite(p))
    r = replay(score, p)

    ann = r.ann()
    to = r.ann_turnover()
    dev_max = max(r.ind_dev.values())

    print(f"\n组合年化        {ann:.2%}")
    print(f"年化换手        {to:.0%}")
    print(f"最大行业偏离    {dev_max:.2%}")
    print(f"净值天数        {len(r.net)}  最后一天 {r.net.index[-1]}")

    # 宽松区间：先确认量级，等 9.65 / 10.10 的矛盾查清后再收紧
    checks = [
        ("年化在 9.4%–10.3%", 0.094 <= ann <= 0.103),
        ("换手在 680%–740%", 6.80 <= to <= 7.40),
        ("最大偏离 ≤ 1.5%", dev_max <= 0.015),
    ]
    print()
    ok = True
    for name, cond in checks:
        print(f"  {'✅' if cond else '❌'} {name}")
        ok = ok and cond

    # 与已存回放持仓比对（若存在）
    f = Path("data/paper/replay_holdings.parquet")
    if f.exists():
        stored = pd.read_parquet(f)
        last = sorted(stored.date.unique())[-1]
        a = set(stored[stored.date == last].ts_code)
        b = set(r.holdings.get(last, []))
        print(f"\n与 paper/replay_holdings.parquet 比对（{last}）")
        print(f"  交集 {len(a & b)} / {TOP_N}")
        if len(a & b) < TOP_N:
            print(f"  只在存储: {sorted(a - b)}")
            print(f"  只在本次: {sorted(b - a)}")

    print("\n" + "=" * 60)
    print("✅ 通过" if ok else "❌ 未通过")
    print("=" * 60)


if __name__ == "__main__":
    main()
