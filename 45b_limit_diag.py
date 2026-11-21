#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
45b_limit_diag.py
检查隔夜/日内因子是否被涨跌停（尤其一字板）污染。

三个输出（每个因子）
  ① IC（全部日）
  ② IC（构造时剔除 |pct_chg| >= 9.5 的交易日）
  ③ 相关性：因子 vs "近20日是否出现过一字板"，以及因子 top50 里的一字板占比

用法
  .venv/bin/python -u 45b_limit_diag.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from factorlab.core import load_panel
from factorlab.factors import load_price_data, fill_alive_zero
from factorlab.gate import forward_returns

LIMIT = 0.095          # 10% 与 20% 板都能覆盖
W = 20


def log(m):
    print(m, flush=True)


def _wide(df, col, dates, codes):
    return (df.pivot(index="trade_date", columns="ts_code", values=col)
              .reindex(index=dates, columns=codes))


def daily_ic(f, fwd, mask, min_n=50):
    a = f.where(mask).stack()
    b = fwd.where(mask).stack()
    d = pd.concat([a.rename("x"), b.rename("y")], axis=1).dropna()
    if d.empty:
        return pd.Series(dtype=float)
    d["rx"] = d.groupby(level=0)["x"].rank()
    d["ry"] = d.groupby(level=0)["y"].rank()
    g = d.groupby(level=0).apply(
        lambda z: z.rx.corr(z.ry) if len(z) >= min_n else np.nan)
    return g.dropna()


def summarize(name, ic):
    n = len(ic)
    if n == 0:
        log(f"   {name:<10} 无 IC")
        return
    t = ic.mean() / ic.std() * np.sqrt(n) if ic.std() > 0 else np.nan
    log(f"   {name:<10} IC {ic.mean():+.4f}  日数 {n}  "
        f"naive t {t:+.2f}（h=5 应打折 √5）")


def main():
    p = load_panel()
    daily, basic = load_price_data(
        daily_cols=("pct_chg", "open", "close", "pre_close"))

    op = _wide(daily, "open", p.idx, p.cols)
    cl = _wide(daily, "close", p.idx, p.cols)
    pc = _wide(daily, "pre_close", p.idx, p.cols)
    pct = _wide(daily, "pct_chg", p.idx, p.cols) / 100.0

    ovn = fill_alive_zero((op / pc.replace(0, np.nan) - 1).clip(-0.2, 0.2))
    idr = fill_alive_zero((cl / op.replace(0, np.nan) - 1).clip(-0.2, 0.2))

    limit = (pct.abs() >= LIMIT) & pct.notna()
    yz = limit & (op - cl).abs().lt(1e-9)          # 一字板：开=收 且 涨跌停
    yz20 = yz.astype(float).rolling(W, min_periods=1).max().fillna(0.0)

    log(f"全样本 涨跌停占比 {float(limit.to_numpy().mean()):.2%}  "
        f"其中一字板 {float(yz.to_numpy().mean()):.2%}")

    mp = int(W * 0.75)
    raw_all = {"ovn_20": -(ovn.rolling(W, min_periods=mp).sum()),
               "idr_20": idr.rolling(W, min_periods=mp).sum()}
    raw_ex = {"ovn_20": -(ovn.where(~limit).rolling(W, min_periods=mp).sum()),
              "idr_20": idr.where(~limit).rolling(W, min_periods=mp).sum()}

    fwd = forward_returns(p, 5)

    for name in ("ovn_20", "idr_20"):
        log(f"\n===== {name} =====")
        summarize("全部日", daily_ic(raw_all[name], fwd, p.mask))
        summarize("剔除涨跌停", daily_ic(raw_ex[name], fwd, p.mask))

        # 因子 vs 一字板指示变量
        a = raw_all[name].where(p.mask).stack()
        b = yz20.where(p.mask).stack()
        d = pd.concat([a.rename("x"), b.rename("y")], axis=1).dropna()
        d["rx"] = d.groupby(level=0)["x"].rank()
        d["ry"] = d.groupby(level=0)["y"].rank()
        c = d.groupby(level=0).apply(
            lambda z: z.rx.corr(z.ry) if len(z) >= 50 else np.nan).dropna()
        log(f"   与「近20日有一字板」的相关：{c.mean():+.3f}")

        # top50 里的一字板占比
        shares = []
        for dt in p.idx[W:]:
            s = raw_all[name].loc[dt].where(p.mask.loc[dt]).dropna()
            if len(s) < 50:
                continue
            top = s.nlargest(50).index
            shares.append(float(yz20.loc[dt, top].mean()))
        if shares:
            log(f"   top50 中一字板占比：均值 {np.mean(shares):.1%}  "
                f"（全池均值 {float(yz20.where(p.mask).mean().mean()):.1%}）")

    return 0


if __name__ == "__main__":
    sys.exit(main())
