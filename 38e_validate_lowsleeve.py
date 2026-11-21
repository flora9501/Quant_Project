#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""38e_validate_lowsleeve.py — 验 ④ 的 +4.42% / +3.28% 是不是真的。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from factorlab.core import load_panel, FACTORS, composite, smooth_score
from factorlab.factors import load_price_data, neutralize
import importlib.util
spec = importlib.util.spec_from_file_location("d", Path(__file__).with_name("38d_lowsleeve.py"))
d = importlib.util.module_from_spec(spec); spec.loader.exec_module(d)   # 复用 replay_n / comp

OUT = Path("data/output_zoo")
HIST_END = "20251231"
DAILY_COLS = ("pct_chg", "amount")


def log(m):
    print(m, flush=True)


def yearly_excess(net, panel):
    bm = d.bench_net(panel, net.index)
    yr = pd.Series([str(x)[:4] for x in net.index], index=net.index)
    out = {}
    for y in sorted(yr.unique()):
        m = (yr == y)
        n, b = net[m], bm[m]
        out[y] = float(((1 + n).cumprod().iloc[-1] /
                        (1 + b).cumprod().iloc[-1]) ** (252 / len(n)) - 1)
    return pd.Series(out)


def main():
    panel = load_panel(hist_end=HIST_END)
    daily, basic = load_price_data(daily_cols=DAILY_COLS)
    ret = d._wide(daily, "pct_chg", panel.idx, panel.cols) / 100.0
    amt = d._wide(daily, "amount", panel.idx, panel.cols)
    cmv = d._wide(basic, "circ_mv", panel.idx, panel.cols)
    ln_mv = -np.log(cmv * 1e4)
    ret_a = d.alive_zero(ret)
    illiq = (ret_a.abs() * 1e4) / (amt * 1000.0).replace(0, np.nan)
    base_score = composite(panel)

    # ---- 基准 20 日 ----
    base20 = d.replay_n(smooth_score(base_score, rebal=20), panel, 20)
    ex20 = d.excess_ann(base20.net, panel)
    log(f"基准 4因子@20: 超额 {ex20:+.2%}  dev_max {max(base20.ind_dev.values()):.2%}")

    # ---- 验证 1 & 2：窗口 × 持有期网格 ----
    log("\n【网格】amihud 窗口 × 持有期（Δ超额 相对同持有期 4 因子）")
    log(f"   {'win':>4} " + " ".join(f"{'@'+str(rb):>10}" for rb in (10, 20, 30)))
    grid = {}
    for win in (10, 20, 40):
        amihud = illiq.rolling(win, min_periods=win).mean()
        neu = neutralize({"amihud": amihud, "ln_mv": ln_mv}, panel.mask, panel.industry,
                         names=["amihud"], verbose=False)["amihud"]
        for rb in (10, 20, 30):
            b = d.replay_n(smooth_score(base_score, rebal=rb), panel, rb)
            exb = d.excess_ann(b.net, panel)
            f = {**panel.fac, "amihud": neu}
            rp = d.replay_n(smooth_score(d.comp(panel, f, FACTORS + ["amihud"]), rebal=rb), panel, rb)
            ex = d.excess_ann(rp.net, panel)
            dd = (rp.net - b.net).dropna()
            t = float(dd.mean() / dd.std() * np.sqrt(len(dd))) if dd.std() > 0 else np.nan
            grid[(win, rb)] = (ex, t, max(rp.ind_dev.values()) if rp.ind_dev else np.nan)
        row = " ".join(f"{grid[(win, rb)][0]-d.excess_ann(d.replay_n(smooth_score(base_score, rebal=rb), panel, rb).net, panel):+6.2%}(t{grid[(win,rb)][1]:+.1f})"
                       for rb in (10, 20, 30))
        log(f"   {win:>4} {row}")

    # ---- 验证 3：分年（用 win=20, rb=20）----
    log("\n【分年】win=20, rb=20")
    amihud20 = neutralize({"amihud": illiq.rolling(20, min_periods=20).mean(),
                           "ln_mv": ln_mv}, panel.mask, panel.industry,
                          names=["amihud"], verbose=False)["amihud"]
    f = {**panel.fac, "amihud": amihud20}
    rp = d.replay_n(smooth_score(d.comp(panel, f, FACTORS + ["amihud"]), rebal=20), panel, 20)
    tb = yearly_excess(base20.net, panel)
    ta = yearly_excess(rp.net, panel)
    df = pd.DataFrame({"base20": tb, "amihud@20": ta, "Δ": ta - tb})
    log(df.round(4).to_string())
    log(f"   Δ>0 年数 {(df['Δ'] > 0).sum()}/{len(df)}")
    log(f"   行业最大偏离 {max(rp.ind_dev.values()):.2%}（基准 {max(base20.ind_dev.values()):.2%}）")
    log(f"   换手 {rp.ann_turnover():.0%}（基准 {base20.ann_turnover():.0%}）")

    df.to_csv(OUT / "lowsleeve_yearly.csv", encoding="utf-8-sig")


if __name__ == "__main__":
    main()
