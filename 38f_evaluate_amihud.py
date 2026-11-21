#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
38f_evaluate_amihud.py
三个检查：① 为什么是 20 日  ② 风险调整后还剩多少  ③ 容量

复用 38d_lowsleeve.py 里的 replay_n / comp / excess_ann（不改 38d）。
只用 2018-2025。

用法
    .venv/bin/python -u 38f_evaluate_amihud.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---- 复用 38d 的 replay_n / comp / excess_ann / _wide / bench_net ----
_spec = importlib.util.spec_from_file_location("d38", Path(__file__).with_name("38d_lowsleeve.py"))
d = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(d)

from factorlab.core import load_panel, FACTORS, composite, smooth_score
from factorlab.factors import load_price_data, neutralize
from factorlab.gate import forward_returns

OUT = Path("data/output_zoo")
HIST_END = "20251231"
DAILY_COLS = ("pct_chg", "amount")


def log(m):
    print(m, flush=True)


# ==================== 风险指标 ====================
def risk_metrics(net, panel):
    bm = d.bench_net(panel, net.index)
    ex = net - bm
    n = len(net)

    ann = float((1 + net).cumprod().iloc[-1] ** (252 / n) - 1)
    vol = float(net.std() * np.sqrt(252))
    sharpe = ann / vol if vol > 0 else np.nan

    te = float(ex.std() * np.sqrt(252))
    ex_ann = d.excess_ann(net, panel)
    ir = ex_ann / te if te > 0 else np.nan

    nav = (1 + net).cumprod()
    mdd = float((nav / nav.cummax() - 1).min())

    ex_nav = (1 + ex).cumprod()
    ex_mdd = float((ex_nav / ex_nav.cummax() - 1).min())

    idx = pd.to_datetime(net.index)
    m_ret = net.groupby(idx.to_period("M")).apply(lambda z: (1 + z).prod() - 1)
    m_ex = ex.groupby(idx.to_period("M")).apply(lambda z: (1 + z).prod() - 1)
    win_ret = float((m_ret > 0).mean())
    win_ex = float((m_ex > 0).mean())

    return {"年化": ann, "波动": vol, "Sharpe": sharpe, "超额": ex_ann, "TE": te,
            "IR": ir, "最大回撤": mdd, "超额最大回撤": ex_mdd,
            "月胜率(绝对)": win_ret, "月胜率(超额)": win_ex}


def rank_ic_fast(f, fwd, mask, min_n=50):
    a = f.where(mask).rank(axis=1)
    b = fwd.where(mask).rank(axis=1)
    s = a.corrwith(b, axis=1)
    cnt = a.notna().sum(axis=1)
    return s.where(cnt >= min_n).dropna()


# ==================== 容量（近似） ====================
def capacity_approx(panel, daily, rp, aums=(1e8, 2e8, 3e8, 5e8)):
    """简化平方根律：impact = 0.7*σ_d*sqrt(Q/ADV) + 3bp
    Q = AUM × 单边换手 × 2（双边交易额）；ADV = 持仓等权日均成交额
    近似点：用平均 ADV、平均 σ，正式测算请用 27 号脚本"""
    amt = d._wide(daily, "amount", panel.idx, panel.cols) * 1000.0   # 千元→元

    advs, tos = [], []
    for date, picks in rp.holdings.items():
        if not picks or date not in amt.index:
            continue
        row = amt.loc[date, picks].dropna()
        if len(row) == 0:
            continue
        advs.append(float(row.mean()))
        tos.append(float(rp.turnover.get(date, 0.0)))

    if not advs:
        log("   ⚠️ 无持仓成交额数据，跳过容量")
        return None

    adv_port = float(np.mean(advs))
    sigma_d = float(rp.net.std())
    years = len(rp.net) / 252.0
    ex0 = d.excess_ann(rp.net, panel)

    rows = []
    for aum in aums:
        extra = 0.0
        for to in tos:
            if to <= 0:
                continue
            q = aum * to * 2.0
            impact = 0.7 * sigma_d * np.sqrt(q / adv_port) + 0.0003
            extra += to * 2.0 * impact
        extra_ann = extra / years
        rows.append({"AUM(亿)": aum / 1e8, "额外成本": extra_ann,
                     "净超额": ex0 - extra_ann})
    return adv_port, rows


# ==================== 主流程 ====================
def main():
    log("① 加载 ...")
    panel = load_panel(hist_end=HIST_END)
    last = str(panel.idx.max())[:10].replace("-", "")
    assert last <= HIST_END, f"越界 {last}"
    daily, basic = load_price_data(daily_cols=DAILY_COLS)

    ret = d._wide(daily, "pct_chg", panel.idx, panel.cols) / 100.0
    amt = d._wide(daily, "amount", panel.idx, panel.cols)
    cmv = d._wide(basic, "circ_mv", panel.idx, panel.cols)
    ln_mv = -np.log(cmv * 1e4)
    ret_a = d.alive_zero(ret)

    illiq = (ret_a.abs() * 1e4) / (amt * 1000.0).replace(0, np.nan)
    amihud20 = neutralize({"amihud": illiq.rolling(20, min_periods=20).mean(), "ln_mv": ln_mv},
                          panel.mask, panel.industry, names=["amihud"], verbose=False)["amihud"]
    fac = {**panel.fac, "amihud": amihud20}
    base_score = composite(panel)
    aug_score = d.comp(panel, fac, FACTORS + ["amihud"])

    # ==================== 检查 1：为什么是 20 日 ====================
    log("\n② 检查 1：持有期 15/20/25/30/40（amihud 固定 win=20）")
    log(f"   {'rebal':>6} {'基准超额':>9} {'+amihud':>9} {'Δ':>8} {'配对t':>8}")
    hold_rows = []
    for rb in (15, 20, 25, 30, 40):
        b = d.replay_n(smooth_score(base_score, rebal=rb), panel, rb)
        r = d.replay_n(smooth_score(aug_score, rebal=rb), panel, rb)
        exb, exa = d.excess_ann(b.net, panel), d.excess_ann(r.net, panel)
        dd = (r.net - b.net).dropna()
        t = float(dd.mean() / dd.std() * np.sqrt(len(dd))) if dd.std() > 0 else np.nan
        log(f"   {rb:>6} {exb:>8.2%} {exa:>8.2%} {exa - exb:>+7.2%} {t:>+7.2f}")
        hold_rows.append({"rebal": rb, "基准超额": round(exb, 4),
                          "加amihud": round(exa, 4), "Δ": round(exa - exb, 4),
                          "t": round(t, 2) if np.isfinite(t) else None})

    log("\n   amihud(win=20) 的 IC 衰减：")
    for h in (1, 5, 10, 20, 40):
        ic = rank_ic_fast(amihud20, forward_returns(panel, h), panel.mask)
        log(f"     h={h:<3} IC {ic.mean():+.4f}  ICIR {ic.mean()/ic.std():+.3f}  days {len(ic)}")

    # ==================== 检查 2：完整风险指标 ====================
    log("\n③ 检查 2：完整风险指标（rb=20）")
    b = d.replay_n(smooth_score(base_score, rebal=20), panel, 20)
    r = d.replay_n(smooth_score(aug_score, rebal=20), panel, 20)
    mb, ma = risk_metrics(b.net, panel), risk_metrics(r.net, panel)
    tb = pd.DataFrame({"base20": mb, "amihud@20": ma})
    tb["差"] = tb["amihud@20"] - tb["base20"]
    log(tb.round(4).to_string())
    log(f"   换手 {b.ann_turnover():.0%} → {r.ann_turnover():.0%}")

    # ==================== 检查 3：容量 ====================
    log("\n④ 检查 3：容量（近似平方根律）")
    for label, rp in (("base20", b), ("amihud@20", r)):
        res = capacity_approx(panel, daily, rp)
        if res is None:
            continue
        adv_port, rows = res
        log(f"   {label}: 持仓等权日均成交额 {adv_port/1e8:.2f} 亿元")
        for x in rows:
            log(f"      AUM {x['AUM(亿)']:>3.0f}亿  额外成本 {x['额外成本']:>6.2%}  "
                f"净超额 {x['净超额']:>+7.2%}")

    # ==================== 输出 ====================
    pd.DataFrame(hold_rows).to_csv(OUT / "amihud_holding.csv", index=False, encoding="utf-8-sig")
    tb.to_csv(OUT / "amihud_risk.csv", encoding="utf-8-sig")
    log(f"\n✅ {OUT}/amihud_holding.csv, amihud_risk.csv")


if __name__ == "__main__":
    main()
