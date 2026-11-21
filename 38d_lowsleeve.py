#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
38d_lowsleeve.py
低频 sleeve：5 / 10 / 20 日持有对比 + amihud / high_52w 在 20 日上的增量。

路线 A：不改 factorlab/core.py，本地带一份 replay_n 副本。
⚠️ 副本仅用于本实验；跑完要么合并回 core.replay，要么删除（坑 #17）。

只用 2018-2025，绝不读 2026（OOS 起点 2026-11-20）。

用法
    .venv/bin/python -u 38d_lowsleeve.py
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from factorlab.core import (load_panel, FACTORS, composite, smooth_score, replay,
                            quota_select, COST, COST_MULT, TOP_N, THRESH)
from factorlab.factors import load_price_data, neutralize
from factorlab.gate import forward_returns

OUT = Path("data/output_zoo")
OUT.mkdir(parents=True, exist_ok=True)
HIST_END = "20251231"
REBALS = (5, 10, 20)
DAILY_COLS = ("pct_chg", "open", "close", "amount")


# ==================== 本地副本：core.replay 唯一改动是 i % rebal ====================
@dataclass
class ReplayN:
    net: pd.Series
    gross: pd.Series
    turnover: pd.Series
    holdings: dict = field(default_factory=dict)
    ind_dev: dict = field(default_factory=dict)
    rebal: int = 5

    def ann(self):
        r = self.net
        return (1 + r).cumprod().iloc[-1] ** (252 / len(r)) - 1

    def ann_turnover(self):
        return float(self.turnover.replace(0, np.nan).mean() * (252 / self.rebal))


def replay_n(score, panel, rebal, top_n=TOP_N, thresh=THRESH,
             cost=COST, cost_mult=COST_MULT):
    w_live = pd.Series(0.0, index=panel.cols)
    held = []
    gross = pd.Series(0.0, index=panel.idx)
    to_s = pd.Series(0.0, index=panel.idx)
    holdings, dev = {}, {}

    for i, d in enumerate(panel.idx):
        if i % rebal == 0:                                     # ← 唯一改动
            s = score.loc[d].where(panel.mask.loc[d]).dropna()
            if len(s) >= top_n:
                picks = quota_select(s, panel.industry, held, top_n, thresh)
                if picks:
                    w_new = (pd.Series(1.0 / len(picks), index=picks)
                               .reindex(panel.cols).fillna(0.0))
                    tot = w_live.sum()
                    w_drift = (w_live / tot) if tot > 0 else w_live
                    to_s.loc[d] = float((w_new - w_drift).abs().sum() / 2)

                    pw = (pd.Series(1.0 / len(picks), index=picks)
                            .groupby(panel.industry.reindex(picks)).sum())
                    valid = panel.mask.loc[d]
                    bw = panel.industry.reindex(valid[valid].index).value_counts(normalize=True)
                    dev[d] = float((pw.reindex(bw.index).fillna(0) - bw).abs().max())

                    holdings[d] = picks
                    held, w_live = picks, w_new

        r = panel.fwd.loc[d].fillna(0.0)
        gross.loc[d] = float((w_live * r).sum())
        w_live = w_live * (1 + r)

    net = (gross - to_s * cost * cost_mult).iloc[:-1]
    return ReplayN(net=net, gross=gross, turnover=to_s,
                   holdings=holdings, ind_dev=dev, rebal=rebal)


# ==================== 工具 ====================
def log(m):
    print(m, flush=True)


def _wide(df, col, dates, codes):
    return (df.pivot(index="trade_date", columns="ts_code", values=col)
              .reindex(index=dates, columns=codes))


def alive_zero(x):
    v = x.notna()
    f = v.astype(np.int8).cummax().astype(bool)
    b = v[::-1].astype(np.int8).cummax()[::-1].astype(bool)
    return x.fillna(0.0).where(f & b)


def build_extra(panel, daily, basic):
    dates, codes = panel.idx, panel.cols
    ret_raw = _wide(daily, "pct_chg", dates, codes) / 100.0
    amt = _wide(daily, "amount", dates, codes)      # 千元
    cmv = _wide(basic, "circ_mv", dates, codes)     # 万元

    ret = alive_zero(ret_raw)

    # Amihud：|r| / 成交额（元），20 日均
    illiq = (ret.abs() * 1e4) / (amt * 1000.0).replace(0, np.nan)
    amihud = illiq.rolling(20, min_periods=20).mean()

    # 52 周高点：pct_chg 累乘作复权价代理，比值为无量纲
    px = (1 + ret).cumprod()
    high52 = -(px / px.rolling(250, min_periods=200).max())

    return {"amihud": amihud, "high_52w": high52, "ln_mv": -np.log(cmv * 1e4)}


def comp(panel, fac, names):
    """等权复合：n 个分量各占 1/n"""
    parts = [fac[n] for n in names]
    out = sum(d.fillna(0.0) for d in parts)
    valid = sum(d.notna().astype(int) for d in parts) > 0
    return (out / len(parts)).where(valid)


def bench_net(panel, index):
    return panel.fwd.where(panel.mask).mean(axis=1).reindex(index).fillna(0.0)


def excess_ann(net, panel):
    """几何超额：(nav_p / nav_b)^(252/n) - 1"""
    bm = bench_net(panel, net.index)
    n = len(net)
    return float(((1 + net).cumprod().iloc[-1] /
                  (1 + bm).cumprod().iloc[-1]) ** (252 / n) - 1)


def gross_ann(rp):
    g = rp.gross.iloc[:-1]
    return float((1 + g).cumprod().iloc[-1] ** (252 / len(g)) - 1)


def quick_ic(f, panel, h=20):
    """向量化 rank IC，用于自动对齐因子方向"""
    a = f.where(panel.mask).rank(axis=1)
    b = forward_returns(panel, h).where(panel.mask).rank(axis=1)
    s = a.corrwith(b, axis=1).dropna()
    return float(s.mean()) if len(s) else np.nan


# ==================== 主流程 ====================
def main():
    log("① 加载（硬截止 20251231）...")
    panel = load_panel(hist_end=HIST_END)
    last = str(panel.idx.max())[:10].replace("-", "")
    assert last <= HIST_END, f"越界 {last}"
    log(f"   {len(panel.idx)} 天 × {len(panel.cols)} 只，末日 {last}")

    # ---- 副本一致性自检（不通过就停）----
    sm5 = smooth_score(composite(panel), rebal=5)
    a = replay(sm5, panel)
    b = replay_n(sm5, panel, 5)
    diff = float((a.net - b.net).abs().max())
    assert diff < 1e-12, f"replay_n 与 core.replay 不一致 max|Δ|={diff:.3e}"
    log(f"   ✅ replay_n 与 core.replay 逐位一致（max|Δ|={diff:.1e}）")

    # ---- 拉数据 + 列自检 ----
    daily, basic = load_price_data(daily_cols=DAILY_COLS)
    miss = set(DAILY_COLS) - set(daily.columns)
    assert not miss, f"daily 缺列 {miss}（实际 {list(daily.columns)}）"
    log(f"   daily {daily.shape} 列 {list(daily.columns)}")

    raw = build_extra(panel, daily, basic)

    log("\n② 中性化 amihud / high_52w ...")
    neu = neutralize(raw, panel.mask, panel.industry,
                     names=["amihud", "high_52w"], verbose=False)
    for n in ("amihud", "high_52w"):
        log(f"   {n:<9} 覆盖率 {neu[n].notna().values.mean():.1%}")

    rows = []

    # ---- ③ 持有期对比 ----
    log("\n③ 持有期对比（4 因子等权）")
    log(f"   {'rebal':>6} {'年化':>9} {'超额':>9} {'毛利':>9} {'成本':>9} {'换手':>8}")
    base_score = composite(panel)
    base20 = None
    for rb in REBALS:
        sm = smooth_score(base_score, rebal=rb)
        rp = replay_n(sm, panel, rb)
        ex = excess_ann(rp.net, panel)
        g = gross_ann(rp)
        log(f"   {rb:>6} {rp.ann():>8.2%} {ex:>8.2%} {g:>8.2%} "
            f"{g - rp.ann():>8.2%} {rp.ann_turnover():>7.0%}")
        rows.append({"配置": f"4因子@{rb}日", "年化": round(rp.ann(), 4),
                     "超额": round(ex, 4), "毛利": round(g, 4),
                     "成本": round(g - rp.ann(), 4),
                     "换手": round(rp.ann_turnover(), 4)})
        if rb == 20:
            base20 = rp

    # ---- ④ 20 日持有下的候选增量 ----
    log("\n④ 20 日持有：amihud / high_52w 是否带来增量")
    ex20 = excess_ann(base20.net, panel)
    log(f"   基准（4 因子@20）超额 {ex20:+.2%}")

    for name in ("amihud", "high_52w"):
        ic = quick_ic(neu[name], panel, h=20)
        sign = +1 if (np.isfinite(ic) and ic > 0) else -1
        f = sign * neu[name]                                   # 自动对齐方向
        log(f"\n   {name}: IC(h=20) {ic:+.4f} → sign {sign:+d}")

        fac2 = {**panel.fac, name: f}
        sm = smooth_score(comp(panel, fac2, FACTORS + [name]), rebal=20)
        rp = replay_n(sm, panel, 20)

        dd = (rp.net - base20.net).dropna()
        t = float(dd.mean() / dd.std() * np.sqrt(len(dd))) if dd.std() > 0 else np.nan
        ex = excess_ann(rp.net, panel)
        log(f"   年化 {rp.ann():>7.2%}  超额 {ex:>+7.2%}  Δ超额 {ex - ex20:>+6.2%}  "
            f"换手 {rp.ann_turnover():>4.0%}  配对t {t:+.2f}")
        rows.append({"配置": f"4因子+{name}@20日", "年化": round(rp.ann(), 4),
                     "超额": round(ex, 4), "Δ超额": round(ex - ex20, 4),
                     "换手": round(rp.ann_turnover(), 4),
                     "配对t": round(t, 2) if np.isfinite(t) else None})

    pd.DataFrame(rows).to_csv(OUT / "lowsleeve.csv", index=False, encoding="utf-8-sig")
    log(f"\n✅ {OUT}/lowsleeve.csv")


if __name__ == "__main__":
    main()
