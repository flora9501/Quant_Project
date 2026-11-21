#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""38b_oi_v2.py — 隔夜/日内 v2：动量方向 + 尾部 + 结构对比。只用 2018-2025。"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from factorlab.core import load_panel, FACTORS, smooth_score, replay
from factorlab.factors import load_price_data, neutralize
from factorlab.gate import evaluate, forward_returns

OUT = Path("data/output_zoo"); OUT.mkdir(parents=True, exist_ok=True)
HIST_END = "20251231"


def log(m): print(m, flush=True)


def _wide(df, col, dates, codes):
    return (df.pivot(index="trade_date", columns="ts_code", values=col)
              .reindex(index=dates, columns=codes))


def alive_zero(x):
    v = x.notna()
    fwd = v.astype(np.int8).cummax().astype(bool)
    bwd = v[::-1].astype(np.int8).cummax()[::-1].astype(bool)
    return x.fillna(0.0).where(fwd & bwd)


def build(panel, daily, basic):
    dates, codes = panel.idx, panel.cols
    ret = _wide(daily, "pct_chg", dates, codes) / 100.0
    o = _wide(daily, "open", dates, codes).where(lambda z: z > 0)
    c = _wide(daily, "close", dates, codes).where(lambda z: z > 0)
    cmv = _wide(basic, "circ_mv", dates, codes)

    intraday = (c / o - 1.0).where(lambda z: z.abs() < 0.5)
    overnight = ((1.0 + ret) / (1.0 + intraday) - 1.0).where(lambda z: z.abs() < 0.5)
    log(f"   恒等式误差 {float(np.nanmax(np.abs(((1+overnight)*(1+intraday)-1-ret).to_numpy()))):.2e}")

    on = alive_zero(overnight); idr = alive_zero(intraday)

    raw = {}
    # 不取负 = 动量方向（按 v1 实测）
    raw["on_20"] = np.log1p(on).rolling(20, min_periods=20).sum()
    raw["id_20"] = np.log1p(idr).rolling(20, min_periods=20).sum()
    # 尾部：过去 20 日隔夜累计收益最差 10%
    cum_on = np.log1p(on).rolling(20, min_periods=20).sum()
    raw["on_tail"] = -(cum_on.rank(axis=1, pct=True) <= 0.10).astype(float).where(cum_on.notna())
    raw["ln_mv"] = -np.log(cmv * 1e4)
    return raw


def rank_ic_fast(f, fwd, mask, min_n=50):
    """向量化 rank IC：corrwith 比 groupby.apply 快 30 倍"""
    a = f.where(mask).rank(axis=1)
    b = fwd.where(mask).rank(axis=1)
    return a.corrwith(b, axis=1).where(a.notna().sum(axis=1) >= min_n).dropna()


def decay(f, panel, hs=(1, 5, 10, 20)):
    return {h: float(rank_ic_fast(f, forward_returns(panel, h), panel.mask).mean())
            for h in hs}


def comp(panel, fac, names):
    parts = [fac[n] for n in names]
    out = sum(d.fillna(0.0) for d in parts)
    valid = sum(d.notna().astype(int) for d in parts) > 0
    return (out / len(parts)).where(valid)


def ann_excess(net, panel):
    bm = panel.fwd.where(panel.mask).mean(axis=1).reindex(net.index).fillna(0.0)
    e = net - bm
    return float((1 + e).cumprod().iloc[-1] ** (252 / len(e)) - 1)


def main():
    panel = load_panel(hist_end=HIST_END)
    assert str(panel.idx.max())[:10].replace("-", "") <= HIST_END
    log(f"{len(panel.idx)} 天 × {len(panel.cols)} 只，末日 {panel.idx.max()}")

    daily, basic = load_price_data(daily_cols=("pct_chg", "open", "close"))
    raw = build(panel, daily, basic)

    log("\n① 中性化 on_20 / id_20 / on_tail ...")
    neu = neutralize(raw, panel.mask, panel.industry,
                     names=["on_20", "id_20", "on_tail"], verbose=False)

    log("\n② 单因子（h=5 完整 + 快衰减）")
    rows = []
    for name in ("on_20", "id_20", "on_tail"):
        f = neu[name]
        r = evaluate(name, f, panel, h=5, with_marginal=False, verbose=False)
        d = decay(f, panel)
        log(f"   {name:<8} IC {r.ic['IC']:+.4f} NW t {r.t_nw:+.2f} "
            f"残差 {r.resid_ic:+.4f} 相关 {r.max_corr:.3f} "
            f"| 衰减 " + " ".join(f"h{h}:{v:+.4f}" for h, v in d.items()))
        rows.append({"因子": name, "IC": round(r.ic["IC"], 4),
                     "t_NW": round(r.t_nw, 2), "残差IC": round(r.resid_ic, 4),
                     "最大相关": round(r.max_corr, 3),
                     **{f"IC_h{h}": round(v, 4) for h, v in d.items()}})

    log("\n③ 结构对比：拆开 rev_5 是否更好？")
    fac = {**panel.fac, **neu}
    variants = {
        "A_base(rev_5合并)":      FACTORS,
        "B_split(on+id)":         ["on_20", "id_20", "turn_20", "vol_20", "ep"],
        "C_split+tail":           ["on_20", "id_20", "on_tail", "turn_20", "vol_20", "ep"],
        "D_base+on_tail":         FACTORS + ["on_tail"],
    }
    for label, names in variants.items():
        s = comp(panel, fac, names)
        sm = smooth_score(s)
        ic = float(rank_ic_fast(sm, forward_returns(panel, 5), panel.mask).mean())
        rp = replay(sm, panel)
        ex = ann_excess(rp.net, panel)
        log(f"   {label:<22} 平滑IC {ic:+.4f}  年化 {rp.ann():.2%}  超额 {ex:+.2%}  "
            f"换手 {rp.ann_turnover():.0%}")
        rows.append({"因子": label, "IC": round(ic, 4), "年化": round(rp.ann(), 4),
                     "超额": round(ex, 4), "换手": round(rp.ann_turnover(), 4)})

    pd.DataFrame(rows).to_csv(OUT / "oi_v2.csv", index=False, encoding="utf-8-sig")
    log(f"\n✅ {OUT}/oi_v2.csv")


if __name__ == "__main__":
    main()
