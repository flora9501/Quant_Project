#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
38_overnight_intraday.py
隔夜 / 日内收益分解（Lou, Polk & Skouras 2019, RFS）

只用 2018-2025。绝不读 2026（OOS 起点 2026-11-20）。

用法
    .venv/bin/python -u 38_overnight_intraday.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from factorlab.core import (load_panel, FACTORS, composite, smooth_score, replay)
from factorlab.factors import load_price_data, neutralize
from factorlab.gate import evaluate

OUT = Path("data/output_zoo")
OUT.mkdir(parents=True, exist_ok=True)
HIST_END = "20251231"
FACTORS_OI = ["on_20", "id_20"]


def log(m):
    print(m, flush=True)


def _wide(df, col, dates, codes):
    return (df.pivot(index="trade_date", columns="ts_code", values=col)
              .reindex(index=dates, columns=codes))


def alive_zero(x):
    """生命期内 NaN→0（停牌），生命期外保持 NaN（未上市/已退市）"""
    valid = x.notna()
    fwd = valid.astype(np.int8).cummax().astype(bool)
    bwd = valid[::-1].astype(np.int8).cummax()[::-1].astype(bool)
    return x.fillna(0.0).where(fwd & bwd)


def build_oi(panel, daily, basic):
    dates, codes = panel.idx, panel.cols

    ret = _wide(daily, "pct_chg", dates, codes) / 100.0
    o = _wide(daily, "open", dates, codes).where(lambda z: z > 0)
    c = _wide(daily, "close", dates, codes).where(lambda z: z > 0)
    cmv = _wide(basic, "circ_mv", dates, codes)

    intraday = (c / o - 1.0).where(lambda z: z.abs() < 0.5)
    overnight = ((1.0 + ret) / (1.0 + intraday) - 1.0).where(lambda z: z.abs() < 0.5)

    # 恒等式自检：拆开再拼回去必须还原 ret
    recon = (1.0 + overnight) * (1.0 + intraday) - 1.0
    err = float(np.nanmax(np.abs((recon - ret).to_numpy())))
    log(f"   分解恒等式最大误差 {err:.2e}（应 < 1e-9）")
    if err > 1e-6:
        raise RuntimeError("隔夜/日内分解不闭合，先查 open/close/pct_chg")

    on = alive_zero(overnight)
    idr = alive_zero(intraday)

    raw = {}
    # 取负 = 反转假设：过去 20 日隔夜/日内涨得多 → 因子值小 → 预期未来弱
    raw["on_20"] = -np.log1p(on).rolling(20, min_periods=20).sum()
    raw["id_20"] = -np.log1p(idr).rolling(20, min_periods=20).sum()
    raw["ln_mv"] = -np.log(cmv * 1e4)          # 中性化自变量（万元→元）
    return raw, on, idr


def composite_with(panel, names, extra=None):
    """等权复合；extra 会给到与每个因子相同的 1 份权重（即 1/5）"""
    parts = [panel.fac[n] for n in names]
    if extra is not None:
        parts = parts + [extra]
    out = sum(d.fillna(0.0) for d in parts)
    valid = sum(d.notna().astype(int) for d in parts) > 0
    return (out / len(parts)).where(valid)


def marginal_correct(panel, f):
    """n+1 等权：新因子占 1/5。这是正确做法。"""
    base = replay(smooth_score(composite_with(panel, FACTORS)), panel)
    aug = replay(smooth_score(composite_with(panel, FACTORS, extra=f)), panel)
    d = (aug.net - base.net).dropna()
    t = float(d.mean() / d.std() * np.sqrt(len(d))) if len(d) > 100 and d.std() > 0 else np.nan
    return t, float(aug.ann() - base.ann()), base, aug


def marginal_old(panel, f):
    """旧口径：composite(4因子均值) + f，新因子方差约占 60%。仅用于对照。"""
    base = replay(smooth_score(composite_with(panel, FACTORS)), panel)
    aug = replay(smooth_score(composite_with(panel, FACTORS) + f.fillna(0.0)), panel)
    d = (aug.net - base.net).dropna()
    t = float(d.mean() / d.std() * np.sqrt(len(d))) if len(d) > 100 and d.std() > 0 else np.nan
    return t, float(aug.ann() - base.ann())


def main():
    log("① 加载面板（硬截止 20251231）...")
    panel = load_panel(hist_end=HIST_END)
    last = str(panel.idx.max())[:10].replace("-", "")
    assert last <= HIST_END, f"面板越界到 {last}，停止"
    log(f"   {len(panel.idx)} 天 × {len(panel.cols)} 只；最后一天 {last}")

    daily, basic = load_price_data(daily_cols=("pct_chg", "open", "close"))
    log(f"   daily {daily.shape}（列：{list(daily.columns)}）")

    log("\n② 构造隔夜 / 日内因子...")
    raw, on, idr = build_oi(panel, daily, basic)
    for k in ("on_20", "id_20"):
        log(f"   {k:<8} 覆盖 {raw[k].notna().values.mean():.1%}  "
            f"截面 std 中位数 {np.nanmedian(raw[k].std(axis=1)):.4f}")

    log("\n③ 中性化（行业 + ln市值，FWL）...")
    neu = neutralize(raw, panel.mask, panel.industry, names=FACTORS_OI, verbose=False)

    rows, results = [], {}
    for name in FACTORS_OI:
        f = neu[name]
        

        # h=5 主评估（含五分组、残差 IC、相关性）
        r = evaluate(name, f, panel, h=5, with_marginal=False, verbose=True)
        results[name] = r

        # IC 衰减（用于简历里的 1/5/10/20 日曲线）
        decay = {}
        for h in (1, 5, 10, 20):
            rr = evaluate(name, f, panel, h=h, with_marginal=False, verbose=False)
            decay[h] = rr.ic["IC"]

        # 两种 marginal
        t_new, d_new, base, aug = marginal_correct(panel, f)
        t_old, d_old = marginal_old(panel, f)
        log(f"\n   边际（等权 5 因子，正确）: t {t_new:+.2f}  Δ {d_new:+.2%}")
        log(f"   边际（旧口径 composite+f）: t {t_old:+.2f}  Δ {d_old:+.2%}")
        log(f"   换手 {base.ann_turnover():.0%} → {aug.ann_turnover():.0%}")
        log(f"   IC 衰减 " + "  ".join(f"h={h}:{v:+.4f}" for h, v in decay.items()))

        rows.append({
            "因子": name,
            "覆盖率": round(float(raw[name].notna().values.mean()), 3),
            "IC_h5": round(r.ic["IC"], 4),
            "ICIR": round(r.ic["ICIR"], 3),
            "t_NW": round(r.t_nw, 2),
            "Q5-Q1_bp": round(r.ls_bp, 0) if np.isfinite(r.ls_bp) else None,
            "LS_t_NW": round(r.ls_t_nw, 2) if np.isfinite(r.ls_t_nw) else None,
            "分年同号": bool((r.yearly > 0).all() or (r.yearly < 0).all()),
            "最大相关": round(r.max_corr, 3),
            "残差IC": round(r.resid_ic, 4),
            "边际t_等权": round(t_new, 2) if np.isfinite(t_new) else None,
            "边际Δ_等权": round(d_new, 3) if np.isfinite(d_new) else None,
            "边际t_旧": round(t_old, 2) if np.isfinite(t_old) else None,
            "IC1": round(decay[1], 4),
            "IC10": round(decay[10], 4),
            "IC20": round(decay[20], 4),
        })

    tab = pd.DataFrame(rows).set_index("因子")
    log("\n" + "=" * 100)
    log(tab.to_string())
    tab.to_csv(OUT / "overnight_intraday.csv", encoding="utf-8-sig")
    log(f"\n✅ 写出 {OUT}/overnight_intraday.csv")


if __name__ == "__main__":
    main()
