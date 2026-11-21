#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
45c_overnight_v2.py
隔夜/日内因子 —— **事后方向修正版**（post-hoc，需独立 OOS 验证）

为什么要改方向
  45 号预注册：ovn_20 = −Σ隔夜 → 期望 IC>0；idr_20 = +Σ日内 → 期望 IC>0。
  实测两者都是显著负 IC，方向反了。
  45b 诊断证明涨停板不是主因（相关 |0.05–0.07|），信号是真的。
  修正后的方向与 Lou-Polk-Skouras (2019) 的 tug-of-war 一致：
      隔夜收益 = 动量（机构收盘买入 → 次日高开 → 持续）
      日内收益 = 反转（散户盘中追高 → 次日回落）
  故此为**理论驱动的事后修正**，不是数据挖掘。但必须在 RESEARCH_LOG 留痕。

本版因子（预注册，跑完不许再改）
  ovn_mom = +Σ20日隔夜收益（剔除 |pct_chg|>=9.5% 的日子）   +1
  idr_rev = −Σ20日日内收益（同上）                          +1

用法
  .venv/bin/python -u 45c_overnight_v2.py [--rebuild]
"""
import argparse
import sys
from datetime import date as _date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from factorlab.core import (load_panel, composite, smooth_score, replay)
from factorlab.factors import (load_price_data, neutralize, save_cached,
                               load_cached, fill_alive_zero)
from factorlab.gate import evaluate

OUT = Path("data/output_zoo")
OUT.mkdir(parents=True, exist_ok=True)
TRIALS = OUT / "factor_trials.csv"

EXPECT = {"ovn_mom": +1, "idr_rev": +1}
LIMIT = 0.095
W = 20


def log(m):
    print(m, flush=True)


def _wide(df, col, dates, codes):
    return (df.pivot(index="trade_date", columns="ts_code", values=col)
              .reindex(index=dates, columns=codes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--window", type=int, default=W)
    a = ap.parse_args()
    W_ = a.window
    mp = int(W_ * 0.75)

    log("① 加载面板与开收盘价...")
    panel = load_panel()
    daily, basic = load_price_data(
        daily_cols=("pct_chg", "open", "close", "pre_close"))
    op = _wide(daily, "open", panel.idx, panel.cols)
    cl = _wide(daily, "close", panel.idx, panel.cols)
    pc = _wide(daily, "pre_close", panel.idx, panel.cols)
    pct = _wide(daily, "pct_chg", panel.idx, panel.cols) / 100.0
    cmv = _wide(basic, "circ_mv", panel.idx, panel.cols)
    log(f"   {len(panel.idx)} 天 × {len(panel.cols)} 只")

    log("\n② 拆隔夜/日内 + 剔除涨跌停...")
    ovn = fill_alive_zero((op / pc.replace(0, np.nan) - 1).clip(-0.2, 0.2))
    idr = fill_alive_zero((cl / op.replace(0, np.nan) - 1).clip(-0.2, 0.2))
    limit = (pct.abs() >= LIMIT) & pct.notna()
    log(f"   涨跌停占比 {float(limit.to_numpy().mean()):.2%}"
        f"（构造时剔除）")

    raw = {
        "ovn_mom": ovn.where(~limit).rolling(W_, min_periods=mp).sum(),
        "idr_rev": -idr.where(~limit).rolling(W_, min_periods=mp).sum(),
    }
    raw["ln_mv"] = -np.log(cmv * 1e4)

    for k in EXPECT:
        v = raw[k]
        full = float(v.notna().values.mean())
        inreg = float(v.where(panel.mask).notna().to_numpy().mean()) \
            / float(panel.mask.to_numpy().mean())
        log(f"   {k:<8} 全网格 {full:.1%}  在册 {inreg:.1%}")

    # 两个因子的互相关（它们共享同一个总收益，很可能相关）
    x, y = raw["ovn_mom"], raw["idr_rev"]
    m = pd.concat([x.where(panel.mask).stack().rename("a"),
                   y.where(panel.mask).stack().rename("b")],
                  axis=1).dropna()
    if not m.empty:
        c = (m.groupby(level=0).apply(lambda z: z.a.corr(z.b))
              .dropna().mean())
        log(f"   ovn_mom 与 idr_rev 的截面相关：{c:+.3f}")

    log("\n③ 中性化...")
    fitted = {}
    for name in EXPECT:
        f = None if a.rebuild else load_cached(name)
        if f is not None:
            log(f"   {name:<8} 缓存命中")
        else:
            f = neutralize(raw, panel.mask, panel.industry,
                           names=[name], verbose=False)[name]
            save_cached(name, f)
            log(f"   {name:<8} 已中性化并落盘")
        fitted[name] = f

    log("\n④ gate（含 80% 权重的组合边际）...")
    rows = []
    for name, f in fitted.items():
        r = evaluate(name, f, panel, h=5, with_marginal=True,
                     verbose=False, require_positive=False)
        same_dir = np.sign(r.ic["IC"]) == np.sign(EXPECT[name])
        incr = (np.isfinite(r.marginal_t) and r.marginal_t >= 2.0
                and abs(r.resid_ic) >= 0.015)
        verdict = ("PASS" if (r.max_corr < 0.30 and abs(r.ic["IC"]) >= 0.02
                              and abs(r.ic["ICIR"]) >= 0.30 and same_dir
                              and incr) else "REJECT")
        rows.append(dict(name=name, r=r, same_dir=same_dir, incr=incr,
                         verdict=verdict))

    log("\n⑤ 公平权重复检（20%）...")
    base = replay(smooth_score(composite(panel)), panel)
    base_ann, base_to = base.ann(), base.ann_turnover()
    log(f"   基准：年化 {base_ann:.2%}  换手 {base_to:.0%}")
    for row in rows:
        name, f = row["name"], fitted[row["name"]]
        aug = replay(smooth_score(composite(panel) * 0.8 + f.fillna(0.0) * 0.2),
                     panel)
        d = (aug.net - base.net).dropna()
        t20 = (float(d.mean() / d.std() * np.sqrt(len(d)))
               if d.std() > 0 else np.nan)
        row["t20"] = t20
        row["d20"] = (aug.ann() - base_ann) * 100
        row["to20"] = aug.ann_turnover()
        log(f"   {name:<8} 年化 {aug.ann():.2%}  Δ {row['d20']:+.2f}pp  "
            f"t {t20:+.2f}  换手 {row['to20']:.0%}")

    log("\n" + "=" * 108)
    log("【隔夜/日内 · 修正方向】")
    log("=" * 108)
    out = []
    for row in rows:
        r = row["r"]
        out.append({
            "因子": row["name"],
            "IC": round(r.ic["IC"], 4),
            "ICIR": round(r.ic["ICIR"], 3),
            "t_NW": round(r.t_nw, 2),
            "Q5-Q1": round(r.ls_bp, 0) if np.isfinite(r.ls_bp) else None,
            "最大相关": round(r.max_corr, 3),
            "残差IC": round(r.resid_ic, 4),
            "方向一致": row["same_dir"],
            "边际t(80%)": round(r.marginal_t, 2)
                          if np.isfinite(r.marginal_t) else None,
            "Δ20%": round(row["d20"], 2),
            "t(20%)": round(row["t20"], 2) if np.isfinite(row["t20"]) else None,
            "换手20%": round(row["to20"], 3),
            "判决": row["verdict"],
            "增量(20%)": "✅" if (np.isfinite(row["t20"]) and row["t20"] >= 2
                               and row["d20"] > 0) else "❌",
        })
    tab = pd.DataFrame(out).set_index("因子")
    log(tab.to_string())
    tab.to_csv(OUT / "overnight_v2_gates.csv", encoding="utf-8-sig")

    log("\n【口径警告】")
    log("  1. 本版方向为 post-hoc 修正（预注册时是反的），必须在 RESEARCH_LOG 留痕。")
    log("  2. 80% 权重的边际 t 是旧的不公平口径；t(20%) 才是公平比较。")
    log("  3. 结论需独立样本（2026 OOS）验证，或至少按年切片看稳定性。")

    # 追加试验日志（新记录，不改旧的）
    new = pd.DataFrame([{
        "date": str(_date.today()), "factor": row["name"],
        "IC": round(row["r"].ic["IC"], 4),
        "ICIR": round(row["r"].ic["ICIR"], 3),
        "t_NW": round(row["r"].t_nw, 2),
        "p_NW": float(f"{row['r'].p_nw:.3g}")
                  if np.isfinite(row["r"].p_nw) else np.nan,
        "resid_ic": round(row["r"].resid_ic, 4),
        "max_corr": round(row["r"].max_corr, 3),
        "marginal_t": round(row["t20"], 2) if np.isfinite(row["t20"]) else None,
        "marginal_delta": round(row["d20"], 4),
        "verdict": row["verdict"],
    } for row in rows])
    old = pd.read_csv(TRIALS) if TRIALS.exists() else pd.DataFrame()
    pd.concat([old, new], ignore_index=True).to_csv(
        TRIALS, index=False, encoding="utf-8-sig")
    log(f"\n试验日志累计 {len(pd.concat([old, new], ignore_index=True))} 条")
    return 0


if __name__ == "__main__":
    sys.exit(main())
