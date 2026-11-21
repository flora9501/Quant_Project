#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
45_overnight.py
隔夜 / 日内 收益分解（Lou-Polk-Skouras 那类 "tug of war"）。

预注册方向（跑完不许改）
  ovn_20 = −Σ20日隔夜收益   +1   隔夜涨得多的，后面差
  idr_20 = +Σ20日日内收益   +1   日内强 = 资金真买

定义
  overnight_t = open_t / pre_close_t − 1
  intraday_t  = close_t / open_t − 1
  用交易所口径的 pre_close，自动处理停牌与除权（不用 close.shift(1)）。

为什么值得单独测
  它和 rev_5 的构造不同：rev_5 用 pct_chg（整日收益），
  这两个只看同一天内部的开盘/收盘结构，理论上与「过去涨了多少」正交。

用法
  .venv/bin/python -u 45_overnight.py [--rebuild]
"""
import argparse
import sys
from datetime import date as _date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from factorlab.core import load_panel
from factorlab.factors import (load_price_data, neutralize, save_cached,
                               load_cached, fill_alive_zero)
from factorlab.gate import evaluate

OUT = Path("data/output_zoo")
OUT.mkdir(parents=True, exist_ok=True)
TRIALS = OUT / "factor_trials.csv"

EXPECT = {"ovn_20": +1, "idr_20": +1}


def log(m):
    print(m, flush=True)


def _wide(df, col, dates, codes):
    return (df.pivot(index="trade_date", columns="ts_code", values=col)
              .reindex(index=dates, columns=codes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--window", type=int, default=20)
    a = ap.parse_args()

    log("① 加载面板与开收盘价...")
    panel = load_panel()
    daily, basic = load_price_data(
        daily_cols=("pct_chg", "open", "close", "pre_close"))

    missing = {"open", "close", "pre_close"} - set(daily.columns)
    if missing:
        raise SystemExit(f"daily 缺列 {missing}；2026 年的文件没有 open，"
                         f"确认 load_price_data 的 years 不含 2026")

    op = _wide(daily, "open", panel.idx, panel.cols)
    cl = _wide(daily, "close", panel.idx, panel.cols)
    pc = _wide(daily, "pre_close", panel.idx, panel.cols)
    cmv = _wide(basic, "circ_mv", panel.idx, panel.cols)
    log(f"   {len(panel.idx)} 天 × {len(panel.cols)} 只")

    log("\n② 拆隔夜 / 日内...")
    ovn = (op / pc.replace(0, np.nan) - 1.0)
    idr = (cl / op.replace(0, np.nan) - 1.0)

    # 单日异常值（送转、复牌）先截尾，避免污染 20 日累计
    ovn = ovn.clip(-0.2, 0.2)
    idr = idr.clip(-0.2, 0.2)

    # 停牌只有 NaN；生命期内按 0 处理（当天没交易 = 没收益）
    ovn = fill_alive_zero(ovn)
    idr = fill_alive_zero(idr)

    # 诊断：A 股隔夜收益应当长期为负
    log(f"   全样本日均 隔夜 {float(np.nanmean(ovn.to_numpy())) * 1e4:+.1f} bp  "
        f"日内 {float(np.nanmean(idr.to_numpy())) * 1e4:+.1f} bp")

    W = a.window
    raw = {}
    raw["ovn_20"] = -ovn.rolling(W, min_periods=int(W * 0.75)).sum()
    raw["idr_20"] = idr.rolling(W, min_periods=int(W * 0.75)).sum()

    cmv_w = cmv
    raw["ln_mv"] = -np.log(cmv_w * 1e4)

    for k in EXPECT:
        v = raw[k]
        inreg = float(v.where(panel.mask).notna().to_numpy().mean()) / \
                float(panel.mask.to_numpy().mean())
        log(f"   {k:<8} 全网格 {v.notna().values.mean():.1%}  在册 {inreg:.1%}")

    log("\n③ 中性化 + gate...")
    rows = []
    for name in EXPECT:
        f = None if a.rebuild else load_cached(name)
        if f is not None:
            log(f"   {name:<8} 缓存命中")
        else:
            f = neutralize(raw, panel.mask, panel.industry,
                           names=[name], verbose=False)[name]
            save_cached(name, f)
            log(f"   {name:<8} 已中性化并落盘")

        r = evaluate(name, f, panel, h=5, with_marginal=True,
                     verbose=False, require_positive=False)
        same_dir = np.sign(r.ic["IC"]) == np.sign(EXPECT[name])
        incr = (np.isfinite(r.marginal_t) and r.marginal_t >= 2.0
                and abs(r.resid_ic) >= 0.015)
        verdict = ("PASS" if (r.max_corr < 0.30 and abs(r.ic["IC"]) >= 0.02
                              and abs(r.ic["ICIR"]) >= 0.30 and same_dir
                              and incr) else "REJECT")

        ls = f"{r.ls_bp:+.0f}bp" if np.isfinite(r.ls_bp) else "n/a"
        log(f"   {name:<8} IC {r.ic['IC']:+.4f}  ICIR {r.ic['ICIR']:+.3f}  "
            f"NW t {r.t_nw:+.2f}  Q5-Q1 {ls}  相关 {r.max_corr:.3f}  "
            f"残差 {r.resid_ic:+.4f}  边际 t {r.marginal_t:+.2f}  → {verdict}")

        rows.append({"因子": name, "IC": round(r.ic["IC"], 4),
                     "ICIR": round(r.ic["ICIR"], 3), "t_NW": round(r.t_nw, 2),
                     "Q5-Q1(bp)": round(r.ls_bp, 0) if np.isfinite(r.ls_bp) else None,
                     "最大相关": round(r.max_corr, 3),
                     "残差IC": round(r.resid_ic, 4),
                     "方向一致": bool(same_dir),
                     "边际t": round(r.marginal_t, 2) if np.isfinite(r.marginal_t) else None,
                     "判决": verdict})

        new = pd.DataFrame([{
            "date": str(_date.today()), "factor": name,
            "IC": round(r.ic["IC"], 4), "ICIR": round(r.ic["ICIR"], 3),
            "t_NW": round(r.t_nw, 2),
            "p_NW": float(f"{r.p_nw:.3g}") if np.isfinite(r.p_nw) else np.nan,
            "resid_ic": round(r.resid_ic, 4), "max_corr": round(r.max_corr, 3),
            "marginal_t": round(r.marginal_t, 2) if np.isfinite(r.marginal_t) else None,
            "marginal_delta": round(r.marginal_delta, 4) if np.isfinite(r.marginal_delta) else None,
            "verdict": verdict}])
        old = pd.read_csv(TRIALS) if TRIALS.exists() else pd.DataFrame()
        pd.concat([old, new], ignore_index=True).to_csv(
            TRIALS, index=False, encoding="utf-8-sig")

    tab = pd.DataFrame(rows).set_index("因子")
    tab.to_csv(OUT / "overnight_gates.csv", encoding="utf-8-sig")
    log("\n" + "=" * 100)
    log("【隔夜 / 日内分解】")
    log("=" * 100)
    log(tab.to_string())
    log(f"\n通过 {int((tab['判决'] == 'PASS').sum())}/{len(tab)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
