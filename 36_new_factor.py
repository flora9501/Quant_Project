#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
36_new_factor.py
新因子验证台。与 35（论文复现）完全分开：
  - 35 的分母只放"已发表论文"，用来算复制率
  - 36 的分母放"你试过的所有变体"，用来算你自己的多重检验

用法
  .venv/bin/python -u 36_new_factor.py nf_xxx        # 验证一个
  .venv/bin/python -u 36_new_factor.py --report      # 看累计尝试与 BH
"""
import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))

from factorlab.core import load_panel
from factorlab.factors import (load_price_data, neutralize, fill_alive_zero,
                               load_cached, save_cached)
from factorlab.gate import evaluate

TRIALS = Path("data/output_zoo/factor_trials.csv")

# 预先声明方向：+1 = 越大越好，-1 = 越小越好。先写死，跑完不许改。
EXPECT = {
    "nf_xxx": +1,
}


def _wide(df, col, dates, codes):
    return (df.pivot(index="trade_date", columns="ts_code", values=col)
              .reindex(index=dates, columns=codes))


def build(name, panel, daily, basic):
    """返回 dict：{因子名: 原始宽表, 'ln_mv': 规模} —— 中性化必须要 ln_mv"""
    dates, codes = panel.idx, panel.cols

    ret = fill_alive_zero(_wide(daily, "pct_chg", dates, codes) / 100.0)
    turn = _wide(basic, "turnover_rate", dates, codes)
    cmv = _wide(basic, "circ_mv", dates, codes)      # 万元
    amt = _wide(daily, "amount", dates, codes)       # 千元
    ln_mv = -np.log(cmv * 1e4)

    if name == "nf_xxx":
        # ↓↓↓ 你的因子只写在这里。原始值，不 winsorize、不 z-score ↓↓↓
        f = -ret.rolling(10, min_periods=10).sum()
        # ↑↑↑ 只写到这里 ↑↑↑
    else:
        raise KeyError(f"没定义 {name}；在 build() 里加分支，并在 EXPECT 里声明方向")

    return {name: f, "ln_mv": ln_mv}


def bh(ps):
    p = np.asarray(ps, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        _, qq, _, _ = multipletests(p[ok], method="fdr_bh")
        q[ok] = qq
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", help="因子名")
    ap.add_argument("--h", type=int, default=5)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()

    # ---------- 只出报表 ----------
    if a.report:
        if not TRIALS.exists():
            raise SystemExit("还没有试验记录")
        d = pd.read_csv(TRIALS)
        d["p_BH"] = bh(d["p_NW"].to_numpy())
        print(d.sort_values("p_NW").to_string(index=False))
        n = len(d)
        print(f"\n累计尝试 {n} 个变体；BH(q=0.05) 存活 {(d['p_BH'] < 0.05).sum()} 个")
        print("这个分母才是你真实的假阳性率来源，也是写进文档的那个数。")
        return

    if not a.name:
        raise SystemExit("给个因子名，或 --report")
    if a.name not in EXPECT:
        raise SystemExit(f"先在 EXPECT 里声明 {a.name} 的方向")

    panel = load_panel()
    daily, basic = load_price_data(daily_cols=("pct_chg", "amount"))
    raw = build(a.name, panel, daily, basic)

    cov = float(raw[a.name].notna().values.mean())
    print(f"覆盖率（全网格）{cov:.1%}；在册覆盖率约 {cov / float(panel.mask.values.mean()):.1%}")

    f = None if a.rebuild else load_cached(a.name)
    if f is not None:
        print("缓存命中")
    else:
        f = neutralize(raw, panel.mask, panel.industry,
                       names=[a.name], verbose=False)[a.name]
        save_cached(a.name, f)
        print("已中性化并落盘")

    # 方向交给 EXPECT 判断，不用 require_positive
    r = evaluate(a.name, f, panel, h=a.h, with_marginal=True,
                 verbose=True, require_positive=False)

    same_dir = np.sign(r.ic["IC"]) == np.sign(EXPECT[a.name])

    # ---- 三关判定 ----
    k1 = r.max_corr < 0.30
    k2 = abs(r.resid_ic) >= 0.015
    k3 = np.isfinite(r.marginal_t) and r.marginal_t >= 2.0
    ok = k1 and k2 and k3 and same_dir and r.t_nw >= 3.0

    print(f"\n── 新因子判定 ──")
    print(f"  方向符合预期 : {'✅' if same_dir else '❌ 符号反了，先查构造'}")
    print(f"  与现有因子正交(max_corr<0.30): {'✅' if k1 else '❌'} ({r.max_corr:.3f})")
    print(f"  残差 IC>=0.015              : {'✅' if k2 else '❌'} ({r.resid_ic:+.4f})")
    print(f"  组合边际 t>=2.0             : {'✅' if k3 else '❌'} ({r.marginal_t:+.2f})")
    print(f"  NW t>=3.0                   : {'✅' if r.t_nw >= 3.0 else '❌'} ({r.t_nw:+.2f})")
    print(f"  → {'PASS（可进候选池）' if ok else 'REJECT'}")

    # ---- 追加试验日志（失败的也记，这才是分母）----
    TRIALS.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{
        "date": str(date.today()), "factor": a.name,
        "IC": round(r.ic["IC"], 4), "ICIR": round(r.ic["ICIR"], 3),
        "t_NW": round(r.t_nw, 2), "p_NW": float(f"{r.p_nw:.3g}"),
        "resid_ic": round(r.resid_ic, 4), "max_corr": round(r.max_corr, 3),
        "marginal_t": round(r.marginal_t, 2) if np.isfinite(r.marginal_t) else None,
        "marginal_delta": round(r.marginal_delta, 4) if np.isfinite(r.marginal_delta) else None,
        "verdict": "PASS" if ok else "REJECT",
    }])
    if TRIALS.exists():
        old = pd.read_csv(TRIALS)
        row = pd.concat([old, row], ignore_index=True)
    row.to_csv(TRIALS, index=False, encoding="utf-8-sig")
    print(f"\n已记入 {TRIALS}（共 {len(row)} 条）")


if __name__ == "__main__":
    main()
