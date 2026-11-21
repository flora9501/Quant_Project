#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
46_regime.py
状态依赖：因子 IC 与组合超额在不同市场环境下的差异。

横截面条件化（按股票切）在 40 号已完成；本脚本做时间序列那一半。

三个环境变量（按日期三分位切）
  mkt_vol20   等权市场 20 日已实现波动（年化）
  mkt_ret20   等权市场 20 日累计收益
  disp20      20 日平均的横截面收益离散度

输出
  A. 各因子 IC × 环境
  B. 各因子 NW t × 环境
  C. 组合超额（年化）× 环境 —— 这张最实用

注意
  1. 分位点是**全样本**切的（in-sample），只作描述，不能当成可交易信号。
  2. 相邻日期环境高度自相关，t 会偏高，只用于排序，不用来判显著。
  3. 每个环境约 640 天，够算均值，但独立样本远少于 640。

用法
  .venv/bin/python -u 46_regime.py
输出
  data/output_zoo/regime_ic.csv
  data/output_zoo/regime_portfolio.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from factorlab.core import (load_panel, composite, smooth_score, replay,
                            benchmark)
from factorlab.gate import forward_returns

OUT = Path("data/output_zoo")
OUT.mkdir(parents=True, exist_ok=True)

H = 5
W = 20
MIN_N = 50
NAMES = ["低", "中", "高"]


def log(m):
    print(m, flush=True)


def nw_t(x, lags=4):
    x = pd.Series(x).dropna().to_numpy(dtype=float)
    n = x.size
    if n < 20:
        return np.nan
    e = x - x.mean()
    g = float((e ** 2).sum() / n)
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        g += 2.0 * w * float((e[L:] * e[:-L]).sum() / n)
    return float(x.mean() / np.sqrt(g / n)) if g > 0 else np.nan


def daily_ic(f, fwd, mask, min_n=MIN_N):
    a = f.where(mask).stack().rename("x")
    b = fwd.where(mask).stack().rename("y")
    d = pd.concat([a, b], axis=1).dropna()
    if d.empty:
        return pd.Series(dtype=float)
    d["rx"] = d.groupby(level=0)["x"].rank()
    d["ry"] = d.groupby(level=0)["y"].rank()
    return (d.groupby(level=0).apply(
        lambda z: z.rx.corr(z.ry) if len(z) >= min_n else np.nan).dropna())


def main():
    t0 = time.time()
    log("① 加载面板...")
    p = load_panel()
    mkt = p.ret.where(p.mask).mean(axis=1)
    disp = p.ret.where(p.mask).std(axis=1)
    log(f"   {len(p.idx)} 天 × {len(p.cols)} 只")

    log("② 构造三个环境变量...")
    regime = pd.DataFrame({
        "mkt_vol20": mkt.rolling(W, min_periods=W).std() * np.sqrt(252),
        "mkt_ret20": (1 + mkt).rolling(W, min_periods=W).apply(np.prod) - 1,
        "disp20": disp.rolling(W, min_periods=W).mean(),
    }).reindex(p.idx)

    labels = {}
    for c in regime.columns:
        s = regime[c].dropna()
        lab = pd.qcut(s, 3, labels=False, duplicates="drop")
        labels[c] = lab
        rng = {int(i): (str(s[lab == i].index.min()),
                        str(s[lab == i].index.max()))
               for i in lab.dropna().unique()}
        log(f"   {c:<10} " + "  ".join(
            f"{NAMES[i]}={rng[i][0]}~{rng[i][1]}" for i in sorted(rng)))

    log("\n③ 各因子日度 IC...")
    fwd = forward_returns(p, H)
    ics = {}
    for name in p.fac:
        name = str(name)
        ics[name] = daily_ic(p.fac[name], fwd, p.mask)
        log(f"   {name:<8} 有效天数 {len(ics[name])}")

    log("\n④ 组合超额...")
    r = replay(smooth_score(composite(p)), p)
    b = benchmark(p).reindex(r.net.index).fillna(0.0)
    ex = (r.net - b).dropna()
    log(f"   年化 {r.ann():.2%}  换手 {r.ann_turnover():.0%}  "
        f"日超额均值 {ex.mean() * 1e4:+.2f} bp")

    # ---------- A/B：因子 IC × 环境（长表 → pivot，避免列索引塌成扁平） ----------
    recs = []
    for c, lab in labels.items():
        for name, s in ics.items():
            d = pd.DataFrame({"ic": s, "r": lab.reindex(s.index)}).dropna()
            for k, x in d.groupby("r")["ic"]:
                recs.append({"regime": c, "bucket": int(k), "factor": name,
                             "ic": float(x.mean()), "t": nw_t(x),
                             "n": int(len(x))})
    long = pd.DataFrame(recs)

    ic_all = long.pivot(index="factor", columns=["regime", "bucket"], values="ic")
    t_all = long.pivot(index="factor", columns=["regime", "bucket"], values="t")

    log("\n" + "=" * 112)
    log("【A】因子 IC × 环境（行=因子，列=环境·分位）")
    log("=" * 112)
    a = ic_all.copy()
    a.columns = [f"{c}·{NAMES[int(i)]}" for c, i in a.columns]
    log(a.round(4).to_string())

    log("\n【B】NW t × 环境")
    log("=" * 112)
    tt = t_all.copy()
    tt.columns = [f"{c}·{NAMES[int(i)]}" for c, i in tt.columns]
    log(tt.round(2).to_string())

    # ---------- C：组合超额 × 环境 ----------
    precs = []
    for c, lab in labels.items():
        d = pd.DataFrame({"ex": ex, "r": lab.reindex(ex.index)}).dropna()
        for k, x in d.groupby("r")["ex"]:
            precs.append({"regime": c, "bucket": int(k),
                          "ann_pct": float(x.mean() * 252 * 100),
                          "t": nw_t(x), "n": int(len(x))})
    plong = pd.DataFrame(precs)

    log("\n" + "=" * 112)
    log("【C】组合超额 × 环境（年化 %）—— 这张最实用")
    log("=" * 112)
    tab_c = plong.pivot(index="regime", columns="bucket", values="ann_pct")
    tab_c.columns = [NAMES[int(i)] for i in tab_c.columns]
    tab_t = plong.pivot(index="regime", columns="bucket", values="t")
    tab_t.columns = [NAMES[int(i)] for i in tab_t.columns]
    tab_n = plong.pivot(index="regime", columns="bucket", values="n")
    tab_n.columns = [NAMES[int(i)] for i in tab_n.columns]

    log("\n年化超额 %")
    log(tab_c.round(2).to_string())
    log("\nNW t（受环境自相关影响，偏大，只看符号与相对大小）")
    log(tab_t.round(2).to_string())
    log("\n天数")
    log(tab_n.to_string())

    log("\n" + "=" * 112)
    log("【极值对比：最不利 vs 最有利环境】")
    log("=" * 112)
    for c in tab_c.index:
        lo, hi = tab_c.loc[c, "低"], tab_c.loc[c, "高"]
        log(f"   {c:<10} 低={lo:+.2f}%  高={hi:+.2f}%  差={hi - lo:+.2f}pp")
    log("\n  若某环境超额为负 → 策略在该状态下失效，需要 regime filter 或降杠杆。")
    log("  但注意：分位点用的是全样本，做成可交易规则必须改成滚动窗口。")

    long.to_csv(OUT / "regime_ic.csv", index=False, encoding="utf-8-sig")
    plong.to_csv(OUT / "regime_portfolio.csv", index=False, encoding="utf-8-sig")
    log(f"\n✅ 输出 {OUT}/regime_*.csv")
    log(f"总耗时 {(time.time() - t0) / 60:.1f} 分钟")
    return 0





if __name__ == "__main__":
    sys.exit(main())
