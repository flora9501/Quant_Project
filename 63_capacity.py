#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
63_capacity.py  (v3)
容量曲线 + 冲击系数 k 的敏感性。

v3 新增
  除主曲线（k=0.7），再跑 k=0.35 与 k=1.4 两条，
  看平衡点如何随 k 移动。

为什么必须做
  平衡点对 k 是**平方敏感**的（breakeven ∝ 1/k²）。
  k 从 0.7 到 1.4，容量降到 1/4；到 0.35 则升 4 倍。
  所以「容量 20 亿」是个点估计，真实的说法是「一个区间」。

关于 R² （重要）
  不要报告拟合优度。成本本来就是用平方根律生成的，
  再拟合 excess = G − c√AUM 必然 R²≈1——这是循环论证，不是验证。
  真正的敏感性检验是这里的 k 扫描。

用法
  .venv/bin/python -u 63_capacity.py     # 约 12 分钟（13 次回放）
输出
  data/output_zoo/capacity_ext.csv
  figures/capacity.png
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from factorlab.core import load_panel, composite, smooth_score, replay, benchmark
from factorlab.factors import load_price_data

OUT = Path("data/output_zoo")
OUT.mkdir(parents=True, exist_ok=True)
FIG = Path("figures")
FIG.mkdir(exist_ok=True)

N = 400
CAP = 0.02
FLOOR = 0.0003
AUMS_MAIN = [0.5, 1, 2, 5, 10, 20, 30]
AUMS_SENS = [10, 20, 30]
K_MAIN = 0.7
K_SENS = [0.35, 1.4]
POS_ADV_WARN, POS_ADV_P90 = 5.0, 20.0


def log(m):
    print(m, flush=True)


def geom(s):
    s = s.dropna()
    return float((1 + s).cumprod().iloc[-1] ** (252 / len(s)) - 1) if len(s) else np.nan


def cost_matrix(p, daily, aum, k, part=1.0 / N):
    sig = p.ret.replace([np.inf, -np.inf], np.nan).rolling(20, min_periods=20).std()
    amt = (daily.pivot(index="trade_date", columns="ts_code", values="amount")
                .reindex(index=p.idx, columns=p.cols)) * 1000.0
    adv = amt.rolling(20, min_periods=20).mean()
    c = k * sig * np.sqrt((aum * part) / adv.where(adv > 0))
    return (c + FLOOR).clip(upper=CAP), adv


def run(p, daily, score, b, aum_yi, k):
    aum = aum_yi * 1e8
    cm, adv = cost_matrix(p, daily, aum, k)
    r = replay(score, p, top_n=N, cost=cm)
    net = r.net
    bn = b.reindex(net.index).fillna(0.0)
    ex = (geom(net) - geom(bn)) * 100
    ratio = ((aum / N) / adv.where(adv > 0)).where(p.mask)
    med = float(np.nanmedian(ratio.to_numpy())) * 100
    p90 = float(np.nanpercentile(ratio.to_numpy(), 90)) * 100
    return ex, med, p90


def breakeven(xs, ys):
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    for i in range(len(xs) - 1):
        if ys[i] > 0 >= ys[i + 1]:
            return xs[i] + (0 - ys[i]) * (xs[i + 1] - xs[i]) / (ys[i + 1] - ys[i])
    return np.nan


def main():
    t0 = time.time()
    log("① 加载...")
    p = load_panel()
    daily, _ = load_price_data(daily_cols=("pct_chg", "amount"))
    score = smooth_score(composite(p))
    b = benchmark(p)
    log(f"   {len(p.idx)} 天 × {len(p.cols)} 只；N={N}")

    log(f"\n② 主曲线 k={K_MAIN}（{len(AUMS_MAIN)} 个点）...")
    main_rows = []
    for aum_yi in AUMS_MAIN:
        ex, med, p90 = run(p, daily, score, b, aum_yi, K_MAIN)
        main_rows.append({"AUM": aum_yi, "excess": round(ex, 2),
                          "pos_adv_med": round(med, 2),
                          "pos_adv_p90": round(p90, 2)})
        log(f"   {aum_yi:>5.0f} 亿   {ex:+.2f}pp   pos/ADV {med:.2f}% / p90 {p90:.2f}%")
    tab_main = pd.DataFrame(main_rows).set_index("AUM")

    log(f"\n③ 敏感性（k ∈ {K_SENS}，各 {len(AUMS_SENS)} 个点）...")
    sens = {}
    for k in K_SENS:
        ys = []
        for aum_yi in AUMS_SENS:
            ex, _, _ = run(p, daily, score, b, aum_yi, k)
            ys.append(round(ex, 2))
            log(f"   k={k:<5} {aum_yi:>4.0f} 亿   {ex:+.2f}pp")
        sens[k] = ys
    tab_sens = pd.DataFrame(sens, index=AUMS_SENS)
    tab_sens.index.name = "AUM"

    # ---------- 平衡点 ----------
    be_main = breakeven(tab_main.index, tab_main["excess"].to_numpy())
    be_sens = {k: breakeven(AUMS_SENS, np.array(v)) for k, v in sens.items()}

    # 可信上限
    bad = tab_main[(tab_main["pos_adv_med"] > POS_ADV_WARN)
                   | (tab_main["pos_adv_p90"] > POS_ADV_P90)]
    limit = float(bad.index.min()) if not bad.empty else np.nan

    log("\n" + "=" * 80)
    log("【主曲线 k=0.7】")
    log("=" * 80)
    log(tab_main.to_string())
    log(f"\n  模型隐含平衡点 k=0.7 : {be_main:.1f} 亿"
        if np.isfinite(be_main) else "\n  区间内无零点")
    if np.isfinite(limit):
        log(f"  可信上限（pos/ADV）  : ≤ {limit:.0f} 亿")

    log("\n【k 敏感性】")
    log(tab_sens.to_string())
    for k, v in be_sens.items():
        if np.isfinite(v):
            log(f"  k={k:<5} → 平衡点约 {v:.0f} 亿")
        else:
            log(f"  k={k:<5} → 区间内（≤{max(AUMS_SENS)} 亿）未见零点")

    # 20 亿处不同 k 的超额
    log("\n  20 亿处的超额：")
    log(f"    k=0.7  {float(tab_main.loc[20, 'excess']):+.2f}pp")
    for k in K_SENS:
        log(f"    k={k:<5}{float(tab_sens.loc[20, k]):+.2f}pp")

    # ---------- 作图 ----------
    fig, ax1 = plt.subplots(figsize=(9.0, 5.0))

    ax1.plot(tab_main.index, tab_main["excess"], "o-", lw=2.2,
             c="#1a5276", ms=6, label=f"k={K_MAIN} (base)", zorder=4)
    styles = {0.35: ("s--", "#27ae60"), 1.4: ("^--", "#c0392b")}
    for k in K_SENS:
        ls, col = styles.get(k, ("--", "#7f8c8d"))
        ax1.plot(AUMS_SENS, sens[k], ls, lw=1.5, ms=5, c=col,
                 label=f"k={k}", zorder=3)

    ax1.axhline(0, ls="-", lw=0.9, c="#999999")
    if np.isfinite(be_main):
        ax1.axvline(be_main, ls=":", lw=1.2, c="#1a5276", alpha=0.8)
        ax1.annotate(f"breakeven ≈ {be_main:.0f}",
                     xy=(be_main, 0), xytext=(be_main * 0.55, max(tab_main["excess"]) * 0.72),
                     arrowprops=dict(arrowstyle="->", color="#1a5276"),
                     color="#1a5276", fontsize=9)

    if np.isfinite(limit):
        ax1.axvspan(limit, max(AUMS_MAIN), color="#bdc3c7", alpha=0.35, zorder=0)
        ax1.text(limit * 1.03, min(tab_main["excess"]) + 0.2,
                 "impact model\nunreliable", fontsize=8, color="#555555")

    ax1.set_xlabel("AUM (RMB 100m)")
    ax1.set_ylabel("Annualised excess (%)")
    ax1.set_title(f"Capacity and impact-coefficient sensitivity (N={N})")
    ax1.grid(alpha=0.25, ls="--")
    ax1.legend(loc="upper right", frameon=False, fontsize=8.5)

    ax2 = ax1.twinx()
    ax2.plot(tab_main.index, tab_main["pos_adv_med"], ":", lw=1.1,
             c="#e67e22", alpha=0.75)
    ax2.set_ylabel("Position / ADV (%, median, k=0.7)", color="#e67e22")
    ax2.tick_params(axis="y", colors="#e67e22")

    fig.tight_layout()
    fig.savefig(FIG / "capacity.png", dpi=160)
    plt.close(fig)

    tab_main.to_csv(OUT / "capacity_ext.csv", encoding="utf-8-sig")
    tab_sens.to_csv(OUT / "capacity_ksens.csv", encoding="utf-8-sig")
    log(f"\n✅ 输出 {OUT/'capacity_ext.csv'}、{OUT/'capacity_ksens.csv'}、{FIG/'capacity.png'}")
    log(f"总耗时 {(time.time() - t0) / 60:.1f} 分钟")
    log("\n提醒：不要报告拟合 R²（成本本就按平方根律生成，属循环论证）。")
    log("      容量应报区间，例如「k∈[0.7,1.4] 时约 XX–YY 亿」。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
