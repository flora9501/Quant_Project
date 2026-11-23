#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
63_capacity.py  (v4)
容量曲线 + 冲击系数 k 的敏感性。

v4 修改
  ★ AUM 标签用 :g —— 修 0.5 被打印成「0 亿」（f"{0.5:.0f}" 会得到 "0"）
  ★ 「可信上限」改为「最后一个干净点」，并同时报出首个越界点
      （v3 用 bad.index.min()，把第一个越界点当成上限，差一格）
  ★ 作图灰底从 first_bad 起画（v3 用 limit=30 与 x 上限同为 30，axvspan 宽度为 0，从未画出）
  ★ σ / ADV 只构建一次（v3 在 13 次回放里重复 pivot 13 次，白烧 2–3 分钟）
  ★ 无零点时给线性外推提示，并明确标注「已超出模型有效区」
  ★ 新增 --replot：从已存的 CSV 重画 + 重新打印，约 20 秒，不必重跑回放

为什么必须做 k 扫描
  平衡点对 k 是平方敏感的（breakeven ∝ 1/k²）。
  k 从 0.7 到 1.4，容量降到 1/4；到 0.35 则升 4 倍。
  所以容量不是一个点，而是一个区间。

关于 R²（重要）
  不要报告拟合优度。成本本来就是用平方根律生成的，
  再拟合 excess = G − c√AUM 必然 R²≈1 —— 循环论证，不是验证。
  真正的敏感性检验是这里的 k 扫描。

口径
  本脚本的超额 = geom(net) − geom(基准)，即「差值口径」，
  与 49_topn_verify.py 的「超额pp」逐位可比（5.37 / 5.11 / 4.75）。
  不要拿它和 33_product_metrics.py 的 4.25%（比值口径）直接比；
  两者满足 比值 ≈ 差值 / (1 + 基准年化)。

用法
  .venv/bin/python -u 63_capacity.py              # 13 次回放，约 15–19 分钟
  .venv/bin/python -u 63_capacity.py --replot     # 从 CSV 重画，约 20 秒
输出
  data/output_zoo/capacity_ext.csv
  data/output_zoo/capacity_ksens.csv
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
    s = pd.Series(s).dropna()
    if not len(s):
        return np.nan
    return float((1 + s).cumprod().iloc[-1] ** (252 / len(s)) - 1)


# ---------- 成本：σ / ADV 只构建一次 ----------
def build_cost_inputs(p, daily):
    sig = p.ret.replace([np.inf, -np.inf], np.nan).rolling(20, min_periods=20).std()
    amt = (daily.pivot(index="trade_date", columns="ts_code", values="amount")
                .reindex(index=p.idx, columns=p.cols)) * 1000.0
    adv = amt.rolling(20, min_periods=20).mean()
    return sig, adv


def cost_from(sig, adv, aum, n=N, k=K_MAIN, floor=FLOOR, cap=CAP):
    """★ part = 1/n，n 必须跟着持仓只数走，不能写死"""
    part = 1.0 / n
    c = k * sig * np.sqrt((aum * part) / adv.where(adv > 0))
    return (c + floor).clip(upper=cap)


def run(p, sig, adv, score, b, aum_yi, k):
    aum = aum_yi * 1e8
    cm = cost_from(sig, adv, aum, N, k)
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


# ==================== 报告 + 作图 ====================
def report(tab_main, tab_sens, do_plot=True):
    be_main = breakeven(tab_main.index, tab_main["excess"].to_numpy())
    sens = {float(k): tab_sens[k].tolist() for k in tab_sens.columns}
    be_sens = {k: breakeven(tab_sens.index, np.array(v)) for k, v in sens.items()}

    # ★ 首个越界点 与 最后一个干净点 分开
    bad = tab_main[(tab_main["pos_adv_med"] > POS_ADV_WARN)
                   | (tab_main["pos_adv_p90"] > POS_ADV_P90)]
    first_bad = float(bad.index.min()) if not bad.empty else np.nan

    clean = tab_main[(tab_main["pos_adv_med"] <= POS_ADV_WARN)
                     & (tab_main["pos_adv_p90"] <= POS_ADV_P90)]
    limit = float(clean.index.max()) if not clean.empty else np.nan

    log("\n" + "=" * 80)
    log("【主曲线 k=0.7】  口径：超额 = geom(net) − geom(基准)（差值，与 49 一致）")
    log("=" * 80)
    log(tab_main.to_string())

    if np.isfinite(be_main):
        log(f"\n  模型隐含平衡点 k=0.7 : {be_main:.1f} 亿")
    else:
        log("\n  区间内无零点")
        xs = tab_main.index.to_numpy(float)
        ys = tab_main["excess"].to_numpy(float)
        slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        if slope < 0:
            hint = xs[-1] + (-ys[-1]) / slope
            log(f"  线性外推零点 ≈ {hint:.0f} 亿"
                f"（仅量级参考，已超出模型有效区，不要对外报）")

    if np.isfinite(limit):
        why = (f"（{first_bad:g} 亿起单只占 ADV 中位 >{POS_ADV_WARN:g}%"
               f" / p90 >{POS_ADV_P90:g}%）") if np.isfinite(first_bad) else ""
        log(f"  可信上限（pos/ADV）  : ≤ {limit:g} 亿{why}")
    elif np.isfinite(first_bad):
        log(f"  可信上限（pos/ADV）  : 全部越界（最小 {first_bad:g} 亿已超阈值）")

    log("\n【k 敏感性】")
    log(tab_sens.to_string())
    for k, v in be_sens.items():
        if np.isfinite(v):
            log(f"  k={k:<5g} → 平衡点约 {v:.0f} 亿")
        else:
            log(f"  k={k:<5g} → 区间内（≤{max(AUMS_SENS)} 亿）未见零点")

    if 20 in tab_main.index:
        log("\n  20 亿处的超额：")
        log(f"    k={K_MAIN:<5g}{float(tab_main.loc[20, 'excess']):+.2f}pp")
        for k in K_SENS:
            if 20 in tab_sens.index and k in tab_sens.columns:
                log(f"    k={k:<5g}{float(tab_sens.loc[20, k]):+.2f}pp")

    if not do_plot:
        return

    # ---------- 作图 ----------
    fig, ax1 = plt.subplots(figsize=(9.0, 5.0))

    ax1.plot(tab_main.index, tab_main["excess"], "o-", lw=2.2,
             c="#1a5276", ms=6, label=f"k={K_MAIN:g} (base)", zorder=4)
    styles = {0.35: ("s--", "#27ae60"), 1.4: ("^--", "#c0392b")}
    for k in K_SENS:
        if k not in sens:
            continue
        ls, col = styles.get(k, ("--", "#7f8c8d"))
        ax1.plot(AUMS_SENS, sens[k], ls, lw=1.5, ms=5, c=col,
                 label=f"k={k:g}", zorder=3)

    ax1.axhline(0, ls="-", lw=0.9, c="#999999")

    if np.isfinite(be_main):
        ax1.axvline(be_main, ls=":", lw=1.2, c="#1a5276", alpha=0.8)
        ax1.annotate(f"breakeven ≈ {be_main:.0f}",
                     xy=(be_main, 0),
                     xytext=(be_main * 0.55, max(tab_main["excess"]) * 0.72),
                     arrowprops=dict(arrowstyle="->", color="#1a5276"),
                     color="#1a5276", fontsize=9)

    # ★ v3 的 axvspan(30, 30) 宽度为零，从未画出；这里从 first_bad 起画
    if np.isfinite(first_bad):
        x0 = first_bad
        x1 = max(AUMS_MAIN) * 0.98
        ax1.axvspan(x0, x1, color="#bdc3c7", alpha=0.35, zorder=0)
        ax1.axvline(x0, ls="-.", lw=1.0, c="#555555", alpha=0.7)
        ax1.text(x0 + (x1 - x0) * 0.04,
                 min(tab_main["excess"]) + 0.35,
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


# ==================== 主流程 ====================
def main():
    t0 = time.time()
    log("① 加载...")
    p = load_panel()
    daily, _ = load_price_data(daily_cols=("pct_chg", "amount"))
    sig, adv = build_cost_inputs(p, daily)          # ★ 只建一次
    score = smooth_score(composite(p))
    b = benchmark(p)
    log(f"   {len(p.idx)} 天 × {len(p.cols)} 只；N={N}")

    log(f"\n② 主曲线 k={K_MAIN}（{len(AUMS_MAIN)} 个点）...")
    main_rows = []
    for aum_yi in AUMS_MAIN:
        ex, med, p90 = run(p, sig, adv, score, b, aum_yi, K_MAIN)
        main_rows.append({"AUM": aum_yi, "excess": round(ex, 2),
                          "pos_adv_med": round(med, 2),
                          "pos_adv_p90": round(p90, 2)})
        log(f"   {aum_yi:>5g} 亿   {ex:+.2f}pp   "
            f"pos/ADV {med:.2f}% / p90 {p90:.2f}%")
    tab_main = pd.DataFrame(main_rows).set_index("AUM")

    log(f"\n③ 敏感性（k ∈ {K_SENS}，各 {len(AUMS_SENS)} 个点）...")
    sens = {}
    for k in K_SENS:
        ys = []
        for aum_yi in AUMS_SENS:
            ex, _, _ = run(p, sig, adv, score, b, aum_yi, k)
            ys.append(round(ex, 2))
            log(f"   k={k:<5g} {aum_yi:>4g} 亿   {ex:+.2f}pp")
        sens[k] = ys
    tab_sens = pd.DataFrame(sens, index=AUMS_SENS)
    tab_sens.index.name = "AUM"

    report(tab_main, tab_sens, do_plot=True)

    tab_main.to_csv(OUT / "capacity_ext.csv", encoding="utf-8-sig")
    tab_sens.to_csv(OUT / "capacity_ksens.csv", encoding="utf-8-sig")
    log(f"\n✅ 输出 {OUT/'capacity_ext.csv'}、{OUT/'capacity_ksens.csv'}、{FIG/'capacity.png'}")
    log(f"总耗时 {(time.time() - t0) / 60:.1f} 分钟")
    log("\n提醒：不要报告拟合 R²（成本本就按平方根律生成，属循环论证）。")
    log("      容量应报区间；N=400 的可报告上限受 pos/ADV 约束，不是零点。")
    return 0


def replot():
    """从已存 CSV 重画 + 重新打印，不重跑回放。"""
    f1 = OUT / "capacity_ext.csv"
    f2 = OUT / "capacity_ksens.csv"
    if not (f1.exists() and f2.exists()):
        log(f"缺 {f1} 或 {f2}，无法重画；请先正常跑一次 63。")
        return 1
    log("（--replot：从 CSV 重画，未重跑回放）")
    tab_main = pd.read_csv(f1, index_col=0)
    tab_main.index = tab_main.index.astype(float)
    tab_main = tab_main[["excess", "pos_adv_med", "pos_adv_p90"]]
    tab_sens = pd.read_csv(f2, index_col=0)
    tab_sens.index = tab_sens.index.astype(float)
    tab_sens.columns = [float(c) for c in tab_sens.columns]
    report(tab_main, tab_sens, do_plot=True)
    log(f"\n✅ 已重画 {FIG/'capacity.png'}（数据未变）")
    return 0


if __name__ == "__main__":
    if "--replot" in sys.argv:
        sys.exit(replot())
    sys.exit(main())
