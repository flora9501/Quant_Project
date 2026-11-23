#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
49_topn_verify.py  (v2)
验证「加宽组合」的三件事。

v2 修改
  ★ cost_matrix 显式接受 n，内部 part = 1/n
      （v1 把 part 写死 1/50，N=200/400 的冲击成本被高估 → 容量被低估约 2.8×）
  ★ 毛超额改为实测（r.gross），不再用「超额 + 换手×12bp×2」回填
      （v1 推导值偏低约 0.18pp：报 6.18，真值 6.36）
  ★ σ / ADV 只构建一次，避免 9 次重复 pivot（省 2–3 分钟）
  ★ 新增 --skip-cost：只跑 12bp 的四条
  ★ 主动份额标注为机械值 1−N/池，不作为风险判据

三个检验（判据预注册）
  ① 逐股成本 0.5 / 1 / 2 亿：N=400 在 1 亿下比 N=50 高 >0.5pp 才算有容量价值
  ② TE / 主动份额：N=400 的 TE 若 <3% → 已接近基准，不能算主动策略
  ③ 暴露：ln 市值差 >0.2 → 赢的是 beta

用法
  .venv/bin/python -u 49_topn_verify.py              # 约 16 分钟
  .venv/bin/python -u 49_topn_verify.py --skip-cost  # 约 5 分钟
输出
  data/output_zoo/topn_verify.csv
  data/output_zoo/topn_cost.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from factorlab.core import (load_panel, composite, smooth_score, replay,
                            benchmark)
from factorlab.factors import load_price_data

OUT = Path("data/output_zoo")
OUT.mkdir(parents=True, exist_ok=True)

NS = [50, 100, 200, 400]
NS_COST = [50, 200, 400]
# 需要平衡点时扩成 [(0.5e8,"0.5亿"),(1e8,"1.0亿"),(2e8,"2.0亿"),
#                    (5e8,"5亿"),(10e8,"10亿"),(20e8,"20亿")]
AUMS = [(0.5e8, "0.5亿"), (1e8, "1.0亿"), (2e8, "2.0亿")]
COST = 0.0012
CAP = 0.02
SKIP_COST = "--skip-cost" in sys.argv


def log(m):
    print(m, flush=True)


def geom(s):
    s = pd.Series(s).dropna()
    if not len(s):
        return np.nan
    return float((1 + s).cumprod().iloc[-1] ** (252 / len(s)) - 1)


# ---------- 成本：σ / ADV 只建一次 ----------
def build_cost_inputs(p, daily):
    sig = p.ret.replace([np.inf, -np.inf], np.nan).rolling(20, min_periods=20).std()
    amt = (daily.pivot(index="trade_date", columns="ts_code", values="amount")
                .reindex(index=p.idx, columns=p.cols)) * 1000.0
    adv = amt.rolling(20, min_periods=20).mean()
    return sig, adv


def cost_from(sig, adv, aum, n, k=0.7, floor=0.0003, cap=CAP):
    """★ part = 1/n，n 必须显式传入，杜绝写死 1/50"""
    part = 1.0 / n
    c = k * sig * np.sqrt((aum * part) / adv.where(adv > 0))
    return (c + floor).clip(upper=cap)


def stats(r, bench):
    net = r.net
    b = bench.reindex(net.index).fillna(0.0)
    ex = (net - b).dropna()
    te = float(ex.std() * np.sqrt(252))
    g_net = geom(net)
    g_bm = geom(b)
    # ★ 实测毛收益；Replay.gross 比 net 多最后一行，必须 reindex
    gross = r.gross.reindex(net.index) if getattr(r, "gross", None) is not None else None
    g_gross = geom(gross) if gross is not None else np.nan
    return {
        "年化": g_net,
        "超额pp": (g_net - g_bm) * 100,
        "毛超额pp": (g_gross - g_bm) * 100,
        "TE": te,
        "IR(算术)": (float(ex.mean() / ex.std() * np.sqrt(252))
                     if ex.std() > 0 else np.nan),
        "IR(比值)": (g_net - g_bm) / te if te > 0 else np.nan,
        "换手": r.ann_turnover(),
    }


def active_share(holdings, mask):
    """机械值 ≈ 1 − N/池，恒等于持仓只数的函数，不是风险度量。保留只为对照。"""
    vals = []
    for d, names in holdings.items():
        if d not in mask.index or not names:
            continue
        n = int(mask.loc[d].sum())
        if n == 0:
            continue
        wb = 1.0 / n
        wp = 1.0 / len(names)
        vals.append(0.5 * (len(names) * abs(wp - wb) + (n - len(names)) * wb))
    return float(np.mean(vals)) if vals else np.nan


def expo(holdings, mat):
    vals = []
    for d, names in holdings.items():
        if d not in mat.index or not names:
            continue
        cols = [x for x in names if x in mat.columns]
        if not cols:
            continue
        x = mat.loc[d, cols]
        x = x[x.notna()]
        if len(x):
            vals.append(float(x.mean()))
    return float(np.mean(vals)) if vals else np.nan


def main():
    t0 = time.time()
    log("① 加载...")
    p = load_panel()
    daily, basic = load_price_data(daily_cols=("pct_chg", "amount"))
    cmv = (basic.pivot(index="trade_date", columns="ts_code", values="circ_mv")
                .reindex(index=p.idx, columns=p.cols))
    turn = (basic.pivot(index="trade_date", columns="ts_code", values="turnover_rate")
                 .reindex(index=p.idx, columns=p.cols))
    lnmv = np.log(cmv * 1e4)
    b = benchmark(p)
    score = smooth_score(composite(p))
    log(f"   {len(p.idx)} 天 × {len(p.cols)} 只")

    log("\n② 各 N 回放（12bp）...")
    rows = []
    for n in NS:
        r = replay(score, p, top_n=n)
        st = stats(r, b)
        st["N"] = n
        st["主动份额"] = active_share(r.holdings, p.mask)
        st["ln市值"] = expo(r.holdings, lnmv)
        st["换手率"] = expo(r.holdings, turn)
        rows.append(st)
        log(f"   N={n:<4} 年化 {st['年化']:+.2%}  超额 {st['超额pp']:+.2f}pp  "
            f"毛 {st['毛超额pp']:+.2f}pp  TE {st['TE']:.2%}  "
            f"IR {st['IR(比值)']:.2f}  换手 {st['换手']:.0%}  "
            f"主动 {st['主动份额']:.0%}")

    tab = pd.DataFrame(rows).set_index("N")[
        ["年化", "超额pp", "毛超额pp", "TE", "IR(算术)", "IR(比值)",
         "换手", "主动份额", "ln市值", "换手率"]]
    for c in ("年化", "超额pp", "毛超额pp", "TE", "换手", "主动份额"):
        tab[c] = tab[c].round(4)
    for c in ("IR(算术)", "IR(比值)"):
        tab[c] = tab[c].round(2)
    for c in ("ln市值", "换手率"):
        tab[c] = tab[c].round(2)

    log("\n" + "=" * 104)
    log("【② TE / IR / 主动份额】")
    log("=" * 104)
    log(tab.to_string())
    log("\n  注：主动份额 ≈ 1 − N/池（N=400 时恒为 ~60%），是机械值；"
        "判断「是否仍为主动策略」只看 TE。")

    # ---------- ① 逐股成本 ----------
    cost_tab = None
    if not SKIP_COST:
        log("\n③ 逐股成本（0.5 / 1 / 2 亿，part = 1/n）...")
        sig, adv = build_cost_inputs(p, daily)
        cost_rows = []
        for n in NS_COST:
            row = {"N": n, "12bp": round(float(tab.loc[n, "超额pp"]), 2)}
            for aum, label in AUMS:
                cm = cost_from(sig, adv, aum, n)
                r = replay(score, p, top_n=n, cost=cm)
                row[label] = round(stats(r, b)["超额pp"], 2)
            cost_rows.append(row)
            log("   " + str(row))
        cost_tab = pd.DataFrame(cost_rows).set_index("N")

        log("\n" + "=" * 104)
        log("【① 成本敏感性（超额 pp）】")
        log("=" * 104)
        log(cost_tab.to_string())
        log("\n  注：0.5 亿时 N=400 可能高于 12bp 行——那是因为实际单边成本"
            "（~9.4bp）低于扁平假设的 12bp，不是错。")

    # ---------- 判读 ----------
    g50 = float(tab.loc[50, "毛超额pp"])
    g400 = float(tab.loc[400, "毛超额pp"])
    te400 = float(tab.loc[400, "TE"])
    dmv = abs(float(tab.loc[400, "ln市值"] - tab.loc[50, "ln市值"]))

    log("\n" + "=" * 104)
    log("【判读】")
    log("=" * 104)
    log(f"  毛超额（实测）N=50 {g50:+.2f}pp  →  N=400 {g400:+.2f}pp   "
        f"（差 {g400 - g50:+.2f}pp，≈0 说明不是 alpha 改善）")
    if cost_tab is not None:
        d_1e = float(cost_tab.loc[400, "1.0亿"] - cost_tab.loc[50, "1.0亿"])
        log(f"  ① 1 亿 AUM 下 N=400 − N=50 = {d_1e:+.2f}pp   （>+0.5pp 才算有容量价值）")
    log(f"  ② N=400 TE {te400:.2%}   （<3% → 接近基准）")
    log(f"  ③ ln 市值差 {dmv:.2f}   （>0.2 → 赢的是 beta）")

    ok_active = te400 >= 0.03
    ok_expo = dmv <= 0.2
    if cost_tab is not None:
        ok_cap = d_1e > 0.5
        log(f"\n  → 容量价值 {'✅' if ok_cap else '❌'}   "
            f"仍是主动策略 {'✅' if ok_active else '❌'}   "
            f"暴露未漂移 {'✅' if ok_expo else '❌'}")

    plat = tab.loc[[n for n in tab.index if n >= 200], "超额pp"]
    log(f"\n  N≥200 的平台：{plat.min():+.2f} ~ {plat.max():+.2f}pp"
        f"（极差 {plat.max() - plat.min():.2f}pp）；N=100 是洼地 "
        f"{float(tab.loc[100, '超额pp']):+.2f}pp（行业配额取整所致）")
    log("  N=400 的平衡点需 5/10/20/30 亿的点位，见 63_capacity.py。")

    tab.to_csv(OUT / "topn_verify.csv", encoding="utf-8-sig")
    if cost_tab is not None:
        cost_tab.to_csv(OUT / "topn_cost.csv", encoding="utf-8-sig")
    log(f"\n✅ 输出 {OUT}/topn_verify.csv" +
        ("、topn_cost.csv" if cost_tab is not None else "（已跳过成本）"))
    log(f"总耗时 {(time.time() - t0) / 60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    sys.exit(main())
