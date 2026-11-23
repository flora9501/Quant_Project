#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
76_optimize.py  (v5)

v5 修改
  ★ 新增 --solver ecos|osqp|scs，透传给 solve_mv
  ★ 失败统计里区分「求解失败」与「可行性不过」

用法
  .venv/bin/python -u 76_optimize.py --diag
  .venv/bin/python -u 76_optimize.py --lam 5 --solver ecos
  .venv/bin/python -u 76_optimize.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from factorlab.core import (load_panel, composite, smooth_score, benchmark,
                            quota_select)
from factorlab.optimize import (estimate_risk, exposures, candidate_block,
                                solve_mv, diagnose, SOLVERS)

OUT = Path("data/output_zoo")
OUT.mkdir(parents=True, exist_ok=True)

REBAL = 5
M_CAND = 400
W_MAX = 0.02
BAND_IND = 0.02
BAND_STY = 0.25
TAU = 0.30
COST = 0.0012
LAMS = [2.0, 5.0, 10.0]
DIAG_DATES = 24

ARGV = sys.argv
if "--lam" in ARGV:
    LAMS = [float(x) for x in ARGV[ARGV.index("--lam") + 1].split(",")]
if "--band-ind" in ARGV:
    BAND_IND = float(ARGV[ARGV.index("--band-ind") + 1])
if "--w-max" in ARGV:
    W_MAX = float(ARGV[ARGV.index("--w-max") + 1])
if "--tau" in ARGV:
    TAU = float(ARGV[ARGV.index("--tau") + 1])
SOLVER = None
if "--solver" in ARGV:
    key = ARGV[ARGV.index("--solver") + 1].lower()
    SOLVER = SOLVERS.get(key)
    if SOLVER is None:
        raise SystemExit(f"未知 solver {key}；可选 {list(SOLVERS)}")


def log(m):
    print(m, flush=True)


def geom(s):
    s = pd.Series(s).dropna()
    return float((1 + s).cumprod().iloc[-1] ** (252 / len(s)) - 1) if len(s) else np.nan


def run_policy(p, policy, rebal=REBAL, cost=COST):
    w_live = pd.Series(0.0, index=p.cols)
    gross = pd.Series(0.0, index=p.idx)
    to_s = pd.Series(0.0, index=p.idx)
    cost_amt = pd.Series(0.0, index=p.idx)
    weights = {}
    for i, d in enumerate(p.idx):
        if i % rebal == 0:
            w_new = policy(d, w_live)
            if w_new is not None and w_new.sum() > 0:
                tot = w_live.sum()
                w_drift = (w_live / tot) if tot > 0 else w_live
                dw = (w_new - w_drift).abs()
                to_s.loc[d] = float(dw.sum() / 2)
                cost_amt.loc[d] = float(dw.sum() * cost)
                w_live = w_new.reindex(p.cols).fillna(0.0)
                weights[d] = w_new[w_new > 0]
        r = p.fwd.loc[d].fillna(0.0)
        gross.loc[d] = float((w_live * r).sum())
        w_live = w_live * (1 + r)
    net = (gross - cost_amt).iloc[:-1]
    return {"net": net, "gross": gross, "turnover": to_s, "weights": weights}


def metrics(res, b, label):
    net = res["net"]
    bb = b.reindex(net.index).fillna(0.0)
    ex = (net - bb).dropna()
    te = float(ex.std() * np.sqrt(252))
    ann = geom(net)
    ex_ann = ann - geom(bb)
    to = float(res["turnover"].replace(0, np.nan).mean() * 252 / REBAL)
    cost_pp = float((net - res["gross"].reindex(net.index)).abs().mean() * 252) * 100
    nw = [len(w) for w in res["weights"].values()] if res["weights"] else []
    return {"组合": label, "年化": ann, "超额": ex_ann, "TE": te,
            "IR": ex_ann / te if te > 0 else np.nan,
            "换手": to, "成本pp": cost_pp,
            "持仓数": float(np.mean(nw)) if nw else 0.0}


def solve_inputs(p, sc, rm, d, w_live):
    s = sc.loc[d].where(p.mask.loc[d]).dropna()
    if len(s) < 100:
        return None
    msk = p.mask.loc[d]
    allnames = msk[msk].index
    cand = s.nlargest(min(M_CAND, len(s))).index
    if len(cand) * W_MAX < 1.0 - 1e-9:
        return None
    Xall = exposures(p, rm.groups, rm.gnames, d, allnames)
    try:
        Xc = candidate_block(Xall, allnames, cand)
    except KeyError:
        return None
    wb_full = pd.Series(1.0 / len(allnames), index=allnames)
    xb = Xall.T @ wb_full.to_numpy()
    wb_c = wb_full.reindex(cand).fillna(0.0).to_numpy()
    d_c = rm.spec_var.loc[d, cand].to_numpy()
    d_c = np.where(np.isfinite(d_c), d_c, 1e-6)
    w0_c = w_live.reindex(cand).fillna(0.0).to_numpy()
    alpha = s.reindex(cand).to_numpy()
    return dict(alpha=alpha, Xc=Xc, xb=xb, d_c=d_c, wb_c=wb_c,
                w0_c=w0_c, cand=cand)


def main():
    t0 = time.time()
    log("① 加载 + 风险模型...")
    p = load_panel(with_index=True)
    sc = smooth_score(composite(p))
    b = benchmark(p)
    rm = estimate_risk(p)
    G = len(rm.gnames)
    band = np.array([BAND_STY] * len(rm.factors) + [BAND_IND] * G)
    labels = list(rm.factors) + list(rm.gnames)
    log(f"   {len(p.idx)} 天 × {len(p.cols)} 只；"
        f"band 风格 {BAND_STY} / 行业 {BAND_IND}；"
        f"w_max={W_MAX}；tau={TAU}；solver={'默认' if SOLVER is None else ARGV[ARGV.index('--solver') + 1]}")

    # ---------- --diag ----------
    if "--diag" in ARGV:
        log("\n② 等权解上的约束诊断（不调 cvxpy）")
        step = max(1, len(p.idx) // REBAL // DIAG_DATES)
        rows, acc = [], []
        for d in p.idx[::REBAL][::step][:DIAG_DATES]:
            msk = p.mask.loc[d]
            allnames = msk[msk].index
            cand = sc.loc[d].where(msk).dropna().nlargest(
                min(M_CAND, int(msk.sum()))).index
            if len(cand) < 50:
                continue
            Xall = exposures(p, rm.groups, rm.gnames, d, allnames)
            try:
                Xc = candidate_block(Xall, allnames, cand)
            except KeyError:
                continue
            wb_full = pd.Series(1.0 / len(allnames), index=allnames)
            xb = Xall.T @ wb_full.to_numpy()
            wb_c = wb_full.reindex(cand).fillna(0.0).to_numpy()
            d_c = rm.spec_var.loc[d, cand].to_numpy()
            d_c = np.where(np.isfinite(d_c), d_c, 1e-6)
            w_eq = np.full(len(cand), 1.0 / len(cand))
            dg = diagnose(np.zeros(len(cand)), Xc, xb, rm.L, d_c, wb_c, w_eq,
                          band, W_MAX, TAU, list(cand), labels)
            acc.append(dg["exposure_ratio"])
            rows.append({"date": d, "n_cand": len(cand),
                         "TE%": dg["te_ann"] * 100,
                         "worst": dg["worst_exposure"],
                         "worst_ratio": dg["worst_ratio"],
                         "n_binding": dg["n_binding"]})
            if len(rows) == 1:
                log(f"   首个日期 {d}：最紧 {dg['worst_exposure']} "
                    f"用掉 {dg['worst_ratio']:.0%}；TE {dg['te_ann']*100:.2f}%")

        if not rows:
            log("   ⚠ 没有可诊断的日期")
            return 1
        dd = pd.DataFrame(rows)
        log("\n" + "=" * 80)
        log("【约束诊断】等权 top-400 解上，每条约束用掉多少预算")
        log("=" * 80)
        log(dd[["n_cand", "TE%", "worst_ratio", "n_binding"]]
            .describe().loc[["mean", "50%", "max"]].round(3).to_string())

        if acc:
            R = pd.concat(acc, axis=1)
            top = R.mean(axis=1).sort_values(ascending=False).head(8)
            log("\n  用掉预算最多的 8 条约束（均值，%）")
            log((top * 100).round(1).to_string())

        med = float(dd["worst_ratio"].median())
        log(f"\n  worst_ratio 中位 {med:.2f}，最大 {dd['worst_ratio'].max():.2f}")
        log(f"  紧约束个数（>0.9）中位 {dd['n_binding'].median():.0f}")
        if med < 0.5:
            log("\n  → 全部大幅松弛：等权解是可行域的内点，**QP 没有可推的方向**")
        elif med < 0.9:
            log("\n  → 部分接近边界：QP 有一点空间，但由最紧那条决定")
        else:
            log("\n  → 约束已顶：QP 的作用是维持可行性，不是提升收益")

        dd.to_csv(OUT / "optimize_diag.csv", index=False, encoding="utf-8-sig")
        log(f"\n✅ 输出 {OUT/'optimize_diag.csv'}")
        log(f"总耗时 {(time.time() - t0) / 60:.1f} 分钟")
        return 0

    # ---------- A / B ----------
    log("\n② A（EW top-400）与 B（quota top-50）")
    def pol_quota(d, w_live):
        s = sc.loc[d].where(p.mask.loc[d]).dropna()
        if len(s) < 50:
            return None
        held = list(w_live[w_live > 0].index)
        picks = quota_select(s, p.industry, held, 50, 0.25)
        return pd.Series(1.0 / len(picks), index=picks) if picks else None

    def pol_ew400(d, w_live):
        s = sc.loc[d].where(p.mask.loc[d]).dropna()
        if len(s) < 400:
            return None
        return pd.Series(1.0 / 400, index=s.nlargest(400).index)

    tA = time.time(); resA = run_policy(p, pol_ew400)
    log(f"   A 完成 {time.time()-tA:.0f}s")
    tB = time.time(); resB = run_policy(p, pol_quota)
    log(f"   B 完成 {time.time()-tB:.0f}s")
    rows = [metrics(resA, b, "A EW top-400"), metrics(resB, b, "B quota top-50")]

    # ---------- C：QP ----------
    for lam in LAMS:
        log(f"\n③ QP  λ={lam:g}")
        tq = time.time()
        stat = {"ok": 0, "fail": 0, "skip": 0}

        def pol_qp(d, w_live, lam=lam):
            inp = solve_inputs(p, sc, rm, d, w_live)
            if inp is None:
                stat["skip"] += 1
                return None
            w = solve_mv(inp["alpha"], inp["Xc"], inp["xb"], rm.L,
                         inp["d_c"], inp["wb_c"], inp["w0_c"],
                         band, W_MAX, TAU, lam, solver=SOLVER,
                         verbose_fail=(stat["ok"] + stat["fail"] == 0))
            if w is None:
                stat["fail"] += 1
                return None
            stat["ok"] += 1
            return pd.Series(w, index=inp["cand"])

        resC = run_policy(p, pol_qp)
        log(f"   C 完成 {time.time()-tq:.0f}s；成功 {stat['ok']} / "
            f"失败 {stat['fail']} / 跳过 {stat['skip']}")
        rows.append(metrics(resC, b, f"C QP λ={lam:g}"))

    tab = pd.DataFrame(rows).set_index("组合")
    for c in ("年化", "超额", "TE", "换手"):
        tab[c] = (tab[c] * 100).round(2)
    tab["IR"] = tab["IR"].round(3)
    tab["成本pp"] = tab["成本pp"].round(2)
    tab["持仓数"] = tab["持仓数"].round(0)

    log("\n" + "=" * 108)
    log("【对照】同一批调仓日、同一 12bp 成本、同一风险模型")
    log("=" * 108)
    log(tab.to_string())
    log("\n  ① QP 的 IR ≤ B → 『拟合权重不如等权』第四次确认")
    log("  ② 持仓数接近 50 → w_max=2% 在替你做决定")
    log("  ③ 『有没有空间』看 --diag 的 worst_ratio")

    tab.to_csv(OUT / "optimize_compare.csv", encoding="utf-8-sig")
    log(f"\n✅ 输出 {OUT/'optimize_compare.csv'}")
    log(f"总耗时 {(time.time() - t0) / 60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    sys.exit(main())
