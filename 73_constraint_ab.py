#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
73_constraint_ab.py  (v6)
换约束集重测：是不是「long-only + 5 日 + 12bp」杀死了这些因子？

v6 修改
  ★ 新增 --topn：把 A 段与 A3 段的持仓只数从 50 改成任意值
      --noquota 已经证明「行业配额不是凶手」（三个因子全部变差），
      剩下的假设是「集中度」：Q5 约 198 只赚 +6.17%，50 只却亏 −4.76%。
      用 --topn 200 可以直接验证这一点。
      注意：N 会同时改变成本模型（per-name 下单量 = AUM/N），这是对的
      —— 加宽本身就会降低成本。

v5
  ★ check_ret_units：用中位 |ret| 判断口径（max 会被 .BJ 的 200,400% 骗到）
  ★ 涨停阈值改为小数（0.095 / 0.195），mask 在平滑之后施加
  ★ Q5−池 / Q1−池、--noquota、A2 取反

用法
  .venv/bin/python -u 73_constraint_ab.py --factors turn_abn --topn 200
  .venv/bin/python -u 73_constraint_ab.py --factors turn_abn,rev_20 --noquota
  .venv/bin/python -u 73_constraint_ab.py --factors ovn_mom --limit-filter
输出
  data/output_zoo/constraint_ab*.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from factorlab.core import (load_panel, composite, smooth_score, replay,
                            benchmark, TOP_N, COST)
from factorlab.factors import load_price_data, load_cached

OUT = Path("data/output_zoo")
OUT.mkdir(parents=True, exist_ok=True)

CAP = 0.02
N = TOP_N
REBALS = [5, 20, 60]
HS = [5, 20, 60]
AUM = 1.0e8
NAMES = ["rev_20", "amihud", "max_ret", "high_52w", "turn_abn",
         "q_ag", "ovn_mom", "idr_rev", "margin_chg"]

# ★ 所有 ARGV 解析必须在 def cost_from 之前（默认参数在 def 时绑定）
ARGV = sys.argv
if "--rebal" in ARGV:
    REBALS = [int(x) for x in ARGV[ARGV.index("--rebal") + 1].split(",")]
if "--factors" in ARGV:
    NAMES = ARGV[ARGV.index("--factors") + 1].split(",")
if "--topn" in ARGV:
    N = int(ARGV[ARGV.index("--topn") + 1])
NOQUOTA = "--noquota" in ARGV
LIMIT_FILTER = "--limit-filter" in ARGV


def log(m):
    print(m, flush=True)


def geom(s):
    s = pd.Series(s).dropna()
    if not len(s):
        return np.nan
    return float((1 + s).cumprod().iloc[-1] ** (252 / len(s)) - 1)


def build_cost_inputs(p, daily):
    sig = p.ret.replace([np.inf, -np.inf], np.nan).rolling(20, min_periods=20).std()
    amt = (daily.pivot(index="trade_date", columns="ts_code", values="amount")
                .reindex(index=p.idx, columns=p.cols)) * 1000.0
    adv = amt.rolling(20, min_periods=20).mean()
    return sig, adv


def cost_from(sig, adv, aum, n=N, k=0.7, floor=0.0003, cap=CAP):
    part = 1.0 / n
    c = k * sig * np.sqrt((aum * part) / adv.where(adv > 0))
    return (c + floor).clip(upper=cap)


def nw_t(x, lag):
    x = pd.Series(x).dropna()
    n = len(x)
    if n < lag + 10:
        return np.nan
    d = (x - x.mean()).to_numpy()
    s = float(d @ d) / n
    for L in range(1, lag + 1):
        w = 1 - L / (lag + 1)
        s += 2 * w * float(d[L:] @ d[:-L]) / n
    return float(x.mean() / np.sqrt(s / n)) if s > 0 else np.nan


def ols_alpha(y, x):
    m = y.notna() & x.notna()
    yv, xv = y[m].to_numpy(), x[m].to_numpy()
    if len(yv) < 100:
        return np.nan, np.nan, np.nan
    X = np.column_stack([np.ones(len(xv)), xv])
    b = np.linalg.lstsq(X, yv, rcond=None)[0]
    resid = yv - X @ b
    n, k = X.shape
    s2 = float(resid @ resid) / (n - k)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return b[0] * 252, b[1], b[0] / se[0]


def quintile_table(score, fwd_h, mask, nq=5):
    s = score.where(mask)
    q = np.ceil(s.rank(axis=1, pct=True).to_numpy() * nq)
    f = fwd_h.to_numpy()
    if q.shape != f.shape:
        raise ValueError(f"quintile 形状不匹配 {q.shape} vs {f.shape}")
    out = np.full((len(q), nq), np.nan)
    for k in range(1, nq + 1):
        sel = (q == k) & np.isfinite(f)
        cnt = sel.sum(1)
        tot = np.where(sel, f, 0.0).sum(1)
        out[:, k - 1] = np.where(cnt >= 10, tot / np.maximum(cnt, 1), np.nan)
    return pd.Series(out[:, -1] - out[:, 0], index=score.index), out


def replay_topn(score, p, top_n, rebal, cost=COST):
    w_live = pd.Series(0.0, index=p.cols)
    gross = pd.Series(0.0, index=p.idx)
    to_s = pd.Series(0.0, index=p.idx)
    cost_amt = pd.Series(0.0, index=p.idx)
    for i, d in enumerate(p.idx):
        if i % rebal == 0:
            s = score.loc[d].where(p.mask.loc[d]).dropna()
            if len(s) >= top_n:
                picks = s.nlargest(top_n).index.tolist()
                w_new = pd.Series(1.0 / len(picks), index=picks).reindex(p.cols).fillna(0.0)
                tot = w_live.sum()
                w_drift = (w_live / tot) if tot > 0 else w_live
                dw = (w_new - w_drift).abs()
                to_s.loc[d] = float(dw.sum() / 2)
                if isinstance(cost, pd.DataFrame):
                    row = cost.loc[d] if d in cost.index else pd.Series(dtype=float)
                    cvec = row.reindex(dw.index).fillna(COST)
                else:
                    cvec = cost
                cost_amt.loc[d] = float((dw * cvec).sum())
                w_live = w_new
        r = p.fwd.loc[d].fillna(0.0)
        gross.loc[d] = float((w_live * r).sum())
        w_live = w_live * (1 + r)
    net = (gross - cost_amt).iloc[:-1]
    return net, gross, to_s


def check_ret_units(p):
    a = np.abs(p.ret.to_numpy())
    med = float(np.nanmedian(a))
    mx_all = float(np.nanmax(a))
    mx_mask = float(np.nanmax(np.abs(p.ret.where(p.mask).to_numpy())))
    log(f"   p.ret：中位 |ret| {med:.4f}   全表最大 {mx_all:.1f}   "
        f"池内最大 {mx_mask:.3f}")
    if not (med < 0.1):
        raise SystemExit(f"p.ret 中位 |ret| = {med:.3f}，像百分数口径")
    if mx_all > 1.0:
        log(f"   ⚠ 全表最大 {mx_all:.0f}（≈{mx_all * 100:.0f}%）——"
            f".BJ 的 pct_chg 垃圾值，靠 mask 隔离")
        if mx_mask > 0.5:
            log(f"   ⚠⚠ 池内最大也有 {mx_mask:.2f}，需人工核查")


def limit_mask(p, thresh_main=0.095, thresh_growth=0.195):
    lim = pd.Series(thresh_main, index=p.cols, dtype=float)
    lim[p.cols.str.startswith(("300", "688"))] = thresh_growth
    return p.ret.ge(lim, axis=1)


def apply_mask(sc, lim):
    return sc.where(~lim) if lim is not None else sc


def main():
    t0 = time.time()
    log("① 加载...")
    p = load_panel(with_index=True)
    daily, _ = load_price_data(daily_cols=("pct_chg", "amount"))
    sig, adv = build_cost_inputs(p, daily)
    b_all = benchmark(p)
    idx_ret = p.index_ret.shift(-1).reindex(p.idx).fillna(0.0)
    log(f"   {len(p.idx)} 天 × {len(p.cols)} 只；N={N}")

    cp = (1 + p.ret.fillna(0.0)).cumprod()
    fwd = {h: (cp.shift(-h) / cp - 1).reindex(p.idx) for h in HS}
    assert fwd[5].shape == p.mask.shape

    log("\n② 口径自检")
    check_ret_units(p)

    lim = None
    if LIMIT_FILTER:
        lim = limit_mask(p)
        rate = float(lim.to_numpy().mean())
        log(f"   涨停近似过滤：阈值 9.5% / 19.5%，"
            f"平均每天剔除 {rate:.2%} 的股票-日")
        if rate <= 1e-6:
            raise SystemExit("涨停过滤未命中任何单元格")
    else:
        log("   未开启涨停过滤")

    log("\n③ 因子尺度自检")
    facs = {}
    for nm in NAMES:
        f = load_cached(nm)
        if f is None:
            log(f"   缺 {nm}，跳过")
            continue
        f = f.reindex(index=p.idx, columns=p.cols)
        facs[nm] = f
        cov = float((p.mask & f.notna()).to_numpy().sum() / p.mask.to_numpy().sum())
        sd = float(np.nanstd(f.where(p.mask).to_numpy()))
        warn = "" if 0.5 < sd < 2.0 else "  ⚠ 尺度异常"
        if cov < 0.9:
            warn += f"  ⚠ 覆盖 {cov:.0%}，只看「同池」"
        log(f"   {nm:<10} std {sd:.3f}  cover {cov:.1%}{warn}")

    log("\n④ B. Q5−Q1 与「Q5−池 / Q1−池」（gross，年化%）")
    qrows = []
    for nm, f in facs.items():
        d = {"factor": nm}
        for h in HS:
            sc = apply_mask(smooth_score(f, rebal=h), lim)
            sp, out = quintile_table(sc, fwd[h], p.mask)
            qm = np.nanmean(out, axis=0)
            pool_m = float(np.nanmean(fwd[h].where(p.mask).to_numpy()))
            mult = 252 / h
            d[f"Q5Q1@{h}%"] = float(sp.mean() * mult) * 100
            d[f"t@{h}"] = nw_t(sp, lag=h - 1)
            d[f"Q5-池@{h}%"] = float((qm[-1] - pool_m) * mult) * 100
            d[f"Q1-池@{h}%"] = float((qm[0] - pool_m) * mult) * 100
        qrows.append(d)
        log(f"   {nm:<10} Q5Q1 " +
            " ".join(f"h{h}:{d[f'Q5Q1@{h}%']:+.1f}" for h in HS) +
            "   |  Q5−池 " +
            " ".join(f"h{h}:{d[f'Q5-池@{h}%']:+.1f}" for h in HS) +
            "   |  Q1−池 " +
            " ".join(f"h{h}:{d[f'Q1-池@{h}%']:+.1f}" for h in HS))
    tabQ = pd.DataFrame(qrows).set_index("factor")

    log(f"\n⑤ A. long-only，rebal × 1 亿逐股成本（N={N}）")
    rows = []
    for nm, f in facs.items():
        pool = p.mask & f.notna()
        b_pool = p.fwd.where(pool).mean(axis=1).reindex(p.idx).fillna(0.0)
        cov = float(pool.to_numpy().sum() / p.mask.to_numpy().sum())
        g_pool, g_all = geom(b_pool), geom(b_all)
        for rb in REBALS:
            sc = apply_mask(smooth_score(f, rebal=rb), lim)
            r = replay(sc, p, top_n=N, rebal=rb, cost=cost_from(sig, adv, AUM))
            g = geom(r.gross.reindex(r.net.index))
            net_ann = geom(r.net)
            to = r.ann_turnover()
            a, beta, at = ols_alpha(r.net, idx_ret)
            rows.append({
                "factor": nm, "rebal": rb, "N": N, "覆盖%": cov * 100,
                "换手%": to * 100, "gross%": g * 100,
                "net@1亿%": net_ann * 100,
                "同池超额%": (net_ann - g_pool) * 100,
                "全池超额%": (net_ann - g_all) * 100,
                "β": beta, "α_ann%": a * 100 if np.isfinite(a) else np.nan,
                "α_t": at,
            })
            log(f"   {nm:<10} rebal {rb:<3} 换手 {to*100:5.0f}%  "
                f"gross {g*100:+6.2f}%  net {net_ann*100:+6.2f}%  "
                f"同池 {(net_ann-g_pool)*100:+6.2f}%  "
                f"全池 {(net_ann-g_all)*100:+6.2f}%  α {a*100:+6.2f}% (t {at:+.2f})")
    tabA = pd.DataFrame(rows)

    log("\n⑥ A2. 取反后的 long-only（Q5−Q1 三档全负且 |t|>2）")
    flip_rows = []
    for nm, f in facs.items():
        signs = [tabQ.loc[nm, f"Q5Q1@{h}%"] for h in HS]
        ts = [tabQ.loc[nm, f"t@{h}"] for h in HS]
        if not (all(s < 0 for s in signs) and all(abs(t) > 2 for t in ts)):
            continue
        pool = p.mask & f.notna()
        b_pool = p.fwd.where(pool).mean(axis=1).reindex(p.idx).fillna(0.0)
        for rb in REBALS:
            sc = apply_mask(smooth_score(-f, rebal=rb), lim)
            r = replay(sc, p, top_n=N, rebal=rb, cost=cost_from(sig, adv, AUM))
            net_ann = geom(r.net)
            a, beta, at = ols_alpha(r.net, idx_ret)
            flip_rows.append({"factor": nm, "rebal": rb,
                              "同池超额_取反%": (net_ann - geom(b_pool)) * 100,
                              "α_ann_取反%": a * 100, "α_t_取反": at})
            log(f"   {nm:<10} rebal {rb:<3} "
                f"同池 {net_ann*100 - geom(b_pool)*100:+6.2f}%  "
                f"α {a*100:+6.2f}% (t {at:+.2f})")
    tabF = pd.DataFrame(flip_rows) if flip_rows else pd.DataFrame()

    tabN = pd.DataFrame()
    if NOQUOTA:
        log(f"\n⑦ A3. 无行业配额、无门槛的纯 top-N（N={N}，只看最优 rebal）")
        nrows = []
        for nm, f in facs.items():
            sub = tabA[tabA.factor == nm]
            if sub.empty:
                continue
            rb = int(sub.sort_values("同池超额%", ascending=False).iloc[0]["rebal"])
            pool = p.mask & f.notna()
            b_pool = p.fwd.where(pool).mean(axis=1).reindex(p.idx).fillna(0.0)
            sc = apply_mask(smooth_score(f, rebal=rb), lim)
            net, gross, to_s = replay_topn(sc, p, N, rb,
                                           cost_from(sig, adv, AUM))
            nrows.append({"factor": nm, "rebal": rb,
                          "同池超额_无配额%": (geom(net) - geom(b_pool)) * 100,
                          "换手_无配额%": float(
                              to_s.replace(0, np.nan).mean() * 252 / rb) * 100})
            log(f"   {nm:<10} rebal {rb:<3} "
                f"同池 {nrows[-1]['同池超额_无配额%']:+6.2f}%  "
                f"换手 {nrows[-1]['换手_无配额%']:.0f}%")
        tabN = pd.DataFrame(nrows).set_index("factor")

    log("\n" + "=" * 140)
    log(f"【A】long-only（1 亿，N={N}）")
    log("=" * 140)
    sh = tabA.copy()
    for c in ("覆盖%", "换手%", "gross%", "net@1亿%", "同池超额%",
              "全池超额%", "α_ann%"):
        sh[c] = sh[c].round(2)
    for c in ("β", "α_t"):
        sh[c] = sh[c].round(2)
    log(sh.to_string(index=False))

    log("\n" + "-" * 140)
    log("【B】Q5−Q1 / Q5−池 / Q1−池（gross，天然同池）")
    log(tabQ.round(2).to_string())
    if len(tabF):
        log("\n【A2 取反后】")
        log(tabF.round(2).to_string(index=False))
    if len(tabN):
        log("\n【A3 无配额】")
        log(tabN.round(2).to_string())

    log("\n" + "=" * 140)
    log("【判读】A 真否定 / B 长持有下活 / C 信息在空头侧 / D 组合构造吃掉信号")
    log("=" * 140)
    best = tabA.sort_values("同池超额%", ascending=False).groupby("factor").head(1)
    for _, row in best.iterrows():
        nm, rb = row["factor"], int(row["rebal"])
        alive = row["同池超额%"] > 0 and abs(row["α_t"] or 0) > 1.5
        q5p = max(tabQ.loc[nm, f"Q5-池@{h}%"] for h in HS)
        q1p = min(tabQ.loc[nm, f"Q1-池@{h}%"] for h in HS)
        q_sp = max(abs(tabQ.loc[nm, f"Q5Q1@{h}%"]) for h in HS)
        if alive and rb != 5:
            v = f"B 长持有下活（rebal={rb}，同池 {row['同池超额%']:+.2f}%）"
        elif alive:
            v = f"原配置即活（同池 {row['同池超额%']:+.2f}%）→ 需核查"
        elif q5p > 2.0:
            extra = ""
            if len(tabN) and nm in tabN.index:
                v2 = tabN.loc[nm, "同池超额_无配额%"]
                extra = (f"；无配额后 {v2:+.2f}% → "
                         f"{'配额是凶手' if v2 > 0 else '配额不是主因'}")
            v = (f"D 多头端有效（Q5−池 {q5p:+.1f}%）但 long-only 负"
                 f"（{row['同池超额%']:+.2f}%）→ 组合构造吃掉信号{extra}")
        elif q1p < -2.0 and q_sp > 2.0:
            v = f"C 信息在空头侧（Q1−池 {q1p:+.1f}%）→ long-only 不可变现"
        else:
            v = f"A 真否定（Q5−池 {q5p:+.1f}%、同池 {row['同池超额%']:+.2f}%）"
        log(f"  {nm:<11} 最优 rebal={rb:<3} N={N}  {v}")

    tag = f".n{N}" if N != TOP_N else ""
    tabA.to_csv(OUT / f"constraint_ab{tag}.csv", index=False, encoding="utf-8-sig")
    tabQ.to_csv(OUT / f"constraint_ab_q5q1{tag}.csv", encoding="utf-8-sig")
    if len(tabF):
        tabF.to_csv(OUT / f"constraint_ab_flip{tag}.csv", index=False, encoding="utf-8-sig")
    if len(tabN):
        tabN.to_csv(OUT / f"constraint_ab_noquota{tag}.csv", index=True, encoding="utf-8-sig")
    log(f"\n✅ 输出 {OUT}/constraint_ab*{tag}.csv")
    log(f"总耗时 {(time.time() - t0) / 60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    sys.exit(main())
