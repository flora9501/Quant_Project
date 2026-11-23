# factorlab/evaluate.py
"""
统一的因子评估台。

v7 修改
  ★ bh_from_log 改用「残差 IC 的单边 p」。
      v6 用原始 IC 的 t → amihud（残差 IC −0.0065）被误判为「通过多重检验」。
      正确假设是「该因子有正增量」，所以要用单边 p 且取残差 IC。

v6
  ★ usable 语义：未跑 deep 时为 None（未知），不再默认 True
  ★ coverage_trend / coverage_drifting（>0.30 视为历史不平稳）

v5
  ★ 「质量门」与「可并入」分离，delta 不参与 pass_all
  ★ resid_ic 门槛加精度条件（≥0.015，或 ≥0.008 且 |t| ≥ 4）
  ★ excess_12bp 是真净超额；另给 gross_excess / cost_pp_12bp
  ★ first_effective_year / low_cov_years
  ★ 成本输入懒加载；rank 与 book 矩阵缓存

已知技术债
  _nw_t 与 factorlab/gate.py 有重叠；应收口为 gate.py 反向 import 本模块。
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from .core import (Panel, load_panel, composite, smooth_score, replay,
                   benchmark, FACTORS, TOP_N, REBAL, COST)

OUT = Path("data/output_zoo")
TRIALS = OUT / "eval_trials.csv"

GATE = {
    "ic_min": 0.02,
    "icir_min": 0.30,
    "ic_t_min": 3.0,

    # resid_ic 二选一：点估计够大，或稍小但极其精确
    "resid_ic_min": 0.015,
    "resid_ic_precise_min": 0.008,
    "resid_ic_t_min": 4.0,

    "corr_max": 0.70,
    "monotone_min": 0.80,
    "coverage_min": 0.50,
}
LOW_COV_FLAG = 0.25
DRIFT_FLAG = 0.30


# ==================== 统计工具 ====================
def _row_corr(A, B, min_n=30):
    m = np.isfinite(A) & np.isfinite(B)
    n = m.sum(1)
    A0 = np.where(m, A, 0.0)
    B0 = np.where(m, B, 0.0)
    Am = A0.sum(1) / np.maximum(n, 1)
    Bm = B0.sum(1) / np.maximum(n, 1)
    a = np.where(m, A0 - Am[:, None], 0.0)
    b = np.where(m, B0 - Bm[:, None], 0.0)
    num = (a * b).sum(1)
    den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    out = np.full(len(num), np.nan)
    ok = (den > 0) & (n >= min_n)
    np.divide(num, den, out=out, where=ok)
    return out


def _nw_t(x, lag=0):
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


def _quintile_from_pct(S_pct, fwd, nq=5):
    q = np.ceil(S_pct * nq)
    f = fwd.to_numpy()
    if q.shape != f.shape:
        raise ValueError(f"形状不匹配 {q.shape} vs {f.shape}")
    out = np.full((len(q), nq), np.nan)
    for k in range(1, nq + 1):
        sel = (q == k) & np.isfinite(f)
        cnt = sel.sum(1)
        tot = np.where(sel, f, 0.0).sum(1)
        out[:, k - 1] = np.where(cnt >= 10, tot / np.maximum(cnt, 1), np.nan)
    return out


def _bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(q, 1.0)
    return out


def _p_one_sided(t, n_eff):
    """H1: IC > 0 的单边 p"""
    if not np.isfinite(t) or n_eff < 10:
        return 1.0
    return float(1 - sps.t.cdf(t, df=max(n_eff - 1, 1)))


def _geom(s):
    s = pd.Series(s).dropna()
    if not len(s):
        return np.nan
    return float((1 + s).cumprod().iloc[-1] ** (252 / len(s)) - 1)


# ==================== 上下文 ====================
@dataclass
class EvalContext:
    panel: Panel
    fwd: dict
    bench: pd.Series
    idx_ret: pd.Series
    book: dict
    _sig: pd.DataFrame | None = None
    _adv: pd.DataFrame | None = None
    _comp: dict = field(default_factory=dict)
    _book_mat: np.ndarray | None = None

    def comp_raw(self):
        if "raw" not in self._comp:
            self._comp["raw"] = composite(self.panel)
        return self._comp["raw"]

    def comp(self, rebal=REBAL):
        if rebal not in self._comp:
            self._comp[rebal] = smooth_score(self.comp_raw(), rebal=rebal)
        return self._comp[rebal]

    def book_matrix(self):
        if self._book_mat is None:
            m = self.panel.mask
            self._book_mat = np.stack(
                [self.book[k].where(m).to_numpy() for k in self.book], axis=2)
        return self._book_mat

    def ensure_cost(self):
        if self._sig is None:
            from .factors import load_price_data
            p = self.panel
            daily, _ = load_price_data(daily_cols=("pct_chg", "amount"))
            self._sig = (p.ret.replace([np.inf, -np.inf], np.nan)
                         .rolling(20, min_periods=20).std())
            amt = (daily.pivot(index="trade_date", columns="ts_code", values="amount")
                        .reindex(index=p.idx, columns=p.cols)) * 1000.0
            self._adv = amt.rolling(20, min_periods=20).mean()
        return self._sig, self._adv


def build_context(panel=None, horizons=(1, 5, 10, 20, 60), with_cost=False):
    p = panel or load_panel(with_index=True)
    cp = (1 + p.ret.fillna(0.0)).cumprod()
    fwd = {h: (cp.shift(-h) / cp - 1).reindex(p.idx) for h in horizons}
    ctx = EvalContext(
        panel=p, fwd=fwd, bench=benchmark(p),
        idx_ret=p.index_ret.shift(-1).reindex(p.idx).fillna(0.0),
        book={k: p.fac[k] for k in FACTORS},
    )
    if with_cost:
        ctx.ensure_cost()
    return ctx


def cost_matrix(ctx, aum, n=TOP_N, k=0.7, floor=0.0003, cap=0.02):
    sig, adv = ctx.ensure_cost()
    part = 1.0 / n
    c = k * sig * np.sqrt((aum * part) / adv.where(adv > 0))
    return (c + floor).clip(upper=cap)


# ==================== 主函数 ====================
def evaluate(ctx, fac, name="factor", rebal=REBAL, deep="auto",
             aums=(1.0,), nq=5, h0=5, w_new=None):
    p = ctx.panel
    mask = p.mask
    if h0 not in ctx.fwd:
        h0 = min(ctx.fwd)
    res = {"name": name, "rebal": rebal, "gate": {}, "h0": h0}

    # ---------- gate 0 ----------
    inmask = fac.where(mask)
    valid = inmask.notna().to_numpy()
    m = mask.to_numpy()
    res["coverage"] = float(valid[m].sum() / m.sum())
    res["scale_std"] = float(np.nanstd(inmask.to_numpy()))
    res["scale_ok"] = 0.5 < res["scale_std"] < 2.0

    yr = pd.Series(fac.index).str[:4].to_numpy()
    cov_y = {}
    for y in pd.unique(yr):
        sel = yr == y
        d, v = m[sel], valid[sel]
        cov_y[str(y)] = round(float(v[d].sum() / d.sum()), 3) if d.sum() else np.nan
    res["coverage_by_year"] = cov_y
    res["low_cov_years"] = [y for y in sorted(cov_y)
                            if np.isfinite(cov_y[y]) and cov_y[y] < LOW_COV_FLAG]
    res["first_effective_year"] = next(
        (y for y in sorted(cov_y) if np.isfinite(cov_y[y])
         and cov_y[y] >= GATE["coverage_min"]), None)

    _c = [cov_y[y] for y in sorted(cov_y) if np.isfinite(cov_y[y])]
    res["coverage_trend"] = round(float(_c[-1] - _c[0]), 3) if len(_c) >= 2 else np.nan
    res["coverage_drifting"] = bool(
        np.isfinite(res["coverage_trend"]) and res["coverage_trend"] > DRIFT_FLAG)

    # ---------- gate 1 ----------
    S_pct = inmask.rank(axis=1, pct=True).to_numpy()
    ic_stats = {}
    for h, fw in ctx.fwd.items():
        F = fw.where(mask).rank(axis=1, pct=True).to_numpy()
        ic = pd.Series(_row_corr(S_pct, F), index=p.idx).dropna()
        ic_stats[h] = {
            "ic": float(ic.mean()) if len(ic) else np.nan,
            "icir": float(ic.mean() / ic.std()) if len(ic) > 5 and ic.std() > 0 else np.nan,
            "t_nw": _nw_t(ic, lag=h - 1),
            "n": int(len(ic)),
        }
    res["ic"] = ic_stats
    res["ic_decay"] = {h: round(v["ic"], 4) for h, v in ic_stats.items()}
    res["best_h"] = max(ic_stats, key=lambda h: abs(ic_stats[h]["ic"] or 0))

    base = ic_stats[h0]
    res["gate"]["ic"] = bool(base["ic"] >= GATE["ic_min"])
    res["gate"]["icir"] = bool(base["icir"] >= GATE["icir_min"])
    res["gate"]["ic_t"] = bool(abs(base["t_nw"] or 0) >= GATE["ic_t_min"])
    res["gate"]["coverage"] = bool(res["coverage"] >= GATE["coverage_min"])

    # ---------- gate 2 ----------
    corrs = {}
    for k, bk in ctx.book.items():
        B = bk.where(mask).rank(axis=1, pct=True).to_numpy()
        corrs[k] = float(np.nanmean(_row_corr(S_pct, B)))
    res["corr_with_book"] = {k: round(v, 3) for k, v in corrs.items()}
    res["corr_max"] = max(abs(v) for v in corrs.values()) if corrs else np.nan
    res["gate"]["corr"] = bool(res["corr_max"] < GATE["corr_max"])

    BM = ctx.book_matrix()
    Y = inmask.to_numpy()
    resid = np.full_like(Y, np.nan)
    for i in range(Y.shape[0]):
        y, X = Y[i], BM[i]
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        if ok.sum() < 100:
            continue
        Xi = np.column_stack([np.ones(int(ok.sum())), X[ok]])
        yy = y[ok]
        b, *_ = np.linalg.lstsq(Xi, yy, rcond=None)
        resid[i, ok] = yy - Xi @ b
    resid_df = pd.DataFrame(resid, index=p.idx, columns=p.cols)

    Sr = resid_df.where(mask).rank(axis=1, pct=True).to_numpy()
    Fr = ctx.fwd[h0].where(mask).rank(axis=1, pct=True).to_numpy()
    ric = pd.Series(_row_corr(Sr, Fr), index=p.idx).dropna()
    res["resid_ic"] = float(ric.mean()) if len(ric) else np.nan
    res["resid_ic_t"] = _nw_t(ric, lag=h0 - 1)

    ok_big = res["resid_ic"] >= GATE["resid_ic_min"]
    ok_precise = (res["resid_ic"] >= GATE["resid_ic_precise_min"]
                  and abs(res["resid_ic_t"] or 0) >= GATE["resid_ic_t_min"])
    res["gate"]["resid_ic"] = bool(ok_big or ok_precise)
    res["resid_ic_via"] = ("点估计" if ok_big
                           else ("精度" if ok_precise else "未过"))

    # ---------- gate 3 ----------
    qm = _quintile_from_pct(S_pct, ctx.fwd[h0], nq)
    means = np.nanmean(qm, axis=0)
    res["quintile_bp"] = [round(float(x) * 1e4, 2) for x in means]
    res["spread_bp"] = res["quintile_bp"][-1] - res["quintile_bp"][0]
    ok = np.isfinite(means)
    res["monotone"] = (float(sps.spearmanr(np.arange(len(means))[ok],
                                           means[ok])[0])
                       if ok.sum() >= 3 else np.nan)
    res["gate"]["monotone"] = bool((res["monotone"] or 0) >= GATE["monotone_min"])

    res["pass_gates"] = all(res["gate"].values())

    # ---------- deep ----------
    if deep == "auto":
        deep = res["pass_gates"]
    res["deep_run"] = bool(deep)
    res["delta_ok"] = None
    res["delta_excess"] = np.nan
    res["delta_t"] = np.nan

    if deep:
        comp_raw = ctx.comp_raw()
        base_sc = ctx.comp(rebal)
        sc = smooth_score(fac, rebal=rebal)

        r12 = replay(sc, p, top_n=TOP_N, rebal=rebal)
        res["turnover"] = r12.ann_turnover()
        bench_ann = _geom(ctx.bench.reindex(r12.net.index))
        res["gross_ann"] = _geom(r12.gross.reindex(r12.net.index))
        res["gross_excess"] = res["gross_ann"] - bench_ann
        res["excess_12bp"] = _geom(r12.net) - bench_ann
        res["cost_pp_12bp"] = res["gross_excess"] - res["excess_12bp"]

        vc = float(np.nanvar(comp_raw.where(mask).to_numpy()))
        vf = float(np.nanvar(fac.where(mask).to_numpy()))
        if w_new is None:
            rr = np.sqrt(0.2 / 0.8) * np.sqrt(vc / vf)
            w_new = rr / (1 + rr)
        res["w_new"] = float(w_new)
        res["w_new_naive_share"] = float(
            (0.2 ** 2 * vf) / (0.8 ** 2 * vc + 0.2 ** 2 * vf))

        aug_sc = smooth_score(comp_raw * (1 - w_new) + fac.fillna(0.0) * w_new,
                              rebal=rebal)

        cost_res, delta = {}, {}
        for aum in aums:
            cm = cost_matrix(ctx, aum * 1e8)
            rb = replay(base_sc, p, top_n=TOP_N, rebal=rebal, cost=cm)
            ra = replay(aug_sc, p, top_n=TOP_N, rebal=rebal, cost=cm)
            ex_b = _geom(rb.net) - _geom(ctx.bench.reindex(rb.net.index))
            ex_a = _geom(ra.net) - _geom(ctx.bench.reindex(ra.net.index))
            key = f"{aum:g}亿"
            cost_res[key] = {"excess": ex_a * 100}
            d = (ra.net - rb.net).dropna()
            delta[key] = {
                "delta": (ex_a - ex_b) * 100,
                "t": float(d.mean() / d.std() * np.sqrt(len(d))) if d.std() > 0 else np.nan,
            }
        res["cost"] = cost_res
        res["delta"] = delta
        k0 = list(delta)[0]
        res["delta_excess"] = delta[k0]["delta"]
        res["delta_t"] = delta[k0]["t"]
        res["delta_ok"] = bool(res["delta_excess"] > 0 and abs(res["delta_t"]) > 2)

    if res["delta_ok"] is None:
        res["usable"] = None
    else:
        res["usable"] = bool(res["pass_gates"] and res["delta_ok"])
    res["pass_all"] = bool(res["usable"]) if res["usable"] is not None else False
    _log_trial(res)
    return res


# ==================== 日志 / BH / 报告 ====================
def _f(v, n=4):
    return (round(float(v), n)
            if isinstance(v, (int, float, np.floating)) and np.isfinite(v)
            else np.nan)


def _log_trial(res):
    OUT.mkdir(parents=True, exist_ok=True)
    h0 = res["h0"]
    row = {
        "name": res["name"], "rebal": res["rebal"], "h0": h0,
        "coverage": _f(res["coverage"], 3),
        "first_year": res.get("first_effective_year"),
        "cov_trend": _f(res.get("coverage_trend", np.nan), 3),
        "ic": _f(res["ic"][h0]["ic"], 5),
        "icir": _f(res["ic"][h0]["icir"], 4),
        "t_nw": _f(res["ic"][h0]["t_nw"], 2),
        "resid_ic": _f(res.get("resid_ic", np.nan), 5),
        "resid_t": _f(res.get("resid_ic_t", np.nan), 2),
        "corr_max": _f(res["corr_max"], 3),
        "monotone": _f(res.get("monotone", np.nan), 2),
        "pass_gates": bool(res["pass_gates"]),
        "delta_ok": res["delta_ok"],
        "usable": res["usable"],
    }
    df = pd.DataFrame([row])
    if TRIALS.exists():
        df = pd.concat([pd.read_csv(TRIALS), df], ignore_index=True)
    df.to_csv(TRIALS, index=False, encoding="utf-8-sig")


def bh_from_log(alpha=0.10, n_eff=1500, dedupe=True):
    """
    ★ 用「残差 IC 的单边 p」：原假设是「该因子有正增量」。
      原始 IC 显著但残差为负的因子（如 amihud）必须在这里被判死。
      t_nw 是原始 IC 的 t，不能直接拿来算这个 p。
    """
    if not TRIALS.exists():
        return None
    df = pd.read_csv(TRIALS).dropna(subset=["ic"])
    if dedupe and {"name", "rebal", "h0"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["name", "rebal", "h0"], keep="last")

    tcol = "resid_t" if "resid_t" in df.columns else "t_nw"
    df["t_used"] = pd.to_numeric(df[tcol], errors="coerce").fillna(0.0)
    df["p"] = [_p_one_sided(t, n_eff) for t in df["t_used"]]
    df["q_bh"] = _bh(df["p"].to_numpy())
    df["pass_bh"] = df["q_bh"] <= alpha
    df["n_trials"] = int(len(df))
    return df


def format_report(res):
    h0 = res["h0"]
    base = res["ic"][h0]
    bars = "=" * 72
    warn = ""
    if res["low_cov_years"]:
        warn += "  ⚠ 早年被截断，聚合统计偏晚"
    L = [bars, f"  {res['name']}   (rebal={res['rebal']}, h0={h0})", bars,
         f"  覆盖 {res['coverage']:.1%}   尺度 std {res['scale_std']:.2f}"
         f"{'' if res['scale_ok'] else '  ⚠'}",
         f"  有效起始年 {res['first_effective_year']}   "
         f"低覆盖年份 {res['low_cov_years'] or '无'}{warn}",
         f"  逐年覆盖 {res['coverage_by_year']}"]
    if res["coverage_drifting"]:
        L.append(f"  ⚠ 覆盖漂移 {res['coverage_trend']:+.2f}"
                 f"（首年→末年）—— IC 可能是 regime 效应，不是稳定属性")
    L += ["",
          "  IC      " + "  ".join(f"h={h}:{v['ic']:+.4f}"
                                   for h, v in res["ic"].items()),
          f"  最佳期限 h={res['best_h']}（仅信息，门禁按 h0={h0}）",
          f"  ICIR(h={h0}) {base['icir']:.3f}   NW t {base['t_nw']:+.2f}",
          f"  残差 IC {res['resid_ic']:+.5f} (t {res['resid_ic_t']:+.2f})"
          f"   → 判定依据：{res['resid_ic_via']}",
          f"  与 book 最大相关 {res['corr_max']:.3f}"
          f"  ({max(res['corr_with_book'], key=lambda k: abs(res['corr_with_book'][k]))})",
          f"  五分组(bp/期) {res['quintile_bp']}  spread {res['spread_bp']:+.1f}"
          f"  单调 {res['monotone']:.2f}"]

    if res.get("deep_run"):
        L += ["",
              f"  换手 {res['turnover']:.0%}",
              f"  毛超额 {res['gross_excess']*100:+.2f}pp   "
              f"12bp 净超额 {res['excess_12bp']*100:+.2f}pp   "
              f"（12bp 成本 {res['cost_pp_12bp']*100:.2f}pp）",
              f"  方差权重 {res['w_new']:.1%}"
              f"（naive 0.8/0.2 只有 {res['w_new_naive_share']:.1%}）",
              "  " + "  ".join(f"{k}: {v['excess']:+.2f}pp"
                               for k, v in res["cost"].items()),
              f"  等权并入 Δ {res['delta_excess']:+.2f}pp  t {res['delta_t']:+.2f}"
              f"   → {'✅ 可并入' if res['delta_ok'] else '❌ 稀释'}"]

    def _tri(v):
        return "—（未跑 deep）" if v is None else ("✅" if v else "❌")

    L += ["",
          "  质量门 " + "  ".join(f"{'✅' if v else '❌'}{k}"
                                 for k, v in res["gate"].items()),
          f"  → 质量门 {'✅ 全过' if res['pass_gates'] else '❌ 未过'}"
          f"；可并入 {_tri(res['delta_ok'])}"
          f"；可用 {_tri(res['usable'])}"]
    return "\n".join(L)
