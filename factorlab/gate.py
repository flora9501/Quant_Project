# factorlab/gate.py
"""
因子准入检验。全项目唯一实现。

一道 gate 输出
    覆盖率 · IC/ICIR/t（多 horizon）· 分年 IC 符号
    与现有 4 因子的截面相关性
    残差 IC（对 4 因子回归后）
    五分组（均值 + 中位数）与 **Q5−Q1 多空**
    组合边际：加入后与基线的配对 t

用法
    from factorlab.core import load_panel
    from factorlab.gate import evaluate
    p = load_panel()
    r = evaluate("acc_ttm", factor_wide, p)

口径说明
    ic["t"] 是 naive t：对**每日** IC 序列算 mean/std*sqrt(n)。
    h>1 时前瞻收益是重叠的，序列自相关约 AR(h)，naive t 高估约 sqrt(h) 倍。
    因此另给 res.t_nw / res.p_nw（Newey-West，lag=h-1）用于多重检验校正。
    ic["t"] 保留原样，避免动已发表的 0.0093 / 0.0229。

    ls_bp / ls_t / ls_t_nw 是五分组多空（Q5−Q1），复现论文时的主判据。
    ls_bp 单位是 bp/期（h=5 即每 5 个交易日），不是年化。
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from factorlab.core import (FACTORS, TOP_N, REBAL, THRESH,
                            composite, smooth_score, replay)

GATE_IC = 0.02
GATE_ICIR = 0.30
GATE_RESID_IC = 0.015
GATE_MAX_CORR = 0.30
GATE_MARGINAL_T = 2.0


# ==================== 工具 ====================
def forward_returns(panel, h=5):
    px = (1 + panel.ret).cumprod()
    return (px.shift(-h) / px - 1).reindex(index=panel.idx, columns=panel.cols)


def rank_ic(frame, x, y, min_n=50):
    """frame 长表，含 trade_date / x / y；返回每日 Spearman IC 序列"""
    s = frame[["trade_date", x, y]].dropna()
    if s.empty:
        return pd.Series(dtype=float)
    s = s.assign(rx=s.groupby("trade_date")[x].rank(),
                 ry=s.groupby("trade_date")[y].rank())
    return (s.groupby("trade_date")
             .apply(lambda z: z.rx.corr(z.ry) if len(z) >= min_n else np.nan)
             .dropna())


def _long(wide, name, panel, h=5):
    fwd = forward_returns(panel, h)
    d = wide.stack().rename(name).reset_index()
    d.columns = ["trade_date", "ts_code", name]
    d["fwd"] = fwd.stack().reindex(
        pd.MultiIndex.from_frame(d[["trade_date", "ts_code"]])).values
    return d.dropna(subset=[name, "fwd"])


def _ic_stats(series, label=""):
    n = len(series)
    if n == 0:
        return {"IC": np.nan, "IC_std": np.nan, "ICIR": np.nan, "t": np.nan, "days": 0}
    m, sd = float(series.mean()), float(series.std())
    return {"IC": m, "IC_std": sd,
            "ICIR": m / sd if sd > 0 else np.nan,
            "t": m / sd * np.sqrt(n) if sd > 0 else np.nan,
            "days": n}


def _nw_t(series, lags):
    """Newey-West(Bartlett) t：校正重叠 h 日收益造成的自相关。lags = h-1。"""
    x = pd.Series(series).dropna().to_numpy(dtype=float)
    n = x.size
    if n < 20 or lags < 0:
        return np.nan
    e = x - x.mean()
    g = float((e ** 2).sum() / n)
    for L in range(1, int(lags) + 1):
        w = 1.0 - L / (lags + 1.0)
        g += 2.0 * w * float((e[L:] * e[:-L]).sum() / n)
    if g <= 0:
        return np.nan
    return float(x.mean() / np.sqrt(g / n))


# ==================== 主函数 ====================
@dataclass
class GateResult:
    name: str
    coverage: float = np.nan
    ic: dict = field(default_factory=dict)
    yearly: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    corr: pd.DataFrame = field(default_factory=pd.DataFrame)
    max_corr: float = np.nan
    resid_ic: float = np.nan
    resid_icir: float = np.nan
    quintiles: pd.DataFrame = field(default_factory=pd.DataFrame)
    marginal_t: float = np.nan
    marginal_delta: float = np.nan
    verdict: str = ""
    # 新增：重叠校正后的推断（不影响既有字段）
    t_nw: float = np.nan
    p_nw: float = np.nan
    p_naive: float = np.nan
    nw_lags: int = 0
    # 新增：五分组多空
    ls_bp: float = np.nan
    ls_t: float = np.nan
    ls_t_nw: float = np.nan

    def summary(self):
        return pd.Series({
            "因子": self.name,
            "覆盖率": round(self.coverage, 3),
            "IC": round(self.ic.get("IC", np.nan), 4),
            "ICIR": round(self.ic.get("ICIR", np.nan), 3),
            "t": round(self.ic.get("t", np.nan), 2),
            "分年同号": bool((self.yearly > 0).all() or (self.yearly < 0).all()),
            "最大相关": round(self.max_corr, 3),
            "残差IC": round(self.resid_ic, 4),
            "边际t": round(self.marginal_t, 2),
            "裁决": self.verdict,
        })


def evaluate(name, factor, panel, h=5, with_marginal=True, verbose=True,
             require_positive=False):
    """factor: (dates × codes)，已是截面 z-score

    require_positive: True 时要求 IC > 0（因子已按"越大越好"对齐，
                      负号视为构造错误，不允许用 abs() 混过）。
                      注意：做论文复现时**不要**用它 —— 方向应由论文登记表判断，
                      否则 George & Hwang 这类"原文预期反向"的因子会被误杀。
    """
    factor = factor.reindex(index=panel.idx, columns=panel.cols)
    f = factor.where(panel.mask)

    res = GateResult(name=name)
    res.coverage = float(f.notna().values.mean())

    # ---- 单因子 IC ----
    long = _long(f, "x", panel, h)
    ic_series = rank_ic(long, "x", "fwd")
    res.ic = _ic_stats(ic_series)

    # Newey-West：滞后 h-1
    res.nw_lags = max(int(h) - 1, 0)
    res.t_nw = _nw_t(ic_series, res.nw_lags)
    if not np.isfinite(res.t_nw):
        res.t_nw = res.ic.get("t", np.nan)
    res.p_nw = (float(2 * (1 - stats.norm.cdf(abs(res.t_nw))))
                if np.isfinite(res.t_nw) else np.nan)
    res.p_naive = (float(2 * (1 - stats.norm.cdf(abs(res.ic.get("t", np.nan)))))
                   if np.isfinite(res.ic.get("t", np.nan)) else np.nan)

    yr = ic_series.groupby(ic_series.index.str[:4]).mean()
    res.yearly = yr

    # ---- 与现有 4 因子的相关性 ----
    rows = []
    for b in FACTORS:
        bf = panel.fac[b].where(panel.mask)
        m = pd.concat([f.stack().rename("a"), bf.stack().rename("b")],
                      axis=1, join="inner").dropna()
        if m.empty:
            continue
        r = (m.groupby(level=0)
              .apply(lambda z: z.a.corr(z.b, method="spearman"))
              .dropna())
        rows.append({"vs": b, "corr": float(r.mean()), "abs_corr": float(r.abs().mean())})
    res.corr = pd.DataFrame(rows)
    res.max_corr = float(res.corr["abs_corr"].max()) if not res.corr.empty else np.nan

    # ---- 残差 IC ----
    base = [panel.fac[b].where(panel.mask) for b in FACTORS]
    stack = pd.concat([x.stack().rename(b) for x, b in zip(base, FACTORS)]
                      + [f.stack().rename("y")], axis=1, join="inner").dropna()

    def _resid(g):
        X = g[FACTORS].to_numpy()
        y = g["y"].to_numpy()
        X = np.c_[np.ones(len(g)), X]
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return pd.Series(y - X @ beta, index=g.index)

    r = (stack.groupby(level=0, group_keys=False)
              .apply(_resid)
              .rename("r")
              .reset_index())
    r.columns = ["trade_date", "ts_code", "r"]

    fwd_long = forward_returns(panel, h).stack().rename("fwd").reset_index()
    fwd_long.columns = ["trade_date", "ts_code", "fwd"]
    r = r.merge(fwd_long, on=["trade_date", "ts_code"], how="left")

    ric = rank_ic(r, "r", "fwd")
    res.resid_ic = float(ric.mean()) if len(ric) else np.nan
    res.resid_icir = (float(ric.mean() / ric.std())
                      if len(ric) and ric.std() > 0 else np.nan)

    # ---- 五分组 ----
    s = long.copy()
    s["g"] = s.groupby("trade_date")["x"].transform(
        lambda z: pd.qcut(z.rank(method="first"), 5, labels=False))
    q = s.groupby("g")["fwd"].agg(["mean", "median", "count"])
    q["mean_bp"] = (q["mean"] * 1e4).round(1)
    q["median_bp"] = (q["median"] * 1e4).round(1)
    res.quintiles = q[["mean_bp", "median_bp", "count"]]

    # ---- 五分组多空（Q5−Q1），bootstrap 只在需要时另做 ----
    gm = q["mean"]
    hi, lo = gm.index.max(), gm.index.min()
    sp = (s[s["g"] == hi].groupby("trade_date")["fwd"].mean()
          - s[s["g"] == lo].groupby("trade_date")["fwd"].mean()).dropna()
    if len(sp) > 20 and sp.std() > 0:
        res.ls_bp = float(gm.loc[hi] - gm.loc[lo]) * 1e4      # bp / 期
        res.ls_t = float(sp.mean() / sp.std() * np.sqrt(len(sp)))
        res.ls_t_nw = _nw_t(sp, res.nw_lags)

    # ---- 组合边际 ----
    if with_marginal:
        base_score = smooth_score(composite(panel))
        r_base = replay(base_score, panel)

        aug_score = smooth_score(composite(panel) + f.fillna(0.0))
        r_aug = replay(aug_score, panel)

        diff = (r_aug.net - r_base.net).dropna()
        if len(diff) > 100 and diff.std() > 0:
            res.marginal_t = float(diff.mean() / diff.std() * np.sqrt(len(diff)))
            res.marginal_delta = float(r_aug.ann() - r_base.ann())

    # ---- 裁决 ----
    same_sign = bool((yr > 0).all() or (yr < 0).all())
    ok = (
        res.max_corr < GATE_MAX_CORR
        and abs(res.ic.get("IC", 0)) >= GATE_IC
        and abs(res.ic.get("ICIR", 0)) >= GATE_ICIR
        and same_sign
        and abs(res.resid_ic) >= GATE_RESID_IC
        and (not require_positive or res.ic.get("IC", 0.0) > 0)
        and (not with_marginal or res.marginal_t >= GATE_MARGINAL_T)
    )
    res.verdict = "PASS" if ok else "REJECT"

    if verbose:
        print(f"\n{'='*70}\n【{name}】\n{'='*70}")
        print(f"覆盖率 {res.coverage:.1%}")
        print(f"IC {res.ic['IC']:+.4f}  ICIR {res.ic['ICIR']:+.3f}  "
              f"t {res.ic['t']:.2f}  days {res.ic['days']}")
        print(f"NW(lag={res.nw_lags}) t {res.t_nw:.2f}  p {res.p_nw:.4g}  "
              f"(naive p {res.p_naive:.4g})")
        print(f"分年 IC:\n{yr.round(4).to_string()}")
        print(f"\n相关性:\n{res.corr.round(3).to_string(index=False)}")
        print(f"\n残差 IC {res.resid_ic:+.4f}（ICIR {res.resid_icir:+.3f}）")
        print(f"\n五分组:\n{res.quintiles.to_string()}")
        print(f"多空 Q5-Q1: {res.ls_bp:+.0f} bp/期  "
              f"t {res.ls_t:.2f}  NW {res.ls_t_nw:.2f}")
        if with_marginal:
            print(f"\n组合边际：Δ年化 {res.marginal_delta:+.2%}，"
                  f"配对 t {res.marginal_t:+.2f}")
        print(f"\n裁决：{res.verdict}")

    return res


def gate_table(results):
    return pd.DataFrame([r.summary() for r in results]).set_index("因子")
