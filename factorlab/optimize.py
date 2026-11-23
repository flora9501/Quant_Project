# factorlab/optimize.py
"""
均值方差优化器（cvxpy）+ 因子风险模型（4 风格 + 行业 + 对角特质）。

v5 修改
  ★ solve_mv 默认 solver 顺序改为 ECOS → OSQP
      OSQP 在「alpha'w 与 risk 差 3 个数量级」这类病态问题上常返回
      optimal_inaccurate，并刷几百行 UserWarning。
  ★ warnings.catch_warnings 抑制求解器警告
  ★ 返回前做可行性检查：约束违反 > 1e-4 就换下一个 solver，
      不把 inaccurate 的烂解当结果
  ★ 新增 SOLVERS 字典，供 CLI 选 --solver ecos|osqp|scs

v4：exposures 末尾 nan_to_num（缺失暴露=行业内均值）
v3：冷启动跳过换手约束（w_0=0 时 ||w||₁=1 > τ，必然不可行）
v2：estimate_risk 改 numpy（几十分钟 → 10–20 秒）、candidate_block
"""
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .core import Panel, FACTORS

try:
    import cvxpy as cp
except ImportError:                                     # pragma: no cover
    cp = None

SOLVERS = {}
if cp is not None:
    for _n, _s in (("ecos", cp.ECOS), ("osqp", cp.OSQP), ("scs", cp.SCS)):
        try:
            SOLVERS[_n] = _s
        except Exception:
            pass


def industry_groups(p: Panel, min_count: int = 20) -> pd.Series:
    ind = p.industry.reindex(p.cols).fillna("OTHER")
    cnt = ind.value_counts()
    return ind.where(ind.isin(cnt[cnt >= min_count].index), "OTHER")


def _zscore(v: np.ndarray) -> np.ndarray:
    v = v.astype(float).copy()
    med = np.nanmedian(v)
    mad = np.nanmedian(np.abs(v - med)) * 1.4826
    if np.isfinite(mad) and mad > 0:
        v = np.clip(v, med - 3 * mad, med + 3 * mad)
    m, s = np.nanmean(v), np.nanstd(v)
    return (v - m) / s if s > 0 else np.zeros_like(v)


def exposures(p: Panel, groups: pd.Series, gnames, d, names,
              factors=FACTORS) -> np.ndarray:
    """返回 (len(names), K)，行顺序与 names 一致，且保证无 NaN。"""
    Z = np.column_stack([_zscore(p.fac[k].loc[d, names].to_numpy())
                         for k in factors])
    g = groups.reindex(names).to_numpy()
    for j in range(Z.shape[1]):
        fin = np.isfinite(Z[:, j])
        sw = pd.Series(np.where(fin, Z[:, j], 0.0)).groupby(g).transform("sum").to_numpy()
        cw = pd.Series(fin.astype(float)).groupby(g).transform("sum").to_numpy()
        mu = np.where(cw > 0, sw / np.maximum(cw, 1e-9), 0.0)
        Z[:, j] = np.where(fin, Z[:, j] - mu, np.nan)

    # 缺失暴露 → 0：行业内去均值后均值=0，所以 0 == 该行业平均暴露
    Z = np.nan_to_num(Z, nan=0.0)

    D = pd.get_dummies(pd.Series(g)).reindex(
        columns=list(gnames), fill_value=0).to_numpy(dtype=float)
    return np.column_stack([Z, D])


def candidate_block(Xall: np.ndarray, allnames, cand) -> np.ndarray:
    """按 cand 自身顺序取行并转置成 (K, m)。不能用 np.isin（顺序会错配）。"""
    pos = {c: i for i, c in enumerate(allnames)}
    missing = [c for c in cand if c not in pos]
    if missing:
        raise KeyError(f"{len(missing)} 个候选不在基准池里，例如 {missing[:3]}")
    idx = np.array([pos[c] for c in cand], dtype=int)
    return Xall[idx].T


@dataclass
class RiskModel:
    F: pd.DataFrame
    L: np.ndarray
    spec_var: pd.DataFrame
    groups: pd.Series
    gnames: list
    factors: list

    @property
    def k(self):
        return len(self.factors) + len(self.gnames)


def estimate_risk(p: Panel, halflife: int = 90, verbose: bool = True) -> RiskModel:
    groups = industry_groups(p)
    gnames = sorted(groups.unique())
    G, K, nf = len(gnames), len(FACTORS) + len(gnames), len(FACTORS)
    gpos = {g: i for i, g in enumerate(gnames)}

    T, N = len(p.idx), len(p.cols)
    spec = np.full((T, N), np.nan)
    fmat = np.zeros((T, K))
    mask_v = p.mask.to_numpy()
    ret_v = p.ret.to_numpy()
    fac_v = [p.fac[k].to_numpy() for k in FACTORS]
    gi = np.array([gpos.get(g, 0) for g in groups.reindex(p.cols).to_numpy()])

    for i, d in enumerate(p.idx):
        sel = mask_v[i]
        if sel.sum() < 100:
            continue
        cols_i = np.flatnonzero(sel)
        n = len(cols_i)
        g_i = gi[cols_i]

        Z = np.empty((n, nf))
        for j in range(nf):
            v = fac_v[j][i, cols_i]
            med = np.nanmedian(v)
            mad = np.nanmedian(np.abs(v - med)) * 1.4826
            if np.isfinite(mad) and mad > 0:
                v = np.clip(v, med - 3 * mad, med + 3 * mad)
            m_, s_ = np.nanmean(v), np.nanstd(v)
            Z[:, j] = (v - m_) / s_ if s_ > 0 else 0.0
            fin = np.isfinite(Z[:, j])
            sw = np.bincount(g_i, weights=np.where(fin, Z[:, j], 0.0), minlength=G)
            cw = np.bincount(g_i, weights=fin.astype(float), minlength=G)
            mu = np.where(cw > 0, sw / np.maximum(cw, 1e-9), 0.0)
            Z[:, j] = np.where(fin, Z[:, j] - mu[g_i], np.nan)

        D = np.zeros((n, G))
        D[np.arange(n), g_i] = 1.0
        X = np.column_stack([Z, D])

        r = ret_v[i, cols_i]
        ok = np.isfinite(r) & np.isfinite(X).all(1)
        if ok.sum() < 100:
            continue
        b, *_ = np.linalg.lstsq(X[ok], r[ok], rcond=None)
        fmat[i] = b
        spec[i, cols_i[ok]] = r[ok] - X[ok] @ b

        if verbose and i % 400 == 0:
            print(f"     risk {i}/{T}", flush=True)

    lam = 0.5 ** (1 / halflife)
    w = lam ** np.arange(T)[::-1]
    w = w / w.sum()
    Zc = fmat - (w @ fmat)
    Fm = (Zc * w[:, None]).T @ Zc + np.eye(K) * 1e-10
    F = pd.DataFrame(Fm, columns=[f"f{j}" for j in range(K)])
    L = np.linalg.cholesky(Fm)

    sv = pd.DataFrame(spec, index=p.idx, columns=p.cols)
    sv = sv.pow(2).ewm(halflife=halflife, min_periods=60).mean()
    sv = sv.clip(lower=(0.02 ** 2) / 252)

    if verbose:
        print(f"   风险模型：{K} 因子（{nf} 风格 + {G} 行业），"
              f"F 对角均值 {np.diag(Fm).mean():.2e}")
    return RiskModel(F=F, L=L, spec_var=sv, groups=groups, gnames=gnames,
                     factors=list(FACTORS))


def _check_shapes(alpha, Xc, xb, d_c, wb_c, w0_c, band, fn):
    m, K = len(alpha), len(xb)
    if Xc.shape != (K, m):
        raise ValueError(
            f"[{fn}] Xc 应为 (K,m)=({K},{m})，收到 {Xc.shape}；"
            f"exposures() 返回 (n,K)，请用 candidate_block()")
    for nm, arr in (("Xc", Xc), ("xb", xb), ("d_c", d_c),
                    ("wb_c", wb_c), ("w0_c", w0_c)):
        if not np.isfinite(arr).all():
            raise ValueError(f"[{fn}] {nm} 含 NaN/inf")
    for nm, arr in (("d_c", d_c), ("wb_c", wb_c), ("w0_c", w0_c)):
        if len(arr) != m:
            raise ValueError(f"[{fn}] {nm} 长度 {len(arr)} ≠ m={m}")
    if len(band) != K:
        raise ValueError(f"[{fn}] band 长度 {len(band)} ≠ K={K}")


def solve_mv(alpha, Xc, xb, L, d_c, wb_c, w0_c, band, w_max, tau, lam,
             solver=None, verbose_fail=False):
    if cp is None:
        raise ImportError("需要 cvxpy：.venv/bin/python -m pip install cvxpy")
    _check_shapes(alpha, Xc, xb, d_c, wb_c, w0_c, band, "solve_mv")

    m = len(alpha)
    w = cp.Variable(m, nonneg=True)
    dev = Xc @ w - xb
    risk = (cp.sum_squares(L @ dev)
            + cp.sum(cp.multiply(d_c, cp.square(w - wb_c))))
    obj = cp.Maximize(cp.sum(cp.multiply(alpha, w)) - 0.5 * lam * risk)

    cons = [cp.sum(w) == 1, w <= w_max, cp.abs(dev) <= band]
    cold = float(np.sum(w0_c)) <= 1e-12
    if not cold:
        cons.append(cp.norm1(w - w0_c) <= tau)

    prob = cp.Problem(obj, cons)
    solvers = [solver] if solver is not None else \
        ([SOLVERS["ecos"]] if "ecos" in SOLVERS else []) + \
        ([SOLVERS["osqp"]] if "osqp" in SOLVERS else [])
    if not solvers:
        raise ImportError("没有可用的 cvxpy solver（ecos/osqp）")

    last = None
    for s in solvers:
        name = getattr(s, "__name__", str(s))
        with warnings.catch_warnings():           # OSQP 的 inaccurate 警告太吵
            warnings.simplefilter("ignore")
            try:
                prob.solve(solver=s, warm_start=True, verbose=False)
            except Exception as e:
                last = f"{name}: {type(e).__name__}: {e}"
                continue

        if w.value is None:
            last = f"{name}: status={prob.status}"
            continue
        v = np.clip(w.value, 0.0, None)
        if v.sum() <= 0:
            last = f"{name}: w.sum()=0"
            continue
        v = v / v.sum()

        # 可行性检查：inaccurate 的烂解不要
        viol = max(abs(float(v.sum()) - 1.0),
                   float(v.max() - w_max),
                   float(np.max(np.abs(Xc @ v - xb) - band)))
        if not cold:
            viol = max(viol, float(np.abs(v - w0_c).sum() - tau))
        if viol > 1e-4:
            last = f"{name}: 约束违反 {viol:.1e}"
            continue
        return v

    if verbose_fail:
        print(f"    [solve_mv 失败] {last}", flush=True)
    return None


def diagnose(alpha, Xc, xb, L, d_c, wb_c, w0_c, band, w_max, tau,
             names, labels):
    """等权解上的约束松弛度。全部松弛 → QP 没有改进空间。"""
    _check_shapes(alpha, Xc, xb, d_c, wb_c, w0_c, band, "diagnose")
    m = Xc.shape[1]
    w_eq = np.full(m, 1.0 / m)
    dev = Xc @ w_eq - xb
    te2 = float(np.sum((L @ dev) ** 2) + np.sum(d_c * (w_eq - wb_c) ** 2))
    ratio = np.abs(dev) / np.maximum(band, 1e-12)
    return {
        "te_ann": float(np.sqrt(max(te2, 0)) * np.sqrt(252)),
        "exposure_slack": pd.Series(band - np.abs(dev), index=labels),
        "exposure_ratio": pd.Series(ratio, index=labels),
        "worst_exposure": labels[int(np.argmax(ratio))],
        "worst_ratio": float(np.max(ratio)),
        "n_binding": int((ratio > 0.9).sum()),
        "turnover": float(np.abs(w_eq - w0_c).sum()),
        "turnover_budget": tau,
        "max_w": float(w_eq.max()),
        "w_max": w_max,
        "n_names": m,
    }
