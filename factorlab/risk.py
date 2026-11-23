# factorlab/risk.py
"""
CNE5 公开近似的截面风险模型。

10 个风格（数据可得范围内的近似）
  Size        ln(circ_mv)
  Beta        252d 滚动 beta（对等权市场）
  Momentum    252d 收益剔除最近 21d
  ResidVol    HSIGMA 解析近似 sqrt(var_i − β²·var_m)，252d
  NLS         Size³ 对 Size 正交化
  BtP         1/PB
  Liquidity   ln(21d 平均换手率)
  EarnYield   1/PE_TTM（仅正 PE）
  Growth      资产增速（复用 q_ag）
  Leverage    负债/总资产（年报 PIT，缺失则自动跳过并告警）

估计（顺序法，稳定且易解释）
  ① f_ind   = 行业内市值加权平均收益
  ② X~      = 行业市值加权去均值后的风格暴露
  ③ f_style = WLS(r − f_ind ~ X~)，权重 ∝ sqrt(circ_mv)
  ④ u       = (r − f_ind) − X~·f_style
  恒等式：r ≈ f_ind + X~·f_style + u

风险
  F = EWMA(halflife=90d) 于 [f_style, f_ind] 的协方差
  D = EWMA(halflife=90d) 于 u²，年化下限 2%
  TE² = Δe' F Δe + w_active' D w_active
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .core import Panel, load_panel, benchmark, COST
from .factors import load_cached

DATA = Path("data")
RAW = DATA / "raw"
FINA = DATA / "raw_fina"
OUT = DATA / "output_zoo"

STYLES = ["Size", "Beta", "Momentum", "ResidVol", "NLS",
          "BtP", "Liquidity", "EarnYield", "Growth", "Leverage"]
HALFLIFE = 90
SPEC_FLOOR = 0.02 / np.sqrt(252)      # 日频特质波动下限


# ==================== 截面标准化 ====================
def _cs_standardize(x, w=None, mad=3.0):
    """逐日 MAD 去极值 + (加权) z-score；w 为 DataFrame 权重（如 sqrt(circ_mv)）"""
    arr = x.to_numpy(dtype=float)
    med = np.nanmedian(arr, axis=1, keepdims=True)
    mad_ = np.nanmedian(np.abs(arr - med), axis=1, keepdims=True) * 1.4826
    good = np.isfinite(mad_) & (mad_ > 0)
    lo = np.where(good, med - mad * mad_, -np.inf)
    hi = np.where(good, med + mad * mad_, np.inf)
    arr = np.clip(arr, lo, hi)

    if w is None:
        mu = np.nanmean(arr, axis=1, keepdims=True)
        var = np.nanvar(arr, axis=1, keepdims=True)
    else:
        wa = w.to_numpy(dtype=float)
        wa = np.where(np.isfinite(arr), wa, np.nan)
        sw = np.nansum(wa, axis=1, keepdims=True)
        mu = np.nansum(arr * wa, axis=1, keepdims=True) / sw
        var = np.nansum(wa * (arr - mu) ** 2, axis=1, keepdims=True) / sw
    sd = np.sqrt(var)
    out = (arr - mu) / np.where(sd > 0, sd, np.nan)
    return pd.DataFrame(out, index=x.index, columns=x.columns)


# ==================== 风格构造 ====================
def build_styles(p: Panel, daily_basic=None, years=range(2018, 2026), verbose=True):
    cols = p.cols
    mask = p.mask
    cap = daily_basic["circ_mv"] if daily_basic is not None else None
    tmv = daily_basic["total_mv"] if daily_basic is not None else None

    def log(m):
        if verbose:
            print(m, flush=True)

    S = {}

    # Size
    S["Size"] = np.log(cap.replace(0, np.nan) * 1e4)

    # Beta / ResidVol（对等权市场）
    mkt = p.ret.where(mask).mean(axis=1)
    win, minp = 252, 120
    mv = mkt.rolling(win, min_periods=minp).var()
    cov = p.ret.rolling(win, min_periods=minp).cov(mkt)
    beta = cov.div(mv, axis=0)
    S["Beta"] = beta
    var_i = p.ret.rolling(win, min_periods=minp).var()
    hsig = np.sqrt((var_i - beta.pow(2).mul(mv, axis=0)).clip(lower=0))
    S["ResidVol"] = hsig

    # Momentum
    cp = (1 + p.ret.fillna(0)).cumprod()
    S["Momentum"] = cp.shift(21) / cp.shift(252) - 1

    # Liquidity
    if daily_basic is not None:
        to = daily_basic["turnover_rate"].where(mask)
        S["Liquidity"] = np.log(to.replace(0, np.nan).rolling(21, min_periods=10).mean())

    # BtP / EarnYield
    if daily_basic is not None:
        pb = daily_basic.get("pb")
        if pb is not None:
            S["BtP"] = 1.0 / pb.where(pb > 0)
        pe = daily_basic.get("pe_ttm")
        if pe is not None:
            S["EarnYield"] = 1.0 / pe.where(pe > 0)

    # Growth（复用已有 q_ag）
    qag = load_cached("q_ag")
    if qag is not None:
        S["Growth"] = qag.reindex(index=p.idx, columns=cols)

    # Leverage（年报 PIT）
    lev = _load_leverage(p)
    if lev is not None:
        S["Leverage"] = lev
    else:
        log("   ⚠ 缺 fina_indicator，Leverage 跳过 → 风险模型为 9 风格")

    # NLS（在 Size 之后）
    if "Size" in S:
        z = _cs_standardize(S["Size"].where(mask))
        S["NLS"] = z.pow(3)
        # 对 Size 正交化
        S["NLS"] = _orth_to(S["NLS"], S["Size"], mask)

    log(f"   风格构建完成：{list(S)}")
    return S


def _orth_to(y, x, mask, min_n=100):
    """逐日把 y 对 [1, x] 回归取残差"""
    out = y.where(mask) * np.nan
    for d in y.index:
        yy = y.loc[d].where(mask.loc[d])
        xx = x.loc[d].where(mask.loc[d])
        df = pd.concat([yy.rename("y"), xx.rename("x")], axis=1).dropna()
        if len(df) < min_n:
            continue
        A = np.column_stack([np.ones(len(df)), df["x"].to_numpy()])
        b = np.linalg.lstsq(A, df["y"].to_numpy(), rcond=None)[0]
        out.loc[d, df.index] = df["y"].to_numpy() - A @ b
    return out


def _load_leverage(p):
    files = sorted(FINA.glob("fina_indicator_part*.parquet"))
    if not files:
        return None
    try:
        fi = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        fi = fi[fi["end_date"].astype(str).str.endswith("1231")]
        fi = fi[["ts_code", "ann_date", "debt_to_assets"]].dropna()
        if fi.empty:
            return None
        fi["ann_date"] = pd.to_datetime(fi["ann_date"]).dt.strftime("%Y%m%d")
        w = fi.pivot_table(index="ann_date", columns="ts_code",
                           values="debt_to_assets", aggfunc="last")
        idx = pd.Index(sorted(set(w.index) | set(p.idx)))
        w = w.reindex(idx).ffill().reindex(p.idx)
        return w.reindex(columns=p.cols)
    except Exception as e:
        print(f"   ⚠ Leverage 读取失败：{e}")
        return None


# ==================== 暴露矩阵（长表） ====================
def build_exposures(p: Panel, styles: dict, min_ind=20, verbose=True):
    mask = p.mask
    use = [k for k in STYLES if k in styles]

    # 行业分组：小行业并入 OTHER
    ind = p.industry.reindex(p.cols).fillna("OTHER")
    cnt = ind.value_counts()
    keep = cnt[cnt >= min_ind].index
    ind_grp = ind.where(ind.isin(keep), "OTHER")

    std = {}
    for k in use:
        std[k] = _cs_standardize(styles[k].where(mask))
    wide = pd.concat({k: v.where(mask).stack() for k, v in std.items()}, axis=1)
    wide.index.names = ["trade_date", "ts_code"]
    wide["industry"] = ind_grp.reindex(wide.index.get_level_values(1)).to_numpy()

    if verbose:
        print(f"   暴露矩阵 {wide.shape[0]:,} 行 × {len(use)} 风格 + 行业"
              f"（{wide['industry'].nunique()} 组）")
    return wide, use


# ==================== 估计 ====================
def estimate(p: Panel, wide, use, verbose=True):
    cap = None
    dates = p.idx
    f_style = pd.DataFrame(np.nan, index=dates, columns=use)
    f_ind = {}
    spec = {}
    ret = p.ret

    for d in dates:
        try:
            g = wide.xs(d, level=0)
        except KeyError:
            continue
        if len(g) < 100:
            continue
        r = ret.loc[d].reindex(g.index)
        df = g.join(r.rename("r"), how="inner").dropna(subset=["r"])
        if len(df) < 100:
            continue
        w = df["Size"].pipe(lambda s: np.exp(s))  # 用 exp(Size)∝市值 作权重
        w = w.where(np.isfinite(w), w.median())
        w = w / w.sum()

        # ① 行业收益
        ir = df.groupby("industry")["r"].apply(lambda s: np.average(
            s, weights=w.reindex(s.index)))
        ind_ret = df["industry"].map(ir).to_numpy()
        f_ind[d] = ir

        # ② 行业去均值
        X = df[use].to_numpy()
        y = df["r"].to_numpy() - ind_ret
        Xd = X - np.average(X, axis=0, weights=w.to_numpy())
        yd = y - np.average(y, weights=w.to_numpy())

        # ③ WLS
        sw = np.sqrt(w.to_numpy())
        A = Xd * sw[:, None]
        bvec = yd * sw
        b, *_ = np.linalg.lstsq(A, bvec, rcond=None)
        f_style.loc[d] = b

        # ④ 特质
        u = yd - Xd @ b
        for c, v in zip(df.index, u):
            spec.setdefault(c, {})[d] = v

    f_ind = pd.DataFrame(f_ind).T
    if verbose:
        print(f"   因子收益 {f_style.shape}；行业 {f_ind.shape}")
    return f_style, f_ind


def ewma_cov(df, halflife=HALFLIFE):
    x = df.dropna()
    lam = 0.5 ** (1 / halflife)
    w = lam ** np.arange(len(x))[::-1]
    w = w / w.sum()
    X = x.to_numpy()
    mu = w @ X
    Z = X - mu
    return pd.DataFrame((Z * w[:, None]).T @ Z, index=x.columns, columns=x.columns)


def specific_risk(spec: dict, dates) -> pd.Series:
    s = pd.Series(spec).sort_index()
    var = s.pow(2).ewm(halflife=HALFLIFE, min_periods=60).mean()
    return np.sqrt(var.clip(lower=SPEC_FLOOR ** 2))


# ==================== 组合风险 ====================
def portfolio_loading(w: pd.Series, wide, use, ind_grp):
    """返回 (风格暴露 Series, 行业权重 Series)"""
    if w.sum() <= 0:
        return pd.Series(0.0, index=use), pd.Series(dtype=float)
    ww = w / w.sum()
    cols = {}
    for k in use:
        cols[k] = wide[k].groupby(level=1).last() if False else None
    # 用最近一天的截面做暴露（组合权重按当日）
    return ww


def te_decompose(w_p: pd.Series, w_b: pd.Series, wide_d: pd.DataFrame,
                 use, F: pd.DataFrame, D: pd.Series):
    """wide_d: 某一天的暴露截面（index=ts_code）"""
    common = wide_d.index.union(w_p.index).union(w_b.index)
    wp = w_p.reindex(common).fillna(0.0)
    wb = w_b.reindex(common).fillna(0.0)
    wp, wb = wp / max(wp.sum(), 1e-12), wb / max(wb.sum(), 1e-12)

    e_p = wide_d[use].reindex(common).fillna(0).mul(wp, axis=0).sum()
    e_b = wide_d[use].reindex(common).fillna(0).mul(wb, axis=0).sum()
    de = (e_p - e_b).reindex(F.index).fillna(0.0)

    var_f = float(de.to_numpy() @ F.to_numpy() @ de.to_numpy())
    w_a = (wp - wb).abs()
    var_s = float((w_a.pow(2) * D.reindex(common).fillna(D.median())).sum())

    contrib = {}
    for s in use:
        v = float(de[s] * (F.loc[s, :] @ de))
        contrib[s] = v

    return {
        "te_ann": np.sqrt(max(var_f + var_s, 0)) * np.sqrt(252),
        "factor_risk_ann": np.sqrt(max(var_f, 0)) * np.sqrt(252),
        "specific_risk_ann": np.sqrt(max(var_s, 0)) * np.sqrt(252),
        "factor_share": var_f / (var_f + var_s) if (var_f + var_s) > 0 else np.nan,
        "style_contrib": contrib,
        "exposure_port": e_p, "exposure_bench": e_b,
    }


# ==================== 因子级风格诊断（核心） ====================
def factor_style_diagnose(fac: pd.DataFrame, wide, use, min_n=100):
    """
    逐日把因子分数对 [1, X~] 回归，得到：
      - R²（有多少是风格）
      - 载荷 β（像哪个风格）
      - 残差因子（真正的"新信息"）
    """
    from .core import smooth_score
    out_r2, loads = [], []
    resid_wide = fac * np.nan
    for d in fac.index:
        try:
            g = wide.xs(d, level=0)[use]
        except KeyError:
            continue
        y = fac.loc[d].reindex(g.index)
        df = pd.concat([y.rename("y"), g], axis=1).dropna()
        if len(df) < min_n:
            continue
        X = np.column_stack([np.ones(len(df)), df[use].to_numpy()])
        yv = df["y"].to_numpy()
        b, *_ = np.linalg.lstsq(X, yv, rcond=None)
        pred = X @ b
        ss_res = float(((yv - pred) ** 2).sum())
        ss_tot = float(((yv - yv.mean()) ** 2).sum())
        out_r2.append(1 - ss_res / ss_tot if ss_tot > 0 else np.nan)
        loads.append(pd.Series(b[1:], index=use))
        resid_wide.loc[d, df.index] = yv - pred
    r2 = pd.Series(out_r2)
    beta = pd.DataFrame(loads)
    return {
        "r2_mean": float(r2.mean()),
        "r2_by_year": r2.groupby(pd.Series(fac.index).str[:4].values).mean().round(3).to_dict(),
        "beta_mean": beta.mean().round(3).to_dict(),
        "beta_abs_rank": beta.abs().mean().sort_values(ascending=False).round(3).to_dict(),
        "resid": resid_wide,
    }
