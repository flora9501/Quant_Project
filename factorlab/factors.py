# factorlab/factors.py
"""
因子构造与中性化。全项目唯一实现。

对外
    load_price_data()            读日线 / 每日指标（可指定需要的列）
    build_price_factors()        6 个价量因子（原始，生产口径）
    fill_alive_zero()            生命期内 NaN→0，生命期外保持 NaN
    neutralize()                 MAD → 行业+市值回归 → z-score
    get_factor()                 统一入口（新旧因子都走这里）
    cached()                     中性化结果落盘，避免每次重算

注意
    build_price_factors 的 ret.fillna(0.0) 是**生产口径**（复现已发表数字），
    不要改成生命期感知 —— vol_20 / rev_5 的回归基线依赖它。
    动物园等新因子请用 fill_alive_zero()。

    load_price_data 的 daily_cols 是 keyword-only：必须写
    load_price_data(daily_cols=("pct_chg", "amount"))，
    避免位置参数静默绑到 years 上。
"""
from pathlib import Path

import numpy as np
import pandas as pd

from factorlab.core import RAW, PANEL

ZOO = PANEL / "factor_zoo"
ZOO.mkdir(parents=True, exist_ok=True)

PRICE_FACTORS = ["rev_5", "turn_20", "vol_20", "mom_60", "ep", "ln_mv"]


# ==================== 原始数据 ====================
def _parquet_columns(f):
    """只读 schema，不读数据（daily_YYYY 是 800 万行，别为了看列名整份读进来）"""
    try:
        import pyarrow.parquet as pq
        return set(pq.ParquetFile(str(f)).schema.names)
    except Exception:
        return set(pd.read_parquet(f).columns)


def load_price_data(years=range(2018, 2026), *, daily_cols=("pct_chg",)):
    """
    daily_cols: 要从 daily_ 里取的列（keyword-only）。默认只取 pct_chg。
    任何一年缺列会立刻报出文件名，而不是静默返回 NaN。
    """
    need = ["ts_code", "trade_date", *daily_cols]
    frames = []
    for y in years:
        f = RAW / f"daily_{y}.parquet"
        if not f.exists():
            continue
        have = _parquet_columns(f)
        missing = [c for c in need if c not in have]
        if missing:
            raise KeyError(f"{f.name} 缺列 {missing}；该文件现有 {sorted(have)}")
        frames.append(pd.read_parquet(f, columns=need))
    if not frames:
        raise FileNotFoundError(
            f"没找到 daily_YYYY.parquet（RAW={RAW}，years={list(years)}）")
    daily = pd.concat(frames, ignore_index=True).drop_duplicates(["trade_date", "ts_code"])

    bneed = ["ts_code", "trade_date", "turnover_rate", "circ_mv", "pe_ttm"]
    bframes = []
    for y in years:
        f = RAW / f"daily_basic_{y}.parquet"
        if not f.exists():
            continue
        have = _parquet_columns(f)
        missing = [c for c in bneed if c not in have]
        if missing:
            raise KeyError(f"{f.name} 缺列 {missing}；该文件现有 {sorted(have)}")
        bframes.append(pd.read_parquet(f, columns=bneed))
    if not bframes:
        raise FileNotFoundError(f"没找到 daily_basic_YYYY.parquet（RAW={RAW}）")
    basic = pd.concat(bframes, ignore_index=True).drop_duplicates(["trade_date", "ts_code"])

    return daily, basic


def _wide(df, col, dates, codes, fillna=None):
    m = (df.pivot(index="trade_date", columns="ts_code", values=col)
           .reindex(index=dates, columns=codes))
    return m.fillna(fillna) if fillna is not None else m


def fill_alive_zero(ret):
    """
    只在"生命期"（首次有效 ~ 末次有效）内把 NaN 填 0（停牌 = 0 收益），
    上市前 / 退市后保持 NaN —— 缺失观测不是 0（PROJECT_STATUS 坑 #29）。

    ret: dates × codes 的日收益（小数）
    """
    if ret.empty:
        return ret.copy()
    valid = ret.notna()
    fwd_seen = valid.astype(np.int8).cummax().astype(bool)
    bwd_seen = valid[::-1].astype(np.int8).cummax()[::-1].astype(bool)
    alive = fwd_seen & bwd_seen
    return ret.fillna(0.0).where(alive)


def build_price_factors(dates, codes, daily, basic):
    """与 04_build_factors.py 完全一致的原始因子（未中性化）。

    注意：ret.fillna(0.0) 是全样本填 0（含上市前），这是生产口径，
    回归测试锁定了它的输出。新因子请另写，不要改这里。
    """
    ret = _wide(daily, "pct_chg", dates, codes) / 100.0
    ret = ret.fillna(0.0)                    # 生产口径：停牌/未上市均按 0
    turn = _wide(basic, "turnover_rate", dates, codes)
    cmv = _wide(basic, "circ_mv", dates, codes)
    pe = _wide(basic, "pe_ttm", dates, codes)

    return {
        "rev_5": -ret.rolling(5, min_periods=5).sum(),
        "turn_20": -turn.rolling(20, min_periods=20).mean(),
        "vol_20": -ret.rolling(20, min_periods=20).std() * np.sqrt(252),
        "mom_60": ret.shift(5).rolling(55, min_periods=55).sum(),
        "ep": 1.0 / pe.where(pe > 0),
        "ln_mv": -np.log(cmv * 1e4),
    }


# ==================== 中性化 ====================
def mad_winsorize(s, n=3):
    med = s.median()
    mad = (s - med).abs().median() * 1.4826
    if not np.isfinite(mad) or mad == 0:
        return s
    return s.clip(med - n * mad, med + n * mad)


def neutralize_one(y, lnmv, ind, min_n=100):
    """
    单日单因子：winsorize → 对 [行业哑变量 + ln市值] 回归 → 残差 z-score

    用 FWL 等价变换代替 140 列 lstsq：
      先把 y 和 lnmv 在行业内去均值，再对去均值后的 lnmv 做一元回归。
    残差与"带行业哑变量的多元回归"完全一致，但快约 10 倍。
    """
    ok = y.notna() & lnmv.notna()
    if int(ok.sum()) < min_n:
        return pd.Series(np.nan, index=y.index)

    y = mad_winsorize(y)

    d = pd.DataFrame({"y": y[ok], "lm": lnmv[ok], "ind": ind[ok]})
    d["y"] = d["y"] - d.groupby("ind")["y"].transform("mean")
    d["lm"] = d["lm"] - d.groupby("ind")["lm"].transform("mean")

    denom = float((d["lm"] ** 2).sum())
    z = pd.Series(np.nan, index=y.index)
    if denom <= 0:
        return z

    beta = float((d["y"] * d["lm"]).sum()) / denom
    res = (d["y"] - beta * d["lm"]).to_numpy()
    sd = res.std()
    z[d.index] = (res - res.mean()) / (sd if sd > 0 else 1.0)
    return z


def neutralize(raw, mask, industry, names=None, verbose=True):
    """
    raw: dict[str, DataFrame(dates × codes)]，必须含 ln_mv
    mask / industry: 与 core.load_panel 返回的一致
    返回 dict[str, DataFrame]，已中性化并 z-score
    """
    if "ln_mv" not in raw:
        raise KeyError("neutralize 需要 raw['ln_mv'] 作为规模变量")

    lnmv = raw["ln_mv"]
    todo = [n for n in (names or raw.keys()) if n != "ln_mv"]

    out = {}
    for name in todo:
        mat = raw[name]
        rows = {}
        for i, d in enumerate(mat.index):
            if d not in mask.index:
                continue
            y = mat.loc[d].where(mask.loc[d])
            rows[d] = neutralize_one(y, lnmv.loc[d], industry)
            if verbose and i % 250 == 0:
                print(f"   {name}: {i}/{len(mat.index)}", flush=True)
        out[name] = pd.DataFrame(rows).T.sort_index()
        if verbose:
            print(f"   {name} 完成，{len(out[name])} 天", flush=True)
    return out


# ==================== 统一入口 ====================
def cache_path(name):
    return ZOO / f"{name}.parquet"


def load_cached(name):
    """读回宽表。缓存文件是长表（trade_date, ts_code, value），必须 pivot。"""
    p = cache_path(name)
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if {"trade_date", "ts_code", "value"} <= set(df.columns):
        return (df.pivot(index="trade_date", columns="ts_code", values="value")
                  .sort_index())
    return df


def save_cached(name, wide):
    long = wide.stack().rename("value").reset_index()
    long.columns = ["trade_date", "ts_code", "value"]
    long.to_parquet(cache_path(name), index=False)


def get_factor(name, panel, rebuild=False):
    """
    返回一个 (dates × codes) 的截面 z-score 因子。
    已有四个因子直接从 factors_neutral 读；价量类按需中性化并缓存。
    """
    if not rebuild:
        c = load_cached(name)
        if c is not None:
            return c.reindex(index=panel.idx, columns=panel.cols)

    if name in ("rev_5", "turn_20", "vol_20", "ep"):
        fac = pd.read_parquet(PANEL / "factors_neutral.parquet")
        w = fac.pivot(index="trade_date", columns="ts_code", values=name)
        return w.reindex(index=panel.idx, columns=panel.cols)

    if name in ("acc_ttm", "sue"):
        f = PANEL / ("factor_accruals.parquet" if name == "acc_ttm" else "factor_pead.parquet")
        d = pd.read_parquet(f)
        w = d.pivot(index="trade_date", columns="ts_code", values="factor_z")
        return w.reindex(index=panel.idx, columns=panel.cols)

    if name in PRICE_FACTORS:
        daily, basic = load_price_data()
        raw = build_price_factors(panel.idx, panel.cols, daily, basic)
        neu = neutralize(raw, panel.mask, panel.industry, names=[name])
        wide = neu[name]
        save_cached(name, wide)
        return wide.reindex(index=panel.idx, columns=panel.cols)

    raise KeyError(f"未知因子: {name}")
