# factorlab/core.py
"""
唯一的策略实现。data / smooth / select / replay 各只有一份。
研究脚本和未来的因子动物园全部 import 这里，不再各写一遍。

参数化约定
    REBAL 是默认值（5）。replay(..., rebal=N) 可覆盖，用于持有期实验（37 号）。
    smooth_score(s, rebal=N) 同样可覆盖 —— 平滑窗口应与调仓周期同步，
    否则等于用 5 日周期的平滑去喂一个 20 日周期的组合。

成本口径（重要）
    cost 可以是标量，也可以是 DataFrame（日期 × 代码，逐股成本）。
    net = gross − Σ|Δw| · cost
    Σ|Δw| 同时包含买入与卖出，所以 cost 已经是"每单位成交额的成本"，
    cost_amt 里不要再乘 COST_MULT。标量路径与旧版逐位等价。
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_DIR

RAW = DATA_DIR / "raw"
REF = DATA_DIR / "ref"
PANEL = DATA_DIR / "panel"

FACTORS = ["rev_5", "turn_20", "vol_20", "ep"]
TOP_N, REBAL, THRESH = 50, 5, 0.25
COST, COST_MULT = 0.0012, 2.0
HIST_END = "20251231"


# ==================== 数据 ====================
@dataclass
class Panel:
    ret: pd.DataFrame
    fwd: pd.DataFrame
    fac: dict
    idx: pd.Index
    cols: pd.Index
    mask: pd.DataFrame
    industry: pd.Series
    index_ret: pd.Series | None = None


def load_panel(years=range(2018, 2026), hist_end=HIST_END, with_index=False):
    frames = []
    for y in years:
        f = RAW / f"daily_{y}.parquet"
        if not f.exists():
            continue
        frames.append(pd.read_parquet(f, columns=["ts_code", "trade_date", "pct_chg"]))
    if not frames:
        raise FileNotFoundError("没找到 daily_YYYY.parquet")

    daily = pd.concat(frames, ignore_index=True).drop_duplicates(["trade_date", "ts_code"])
    ret = (daily.pivot(index="trade_date", columns="ts_code", values="pct_chg") / 100).sort_index()
    del daily
    fwd = ret.shift(-1)

    fac_raw = pd.read_parquet(PANEL / "factors_neutral.parquet")
    fac = {n: fac_raw.pivot(index="trade_date", columns="ts_code", values=n) for n in FACTORS}

    idx = ret.index.intersection(fac["rev_5"].index).sort_values()
    if hist_end:
        idx = idx[idx <= hist_end]
    cols = ret.columns

    for n in FACTORS:
        fac[n] = fac[n].reindex(index=idx, columns=cols)
    fwd = fwd.reindex(index=idx, columns=cols)

    tr = pd.read_parquet(PANEL / "tradable.parquet")
    if isinstance(tr.index, pd.DatetimeIndex):
        tr.index = tr.index.strftime("%Y%m%d")
    mask = tr.reindex(index=idx, columns=cols).fillna(0).astype(bool)

    sb = pd.concat([pd.read_parquet(REF / "stock_basic.parquet"),
                    pd.read_parquet(REF / "stock_delisted.parquet")],
                   ignore_index=True).drop_duplicates("ts_code")
    industry = sb.set_index("ts_code")["industry"].reindex(cols).fillna("UNKNOWN")

    p = Panel(ret=ret, fwd=fwd, fac=fac, idx=idx, cols=cols, mask=mask, industry=industry)

    if with_index:
        f = RAW / "index_daily_000852_SH.parquet"
        if not f.exists():
            raise FileNotFoundError("缺 index_daily_000852_SH.parquet，先跑 33")
        d = pd.read_parquet(f)
        s = d.set_index("trade_date")["pct_chg"].astype(float) / 100.0
        s.index = s.index.astype(str)
        p.index_ret = s.sort_index()

    return p


# ==================== 分数 ====================
def composite(panel, names=FACTORS):
    """等权复合；全缺失返回 NaN（不填 0）"""
    parts = [panel.fac[n] for n in names]
    out = sum(d.fillna(0.0) for d in parts)
    valid = sum(d.notna().astype(int) for d in parts) > 0
    return (out / len(parts)).where(valid)


def smooth_score(s, rebal=REBAL):
    """3 期等权：mean(s_t, s_{t-rebal}, s_{t-2*rebal})"""
    acc = s.fillna(0.0).copy()
    cnt = s.notna().astype(int).copy()
    for k in (rebal, 2 * rebal):
        sh = s.shift(k)
        acc = acc + sh.fillna(0.0)
        cnt = cnt + sh.notna().astype(int)
    return acc / cnt.replace(0, np.nan)


# ==================== 选股 ====================
def quota_select(s, industry, held, top_n=TOP_N, thresh=THRESH):
    """行业配额守恒 + 分数差门槛。全项目唯一实现。"""
    s = s.dropna()
    if len(s) < top_n:
        return []

    df = pd.DataFrame({"score": s,
                       "ind": industry.reindex(s.index).fillna("UNKNOWN")}).dropna()
    if df.empty or len(df) < top_n:
        return []

    share = df.groupby("ind").size() / len(df)
    raw = share * top_n
    base = np.floor(raw).astype(int)
    remain = top_n - int(base.sum())
    if remain > 0:
        for key in list((raw - base).sort_values(ascending=False).index)[:remain]:
            base.loc[key] += 1

    delta = thresh * float(df["score"].std())
    picks = []
    for ind_, k in base.items():
        if k <= 0:
            continue
        pool = df[df["ind"] == ind_].sort_values("score", ascending=False)
        if pool.empty:
            continue
        cut = float(pool["score"].iloc[min(int(k), len(pool)) - 1])
        kept = [c for c in held if c in pool.index
                and float(pool.loc[c, "score"]) >= cut - delta]
        kept = sorted(kept, key=lambda c: -float(pool.loc[c, "score"]))[:int(k)]
        fill = [c for c in pool.index if c not in set(kept)][:int(k) - len(kept)]
        picks.extend(kept + fill)

    picks = list(dict.fromkeys(picks))[:top_n]
    if len(picks) < top_n:
        rest = df[~df.index.isin(set(picks))].nlargest(top_n - len(picks), "score")
        picks += rest.index.tolist()
    return list(dict.fromkeys(picks))[:top_n]


# ==================== 回放 ====================
@dataclass
class Replay:
    net: pd.Series
    gross: pd.Series
    turnover: pd.Series
    holdings: dict = field(default_factory=dict)
    ind_dev: dict = field(default_factory=dict)
    rebal: int = REBAL
    cost_amt: pd.Series | None = None      # ★ 逐日成本（收益率单位，已含双边）

    def ann(self, cost_mult=COST_MULT):
        r = self.net
        return (1 + r).cumprod().iloc[-1] ** (252 / len(r)) - 1

    def ann_turnover(self):
        return float(self.turnover.replace(0, np.nan).mean() * (252 / self.rebal))

    def cost_pp(self):
        """年化成本（百分点）"""
        if self.cost_amt is None:
            return np.nan
        return float(self.cost_amt.mean() * 252 * 100)


def replay(score, panel, top_n=TOP_N, thresh=THRESH, cost=COST, rebal=REBAL):
    """rebal: 调仓周期（交易日）。默认 REBAL=5。

    cost: 标量（每单位成交额的成本）或 DataFrame（日期 × 代码，逐股成本）。
          标量路径与旧版完全等价：net = gross - Σ|Δw| * cost
    """
    if rebal <= 0:
        raise ValueError(f"rebal 必须为正整数，收到 {rebal}")

    w_live = pd.Series(0.0, index=panel.cols)
    held = []
    gross = pd.Series(0.0, index=panel.idx)
    to_s = pd.Series(0.0, index=panel.idx)
    cost_amt = pd.Series(0.0, index=panel.idx)          # ★
    holdings, dev = {}, {}

    for i, d in enumerate(panel.idx):
        if i % rebal == 0:
            s = score.loc[d].where(panel.mask.loc[d]).dropna()
            if len(s) >= top_n:
                picks = quota_select(s, panel.industry, held, top_n, thresh)
                if picks:
                    w_new = pd.Series(1.0 / len(picks), index=picks).reindex(panel.cols).fillna(0.0)
                    tot = w_live.sum()
                    w_drift = (w_live / tot) if tot > 0 else w_live

                    dw = (w_new - w_drift).abs()                    # ★
                    to_s.loc[d] = float(dw.sum() / 2)

                    if isinstance(cost, pd.DataFrame):              # ★ 逐股成本
                        row = cost.loc[d] if d in cost.index else pd.Series(dtype=float)
                        cvec = row.reindex(dw.index).fillna(COST)
                    else:                                           # 标量（旧口径）
                        cvec = cost
                    cost_amt.loc[d] = float((dw * cvec).sum())

                    pw = (pd.Series(1.0 / len(picks), index=picks)
                            .groupby(panel.industry.reindex(picks)).sum())
                    valid = panel.mask.loc[d]
                    bw = panel.industry.reindex(valid[valid].index).value_counts(normalize=True)
                    dev[d] = float((pw.reindex(bw.index).fillna(0) - bw).abs().max())

                    holdings[d] = picks
                    held, w_live = picks, w_new

        r = panel.fwd.loc[d].fillna(0.0)
        gross.loc[d] = float((w_live * r).sum())
        w_live = w_live * (1 + r)

    net = (gross - cost_amt).iloc[:-1]                          # ★
    return Replay(net=net, gross=gross, turnover=to_s,
                  holdings=holdings, ind_dev=dev, rebal=rebal,
                  cost_amt=cost_amt)                            # ★


def benchmark(panel):
    return panel.fwd.where(panel.mask).mean(axis=1).reindex(panel.idx).fillna(0.0)
