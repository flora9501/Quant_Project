#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
34_market_neutral.py  (v4)
市场中性版本：IM/IC 对冲 + 基差敏感性。

v4 修改
  ★ composite 改为 NaN-aware
  ★ 指数与期货收益均 shift(-1)，与组合 T+1 约定对齐

用法
  .venv/bin/python 34_market_neutral.py --probe
  .venv/bin/python 34_market_neutral.py
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tushare as ts
from dotenv import load_dotenv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DATA_DIR

load_dotenv()

RAW = DATA_DIR / "raw"
REF = DATA_DIR / "ref"
PANEL = DATA_DIR / "panel"
OUT = DATA_DIR / "output_neutral"
OUT.mkdir(parents=True, exist_ok=True)

FACTORS = ["rev_5", "turn_20", "vol_20", "ep"]
TOP_N, REBAL, COST, THRESH = 50, 5, 0.0012, 0.25
COST_MULT = 2.0
YEARS = 252
HIST_END = "20251231"
BASIS_GRID = [0.0, 0.05, 0.08, 0.12]
BETA_WIN = 60
READ_YEARS = [str(y) for y in range(2018, 2026)]

PERIODS = [
    ("全样本", "20180101", "20251231"),
    ("IC 时代", "20180101", "20220630"),
    ("IM 时代", "20220701", "20251231"),
]


def log(m):
    print(m, flush=True)


def probe_futures():
    pro = ts.pro_api(os.getenv("TUSHARE_TOKEN"))
    log("=" * 60)
    log("期货数据探针")
    log("=" * 60)
    for code in ["IM.CFX", "IC.CFX", "IF.CFX"]:
        try:
            df = pro.fut_daily(ts_code=code, start_date="20180101", end_date="20251231",
                               fields="trade_date,close,settle")
            if df is None or df.empty:
                log(f"  {code:<10} ❌ 返回空")
            else:
                log(f"  {code:<10} ✅ {len(df):>5} 行   "
                    f"{df.trade_date.min()} → {df.trade_date.max()}")
        except Exception as e:
            log(f"  {code:<10} ❌ {repr(e)[:80]}")
    log("=" * 60)


def load_fut_close(code):
    f = RAW / f"fut_daily_{code.replace('.', '_')}.parquet"
    if f.exists():
        df = pd.read_parquet(f)
    else:
        pro = ts.pro_api(os.getenv("TUSHARE_TOKEN"))
        try:
            df = pro.fut_daily(ts_code=code, start_date="20180101", end_date="20251231",
                               fields="trade_date,close,settle")
        except Exception:
            return None
        if df is None or df.empty:
            return None
        df.to_parquet(f, index=False)

    s = df.set_index("trade_date")["close"].astype(float)
    s.index = s.index.astype(str)
    return s.sort_index()


def load_fut_returns():
    """IC（旧）→ IM（新）拼接；拼接日剔除；最后 shift(-1)"""
    im = load_fut_close("IM.CFX")
    ic = load_fut_close("IC.CFX")

    if im is not None and ic is not None:
        cut = im.index.min()
        s = pd.concat([ic[ic.index < cut], im]).sort_index()
        stitch = cut
    elif im is not None:
        s, stitch = im, None
    elif ic is not None:
        s, stitch = ic, None
    else:
        return None

    r = s.pct_change()
    if stitch is not None and stitch in r.index:
        r.loc[stitch] = np.nan
    return r.dropna().shift(-1).dropna()


def load_all():
    log("① 读取数据...")
    frames = []
    for y in READ_YEARS:
        f = RAW / f"daily_{y}.parquet"
        if f.exists():
            frames.append(pd.read_parquet(f, columns=["ts_code", "trade_date", "pct_chg"]))
    daily = pd.concat(frames, ignore_index=True).drop_duplicates(["trade_date", "ts_code"])
    ret = (daily.pivot(index="trade_date", columns="ts_code", values="pct_chg") / 100).sort_index()
    del daily
    fwd = ret.shift(-1)

    fac = pd.read_parquet(PANEL / "factors_neutral.parquet")
    wide = {n: fac.pivot(index="trade_date", columns="ts_code", values=n) for n in FACTORS}

    idx = ret.index.intersection(wide["rev_5"].index).sort_values()
    idx = idx[idx <= HIST_END]
    cols = ret.columns
    for n in FACTORS:
        wide[n] = wide[n].reindex(index=idx, columns=cols)
    fwd = fwd.reindex(index=idx, columns=cols)

    tr = pd.read_parquet(PANEL / "tradable.parquet")
    if isinstance(tr.index, pd.DatetimeIndex):
        tr.index = tr.index.strftime("%Y%m%d")
    mask = tr.reindex(index=idx, columns=cols).fillna(0).astype(bool)

    sb = pd.concat([pd.read_parquet(REF / "stock_basic.parquet"),
                    pd.read_parquet(REF / "stock_delisted.parquet")],
                   ignore_index=True).drop_duplicates("ts_code")
    industry = sb.set_index("ts_code")["industry"].reindex(cols).fillna("UNKNOWN")

    log(f"   {len(idx)} 天 × {len(cols)} 只（终点 {HIST_END}）")
    return fwd, wide, idx, cols, mask, industry


def load_index():
    f = RAW / "index_daily_000852_SH.parquet"
    df = pd.read_parquet(f) if f.exists() else pd.DataFrame()

    if df.empty or df.trade_date.astype(str).nunique() < 1800:
        pro = ts.pro_api(os.getenv("TUSHARE_TOKEN"))
        parts = []
        for y in range(2018, 2026):
            d = pro.index_daily(ts_code="000852.SH",
                                start_date=f"{y}0101", end_date=f"{y}1231",
                                fields="trade_date,close,pct_chg")
            if d is not None and not d.empty:
                parts.append(d)
        if not parts:
            raise SystemExit("指数数据不可用")
        df = pd.concat(parts, ignore_index=True).drop_duplicates("trade_date")
        df.to_parquet(f, index=False)

    s = df.set_index("trade_date")["pct_chg"].astype(float) / 100.0
    s.index = s.index.astype(str)
    return s.sort_index()


def replay(fwd, wide, idx, cols, mask, industry):
    _tot = sum(wide[n].fillna(0.0) for n in FACTORS)
    _val = sum(wide[n].notna().astype(int) for n in FACTORS) > 0
    score_raw = (_tot / len(FACTORS)).where(_val)

    acc = score_raw.fillna(0.0).copy()
    cnt = score_raw.notna().astype(int).copy()
    for k in (REBAL, 2 * REBAL):
        sh = score_raw.shift(k)
        acc = acc + sh.fillna(0.0)
        cnt = cnt + sh.notna().astype(int)
    score = acc / cnt.replace(0, np.nan)

    def quota_select(s, held):
        df = pd.DataFrame({"score": s,
                           "ind": industry.reindex(s.index).fillna("UNKNOWN")}).dropna()
        if df.empty or len(df) < TOP_N:
            return []
        share = df.groupby("ind").size() / len(df)
        raw = share * TOP_N
        base = np.floor(raw).astype(int)
        remain = TOP_N - int(base.sum())
        if remain > 0:
            for key in list((raw - base).sort_values(ascending=False).index)[:remain]:
                base.loc[key] += 1

        delta = THRESH * float(df["score"].std())
        picks = []
        for ind_, k in base.items():
            if k <= 0:
                continue
            pool = df[df["ind"] == ind_].sort_values("score", ascending=False)
            if pool.empty:
                continue
            cut = float(pool["score"].iloc[min(int(k), len(pool)) - 1])
            kept = [c for c in held if c in pool.index and float(pool.loc[c, "score"]) >= cut - delta]
            kept = sorted(kept, key=lambda c: -float(pool.loc[c, "score"]))[:int(k)]
            fill = [c for c in pool.index if c not in set(kept)][:int(k) - len(kept)]
            picks.extend(kept + fill)
        picks = list(dict.fromkeys(picks))[:TOP_N]
        if len(picks) < TOP_N:
            rest = df[~df.index.isin(set(picks))].nlargest(TOP_N - len(picks), "score")
            picks += rest.index.tolist()
        return list(dict.fromkeys(picks))[:TOP_N]

    w_live = pd.Series(0.0, index=cols)
    held = []
    gross = pd.Series(0.0, index=idx)
    to_s = pd.Series(0.0, index=idx)

    for i, d in enumerate(idx):
        if i % REBAL == 0:
            s = score.loc[d].where(mask.loc[d]).dropna()
            if len(s) >= TOP_N:
                picks = quota_select(s, held)
                if picks:
                    w_new = pd.Series(1.0 / len(picks), index=picks).reindex(cols).fillna(0.0)
                    tot = w_live.sum()
                    w_drift = (w_live / tot) if tot > 0 else w_live
                    to_s.loc[d] = float((w_new - w_drift).abs().sum() / 2)
                    held, w_live = picks, w_new

        r = fwd.loc[d].fillna(0.0)
        gross.loc[d] = float((w_live * r).sum())
        w_live = w_live * (1 + r)

    return (gross - to_s * COST * COST_MULT).iloc[:-1]


def perf(r, label):
    n = len(r)
    nav = (1 + r).cumprod()
    ann = nav.iloc[-1] ** (YEARS / n) - 1
    vol = float(r.std() * np.sqrt(YEARS))
    dd = float((nav / nav.cummax() - 1).min())
    return {"策略": label, "年化": ann, "波动": vol,
            "夏普": ann / vol if vol > 0 else np.nan, "最大回撤": dd}


def window(s, start, end):
    return s[(s.index >= start) & (s.index <= end)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()

    if args.probe:
        probe_futures()
        return

    t0 = time.time()
    fwd, wide, idx, cols, mask, industry = load_all()
    idx_ret = load_index()

    log("② 回放多头组合...")
    net = replay(fwd, wide, idx, cols, mask, industry)

    # ★ 与组合的 T+1 约定对齐
    idx_ret = idx_ret.shift(-1).reindex(net.index).fillna(0.0)

    log("③ 估计滚动对冲比例...")
    cov = net.rolling(BETA_WIN).cov(idx_ret)
    var = idx_ret.rolling(BETA_WIN).var()
    beta = (cov / var).shift(1).clip(0.5, 1.2).fillna(1.0)

    corr = float(net.corr(idx_ret))
    log(f"   beta 均值 {beta.mean():.3f}，区间 [{beta.min():.2f}, {beta.max():.2f}]")
    log(f"   与指数日收益相关性 {corr:.3f}（正常应 > 0.85）")
    if beta.mean() < 0.7:
        log("   ⚠️ beta 异常低，检查日期对齐")

    hedge = beta * idx_ret

    log("④ 基差敏感性...")
    rows = []
    for pname, ps, pe in PERIODS:
        n_w = window(net, ps, pe)
        h_w = window(hedge, ps, pe)
        if len(n_w) < 60:
            continue
        rows.append({**perf(n_w, f"{pname}/多头"), "区间": pname, "口径": "多头 net"})
        rows.append({**perf(n_w - h_w, f"{pname}/对冲0"), "区间": pname, "口径": "对冲 0% 基差"})
        for b in BASIS_GRID:
            if b == 0:
                continue
            rows.append({**perf(n_w - h_w - b / YEARS, f"{pname}/{int(b*100)}%"),
                         "区间": pname, "口径": f"对冲 {int(b*100)}% 基差"})

    tab = pd.DataFrame(rows)

    log("⑤ 实际期货交叉验证...")
    fr = load_fut_returns()
    fut_line = None
    if fr is not None:
        common = net.index.intersection(fr.index)
        if len(common) > 100:
            realized = net.reindex(common) - beta.reindex(common) * fr.reindex(common)
            fut_line = {**perf(realized, "实际期货对冲"), "天数": len(common)}
            log(f"   可用 {len(common)} 天，年化 {fut_line['年化']:.2%}")
        else:
            log("   重叠天数不足")
    else:
        log("   期货数据不可用，跳过")

    log("\n" + "=" * 92)
    log("【市场中性：年化（%）】")
    log("=" * 92)
    log((tab.pivot(index="口径", columns="区间", values="年化") * 100).round(2).to_string())

    log("\n【波动（%）】")
    log((tab.pivot(index="口径", columns="区间", values="波动") * 100).round(2).to_string())

    log("\n【最大回撤（%）】")
    log((tab.pivot(index="口径", columns="区间", values="最大回撤") * 100).round(2).to_string())

    if fut_line:
        log("\n【实际期货对冲（IC→IM 拼接）】")
        log(pd.DataFrame([fut_line]).round(4).to_string(index=False))

    log("\n【盈亏平衡基差（年化）】")
    for pname, _, _ in PERIODS:
        sub = tab[(tab["区间"] == pname) & (tab["口径"] == "对冲 0% 基差")]
        if sub.empty:
            continue
        r0 = float(sub["年化"].iloc[0])
        log(f"   {pname:<8} 0 基差下 {r0:+7.2%}  →  可承受贴水上限 {r0:+7.2%}")

    nav_u = (1 + net).cumprod()
    nav_h = (1 + (net - hedge)).cumprod()
    nav_8 = (1 + (net - hedge - 0.08 / YEARS)).cumprod()

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(pd.to_datetime(nav_u.index), nav_u.values, label="Long-only (net)", linewidth=1.4)
    ax.plot(pd.to_datetime(nav_h.index), nav_h.values,
            label="Market neutral (0 basis)", linewidth=1.4)
    ax.plot(pd.to_datetime(nav_8.index), nav_8.values,
            label="Market neutral (8% basis)", linewidth=1.4)
    ax.set_yscale("log")
    ax.set_title("Long-only vs market-neutral")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(OUT / "nav.png", dpi=130)
    plt.close()

    tab.to_csv(OUT / "neutral_summary.csv", index=False, encoding="utf-8-sig")
    beta.to_frame("beta").to_csv(OUT / "beta.csv", encoding="utf-8-sig")

    log(f"\n✅ 输出到 {OUT}/")
    log(f"总耗时 {(time.time() - t0) / 60:.1f} 分钟")


if __name__ == "__main__":
    main()
