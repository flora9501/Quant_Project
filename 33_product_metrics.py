#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
33_product_metrics.py  (v4)
指数增强口径的产品指标。

v4 修改
  ★ composite 改为 NaN-aware（与 28 / core.py 一致）
  ★ 指数收益 shift(-1)，与组合的 T+1 约定对齐

输出
  data/output_product/product_sheet.csv
  data/output_product/constraints.csv
  data/output_product/monthly.csv
  data/output_product/excess_nav.png
"""
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
OUT = DATA_DIR / "output_product"
OUT.mkdir(parents=True, exist_ok=True)

FACTORS = ["rev_5", "turn_20", "vol_20", "ep"]
TOP_N, REBAL, COST, THRESH = 50, 5, 0.0012, 0.25
COST_MULT = 2.0
INDEX_CODE = "000852.SH"
YEARS = 252
HIST_END = "20251231"
READ_YEARS = [str(y) for y in range(2018, 2026)]

PCT_ROWS = {"年化收益", "年化波动", "基准年化", "年化超额",
            "跟踪误差", "月度胜率", "超额最大回撤", "组合最大回撤"}
RAW_ROWS = {"夏普", "信息比率", "Beta", "上行捕获", "下行捕获"}


def log(m):
    print(m, flush=True)


t0 = time.time()

# ==================== 1. 数据（只读 2018–2025） ====================
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


# ==================== 2. 指数 ====================
def load_index_ret(code=INDEX_CODE, min_days=1800):
    f = RAW / f"index_daily_{code.replace('.', '_')}.parquet"
    df = pd.read_parquet(f) if f.exists() else pd.DataFrame()

    if df.empty or "trade_date" not in df.columns \
            or df.trade_date.astype(str).nunique() < min_days:
        log(f"   指数缓存不足，按年重拉...")
        pro = ts.pro_api(os.getenv("TUSHARE_TOKEN"))
        parts = []
        for y in range(2018, 2026):
            d = pro.index_daily(ts_code=code, start_date=f"{y}0101", end_date=f"{y}1231",
                                fields="trade_date,close,pct_chg")
            if d is not None and not d.empty:
                parts.append(d)
        if not parts:
            raise SystemExit("指数数据不可用")
        df = pd.concat(parts, ignore_index=True).drop_duplicates("trade_date")
        df.to_parquet(f, index=False)
        log(f"   已保存 {f.name}，{len(df)} 行")
    else:
        log(f"   指数缓存命中（{df.trade_date.nunique()} 天）")

    s = df.set_index("trade_date")["pct_chg"].astype(float) / 100.0
    s.index = s.index.astype(str)
    return s.sort_index()


log("② 读取中证 1000 指数...")
idx_ret = load_index_ret()


# ==================== 3. 分数 + 回放 ====================
_tot = sum(wide[n].fillna(0.0) for n in FACTORS)
_val = sum(wide[n].notna().astype(int) for n in FACTORS) > 0
score_raw = (_tot / len(FACTORS)).where(_val)


def smooth_score(s):
    acc = s.fillna(0.0).copy()
    cnt = s.notna().astype(int).copy()
    for k in (REBAL, 2 * REBAL):
        sh = s.shift(k)
        acc = acc + sh.fillna(0.0)
        cnt = cnt + sh.notna().astype(int)
    return acc / cnt.replace(0, np.nan)


score = smooth_score(score_raw)


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


log("③ 回放...")
w_live = pd.Series(0.0, index=cols)
held = []
gross_s = pd.Series(0.0, index=idx)
to_s = pd.Series(0.0, index=idx)
ind_dev, n_picks = {}, {}

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

                pw = (pd.Series(1.0 / len(picks), index=picks)
                        .groupby(industry.reindex(picks)).sum())
                valid = mask.loc[d]
                bw = industry.reindex(valid[valid].index).value_counts(normalize=True)
                ind_dev[d] = float((pw.reindex(bw.index).fillna(0) - bw).abs().max())
                n_picks[d] = len(picks)

                held, w_live = picks, w_new

    r = fwd.loc[d].fillna(0.0)
    gross_s.loc[d] = float((w_live * r).sum())
    w_live = w_live * (1 + r)

net = (gross_s - to_s * COST * COST_MULT).iloc[:-1]
bench_ew = fwd.where(mask).mean(axis=1).reindex(net.index).fillna(0.0)

# ★ 指数必须 shift(-1)，与组合的 T+1 收益对齐
idx_ret_aligned = idx_ret.shift(-1).reindex(net.index).fillna(0.0) if idx_ret is not None else None

log(f"   回放完成，{len(net)} 天")


# ==================== 4. 指标 ====================
def stats(net, bench, label, rf=0.0):
    n = len(net)
    active = net - bench
    nav_p = (1 + net).cumprod()
    nav_b = (1 + bench).cumprod()

    ann_p = nav_p.iloc[-1] ** (YEARS / n) - 1
    ann_b = nav_b.iloc[-1] ** (YEARS / n) - 1
    ex_nav = nav_p / nav_b
    ex_ann = ex_nav.iloc[-1] ** (YEARS / n) - 1

    te = float(active.std() * np.sqrt(YEARS))
    vol_p = float(net.std() * np.sqrt(YEARS))
    var_b = float(np.var(bench))

    up, down = bench > 0, bench < 0
    mkey = net.index.str[:6]
    m_active = ((1 + net).groupby(mkey).prod() - 1) - ((1 + bench).groupby(mkey).prod() - 1)

    return {
        "基准": label,
        "年化收益": ann_p,
        "年化波动": vol_p,
        "夏普": (ann_p - rf) / vol_p if vol_p > 0 else np.nan,
        "基准年化": ann_b,
        "年化超额": ex_ann,
        "跟踪误差": te,
        "信息比率": ex_ann / te if te > 0 else np.nan,
        "月度胜率": float((m_active > 0).mean()),
        "月数": len(m_active),
        "超额最大回撤": float((ex_nav / ex_nav.cummax() - 1).min()),
        "组合最大回撤": float((nav_p / nav_p.cummax() - 1).min()),
        "Beta": float(np.cov(net, bench)[0, 1] / var_b) if var_b > 0 else np.nan,
        "上行捕获": float(net[up].mean() / bench[up].mean()) if up.sum() > 0 else np.nan,
        "下行捕获": float(net[down].mean() / bench[down].mean()) if down.sum() > 0 else np.nan,
    }


log("④ 计算产品指标...")
rows = [stats(net, bench_ew, "等权全池（研究口径）")]
if idx_ret_aligned is not None:
    rows.append(stats(net, idx_ret_aligned, "中证1000指数（产品口径）"))
tab = pd.DataFrame(rows).set_index("基准")

ann_to = float(to_s.replace(0, np.nan).mean() * (252 / REBAL))
dev_s = pd.Series(ind_dev)
extra = pd.DataFrame([{
    "年化换手": ann_to,
    "年化成本": ann_to * COST * COST_MULT,
    "行业偏离均值": float(dev_s.mean()),
    "行业偏离最大": float(dev_s.max()),
    "持仓只数": int(np.mean(list(n_picks.values()))),
    "最大个股权重": 1.0 / TOP_N,
    "理论容量(亿)": 2.4,
}], index=["组合约束"])


# ==================== 5. 打印 ====================
log("\n" + "=" * 96)
log("【产品指标】")
log("=" * 96)
show = tab.T.copy()
for r in show.index:
    if r == "月数":
        show.loc[r] = show.loc[r].round(0).astype(int)
    elif r in PCT_ROWS:
        show.loc[r] = (show.loc[r] * 100).round(2)
    elif r in RAW_ROWS:
        show.loc[r] = show.loc[r].round(2)
log(show.to_string())
log("\n（百分比类单位为 %；夏普/IR/Beta/捕获率为倍数）")

log("\n【交易与约束】")
ex_show = extra.T.copy()
for r in ["年化换手", "年化成本", "行业偏离均值", "行业偏离最大", "最大个股权重"]:
    ex_show.loc[r] = (ex_show.loc[r] * 100).round(2)
ex_show.loc["持仓只数"] = ex_show.loc["持仓只数"].astype(int)
ex_show.loc["理论容量(亿)"] = ex_show.loc["理论容量(亿)"].round(1)
log(ex_show.to_string())


# ==================== 6. 月度 ====================
month_key = net.index.str[:6]
mdf = pd.DataFrame({
    "组合": (1 + net).groupby(month_key).prod() - 1,
    "等权基准": (1 + bench_ew).groupby(month_key).prod() - 1,
})
if idx_ret_aligned is not None:
    mdf["中证1000"] = (1 + idx_ret_aligned).groupby(month_key).prod() - 1
mdf["超额(等权)"] = mdf["组合"] - mdf["等权基准"]
if "中证1000" in mdf.columns:
    mdf["超额(指数)"] = mdf["组合"] - mdf["中证1000"]

log("\n【近 12 个月（%）】")
log((mdf.tail(12) * 100).round(2).to_string())


# ==================== 7. 图 ====================
nav_p = (1 + net).cumprod()
nav_b_ew = (1 + bench_ew).cumprod()

fig, ax = plt.subplots(figsize=(11, 5.5))
ax.plot(pd.to_datetime(nav_p.index), (nav_p / nav_b_ew).values,
        label="Excess vs EW universe", linewidth=1.4)
if idx_ret_aligned is not None:
    nav_b_ix = (1 + idx_ret_aligned).cumprod()
    ax.plot(pd.to_datetime(nav_p.index), (nav_p / nav_b_ix).values,
            label="Excess vs CSI 1000 index", linewidth=1.4)
ax.axhline(1, color="gray", linestyle="--", linewidth=1)
ax.set_title("Cumulative excess NAV")
ax.legend()
ax.grid(alpha=0.3)
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig(OUT / "excess_nav.png", dpi=130)
plt.close()

tab.T.to_csv(OUT / "product_sheet.csv", encoding="utf-8-sig")
extra.T.to_csv(OUT / "constraints.csv", encoding="utf-8-sig")
mdf.to_csv(OUT / "monthly.csv", encoding="utf-8-sig")

log(f"\n✅ 输出到 {OUT}/")
log(f"总耗时 {(time.time() - t0) / 60:.1f} 分钟")
