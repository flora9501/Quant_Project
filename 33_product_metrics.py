#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
33_product_metrics.py  (v5)
指数增强口径的产品指标。

v5 修改
  ★ composite / smooth_score / quota_select 改为 import factorlab.core（唯一实现）
      —— 消除本地副本；core 改 composite 口径时本脚本自动跟随
  ★ stats() 同时输出「差值口径」与「比值口径」两种超额，并打印恒等式
      比值 ≈ 差值 / (1 + 基准年化)          （4.25 ≈ 4.47 / 1.0518）
  ★ 新增 毛收益 / 毛超额_差值 / 信息比率_差值，用于「毛利 − 成本 ≠ 净」的闭包核对
  ★ 理论容量从硬编码改为常量 CAPACITY_YI（改 N 时必须同步）

输出
  data/output_product/product_sheet.csv
  data/output_product/constraints.csv
  data/output_product/monthly.csv
  data/output_product/excess_nav.png
"""
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import tushare as ts
from dotenv import load_dotenv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DATA_DIR
from factorlab.core import (composite as core_composite,
                            smooth_score as core_smooth,
                            quota_select as core_quota,
                            FACTORS, TOP_N, REBAL, COST, THRESH, COST_MULT)

load_dotenv()

RAW = DATA_DIR / "raw"
REF = DATA_DIR / "ref"
PANEL = DATA_DIR / "panel"
OUT = DATA_DIR / "output_product"
OUT.mkdir(parents=True, exist_ok=True)

INDEX_CODE = "000852.SH"
YEARS = 252
HIST_END = "20251231"
READ_YEARS = [str(y) for y in range(2018, 2026)]

# ★ 与 41 / 63 对齐；换 N 时必须改这里（目前只有 N=50 的容量有出处）
CAPACITY_YI = 2.4

PCT_ROWS = {"年化收益", "年化波动", "基准年化", "年化超额", "年化超额_差值",
            "毛收益", "毛超额_差值", "跟踪误差", "月度胜率",
            "超额最大回撤", "组合最大回撤"}
RAW_ROWS = {"夏普", "信息比率", "信息比率_差值", "Beta", "上行捕获", "下行捕获"}


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
        log("   指数缓存不足，按年重拉...")
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


# ==================== 3. 分数（全部走 core） + 回放 ====================
# ★ 唯一实现：core.composite 的 NaN 处理口径若改，这里自动跟随
score_raw = core_composite(SimpleNamespace(fac=wide))
score = core_smooth(score_raw)


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
            picks = core_quota(s, industry, held)          # ★ 不再用本地副本
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
gross_ret = gross_s.iloc[:-1]                                  # ★ 与 net 同索引
bench_ew = fwd.where(mask).mean(axis=1).reindex(net.index).fillna(0.0)

# ★ 指数必须 shift(-1)，与组合的 T+1 收益对齐
idx_ret_aligned = idx_ret.shift(-1).reindex(net.index).fillna(0.0) if idx_ret is not None else None

log(f"   回放完成，{len(net)} 天")


# ==================== 4. 指标 ====================
def stats(net, bench, label, gross=None, rf=0.0):
    n = len(net)
    active = net - bench
    nav_p = (1 + net).cumprod()
    nav_b = (1 + bench).cumprod()

    ann_p = nav_p.iloc[-1] ** (YEARS / n) - 1
    ann_b = nav_b.iloc[-1] ** (YEARS / n) - 1
    ex_nav = nav_p / nav_b
    ex_ann = ex_nav.iloc[-1] ** (YEARS / n) - 1             # 比值口径（4.25 / 8.05）

    te = float(active.std() * np.sqrt(YEARS))
    vol_p = float(net.std() * np.sqrt(YEARS))
    var_b = float(np.var(bench))

    gross_ann = np.nan
    if gross is not None:
        gross_ann = float((1 + gross).cumprod().iloc[-1] ** (YEARS / n) - 1)

    up, down = bench > 0, bench < 0
    mkey = net.index.str[:6]
    m_active = ((1 + net).groupby(mkey).prod() - 1) - ((1 + bench).groupby(mkey).prod() - 1)

    return {
        "基准": label,
        "年化收益": ann_p,
        "年化波动": vol_p,
        "夏普": (ann_p - rf) / vol_p if vol_p > 0 else np.nan,
        "基准年化": ann_b,
        "年化超额": ex_ann,                                 # 比值口径
        "年化超额_差值": ann_p - ann_b,                      # ★ 差值口径（4.47）
        "毛收益": gross_ann,                                # ★
        "毛超额_差值": (gross_ann - ann_b) if np.isfinite(gross_ann) else np.nan,
        "跟踪误差": te,
        "信息比率": ex_ann / te if te > 0 else np.nan,
        "信息比率_差值": (ann_p - ann_b) / te if te > 0 else np.nan,   # ★
        "月度胜率": float((m_active > 0).mean()),
        "月数": len(m_active),
        "超额最大回撤": float((ex_nav / ex_nav.cummax() - 1).min()),
        "组合最大回撤": float((nav_p / nav_p.cummax() - 1).min()),
        "Beta": float(np.cov(net, bench)[0, 1] / var_b) if var_b > 0 else np.nan,
        "上行捕获": float(net[up].mean() / bench[up].mean()) if up.sum() > 0 else np.nan,
        "下行捕获": float(net[down].mean() / bench[down].mean()) if down.sum() > 0 else np.nan,
    }


log("④ 计算产品指标...")
rows = [stats(net, bench_ew, "等权全池（研究口径）", gross=gross_ret)]
if idx_ret_aligned is not None:
    rows.append(stats(net, idx_ret_aligned, "中证1000指数（产品口径）", gross=gross_ret))
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
    "理论容量(亿)": CAPACITY_YI,                        # ★ 常量，不再硬编码在 dict 里
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

log("\n【口径恒等式核对】")
_r = tab.loc["等权全池（研究口径）"]
_p = float(_r["年化超额"]) * 100
_d = float(_r["年化超额_差值"]) * 100
_b = float(_r["基准年化"])
log(f"  比值口径 {_p:.2f}%  ≈  差值口径 {_d:.2f}% / (1+{_b*100:.2f}%) = {_d/(1+_b):.2f}%")
log("  两者差额不是误差：4.25 ≈ 4.47 / 1.0518")

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
