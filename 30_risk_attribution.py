#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
30_risk_attribution.py  (v2)
跟踪误差及其分解（5 日 Brinson，385 个周期）。

v2 修改
  ★ 改为使用 3 期平滑分数（与 28 的 "Strat + A+B"、31、daily_run 一致）
    此前用原始分数，持仓与最终策略不同，TE 数字需重算

输出
  data/output_risk/te_summary.csv
  data/output_risk/return_attribution.csv
  data/output_risk/brinson_industry.csv
  data/output_risk/period_detail.csv
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_DIR

RAW = DATA_DIR / "raw"
REF = DATA_DIR / "ref"
PANEL = DATA_DIR / "panel"
OUT = DATA_DIR / "output_risk"
OUT.mkdir(parents=True, exist_ok=True)

FACTORS = ["rev_5", "turn_20", "vol_20", "ep"]
TOP_N, REBAL = 50, 5
THRESH = 0.25
PERIODS_PER_YEAR = 252 / REBAL


def log(m):
    print(m, flush=True)


t0 = time.time()

# ==================== 1. 数据 ====================
log("① 读取数据...")
frames = [pd.read_parquet(f, columns=["ts_code", "trade_date", "pct_chg"])
          for f in sorted(RAW.glob("daily_[0-9][0-9][0-9][0-9].parquet"))]
daily = pd.concat(frames, ignore_index=True).drop_duplicates(["trade_date", "ts_code"])
ret = (daily.pivot(index="trade_date", columns="ts_code", values="pct_chg") / 100).sort_index()
fwd = ret.shift(-1)
del daily

px = (1 + ret).cumprod()
fwd5 = px.shift(-REBAL) / px - 1

fac = pd.read_parquet(PANEL / "factors_neutral.parquet")
wide = {n: fac.pivot(index="trade_date", columns="ts_code", values=n) for n in FACTORS}

idx = ret.index.intersection(wide["rev_5"].index).sort_values()
cols = ret.columns
for n in FACTORS:
    wide[n] = wide[n].reindex(index=idx, columns=cols)
fwd = fwd.reindex(index=idx, columns=cols)
fwd5 = fwd5.reindex(index=idx, columns=cols)

tr = pd.read_parquet(PANEL / "tradable.parquet")
if isinstance(tr.index, pd.DatetimeIndex):
    tr.index = tr.index.strftime("%Y%m%d")
mask = tr.reindex(index=idx, columns=cols).fillna(False).astype(bool)

sb = pd.concat([pd.read_parquet(REF / "stock_basic.parquet"),
                pd.read_parquet(REF / "stock_delisted.parquet")],
               ignore_index=True).drop_duplicates("ts_code")
industry = sb.set_index("ts_code")["industry"].reindex(cols).fillna("UNKNOWN")

log(f"   {len(idx)} 天 × {len(cols)} 只，行业 {industry.nunique()} 个")


# ==================== 2. 分数（含平滑） ====================
score_raw = sum(wide[n].fillna(0) for n in FACTORS) / len(FACTORS)


def smooth_score(s):
    acc = s.fillna(0.0).copy()
    cnt = s.notna().astype(int).copy()
    for k in (REBAL, 2 * REBAL):
        sh = s.shift(k)
        acc = acc + sh.fillna(0.0)
        cnt = cnt + sh.notna().astype(int)
    return acc / cnt.replace(0, np.nan)


score = smooth_score(score_raw)
log(f"   平滑分数完成；有效天数 {int(score.notna().any(axis=1).sum())}")


# ==================== 3. 选股 ====================
def industry_quota(df, top_n=TOP_N):
    share = df.groupby("ind").size() / len(df)
    raw = share * top_n
    base = np.floor(raw).astype(int)
    remain = top_n - int(base.sum())
    if remain > 0:
        for key in list((raw - base).sort_values(ascending=False).index)[:remain]:
            base.loc[key] += 1
    return base


def select_quota_hyst(s, held, sigma_mult=THRESH):
    df = pd.DataFrame({"score": s,
                       "ind": industry.reindex(s.index).fillna("UNKNOWN")}).dropna()
    if df.empty:
        return []
    if len(df) <= TOP_N:
        return list(df.nlargest(TOP_N, "score").index)

    base = industry_quota(df)
    delta = sigma_mult * float(df["score"].std())
    picks = []
    for ind_, k in base.items():
        if k <= 0:
            continue
        pool = df[df["ind"] == ind_].sort_values("score", ascending=False)
        if pool.empty:
            continue
        cut = float(pool["score"].iloc[min(int(k), len(pool)) - 1])
        held_in = [c for c in held if c in pool.index]
        keep = [c for c in held_in if float(pool.loc[c, "score"]) >= cut - delta]
        keep = sorted(keep, key=lambda c: -float(pool.loc[c, "score"]))[:int(k)]
        fill = [c for c in pool.index if c not in set(keep)][:int(k) - len(keep)]
        picks.extend(keep + fill)

    picks = list(dict.fromkeys(picks))[:TOP_N]
    if len(picks) < TOP_N:
        rest = df[~df.index.isin(set(picks))].nlargest(TOP_N - len(picks), "score")
        picks += rest.index.tolist()
    return list(dict.fromkeys(picks))[:TOP_N]


log("② 回放最终版本，记录每期持仓...")
held = []
holdings = {}
for i, d in enumerate(idx):
    if i % REBAL != 0:
        continue
    s = score.loc[d].where(mask.loc[d]).dropna()
    if len(s) < TOP_N:
        continue
    held = select_quota_hyst(s, held)
    if held:
        holdings[d] = list(held)

log(f"   {len(holdings)} 个调仓期")


# ==================== 4. Brinson ====================
log("③ 逐期做 Brinson 分解...")
rows = []

for d, picks in holdings.items():
    r5 = fwd5.loc[d]
    if not np.isfinite(r5.reindex(picks)).any():
        continue

    bench_codes = r5.where(mask.loc[d]).dropna().index
    if len(bench_codes) < 100:
        continue
    w_b = pd.Series(1.0 / len(bench_codes), index=bench_codes)
    r_b = float((w_b * r5.reindex(bench_codes).fillna(0)).sum())

    w_p = pd.Series(1.0 / len(picks), index=picks)

    ind_p = industry.reindex(w_p.index)
    ind_b = industry.reindex(w_b.index)

    wp_ind = w_p.groupby(ind_p).sum()
    wb_ind = w_b.groupby(ind_b).sum()

    tmp_b = pd.DataFrame({"ind": ind_b, "r": r5.reindex(w_b.index).fillna(0)})
    rb_ind = tmp_b.groupby("ind")["r"].mean()

    tmp_p = pd.DataFrame({"ind": ind_p,
                          "wr": w_p.values * r5.reindex(w_p.index).fillna(0).values})
    sum_wr = tmp_p.groupby("ind")["wr"].sum()
    rp_ind = (sum_wr / wp_ind).dropna()

    inds = wp_ind.index.union(wb_ind.index).union(rp_ind.index)
    wp_i = wp_ind.reindex(inds).fillna(0)
    wb_i = wb_ind.reindex(inds).fillna(0)
    rb_i = rb_ind.reindex(inds).fillna(0)
    rp_i = rp_ind.reindex(inds).fillna(rb_i)

    alloc = (wp_i - wb_i) * (rb_i - r_b)
    sel = wb_i * (rp_i - rb_i)
    inter = (wp_i - wb_i) * (rp_i - rb_i)

    rows.append({
        "date": d,
        "r_p": float((w_p * r5.reindex(w_p.index).fillna(0)).sum()),
        "r_b": r_b,
        "alloc": float(alloc.sum()),
        "sel": float(sel.sum()),
        "inter": float(inter.sum()),
        "alloc_by_ind": alloc.to_dict(),
        "sel_by_ind": sel.to_dict(),
    })

br = pd.DataFrame(rows).set_index("date")
br["active"] = br.r_p - br.r_b
br["resid"] = br.active - (br.alloc + br.sel + br.inter)

log(f"   {len(br)} 个有效周期")
log(f"   恒等式校验：|残差| 均值 {br.resid.abs().mean():.2e}（应接近 0）")


# ==================== 5. 跟踪误差 ====================
def te(x):
    return float(x.std() * np.sqrt(PERIODS_PER_YEAR))


te_total = te(br.active)
te_alloc, te_sel, te_inter = te(br.alloc), te(br.sel), te(br.inter)

var_total = br.active.var()
share_alloc = br.alloc.var() / var_total
share_sel = br.sel.var() / var_total
share_inter = br.inter.var() / var_total
cov_sum = 1 - (share_alloc + share_sel + share_inter)


# ==================== 6. 汇总 ====================
log("\n" + "=" * 78)
log("【跟踪误差】")
log("=" * 78)
log(f"   TE（年化）            {te_total:7.2%}")
log(f"   组合年化波动           {te(br.r_p):7.2%}")
log(f"   基准年化波动           {te(br.r_b):7.2%}")

log("\n【风险来源分解（方差占比）】")
risk_tab = pd.DataFrame({
    "来源": ["行业配置 A", "行业内选股 S", "交互项 I", "协方差项", "合计"],
    "TE 贡献": [te_alloc, te_sel, te_inter, np.nan, te_total],
    "方差占比": [share_alloc, share_sel, share_inter, cov_sum, 1.0],
})
log(risk_tab.to_string(index=False,
                       formatters={"TE 贡献": lambda x: f"{x:7.2%}" if pd.notna(x) else "   —  ",
                                   "方差占比": lambda x: f"{x:7.1%}"}))

log("\n【收益归因（年化）】")
ret_tab = pd.DataFrame({
    "来源": ["行业配置 A", "行业内选股 S", "交互项 I", "合计超额"],
    "年化贡献": [br.alloc.mean() * PERIODS_PER_YEAR,
                 br.sel.mean() * PERIODS_PER_YEAR,
                 br.inter.mean() * PERIODS_PER_YEAR,
                 br.active.mean() * PERIODS_PER_YEAR],
})
log(ret_tab.to_string(index=False,
                      formatters={"年化贡献": lambda x: f"{x:+7.2%}"}))

log("\n【稳定性】")
log(f"   5 日超额 > 0 的比例    {(br.active > 0).mean():7.1%}")
log(f"   5 日选股贡献 > 0 的比例 {(br.sel > 0).mean():7.1%}")


# ==================== 7. 行业明细（行=期，列=行业） ====================
log("\n④ 行业明细...")
alloc_df = pd.DataFrame(list(br["alloc_by_ind"]), index=br.index).fillna(0.0)
sel_df = pd.DataFrame(list(br["sel_by_ind"]), index=br.index).fillna(0.0)

assert set(alloc_df.columns) <= set(industry.unique()) | {"UNKNOWN"}, \
    f"列名异常: {list(alloc_df.columns)[:10]}"

ind_tab = pd.DataFrame({
    "配置年化": alloc_df.mean(axis=0) * PERIODS_PER_YEAR,
    "选股年化": sel_df.mean(axis=0) * PERIODS_PER_YEAR,
    "配置波动": alloc_df.std(axis=0) * np.sqrt(PERIODS_PER_YEAR),
    "选股波动": sel_df.std(axis=0) * np.sqrt(PERIODS_PER_YEAR),
}).sort_values("选股年化", ascending=False)

log(f"   行业数校验: {len(ind_tab)}（应约 136）")

log("\n【选股贡献最高的 10 个行业（年化 %）】")
log((ind_tab.head(10) * 100).round(2).to_string())

log("\n【选股贡献最低的 10 个行业（年化 %）】")
log((ind_tab.tail(10) * 100).round(2).to_string())

log("\n【配置贡献绝对值最大的 10 个行业（年化 %）】")
order = ind_tab["配置年化"].abs().sort_values(ascending=False).index
log((ind_tab.loc[order].head(10)[["配置年化", "选股年化"]] * 100).round(2).to_string())


# ==================== 8. 集中度 ====================
log("\n⑤ 主动权重集中度...")
aw_records = []
for d, picks in holdings.items():
    r5 = fwd5.loc[d]
    bench_codes = r5.where(mask.loc[d]).dropna().index
    if len(bench_codes) < 100:
        continue
    w_b = pd.Series(1.0 / len(bench_codes), index=bench_codes)
    w_p = pd.Series(1.0 / len(picks), index=picks)
    aw = w_p.sub(w_b.reindex(w_p.index).fillna(0))
    aw_records.append({"date": d, "top10_abs": aw.abs().nlargest(10).sum()})

awt = pd.DataFrame(aw_records).set_index("date")
log(f"   前 10 大主动权重绝对值之和（均值）: {awt.top10_abs.mean():.3f}")

freq = {}
for d, picks in holdings.items():
    for c in picks:
        freq[c] = freq.get(c, 0) + 1
log("\n   最常被持有的 10 只股票（期数 / 总期数）:")
for c, n in pd.Series(freq).sort_values(ascending=False).head(10).items():
    log(f"     {c:<12} {industry.get(c, '?'):<10} {n:>3} / {len(holdings)}")


# ==================== 9. 存盘 ====================
risk_tab.to_csv(OUT / "te_summary.csv", index=False, encoding="utf-8-sig")
ret_tab.to_csv(OUT / "return_attribution.csv", index=False, encoding="utf-8-sig")
ind_tab.to_csv(OUT / "brinson_industry.csv", encoding="utf-8-sig")
br[["r_p", "r_b", "active", "alloc", "sel", "inter"]].to_csv(
    OUT / "period_detail.csv", encoding="utf-8-sig")

log(f"\n✅ 输出到 {OUT}/")
log(f"总耗时 {(time.time() - t0) / 60:.1f} 分钟")
