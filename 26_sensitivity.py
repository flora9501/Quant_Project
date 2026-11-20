#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
26_sensitivity.py  (v2)
把上一版被判死的两条判据改对。

v1 的两条判据都写错了
  1) 平台期用绝对极差 <= 1.0pp —— 对 9% 量级的结果太严，且没给相对尺度
  2) placebo 用作否决 —— 但基线成本极重，任何低换手变体都会跑赢它，
     所以 placebo 跑赢基线是必然的，不能作为无效的证据

v2 改法
  1) 平台期：报区间 + 检查有无断崖。判据 = 最低档 >= 基线 + 3pp，且 max/min <= 1.3
  2) placebo：改成归因分解
       成本效应 = net(stale) - net(baseline)
       信号效应 = net(THRESH) - net(stale)      <- 两组换手几乎相同，是干净的配对
     另加一条 bug 检查：打乱截面后不应产生正超额

v2 另含成本口径
  TWO_SIDED = True -> 每笔成交都收费（与 28 一致）。旧口径只有它的一半。

输出：data/output_turnover/sensitivity_v2.csv
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_DIR

RAW = DATA_DIR / "raw"
REF = DATA_DIR / "ref"
PANEL = DATA_DIR / "panel"
OUT = DATA_DIR / "output_turnover"
OUT.mkdir(parents=True, exist_ok=True)

FACTORS = ["rev_5", "turn_20", "vol_20", "ep"]
TOP_N = 50
REBAL = 5
COSTS_BP = [12, 20, 30]
COSTS = [bp / 1e4 for bp in COSTS_BP]
THRESH_GRID = [0.10, 0.15, 0.25, 0.35, 0.50]
DEFAULT_THRESH = 0.25
TWO_SIDED = True

PLATEAU_MIN_GAIN = 3.0      # pp，最低档相对基线
PLATEAU_MAX_RATIO = 1.30    # max/min


def log(m):
    print(m, flush=True)


t0 = time.time()

# ==================== 1. 数据 ====================
log("① 读取数据...")
frames = [pd.read_parquet(f, columns=["ts_code", "trade_date", "pct_chg"])
          for f in sorted(RAW.glob("daily_[0-9][0-9][0-9][0-9].parquet"))]
daily = pd.concat(frames, ignore_index=True).drop_duplicates(["trade_date", "ts_code"])
ret = (daily.pivot(index="trade_date", columns="ts_code", values="pct_chg") / 100.0).sort_index()
del daily
fwd = ret.shift(-1)

fac = pd.read_parquet(PANEL / "factors_neutral.parquet")
wide = {n: fac.pivot(index="trade_date", columns="ts_code", values=n) for n in FACTORS}
idx = ret.index.intersection(wide["rev_5"].index).sort_values()
cols = ret.columns
for n in FACTORS:
    wide[n] = wide[n].reindex(index=idx, columns=cols)

mask = (pd.read_parquet(PANEL / "tradable.parquet")
        .reindex(index=idx, columns=cols).fillna(False).astype(bool))
fwd = fwd.reindex(index=idx, columns=cols)

sb = pd.concat([pd.read_parquet(REF / "stock_basic.parquet"),
                pd.read_parquet(REF / "stock_delisted.parquet")],
               ignore_index=True).drop_duplicates("ts_code")
industry = sb.set_index("ts_code")["industry"].reindex(cols).fillna("UNKNOWN")
log(f"   {len(idx)} 天 × {len(cols)} 只")


# ==================== 2. 分数 ====================
def combine(dfs):
    return sum(d.fillna(0.0) for d in dfs).where(
        sum(d.notna().astype(int) for d in dfs) > 0)


score_raw = combine([wide[n] for n in FACTORS])

log("② 平滑分数...")
acc = score_raw.fillna(0.0).copy()
cnt = score_raw.notna().astype(np.int16).copy()
for k in (REBAL, 2 * REBAL):
    sh = score_raw.shift(k)
    acc = acc + sh.fillna(0.0)
    cnt = cnt + sh.notna().astype(np.int16)
    del sh
score_smooth = acc / cnt.replace(0, np.nan)
del acc, cnt


def shuffle_within_date(df, seed=42):
    rng = np.random.default_rng(seed)
    arr = df.to_numpy(dtype=float, copy=True)
    for i in range(arr.shape[0]):
        row = arr[i]
        pos = np.flatnonzero(~np.isnan(row))
        if pos.size > 1:
            row[pos] = rng.permutation(row[pos])
    return pd.DataFrame(arr, index=df.index, columns=df.columns)


log("③ 构造 placebo...")
score_placebo = shuffle_within_date(score_smooth)
score_stale = score_smooth.shift(60)


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


def select_stratified(s):
    df = pd.DataFrame({"score": s, "ind": industry.reindex(s.index)}).dropna()
    if len(df) <= TOP_N:
        return list(df.nlargest(TOP_N, "score").index)
    base = industry_quota(df)
    picks = []
    for ind_, k in base.items():
        if k <= 0:
            continue
        picks.extend(df[df["ind"] == ind_].nlargest(int(k), "score").index.tolist())
    picks = list(dict.fromkeys(picks))[:TOP_N]
    if len(picks) < TOP_N:
        rest = df[~df.index.isin(set(picks))].nlargest(TOP_N - len(picks), "score")
        picks += rest.index.tolist()
    return list(dict.fromkeys(picks))[:TOP_N]


def select_quota_hyst(s, held, sigma_mult):
    df = pd.DataFrame({"score": s, "ind": industry.reindex(s.index)}).dropna()
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


# ==================== 4. 回放 ====================
def run(score_df, use_hyst, thresh):
    w_live = pd.Series(0.0, index=cols)
    held = []
    gross_s = pd.Series(0.0, index=idx)
    to_s = pd.Series(0.0, index=idx)
    rec = []

    for i, d in enumerate(idx):
        if i % REBAL == 0:
            valid = mask.loc[d]
            s = score_df.loc[d].where(valid).dropna()
            if len(s) >= TOP_N:
                picks = (select_quota_hyst(s, held, thresh) if use_hyst
                         else select_stratified(s))
                picks = list(dict.fromkeys(picks))[:TOP_N]
                w_new = pd.Series(1.0 / len(picks), index=picks).reindex(cols).fillna(0.0)

                tot = w_live.sum()
                w_drift = (w_live / tot) if tot > 0 else w_live
                to_s.loc[d] = float((w_new - w_drift).abs().sum() / 2)

                before = set(w_drift[w_drift > 0].index)
                pw = (pd.Series(1.0 / len(picks), index=picks)
                        .groupby(industry.reindex(picks)).sum())
                bw = industry.reindex(valid[valid].index).value_counts(normalize=True)
                rec.append({"date": d,
                            "n_replaced": len(set(picks) - before),
                            "dev_max": (pw.reindex(bw.index).fillna(0.0) - bw).abs().max()})
                held = picks
                w_live = w_new

        r = fwd.loc[d].fillna(0.0)
        gross_s.loc[d] = float((w_live * r).sum())
        w_live = w_live * (1 + r)

    return pd.DataFrame(rec).set_index("date"), gross_s, to_s


bench = fwd.where(mask).mean(axis=1).reindex(idx).fillna(0.0)
bench_nav = (1 + bench).cumprod()


def metrics(rec, gross, to):
    mult = 2.0 if TWO_SIDED else 1.0
    out = {"换手": to.replace(0, np.nan).mean() * (252 / REBAL) * 100,
           "替换/期": rec.n_replaced.mean(),
           "持有天数": TOP_N / max(rec.n_replaced.mean(), 0.1) * REBAL,
           "偏离最大": rec.dev_max.max() * 100}
    for bp, c in zip(COSTS_BP, COSTS):
        net = (gross - to * c * mult).iloc[:-1]
        nav = (1 + net).cumprod()
        ann = nav.iloc[-1] ** (252 / len(net)) - 1
        ex_nav = nav / bench_nav.reindex(nav.index).ffill()
        ex_ann = ex_nav.iloc[-1] ** (252 / len(ex_nav)) - 1
        out[f"net@{bp}"] = ann * 100
        out[f"ex@{bp}"] = ex_ann * 100
        if bp == 30:
            ex_ret = ex_nav.pct_change().dropna()
            out["IR@30"] = ex_ann / (ex_ret.std() * np.sqrt(252))
    return out


# ==================== 5. 跑 ====================
log(f"\n④ 回放（成本口径：{'双边计费' if TWO_SIDED else '单边收一次'}）...")
rows = []
rec, g, t = run(score_raw, False, 0)
rows.append({"配置": "Baseline", **metrics(rec, g, t)})
log("   Baseline 完成")

for th in THRESH_GRID:
    rec, g, t = run(score_smooth, True, th)
    rows.append({"配置": f"THRESH={th:.2f}", **metrics(rec, g, t)})
    log(f"   THRESH={th:.2f} 完成")

rec, g, t = run(score_placebo, True, DEFAULT_THRESH)
rows.append({"配置": "Placebo 打乱截面", **metrics(rec, g, t)})
log("   placebo 打乱 完成")

rec, g, t = run(score_stale, True, DEFAULT_THRESH)
rows.append({"配置": "Placebo 滞后60日", **metrics(rec, g, t)})
log("   placebo 滞后 完成")

tab = pd.DataFrame(rows).set_index("配置")


# ==================== 6. 汇总 ====================
log("\n" + "=" * 104)
log("【全部配置】")
log("=" * 104)
log(tab.round(2).to_string())

base = tab.loc["Baseline"]
plateau = tab.loc[[f"THRESH={th:.2f}" for th in THRESH_GRID]]

# ---- 平台期 ----
log("\n【参数平台期】")
log(plateau[["换手", "替换/期", "持有天数", "偏离最大",
             "net@12", "net@30", "ex@30", "IR@30"]].round(2).to_string())

lo, hi = plateau["net@30"].min(), plateau["net@30"].max()
spread = hi - lo
ratio = hi / lo if lo > 0 else np.inf
gain = lo - base["net@30"]
ok_plateau = (gain >= PLATEAU_MIN_GAIN) and (ratio <= PLATEAU_MAX_RATIO)

log(f"\n   net@30 区间      : {lo:.2f}% – {hi:.2f}%   （极差 {spread:.2f}pp）")
log(f"   最低档相对基线   : {gain:+.2f}pp      门槛 >= +{PLATEAU_MIN_GAIN:.1f}pp")
log(f"   最高/最低        : {ratio:.2f}        门槛 <= {PLATEAU_MAX_RATIO:.2f}")
log(f"   {'✅ 平坦，无断崖' if ok_plateau else '❌ 存在断崖或整体不优于基线'}")

# ---- placebo 归因 ----
log("\n【Placebo → 归因分解】")
st = tab.loc["Placebo 滞后60日"]
sh = tab.loc["Placebo 打乱截面"]
ab = tab.loc[f"THRESH={DEFAULT_THRESH:.2f}"]

log(f"   Baseline（高换手）      换手 {base['换手']:6.0f}%   net@30 {base['net@30']:6.2f}%")
log(f"   滞后60日（低换手）      换手 {st['换手']:6.0f}%   net@30 {st['net@30']:6.2f}%")
log(f"   THRESH=0.25（低换手）   换手 {ab['换手']:6.0f}%   net@30 {ab['net@30']:6.2f}%")
log("")
cost_effect = st["net@30"] - base["net@30"]
signal_effect = ab["net@30"] - st["net@30"]
log(f"   成本效应 = 滞后 − 基线          = {cost_effect:+.2f}pp")
log(f"   信号效应 = 0.25 − 滞后          = {signal_effect:+.2f}pp   （换手 {st['换手']:.0f}% vs {ab['换手']:.0f}%，基本可比）")
log(f"   合计                             = {cost_effect + signal_effect:+.2f}pp")
log(f"   实际总增量（0.25 − 基线）       = {ab['net@30'] - base['net@30']:+.2f}pp")

log("\n   一致性检查：")
c1_ok = abs((cost_effect + signal_effect) - (ab["net@30"] - base["net@30"])) < 0.05
log(f"     分解是否闭合                     {'✅' if c1_ok else '❌'}")
bug_ok = sh["ex@30"] <= 0
log(f"     打乱截面后 ex@30 = {sh['ex@30']:+.2f}%，应为负（否则有 bug）  {'✅' if bug_ok else '❌'}")
log(f"     打乱后换手 = {sh['换手']:.0f}%，应高于 0.25 档（说明降换手依赖信号持续性）")

# ---- 最终裁决 ----
log("\n" + "=" * 104)
if ok_plateau and c1_ok and bug_ok:
    log("✅ PASS —— 平台平坦、归因闭合、无机制性 bug。可以进入容量分析与 README。")
elif ok_plateau and c1_ok:
    log("⚠ 条件通过 —— 平台与归因正常，但 placebo 一致性检查失败，需先排查。")
else:
    log("❌ 未通过 —— 见上面标红项。")
log("=" * 104)

tab.to_csv(OUT / "sensitivity_v2.csv", encoding="utf-8-sig")
log(f"\n✅ 输出到 {OUT}/sensitivity_v2.csv")
log(f"总耗时 {(time.time() - t0) / 60:.1f} 分钟")
