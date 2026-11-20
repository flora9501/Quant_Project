"""
08_attribution.py
A/B 对比归因：
  A) Top50 等权（行业未控）
  B) Top50 行业分层抽样（行业受控）
输出 data/output_attribution/
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DATA_DIR

RAW = DATA_DIR / "raw"
REF = DATA_DIR / "ref"
PANEL = DATA_DIR / "panel"
OUT = DATA_DIR / "output_attribution"
OUT.mkdir(parents=True, exist_ok=True)

FACTORS = ["rev_5", "turn_20", "vol_20", "ep"]
TOP_N, REBAL, COST = 50, 5, 0.0012
SIZE_BUCKETS = 1


def log(m):
    print(m, flush=True)


# ==================== 数据 ====================
log("① 读取数据...")
frames = []
for f in sorted(RAW.glob("daily_[0-9][0-9][0-9][0-9].parquet")):
    frames.append(pd.read_parquet(f, columns=["ts_code", "trade_date", "pct_chg"]))
daily = pd.concat(frames, ignore_index=True).drop_duplicates(["trade_date", "ts_code"])
ret = (daily.pivot(index="trade_date", columns="ts_code", values="pct_chg") / 100).sort_index()
fwd = ret.shift(-1)

fac = pd.read_parquet(PANEL / "factors_neutral.parquet")
wide = {n: fac.pivot(index="trade_date", columns="ts_code", values=n) for n in FACTORS}

idx = ret.index.intersection(wide["rev_5"].index).sort_values()
cols = ret.columns
for n in FACTORS:
    wide[n] = wide[n].reindex(index=idx, columns=cols)
fwd = fwd.reindex(index=idx, columns=cols)

bb = []
for f in sorted(RAW.glob("daily_basic_*.parquet")):
    bb.append(pd.read_parquet(f, columns=["ts_code", "trade_date", "circ_mv"]))
bb = pd.concat(bb, ignore_index=True).drop_duplicates(["trade_date", "ts_code"])
logmv = np.log(bb.pivot(index="trade_date", columns="ts_code", values="circ_mv")
                .reindex(index=idx, columns=cols) * 1e4)

mask = (pd.read_parquet(PANEL / "tradable.parquet")
        .reindex(index=idx, columns=cols).fillna(False).astype(bool))

sb = pd.concat([pd.read_parquet(REF / "stock_basic.parquet"),
                pd.read_parquet(REF / "stock_delisted.parquet")],
               ignore_index=True).drop_duplicates("ts_code")
industry = sb.set_index("ts_code")["industry"].reindex(cols).fillna("UNKNOWN")

score = sum(wide[n].fillna(0) for n in FACTORS) / len(FACTORS)
bench = fwd.where(mask).mean(axis=1)

log(f"   {len(idx)} 天 × {len(cols)} 只   行业数 {industry.nunique()}")


# ==================== 选股 ====================
def select_plain(s):
    return list(s.nlargest(TOP_N).index)


def select_stratified(s, valid, d):
    lm = logmv.loc[d]
    sz = np.ceil(lm.where(valid).rank(pct=True) * SIZE_BUCKETS).clip(1, max(SIZE_BUCKETS, 1))

    df = pd.DataFrame({"score": s,
                       "ind": industry.reindex(s.index),
                       "sz": sz.reindex(s.index)}).dropna()
    if len(df) <= TOP_N:
        return list(df.nlargest(TOP_N, "score").index)

    share = df.groupby(["ind", "sz"]).size() / len(df)
    raw = share * TOP_N
    base = np.floor(raw).astype(int)
    remain = TOP_N - int(base.sum())
    if remain > 0:
        for key in list((raw - base).sort_values(ascending=False).index)[:remain]:
            base.loc[key] += 1

    picks = []
    for (ind_, sz_), k in base.items():
        if k <= 0:
            continue
        sub = df[(df["ind"] == ind_) & (df["sz"] == sz_)].nlargest(int(k), "score")
        picks.extend(sub.index.tolist())
    picks = list(dict.fromkeys(picks))[:TOP_N]

    if len(picks) < TOP_N:
        taken = set(picks)
        rest = df[~df.index.isin(taken)].nlargest(TOP_N - len(picks), "score")
        picks += rest.index.tolist()
    return list(dict.fromkeys(picks))[:TOP_N]


def _ind_dev(picks, universe_codes):
    if not picks:
        return np.nan
    pw = pd.Series(1.0 / len(picks), index=picks)
    port = pw.groupby(industry.reindex(picks)).sum()
    bw = industry.reindex(universe_codes).value_counts(normalize=True)
    return float((port.reindex(bw.index).fillna(0) - bw).abs().max())


def run(mode):
    w_prev = pd.Series(0.0, index=cols)
    rows = []
    for i, d in enumerate(idx):
        to = 0.0
        ind_dev = np.nan
        if i % REBAL == 0:
            valid = mask.loc[d]
            s = score.loc[d].where(valid).dropna()
            if len(s) >= TOP_N:
                picks = select_stratified(s, valid, d) if mode == "stratified" else select_plain(s)
                if picks:
                    ind_dev = _ind_dev(picks, valid[valid].index)
                    w_new = pd.Series(1.0 / len(picks), index=picks).reindex(cols).fillna(0.0)
                    to = (w_new - w_prev).abs().sum() / 2
                    w_prev = w_new
        r = fwd.loc[d].fillna(0)
        rows.append({"date": d, "net": float((w_prev * r).sum()) - to * COST,
                     "turnover": to, "ind_dev": ind_dev})
    res = pd.DataFrame(rows).set_index("date")
    return res["net"].iloc[:-1], res["ind_dev"].dropna()


# ==================== 风格因子 ====================
def q_spread(factor, q=0.2):
    qq = factor.where(mask).rank(axis=1, pct=True)
    return fwd.where(qq > 1 - q).mean(axis=1) - fwd.where(qq <= q).mean(axis=1)


style = pd.DataFrame({
    "size":   q_spread(-logmv),
    "value":  q_spread(wide["ep"]),
    "lowvol": q_spread(wide["vol_20"]),
}).reindex(idx).fillna(0)

log("\n===== 风格因子年化收益（%）=====")
print((style.mean() * 252 * 100).round(2).to_string())


# ==================== 归因 ====================
def attribute(rets, label):
    excess = (rets - bench.reindex(rets.index).fillna(0)).dropna()
    X = sm.add_constant(style.reindex(excess.index).fillna(0))
    m = sm.OLS(excess, X).fit()

    log(f"\n===== {label} =====")
    log(f"R² = {m.rsquared:.3f}")
    tab = pd.DataFrame({"beta": m.params, "t": m.tvalues})
    print(tab.round(4).to_string())
    log(f"alpha = {m.params['const'] * 252:.2%}   t = {m.tvalues['const']:.2f}")

    return {
        "alpha年化": m.params["const"] * 252,
        "alpha_t": m.tvalues["const"],
        "size": m.params["size"],
        "value": m.params["value"],
        "lowvol": m.params["lowvol"],
        "size_t": m.tvalues["size"],
        "value_t": m.tvalues["value"],
        "lowvol_t": m.tvalues["lowvol"],
        "R2": m.rsquared,
    }


log("\n② 回测两个组合...")
net_plain, dev_plain = run("plain")
net_strat, dev_strat = run("stratified")

log(f"   A 行业偏离  均值 {dev_plain.mean():.4f}  最大 {dev_plain.max():.4f}")
log(f"   B 行业偏离  均值 {dev_strat.mean():.4f}  最大 {dev_strat.max():.4f}")

a = attribute(net_plain, "A. Top50 等权（行业未控）")
b = attribute(net_strat, "B. Top50 分层抽样（行业受控）")

# ==================== 对比表 ====================
cmp = pd.DataFrame({"A_行业未控": a, "B_行业受控": b})
cmp["变化"] = (cmp["B_行业受控"] - cmp["A_行业未控"]).round(4)

log("\n===== 对比 =====")
print(cmp.round(4).to_string())
cmp.to_csv(OUT / "attribution_compare.csv", encoding="utf-8-sig")


# ==================== 关键解读（修好后的版本） ====================
log("\n===== 关键解读 =====")
for k in ["size", "value", "lowvol"]:
    if abs(a[k]) < 0.02:
        log(f"  {k:7s} beta: {a[k]:+.4f} → {b[k]:+.4f}   （基准接近 0，占比无意义）")
    else:
        change = (a[k] - b[k]) / a[k]
        log(f"  {k:7s} beta: {a[k]:+.3f} → {b[k]:+.3f}   （行业配置贡献 {change:.0%}）")

log(f"\n  alpha : {a['alpha年化']:.2%} → {b['alpha年化']:.2%}")
log(f"  t 值  : {a['alpha_t']:.2f} → {b['alpha_t']:.2f}")
log(f"  R²    : {a['R2']:.3f} → {b['R2']:.3f}")


# ==================== 保存超额净值对比 ====================
def excess_nav(net):
    ex = net - bench.reindex(net.index).fillna(0)
    return (1 + ex).cumprod()


nav_a = excess_nav(net_plain)
nav_b = excess_nav(net_strat)

fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(pd.to_datetime(nav_a.index), nav_a.values, label="A: Top50 (industry uncontrolled)", linewidth=1.2)
ax.plot(pd.to_datetime(nav_b.index), nav_b.values, label="B: Top50 stratified (industry controlled)", linewidth=1.2)
ax.axhline(1, color="gray", linestyle="--", linewidth=1)
ax.set_title("Excess NAV: industry bets removed")
ax.legend()
ax.grid(alpha=0.3)
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig(OUT / "excess_nav_compare.png", dpi=130)
plt.close()

pd.DataFrame({"A_行业未控": net_plain, "B_行业受控": net_strat}).to_csv(
    OUT / "daily_returns_ab.csv", encoding="utf-8-sig")

log(f"\n✅ 保存到 {OUT}/")
