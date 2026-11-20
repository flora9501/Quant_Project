"""
07_backtest.py
第 6 天：因子合成 → 组合构建 → 扣成本 → 净值 / 超额 / 多空 / 行业偏离
五种方案：
  Equal Weight / IC Weighted / Orthogonalized / EW + Buffer / EW + Stratified
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DATA_DIR

RAW = DATA_DIR / "raw"
REF = DATA_DIR / "ref"
PANEL = DATA_DIR / "panel"
OUT = DATA_DIR / "output_backtest"
OUT.mkdir(parents=True, exist_ok=True)

FACTORS = ["rev_5", "turn_20", "vol_20", "ep"]
TOP_N = 50
BUFFER_N = 100
REBAL = 5
COST = 0.0012
SIZE_BUCKETS = 1


def log(m):
    print(m, flush=True)


t0 = time.time()

# ==================== 1. 收益 ====================
log("① 收益矩阵...")
frames = []
for f in sorted(RAW.glob("daily_[0-9][0-9][0-9][0-9].parquet")):
    frames.append(pd.read_parquet(f, columns=["ts_code", "trade_date", "pct_chg"]))
daily = pd.concat(frames, ignore_index=True).drop_duplicates(["trade_date", "ts_code"])
ret = (daily.pivot(index="trade_date", columns="ts_code", values="pct_chg") / 100.0).sort_index()
del daily
fwd = ret.shift(-1)

# ==================== 2. 因子 / 掩码 / 市值 / 行业 ====================
log("② 读取因子、掩码、市值、行业...")
fac = pd.read_parquet(PANEL / "factors_neutral.parquet")
wide = {n: fac.pivot(index="trade_date", columns="ts_code", values=n) for n in FACTORS}

idx = ret.index.intersection(wide["rev_5"].index).sort_values()
cols = ret.columns
for n in FACTORS:
    wide[n] = wide[n].reindex(index=idx, columns=cols)

b = []
for f in sorted(RAW.glob("daily_basic_*.parquet")):
    b.append(pd.read_parquet(f, columns=["ts_code", "trade_date", "circ_mv"]))
bb = pd.concat(b, ignore_index=True).drop_duplicates(["trade_date", "ts_code"])
logmv = np.log(bb.pivot(index="trade_date", columns="ts_code", values="circ_mv")
                .reindex(index=idx, columns=cols) * 1e4)

mask = (pd.read_parquet(PANEL / "tradable.parquet")
        .reindex(index=idx, columns=cols).fillna(False).astype(bool))
fwd = fwd.reindex(index=idx, columns=cols)

sb = pd.concat([pd.read_parquet(REF / "stock_basic.parquet"),
                pd.read_parquet(REF / "stock_delisted.parquet")],
               ignore_index=True).drop_duplicates("ts_code")
industry = (sb.set_index("ts_code")["industry"]
              .reindex(cols).fillna("UNKNOWN"))

log(f"   {len(idx)} 天 × {len(cols)} 只   行业数 {industry.nunique()}")


# ==================== 3. Rank IC ====================
def rank_ic(f, r):
    fr, rr = f.rank(axis=1), r.rank(axis=1)
    both = fr.notna() & rr.notna()
    fr, rr = fr.where(both), rr.where(both)
    a = fr.sub(fr.mean(axis=1), axis=0)
    b_ = rr.sub(rr.mean(axis=1), axis=0)
    num = (a * b_).sum(axis=1)
    den = np.sqrt((a ** 2).sum(axis=1) * (b_ ** 2).sum(axis=1))
    return (num / den).replace([np.inf, -np.inf], np.nan)


log("③ 计算 IC 序列...")
ic_df = pd.DataFrame({n: rank_ic(wide[n], fwd) for n in FACTORS}, index=idx)


# ==================== 4. 三种合成 ====================
log("④ 因子合成...")


def combine(dfs, weights=None):
    if weights is None:
        parts = [d.fillna(0.0) for d in dfs]
    else:
        parts = [d.fillna(0.0).mul(w, axis=0) for d, w in zip(dfs, weights)]
    out = sum(parts)
    valid = sum(d.notna().astype(int) for d in dfs) > 0
    return out.where(valid)


score_eq = combine([wide[n] for n in FACTORS])

w_ic = ic_df.rolling(252, min_periods=60).mean().shift(1)
w_norm = w_ic.div(w_ic.abs().sum(axis=1).replace(0, np.nan), axis=0)
w_norm = w_norm.fillna(1 / len(FACTORS))
score_ic = combine([wide[n] for n in FACTORS], [w_norm[n] for n in FACTORS])


def orthogonalize(y, x):
    Y, X = y.values.copy(), x.values
    for i in range(Y.shape[0]):
        yy, xx = Y[i], X[i]
        ok = np.isfinite(yy) & np.isfinite(xx)
        if ok.sum() < 20:
            continue
        b1, b0 = np.polyfit(xx[ok], yy[ok], 1)
        r = yy - (b1 * xx + b0)
        s = r[ok].std()
        if s > 0:
            Y[i] = (r - r[ok].mean()) / s
    return pd.DataFrame(Y, index=y.index, columns=y.columns)


log("   正交化 vol_20 ...")
vol_orth = orthogonalize(wide["vol_20"], wide["turn_20"])
score_orth = combine([wide["rev_5"], wide["turn_20"], vol_orth, wide["ep"]])


# ==================== 5. 选股方法 ====================
def _ind_dev(picks, universe_codes):
    """组合行业权重 vs 基准行业权重的最大绝对偏离"""
    if len(picks) == 0:
        return np.nan
    pw = pd.Series(1.0 / len(picks), index=picks)
    port = pw.groupby(industry.reindex(picks)).sum()
    bench_w = industry.reindex(universe_codes).value_counts(normalize=True)
    return float((port.reindex(bench_w.index).fillna(0) - bench_w).abs().max())


def select_plain(s, valid):
    return list(s.nlargest(TOP_N).index)


def select_buffer(s, valid, held):
    ranked = s.sort_values(ascending=False)
    rank = {c: k for k, c in enumerate(ranked.index)}
    keep = [c for c in held if rank.get(c, 10 ** 9) < BUFFER_N]
    for c in ranked.index:
        if len(keep) >= TOP_N:
            break
        if c not in keep:
            keep.append(c)
    return keep


def select_stratified(s, valid, d, top_n=TOP_N, size_buckets=SIZE_BUCKETS):
    """
    行业 × 规模分层抽样，用最大余数法分配名额，保证 Σquota == top_n
    """
    lm = logmv.loc[d]
    sz = np.ceil(lm.where(valid).rank(pct=True) * size_buckets).clip(1, size_buckets)

    df = pd.DataFrame({
        "score": s,
        "ind": industry.reindex(s.index),
        "sz": sz.reindex(s.index),
    }).dropna()

    if len(df) <= top_n:
        return list(df.nlargest(top_n, "score").index)

    # 按桶的基准占比 → 期望名额
    share = df.groupby(["ind", "sz"]).size() / len(df)
    raw = share * top_n
    base = np.floor(raw).astype(int)

    # 最大余数法补齐到 top_n
    remain = top_n - int(base.sum())
    if remain > 0:
        order = (raw - base).sort_values(ascending=False).index
        for key in list(order)[:remain]:
            base.loc[key] += 1

    picks = []
    for (ind_, sz_), k in base.items():
        if k <= 0:
            continue
        sub = df[(df["ind"] == ind_) & (df["sz"] == sz_)].nlargest(int(k), "score")
        picks.extend(sub.index.tolist())

    picks = list(dict.fromkeys(picks))[:top_n]

    # 某桶股票不足时，用剩余最高分补齐
    if len(picks) < top_n:
        taken = set(picks)
        rest = df[~df.index.isin(taken)].nlargest(top_n - len(picks), "score")
        picks += rest.index.tolist()

    return list(dict.fromkeys(picks))[:top_n]


# ==================== 6. 回测 ====================
def backtest(score, mode="plain"):
    w_prev = pd.Series(0.0, index=cols)
    held = set()
    rows = []

    for i, d in enumerate(idx):
        to = 0.0
        ind_dev = np.nan

        if i % REBAL == 0:
            valid = mask.loc[d]
            s = score.loc[d].where(valid).dropna()
            if len(s) >= TOP_N:
                if mode == "buffer":
                    picks = select_buffer(s, valid, held)
                    held = set(picks)
                elif mode == "stratified":
                    picks = select_stratified(s, valid, d)
                else:
                    picks = select_plain(s, valid)

                if picks:
                    ind_dev = _ind_dev(picks, valid[valid].index)
                    w_new = pd.Series(1.0 / len(picks), index=picks).reindex(cols).fillna(0.0)
                    to = (w_new - w_prev).abs().sum() / 2
                    w_prev = w_new

        r = fwd.loc[d].reindex(cols).fillna(0.0)
        gross = float((w_prev * r).sum())
        rows.append({"date": d, "gross": gross, "cost": to * COST,
                     "net": gross - to * COST, "turnover": to, "ind_dev": ind_dev})

    return pd.DataFrame(rows).set_index("date")


def metrics(res):
    net = res["net"].iloc[:-1]
    nav = (1 + net).cumprod()
    ann = nav.iloc[-1] ** (252 / len(net)) - 1
    vol = net.std() * np.sqrt(252)
    dd = (nav / nav.cummax() - 1).min()
    dev = res["ind_dev"].dropna()
    return {
        "年化": ann,
        "波动": vol,
        "夏普": ann / vol if vol > 0 else np.nan,
        "最大回撤": dd,
        "Calmar": ann / abs(dd) if dd < 0 else np.nan,
        "年化换手": res["turnover"].mean() * 252,
        "年化成本": res["cost"].mean() * 252,
        "行业偏离均值": dev.mean(),
        "行业偏离最大": dev.max(),
        "nav": nav,
    }


VARIANTS = [
    ("Equal Weight",    score_eq,   "plain"),
    ("IC Weighted",     score_ic,   "plain"),
    ("Orthogonalized",  score_orth, "plain"),
    ("EW + Buffer",     score_eq,   "buffer"),
    ("EW + Stratified", score_eq,   "stratified"),
]

log("⑤ 回测五种方案...")
bench = fwd.where(mask).mean(axis=1)
results, navs = {}, {}
for name, sc, mode in VARIANTS:
    res = backtest(sc, mode=mode)
    m = metrics(res)
    navs[name] = m.pop("nav")
    results[name] = m
    log(f"   {name} 完成")

summary = pd.DataFrame(results).T
summary.to_csv(OUT / "backtest_summary.csv", encoding="utf-8-sig")

log("\n===== 回测结果（扣费后）=====")
print(summary.round(4).to_string())

# ==================== 7. 基准 ====================
bench_aligned = bench.reindex(navs["Equal Weight"].index).fillna(0.0)
bench_nav = (1 + bench_aligned).cumprod()
b_ann = bench_nav.iloc[-1] ** (252 / len(bench_aligned)) - 1
log("\n===== 基准（等权全池）=====")
log(f"   年化 {b_ann:.2%}   波动 {bench_aligned.std() * np.sqrt(252):.2%}")


# ==================== 8. 分年度 ====================
log("\n===== 分年度收益（扣费后，%）=====")
year_arr = pd.Series(navs["Equal Weight"].index).str[:4].values
yearly = pd.DataFrame({
    name: (1 + nav).groupby(year_arr).apply(lambda x: x.iloc[-1] / x.iloc[0] - 1)
    for name, nav in navs.items()
})
print((yearly * 100).round(2).to_string())
yearly.to_csv(OUT / "yearly.csv", encoding="utf-8-sig")


# ==================== 9. 超额 ====================
log("\n===== 超额（相对等权基准）=====")
extra = {}
for name, nav in navs.items():
    ex_nav = nav / bench_nav.reindex(nav.index).ffill()
    ex_ret = ex_nav.pct_change().dropna()
    ex_ann = ex_nav.iloc[-1] ** (252 / len(ex_nav)) - 1
    ex_vol = ex_ret.std() * np.sqrt(252)
    extra[name] = {
        "超额年化": ex_ann,
        "超额波动": ex_vol,
        "信息比率": ex_ann / ex_vol if ex_vol > 0 else np.nan,
        "超额最大回撤": (ex_nav / ex_nav.cummax() - 1).min(),
    }

extra_df = pd.DataFrame(extra).T
print(extra_df.round(4).to_string())
extra_df.to_csv(OUT / "excess.csv", encoding="utf-8-sig")


# ==================== 10. 多空诊断 ====================
def long_short(score):
    rows = []
    for i, d in enumerate(idx):
        if i % REBAL != 0:
            continue
        s = score.loc[d].where(mask.loc[d]).dropna()
        if len(s) < TOP_N * 2:
            continue
        hi = s.nlargest(TOP_N).index
        lo = s.nsmallest(TOP_N).index
        r = fwd.loc[d]
        rows.append({"date": d, "ls": float(r.reindex(hi).mean() - r.reindex(lo).mean())})
    return pd.DataFrame(rows).set_index("date")["ls"]


log("\n===== 多空（Top50 - Bottom50，诊断）=====")
ls_stats = {}
for name, sc, _ in VARIANTS[:3]:
    ls = long_short(sc).dropna()
    nav_ls = (1 + ls).cumprod()
    ann = nav_ls.iloc[-1] ** (252 / len(ls)) - 1
    vol = ls.std() * np.sqrt(252)
    ls_stats[name] = {"年化": ann, "波动": vol,
                      "夏普": ann / vol if vol > 0 else np.nan,
                      "最大回撤": (nav_ls / nav_ls.cummax() - 1).min()}
ls_df = pd.DataFrame(ls_stats).T
print(ls_df.round(4).to_string())
ls_df.to_csv(OUT / "long_short.csv", encoding="utf-8-sig")


# ==================== 11. 换手 / 成本 / 行业偏离 ====================
log("\n===== 换手 · 成本 · 行业偏离 =====")
tc = summary[["年化换手", "年化成本", "年化", "行业偏离均值", "行业偏离最大"]].copy()
tc["成本占毛收益"] = (tc["年化成本"] / (tc["年化"] + tc["年化成本"])).round(3)
print(tc.round(4).to_string())
tc.to_csv(OUT / "turnover_industry.csv", encoding="utf-8-sig")


# ==================== 12. 画图 ====================
log("\n⑦ 画图...")
fig, axes = plt.subplots(3, 1, figsize=(12, 13))

for name, nav in navs.items():
    axes[0].plot(pd.to_datetime(nav.index), nav.values, label=name, linewidth=1.1)
axes[0].plot(pd.to_datetime(bench_nav.index), bench_nav.values,
             label="Benchmark (EW)", color="gray", linestyle="--", linewidth=1)
axes[0].set_yscale("log")
axes[0].set_title(f"NAV | Top{TOP_N} / {REBAL}d / cost {COST:.2%} per side")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)

for name, nav in navs.items():
    ex = nav / bench_nav.reindex(nav.index).ffill()
    axes[1].plot(pd.to_datetime(ex.index), ex.values, label=name, linewidth=1.1)
axes[1].axhline(1, color="gray", linestyle="--", linewidth=1)
axes[1].set_title("Excess NAV vs benchmark")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)

for name, sc, _ in VARIANTS[:3]:
    ls = long_short(sc).dropna()
    axes[2].plot(pd.to_datetime(ls.index), (1 + ls).cumprod().values,
                 label=name, linewidth=1.1)
axes[2].set_title("Long-Short (Top50 - Bottom50) | diagnostic only")
axes[2].legend(fontsize=8)
axes[2].grid(alpha=0.3)

for ax in axes:
    ax.tick_params(axis="x", rotation=30, labelsize=8)

plt.tight_layout()
plt.savefig(OUT / "nav.png", dpi=130)
plt.close()

log(f"   ✅ 保存到 {OUT}/")
log(f"\n总耗时 {(time.time() - t0) / 60:.1f} 分钟")
