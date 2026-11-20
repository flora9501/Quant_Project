"""
09_style_neutral_test.py
A) 超额收益归因（不含市场）
B) 风格中性组合（诊断性，方法粗糙，结论需谨慎）
输出 data/output_style/
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DATA_DIR

RAW = DATA_DIR / "raw"
PANEL = DATA_DIR / "panel"
OUT = DATA_DIR / "output_style"
OUT.mkdir(parents=True, exist_ok=True)

FACTORS = ["rev_5", "turn_20", "vol_20", "ep"]
TOP_N, REBAL, COST = 50, 5, 0.0012


def log(m):
    print(m, flush=True)


# ==================== 数据 ====================
frames = []
for f in sorted(RAW.glob("daily_[0-9][0-9][0-9][0-9].parquet")):
    frames.append(pd.read_parquet(f, columns=["ts_code", "trade_date", "pct_chg"]))
daily = pd.concat(frames, ignore_index=True).drop_duplicates(["trade_date", "ts_code"])
ret = (daily.pivot(index="trade_date", columns="ts_code", values="pct_chg") / 100).sort_index()
fwd = ret.shift(-1)

bframes = []
for f in sorted(RAW.glob("daily_basic_*.parquet")):
    bframes.append(pd.read_parquet(f, columns=["ts_code", "trade_date", "circ_mv"]))
bb = pd.concat(bframes, ignore_index=True).drop_duplicates(["trade_date", "ts_code"])

fac = pd.read_parquet(PANEL / "factors_neutral.parquet")
wide = {n: fac.pivot(index="trade_date", columns="ts_code", values=n) for n in FACTORS}

idx = ret.index.intersection(wide["rev_5"].index).sort_values()
cols = ret.columns

fwd = fwd.reindex(index=idx, columns=cols)
for n in FACTORS:
    wide[n] = wide[n].reindex(index=idx, columns=cols)

logmv = np.log(bb.pivot(index="trade_date", columns="ts_code", values="circ_mv")
                .reindex(index=idx, columns=cols) * 1e4)

mask = (pd.read_parquet(PANEL / "tradable.parquet")
        .reindex(index=idx, columns=cols).fillna(False).astype(bool))

score = sum(wide[n].fillna(0) for n in FACTORS) / len(FACTORS)
bench = fwd.where(mask).mean(axis=1)


# ==================== 回测 ====================
def run(score_mat):
    w_prev = pd.Series(0.0, index=cols)
    rows = []
    for i, d in enumerate(idx):
        to = 0.0
        if i % REBAL == 0:
            s = score_mat.loc[d].where(mask.loc[d]).dropna()
            if len(s) >= TOP_N:
                held = s.nlargest(TOP_N).index
                w_new = pd.Series(1 / TOP_N, index=held).reindex(cols).fillna(0)
                to = (w_new - w_prev).abs().sum() / 2
                w_prev = w_new
        r = fwd.loc[d].fillna(0)
        rows.append({"date": d, "net": float((w_prev * r).sum()) - to * COST})
    return pd.DataFrame(rows).set_index("date")["net"].iloc[:-1]


# ==================== 风格因子 ====================
def q_spread(factor, q=0.2):
    qq = factor.where(mask).rank(axis=1, pct=True)
    return fwd.where(qq > 1 - q).mean(axis=1) - fwd.where(qq <= q).mean(axis=1)


# wide["vol_20"] = -σ√252，数值越大 = 波动越低
# q_spread(wide["vol_20"]) = 低波组 - 高波组
style = pd.DataFrame({
    "size":   q_spread(-logmv),
    "value":  q_spread(wide["ep"]),
    "lowvol": q_spread(wide["vol_20"]),
}).reindex(idx).fillna(0)

log("\n===== 风格因子年化收益（%）=====")
prem = (style.mean() * 252 * 100).round(2)
print(prem.to_string())

prem.to_frame("annualized_return_pct").to_csv(OUT / "style_premiums.csv",
                                              encoding="utf-8-sig")


# ==================== A. 超额回归 ====================
strat = run(score)
excess = (strat - bench.reindex(strat.index).fillna(0)).dropna()
X = sm.add_constant(style.reindex(excess.index).fillna(0))
m1 = sm.OLS(excess, X).fit()

log("\n===== A. 超额收益归因（不含市场）=====")
log(f"R² = {m1.rsquared:.3f}")
tab_a = pd.DataFrame({"beta": m1.params, "t": m1.tvalues, "p": m1.pvalues})
print(tab_a.round(4).to_string())

log(f"\n纯 alpha（年化）= {m1.params['const'] * 252:.2%}")
log(f"alpha t 值       = {m1.tvalues['const']:.2f}   "
    f"{'显著' if abs(m1.tvalues['const']) > 2 else '不显著'}")

tab_a.to_csv(OUT / "attribution_excess.csv", encoding="utf-8-sig")


# ==================== B. 风格中性组合 ====================
log("\n② 构建风格中性组合（逐日对 size/value/lowvol 残差化）...")

SC = score.values
REG = np.stack([logmv.values,
                wide["ep"].values,
                wide["vol_20"].values], axis=2)
NEU = np.full_like(SC, np.nan, dtype=float)

for i in range(SC.shape[0]):
    y = SC[i]
    R = REG[i]
    ok = np.isfinite(y)
    for j in range(R.shape[1]):
        ok &= np.isfinite(R[:, j])
    if ok.sum() < 50:
        continue
    Xd = np.column_stack([np.ones(ok.sum()), R[ok]])
    beta, *_ = np.linalg.lstsq(Xd, y[ok], rcond=None)
    r = np.full_like(y, np.nan)
    r[ok] = y[ok] - Xd @ beta
    NEU[i] = r

neutral_score = pd.DataFrame(NEU, index=idx, columns=cols)
log(f"   完成，有效格子 {int(np.isfinite(NEU).sum()):,}")

strat_n = run(neutral_score)
ex_n = (strat_n - bench.reindex(strat_n.index).fillna(0)).dropna()
m2 = sm.OLS(ex_n, sm.add_constant(style.reindex(ex_n.index).fillna(0))).fit()

log("\n===== B. 风格中性组合（诊断性）=====")
log(f"R² = {m2.rsquared:.3f}")
tab_b = pd.DataFrame({"beta": m2.params, "t": m2.tvalues})
print(tab_b.round(4).to_string())

nav_n = (1 + strat_n).cumprod()
ann_n = nav_n.iloc[-1] ** (252 / len(strat_n)) - 1
log(f"\n年化        {ann_n:.2%}")
log(f"超额年化     {ex_n.mean() * 252:.2%}")
log(f"纯 alpha     {m2.params['const'] * 252:.2%}   t = {m2.tvalues['const']:.2f}")
log("⚠️ 注意：个股分数残差化 ≠ 组合中性化，组合仍有显著负暴露，结论仅作参考")

tab_b.to_csv(OUT / "attribution_neutral.csv", encoding="utf-8-sig")

pd.DataFrame({
    "original": {
        "年化": (1 + strat).cumprod().iloc[-1] ** (252 / len(strat)) - 1,
        "超额年化": excess.mean() * 252,
        "alpha_年化": m1.params["const"] * 252,
        "alpha_t": m1.tvalues["const"],
    },
    "style_neutral": {
        "年化": ann_n,
        "超额年化": ex_n.mean() * 252,
        "alpha_年化": m2.params["const"] * 252,
        "alpha_t": m2.tvalues["const"],
    },
}).to_csv(OUT / "summary_compare.csv", encoding="utf-8-sig")


# ==================== 画图并保存 ====================
log("\n③ 画图...")

# 图 1：风格因子累计收益
fig, ax = plt.subplots(figsize=(10, 5))
for c in ["size", "value", "lowvol"]:
    ax.plot(pd.to_datetime(style.index), (1 + style[c]).cumprod().values,
            label=c, linewidth=1.2)
ax.set_yscale("log")
ax.set_title("Style factor cumulative returns (long-short)")
ax.legend()
ax.grid(alpha=0.3)
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig(OUT / "style_premiums.png", dpi=130)
plt.close()

# 图 2：原组合 / 风格中性组 合 / 基准
fig, ax = plt.subplots(figsize=(11, 6))
b_nav = (1 + bench.reindex(idx).fillna(0)).cumprod()
ax.plot(pd.to_datetime(idx), b_nav.values, label="Benchmark (EW)",
        color="gray", linestyle="--", linewidth=1)
ax.plot(pd.to_datetime(strat.index), (1 + strat).cumprod().values,
        label="Original (Top50)", linewidth=1.3)
ax.plot(pd.to_datetime(strat_n.index), nav_n.values,
        label="Style-neutral (diagnostic)", linewidth=1.3)
ax.set_yscale("log")
ax.set_title("Original vs Style-Neutral vs Benchmark")
ax.legend()
ax.grid(alpha=0.3)
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig(OUT / "nav_compare.png", dpi=130)
plt.close()

log(f"   ✅ 已保存到 {OUT}/")
log("   文件：style_premiums.csv / attribution_excess.csv / "
    "attribution_neutral.csv / summary_compare.csv / style_premiums.png / nav_compare.png")
