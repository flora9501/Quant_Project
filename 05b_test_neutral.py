"""
05b_test_neutral.py
中性化后的单因子检验：IC / ICIR / 分层 / 衰减
输出 data/output_neutral/
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DATA_DIR

RAW = DATA_DIR / "raw"
PANEL = DATA_DIR / "panel"
OUT = DATA_DIR / "output_neutral"
OUT.mkdir(parents=True, exist_ok=True)

FACTORS = ["rev_5", "turn_20", "vol_20", "ep"]


def log(m):
    print(m, flush=True)


t0 = time.time()

# ==================== 1. 收益矩阵 ====================
log("① 构建收益矩阵...")
frames = []
for f in sorted(RAW.glob("daily_[0-9][0-9][0-9][0-9].parquet")):
    frames.append(pd.read_parquet(f, columns=["ts_code", "trade_date", "pct_chg"]))
daily = pd.concat(frames, ignore_index=True).drop_duplicates(["trade_date", "ts_code"])

ret = (daily.pivot(index="trade_date", columns="ts_code", values="pct_chg") / 100.0)
ret = ret.sort_index()
del daily
log(f"   {ret.shape}")

fwd1 = ret.shift(-1)

# ==================== 2. 因子宽表 ====================
log("② 读取中性化因子...")
fac = pd.read_parquet(PANEL / "factors_neutral.parquet")

fac_wide = {
    name: fac.pivot(index="trade_date", columns="ts_code", values=name)
    for name in FACTORS
}

common_idx = ret.index.intersection(fac_wide["rev_5"].index).sort_values()
common_cols = ret.columns

for name in FACTORS:
    fac_wide[name] = fac_wide[name].reindex(index=common_idx, columns=common_cols)

fwd1 = fwd1.reindex(index=common_idx, columns=common_cols)
log(f"   对齐后 {len(common_idx)} 天 × {len(common_cols)} 只")

# 因子相关性
log("\n===== 因子相关性（中性化后）=====")
corr = fac[FACTORS].corr().round(3)
print(corr.to_string())
corr.to_csv(OUT / "factor_corr.csv", encoding="utf-8-sig")


# ==================== 3. Rank IC ====================
log("\n③ 计算 Rank IC...")


def rank_ic(f, r):
    """逐日截面 Spearman IC，只对两边都非缺失的格子计算"""
    fr = f.rank(axis=1)
    rr = r.rank(axis=1)

    both = fr.notna() & rr.notna()
    fr = fr.where(both)
    rr = rr.where(both)

    a = fr.sub(fr.mean(axis=1), axis=0)
    b = rr.sub(rr.mean(axis=1), axis=0)

    num = (a * b).sum(axis=1)
    den = np.sqrt((a ** 2).sum(axis=1) * (b ** 2).sum(axis=1))
    return (num / den).replace([np.inf, -np.inf], np.nan)


rows = []
ic_series = {}
for name in FACTORS:
    ic = rank_ic(fac_wide[name], fwd1).dropna()
    ic_series[name] = ic
    rows.append({
        "IC均值": ic.mean(),
        "IC标准差": ic.std(),
        "ICIR": ic.mean() / ic.std() if ic.std() > 0 else np.nan,
        "IC>0占比": (ic > 0).mean(),
        "样本天数": len(ic),
    })

ic_summary = pd.DataFrame(rows, index=FACTORS)
ic_summary.index.name = "factor"
ic_summary.to_csv(OUT / "ic_summary.csv", encoding="utf-8-sig")
log(f"   完成 {time.time() - t0:.0f}s")


# ==================== 4. 分层回测 ====================
log("④ 5 组分层...")


def layer_returns(f, r, n=5):
    q = f.rank(axis=1, pct=True)
    out = {}
    for k in range(1, n + 1):
        m = (q > (k - 1) / n) & (q <= k / n)
        out[f"L{k}"] = r.where(m).mean(axis=1)
    return pd.DataFrame(out)


layer_all = {}
for name in FACTORS:
    layer_all[name] = layer_returns(fac_wide[name], fwd1)
    layer_all[name].to_csv(OUT / f"layer_{name}.csv", encoding="utf-8-sig")

log("\n===== 分层年化收益（%，未扣费）=====")
for name in FACTORS:
    ann = (layer_all[name].mean() * 252 * 100).round(2)
    print(f"  {name:<9}", ann.to_dict())


# ==================== 5. IC 衰减 ====================
log("\n⑤ IC 衰减...")
LAGS = [1, 5, 10, 20]
decay = {}
for name in FACTORS:
    vals = {}
    for L in LAGS:
        # t+1 到 t+L 的累计收益
        fwdL = ret.rolling(L).sum().shift(-L)
        fwdL = fwdL.reindex(index=common_idx, columns=common_cols)
        ic = rank_ic(fac_wide[name], fwdL).dropna()
        vals[L] = ic.mean()
    decay[name] = vals

decay_df = pd.DataFrame(decay).T
decay_df.columns = [f"{L}d" for L in LAGS]
decay_df.to_csv(OUT / "ic_decay.csv", encoding="utf-8-sig")
log("   完成")


# ==================== 6. 多空组合 ====================
log("⑥ 多空组合（L5 - L1）...")
ls_stats = {}
for name in FACTORS:
    ls = (layer_all[name]["L5"] - layer_all[name]["L1"]).dropna()
    nav = (1 + ls).cumprod()
    ann = nav.iloc[-1] ** (252 / len(ls)) - 1
    vol = ls.std() * np.sqrt(252)
    dd = (nav / nav.cummax() - 1).min()
    ls_stats[name] = {
        "多空年化": ann,
        "多空波动": vol,
        "多空夏普": ann / vol if vol > 0 else np.nan,
        "多空最大回撤": dd,
    }

ls_df = pd.DataFrame(ls_stats).T
ls_df.to_csv(OUT / "long_short.csv", encoding="utf-8-sig")


# ==================== 7. 汇总 ====================
log("\n===== 中性化后 因子检验汇总 =====")
summary = ic_summary.join(ls_df)
print(summary.round(4).to_string())


# ==================== 8. 与中性化前对比 ====================
old_path = DATA_DIR / "output" / "ic_summary.csv"
if old_path.exists():
    log("\n===== 中性化前 vs 后 =====")
    try:
        old = pd.read_csv(old_path, index_col=0)
        cmp = pd.DataFrame({
            "IC前": old.loc[FACTORS, "IC均值"],
            "IC后": ic_summary["IC均值"],
            "ICIR前": old.loc[FACTORS, "ICIR"],
            "ICIR后": ic_summary["ICIR"],
        })
        cmp["IC变化"] = (cmp["IC后"] - cmp["IC前"]).round(4)
        cmp["ICIR变化"] = (cmp["ICIR后"] - cmp["ICIR前"]).round(4)
        print(cmp.round(4).to_string())
        cmp.to_csv(OUT / "compare_raw_vs_neutral.csv", encoding="utf-8-sig")
    except Exception as e:
        log(f"   跳过对比：{e}")


# ==================== 9. 画图 ====================
log("\n⑦ 画图...")

# 分层净值
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
for ax, name in zip(axes.flat, FACTORS):
    nav = (1 + layer_all[name]).cumprod()
    for col in nav.columns:
        ax.plot(nav.index, nav[col], label=col, linewidth=0.9)
    ax.set_title(f"{name} - 5 group cumulative (neutralized)", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    n_ticks = 6
    ax.set_xticks(nav.index[:: max(1, len(nav) // n_ticks)])
    ax.tick_params(axis="x", rotation=30, labelsize=7)
plt.tight_layout()
plt.savefig(OUT / "layers.png", dpi=120)
plt.close()

# IC 衰减
fig, ax = plt.subplots(figsize=(8, 5))
for name in FACTORS:
    ax.plot(decay_df.columns, decay_df.loc[name].values, marker="o", label=name)
ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax.set_title("IC Decay (neutralized)")
ax.set_ylabel("mean Rank IC")
ax.set_xlabel("forward horizon")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "ic_decay.png", dpi=120)
plt.close()

# IC 时序
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for ax, name in zip(axes.flat, FACTORS):
    ic = ic_series[name]
    ax.plot(ic.index, ic.rolling(60).mean(), color="steelblue")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title(f"{name} - 60d rolling mean IC", fontsize=9)
    ax.tick_params(axis="x", rotation=30, labelsize=6)
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "ic_series.png", dpi=120)
plt.close()

log(f"   ✅ 图片已保存到 {OUT}/")
log(f"\n总耗时 {(time.time() - t0) / 60:.1f} 分钟")
