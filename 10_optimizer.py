"""
10_optimizer.py
方法 2：组合层面约束优化
  min  -w's + gamma/2 * ||w - w_bench||^2
  s.t. sum(w)=1, 0<=w<=cap, X'w = X'w_bench
输出 data/output_optimizer/
"""
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from config import DATA_DIR

RAW = DATA_DIR / "raw"
REF = DATA_DIR / "ref"
PANEL = DATA_DIR / "panel"
OUT = DATA_DIR / "output_optimizer"
OUT.mkdir(parents=True, exist_ok=True)

FACTORS = ["rev_5", "turn_20", "vol_20", "ep"]
REBAL = 5
COST = 0.0012
CAP = 0.02          # 单票上限 2%
GAMMA = 1.0         # 向基准收缩的力度


def log(m):
    print(m, flush=True)


t0 = time.time()

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

b = []
for f in sorted(RAW.glob("daily_basic_*.parquet")):
    b.append(pd.read_parquet(f, columns=["ts_code", "trade_date", "circ_mv"]))
bb = pd.concat(b, ignore_index=True).drop_duplicates(["trade_date", "ts_code"])
logmv = np.log(bb.pivot(index="trade_date", columns="ts_code", values="circ_mv")
                .reindex(index=idx, columns=cols) * 1e4)

mask = (pd.read_parquet(PANEL / "tradable.parquet")
        .reindex(index=idx, columns=cols).fillna(False).astype(bool))

sb = pd.concat([pd.read_parquet(REF / "stock_basic.parquet"),
                pd.read_parquet(REF / "stock_delisted.parquet")],
               ignore_index=True).drop_duplicates("ts_code")
industry = sb.set_index("ts_code")["industry"].reindex(cols).fillna("UNKNOWN")

# 行业哑变量（去掉一列，避免与 sum(w)=1 共线）
D = pd.get_dummies(industry, drop_first=True).astype(float)
Dv = D.values                       # (n_cols, n_ind-1)
log(f"   {len(idx)} 天 × {len(cols)} 只   行业 {Dv.shape[1] + 1}")

score = sum(wide[n].fillna(0) for n in FACTORS) / len(FACTORS)
bench = fwd.where(mask).mean(axis=1)


# ==================== 单期求解 ====================
def solve_one(d):
    valid = mask.loc[d].values
    s = score.loc[d].values
    ok = valid & np.isfinite(s)
    if ok.sum() < 200:
        return None, np.nan

    n = int(ok.sum())
    s_ = s[ok]

    def z(v):
        v = v[ok]
        v = np.where(np.isfinite(v), v, np.nan)
        m = np.nanmean(v)
        sd = np.nanstd(v)
        if not np.isfinite(sd) or sd == 0:
            return np.zeros(n)
        return np.where(np.isfinite(v), (v - m) / sd, 0.0)

    X = np.column_stack([
        z(logmv.loc[d].values),
        z(wide["ep"].loc[d].values),
        z(wide["vol_20"].loc[d].values),
        Dv[ok],
    ])                                   # (n, k)

    wb = np.full(n, 1.0 / n)
    A = X.T
    bvec = A @ wb

    def obj(w):
        diff = w - wb
        return -w @ s_ + 0.5 * GAMMA * (diff @ diff)

    def jac(w):
        return -s_ + GAMMA * (w - wb)

    cons = [{"type": "eq",
             "fun": lambda w: A @ w - bvec,
             "jac": lambda w: A}]

    res = minimize(obj, wb, jac=jac, method="SLSQP",
                   bounds=[(0.0, CAP)] * n,
                   constraints=cons,
                   options={"maxiter": 300, "ftol": 1e-12})

    if not res.success:
        return None, np.nan

    w = np.clip(res.x, 0, CAP)
    tot = w.sum()
    if tot <= 0:
        return None, np.nan
    w = w / tot

    # 组合的实际暴露偏离
    expo = A @ w - bvec
    dev = float(np.abs(expo).max())

    codes = np.array(cols)[ok]
    return pd.Series(w, index=codes), dev


# ==================== 逐期回测 ====================
log("② 逐期求解（可能要 5–15 分钟）...")
w_prev = pd.Series(0.0, index=cols)
rows = []

for i, d in enumerate(idx):
    if i % REBAL != 0:
        continue

    w_new, dev = solve_one(d)

    if w_new is None:
        rows.append({"date": d, "gross": np.nan, "cost": np.nan,
                     "net": np.nan, "turnover": np.nan, "expo_dev": np.nan})
        continue

    w_new = w_new.reindex(cols).fillna(0.0)
    to = (w_new - w_prev).abs().sum() / 2

    # 持有 5 天（每天按当期收益）
    dates_hold = []
    for k in range(REBAL):
        j = i + k
        if j >= len(idx) - 1:
            break
        d_h = idx[j]
        r = fwd.loc[d_h].reindex(cols).fillna(0.0)
        dates_hold.append(float((w_new * r).sum()))

    for k, g in enumerate(dates_hold):
        c = to * COST if k == 0 else 0.0
        rows.append({"date": idx[i + k], "gross": g, "cost": c,
                     "net": g - c, "turnover": to if k == 0 else 0.0,
                     "expo_dev": dev if k == 0 else np.nan})

    w_prev = w_new
    if (i // REBAL) % 20 == 0:
        log(f"   {i}/{len(idx)}  已用 {time.time() - t0:.0f}s")

res = pd.DataFrame(rows).set_index("date").sort_index()
res = res[~res.index.duplicated(keep="first")]
res = res.dropna(subset=["net"])

# ==================== 指标 ====================
nav = (1 + res["net"]).cumprod()
ann = nav.iloc[-1] ** (252 / len(res)) - 1
vol = res["net"].std() * np.sqrt(252)
dd = (nav / nav.cummax() - 1).min()

bench_al = bench.reindex(res.index).fillna(0)
ex_nav = nav / (1 + bench_al).cumprod()
ex_ret = ex_nav.pct_change().dropna()
ex_ann = ex_nav.iloc[-1] ** (252 / len(ex_nav)) - 1
ex_vol = ex_ret.std() * np.sqrt(252)

log("\n===== 优化器结果（扣费后）=====")
log(f"年化          {ann:.2%}")
log(f"波动          {vol:.2%}")
log(f"夏普          {ann / vol:.2f}")
log(f"最大回撤      {dd:.2%}")
log(f"年化换手      {res['turnover'].sum() * 252 / REBAL:.2f} 倍")
log(f"年化成本      {res['cost'].sum() * 252 / REBAL:.2%}")
log(f"超额年化      {ex_ann:.2%}")
log(f"信息比率      {ex_ann / ex_vol:.2f}")
log(f"暴露最大偏离  {res['expo_dev'].max():.4f}")

pd.DataFrame({
    "年化": [ann], "波动": [vol], "夏普": [ann / vol],
    "最大回撤": [dd], "年化换手": [res["turnover"].sum() * 252 / REBAL],
    "年化成本": [res["cost"].sum() * 252 / REBAL],
    "超额年化": [ex_ann], "信息比率": [ex_ann / ex_vol],
    "暴露最大偏离": [res["expo_dev"].max()],
}).to_csv(OUT / "optimizer_summary.csv", index=False, encoding="utf-8-sig")

res.to_csv(OUT / "optimizer_daily.csv", encoding="utf-8-sig")
log(f"\n✅ 保存到 {OUT}/   总耗时 {(time.time() - t0) / 60:.1f} 分钟")
