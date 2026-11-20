"""
第 5 天：因子预处理 —— MAD 去极值 → 行业+市值中性化 → z-score
输出 data/panel/factors_neutral.parquet
"""
import time

import numpy as np
import pandas as pd

from config import DATA_DIR

RAW = DATA_DIR / "raw"
REF = DATA_DIR / "ref"
PANEL = DATA_DIR / "panel"

FACTORS = ["rev_5", "turn_20", "vol_20", "ep"]     # mom_60 已剔除


def log(m):
    print(m, flush=True)


t0 = time.time()

# ==================== 1. 因子宽表 ====================
log("① 读取因子...")
fac = pd.read_parquet(PANEL / "factors.parquet")
wide = {n: fac.pivot(index="trade_date", columns="ts_code", values=n)
        for n in FACTORS}
dates = wide["rev_5"].index
codes = wide["rev_5"].columns
log(f"   {len(dates)} 天 × {len(codes)} 只")

# ==================== 2. 对数市值 ====================
log("② 读取市值...")
frames = []
for f in sorted(RAW.glob("daily_basic_*.parquet")):
    frames.append(pd.read_parquet(f, columns=["ts_code", "trade_date", "circ_mv"]))
b = (pd.concat(frames, ignore_index=True)
       .drop_duplicates(["trade_date", "ts_code"]))
cmv = (b.pivot(index="trade_date", columns="ts_code", values="circ_mv")
        .reindex(index=dates, columns=codes))
logmv = np.log(cmv * 1e4)           # 万元 → 元
log(f"   完成 {time.time() - t0:.0f}s")

# ==================== 3. 行业哑变量 ====================
log("③ 构建行业哑变量...")
sb = pd.concat([pd.read_parquet(REF / "stock_basic.parquet"),
                pd.read_parquet(REF / "stock_delisted.parquet")],
               ignore_index=True).drop_duplicates("ts_code")
ind = (sb.set_index("ts_code")["industry"]
         .reindex(codes).fillna("UNKNOWN"))
D = pd.get_dummies(ind, drop_first=True).astype(float)
Dv = D.values
log(f"   行业数 {Dv.shape[1] + 1}")

# ==================== 4. 逐日中性化 ====================
log("④ 逐日处理（去极值 → 回归取残差 → z-score）...")
Y = {n: wide[n].values for n in FACTORS}
M = logmv.values
resid = {n: np.full((len(dates), len(codes)), np.nan) for n in FACTORS}

for i in range(len(dates)):
    lm = M[i]
    for n in FACTORS:
        y = Y[n][i]
        ok = ~np.isnan(y) & ~np.isnan(lm)
        if ok.sum() < Dv.shape[1] + 20:
            continue

        yy = y[ok]

        # MAD 去极值
        med = np.median(yy)
        mad = np.median(np.abs(yy - med))
        if mad > 0:
            k = 1.4826
            yy = np.clip(yy, med - 3 * k * mad, med + 3 * k * mad)

        # 对 [截距 + 行业哑变量 + log市值] 回归
        X = np.column_stack([np.ones(ok.sum()), Dv[ok], lm[ok]])
        beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
        r = yy - X @ beta

        # z-score
        s = r.std()
        if s > 0:
            r = (r - r.mean()) / s

        resid[n][i, ok] = r

    if (i + 1) % 200 == 0:
        log(f"   {i + 1}/{len(dates)}  完成 {time.time() - t0:.0f}s")

log(f"   完成 {time.time() - t0:.0f}s")

# ==================== 5. 输出 ====================
log("⑤ 保存...")
long = pd.DataFrame({
    "trade_date": np.repeat(dates.values, len(codes)),
    "ts_code": np.tile(codes.values, len(dates)),
})
for n in FACTORS:
    long[n] = resid[n].ravel()

long = long.dropna(subset=FACTORS, how="all")
long.to_parquet(PANEL / "factors_neutral.parquet", index=False)
log(f"   ✅ {len(long):,} 行 → data/panel/factors_neutral.parquet")

# ==================== 6. 诊断 ====================
log("\n===== 中性化后覆盖率 =====")
print(long[FACTORS].notna().mean().round(3).to_string())

log("\n===== 中性化后分布 =====")
print(long[FACTORS].describe().T[["mean", "std", "min", "50%", "max"]].round(4).to_string())

log("\n===== 因子间相关性（中性化后）=====")
print(long[FACTORS].corr().round(3).to_string())

log(f"\n总耗时 {(time.time() - t0) / 60:.1f} 分钟")
