"""
第 3 天：构建六个因子
输出 data/panel/factors.parquet（长表：trade_date, ts_code, 6 个因子）
"""
import time

import numpy as np
import pandas as pd

from config import DATA_DIR

RAW = DATA_DIR / "raw"
PANEL = DATA_DIR / "panel"
PANEL.mkdir(parents=True, exist_ok=True)


def log(m):
    print(m, flush=True)


t0 = time.time()

# ==================== 1. 读取 ====================
log("① 读取行情...")
frames = []
for f in sorted(RAW.glob("daily_[0-9][0-9][0-9][0-9].parquet")):
    frames.append(pd.read_parquet(
        f, columns=["ts_code", "trade_date", "close", "pct_chg"]))
daily = pd.concat(frames, ignore_index=True).drop_duplicates(["trade_date", "ts_code"])

log("② 读取每日指标...")
bframes = []
for f in sorted(RAW.glob("daily_basic_*.parquet")):
    bframes.append(pd.read_parquet(
        f, columns=["ts_code", "trade_date", "turnover_rate", "circ_mv", "pe_ttm"]))
basic = pd.concat(bframes, ignore_index=True).drop_duplicates(["trade_date", "ts_code"])

all_dates = np.sort(daily["trade_date"].unique())
all_codes = np.sort(daily["ts_code"].unique())
log(f"   交易日 {len(all_dates)}  股票 {len(all_codes)}")


def to_matrix(df, col):
    """长表 → 宽表（date × code），并对齐到全量网格"""
    return (df.pivot(index="trade_date", columns="ts_code", values=col)
              .reindex(index=all_dates, columns=all_codes))


# ==================== 2. 收益矩阵 ====================
log("③ 构建收益矩阵...")
# 直接用 pct_chg（Tushare 已按前收盘调整，除权日也是对的）
ret = to_matrix(daily, "pct_chg") / 100.0
ret = ret.fillna(0.0)          # 停牌日按 0 收益处理（持有不动）
log(f"   完成 {time.time() - t0:.0f}s")

# ==================== 3. 因子 ====================
log("④ 计算六个因子...")

turn = to_matrix(basic, "turnover_rate")           # 单位 %
cmv = to_matrix(basic, "circ_mv")                  # 单位 万元
pe = to_matrix(basic, "pe_ttm")

# 反转：过去5日累计收益，取负
f_rev5 = -ret.rolling(5, min_periods=5).sum()

# 换手：20日均换手，取负
f_turn20 = -turn.rolling(20, min_periods=20).mean()

# 波动：20日收益标准差（年化），取负
f_vol20 = -ret.rolling(20, min_periods=20).std() * np.sqrt(252)

# 动量：跳过最近5日的60日累计收益
f_mom60 = ret.shift(5).rolling(55, min_periods=55).sum()

# 规模：负对数流通市值（万元→元）
f_lnmv = -np.log(cmv * 1e4)

# 价值：EP = 1/PE，PE<=0 记 NaN
f_ep = 1.0 / pe.where(pe > 0)

factors = {
    "rev_5":  f_rev5,
    "turn_20": f_turn20,
    "vol_20": f_vol20,
    "mom_60": f_mom60,
    "ln_mv":  f_lnmv,
    "ep":     f_ep,
}
log(f"   完成 {time.time() - t0:.0f}s")

# ==================== 4. 套上可交易掩码 ====================
log("⑤ 应用股票池掩码...")
# ==================== 5. 应用股票池掩码 ====================
log("⑤ 应用股票池掩码...")

# 读取掩码并对齐到全量网格
mask = pd.read_parquet(PANEL / "tradable.parquet")
mask = mask.reindex(index=all_dates, columns=all_codes).fillna(False).astype(bool)

# 拿到可交易的 (date, code) 索引
mask_stack = mask.stack()
keep = mask_stack[mask_stack].index
log(f"   可交易格子数 {len(keep):,}")

# 按格子取值，缺失保留 NaN
long = pd.DataFrame(index=keep)
for name, mat in factors.items():
    long[name] = mat.stack().reindex(keep)

long.index.names = ["trade_date", "ts_code"]
long = long.reset_index()

long.to_parquet(PANEL / "factors.parquet", index=False)
log(f"   ✅ 已保存 {len(long):,} 行 → data/panel/factors.parquet")


# ==================== 5. 诊断 ====================
log("\n===== 因子覆盖率（非空比例）=====")
print((long[[c for c in long.columns if c not in ("trade_date", "ts_code")]]
       .notna().mean().round(3)).to_string())

log("\n===== 因子分布 =====")
cols = ["rev_5", "turn_20", "vol_20", "mom_60", "ln_mv", "ep"]
print(long[cols].describe().T[["mean", "std", "min", "50%", "max"]].round(4).to_string())

log("\n===== 每日截面样本数 =====")
n_per_day = long.groupby("trade_date").size()
print(n_per_day.describe().round(0).to_string())

log(f"\n总耗时 {(time.time() - t0) / 60:.1f} 分钟")
