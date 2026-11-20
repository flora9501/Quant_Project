"""
阶段零自检：验证数据源、字段名、单位。
运行：
    pip install -r requirements.txt
    cp .env.example .env        # 填入 TUSHARE_TOKEN
    python step0_check.py
"""
import sys
import pandas as pd
import tushare as ts

from config import TUSHARE_TOKEN, CHECK_DATE, INDEX_CODE, CHECK_DIR

if not TUSHARE_TOKEN:
    sys.exit("❌ 没找到 TUSHARE_TOKEN，请在 .env 里填写")

pro = ts.pro_api(TUSHARE_TOKEN)


def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def show(df, name, n=5):
    print(f"\n[{name}] shape={df.shape}")
    print("columns:", list(df.columns))
    print(df.head(n).to_string(index=False))
    na = df.isna().mean().round(4)
    bad = na[na > 0]
    print("缺失率 > 0 的列：")
    print(bad.to_string() if len(bad) else "  无")


# ---------- 1. 交易日 ----------
hr("1. 交易日历")
cal = pro.trade_cal(exchange="SSE", start_date="20231220", end_date="20240115")
open_days = cal.loc[cal["is_open"] == 1, "cal_date"].tolist()
assert CHECK_DATE in open_days, f"❌ {CHECK_DATE} 不是交易日"
print(f"✅ {CHECK_DATE} 是交易日")


# ---------- 2. 全 A 行情 ----------
hr("2. daily（全 A 日行情）")
daily = pro.daily(trade_date=CHECK_DATE)
show(daily, "daily")

print("\n--- 单位核对 ---")
print(f"  close  中位数 = {daily['close'].median():.2f}   → 元")
print(f"  vol    中位数 = {daily['vol'].median():.0f}     → 手（1手=100股）")
print(f"  amount 中位数 = {daily['amount'].median():.0f}   → 千元")
assert 1 < daily["close"].median() < 200, "❌ close 量级异常"


# ---------- 3. daily_basic 与 total_mv 单位 ----------
hr("3. daily_basic（重点：total_mv 单位）")
fields = ("ts_code,trade_date,close,turnover_rate,volume_ratio,pe,pe_ttm,pb,"
          "ps_ttm,dv_ratio,total_share,float_share,free_share,total_mv,circ_mv")
basic = pro.daily_basic(trade_date=CHECK_DATE, fields=fields)
show(basic, "daily_basic")

med = basic["total_mv"].median()
print(f"\ntotal_mv 原始中位数 = {med:,.0f}")

if med > 1e8:
    print("→ 单位是【元】")
    factor = 1.0
elif 1e5 < med < 1e7:
    print("→ 单位是【万元】，需 × 10000")
    factor = 1e4
else:
    raise ValueError(f"❌ total_mv 量级异常：{med}")

basic["total_mv_yuan"] = basic["total_mv"] * factor
basic["circ_mv_yuan"] = basic["circ_mv"] * factor

print(f"转换后 total_mv 中位数 = {basic['total_mv_yuan'].median():,.0f} 元 "
      f"≈ {basic['total_mv_yuan'].median()/1e8:.1f} 亿")
assert 5e8 < basic["total_mv_yuan"].median() < 5e10, "❌ total_mv 单位仍不对"


# ---------- 4. 中证 1000 成分 ----------
hr("4. index_weight（中证 1000）")
w = pro.index_weight(index_code=INDEX_CODE,
                     start_date="20240101", end_date="20240131")
show(w, "index_weight")

cnt = w.groupby("trade_date")["con_code"].nunique()
print("\n每日成分数量：")
print(cnt.to_string())
print("\n每日权重之和：")
print(w.groupby("trade_date")["weight"].sum().round(2).to_string())

if cnt.max() < 500:
    print("\n⚠️ 成分数偏少，可能只返回前 N 大权重，改用 index_member")
else:
    print("\n✅ 成分数正常（中证 1000 应约 1000 只）")


# ---------- 5. 覆盖度交叉检查 ----------
hr("5. 覆盖度交叉检查")
print(f"daily 股票数       = {daily['ts_code'].nunique()}")
print(f"daily_basic 股票数 = {basic['ts_code'].nunique()}")
merged = daily.merge(basic[["ts_code", "total_mv_yuan"]], on="ts_code", how="inner")
print(f"两者交集           = {len(merged)}")
print(f"交集内 total_mv 缺失 = {merged['total_mv_yuan'].isna().sum()}")
assert len(merged) > 4000, "❌ 交集过小，检查是否同一交易日"


# ---------- 6. 保存 ----------
hr("6. 保存样例")
daily.to_parquet(CHECK_DIR / f"daily_{CHECK_DATE}.parquet", index=False)
basic.to_parquet(CHECK_DIR / f"daily_basic_{CHECK_DATE}.parquet", index=False)
w.to_parquet(CHECK_DIR / f"index_weight_{INDEX_CODE}_202401.parquet", index=False)
print(f"✅ 已保存到 {CHECK_DIR}/")


# ---------- 7. 通过标准 ----------
hr("阶段零通过标准")
print("""
[ ] 1. CHECK_DATE 是交易日
[ ] 2. daily 的 close/vol/amount 量级正常
[ ] 3. total_mv 已确认单位并转成元
[ ] 4. index_weight 有 con_code + weight，成分数 > 500
[ ] 5. daily 与 daily_basic 能按 ts_code merge
[ ] 6. 三个 parquet 已落盘
""")
