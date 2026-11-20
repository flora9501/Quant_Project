# 21_probe_industry.py
import pandas as pd
from config import DATA_DIR

PANEL, REF = DATA_DIR / "panel", DATA_DIR / "ref"

cols = pd.read_parquet(PANEL / "tradable.parquet").columns
sb = pd.read_parquet(REF / "stock_basic.parquet")
sd = pd.read_parquet(REF / "stock_delisted.parquet")

print("股票池代码数 :", len(cols))
print("stock_basic  :", len(sb), "| 列:", sb.columns.tolist())
print("stock_delisted:", len(sd), "| 列:", sd.columns.tolist())

in_sb = set(sb.ts_code)
in_sd = set(sd.ts_code)
unknown = [c for c in cols if c not in in_sb and c not in in_sd]
print("\n完全查不到行业 :", len(unknown))
print("示例:", unknown[:10])

# 有记录但 industry 为空
null_ind = sb[sb.industry.isna()].ts_code.tolist()
print("stock_basic 里 industry 为空 :", len(null_ind))

# 在不在册
mask = pd.read_parquet(PANEL / "tradable.parquet")
unk_cols = [c for c in (unknown + null_ind) if c in mask.columns]
if unk_cols:
    print("\nUNKNOWN 日均占在册比例 :",
          round(mask[unk_cols].sum(axis=1).mean() / mask.sum(axis=1).mean(), 3))
