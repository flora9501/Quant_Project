# tests/test_neutralize_equiv.py
"""验证 FWL 快速中性化与 lstsq 版本等价。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from factorlab.core import load_panel
from factorlab.factors import build_price_factors, load_price_data, mad_winsorize


def main():
    p = load_panel()
    daily, basic = load_price_data()
    raw = build_price_factors(p.idx[:60], p.cols, daily, basic)
    d = p.idx[50]

    y = raw["rev_5"].loc[d].where(p.mask.loc[d])
    lm = raw["ln_mv"].loc[d]
    ind = p.industry

    ok = y.notna() & lm.notna()
    y = mad_winsorize(y)

    # 旧：带行业哑变量 + lnmv 的 lstsq
    X = pd.get_dummies(ind[ok], drop_first=True).to_numpy(dtype=float)
    X = np.c_[np.ones(int(ok.sum())), lm[ok].to_numpy(dtype=float), X]
    b, *_ = np.linalg.lstsq(X, y[ok].to_numpy(dtype=float), rcond=None)
    r_old = y[ok].to_numpy(dtype=float) - X @ b

    # 新：FWL，组内去均值后一元回归
    dd = pd.DataFrame({"y": y[ok], "lm": lm[ok], "ind": ind[ok]})
    dd["y"] = dd["y"] - dd.groupby("ind")["y"].transform("mean")
    dd["lm"] = dd["lm"] - dd.groupby("ind")["lm"].transform("mean")

    beta = float((dd["y"] * dd["lm"]).sum()) / float((dd["lm"] ** 2).sum())
    r_new = (dd["y"] - beta * dd["lm"]).to_numpy()

    diff = float(np.abs(r_old - r_new).max())

    print(f"lstsq 残差 std : {r_old.std():.10f}")
    print(f"FWL   残差 std : {r_new.std():.10f}")
    print(f"最大绝对差     : {diff:.3e}")
    print(f"样本数         : {int(ok.sum())}")
    print()
    if diff < 1e-8:
        print("✅ 等价，可以替换 neutralize_one")
    elif diff < 1e-3:
        print("⚠️ 接近但不等价，检查 get_dummies 的基准组")
    else:
        print("❌ 不等价，别替换")
        print("   industry 空值:", int(p.industry.isna().sum()))
        print("   UNKNOWN 只数 :", int((p.industry == "UNKNOWN").sum()))


if __name__ == "__main__":
    main()
