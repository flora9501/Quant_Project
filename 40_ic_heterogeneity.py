#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
40_ic_heterogeneity.py
因子 IC 的条件异质性。三个切分维度：

  A. 市值 × 换手率（3×3 = 9 桶）—— 原版
  B. 上市年龄（固定阈值 3 档）—— 新增
  C. 上市年龄 × 市值（3×3 = 9 桶）—— 新增，用来回答
     "次新效应是不是只是小市值的换皮"

上市年龄用**自然日**（date − list_date），固定阈值：
  次新   < 365 天
  中生代 365–1095 天
  老股   > 1095 天
不用三分位，因为固定阈值更好解释；每桶日均只数会打印，<30 的格子不可信。

同时输出每桶的前瞻收益（bp/期）：
  IC 高 + 桶收益平 → 选股能力
  IC 高 + 桶收益也高 → 该桶是 beta

用法
  .venv/bin/python -u 40_ic_heterogeneity.py
输出
  data/output_zoo/ic_heterogeneity.csv
  data/output_zoo/ic_heterogeneity_age.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DATA_DIR
from factorlab.core import load_panel
from factorlab.factors import load_price_data
from factorlab.gate import forward_returns

OUT = Path("data/output_zoo")
OUT.mkdir(parents=True, exist_ok=True)
REF = DATA_DIR / "ref"

H = 5
MIN_N = 30
SIZE = ["小", "中", "大"]
LIQ = ["低换手", "中换手", "高换手"]
AGEL = ["次新<1y", "中生代1-3y", "老股>3y"]
AGE_EDGES = [-np.inf, 365, 1095, np.inf]


def log(m):
    print(m, flush=True)


def nw_t(series, lags):
    x = pd.Series(series).dropna().to_numpy(dtype=float)
    n = x.size
    if n < 20:
        return np.nan
    e = x - x.mean()
    g = float((e ** 2).sum() / n)
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        g += 2.0 * w * float((e[L:] * e[:-L]).sum() / n)
    return float(x.mean() / np.sqrt(g / n)) if g > 0 else np.nan


def bucket_stats(df, factor, bk_col, n_bk):
    """返回 (IC 均值, NW t, 桶收益 bp) 三个 Series，索引为桶编号"""
    sub = df[["date", bk_col, factor, "fwd5"]].dropna()
    if sub.empty:
        return None, None, None
    sub = sub.copy()
    sub["rx"] = sub.groupby(["date", bk_col])[factor].rank()
    sub["ry"] = sub.groupby(["date", bk_col])["fwd5"].rank()
    sub["pr"] = sub["rx"] * sub["ry"]

    gg = sub.groupby(["date", bk_col]).agg(
        mp=("pr", "mean"), mx=("rx", "mean"), my=("ry", "mean"),
        sx=("rx", "std"), sy=("ry", "std"), n=("rx", "size"))
    gg = gg[gg["n"] >= MIN_N]
    if gg.empty:
        return None, None, None

    ic = ((gg["mp"] - gg["mx"] * gg["my"])
          / (gg["sx"] * gg["sy"]).replace(0, np.nan)).unstack(bk_col)
    ic = ic.reindex(columns=range(n_bk))

    t = pd.Series({k: nw_t(ic[k], H - 1) for k in ic.columns})
    bm = (sub.groupby(["date", bk_col])["fwd5"].mean()
             .unstack(bk_col).reindex(columns=range(n_bk)))
    return ic.mean(), t, bm.mean() * 1e4


def print_block(title, ic_tab, t_tab, bm_tab, labels):
    log("\n" + "=" * 118)
    log(title)
    log("=" * 118)
    for name, tab, fmt in (("IC（描述，不是结论）", ic_tab, "{:.4f}"),
                           ("NW t（|t|>=2 才算有东西）", t_tab, "{:.2f}"),
                           ("桶收益 bp/期（区分能力 vs beta）", bm_tab, "{:.1f}")):
        if tab is None or tab.empty:
            continue
        t = tab.copy()
        t.columns = labels
        log(f"\n【{name}】")
        log(t.round(4).to_string())


def main():
    t0 = time.time()
    log("① 加载面板、分位数据与上市日...")
    p = load_panel()
    daily, basic = load_price_data()
    cmv = (basic.pivot(index="trade_date", columns="ts_code", values="circ_mv")
                .reindex(index=p.idx, columns=p.cols))
    turn = (basic.pivot(index="trade_date", columns="ts_code", values="turnover_rate")
                 .reindex(index=p.idx, columns=p.cols))
    lnmv = np.log(cmv * 1e4)
    fwd5 = forward_returns(p, H)

    sb = pd.concat([pd.read_parquet(REF / "stock_basic.parquet"),
                    pd.read_parquet(REF / "stock_delisted.parquet")],
                   ignore_index=True).drop_duplicates("ts_code")
    ld = sb.set_index("ts_code")["list_date"].astype(str)
    ld_ord = (pd.to_datetime(ld, format="%Y%m%d", errors="coerce")
                .map(pd.Timestamp.toordinal))
    d_ord = (pd.to_datetime(pd.Index(p.idx), format="%Y%m%d")
               .map(pd.Timestamp.toordinal).to_numpy())
    age = pd.DataFrame(
        d_ord[:, None] - ld_ord.reindex(p.cols).to_numpy(dtype=float)[None, :],
        index=p.idx, columns=p.cols)
    log(f"   {len(p.idx)} 天 × {len(p.cols)} 只；"
        f"list_date 缺失 {int(ld_ord.reindex(p.cols).isna().sum())} 只")

    log("\n② 摊平成长表（只保留在册样本）...")
    def flat(wide, name):
        return wide.where(p.mask).stack().rename(name)

    parts = {str(n): flat(p.fac[n], str(n)) for n in p.fac}
    parts["fwd5"] = flat(fwd5, "fwd5")
    parts["lnmv"] = flat(lnmv, "lnmv")
    parts["turn"] = flat(turn, "turn")
    parts["age"] = flat(age, "age")

    df = pd.DataFrame(parts).reset_index().rename(
        columns={"trade_date": "date"})
    df = df.dropna(subset=["lnmv", "turn", "fwd5"])
    log(f"   {len(df):,} 行")

    log("\n③ 分桶...")
    # A. 市值 × 换手
    df["sz"] = (df.groupby("date")["lnmv"]
                  .transform(lambda s: pd.qcut(s, 3, labels=False,
                                               duplicates="drop")))
    df["lq"] = (df.groupby("date")["turn"]
                  .transform(lambda s: pd.qcut(s, 3, labels=False,
                                               duplicates="drop")))
    df["bk_sl"] = df["sz"] * 3 + df["lq"]

    # B/C. 上市年龄（固定阈值）
    df["ag"] = pd.cut(df["age"], bins=AGE_EDGES, labels=False,
                      right=False)
    df["bk_age"] = df["ag"]
    df["bk_as"] = df["ag"] * 3 + df["sz"]          # 年龄 × 市值

    n_dates = df["date"].nunique()
    for col, labels, name in (("bk_sl", [f"{s}-{q}" for s in SIZE for q in LIQ],
                              "市值 × 换手"),
                              ("bk_age", AGEL, "上市年龄"),
                              ("bk_as", [f"{a}·{s}" for a in AGEL for s in SIZE],
                              "年龄 × 市值")):
        cnt = df.groupby(col).size().div(n_dates).reindex(
            range(len(labels)))
        thin = [labels[i] for i, v in cnt.items()
                if pd.isna(v) or v < MIN_N]
        log(f"   {name}：日均只数 "
            + ", ".join(f"{labels[i]}={0 if pd.isna(v) else int(v)}"
                        for i, v in cnt.items())
            + (f"  ⚠️ 过窄 {thin}" if thin else ""))

    log("\n④ 逐桶算 IC（约 3–5 分钟）...")
    res = {}
    for factor in p.fac:
        factor = str(factor)
        for key, bk_col, n_bk in (("sl", "bk_sl", 9),
                                  ("age", "bk_age", 3),
                                  ("as", "bk_as", 9)):
            ic, t, bm = bucket_stats(df, factor, bk_col, n_bk)
            res[(factor, key)] = (ic, t, bm)
        log(f"   {factor} 完成")

    # ---------- 输出 ----------
    def matrix(key, idx):
        out = {}
        for factor in p.fac:
            factor = str(factor)
            v = res.get((factor, key), (None, None, None))[idx]
            if v is not None:
                out[factor] = v
        return pd.DataFrame(out).T if out else pd.DataFrame()

    print_block("【A】市值 × 换手（9 桶）",
                matrix("sl", 0), matrix("sl", 1), matrix("sl", 2),
                [f"{s}-{q}" for s in SIZE for q in LIQ])

    print_block("【B】上市年龄（3 档）",
                matrix("age", 0), matrix("age", 1), matrix("age", 2), AGEL)

    log("\n" + "=" * 118)
    log("【C】年龄 × 市值：NW t 矩阵（9 桶）")
    log("=" * 118)
    tas = matrix("as", 1)
    if not tas.empty:
        tas.columns = [f"{a}·{s}" for a in AGEL for s in SIZE]
        log(tas.round(2).to_string())

    # ---------- 判读 ----------
    log("\n" + "=" * 118)
    log("【判读】")
    log("  B 表：若「次新<1y」与「老股>3y」的 IC 差异 < 0.01 且 t 同号 → 次新不是独立维度")
    log("  C 表：若次新的 IC 优势在「小市值」列里消失 → 次新只是小市值的换皮")
    log(f"  36 组以上 = 多重检验；每桶日均 <{MIN_N} 只的格子不可信")

    pd.DataFrame([{**{"因子": f, "维度": k, "桶": int(b)},
                   "IC": round(float(v), 4)}
                  for (f, k), (ic, _, _) in res.items() if ic is not None
                  for b, v in ic.items()]).to_csv(
        OUT / "ic_heterogeneity.csv", index=False, encoding="utf-8-sig")

    log(f"\n✅ 输出 {OUT / 'ic_heterogeneity.csv'}")
    log(f"总耗时 {(time.time() - t0) / 60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    sys.exit(main())
