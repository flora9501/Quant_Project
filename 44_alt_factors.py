#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
44_alt_factors.py
交易结构因子（预注册方向，跑完不许改）

  margin_chg  融资余额 20 日变化率，取负        +1
  big_order   大单(大+特大)净流入占买入比，20日均  +1
  tiger_inst  机构专用席位净买入 / 成交额，20日累计 +1
  block_disc  大宗折价率（20日按金额加权），取负   +1

数据缺失处理
  余额类（margin）缺失 = 非两融标的 → NaN
  事件类（龙虎榜/大宗）缺失 = 当日无事件 → 0，但会同时报"非零占比"

用法
  .venv/bin/python -u 44_alt_factors.py [--rebuild]
"""
import argparse
import sys
from datetime import date as _date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DATA_DIR
from factorlab.core import load_panel
from factorlab.factors import load_price_data, neutralize, save_cached, load_cached
from factorlab.gate import evaluate

ALT = DATA_DIR / "raw" / "alt"
OUT = Path("data/output_zoo")
OUT.mkdir(parents=True, exist_ok=True)
TRIALS = OUT / "factor_trials.csv"

EXPECT = {"margin_chg": +1, "big_order": +1, "tiger_inst": +1, "block_disc": +1}
STOCK_ONLY = ("000", "001", "002", "003", "300", "301", "600", "601", "603", "605", "688")


def log(m):
    print(m, flush=True)


def load_alt(api):
    files = sorted(ALT.glob(f"{api}_*.parquet"))
    if not files:
        return None
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["trade_date"] = df["trade_date"].astype(str)
    return df


def is_stock(s):
    return s.str[:3].isin(STOCK_ONLY) & s.str.endswith((".SH", ".SZ"))


def to_wide(df, col, agg="sum", dates=None, codes=None):
    w = df.pivot_table(index="trade_date", columns="ts_code",
                       values=col, aggfunc=agg)
    return w.reindex(index=dates, columns=codes)


# ---------- 因子 ----------
def build_margin_chg(panel, amt):
    df = load_alt("margin_detail")
    if df is None:
        log("   ⚠️ 缺 margin_detail"); return None
    df = df[is_stock(df["ts_code"])]
    w = to_wide(df, "rzye", "last", panel.idx, panel.cols)
    chg = w / w.shift(20) - 1
    return -chg                       # 融资余额增长 → 未来差


def build_big_order(panel, amt):
    df = load_alt("moneyflow")
    if df is None:
        log("   ⚠️ 缺 moneyflow"); return None
    df = df[is_stock(df["ts_code"])].copy()
    for c in ("buy_lg_amount", "buy_elg_amount", "sell_lg_amount",
              "sell_elg_amount", "buy_sm_amount", "buy_md_amount"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    net = (df.buy_lg_amount + df.buy_elg_amount
           - df.sell_lg_amount - df.sell_elg_amount)
    tot = (df.buy_sm_amount + df.buy_md_amount
           + df.buy_lg_amount + df.buy_elg_amount)
    df["ratio"] = net / tot.where(tot > 0)
    w = to_wide(df, "ratio", "mean", panel.idx, panel.cols)
    return w.rolling(20, min_periods=15).mean()


def build_tiger_inst(panel, amt):
    df = load_alt("top_inst")
    if df is None:
        log("   ⚠️ 缺 top_inst"); return None
    df = df[is_stock(df["ts_code"])].copy()
    df["net_buy"] = pd.to_numeric(df["net_buy"], errors="coerce")
    inst = df[df["exalter"].astype(str).str.contains("机构专用", na=False)]
    g = inst.groupby(["trade_date", "ts_code"])["net_buy"].sum().rename("v").reset_index()

    w = to_wide(g, "v", "sum", panel.idx, panel.cols).fillna(0.0)
    # 事件占比：用于报告
    nz = float((w > 0).values.mean())
    log(f"   tiger_inst 非零占比 {nz:.2%}（每日约 {w.gt(0).sum(axis=1).mean():.0f} 只）")
    ratio = w / (amt * 1000.0)        # amt 是千元 → 元
    return ratio.rolling(20, min_periods=1).sum()


def build_block_disc(panel, amt, close):
    df = load_alt("block_trade")
    if df is None:
        log("   ⚠️ 缺 block_trade"); return None
    df = df[is_stock(df["ts_code"])].copy()
    for c in ("price", "vol", "amount"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    cl = close.stack().rename("close")
    df = df.join(cl, on=["trade_date", "ts_code"])
    df["disc"] = df["price"] / df["close"] - 1
    df["wd"] = df["disc"] * df["amount"]

    g = df.groupby(["trade_date", "ts_code"]).agg(wd=("wd", "sum"),
                                                  amt=("amount", "sum"))
    g["disc"] = g["wd"] / g["amt"].where(g["amt"] > 0)
    w = g["disc"].reset_index().pipe(
        lambda x: to_wide(x, "disc", "mean", panel.idx, panel.cols))
    nz = float(w.notna().values.mean())
    log(f"   block_disc 覆盖 {nz:.2%}")
    return -w.rolling(20, min_periods=1).mean()


# ---------- 主流程 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()

    log("① 加载面板与行情...")
    panel = load_panel()
    daily, basic = load_price_data(daily_cols=("pct_chg", "amount", "close"))
    amt = (daily.pivot(index="trade_date", columns="ts_code", values="amount")
                .reindex(index=panel.idx, columns=panel.cols))
    close = (daily.pivot(index="trade_date", columns="ts_code", values="close")
                  .reindex(index=panel.idx, columns=panel.cols))
    cmv = (basic.pivot(index="trade_date", columns="ts_code", values="circ_mv")
                .reindex(index=panel.idx, columns=panel.cols))
    log(f"   {len(panel.idx)} 天 × {len(panel.cols)} 只")

    log("\n② 构造原始因子...")
    raw = {}
    for name, fn in (("margin_chg", build_margin_chg),
                     ("big_order", build_big_order),
                     ("tiger_inst", build_tiger_inst),
                     ("block_disc", lambda p, a_: build_block_disc(p, a_, close))):
        log(f"   {name} ...")
        v = fn(panel, amt) if name != "block_disc" else fn(panel, amt)
        if v is None:
            continue
        raw[name] = v
        inreg = float(v.where(panel.mask).notna().to_numpy().mean())
        log(f"   {name:<12} 全网格 {v.notna().values.mean():.1%}  "
            f"在册 {inreg / panel.mask.to_numpy().mean():.1%}")

    if not raw:
        log("❌ 没有可用因子；先跑 43_pull_alt.py")
        return 1
    raw["ln_mv"] = -np.log(cmv * 1e4)

    log("\n③ 中性化 + gate...")
    rows = []
    for name in raw:
        if name == "ln_mv":
            continue
        f = None if a.rebuild else load_cached(name)
        if f is not None:
            log(f"   {name:<12} 缓存命中")
        else:
            f = neutralize(raw, panel.mask, panel.industry,
                           names=[name], verbose=False)[name]
            save_cached(name, f)
            log(f"   {name:<12} 已中性化并落盘")

        r = evaluate(name, f, panel, h=5, with_marginal=True,
                     verbose=False, require_positive=False)
        same_dir = np.sign(r.ic["IC"]) == np.sign(EXPECT[name])
        incr = (np.isfinite(r.marginal_t) and r.marginal_t >= 2.0
                and abs(r.resid_ic) >= 0.015)
        verdict = ("PASS" if (r.max_corr < 0.30 and abs(r.ic["IC"]) >= 0.02
                              and abs(r.ic["ICIR"]) >= 0.30 and same_dir
                              and incr) else "REJECT")

        ls = f"{r.ls_bp:+.0f}bp" if np.isfinite(r.ls_bp) else "n/a"
        log(f"   {name:<12} IC {r.ic['IC']:+.4f}  ICIR {r.ic['ICIR']:+.3f}  "
            f"NW t {r.t_nw:+.2f}  Q5-Q1 {ls}  相关 {r.max_corr:.3f}  "
            f"残差 {r.resid_ic:+.4f}  边际 t {r.marginal_t:+.2f}  → {verdict}")

        rows.append({"因子": name, "IC": round(r.ic["IC"], 4),
                     "ICIR": round(r.ic["ICIR"], 3), "t_NW": round(r.t_nw, 2),
                     "Q5-Q1(bp)": round(r.ls_bp, 0) if np.isfinite(r.ls_bp) else None,
                     "最大相关": round(r.max_corr, 3),
                     "残差IC": round(r.resid_ic, 4),
                     "方向一致": bool(same_dir),
                     "边际t": round(r.marginal_t, 2) if np.isfinite(r.marginal_t) else None,
                     "判决": verdict})

        new = pd.DataFrame([{
            "date": str(_date.today()), "factor": name,
            "IC": round(r.ic["IC"], 4), "ICIR": round(r.ic["ICIR"], 3),
            "t_NW": round(r.t_nw, 2),
            "p_NW": float(f"{r.p_nw:.3g}") if np.isfinite(r.p_nw) else np.nan,
            "resid_ic": round(r.resid_ic, 4), "max_corr": round(r.max_corr, 3),
            "marginal_t": round(r.marginal_t, 2) if np.isfinite(r.marginal_t) else None,
            "marginal_delta": round(r.marginal_delta, 4) if np.isfinite(r.marginal_delta) else None,
            "verdict": verdict}])
        old = pd.read_csv(TRIALS) if TRIALS.exists() else pd.DataFrame()
        pd.concat([old, new], ignore_index=True).to_csv(
            TRIALS, index=False, encoding="utf-8-sig")

    tab = pd.DataFrame(rows).set_index("因子")
    tab.to_csv(OUT / "alt_gates.csv", encoding="utf-8-sig")
    log("\n" + "=" * 100)
    log("【交易结构因子】")
    log("=" * 100)
    log(tab.to_string())
    log(f"\n通过 {int((tab['判决'] == 'PASS').sum())}/{len(tab)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
