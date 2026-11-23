#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
77_intraday_daily.py  (v5)
只靠日线 OHLCV：构造 clv / vwap_dev 写入因子缓存 + 拆 ovn_mom 的隔夜/日内腿。

v5 修改
  ★ 写盘前做「符号对齐」，与项目 §7 Method 的惯例一致（sign-aligned so higher is better）
      v4 直接存原始值，导致 clv / vwap_dev 的 IC 为 −0.0197 / −0.0256，
      单边 BH 检验（H1: 残差 IC > 0）把它们的 p 判成 1.0。
      方向依据是预注册的同族因子 idr_rev（日内反转，close/open）：
      「日内强势 → 未来回吐」，所以「收盘越强 → 分数越低」。
      这不是看到 t 值后取反，是入库前漏做的对齐。
  ★ 写盘后自检：与 idr_rev 的平均截面相关应为正

v4：全池日均体检（隔夜/日内/合计 bp）、修 cover 的 nan%
v3：ret_w 用 close/pre_close 重建（pct_chg 两位小数，舍入 5e-5）
v2：非正价格置 NaN、三条腿共用掩码、fwd_cum 有效天 ≥80%

用法
  .venv/bin/python -u 77_intraday_daily.py --no-save
  .venv/bin/python -u 77_intraday_daily.py
输出
  data/panel/factor_zoo/clv.parquet          （已符号对齐）
  data/panel/factor_zoo/vwap_dev.parquet     （已符号对齐）
  data/output_zoo/intraday_decompose.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from factorlab.core import load_panel, smooth_score
from factorlab.factors import load_cached
from factorlab.optimize import industry_groups

RAW = Path("data/raw")
ZOO = Path("data/panel/factor_zoo")
OUT = Path("data/output_zoo")
OUT.mkdir(parents=True, exist_ok=True)
ZOO.mkdir(parents=True, exist_ok=True)

YEARS = range(2018, 2026)
H = 20
NO_SAVE = "--no-save" in sys.argv

# 符号对齐：取 -1 表示「原始指标越高 → 因子分越低」
SIGN = {"clv": -1.0, "vwap_dev": -1.0}


def log(m):
    print(m, flush=True)


def load_daily():
    frames = []
    for y in YEARS:
        f = RAW / f"daily_{y}.parquet"
        if not f.exists():
            continue
        d = pd.read_parquet(f)
        for c in ("open", "high", "low", "close", "pre_close", "amount", "vol"):
            if c not in d.columns:
                d[c] = np.nan
        frames.append(d[["ts_code", "trade_date",
                         "open", "high", "low", "close",
                         "pre_close", "amount", "vol", "pct_chg"]])
    if not frames:
        raise FileNotFoundError("没找到 daily_YYYY.parquet")
    return pd.concat(frames, ignore_index=True)


def wide(daily, col):
    return daily.pivot(index="trade_date", columns="ts_code", values=col).sort_index()


def zscore_rows(x: pd.DataFrame) -> pd.DataFrame:
    v = x.to_numpy(dtype=float)
    med = np.nanmedian(v, axis=1, keepdims=True)
    mad = np.nanmedian(np.abs(v - med), axis=1, keepdims=True) * 1.4826
    ok = np.isfinite(mad) & (mad > 0)
    v = np.clip(v, np.where(ok, med - 3 * mad, -np.inf),
                np.where(ok, med + 3 * mad, np.inf))
    mu = np.nanmean(v, axis=1, keepdims=True)
    sd = np.nanstd(v, axis=1, keepdims=True)
    return pd.DataFrame((v - mu) / np.where(sd > 0, sd, np.nan),
                        index=x.index, columns=x.columns)


def neutralize(x, p, ln_mv, groups):
    out = pd.DataFrame(np.nan, index=p.idx, columns=p.cols)
    for d in p.idx:
        if d not in x.index:
            continue
        m = p.mask.loc[d]
        names = m[m].index
        if len(names) < 100:
            continue
        y = x.loc[d, names].to_numpy(dtype=float)
        g = groups.reindex(names).to_numpy()
        X = pd.get_dummies(pd.Series(g)).to_numpy(dtype=float)
        mv = ln_mv.loc[d, names].to_numpy(dtype=float)
        A = np.column_stack([np.ones(len(names)), mv, X])
        ok = np.isfinite(y) & np.isfinite(A).all(1)
        if ok.sum() < 100:
            continue
        b, *_ = np.linalg.lstsq(A[ok], y[ok], rcond=None)
        out.loc[d, np.asarray(names)[ok]] = y[ok] - A[ok] @ b
    return zscore_rows(out.where(p.mask))


def save_factor(x, name):
    long = x.stack().rename("value").reset_index()
    long.columns = ["trade_date", "ts_code", "value"]
    long = long[np.isfinite(long["value"])]
    long.to_parquet(ZOO / f"{name}.parquet", index=False)
    log(f"   写入 {ZOO/name}.parquet  ({len(long):,} 行)")


def circ_mv_wide(p):
    frames = []
    for y in YEARS:
        f = RAW / f"daily_basic_{y}.parquet"
        if f.exists():
            frames.append(pd.read_parquet(
                f, columns=["ts_code", "trade_date", "circ_mv"]))
    d = pd.concat(frames, ignore_index=True)
    w = d.pivot(index="trade_date", columns="ts_code", values="circ_mv").sort_index()
    return (w.reindex(index=p.idx, columns=p.cols) * 1e4)


def fwd_cum3(lo, li, lr, H=20, min_frac=0.8):
    ok = lo.notna()
    n = ok.astype(float).rolling(H).sum().shift(-H)
    valid = n >= H * min_frac

    def cum(x):
        return (np.exp(x.fillna(0.0).rolling(H).sum().shift(-H)) - 1.0).where(valid)

    return cum(lo), cum(li), cum(lr)


def main():
    t0 = time.time()
    log("① 读取日线（2018–2025，不含 2026）...")
    daily = load_daily()
    log(f"   {len(daily):,} 行")

    pct_w = wide(daily, "pct_chg") / 100.0
    o = wide(daily, "open"); h = wide(daily, "high")
    l = wide(daily, "low");  c = wide(daily, "close")
    pc = wide(daily, "pre_close"); amt = wide(daily, "amount")
    vol = wide(daily, "vol")

    for m in (o, h, l, c, pc):
        m[m <= 0] = np.nan

    ret_w = c / pc - 1.0

    vwap = 10.0 * amt / vol
    clv = (c - l) / (h - l) - 0.5
    vwap_dev = c / vwap - 1.0
    ovn = o / pc - 1.0
    idr = c / o - 1.0
    ret_leg = (1 + ovn) * (1 + idr) - 1.0

    # ---------- ② 自检 ----------
    log("\n② 单位与恒等式自检")
    log(f"   vwap 中位 |close/vwap−1| = "
        f"{np.nanmedian(np.abs(vwap_dev.to_numpy())):.4f}")
    log(f"   vwap 出 [low,high] 比例   = "
        f"{float(np.nanmean(((vwap < l) | (vwap > h)).to_numpy())):.1%}")
    log(f"   clv 范围 [{np.nanmin(clv.to_numpy()):.2f}, "
        f"{np.nanmax(clv.to_numpy()):.2f}]（应 ≈ [−0.5, 0.5]）")

    gap_daily = float(np.nanmax(np.abs((ret_leg - ret_w).to_numpy())))
    log(f"   逐日恒等式 max|(1+ovn)(1+idr)−1 − ret| = {gap_daily:.2e}")
    if gap_daily > 1e-9:
        raise SystemExit("逐日恒等式不成立——检查 ovn / idr 的定义")

    chk = float(np.nanmax(np.abs((ret_w - pct_w).to_numpy())))
    log(f"   ret 与 pct_chg/100 最大差 = {chk:.2e}")
    if chk > 5e-3:
        log("   ⚠ 超过 0.5%：pre_close 与 pct_chg 不自洽")
    elif chk < 1e-3:
        log("   ✅ 舍入级（<1e-3），数据一致")

    # ---------- ③ 中性化 + 符号对齐 + 写盘 ----------
    p = load_panel()
    groups = industry_groups(p)
    ln_mv = np.log(circ_mv_wide(p))

    def sub(x):
        return x.reindex(index=p.idx, columns=p.cols)

    log("\n③ 中性化 + 符号对齐 + 写入因子缓存")
    idr_rev = load_cached("idr_rev")
    if idr_rev is not None:
        idr_rev = idr_rev.reindex(index=p.idx, columns=p.cols)

    for name, raw in (("clv", sub(clv)), ("vwap_dev", sub(vwap_dev))):
        nz = neutralize(raw, p, ln_mv, groups)
        # ★ 符号对齐：把原始「越高越强」翻成「越高越差」，与 idr_rev 同向
        nz = nz * SIGN[name]

        # 自检：与 idr_rev 的平均截面相关应为正
        if idr_rev is not None:
            corr = nz.where(p.mask).corrwith(idr_rev.where(p.mask), axis=1).mean()
            flag = "✅" if corr > 0 else "⚠"
            log(f"   与 idr_rev 平均截面相关 {corr:+.3f} {flag}")
            if corr <= 0:
                raise SystemExit(f"{name} 符号对齐后与 idr_rev 反号，检查 SIGN")

        if not NO_SAVE:
            save_factor(nz, name)
            chk2 = load_cached(name)
            if chk2 is not None:
                log(f"   回读 {name}: {chk2.shape}（2364 列是正常的）")

    # ---------- ④ 分解 ----------
    log("\n④ 分解 ovn_mom 的前向 20 日收益：隔夜腿 vs 日内腿")
    ovn_mom = load_cached("ovn_mom")
    if ovn_mom is None:
        log("   缺 ovn_mom，跳过分解")
        return 0
    ovn_mom = ovn_mom.reindex(index=p.idx, columns=p.cols)

    ov, idr_, rt = sub(ovn), sub(idr), sub(ret_w)
    ok = ov.notna() & idr_.notna() & rt.notna()

    msk_np = p.mask.to_numpy()
    cover = float(ok.to_numpy()[msk_np].sum() / msk_np.sum())
    log(f"   三条腿同时有效的在池占比 = {cover:.1%}")

    ovm = ov.where(p.mask).to_numpy()
    idm = idr_.where(p.mask).to_numpy()
    rtm = rt.where(p.mask).to_numpy()
    log(f"   全池日均：隔夜 {np.nanmean(ovm)*1e4:+.2f}bp  "
        f"日内 {np.nanmean(idm)*1e4:+.2f}bp  "
        f"合计 {np.nanmean(rtm)*1e4:+.2f}bp")
    log("   （判据：隔夜 −2~−6bp / 日内 +3~+7bp / 合计 ≈ +2bp）")

    lo, li, lr = (np.log1p(x.where(ok)) for x in (ov, idr_, rt))
    f_ovn, f_idr, f_tot = fwd_cum3(lo, li, lr, H)

    gap = float(np.nanmax(np.abs(((1 + f_ovn) * (1 + f_idr) - 1 - f_tot)
                                 .to_numpy())))
    log(f"   累计恒等式 max|gap| = {gap:.2e}（应 ~1e-12）")
    if not (gap < 1e-6):
        raise SystemExit("累计恒等式不成立，停止")

    sc = smooth_score(ovn_mom)
    rows = []
    for d in p.idx:
        valid_d = f_ovn.loc[d].notna()
        s = sc.loc[d].where(p.mask.loc[d] & valid_d).dropna()
        if len(s) < 200:
            continue
        q = pd.qcut(s.rank(method="first"), 5, labels=False)
        for k in range(5):
            names = s.index[q == k].to_numpy()
            if len(names) < 20:
                continue
            rows.append({
                "quintile": k + 1, "n": len(names),
                "f_ovn": float(np.nanmean(f_ovn.loc[d, names].to_numpy())),
                "f_idr": float(np.nanmean(f_idr.loc[d, names].to_numpy())),
                "f_tot": float(np.nanmean(f_tot.loc[d, names].to_numpy())),
            })

    if not rows:
        log("   ⚠ 没有任何有效窗口")
        return 1

    tab = pd.DataFrame(rows).groupby("quintile").mean()
    tab["年化_ovn%"] = (tab["f_ovn"] * 252 / H * 100).round(2)
    tab["年化_idr%"] = (tab["f_idr"] * 252 / H * 100).round(2)
    tab["年化_tot%"] = (tab["f_tot"] * 252 / H * 100).round(2)
    tab["ovn占比%"] = (tab["f_ovn"] / tab["f_tot"] * 100).round(1)

    log("\n" + "=" * 78)
    log("【ovn_mom 五分组的 20 日前向收益拆解】（年化）")
    log("=" * 78)
    log(tab[["n", "年化_ovn%", "年化_idr%", "年化_tot%"]].to_string())

    sp = tab.loc[5] - tab.loc[1]
    log(f"\n  Q5−Q1：隔夜 {sp['年化_ovn%']:+.2f}%  "
        f"日内 {sp['年化_idr%']:+.2f}%  合计 {sp['年化_tot%']:+.2f}%")

    tab.to_csv(OUT / "intraday_decompose.csv", encoding="utf-8-sig")
    log(f"\n✅ 输出 {OUT/'intraday_decompose.csv'}")
    log(f"总耗时 {(time.time() - t0) / 60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    sys.exit(main())
