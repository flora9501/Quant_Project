#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
38c_oi_validate.py
验证 on_tail（隔夜尾部哑变量）的 +0.76pp 超额是真增量还是噪声。

只用 2018-2025，绝不读 2026。

修了什么
    - 删掉 neutralize：tail 是 0/1 哑变量，不做中性化
    - ann_excess 改为几何口径：(nav_p/nav_b)^(252/n)-1（上一版低估约 0.96pp）
    - 分年 groupby 兼容 datetime64 索引

用法
    .venv/bin/python -u 38c_oi_validate.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from factorlab.core import load_panel, FACTORS, smooth_score, replay
from factorlab.factors import load_price_data

OUT = Path("data/output_zoo")
OUT.mkdir(parents=True, exist_ok=True)
HIST_END = "20251231"


def log(m):
    print(m, flush=True)


def _wide(df, col, dates, codes):
    return (df.pivot(index="trade_date", columns="ts_code", values=col)
              .reindex(index=dates, columns=codes))


def alive_zero(x):
    """生命期内 NaN→0（停牌），生命期外保持 NaN（未上市/已退市）"""
    v = x.notna()
    fwd = v.astype(np.int8).cummax().astype(bool)
    bwd = v[::-1].astype(np.int8).cummax()[::-1].astype(bool)
    return x.fillna(0.0).where(fwd & bwd)


def build_tails(panel, daily, qs=(0.05, 0.10, 0.15, 0.20)):
    """隔夜累计收益最差 q 分位 → 哑变量 0 / -1"""
    dates, codes = panel.idx, panel.cols
    ret = _wide(daily, "pct_chg", dates, codes) / 100.0
    o = _wide(daily, "open", dates, codes).where(lambda z: z > 0)
    c = _wide(daily, "close", dates, codes).where(lambda z: z > 0)

    intraday = (c / o - 1.0).where(lambda z: z.abs() < 0.5)
    overnight = ((1.0 + ret) / (1.0 + intraday) - 1.0).where(lambda z: z.abs() < 0.5)

    err = float(np.nanmax(np.abs(
        ((1 + overnight) * (1 + intraday) - 1 - ret).to_numpy())))
    log(f"   恒等式误差 {err:.2e}")

    on = alive_zero(overnight)
    cum = np.log1p(on).rolling(20, min_periods=20).sum()
    rk = cum.rank(axis=1, pct=True)

    out = {}
    for q in qs:
        out[q] = -(rk <= q).astype(float).where(cum.notna())
    return out


def comp(panel, fac, names):
    """等权复合：n 个分量各占 1/n"""
    parts = [fac[n] for n in names]
    out = sum(d.fillna(0.0) for d in parts)
    valid = sum(d.notna().astype(int) for d in parts) > 0
    return (out / len(parts)).where(valid)


def ann_excess(net, panel):
    """几何超额：(组合净值 / 基准净值)^(252/n) - 1"""
    bm = panel.fwd.where(panel.mask).mean(axis=1).reindex(net.index).fillna(0.0)
    n = len(net)
    nav_p = (1 + net).cumprod().iloc[-1]
    nav_b = (1 + bm).cumprod().iloc[-1]
    return float((nav_p / nav_b) ** (252 / n) - 1)


def paired(base_net, aug_net):
    """日收益差的配对 t + 分年换算"""
    d = (aug_net - base_net).dropna()
    if len(d) < 100 or d.std() == 0:
        return np.nan, pd.Series(dtype=float)
    t = float(d.mean() / d.std() * np.sqrt(len(d)))
    yr = pd.Series([str(x)[:4] for x in d.index], index=d.index)   # 兼容 datetime64
    by_year = d.groupby(yr).apply(lambda z: float(z.mean() * (252 / max(len(z), 1))))
    return t, by_year


def main():
    log("① 加载（硬截止 20251231）...")
    panel = load_panel(hist_end=HIST_END)
    last = str(panel.idx.max())[:10].replace("-", "")
    assert last <= HIST_END, f"面板越界 {last}"
    log(f"   {len(panel.idx)} 天 × {len(panel.cols)} 只，末日 {last}")

    daily, _ = load_price_data(daily_cols=("pct_chg", "open", "close"))

    log("\n② 构造隔夜尾部哑变量...")
    tails = build_tails(panel, daily)

    base = replay(smooth_score(comp(panel, panel.fac, FACTORS)), panel)
    base_ex = ann_excess(base.net, panel)
    log(f"\n基准 A: 年化 {base.ann():.2%}  超额 {base_ex:+.2%}  换手 {base.ann_turnover():.0%}")

    rows = []

    log("\n③ 阈值敏感性（加分版：tail 进 composite，权重 1/5）")
    for q, f in tails.items():
        fac = {**panel.fac, "__tail": f}
        rp = replay(smooth_score(comp(panel, fac, FACTORS + ["__tail"])), panel)
        t, _ = paired(base.net, rp.net)
        ex = ann_excess(rp.net, panel)
        log(f"   q={q:>4.0%}  年化 {rp.ann():7.2%}  超额 {ex:+7.2%}  "
            f"换手 {rp.ann_turnover():4.0%}  配对t {t:+.2f}")
        rows.append({"方式": "加分", "q": q, "年化": round(rp.ann(), 4),
                     "超额": round(ex, 4), "换手": round(rp.ann_turnover(), 4),
                     "配对t": round(t, 2) if np.isfinite(t) else None})

    log("\n④ 硬过滤（尾部股票分数置 NaN，被 replay 的 dropna 剔除）")
    base_score = smooth_score(comp(panel, panel.fac, FACTORS))
    for q, f in tails.items():
        sm_hard = base_score.where(f != 1.0)
        rp = replay(sm_hard, panel)
        t, _ = paired(base.net, rp.net)
        ex = ann_excess(rp.net, panel)
        log(f"   q={q:>4.0%}  年化 {rp.ann():7.2%}  超额 {ex:+7.2%}  "
            f"换手 {rp.ann_turnover():4.0%}  配对t {t:+.2f}")
        rows.append({"方式": "硬过滤", "q": q, "年化": round(rp.ann(), 4),
                     "超额": round(ex, 4), "换手": round(rp.ann_turnover(), 4),
                     "配对t": round(t, 2) if np.isfinite(t) else None})

    log("\n⑤ 分年（q=10% 加分版）")
    fac = {**panel.fac, "__tail": tails[0.10]}
    rp = replay(smooth_score(comp(panel, fac, FACTORS + ["__tail"])), panel)
    t, by_year = paired(base.net, rp.net)
    log(f"   配对 t = {t:+.2f}   全样本 Δ超额 {ann_excess(rp.net, panel) - base_ex:+.2%}")
    if len(by_year):
        log(by_year.round(4).to_string())
        pos = int((by_year > 0).sum())
        log(f"   正贡献年数 {pos}/{len(by_year)}")

    pd.DataFrame(rows).to_csv(OUT / "oi_validate.csv", index=False, encoding="utf-8-sig")
    log(f"\n✅ {OUT}/oi_validate.csv")


if __name__ == "__main__":
    main()
