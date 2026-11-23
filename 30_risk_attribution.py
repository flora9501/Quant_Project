#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
30_risk_attribution.py  (v3.2)

v3 新增
  ★ --bridge        alpha 收益桥：风格暴露 / alpha / 基准差，恒等式闭合
  ★ --bridge-only   只跑桥，跳过 Brinson（约 1 分钟，迭代用）
  ★ 回放循环补 gross / cost 记账，口径与 factorlab.core.replay、33 号一致：
        net = gross − Σ|Δw|·COST      （Σ|Δw| 已含双边，不再乘 COST_MULT）
  ★ 与 33 号对账：毛收益 / 净收益 / 成本 pp
  ★ --drop-2018     剔除 2018 重算（对照 SUE 那次空仓假象）

v3.1 修正
  ★ ex_ratio 先算后用（修 f-string 括号错误）
  ★ 恒等式断言用未取整值（round(2) 会让 assert 误报）
  ★ 常数项名兼容 const / Intercept
  ★ 缺 daily_basic_*.parquet 时自动降级为两因子回归

v3.2 修正
  ★ score_raw 补回 NaN-aware 语义（四因子全缺失 → NaN，不是 0）
    此前漏了这一步，复现出已废弃的旧口径 10.10%（应为 9.65%）
  ★ mask 用 fillna(0) 消除 Pandas4Warning

原有 Brinson / TE 逻辑与输出保持不变。

用法
  .venv/bin/python 30_risk_attribution.py
  .venv/bin/python 30_risk_attribution.py --bridge
  .venv/bin/python 30_risk_attribution.py --bridge-only
  .venv/bin/python 30_risk_attribution.py --bridge-only --drop-2018
"""
import argparse
import time

import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import DATA_DIR

RAW = DATA_DIR / "raw"
REF = DATA_DIR / "ref"
PANEL = DATA_DIR / "panel"
OUT = DATA_DIR / "output_risk"
OUT.mkdir(parents=True, exist_ok=True)

FACTORS = ["rev_5", "turn_20", "vol_20", "ep"]
TOP_N, REBAL = 50, 5
THRESH = 0.25
COST = 0.0012
ANN = 252
PERIODS_PER_YEAR = ANN / REBAL
HIST_END = "20251231"

ap = argparse.ArgumentParser()
ap.add_argument("--bridge", action="store_true", help="追加 alpha 收益桥")
ap.add_argument("--bridge-only", action="store_true", help="只跑桥，跳过 Brinson")
ap.add_argument("--drop-2018", action="store_true", help="剔除 2018 年")
args = ap.parse_args()
BRIDGE = args.bridge or args.bridge_only


def log(m):
    print(m, flush=True)


t0 = time.time()

# ==================== 1. 数据 ====================
log("① 读取数据...")
frames = [pd.read_parquet(f, columns=["ts_code", "trade_date", "pct_chg"])
          for f in sorted(RAW.glob("daily_[0-9][0-9][0-9][0-9].parquet"))]
daily = pd.concat(frames, ignore_index=True).drop_duplicates(["trade_date", "ts_code"])
ret = (daily.pivot(index="trade_date", columns="ts_code", values="pct_chg") / 100).sort_index()
fwd = ret.shift(-1)
del daily

px = (1 + ret).cumprod()
fwd5 = px.shift(-REBAL) / px - 1

fac = pd.read_parquet(PANEL / "factors_neutral.parquet")
wide = {n: fac.pivot(index="trade_date", columns="ts_code", values=n) for n in FACTORS}

idx = ret.index.intersection(wide["rev_5"].index).sort_values()
idx = idx[idx <= HIST_END]
cols = ret.columns
for n in FACTORS:
    wide[n] = wide[n].reindex(index=idx, columns=cols)
fwd = fwd.reindex(index=idx, columns=cols)
fwd5 = fwd5.reindex(index=idx, columns=cols)

tr = pd.read_parquet(PANEL / "tradable.parquet")
if isinstance(tr.index, pd.DatetimeIndex):
    tr.index = tr.index.strftime("%Y%m%d")
mask = tr.reindex(index=idx, columns=cols).fillna(0).astype(bool)

sb = pd.concat([pd.read_parquet(REF / "stock_basic.parquet"),
                pd.read_parquet(REF / "stock_delisted.parquet")],
               ignore_index=True).drop_duplicates("ts_code")
industry = sb.set_index("ts_code")["industry"].reindex(cols).fillna("UNKNOWN")

# 规模因子（桥需要；缺文件则降级）
logmv = None
SIZE_OK = False
if BRIDGE:
    bfiles = sorted(RAW.glob("daily_basic_*.parquet"))
    if bfiles:
        bframes = [pd.read_parquet(f, columns=["ts_code", "trade_date", "circ_mv"])
                   for f in bfiles]
        bb = pd.concat(bframes, ignore_index=True).drop_duplicates(["trade_date", "ts_code"])
        logmv = np.log(bb.pivot(index="trade_date", columns="ts_code", values="circ_mv")
                       .reindex(index=idx, columns=cols) * 1e4)
        del bb, bframes
        SIZE_OK = True
    else:
        log("   ⚠ 缺 daily_basic_*.parquet → 规模因子不进回归")

log(f"   {len(idx)} 天 × {len(cols)} 只，行业 {industry.nunique()} 个")


# ==================== 2. 分数（含平滑） ====================
# ★ NaN-aware：四因子全缺失 → NaN（不可选），与 factorlab.core.composite 一致
_parts = [wide[n] for n in FACTORS]
_valid = sum(d.notna().astype(int) for d in _parts) > 0
score_raw = (sum(d.fillna(0.0) for d in _parts) / len(FACTORS)).where(_valid)


def smooth_score(s):
    acc = s.fillna(0.0).copy()
    cnt = s.notna().astype(int).copy()
    for k in (REBAL, 2 * REBAL):
        sh = s.shift(k)
        acc = acc + sh.fillna(0.0)
        cnt = cnt + sh.notna().astype(int)
    return acc / cnt.replace(0, np.nan)


score = smooth_score(score_raw)
log(f"   平滑分数完成；有效天数 {int(score.notna().any(axis=1).sum())}")


# ==================== 3. 选股 ====================
def industry_quota(df, top_n=TOP_N):
    share = df.groupby("ind").size() / len(df)
    raw = share * top_n
    base = np.floor(raw).astype(int)
    remain = top_n - int(base.sum())
    if remain > 0:
        for key in list((raw - base).sort_values(ascending=False).index)[:remain]:
            base.loc[key] += 1
    return base


def select_quota_hyst(s, held, sigma_mult=THRESH):
    df = pd.DataFrame({"score": s,
                       "ind": industry.reindex(s.index).fillna("UNKNOWN")}).dropna()
    if df.empty:
        return []
    if len(df) <= TOP_N:
        return list(df.nlargest(TOP_N, "score").index)

    base = industry_quota(df)
    delta = sigma_mult * float(df["score"].std())
    picks = []
    for ind_, k in base.items():
        if k <= 0:
            continue
        pool = df[df["ind"] == ind_].sort_values("score", ascending=False)
        if pool.empty:
            continue
        cut = float(pool["score"].iloc[min(int(k), len(pool)) - 1])
        held_in = [c for c in held if c in pool.index]
        keep = [c for c in held_in if float(pool.loc[c, "score"]) >= cut - delta]
        keep = sorted(keep, key=lambda c: -float(pool.loc[c, "score"]))[:int(k)]
        fill = [c for c in pool.index if c not in set(keep)][:int(k) - len(keep)]
        picks.extend(keep + fill)

    picks = list(dict.fromkeys(picks))[:TOP_N]
    if len(picks) < TOP_N:
        rest = df[~df.index.isin(set(picks))].nlargest(TOP_N - len(picks), "score")
        picks += rest.index.tolist()
    return list(dict.fromkeys(picks))[:TOP_N]


log("② 回放最终版本，记录持仓 / 毛收益 / 成本...")
w_live = pd.Series(0.0, index=cols)
held = []
holdings = {}
gross_s = pd.Series(0.0, index=idx)
cost_s = pd.Series(0.0, index=idx)
to_s = pd.Series(0.0, index=idx)

for i, d in enumerate(idx):
    if i % REBAL == 0:
        s = score.loc[d].where(mask.loc[d]).dropna()
        if len(s) >= TOP_N:
            new = select_quota_hyst(s, held)
            held = new
            if new:
                holdings[d] = list(new)
                w_new = (pd.Series(1.0 / len(new), index=new)
                         .reindex(cols).fillna(0.0))
                tot = w_live.sum()
                w_drift = (w_live / tot) if tot > 0 else w_live
                dw = (w_new - w_drift).abs()
                to_s.loc[d] = float(dw.sum() / 2)
                cost_s.loc[d] = float(dw.sum() * COST)      # 已含双边
                w_live = w_new

    r = fwd.loc[d].fillna(0.0)
    gross_s.loc[d] = float((w_live * r).sum())
    w_live = w_live * (1 + r)

log(f"   {len(holdings)} 个调仓期")

gross_ret = gross_s.iloc[:-1]
cost_ret = cost_s.iloc[:-1]
net = (gross_s - cost_s).iloc[:-1]
bench = fwd.where(mask).mean(axis=1).reindex(net.index).fillna(0.0)

# ---- 与 33 号对账 ----
g_ann = float((1 + gross_ret).cumprod().iloc[-1] ** (ANN / len(gross_ret)) - 1)
n_ann = float((1 + net).cumprod().iloc[-1] ** (ANN / len(net)) - 1)
b_ann = float((1 + bench).cumprod().iloc[-1] ** (ANN / len(bench)) - 1)
log(f"   对账  毛 {g_ann*100:.2f}%  净 {n_ann*100:.2f}%  "
    f"成本 {(g_ann-n_ann)*100:.2f}pp  基准 {b_ann*100:.2f}%")
log("   （33 号应输出 毛 11.54% / 净 9.65% / 基准等权 5.18%）")


# ==================== 3b. alpha 收益桥 ====================
def _ls(f, r, m, q=5, sign=1, min_n=100):
    """池内分 q 组等权，返回 sign·(高组 − 低组) 的日收益"""
    fq = f.where(m).rank(axis=1, pct=True)
    hi = fq > (q - 1) / q
    lo = fq <= 1 / q
    rr = r.where(m)
    out = rr.where(hi).mean(axis=1) - rr.where(lo).mean(axis=1)
    out[(hi.sum(axis=1) < min_n) | (lo.sum(axis=1) < min_n)] = np.nan
    return sign * out


def load_ix_ret():
    f = RAW / "index_daily_000852_SH.parquet"
    if not f.exists():
        log("   ⚠ 缺 index_daily_000852_SH.parquet，基准差一层跳过")
        return None
    d = pd.read_parquet(f)
    s = d.set_index("trade_date")["pct_chg"].astype(float) / 100.0
    s.index = s.index.astype(str)
    return s.sort_index().shift(-1).reindex(net.index).fillna(0.0)


def run_bridge():
    log("\n" + "=" * 78)
    log("【alpha 收益桥】")
    log("=" * 78)

    y = (net - bench).reindex(net.index)
    if args.drop_2018:
        y = y[y.index > "20181231"]

    R = {}
    if SIZE_OK:
        R["规模(小−大)"] = _ls(logmv, fwd, mask, sign=-1)
    R["价值(EP)"] = _ls(wide["ep"], fwd, mask, sign=+1)
    R["低波(vol_20)"] = _ls(wide["vol_20"], fwd, mask, sign=+1)
    R = pd.DataFrame(R).reindex(y.index)

    both = pd.concat([y.rename("ex"), R], axis=1).dropna()
    yy, XX = both["ex"], both[list(R.columns)]

    res = sm.OLS(yy, sm.add_constant(XX)).fit(
        cov_type="HAC", cov_kwds={"maxlags": 5})

    cname = res.params.index[0]                 # 'const' 或 'Intercept'
    alpha = float(res.params[cname]) * ANN
    t_alpha = float(res.tvalues[cname])
    beta = res.params.drop(cname)
    f_ann = XX.mean() * ANN
    contrib = beta * f_ann

    alpha_pp = alpha * 100
    contrib_pp = contrib * 100
    ex_arith = float(yy.mean()) * ANN
    ex_arith_pp = ex_arith * 100

    # 恒等式用未取整值校验，避免 round(2) 误报
    assert abs((float(contrib_pp.sum()) + alpha_pp) - ex_arith_pp) < 1e-8, \
        f"恒等式不闭合：{float(contrib_pp.sum()) + alpha_pp:.6f} vs {ex_arith_pp:.6f}"

    tbl = pd.DataFrame({
        "beta": beta.round(4),
        "因子年化%": (f_ann * 100).round(2),
        "贡献pp": contrib_pp.round(2),
    })
    tbl.loc["alpha"] = [np.nan, np.nan, round(alpha_pp, 2)]
    tbl.loc["合计"] = [np.nan, np.nan, round(float(contrib_pp.sum()) + alpha_pp, 2)]

    log(tbl.to_string())
    log(f"\n   样本 {len(yy)} 天  R² = {res.rsquared:.3f}  alpha t = {t_alpha:.2f}")
    log(f"   日均超额算术年化 = {ex_arith_pp:.2f}pp（合计行已闭合）")

    # ---- 复利口径：用真实净值比，不用 prod(1+active) 近似 ----
    nav_p = (1 + net).reindex(yy.index).cumprod()
    nav_b = (1 + bench).reindex(yy.index).cumprod()
    ex_ratio = (nav_p / nav_b).iloc[-1] ** (ANN / len(yy)) - 1
    log(f"   复利（比值口径）   = {ex_ratio*100:.2f}%")

    # ---- 第二层：基准构造差 ----
    ix = load_ix_ret()
    if ix is not None:
        ix = ix.reindex(yy.index).fillna(0.0)
        gap_arith = float(bench.mean()) * ANN - float(ix.mean()) * ANN
        gap_arith_pp = gap_arith * 100

        nav_ix = (1 + ix).cumprod()
        br_ratio = nav_b.iloc[-1] ** (ANN / len(yy)) - 1
        ix_ratio = nav_ix.iloc[-1] ** (ANN / len(yy)) - 1
        gap_ratio = (1 + br_ratio) / (1 + ix_ratio) - 1
        total_ratio = (1 + ex_ratio) * (1 + gap_ratio) - 1
        total_arith_pp = ex_arith_pp + gap_arith_pp

        log("\n   基准构造差（等权全池 − 中证1000）")
        log(f"     算术 = {gap_arith_pp:+.2f}pp")
        log(f"     复利 = {gap_ratio*100:+.2f}pp")
        log(f"   vs 等权      算术 {ex_arith_pp:+.2f}pp / 复利 {ex_ratio*100:+.2f}%")
        log(f"   vs 中证1000  算术 {total_arith_pp:+.2f}pp / "
            f"复利 {total_ratio*100:+.2f}%")
        log("     （对照文档 8.05%；差 >0.3pp 说明基准定义或样本期不同）")

    # ---- 文档待替换 ----
    log("\n   ── 替换用 ──")
    log(f"   alpha = {alpha_pp:+.2f}%  (t={t_alpha:.2f})")
    for k in beta.index:
        log(f"   贡献 {k:<12} = {contrib_pp[k]:+.2f}pp "
            f"(β={beta[k]:.4f}, 溢价={f_ann[k]*100:.2f}%)")
    log(f"   因子合计 = {float(contrib_pp.sum()):+.2f}pp")

    p_out = OUT / "alpha_bridge.csv"
    tbl.to_csv(p_out, encoding="utf-8-sig")
    log(f"\n   ✅ {p_out}")


if BRIDGE:
    run_bridge()
    if args.bridge_only:
        log(f"\n总耗时 {(time.time() - t0) / 60:.1f} 分钟")
        raise SystemExit(0)


# ==================== 4. Brinson（原有，未改） ====================
log("③ 逐期做 Brinson 分解...")
rows = []

for d, picks in holdings.items():
    r5 = fwd5.loc[d]
    if not np.isfinite(r5.reindex(picks)).any():
        continue

    bench_codes = r5.where(mask.loc[d]).dropna().index
    if len(bench_codes) < 100:
        continue
    w_b = pd.Series(1.0 / len(bench_codes), index=bench_codes)
    r_b = float((w_b * r5.reindex(bench_codes).fillna(0)).sum())

    w_p = pd.Series(1.0 / len(picks), index=picks)

    ind_p = industry.reindex(w_p.index)
    ind_b = industry.reindex(w_b.index)

    wp_ind = w_p.groupby(ind_p).sum()
    wb_ind = w_b.groupby(ind_b).sum()

    tmp_b = pd.DataFrame({"ind": ind_b, "r": r5.reindex(w_b.index).fillna(0)})
    rb_ind = tmp_b.groupby("ind")["r"].mean()

    tmp_p = pd.DataFrame({"ind": ind_p,
                          "wr": w_p.values * r5.reindex(w_p.index).fillna(0).values})
    sum_wr = tmp_p.groupby("ind")["wr"].sum()
    rp_ind = (sum_wr / wp_ind).dropna()

    inds = wp_ind.index.union(wb_ind.index).union(rp_ind.index)
    wp_i = wp_ind.reindex(inds).fillna(0)
    wb_i = wb_ind.reindex(inds).fillna(0)
    rb_i = rb_ind.reindex(inds).fillna(0)
    rp_i = rp_ind.reindex(inds).fillna(rb_i)

    alloc = (wp_i - wb_i) * (rb_i - r_b)
    sel = wb_i * (rp_i - rb_i)
    inter = (wp_i - wb_i) * (rp_i - rb_i)

    rows.append({
        "date": d,
        "r_p": float((w_p * r5.reindex(w_p.index).fillna(0)).sum()),
        "r_b": r_b,
        "alloc": float(alloc.sum()),
        "sel": float(sel.sum()),
        "inter": float(inter.sum()),
        "alloc_by_ind": alloc.to_dict(),
        "sel_by_ind": sel.to_dict(),
    })

br = pd.DataFrame(rows).set_index("date")
br["active"] = br.r_p - br.r_b
br["resid"] = br.active - (br.alloc + br.sel + br.inter)

log(f"   {len(br)} 个有效周期")
log(f"   恒等式 |残差| 均值 {br.resid.abs().mean():.2e}")
assert br.resid.abs().max() < 1e-9, \
    f"Brinson 残差 {br.resid.abs().max():.2e} → 口径错位"


# ==================== 5. 跟踪误差 ====================
def te(x):
    return float(x.std() * np.sqrt(PERIODS_PER_YEAR))


te_total = te(br.active)
te_alloc, te_sel, te_inter = te(br.alloc), te(br.sel), te(br.inter)

var_total = br.active.var()
share_alloc = br.alloc.var() / var_total
share_sel = br.sel.var() / var_total
share_inter = br.inter.var() / var_total
cov_sum = 1 - (share_alloc + share_sel + share_inter)


# ==================== 6. 汇总 ====================
log("\n" + "=" * 78)
log("【跟踪误差】period 口径（√(252/5)）")
log("=" * 78)
log(f"   TE（年化）            {te_total:7.2%}")
log(f"   组合年化波动           {te(br.r_p):7.2%}")
log(f"   基准年化波动           {te(br.r_b):7.2%}")
log(f"   日频 TE（√252）        {float(br.active.std() * np.sqrt(ANN)):7.2%}")

log("\n【风险来源分解（方差占比）】")
risk_tab = pd.DataFrame({
    "来源": ["行业配置 A", "行业内选股 S", "交互项 I", "协方差项", "合计"],
    "TE 贡献": [te_alloc, te_sel, te_inter, np.nan, te_total],
    "方差占比": [share_alloc, share_sel, share_inter, cov_sum, 1.0],
})
log(risk_tab.to_string(
    index=False,
    formatters={"TE 贡献": lambda x: f"{x:7.2%}" if pd.notna(x) else "   —  ",
                "方差占比": lambda x: f"{x:7.1%}"}))

log("\n【收益归因（年化）】")
ret_tab = pd.DataFrame({
    "来源": ["行业配置 A", "行业内选股 S", "交互项 I", "合计超额"],
    "年化贡献": [br.alloc.mean() * PERIODS_PER_YEAR,
                 br.sel.mean() * PERIODS_PER_YEAR,
                 br.inter.mean() * PERIODS_PER_YEAR,
                 br.active.mean() * PERIODS_PER_YEAR],
})
log(ret_tab.to_string(index=False,
                      formatters={"年化贡献": lambda x: f"{x:+7.2%}"}))

log("\n【稳定性】")
log(f"   5 日超额 > 0 的比例    {(br.active > 0).mean():7.1%}")
log(f"   5 日选股贡献 > 0 的比例 {(br.sel > 0).mean():7.1%}")


# ==================== 7. 行业明细 ====================
log("\n④ 行业明细...")
alloc_df = pd.DataFrame(list(br["alloc_by_ind"]), index=br.index).fillna(0.0)
sel_df = pd.DataFrame(list(br["sel_by_ind"]), index=br.index).fillna(0.0)

assert set(alloc_df.columns) <= set(industry.unique()) | {"UNKNOWN"}, \
    f"列名异常: {list(alloc_df.columns)[:10]}"

ind_tab = pd.DataFrame({
    "配置年化": alloc_df.mean(axis=0) * PERIODS_PER_YEAR,
    "选股年化": sel_df.mean(axis=0) * PERIODS_PER_YEAR,
    "配置波动": alloc_df.std(axis=0) * np.sqrt(PERIODS_PER_YEAR),
    "选股波动": sel_df.std(axis=0) * np.sqrt(PERIODS_PER_YEAR),
}).sort_values("选股年化", ascending=False)

log(f"   行业数校验: {len(ind_tab)}（应约 136）")
log("\n【选股贡献最高的 10 个行业（年化 %）】")
log((ind_tab.head(10) * 100).round(2).to_string())
log("\n【选股贡献最低的 10 个行业（年化 %）】")
log((ind_tab.tail(10) * 100).round(2).to_string())
log("\n【配置贡献绝对值最大的 10 个行业（年化 %）】")
order = ind_tab["配置年化"].abs().sort_values(ascending=False).index
log((ind_tab.loc[order].head(10)[["配置年化", "选股年化"]] * 100).round(2).to_string())


# ==================== 8. 集中度 ====================
log("\n⑤ 主动权重集中度...")
aw_records = []
for d, picks in holdings.items():
    r5 = fwd5.loc[d]
    bench_codes = r5.where(mask.loc[d]).dropna().index
    if len(bench_codes) < 100:
        continue
    w_b = pd.Series(1.0 / len(bench_codes), index=bench_codes)
    w_p = pd.Series(1.0 / len(picks), index=picks)
    aw = w_p.sub(w_b.reindex(w_p.index).fillna(0))
    aw_records.append({"date": d, "top10_abs": aw.abs().nlargest(10).sum()})

awt = pd.DataFrame(aw_records).set_index("date")
log(f"   前 10 大主动权重绝对值之和（均值）: {awt.top10_abs.mean():.3f}")

freq = {}
for d, picks in holdings.items():
    for c in picks:
        freq[c] = freq.get(c, 0) + 1
log("\n   最常被持有的 10 只股票（期数 / 总期数）:")
for c, n in pd.Series(freq).sort_values(ascending=False).head(10).items():
    log(f"     {c:<12} {industry.get(c, '?'):<10} {n:>3} / {len(holdings)}")


# ==================== 9. 存盘 ====================
risk_tab.to_csv(OUT / "te_summary.csv", index=False, encoding="utf-8-sig")
ret_tab.to_csv(OUT / "return_attribution.csv", index=False, encoding="utf-8-sig")
ind_tab.to_csv(OUT / "brinson_industry.csv", encoding="utf-8-sig")
br[["r_p", "r_b", "active", "alloc", "sel", "inter"]].to_csv(
    OUT / "period_detail.csv", encoding="utf-8-sig")

log(f"\n✅ 输出到 {OUT}/")
log(f"总耗时 {(time.time() - t0) / 60:.1f} 分钟")
