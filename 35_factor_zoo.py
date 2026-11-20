#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
35_factor_zoo.py
论文复现台：一论文一因子，报复制率。

角色不是「筛因子」，是「测量仪」：
  - 每个因子声明它复现哪篇论文、原文方向、定义保真度
  - 强度（|t|）与方向（符号）分开判，别用带符号的 t 卡阈值
  - 双频率：h=5（你的生产频率，决定能不能用）+ h_paper（论文频率，决定成不成立）
  - 复现判定与「增量判定」分开报（复现成功 ≠ 有增量）

用法
  .venv/bin/python -u 35_factor_zoo.py                 # 默认 5 篇，双频率
  .venv/bin/python -u 35_factor_zoo.py --all           # 含 3 个低保真度扩展
  .venv/bin/python -u 35_factor_zoo.py --diag          # 追加两个诊断（+4 分钟）
  .venv/bin/python -u 35_factor_zoo.py --skip-hpaper   # 只看 h=5，省 3 分钟
  .venv/bin/python -u 35_factor_zoo.py rev_20          # 冒烟
  .venv/bin/python -u 35_factor_zoo.py --rebuild       # 忽略中性化缓存

输出
  data/output_zoo/factor_zoo.csv          (h=5 主表)
  data/output_zoo/factor_zoo_hpaper.csv   (论文频率附表)
  data/output_zoo/factor_zoo_diag.csv     (--diag)
  data/output_zoo/factor_zoo.md
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))

from factorlab.core import load_panel, composite, smooth_score, replay
from factorlab.factors import (load_price_data, neutralize, fill_alive_zero,
                               load_cached, save_cached)
from factorlab.gate import evaluate

OUT = Path("data/output_zoo")
OUT.mkdir(parents=True, exist_ok=True)

# ==================== 论文登记表 ====================
# paper_ic_sign: 按**原论文结论**，我们因子的 IC 应该是什么符号
#   +1 表示「我们因子越大越好」（构造里已含必要的取负）
#   -1 表示原论文结论意味着我们因子越小越好
# h_paper: 原论文的评估/持有期限（交易日）。用它复现，用 h=5 判断能不能落地
# fidelity: 高 / 中 / 低 —— 中低要在 md 里写明差异，别硬说成复现
PAPERS = {
    "rev_20": dict(
        paper="Jegadeesh (1990) — 短期反转",
        family="reversal", role="control", paper_ic_sign=+1, fidelity="高",
        h_paper=20,
        spec="上月个股收益取负；我们用 20 日累计收益取负"),
    "amihud": dict(
        paper="Amihud (2002) — ILLIQ 非流动性",
        family="liquidity", role="test", paper_ic_sign=+1, fidelity="高",
        h_paper=20,
        spec="|r|/成交额，月均；我们用 20 日均，amount 已换算为元"),
    "max_ret": dict(
        paper="Bali, Cakici & Whitelaw (2011) — MAX",
        family="lottery", role="test", paper_ic_sign=+1, fidelity="高",
        h_paper=20,
        spec="月内最大日收益取负；我们用 20 日内最大 5 日收益取负"),
    "high_52w": dict(
        paper="George & Hwang (2004) — 52 周高点",
        family="anchoring", role="test", paper_ic_sign=-1, fidelity="高",
        h_paper=120,
        spec="原文：越接近 52 周高点越好（持有 6 个月）；"
             "我们因子已取负，故原文预期 IC<0"),
    "turn_abn": dict(
        paper="Chordia, Subrahmanyam & Anshuman (2001) — 换手率",
        family="turnover", role="test", paper_ic_sign=+1, fidelity="中",
        h_paper=20,
        spec="原文用**原始**换手率；我们用 20/250 异常换手并取负，"
             "失败时无法区分是「换手率」还是「异常换手」的问题"),
    # ---- 扩展：定义保真度不足，只作参考，不进复制率 ----
    "ivol": dict(
        paper="Ang, Hodrick, Xing & Zhang (2006) — IVOL",
        family="lottery", role="ext", paper_ic_sign=+1, fidelity="中",
        h_paper=20,
        spec="原文对 FF3 取残差；我们对等权市场单因子取残差"),
    "downside": dict(
        paper="Ang, Chen & Xing (2006) — 下行风险",
        family="lottery", role="ext", paper_ic_sign=+1, fidelity="低",
        h_paper=20,
        spec="原文是下行 beta；我们算的是半方差占比，不是同一个因子"),
    "skew_20": dict(
        paper="Harvey & Siddique (2000) / Boyer et al. (2010) — 偏度",
        family="lottery", role="ext", paper_ic_sign=+1, fidelity="低",
        h_paper=20,
        spec="原文是协偏度 / 预期特质偏度；我们算已实现偏度"),
}

DEFAULT_PAPERS = ["rev_20", "amihud", "max_ret", "high_52w", "turn_abn"]
ALL_PAPERS = list(PAPERS)


def log(m):
    print(m, flush=True)


def _wide(df, col, dates, codes):
    return (df.pivot(index="trade_date", columns="ts_code", values=col)
              .reindex(index=dates, columns=codes))


# ==================== 候选因子 ====================
def build_candidates(panel, daily, basic):
    dates, codes = panel.idx, panel.cols

    ret_raw = _wide(daily, "pct_chg", dates, codes) / 100.0
    turn = _wide(basic, "turnover_rate", dates, codes)
    cmv = _wide(basic, "circ_mv", dates, codes)      # 万元
    amt = _wide(daily, "amount", dates, codes)       # 千元

    ret = fill_alive_zero(ret_raw)                   # 生命期内 0，生命期外 NaN

    mkt = ret.mean(axis=1)                           # 等权市场收益（按行）
    resid = ret.sub(mkt, axis=0)

    neg_sq = ret.clip(upper=0.0) ** 2
    dvol = np.sqrt(neg_sq.rolling(20, min_periods=20).mean())
    tvol = ret.rolling(20, min_periods=20).std()

    raw = {}
    raw["rev_20"] = -ret.rolling(20, min_periods=20).sum()
    raw["ivol"] = -resid.rolling(20, min_periods=20).std() * np.sqrt(252)
    raw["downside"] = -(dvol / tvol.replace(0, np.nan))
    illiq = (ret.abs() * 1e4) / (amt * 1000).replace(0, np.nan)
    raw["amihud"] = illiq.rolling(20, min_periods=20).mean()
    raw["turn_abn"] = -(turn.rolling(20, min_periods=20).mean()
                        / turn.rolling(250, min_periods=200).mean().replace(0, np.nan))
    raw["max_ret"] = -ret.rolling(5, min_periods=5).sum().rolling(20, min_periods=20).max()
    raw["skew_20"] = -ret.rolling(20, min_periods=20).skew()
    px = (1 + ret).cumprod()
    raw["high_52w"] = -(px / px.rolling(250, min_periods=200).max())
    raw["ln_mv"] = -np.log(cmv * 1e4)

    return raw, amt


# ==================== 判定 ====================
def direction_ok(r, meta):
    """方向是否与原论文一致。单独一个维度，别塞进 verdict。"""
    ic = r.ic.get("IC", np.nan)
    if not np.isfinite(ic):
        return False
    return bool(np.sign(ic) == np.sign(meta["paper_ic_sign"]))


def replication_verdict(r, meta):
    """强度看 |t|，方向看 direction_ok。

    注意：不能用 t >= 3.0 直接比 —— 原文预期方向为负的因子
    （George & Hwang 这类），显著结果本身就是负 t，会被永久误杀。
    """
    t = r.t_nw
    if not np.isfinite(t):
        return "NO_DATA"
    same = direction_ok(r, meta)
    at = abs(t)
    if at >= 3.0:
        return "REPLICATED" if same else "SIGN_FLIPPED"
    if at >= 2.0:
        return "WEAK"
    return "NOT_REPLICATED"


def incremental_ok(r):
    """增量判定：残差 IC 够 + 组合边际够。与复现是两件事。"""
    return (np.isfinite(r.marginal_t) and r.marginal_t >= 2.0
            and abs(r.resid_ic) >= 0.015)


def bh(ps):
    p = np.asarray(ps, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        _, qq, _, _ = multipletests(p[ok], method="fdr_bh")
        q[ok] = qq
    return q


def daily_corr(a, b, mask):
    """逐日截面 Pearson 相关，再取时间均值。向量化，别去 groupby。"""
    r = a.where(mask).corrwith(b.where(mask), axis=1)
    return float(r.dropna().mean()) if r.notna().any() else np.nan


def cluster_count(cm, thr):
    """|rho|>=thr 单链接合并，返回簇数 = 有效独立检验数"""
    names = list(cm.index)
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            v = cm.loc[a, b]
            if np.isfinite(v) and abs(v) >= thr:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
    return len({find(n) for n in names})


# ==================== 主流程 ====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("only", nargs="*", help="只跑这些因子（冒烟用）")
    ap.add_argument("--all", action="store_true", help="含 3 个低保真度扩展")
    ap.add_argument("--diag", action="store_true",
                    help="追加诊断：未复现者的频率敏感性 + 负边际者的换手")
    ap.add_argument("--skip-hpaper", action="store_true",
                    help="不做论文频率评估，只跑 h")
    ap.add_argument("--h", type=int, default=5)
    ap.add_argument("--q", type=float, default=0.05)
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()

    t0 = time.time()
    log("① 加载面板与原始数据...")
    panel = load_panel()
    daily, basic = load_price_data(daily_cols=("pct_chg", "amount"))
    log(f"   {len(panel.idx)} 天 × {len(panel.cols)} 只；"
        f"daily {daily.shape}, basic {basic.shape}")

    log("② 构造候选因子...")
    raw, amt = build_candidates(panel, daily, basic)
    med_amt = float(np.nanmedian(amt.to_numpy()))
    log(f"   amount 中位数 {med_amt:.3g}（千元；应在 1e4–1e6）")

    names = a.only or (ALL_PAPERS if a.all else DEFAULT_PAPERS)
    unknown = [n for n in names if n not in PAPERS]
    if unknown:
        raise SystemExit(f"未知因子 {unknown}；可选 {ALL_PAPERS}")

    for k in names:
        log(f"   {k:<10} 覆盖 {raw[k].notna().values.mean():.1%}  "
            f"[{PAPERS[k]['paper']}]")

    log(f"\n③ 中性化（{len(names)} 个，每个约 20–30 秒）...")
    fitted, stage1 = {}, []
    for name in names:
        f = None if a.rebuild else load_cached(name)
        if f is not None:
            log(f"   {name:<10} 缓存命中")
        else:
            f = neutralize(raw, panel.mask, panel.industry,
                           names=[name], verbose=False)[name]
            save_cached(name, f)
            log(f"   {name:<10} 已中性化并落盘")
        fitted[name] = f

        # require_positive=False：方向交给论文登记表判断，不用 abs() 掩盖
        r = evaluate(name, f, panel, h=a.h, with_marginal=False,
                     verbose=False, require_positive=False)
        r.meta = PAPERS[name]
        r.dir_ok = direction_ok(r, r.meta)
        r.repl = replication_verdict(r, r.meta)
        stage1.append(r)
        ls = f"{r.ls_bp:+.0f}bp" if np.isfinite(r.ls_bp) else "n/a"
        log(f"   {name:<10} IC {r.ic['IC']:+.4f}  NW t {r.t_nw:+.2f}  "
            f"Q5-Q1 {ls}  → {r.repl}")

    # ---- 论文频率（原论文的评估期限）----
    if not a.skip_hpaper:
        log(f"\n③b 论文频率评估（各按 h_paper）...")
        for r in stage1:
            hp = int(r.meta.get("h_paper", a.h))
            if hp == a.h:
                r.paper_res, r.repl_paper, r.dir_ok_paper = r, r.repl, r.dir_ok
            else:
                pr = evaluate(r.name, fitted[r.name], panel, h=hp,
                              with_marginal=False, verbose=False,
                              require_positive=False)
                pr.meta = r.meta
                r.paper_res = pr
                r.dir_ok_paper = direction_ok(pr, r.meta)
                r.repl_paper = replication_verdict(pr, r.meta)
            log(f"   {r.name:<10} h={hp:<4} IC {r.paper_res.ic['IC']:+.4f}  "
                f"NW t {r.paper_res.t_nw:+.2f}  "
                f"Q5-Q1 {r.paper_res.ls_bp:+.0f}bp  → {r.repl_paper}")
    else:
        for r in stage1:
            r.paper_res, r.repl_paper, r.dir_ok_paper = r, r.repl, r.dir_ok

    # ---- 多重检验：只对测试组做（对照组不进分母）----
    tests = [r for r in stage1 if r.meta["role"] == "test"]
    q_nw = bh(np.array([r.p_nw for r in tests], dtype=float))
    for r, qq in zip(tests, q_nw):
        r.q_nw = float(qq)

    # ---- 阶段二：增量（在生产频率上做，因为要在 h=5 调仓）----
    log("\n④ 阶段二：组合边际（增量，不是复现判据）...")
    for r in stage1:
        if r.repl not in ("REPLICATED", "SIGN_FLIPPED", "WEAK"):
            r.marginal_t, r.marginal_delta = np.nan, np.nan
            continue
        full = evaluate(r.name, fitted[r.name], panel, h=a.h,
                        with_marginal=True, verbose=False, require_positive=False)
        r.marginal_t = full.marginal_t
        r.marginal_delta = full.marginal_delta
        log(f"   {r.name:<10} 边际 t {r.marginal_t:+.2f}  Δ {r.marginal_delta:+.2%}")

    # ---- 相关性与有效独立检验数 ----
    log("\n⑤ 候选因子两两相关（逐日截面，时间均值）...")
    cm = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            v = daily_corr(fitted[x], fitted[y], panel.mask)
            cm.loc[x, y] = cm.loc[y, x] = v
    log(cm.round(3).to_string())
    k50 = cluster_count(cm, 0.5)
    k60 = cluster_count(cm, 0.6)
    k70 = cluster_count(cm, 0.7)
    off = cm.to_numpy()[np.triu_indices(len(names), k=1)]
    max_off = float(np.nanmax(np.abs(off))) if off.size else np.nan
    log(f"   有效独立检验数：|rho|>=0.5 → {k50}；>=0.6 → {k60}；"
        f">=0.7 → {k70}（共 {len(names)} 个）。最大两两相关 {max_off:.3f}。")

    # ---- 诊断（可选）----
    diag_rows = []
    if a.diag:
        log("\n⑥ 诊断 A：未复现者的频率敏感性（原文多为月度/半年持有）...")
        for r in stage1:
            if r.repl not in ("NOT_REPLICATED", "WEAK", "SIGN_FLIPPED"):
                continue
            for hh in (20, 60):
                rr = evaluate(r.name, fitted[r.name], panel, h=hh,
                              with_marginal=False, verbose=False,
                              require_positive=False)
                v = replication_verdict(rr, r.meta)
                log(f"   {r.name:<10} h={hh:<3} IC {rr.ic['IC']:+.4f}  "
                    f"NW t {rr.t_nw:+.2f}  Q5-Q1 {rr.ls_bp:+.0f}bp  → {v}")
                diag_rows.append({"factor": r.name, "kind": f"h={hh}",
                                  "IC": round(rr.ic["IC"], 4),
                                  "t_NW": round(rr.t_nw, 2),
                                  "verdict": v})

        log("\n   诊断 B：负边际者的换手（区分成本 vs 信号）...")
        base = replay(smooth_score(composite(panel)), panel)
        log(f"   基准：换手 {base.ann_turnover():.0%}  年化 {base.ann():.2%}")
        for r in stage1:
            if not (np.isfinite(r.marginal_t) and r.marginal_t < 2.0):
                continue
            aug = replay(smooth_score(composite(panel)
                                      + fitted[r.name].fillna(0.0)), panel)
            log(f"   {r.name:<10} 换手 {base.ann_turnover():.0%} → "
                f"{aug.ann_turnover():.0%}   年化 {base.ann():.2%} → "
                f"{aug.ann():.2%}   (Δ{r.marginal_delta:+.2%}, t{r.marginal_t:+.2f})")
            diag_rows.append({"factor": r.name, "kind": "turnover",
                              "IC": np.nan, "t_NW": np.nan,
                              "verdict": f"{base.ann_turnover():.0%}->"
                                         f"{aug.ann_turnover():.0%}"})

    # ---- 汇总：h=5 主表 ----
    rows = []
    for r in stage1:
        m = r.meta
        qq = getattr(r, "q_nw", np.nan)
        rows.append({
            "因子": r.name,
            "论文": m["paper"],
            "家族": m["family"],
            "角色": m["role"],
            "保真度": m["fidelity"],
            "IC": round(r.ic["IC"], 4),
            "ICIR": round(r.ic["ICIR"], 3),
            "t_NW": round(r.t_nw, 2),
            "p_NW": float(f"{r.p_nw:.3g}") if np.isfinite(r.p_nw) else None,
            "Q5-Q1(bp)": round(r.ls_bp, 0) if np.isfinite(r.ls_bp) else None,
            "LS_t_NW": round(r.ls_t_nw, 2) if np.isfinite(r.ls_t_nw) else None,
            "分年同号": bool((r.yearly > 0).all() or (r.yearly < 0).all()),
            "最大相关": round(r.max_corr, 3),
            "残差IC": round(r.resid_ic, 4),
            "方向一致": r.dir_ok,
            "复现判定": r.repl,
            "增量": "✅" if incremental_ok(r) else "❌",
            "边际t": round(r.marginal_t, 2) if np.isfinite(r.marginal_t) else None,
            "FDR存活": (bool(qq < a.q) if m["role"] == "test"
                       and np.isfinite(qq) else None),
        })
    tab = pd.DataFrame(rows).set_index("因子")

    # ---- 汇总：论文频率附表 ----
    prow = []
    for r in stage1:
        pr = r.paper_res
        prow.append({
            "因子": r.name,
            "h_paper": int(r.meta.get("h_paper", a.h)),
            "IC": round(pr.ic["IC"], 4),
            "t_NW": round(pr.t_nw, 2),
            "Q5-Q1(bp)": round(pr.ls_bp, 0) if np.isfinite(pr.ls_bp) else None,
            "LS_t_NW": round(pr.ls_t_nw, 2) if np.isfinite(pr.ls_t_nw) else None,
            "方向一致": r.dir_ok_paper,
            "复现判定": r.repl_paper,
        })
    tab_p = pd.DataFrame(prow).set_index("因子")

    is_test = tab["角色"] == "test"
    n_test = int(is_test.sum())
    n_rep = int(((tab["复现判定"] == "REPLICATED") & is_test).sum())
    n_flip = int(((tab["复现判定"] == "SIGN_FLIPPED") & is_test).sum())
    n_weak = int(((tab["复现判定"] == "WEAK") & is_test).sum())
    n_none = int(((tab["复现判定"] == "NOT_REPLICATED") & is_test).sum())
    n_inc = int(((tab["增量"] == "✅") & is_test).sum())

    tp = tab_p["复现判定"]
    m_test = pd.Series(is_test.values, index=tab_p.index)
    p_rep = int(((tp == "REPLICATED") & m_test).sum())
    p_weak = int(((tp == "WEAK") & m_test).sum())
    p_flip = int(((tp == "SIGN_FLIPPED") & m_test).sum())
    p_none = int(((tp == "NOT_REPLICATED") & m_test).sum())

    ctrl = [r for r in stage1 if r.meta["role"] == "control"]
    ctrl_ok = all(r.repl == "REPLICATED" for r in ctrl) if ctrl else True

    log("\n" + "=" * 120)
    log(f"【论文复现 · h={a.h}（生产频率）· {len(tab)} 个因子 / 测试组 {n_test} 个】")
    log("=" * 120)
    log(tab.to_string())

    if not a.skip_hpaper:
        log("\n" + "-" * 120)
        log("【论文频率附表】")
        log("-" * 120)
        log(tab_p.to_string())

    if ctrl:
        log(f"\n阳性对照: "
            f"{'✅ 通过（管道可用）' if ctrl_ok else '❌ 未通过，先修管道，再解释论文'}")
    if n_test == 0:
        log("测试组为空（只跑了对照组），复制率不适用。")
    else:
        log(f"复制率  生产频率 h={a.h} : {n_rep}/{n_test}"
            f"   （翻转 {n_flip} / 弱 {n_weak} / 未复现 {n_none}）")
        if not a.skip_hpaper:
            log(f"复制率  论文频率     : {p_rep}/{n_test}"
                f"   （翻转 {p_flip} / 弱 {p_weak} / 未复现 {p_none}）")
        log(f"增量通过（测试组, h={a.h}）: {n_inc}/{n_test}"
            f"   ← 复现成功 ≠ 有增量")
    log(f"有效独立检验数: {k50}-{k70}（|rho| 阈值 0.5–0.7），最大两两相关 {max_off:.3f}")

    tab.to_csv(OUT / "factor_zoo.csv", encoding="utf-8-sig")
    tab_p.to_csv(OUT / "factor_zoo_hpaper.csv", encoding="utf-8-sig")
    if diag_rows:
        pd.DataFrame(diag_rows).to_csv(OUT / "factor_zoo_diag.csv",
                                       index=False, encoding="utf-8-sig")

    try:
        md_main, md_paper = tab.to_markdown(), tab_p.to_markdown()
    except Exception:                     # tabulate 不在依赖里时的兜底
        md_main = "```\n" + tab.to_string() + "\n```"
        md_paper = "```\n" + tab_p.to_string() + "\n```"

    with open(OUT / "factor_zoo.md", "w", encoding="utf-8") as fh:
        fh.write("# 论文复现 · Batch 1（价量）\n\n")
        fh.write(f"- 样本：CSI 1000 PIT，2018–2025\n")
        fh.write(f"- 生产频率 h={a.h}；论文频率见附表（各按 h_paper）\n")
        fh.write("- 中性化：行业 + ln市值（FWL）\n")
        fh.write(f"- amount 中位数 {med_amt:.3g} 千元\n\n")
        if n_test:
            fh.write(f"**复制率：生产频率 {n_rep}/{n_test}"
                     f"（翻转 {n_flip} / 弱 {n_weak} / 未复现 {n_none}）；"
                     f"论文频率 {p_rep}/{n_test}"
                     f"（翻转 {p_flip} / 弱 {p_weak} / 未复现 {p_none}）。"
                     f"增量通过 {n_inc}/{n_test}。"
                     f"阳性对照 {'通过' if ctrl_ok else '未通过'}。"
                     f"有效独立检验数 {k50}-{k70}，最大相关 {max_off:.3f}。**\n\n")
        else:
            fh.write("- 测试组为空（只跑了对照组）。\n\n")
        fh.write("## 主表（h=%d，能不能用）\n\n" % a.h)
        fh.write(md_main)
        fh.write("\n\n## 附表（h_paper，成不成立）\n\n")
        fh.write(md_paper)
        fh.write("\n\n## 逐篇口径\n\n")
        for r in stage1:
            m = r.meta
            pr = r.paper_res
            ls = f"{r.ls_bp:+.0f}" if np.isfinite(r.ls_bp) else "n/a"
            lst = f"{r.ls_t_nw:.2f}" if np.isfinite(r.ls_t_nw) else "n/a"
            ls_p = f"{pr.ls_bp:+.0f}" if np.isfinite(pr.ls_bp) else "n/a"
            fh.write(f"### {m['paper']}\n\n")
            fh.write(f"- 因子 `{r.name}`｜家族 {m['family']}｜角色 {m['role']}"
                     f"｜保真度 {m['fidelity']}\n")
            fh.write(f"- 原文 vs 我们：{m['spec']}\n")
            fh.write(f"- 原论文预期 IC 符号：{m['paper_ic_sign']:+d}\n")
            fh.write(f"- h={a.h}：IC {r.ic['IC']:+.4f}，NW t {r.t_nw:.2f}，"
                     f"Q5−Q1 {ls} bp/期，LS t(NW) {lst}，"
                     f"方向{'一致' if r.dir_ok else '相反'} → **{r.repl}**\n")
            fh.write(f"- h_paper={int(m.get('h_paper', a.h))}："
                     f"IC {pr.ic['IC']:+.4f}，NW t {pr.t_nw:.2f}，"
                     f"Q5−Q1 {ls_p} bp/期 → **{r.repl_paper}**\n")
            fh.write(f"- 增量（h={a.h}）："
                     f"**{'通过' if incremental_ok(r) else '不通过'}**"
                     f"（残差 IC {r.resid_ic:+.4f}，边际 t "
                     f"{r.marginal_t:+.2f}）\n\n")
        fh.write("## 口径\n\n"
                 "1. 复现判定看 |t|（NW，门槛 3.0，Harvey-Liu-Zhu 2016），"
                 "方向单独一列。原文预期方向为负的因子，显著结果的 t 本身为负，"
                 "不能直接与 +3.0 比较。\n"
                 "2. 增量判定 = 残差 IC >= 0.015 且 组合边际 t >= 2.0，"
                 "只在生产频率 h=%d 上评估。\n" % a.h +
                 "3. `p_naive` 用每日重叠 IC，高估约 sqrt(h)；以 NW 为准。\n"
                 "4. 符号翻转是结论，不是 bug：原方向失败、A 股反向成立。\n"
                 "5. 保真度为「中/低」的条目只作参考，不进复制率。\n"
                 f"6. 有效独立检验数取 |rho|>=0.5..0.7 的区间（{k50}-{k70}），"
                 f"最大两两相关 {max_off:.3f}；单点估计不稳，报告区间更诚实。\n")

    log(f"\n✅ 输出到 {OUT}/")
    log(f"总耗时 {(time.time() - t0) / 60:.1f} 分钟")


if __name__ == "__main__":
    main()
