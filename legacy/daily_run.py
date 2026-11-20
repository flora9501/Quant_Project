#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
daily_run.py  (v3)
每日增量拉取 + 因子更新 + 目标持仓 + 模拟盘记录

用法
    .venv/bin/python daily_run.py --check
    .venv/bin/python daily_run.py
    .venv/bin/python daily_run.py --date 20251231
    .venv/bin/python daily_run.py --date 20251231 --force   # 强制调仓（验证用）

修订
  v2  补 MAD 去极值；净值改为区间累计收益；state 缺失时报错；非交易日退出
  v3  ★ 中性化截面限制在当日成分股内（与 06_preprocess 一致）
      —— 之前在全市场上做回归，与回测的 z-score 不可比，smoke test 必然失败

纪律
    2026-01 起的行情只用于运行，不用于调参。
    任何基于这些日期做出的策略修改，都会让样本外记录失效。
"""
import argparse
import json
import os
import time
from pathlib import Path
import sys
from pathlib import Path
for _p in Path(__file__).resolve().parents[:2]:
    if (_p / "config.py").exists():
        sys.path.insert(0, str(_p))
        break




import numpy as np
import pandas as pd
import tushare as ts
from dotenv import load_dotenv

from config import DATA_DIR

load_dotenv()

RAW = DATA_DIR / "raw"
REF = DATA_DIR / "ref"
PANEL = DATA_DIR / "panel"
PAPER = DATA_DIR / "paper"
PAPER.mkdir(parents=True, exist_ok=True)

FACTORS = ["rev_5", "turn_20", "vol_20", "ep"]
TOP_N = 50
REBAL = 5
THRESH = 0.25
LOOKBACK_DAYS = 120
SMOOTH_LAGS = (REBAL, 2 * REBAL)
SCORE_DAYS = max(SMOOTH_LAGS) + 2          # 需要多少天的分数（含平滑回看）

TOKEN = os.getenv("TUSHARE_TOKEN")
if not TOKEN:
    raise SystemExit("缺少 TUSHARE_TOKEN，检查 .env")


def log(m):
    print(m, flush=True)


# ==================== 依赖检查 ====================
def check_inputs():
    log("=" * 62)
    log("依赖检查")
    log("=" * 62)
    fatal = True

    required = {
        "日线": sorted(RAW.glob("daily_[0-9][0-9][0-9][0-9].parquet")),
        "每日指标": sorted(RAW.glob("daily_basic_[0-9][0-9][0-9][0-9].parquet")),
        "复权因子": sorted(RAW.glob("adj_factor_[0-9][0-9][0-9][0-9].parquet")),
        "股票基础": sorted(REF.glob("stock_basic*.parquet")),
        "退市清单": sorted(REF.glob("stock_delisted*.parquet")),
    }
    optional = {
        "指数权重": sorted(RAW.glob("index_weight*.parquet")) + sorted(REF.glob("index_weight*.parquet")),
    }

    for k, v in required.items():
        log(f"  {k:<8} {len(v)} 个文件" if v else f"  {k:<8} ❌ 缺失")
        fatal = fatal and bool(v)
    for k, v in optional.items():
        log(f"  {k:<8} {len(v)} 个文件" if v else f"  {k:<8} ⚠️  缺失（v1 不依赖，可忽略）")

    for p in ["tradable.parquet", "factors_neutral.parquet"]:
        f = PANEL / p
        log(f"  {p:<26} {'✅' if f.exists() else '❌ 缺失'}")
        fatal = fatal and f.exists()

    if required["日线"]:
        last = pd.read_parquet(required["日线"][-1], columns=["trade_date"])
        log(f"\n  本地最新交易日: {last.trade_date.max()}")

    if required["股票基础"]:
        sb = pd.read_parquet(required["股票基础"][0], columns=["industry"])
        log(f"  在市股票行业标签数: {sb.industry.nunique()}")

    if (PANEL / "tradable.parquet").exists():
        tr = pd.read_parquet(PANEL / "tradable.parquet")
        log(f"  日线面板宽度 {len(tr.columns)}；"
            f"曾进过成分股 {int(tr.any(axis=0).sum())} 只；"
            f"日均在册 {tr.sum(axis=1).mean():.0f} 只")

    st = PAPER / "state.json"
    if st.exists():
        s = json.loads(st.read_text())
        log(f"  state.json: last_rebalance={s.get('last_rebalance')}, "
            f"holdings={len(s.get('holdings', []))}")
    else:
        log("  state.json: ⚠️  不存在（首次运行前需先跑 31 播种）")

    log("=" * 62)
    log("✅ 依赖齐全" if fatal else "❌ 有缺失，先补齐再跑")
    return fatal


# ==================== 工具 ====================
def get_pro():
    return ts.pro_api(TOKEN)


def is_trading_day(pro, d):
    try:
        cal = pro.trade_cal(exchange="SSE", start_date=d, end_date=d,
                            fields="cal_date,is_open")
        return (not cal.empty) and int(cal.is_open.iloc[0]) == 1
    except Exception:
        return True


def panel_dates():
    ds = set()
    for f in sorted(RAW.glob("daily_[0-9][0-9][0-9][0-9].parquet")):
        ds |= set(pd.read_parquet(f, columns=["trade_date"]).trade_date.unique())
    return sorted(ds)


def latest_trade_date(pro):
    cal = pro.trade_cal(exchange="SSE",
                        end_date=pd.Timestamp.today().strftime("%Y%m%d"),
                        is_open="1", fields="cal_date")
    return cal.cal_date.max()


# ==================== 增量拉取 ====================
def pull_one_day(pro, d):
    year = d[:4]
    specs = [
        ("daily", ["ts_code", "trade_date", "close", "pct_chg", "amount"]),
        ("daily_basic", ["ts_code", "trade_date", "turnover_rate", "circ_mv", "pe_ttm"]),
        ("adj_factor", ["ts_code", "trade_date", "adj_factor"]),
    ]
    got_any = False

    for api, fields in specs:
        path = RAW / f"{api}_{year}.parquet"
        old = pd.read_parquet(path) if path.exists() else pd.DataFrame()

        if not old.empty and d in set(old.trade_date.unique()):
            log(f"   {api} {d} 已存在，跳过")
            got_any = True
            continue

        try:
            df = getattr(pro, api)(trade_date=d, fields=",".join(fields))
        except Exception as e:
            log(f"   ❌ {api} {d} 拉取失败: {repr(e)[:120]}")
            continue

        if df is None or df.empty:
            log(f"   ⚠️  {api} {d} 返回空")
            continue

        out = pd.concat([old, df[fields]], ignore_index=True) if not old.empty else df[fields]
        out = out.drop_duplicates(["ts_code", "trade_date"])
        out.to_parquet(path, index=False)
        log(f"   {api} {d} 新增 {len(df)} 行 → {path.name}")
        got_any = True

    return got_any


# ==================== 窗口数据 ====================
def load_window(dates):
    keep = set(dates[-LOOKBACK_DAYS:])
    years = {d[:4] for d in keep}
    dparts, bparts = [], []

    for y in sorted(years):
        f = RAW / f"daily_{y}.parquet"
        if f.exists():
            x = pd.read_parquet(f, columns=["ts_code", "trade_date", "pct_chg"])
            dparts.append(x[x.trade_date.isin(keep)])
        f = RAW / f"daily_basic_{y}.parquet"
        if f.exists():
            x = pd.read_parquet(f, columns=["ts_code", "trade_date",
                                            "turnover_rate", "circ_mv", "pe_ttm"])
            bparts.append(x[x.trade_date.isin(keep)])

    return pd.concat(dparts, ignore_index=True), pd.concat(bparts, ignore_index=True)


def to_wide(df, col, dates, codes):
    return (df.pivot(index="trade_date", columns="ts_code", values=col)
              .reindex(index=dates, columns=codes))


# ==================== 成分股 ====================
def build_universe_map(dates_list):
    """每个交易日对应的成分股集合；无当日快照则沿用最近一次"""
    tr = pd.read_parquet(PANEL / "tradable.parquet")
    if isinstance(tr.index, pd.DatetimeIndex):
        tr.index = tr.index.strftime("%Y%m%d")
    keys = sorted(tr.index)

    out = {}
    for d in dates_list:
        avail = [x for x in keys if x <= d]
        if not avail:
            out[d] = set()
            continue
        row = tr.loc[avail[-1]]
        out[d] = set(row[row.astype(bool)].index)
    return out


# ==================== 因子 ====================
def mad_winsorize(s, n=3):
    med = s.median()
    mad = (s - med).abs().median() * 1.4826
    if not np.isfinite(mad) or mad == 0:
        return s
    return s.clip(med - n * mad, med + n * mad)


def build_scores(dates, daily, basic, industry, uni_map):
    """
    顺序严格对齐 06_preprocess_factors.py：
        限制到当日成分股 → MAD 去极值 → 对行业+市值回归 → z-score
    """
    codes = sorted(set(daily.ts_code) | set(basic.ts_code))
    ret = (to_wide(daily, "pct_chg", dates, codes) / 100.0).fillna(0.0)
    turn = to_wide(basic, "turnover_rate", dates, codes)
    cmv = to_wide(basic, "circ_mv", dates, codes)
    pe = to_wide(basic, "pe_ttm", dates, codes)

    raw = {
        "rev_5": -ret.rolling(5, min_periods=5).sum(),
        "turn_20": -turn.rolling(20, min_periods=20).mean(),
        "vol_20": -ret.rolling(20, min_periods=20).std() * np.sqrt(252),
        "ep": 1.0 / pe.where(pe > 0),
    }
    lnmv = -np.log(cmv * 1e4)
    ind_all = industry.reindex(codes).fillna("UNKNOWN")

    score_rows = {}
    for d in dates[-SCORE_DAYS:]:
        uni = uni_map.get(d, set())
        mask_codes = [c for c in uni if c in set(codes)]
        if len(mask_codes) < 100:
            score_rows[d] = pd.Series(np.nan, index=codes)
            continue

        keep = pd.Series(False, index=codes)
        keep[ mask_codes ] = True

        parts = []
        for n in FACTORS:
            y = raw[n].loc[d].where(keep)        # ★ 先进截面
            if y.notna().sum() < 100:
                parts.append(pd.Series(np.nan, index=codes))
                continue

            y = mad_winsorize(y)                 # ★ 再 winsorize
            lm = lnmv.loc[d]
            ok = y.notna() & lm.notna()

            X = pd.get_dummies(ind_all[ok], drop_first=True).to_numpy(dtype=float)
            X = np.c_[np.ones(int(ok.sum())), lm[ok].to_numpy(dtype=float), X]
            yy = y[ok].to_numpy(dtype=float)

            beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
            res = yy - X @ beta
            sd = res.std()

            z = pd.Series(np.nan, index=codes)
            z[ok] = (res - res.mean()) / (sd if sd > 0 else 1.0)
            parts.append(z)

        tot = sum(p.fillna(0.0) for p in parts)
        valid = sum(p.notna().astype(int) for p in parts) > 0
        score_rows[d] = (tot / len(FACTORS)).where(valid)


    return pd.DataFrame(score_rows).T.sort_index()


def smooth_score(scores):
    acc = scores.fillna(0.0).copy()
    cnt = scores.notna().astype(int).copy()
    for k in SMOOTH_LAGS:
        sh = scores.shift(k)
        acc = acc + sh.fillna(0.0)
        cnt = cnt + sh.notna().astype(int)
    return acc / cnt.replace(0, np.nan)


# ==================== 股票池 ====================
def load_universe(d):
    tr = pd.read_parquet(PANEL / "tradable.parquet")
    if isinstance(tr.index, pd.DatetimeIndex):
        tr.index = tr.index.strftime("%Y%m%d")
    avail = [x for x in tr.index if x <= d]
    if not avail:
        raise RuntimeError(f"tradable.parquet 里没有 {d} 及之前的快照")

    snap = avail[-1]
    if snap < d:
        log(f"   ⚠️  股票池快照 {snap} 早于目标日 {d}（本地无更新快照）")

    row = tr.loc[snap]
    return row[row.astype(bool)].index.tolist(), snap


# ==================== 选股 ====================
def select_with_quota(s, universe, industry, held, sigma_mult=THRESH):
    s = s.reindex(universe).dropna()
    if len(s) < TOP_N:
        return []

    df = pd.DataFrame({"score": s,
                       "ind": industry.reindex(s.index).fillna("UNKNOWN")}).dropna()

    share = df.groupby("ind").size() / len(df)
    raw = share * TOP_N
    base = np.floor(raw).astype(int)
    remain = TOP_N - int(base.sum())
    if remain > 0:
        for key in list((raw - base).sort_values(ascending=False).index)[:remain]:
            base.loc[key] += 1

    delta = sigma_mult * float(df["score"].std())
    picks = []
    for ind_, k in base.items():
        if k <= 0:
            continue
        pool = df[df["ind"] == ind_].sort_values("score", ascending=False)
        if pool.empty:
            continue
        cut = float(pool["score"].iloc[min(int(k), len(pool)) - 1])
        kept = [c for c in held if c in pool.index and float(pool.loc[c, "score"]) >= cut - delta]
        kept = sorted(kept, key=lambda c: -float(pool.loc[c, "score"]))[:int(k)]
        fill = [c for c in pool.index if c not in set(kept)][:int(k) - len(kept)]
        picks.extend(kept + fill)

    picks = list(dict.fromkeys(picks))[:TOP_N]
    if len(picks) < TOP_N:
        rest = df[~df.index.isin(set(picks))].nlargest(TOP_N - len(picks), "score")
        picks += rest.index.tolist()
    return list(dict.fromkeys(picks))[:TOP_N]


# ==================== 状态 ====================
def load_state():
    f = PAPER / "state.json"
    return json.loads(f.read_text()) if f.exists() else None


def save_state(st):
    (PAPER / "state.json").write_text(json.dumps(st, ensure_ascii=False, indent=2))


def portfolio_return_since(held, ret_wide, d0, d1):
    between = [x for x in ret_wide.index if d0 < x <= d1]
    if not between:
        return 0.0
    cols = [c for c in held if c in ret_wide.columns]
    if not cols:
        return 0.0
    daily = ret_wide.loc[between, cols].fillna(0.0).mean(axis=1)
    return float((1 + daily).prod() - 1)


# ==================== 主流程 ====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--date", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.check:
        raise SystemExit(0 if check_inputs() else 1)

    t0 = time.time()
    pro = get_pro()

    d = args.date or latest_trade_date(pro)
    log(f"① 目标交易日 {d}")

    if not is_trading_day(pro, d):
        log(f"   {d} 非交易日，退出")
        return

    if not pull_one_day(pro, d):
        log("   拉取失败或返回空（可能数据尚未更新），退出")
        return

    dates_all = panel_dates()
    if d not in dates_all:
        raise SystemExit(f"{d} 不在本地交易日序列里")

    # ★ 窗口截止到目标日：补了 2026 数据后，dates 的尾部不是 d
    dates = [x for x in dates_all if x <= d]

    st = load_state()
    if st is None:
        raise SystemExit(
            "\nstate.json 不存在。首次运行前必须先播种：\n"
            "    .venv/bin/python 31_export_replay_holdings.py\n"
            "否则 hysteresis 没有历史持仓可依，选股会退化为纯配额选股。"
        )
    held = st.get("holdings", [])
    if not held:
        raise SystemExit("state.json 里 holdings 为空，重跑 31_export_replay_holdings.py")

    # ---- 窗口数据 ----
    daily, basic = load_window(dates)
    ret_wide = (to_wide(daily, "pct_chg", dates, sorted(daily.ts_code.unique()))
                / 100.0).fillna(0.0)

    # ---- 净值：先按旧持仓结算到今日 ----
    log("② 更新模拟盘净值...")
    last_date = st.get("last_date")
    if last_date and last_date < d:
        r = portfolio_return_since(held, ret_wide, last_date, d)
        st["nav"] = float(st.get("nav", 1.0) * (1 + r))
        navf = PAPER / "nav.csv"
        pd.DataFrame([{"date": d, "ret": r, "nav": st["nav"]}]).to_csv(
            navf, mode="a", header=not navf.exists(), index=False)
        log(f"   自 {last_date} 累计 {r:+.2%}，模拟净值 {st['nav']:.4f}")
    else:
        log("   今日已结算或首次运行，跳过")

    # ---- 因子 ----
    log("③ 计算因子并中性化（截面 = 当日成分股）...")
    sb = pd.concat([pd.read_parquet(REF / "stock_basic.parquet"),
                    pd.read_parquet(REF / "stock_delisted.parquet")],
                   ignore_index=True).drop_duplicates("ts_code")
    industry = sb.set_index("ts_code")["industry"].fillna("UNKNOWN")

    uni_map = build_universe_map(dates[-SCORE_DAYS:])
    scores = build_scores(dates, daily, basic, industry, uni_map)
    smooth = smooth_score(scores)
    if d not in smooth.index:
        raise SystemExit("平滑分数缺失，检查回看窗口")

    # ---- 股票池 ----
    log("④ 股票池...")
    universe, snap = load_universe(d)
    log(f"   快照 {snap}，{len(universe)} 只")

    # ---- 调仓判断 ----
    last_reb = st.get("last_rebalance")
    pos_now = dates.index(d)
    gap = None if not last_reb else pos_now - dates.index(last_reb)
    need_rebal = args.force or last_reb is None or (gap is not None and gap >= REBAL)
    log(f"   上次调仓 {last_reb}，间隔 {gap if gap is not None else '—'}；"
        f"{'需要调仓' if need_rebal else '维持持仓'}")

    if need_rebal:
        log("⑤ 生成目标持仓...")
        picks = select_with_quota(smooth.loc[d], universe, industry, held)
        if not picks:
            raise SystemExit("选股结果为空")

        replaced = len(set(picks) - set(held))
        log(f"   {len(picks)} 只，替换 {replaced} 只（换手 {replaced / TOP_N:.0%}）")

        tbl = pd.DataFrame({"ts_code": picks, "weight": 1.0 / len(picks)})
        tbl.to_csv(PAPER / f"target_{d}.csv", index=False)

        hlog = PAPER / "holdings_log.parquet"
        row = tbl.assign(date=d)
        if hlog.exists():
            row = pd.concat([pd.read_parquet(hlog), row], ignore_index=True)
        row.to_parquet(hlog, index=False)

        st["last_rebalance"] = d
        st["holdings"] = picks
        log(f"   → paper/target_{d}.csv")

    st["last_date"] = d
    save_state(st)

    log(f"\n✅ 完成，耗时 {(time.time() - t0) / 60:.1f} 分钟")
    log("\n若今天改过任何策略参数，请在 RESEARCH_LOG 追加一条：")
    log("  | 日期 | 修改内容 | 理由 | 影响哪些历史结果 |")


if __name__ == "__main__":
    main()
