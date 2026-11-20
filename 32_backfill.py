#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
32_backfill.py
补齐 2026-01-01 至今的日线 / 每日指标 / 复权因子。
只为让滚动窗口有历史，不做任何回放。

运维约定
  - 补数据是运维动作，不是策略动作。不要让 2026 行情影响任何参数选择。
  - **失败必须显式**：任何一天取不到 → 退出码 1，便于 launchd / CI 报警。
  - 幂等：可反复重跑，已存在的天由 pull_one_day 内部跳过。
  - 自检：三类文件在 2026 内每个目标日的行数必须 > 0。

注意
  原子写不在本文件。写盘的是 daily_run.pull_one_day，
  请在那里把 df.to_parquet(...) 换成 factorlab.io.atomic_write_parquet(...)。
  本文件只负责调度、重试、校验、退出码。

用法
  .venv/bin/python -u 32_backfill.py --dry-run     # 只看要补哪些天
  .venv/bin/python -u 32_backfill.py
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from daily_run import get_pro, pull_one_day

RAW = Path("data/raw")
KINDS = ("daily", "daily_basic", "adj_factor")
SLEEP = 5.0


def _norm_date(s: pd.Series) -> pd.Series:
    """trade_date 可能是 str / int / datetime，统一成 YYYYMMDD"""
    return s.astype(str).str.replace("-", "").str[:8]


def verify(days, year="2026"):
    """一次性校验：三类文件在目标日各有多少行。每个文件只读一列。"""
    report = {}
    for k in KINDS:
        p = RAW / f"{k}_{year}.parquet"
        if not p.exists():
            report[k] = None
            continue
        x = pd.read_parquet(p, columns=["trade_date"])
        vc = _norm_date(x["trade_date"]).value_counts()
        report[k] = {d: int(vc.get(d, 0)) for d in days}
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20260101")
    ap.add_argument("--end", default=None, help="默认今天")
    ap.add_argument("--retry", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    pro = get_pro()
    end = a.end or pd.Timestamp.today().strftime("%Y%m%d")
    cal = pro.trade_cal(exchange="SSE", start_date=a.start, end_date=end,
                        is_open="1", fields="cal_date")
    days = sorted(str(d) for d in cal.cal_date)
    if not days:
        print("区间内无交易日")
        return 0

    print(f"待补 {len(days)} 个交易日：{days[0]} → {days[-1]}")
    if a.dry_run:
        print("（dry-run，不写入）")
        return 0

    failed = []
    for i, d in enumerate(days, 1):
        ok = False
        for k in range(max(1, a.retry)):
            try:
                ok = bool(pull_one_day(pro, d))
            except Exception as e:
                print(f"  ⚠️ {d} 第 {k + 1} 次异常：{e}")
                ok = False
            if ok:
                break
            if k + 1 < a.retry:
                time.sleep(SLEEP)
        if not ok:
            failed.append(d)
            print(f"  ❌ {d} 取数失败（已重试 {a.retry} 次）")
        if i % 20 == 0:
            print(f"  {i}/{len(days)}")

    print("\n自检（按文件统计目标日行数）...")
    report = verify(days, year=days[-1][:4])
    bad = []
    for k, v in report.items():
        if v is None:
            print(f"  ❌ {k}_{days[-1][:4]}.parquet 不存在")
            bad.append(k)
            continue
        zero = [d for d, n in v.items() if n == 0]
        if zero:
            print(f"  ❌ {k}: {len(zero)} 天无数据 → {zero[:5]}")
            bad.append(k)
        else:
            print(f"  ✅ {k}: 全部 {len(v)} 天有数据")

    if failed or bad:
        print(f"\n❌ 未完成：取数失败 {len(failed)} 天（{failed[:5]}）；文件异常 {bad}")
        return 1
    print("\n✅ 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
