#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
43_pull_alt.py
拉取交易结构数据，按年分文件，可续跑，**多线程**。

为什么用线程
  SLEEP 只是下限，真实速率 = 1/(SLEEP + 延迟)。
  margin_detail 每天 4265 行，单次约 0.9 秒延迟 → 单线程只有 ~49 次/分，
  而配额是 200 次/分。3 线程约 147 次/分，安全且快 3 倍。

用法
  .venv/bin/python -u 43_pull_alt.py                      # 默认 3 线程
  .venv/bin/python -u 43_pull_alt.py --apis top_inst --workers 2
  .venv/bin/python -u 43_pull_alt.py --start 20190101
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import groupby
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
import tushare as ts

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR
from factorlab.io import atomic_write_parquet

load_dotenv()
TOKEN = os.getenv("TUSHARE_TOKEN")
if not TOKEN:
    raise SystemExit("缺少 TUSHARE_TOKEN")

OUT = DATA_DIR / "raw" / "alt"
OUT.mkdir(parents=True, exist_ok=True)

ALL_APIS = ["margin_detail", "moneyflow", "top_inst", "block_trade"]
SLEEP = 0.32
RETRY = 2


def log(m):
    print(m, flush=True)


def fetch(pro, api, d):
    """返回 (date, DataFrame | Exception)"""
    time.sleep(SLEEP)
    for k in range(RETRY):
        try:
            return d, getattr(pro, api)(trade_date=d)
        except Exception as e:
            if k == RETRY - 1:
                return d, e
            time.sleep(1.5 * (k + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apis", nargs="*", default=ALL_APIS)
    ap.add_argument("--start", default="20180101")
    ap.add_argument("--end", default="20251231")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()

    bad = [x for x in a.apis if x not in ALL_APIS]
    if bad:
        raise SystemExit(f"未知接口 {bad}；可选 {ALL_APIS}")

    pro = ts.pro_api(TOKEN)
    cal = pro.trade_cal(exchange="SSE", start_date=a.start, end_date=a.end,
                        is_open="1", fields="cal_date")
    days = sorted(str(x) for x in cal.cal_date)
    log(f"交易日 {len(days)} 天：{days[0]} → {days[-1]}")
    log(f"接口 {a.apis}；{a.workers} 线程\n")

    t0 = time.time()
    for api in a.apis:
        log(f"===== {api} =====")
        for year, grp in groupby(days, key=lambda d: d[:4]):
            ydates = list(grp)
            path = OUT / f"{api}_{year}.parquet"
            frames, have = [], set()
            if path.exists():
                old = pd.read_parquet(path)
                frames.append(old)
                have = set(old["trade_date"].astype(str).unique())

            todo = [d for d in ydates if d not in have]
            if not todo:
                log(f"  {year}: 已有 {len(have)} 天，跳过")
                continue

            n_ok = n_empty = n_fail = 0
            ty = time.time()
            with ThreadPoolExecutor(max_workers=a.workers) as ex:
                futs = [ex.submit(fetch, pro, api, d) for d in todo]
                for i, fut in enumerate(as_completed(futs), 1):
                    d, res = fut.result()
                    if isinstance(res, Exception):
                        n_fail += 1
                        log(f"    ⚠️ {d}: {str(res)[:90]}")
                    elif res is None or res.empty:
                        n_empty += 1
                    else:
                        frames.append(res)
                        n_ok += 1
                    if i % 100 == 0:
                        el = time.time() - ty
                        log(f"    {i}/{len(todo)}  成功 {n_ok}  空 {n_empty}  "
                            f"失败 {n_fail}  {i / max(el, 1e-9) * 60:.0f} 次/分")

            if frames:
                out = pd.concat(frames, ignore_index=True).drop_duplicates()
                atomic_write_parquet(out, path, index=False)
                log(f"  {year}: {len(out):,} 行 → {path.name}  "
                    f"（成功 {n_ok} / 空 {n_empty} / 失败 {n_fail}，"
                    f"{(time.time() - ty) / 60:.1f} 分钟）")
        log("")

    log(f"✅ 完成，耗时 {(time.time() - t0) / 60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    sys.exit(main())
