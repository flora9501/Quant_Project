#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
10_pull_fina.py  (v2 并发版)
串行 27 次/分钟 -> 并发跑满账户上限 ~185 次/分钟。

约束（2000 积分档）：
  单次 100 行 / ts_code 必填 / 200 次每分钟
"""
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import tushare as ts
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("TUSHARE_TOKEN")

OUT = Path("data/raw_fina")
OUT.mkdir(parents=True, exist_ok=True)

START, END = "20170101", "20251231"
PAGE_LIMIT = 100
RATE_PER_MIN = 185        # 官方 200，留余量防突发
WORKERS = 8               # 8 × (1/2.2s) ≈ 218/min，被限流器压到 185
FLUSH_EVERY = 400

FIELDS = {
    "income":       "ts_code,ann_date,f_ann_date,end_date,report_type,update_flag,n_income,n_income_attr_p",
    "cashflow":     "ts_code,ann_date,f_ann_date,end_date,report_type,update_flag,n_cashflow_act",
    "balancesheet": "ts_code,ann_date,f_ann_date,end_date,report_type,update_flag,comp_type,total_assets",
}


class RateLimiter:
    """全局令牌间隔：线程共享，保证总速率不超过账户上限。"""

    def __init__(self, per_min):
        self.interval = 60.0 / per_min
        self.lock = threading.Lock()
        self.next_at = time.time()

    def acquire(self):
        with self.lock:
            now = time.time()
            if self.next_at < now:
                self.next_at = now
            wait = self.next_at - now
            self.next_at += self.interval
        if wait > 0:
            time.sleep(wait)


limiter = RateLimiter(RATE_PER_MIN)
_tl = threading.local()


def get_pro():
    """每个线程一个 client：requests.Session 不是严格线程安全的。"""
    if not hasattr(_tl, "p"):
        try:
            _tl.p = ts.pro_api(TOKEN, timeout=30)
        except TypeError:          # 老版本 tushare 不支持 timeout 参数
            _tl.p = ts.pro_api(TOKEN)
    return _tl.p


def fetch(api, code, fields, retry=3):
    rows, offset = [], 0
    while True:
        df = None
        for attempt in range(retry):
            limiter.acquire()
            try:
                df = getattr(get_pro(), api)(
                    ts_code=code, start_date=START, end_date=END,
                    report_type="1", fields=fields,
                    offset=offset, limit=PAGE_LIMIT,
                )
                break
            except Exception:
                if attempt == retry - 1:
                    raise
                time.sleep(2 * (attempt + 1))
        if df is None or df.empty:
            break
        rows.append(df)
        if len(df) < PAGE_LIMIT:
            break
        offset += PAGE_LIMIT
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_codes():
    pan = pd.read_parquet("data/panel/tradable.parquet")
    codes = sorted({str(c) for c in pan.columns if str(c).endswith((".SH", ".SZ", ".BJ"))})
    assert len(codes) > 4000, f"列名解析异常: {list(pan.columns)[:5]}"
    return codes


def run(api, fields, codes):
    parts = sorted(OUT.glob(f"{api}_part*.parquet"))
    done = set()
    for p in parts:
        done |= set(pd.read_parquet(p, columns=["ts_code"]).ts_code.astype(str))

    todo = [c for c in codes if c not in done]
    print(f"[{api}] 已完成 {len(done)}，待拉 {len(todo)}", flush=True)
    if not todo:
        return

    seq = max([0] + [int(p.stem.split("part")[-1]) for p in parts])
    buf, failed = [], []
    t0 = time.time()

    def flush():
        nonlocal buf, seq
        if not buf:
            return
        seq += 1
        path = OUT / f"{api}_part{seq:05d}.parquet"
        pd.concat(buf, ignore_index=True).to_parquet(path, index=False)
        print(f"  落盘 {path.name}  rows={sum(len(b) for b in buf)}", flush=True)
        buf = []

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(fetch, api, c, fields): c for c in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            code = futs[fut]
            try:
                df = fut.result()
                if not df.empty:
                    buf.append(df)
            except KeyboardInterrupt:
                flush()
                raise
            except Exception as e:
                failed.append(code)
                print(f"  FAIL {code}: {repr(e)[:80]}", flush=True)

            if i % FLUSH_EVERY == 0:
                flush()
                elapsed = (time.time() - t0) / 60
                print(f"  [{api}] {i}/{len(todo)}  {elapsed:.0f}min  "
                      f"ETA {elapsed / i * (len(todo) - i):.0f}min", flush=True)

    flush()

    if failed:
        path = OUT / f"failures_{api}.txt"
        old = set(path.read_text().split()) if path.exists() else set()
        path.write_text("\n".join(sorted(old | set(failed))))
        print(f"[{api}] 失败 {len(failed)} 只 -> {path}", flush=True)


def main():
    codes = load_codes()
    total = len(codes) * 3
    print(f"股票池 {len(codes)} 只，预计 {total} 次调用，"
          f"约 {total / RATE_PER_MIN:.0f} 分钟", flush=True)
    for api, fields in FIELDS.items():
        run(api, fields, codes)
    print("全部完成。", flush=True)


if __name__ == "__main__":
    main()
