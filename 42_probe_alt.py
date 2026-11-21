#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
42_probe_alt.py
探测交易结构类接口的权限与字段。只读，不写任何数据。

用法
  .venv/bin/python 42_probe_alt.py                  # 默认用最近一个交易日
  .venv/bin/python 42_probe_alt.py --date 20251120
"""
import argparse
import os
import sys

import pandas as pd
from dotenv import load_dotenv
import tushare as ts

load_dotenv()
TOKEN = os.getenv("TUSHARE_TOKEN")
if not TOKEN:
    raise SystemExit("缺少 TUSHARE_TOKEN")

APIS = [
    ("top_list",      "龙虎榜每日明细"),
    ("top_inst",      "龙虎榜机构/营业部席位明细"),
    ("margin_detail", "融资融券明细"),
    ("block_trade",   "大宗交易"),
    ("moneyflow",     "个股资金流向"),
    ("hsgt_top10",    "沪深股通十大成交股"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="20251120")
    a = ap.parse_args()
    pro = ts.pro_api(TOKEN)
    d = a.date

    print(f"探测日期 {d}\n" + "=" * 78)
    ok_list = []

    for name, desc in APIS:
        print(f"\n【{name}】{desc}")
        try:
            df = getattr(pro, name)(trade_date=d)
        except Exception as e:
            print(f"  ❌ {type(e).__name__}: {str(e)[:180]}")
            continue

        if df is None or df.empty:
            print("  ⚠️ 返回空（可能当天无数据，或权限不足但未报错）")
            continue

        print(f"  ✅ shape={df.shape}")
        print(f"  columns={df.columns.tolist()}")
        print(df.head(3).to_string())
        ok_list.append(name)

    print("\n" + "=" * 78)
    print(f"可用接口：{ok_list if ok_list else '无'}")
    if not ok_list:
        print("→ 2000 积分可能不够这些接口，需要确认权限或换数据源")
    return 0


if __name__ == "__main__":
    sys.exit(main())
