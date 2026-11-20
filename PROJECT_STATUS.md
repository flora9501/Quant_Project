# PROJECT STATUS

> 项目交接文档。换人接手或重开会话时先读这里，30 秒可恢复上下文。
> 最后更新：2025-11-20（Phase 4 完成，Phase 5 进行中）

## 环境

- macOS，Python 3.13，venv 在 `.venv`
- 终端提示符常为 `(.venv) (base)`：conda base 同时存在，裸敲 `python` 可能落到 base（没有 pandas）。
  **一律用 `.venv/bin/python`。**
- Tushare Pro **2000 积分**（200 元/年）：财务三表 100 行/次、限 200 次/分钟、无 `_vip`。
- 依赖：tushare / pandas / numpy / pyarrow / python-dotenv / statsmodels / matplotlib

## 最终数字（唯一有效口径：双边成本 + buy-and-hold）

| 配置 | 换手 | 毛利 | 成本 | 净收益 | 超额 | 行业偏离 |
|---|---:|---:|---:|---:|---:|---:|
| EW Baseline | 2437% | 13.75% | 5.86% | 7.29% | 2.01% | 17.50% |
| EW + 门槛 | 619% | 10.77% | 1.49% | 9.14% | 3.77% | 16.60% |
| Strat Baseline | 2514% | 11.96% | 6.05% | 5.41% | 0.22% | 1.35% |
| **Strat + 门槛** | **714%** | 11.54% | **1.72%** | **9.65%** | **4.25%** | **1.35%** |

容量约 **2.4 亿**（1 亿 +1.91%，2 亿 +0.42%，3 亿 −0.71%）。

**归因**

| | 行业未控 | 行业已控 |
|---|---:|---:|
| alpha（年化） | **−4.45%** | **−6.85%** |
| alpha t | **−2.63** | **−4.28** |
| value β | 0.2979 | 0.2813 |
| low-vol β | 0.2774 | 0.3281 |
| R² | 0.526 | 0.595 |

残差化组合：净 −5.33%、超额 −9.71%。**alpha 显著为负，不是“没有 alpha”。**

**风险归因（30 号脚本，5 日 Brinson，384 期）**

| | 行业配置 | 行业内选股 | 交互 |
|---|---:|---:|---:|
| TE 贡献 | 1.44% | **6.17%** | 1.71% |
| 方差占比 | 4.3% | **78.5%** | 6.0% |
| 收益贡献 | +0.30% | **+4.04%** | +0.81% |

**TE 6.97%**（组合波动 23.17%，基准 25.52%）；5 日胜率 52.1%。
结论：风险与收益都来自选股；行业约束几乎免费；**QP 无空间，明确不做**。

## 目录与关键数据形状

| 路径 | 内容 | 形状 / 注意 |
|---|---|---|
| `data/raw/daily_YYYY.parquet` | 日线 | 含 `pct_chg` / `amount`；glob 用 `daily_????` |
| `data/raw/daily_basic_YYYY.parquet` | 每日指标 | turnover_rate / circ_mv / pe_ttm |
| `data/raw/adj_factor_YYYY.parquet` | 复权因子 | — |
| `data/raw_fina/{api}_part*.parquet` | 财务三表原始 | 保留 update_flag 0/1 重复行 |
| `data/ref/` | stock_basic / stock_delisted / namechange | **两者都带 industry** |
| `data/panel/tradable.parquet` | **宽表** | `(1921, 5696)`，index = `YYYYMMDD` **字符串** |
| `data/panel/factors.parquet` | **长表** | `(1902858, 8)`，trade_date 为 **str** |
| `data/panel/factors_neutral.parquet` | 中性化后 | 07/08/09/19…30 都读它 |
| `data/panel/factor_accruals.parquet` | Phase 2a | `(1827435, 7)` |
| `data/panel/factor_pead.parquet` | Phase 2b | `(1119547, 7)` |
| `paper/` | Phase 5 运行态 | **脚本自动创建**，不需要手动 mkdir |

## 脚本状态

| 脚本 | 状态 |
|---|---|
| 01–06 | ✅ universe 1921 天/日均 991；factors 1,902,858 行；中性化 IC 0.017→0.035 |
| 07_backtest | ✅ 5 变体（旧口径，仅作历史对照） |
| 08 / 09 | ✅ 双边口径：alpha −4.45% (t=−2.63) / −6.85% (t=−4.28) |
| 10–12 | ✅ 应计项，REJECT |
| 14–18 | ✅ PEAD/SUE，REJECT |
| 19–28 | ✅ 换手控制、冲击成本、口径统一 |
| 30_risk_attribution | ✅ TE 6.97%，选股占 78.5% |
| daily_run.py | 🔄 **进行中**：依赖检查 + 增量拉取 + 目标持仓 + 模拟盘 |

## 核心结论（五条）

1. **Phase 1**：smart beta，**显著负 alpha**（−4.45% / −6.85%）。
2. **Phase 2a 应计项**：相关性 0.054，但 IC 0.0093、两年为负 → 拒绝。**信号太弱。**
3. **Phase 2b PEAD/SUE**：截面全过，组合不显著（t = −0.27 / +0.66）→ 拒绝。**信号期与调仓期错配。**
4. **Phase 3 换手控制**：换手 2514% → 714%，超额 0.22% → **4.25%**。成本效应 +8.03pp、信号效应 +2.57pp。
5. **Phase 4 风险归因**：TE 6.97%，78.5% 来自选股、4.3% 来自行业。**QP 无空间。**

## 样本外说明

2018–2025 全样本已被反复使用，**不构成样本外**。
**2026-01 起的数据只允许运行，不允许用于调参。** `paper/nav.csv` 就是 OOS 记录，任何基于这些日期做出的策略修改都会让它失效。

## 下一步

| 优先级 | 事项 | 说明 |
|---|---|---|
| 1 | 跑通 `daily_run.py --check` | 确认依赖齐全；若只有“指数权重”缺失可忽略 |
| 2 | 用历史日期 smoke test | `--date 20251231`，与 30 号脚本同日持仓比对 |
| 3 | 设 launchd / cron | 每日 18:30 自动跑 |
| 4 | 积累 `paper/nav.csv` | 这就是 OOS 记录 |
| — | 不再尝试 | 应计项、PEAD、QP |

## 已知坑（接手必读）

1. **`.venv/bin/python`**，别用裸 `python`。
2. `tradable.parquet` 是**宽表**，代码在 `columns` 上。
3. `glob('daily*.parquet')` 会匹配 `daily_basic_*`，必须 `daily_????`。
4. `merge_asof` 要求 **`on` 键全局单调**，排序时 on 键放第一位。
5. `Series.reset_index()` 产生 **3 列**（两个索引层 + 值列）。
6. 字符串日期取月份是 **`[4:6]`**，`[5:7]` 全 NaN 且不报错。
7. `update_flag` 的 NaN 转 str 变 `'nan'`，排序时排在 `'1'` 前 → 先 `fillna('0')`。
8. **启动长任务前 `ps aux | grep 脚本名`**：曾有两个进程同时拉 income，重复率 47%。
9. `.BJ` 的 `pct_chg` 有垃圾值（max 200,400%，占 3.0%），全市场计算要滤 `.SH` / `.SZ`。
10. **日期类型有两种**：`tradable`/新因子表是 datetime64，`factors`/`daily` 是 str。
11. **覆盖率分母要写清**：SUE 10.2% 是全网格口径，在册口径 58.8%。
12. **`combine()` 是 fillna(0) 后相加**，不是取均值。
13. **`int(0.0012 * 1e4)` = 11**，别用 `int()` 把浮点转标签。
14. **成本必须双边计费**：`Σ|Δw| × c`，不是 `单边换手 × c`。
15. **换手率汇报用 `to.replace(0, np.nan).mean()`**，否则被 0 摊薄 5 倍。
16. **宽表 `.mean()/.std()` 默认按列聚合**，行列搞反不报错，只会给出错位的索引 → 聚合前断言列名集合。
17. `paper/` 目录由脚本自动创建，不需要手动 `mkdir`。

## 一句话交接

# PROJECT STATUS

> 项目交接文档。换人接手或重开会话时先读这里，30 秒可恢复上下文。
> 最后更新：2025-12-31（Phase 5 建成并验证，等待 2026 样本外）

## 环境

- macOS，Python 3.13，venv 在 `.venv`
- 终端提示符常为 `(.venv) (base)`：conda base 同时存在，裸敲 `python` 可能落到 base（没有 pandas）。
  **一律用 `.venv/bin/python`。**
- Tushare Pro **2000 积分**（200 元/年）：财务三表 100 行/次、限 200 次/分钟、无 `_vip`。
- 依赖：tushare / pandas / numpy / pyarrow / python-dotenv / statsmodels / matplotlib

## 最终数字（唯一有效口径：双边成本 + buy-and-hold + 3 期平滑分数）

| 配置 | 换手 | 毛利 | 成本 | 净收益 | 超额 | 行业偏离 |
|---|---:|---:|---:|---:|---:|---:|
| EW Baseline | 2437% | 13.75% | 5.86% | 7.29% | 2.01% | 17.50% |
| EW + 门槛 | 619% | 10.77% | 1.49% | 9.14% | 3.77% | 16.60% |
| Strat Baseline | 2514% | 11.96% | 6.05% | 5.41% | 0.22% | 1.35% |
| **Strat + 门槛** | **714%** | 11.54% | **1.72%** | **9.65%** | **4.25%** | **1.35%** |

容量约 **2.4 亿**（1 亿 +1.91%，2 亿 +0.42%，3 亿 −0.71%）。

**归因**

| | 行业未控 | 行业已控 |
|---|---:|---:|
| alpha（年化） | **−4.45%** | **−6.85%** |
| alpha t | **−2.63** | **−4.28** |
| value β | 0.2979 | 0.2813 |
| low-vol β | 0.2774 | 0.3281 |
| R² | 0.526 | 0.595 |

残差化组合：净 −5.33%、超额 −9.71%。**alpha 显著为负。**

**风险归因（30 号脚本，5 日 Brinson，384 期）**

| | 行业配置 | 行业内选股 | 交互 |
|---|---:|---:|---:|
| TE 贡献 | 1.44% | **6.20%** | 1.71% |
| 方差占比 | 4.2% | **77.2%** | 5.9% |
| 收益贡献 | +0.30% | **+4.05%** | +0.70% |

**TE 7.06%**（组合波动 22.67%，基准 25.52%）；5 日胜率 53.4%。
**QP 无空间，明确不做。**

行业层面：选股在软件服务(+0.72)/食品(+0.69)/半导体(+0.65) 最有效，
在电气设备(−0.33) 最差——而电气设备正是未约束时超配 8.99% 的那个行业。
**行业约束解决了超配，但解决不了行业内选股失效。**

## 目录与关键数据形状

| 路径 | 内容 | 注意 |
|---|---|---|
| `data/raw/daily_YYYY.parquet` | 日线 | 含 `pct_chg` / `amount`；glob 用 `daily_????` |
| `data/raw/daily_basic_YYYY.parquet` | 每日指标 | turnover_rate / circ_mv / pe_ttm |
| `data/raw/adj_factor_YYYY.parquet` | 复权因子 | — |
| `data/raw_fina/{api}_part*.parquet` | 财务三表原始 | 保留 update_flag 0/1 重复行 |
| `data/ref/` | stock_basic / stock_delisted / namechange | **两者都带 industry** |
| `data/panel/tradable.parquet` | **宽表** | `(1921, 5696)`；5696 是面板宽度，**真实标的 2364 只** |
| `data/panel/factors.parquet` | **长表** | `(1902858, 8)`，trade_date 为 **str** |
| `data/panel/factors_neutral.parquet` | 中性化后 | 全项目读它 |
| `paper/` | Phase 5 运行态 | **脚本自动创建** |

## 脚本状态

| 脚本 | 状态 |
|---|---|
| 01–06 | ✅ universe 1921 天 / 日均 991 / 累计 2364 只；中性化 IC 0.017→0.035 |
| 07_backtest | ✅ 5 变体（旧口径，仅作历史对照） |
| 08 / 09 | ✅ 双边口径：alpha −4.45% (t=−2.63) / −6.85% (t=−4.28) |
| 10–12 | ✅ 应计项，REJECT |
| 14–18 | ✅ PEAD/SUE，REJECT |
| 19–28 | ✅ 换手控制、冲击成本、口径统一 |
| 30_risk_attribution | ✅ TE 7.06%，选股占 77.2% |
| 31_export_replay_holdings | ✅ 平滑分数回放 + state 播种 |
| daily_run.py | ✅ 与回测**逐只一致（50/50）** |

## 核心结论

1. **Phase 1**：smart beta，**显著负 alpha**（−4.45% / −6.85%）。
2. **Phase 2a 应计项**：相关性 0.054，但 IC 0.0093、两年为负 → 拒绝。
3. **Phase 2b PEAD/SUE**：截面全过，组合不显著 → 拒绝。
4. **Phase 3 换手控制**：换手 2514% → 714%，超额 0.22% → **4.25%**；成本效应 +8.03pp、信号效应 +2.57pp。
5. **Phase 4 风险归因**：TE 7.06%，77.2% 来自选股、4.2% 来自行业。
6. **Phase 5 增量管线**：已验证与全量回测 50/50 一致。

## 样本外说明

2018–2025 全样本已被反复使用，**不构成样本外**。
**2026-01 起的 `paper/nav.csv` 是唯一的样本外记录。** Data 之后只允许运行，不允许用于调参。

## 下一步

| 优先级 | 事项 | 说明 |
|---|---|---|
| 1 | 定时任务 | `launchctl load ~/Library/LaunchAgents/com.wangfang.dailyrun.plist` |
| 2 | 每日检查 `data/paper/run.log` | 确认脚本在跑 |
| 3 | 积累 `paper/nav.csv` | OOS 记录，不可回看调参 |
| — | 不再尝试 | 应计项、PEAD、QP |

## 已知坑（接手必读）

1. **`.venv/bin/python`**，别用裸 `python`。
2. `tradable.parquet` 是**宽表**；`len(columns)=5696` 是面板宽度，**真实标的 2364 只**（用 `mask.any(axis=0).sum()`）。
3. `glob('daily*.parquet')` 会匹配 `daily_basic_*`，必须 `daily_????`。
4. `merge_asof` 要求 **`on` 键全局单调**，on 键放排序第一位。
5. `Series.reset_index()` 产生 **3 列**。
6. 字符串日期取月份是 **`[4:6]`**。
7. `update_flag` 的 NaN 转 str 变 `'nan'` → 先 `fillna('0')`。
8. **启动长任务前 `ps aux | grep 脚本名`**。
9. `.BJ` 的 `pct_chg` 有垃圾值，全市场计算要滤 `.SH` / `.SZ`。
10. **日期类型有两种**：`tradable`/新因子表是 datetime64，`factors`/`daily` 是 str。
11. **覆盖率分母要写清**（全网格 vs 在册）。
12. `combine()` 是 `fillna(0)` 后相加，不是取均值。
13. **`int(0.0012 * 1e4)` = 11**，浮点转标签用整数常量。
14. **成本必须双边计费**。
15. **换手率用 `to.replace(0, np.nan).mean()`**。
16. 宽表 `.mean()/.std()` 默认按列聚合，**聚合前断言列名集合**。
17. `paper/` 由脚本自动创建。
18. **同一策略的 `score` 构造不要在多处各写一遍**：曾出现 28 用平滑、31/30 用原始分数，交集只有 36/50。
19. macOS 上 `cron` 访问 `~/Desktop` 会被 TCC 拦截，**用 launchd**。

## 一句话交接

**显著负 alpha 的 smart-beta 组合；换手控制把 12bp 下超额从 0.22% 做到 4.25%，TE 7.06%、77.2% 来自选股，容量约 2.4 亿。增量管线已与回测逐只一致，2026 年起进入真正的样本外。**
