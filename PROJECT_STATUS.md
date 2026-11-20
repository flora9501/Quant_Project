# PROJECT STATUS

> 项目交接文档。换人接手或重开会话时先读这里，30 秒可恢复上下文。
> 最后更新：2025-11-20（Phase 3 + 归因重算完成，全部数字为双边成本口径）

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
基线构造在任何 AUM 下超额为负，不可投资。

**归因（08/09 重算后）**

| | 行业未控 | 行业已控 |
|---|---:|---:|
| alpha（年化） | **−4.45%** | **−6.85%** |
| alpha t | **−2.63** | **−4.28** |
| value β | 0.2979 | 0.2813 |
| low-vol β | 0.2774 | 0.3281 |
| R² | 0.526 | 0.595 |

残差化重建组合的超额 **−9.71%**（净收益 −5.33%）。
**alpha 已跨过显著性门槛——不是"没有 alpha"，是"显著负 alpha"。**

**Phase 1 头条已修订**：9.79% → 7.29%，超额 4.38% → 2.01%（成本口径 −2.9pp、权重口径 +0.7pp）。

## 目录与关键数据形状

| 路径 | 内容 | 形状 / 注意 |
|---|---|---|
| `data/raw/daily_YYYY.parquet` | 日线 | 8 个文件，含 `pct_chg` / `amount`；glob 用 `daily_????` |
| `data/raw/daily_basic_YYYY.parquet` | 每日指标 | turnover_rate / circ_mv / pe_ttm |
| `data/raw/adj_factor_YYYY.parquet` | 复权因子 | 8 个文件 |
| `data/raw_fina/{api}_part*.parquet` | 财务三表原始 | 每表 25–35 万行，保留 update_flag 0/1 重复行 |
| `data/ref/` | stock_basic / stock_delisted / namechange | **两者都带 industry** |
| `data/panel/tradable.parquet` | **宽表** | `(1921, 5696)`，index = `YYYYMMDD` **字符串** |
| `data/panel/factors.parquet` | **长表** | `(1902858, 8)`，trade_date 为 **str** |
| `data/panel/factors_neutral.parquet` | 中性化后 | 07/08/09/19…28 都读它 |
| `data/panel/factor_accruals.parquet` | Phase 2a | `(1827435, 7)` |
| `data/panel/factor_pead.parquet` | Phase 2b | `(1119547, 7)` |
| `data/output_attribution/` | 归因输出 | `excess_nav_compare.png` 需拷到 `figures/` |
| `data/output_style/` | 风格中性 | `nav_compare.png`、`style_premiums.png` |
| `data/output_turnover/` | Phase 3 | `sensitivity_v2.csv` / `final_numbers.csv` / `capacity.csv` |

## 脚本状态

| 脚本 | 状态 |
|---|---|
| 01–06 | ✅ universe 1921 天/日均 991；factors 1,902,858 行；中性化 IC 0.017→0.035 |
| 07_backtest | ✅ 5 变体（旧口径，仅作历史对照） |
| 08_attribution | ✅ 双边口径：alpha −4.45% (t=−2.63) / −6.85% (t=−4.28) |
| 09_style_neutral | ✅ 残差化组合超额 −9.71%，净收益 −5.33% |
| 10–12 | ✅ 应计项，REJECT |
| 14–18 | ✅ PEAD/SUE，REJECT |
| 19 / 20 / 24 / 25 | ✅ 换手诊断与两轮方案 |
| 26_sensitivity | ✅ Δ 平台 + placebo 归因 |
| 27_impact_cost | ✅ 容量约 2.4 亿 |
| 28_fix_cost_convention | ✅ 最终数字（双边口径） |

## 核心结论（四条）

1. **Phase 1**：smart beta，**显著负 alpha**。value β 0.298 / lowvol β 0.277；剥离后 alpha −4.45%（t=−2.63）/ −6.85%（t=−4.28）。
2. **Phase 2a 应计项**：相关性 0.054（真新维度），但 IC 0.0093、两年为负 → 拒绝。**信号太弱。**
3. **Phase 2b PEAD/SUE**：截面全过（IC 0.023@20d、七年同号），组合不显著（t = −0.27 / +0.66）→ 拒绝。**信号期与调仓期错配。**
4. **Phase 3 换手控制**：配额守恒的分数门槛把换手从 2514% 降到 714%，超额从 0.22% 升到 **4.25%**。其中 **+8.03pp 来自成本、+2.57pp 来自信号**（同换手配对）。

## 样本外说明

2018–2025 全样本已被反复使用（Phase 1 归因、Phase 2 两个候选、Phase 3 全部实验）。
**不构成样本外。** 任何新配置必须在 **2026-01 之后**的数据上做首次、也是唯一一次验证。

## 下一步

| 优先级 | 事项 | 说明 |
|---|---|---|
| 1 | 更新 `figures/` | `excess_nav_compare.png`、`nav_compare.png` 已重新生成，需覆盖旧图 |
| 2 | 2026 数据后做 OOS | 唯一的样本外机会 |
| 3 | 实盘/模拟盘记录 | 目前完全没有 |
| — | 不再尝试 | 应计项与 PEAD 的任何变体 |

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
14. **成本必须双边计费**：`Σ|Δw| × c`，不是 `单边换手 × c`。旧代码少收一半。
15. **换手率汇报要用 `to.replace(0, np.nan).mean()`**，否则被非调仓日的 0 摊薄 5 倍。

## 一句话交接

四次迭代后结论稳定：**显著负 alpha 的 smart-beta 组合，靠换手控制把 12bp 下超额从 0.22% 做到 4.25%，容量约 2.4 亿**。
收益的 3/4 来自成本控制，1/4 来自信号时效性——这个比例必须诚实地说出来。
