# PROJECT STATUS


> 最后更新：2025-11-20（Phase 2 全部关闭）

## 环境

- macOS，Python 3.13，venv 在 `.venv`
- 终端提示符常为 `(.venv) (base)`：conda base 同时存在，裸敲 `python` 可能落到 base（没有 pandas）。
  **一律用 `.venv/bin/python`。**
- Tushare Pro **2000 积分**（200 元/年）：财务三表 100 行/次、限 200 次/分钟、无 `_vip`。
- 依赖：tushare / pandas / numpy / pyarrow / python-dotenv / statsmodels / matplotlib

## 目录与关键数据形状

| 路径 | 内容 | 形状 / 注意 |
|---|---|---|
| `data/raw/daily_YYYY.parquet` | 日线 | 8 个文件，含 `pct_chg`；**glob 用 `daily_????`** |
| `data/raw/daily_basic_YYYY.parquet` | 每日指标 | turnover_rate / circ_mv / pe_ttm |
| `data/raw/adj_factor_YYYY.parquet` | 复权因子 | 8 个文件 |
| `data/raw_fina/{api}_part*.parquet` | 财务三表原始 | 每表 25–35 万行，**保留 update_flag 0/1 重复行** |
| `data/ref/` | stock_basic / stock_delisted / namechange | namechange 14,266 条 |
| `data/panel/tradable.parquet` | **宽表** | `(1921, 5696)`，index = `YYYYMMDD` **字符串**，columns = 代码 |
| `data/panel/factors.parquet` | **长表** | `(1902858, 8)`，trade_date 为 **str** |
| `data/panel/factors_neutral.parquet` | 中性化后 | 07 读的是这个 |
| `data/panel/factor_accruals.parquet` | Phase 2a | `(1827435, 7)` |
| `data/panel/factor_pead.parquet` | Phase 2b | `(1119547, 7)` |
| `data/output_backtest/daily_net.parquet` | 各方案日净收益 | `(1921, 8)`，供 18 用 |

`data/` 全部在 `.gitignore` 内，仓库里只有代码与文档。

## 脚本状态

| 脚本 | 状态 | 关键输出 |
|---|---|---|
| 01–06 | ✅ | universe 1921 天 / 日均 991；factors 1,902,858 行；中性化 IC 0.017→0.035 |
| 07_backtest | ✅ | 5 变体；EW 9.79% / 超额 4.38% / IR 0.663 |
| 08_attribution | ✅ | alpha −2.18% / −4.37% |
| 09_style_neutral | ✅ | 残差化后超额 −5.44% |
| 10_pull_fina | ✅ | 92 分钟，17,088 次调用 |
| 11_build_accruals | ✅ | 覆盖率 96.0% |
| 12_check_accruals | ✅ | **REJECT**（IC 0.0093） |
| 14_build_pead | ✅ | SUE 可用 116,075 行，在册覆盖 58.8% |
| 15_check_pead_horizon | ✅ | IC 5/20/60d = 0.0142 / 0.0229 / 0.0297 |
| 16_check_pead_marginal | ✅ | 残差 IC 0.0300；comp ICIR 0.462→0.538 |
| 16b_check_selection | ✅ | 无正向选择偏误（子样本 0.0880 < 全样本 0.0964） |
| 18_significance | ✅ | **全部 t < 1，CI 含 0 → 拒绝** |

## 核心结论

**Phase 1**：smart beta，无 alpha。超额 4.38% 来自 value（β 0.285）+ lowvol（β 0.288）；
控行业后 alpha −4.37%（t = −2.91）。

**Phase 2a 应计项**：与现有因子相关性极低（max 0.054，确实是新维度），
但 IC 0.0093、2020/2025 两年为负、五分组 Q5 < Q4 → 拒绝。**失败原因：信号太弱。**

**Phase 2b PEAD/SUE**：截面全部过关（IC 0.0229@20d、七年同号、残差 0.0300、分组单调），
但组合层面：
- EW：超额 4.38% → 3.96%，配对 t = −0.27
- Stratified：2.77% → 3.43%，配对 t = +0.66
- 换手 23.94% → 24.00%（Top50 名单几乎没动）
→ 拒绝。**失败原因：信号真实，但在 5 日换手的 Top50 组合里无法变现。**

**三次尝试，三种不同的失败原因。这是项目最有价值的结论。**

## 样本外说明

现有 2018–2025 全样本已被反复使用（Phase 1 归因、Phase 2 两个候选的全部检验），
**不构成样本外**。分年表逐条讨论过，等于把每一年都"看过了"。

因此：任何新版本必须在 **2026-01 之后**的数据上做首次、也是唯一一次验证。
在此之前不得使用 2026 年数据做任何选择、调参或方向判断。

## 下一步

**Phase 3 的第一件事不是写优化器，而是量化当前持仓特征：**

- Top50 的规模分位分布
- 行业分布与基准的偏差（均值 7.37% / 最大 17.50% 的结构性来源）
- 换手来源拆解：新进榜 vs 权重漂移
- 成本敏感度：12bp → 20bp / 30bp 下的净收益曲线

没有这个基线，"加了约束之后改了什么"就说不清。预计半天，只依赖现有 `07` 输出。

之后才是：行业 ±2% 约束、规模中性（`SIZE_BUCKETS` 提到 2–3）、换手上限、平方根律冲击成本。

**明确不再尝试**：应计项的任何变体、PEAD 的任何变体（换 horizon / 换 SUE 定义 / 调 SIZE_BUCKETS）。
**需 10000 积分才能做**：分析师预期修正、资金流（5000 分不够，别买错档）。

## 已知坑（接手必读）

1. **`.venv/bin/python`**，别用裸 `python`。
2. `tradable.parquet` 是**宽表**，代码在 `columns` 上，不是 `ts_code` 列。
3. `glob('daily*.parquet')` 会匹配 `daily_basic_*`，必须 `daily_????`。
4. `merge_asof` 要求 **`on` 键全局单调**，排序时 on 键放第一位（不是 `[by, on]`）。
5. `Series.reset_index()` 产生 **3 列**（两个索引层 + 值列），别直接赋 2 个列名。
6. 字符串日期取月份是 **`[4:6]`**，`[5:7]` 会全 NaN 且**不报错**。
7. `update_flag` 的 NaN 转 str 会变 `'nan'`，排序时排在 `'1'` 前面 → 先 `fillna('0')`。
8. **启动长任务前 `ps aux | grep 脚本名`**：曾有两个进程同时拉 income，重复率 47%。
9. `.BJ` 的 `pct_chg` 有垃圾值（max 200,400%，占 3.0%），全市场计算要滤 `.SH` / `.SZ`。
10. **日期类型有两种**：`tradable` / 新因子表是 datetime64，`factors` / `daily` 是 str。跨表合并前先统一。
11. **覆盖率的分母要看清**：SUE 显示 10.2% 是全网格口径（1094 万格），在册口径是 58.8%（190 万格）。同一件事。
12. **`combine()` 是 `fillna(0)` 之后相加**，不是取均值。加因子要按"和"来想权重。

## 一句话交接

Phase 1 证明超额 = value + lowvol（smart beta，无 alpha）；Phase 2 两个正交候选分别因
"信号太弱"和"信号期与调仓期错配"被拒。**下一步不再找 alpha，做约束优化与冲击成本，
把现有的收益做到低换手、可交付；任何新结论只能等 2026 年后的数据验证。**
