# 第五阶段复现入口

> 公开版本说明：以下命令记录原本地归档流程，路径占位符不代表可直接执行的恢复命令。干净克隆请使用 [可移植复现入口](PROBABILITY_PUBLIC_REPRODUCE.md)。原 seal/清单保留为历史证据，公开脱敏文件使用独立 publication manifest。


从隔离仓库根目录执行。第五阶段仅提取和描述第四阶段已有预测，不训练或重新校准。原始NetCDF不需要重读，旧界面不需要启动或修改。

## 前置文件

- `data/probability_phase4/`的三个seal、完成状态、字典、calibrators、checkpoint、artifact_manifest，以及120份本阶段所需enriched NPZ。大数组只保留本地，轻量Git bundle不含它们；缺数组时按[第四阶段入口](PROBABILITY_PHASE4_REPRODUCE.md)恢复原冻结产物，不能创建新候选。
- `data/research_phase1/temporal/`、`spatial/`、`data/site_inputs/`与 `apps/factory-pilot/dist/research-data.js`：使用现有哈希快照，不需要重新访问公开网站。
- Python、numpy、pandas、matplotlib及既有项目依赖。当前可用解释器为 `<ERA5_SOURCE_ROOT>\.venv\Scripts\python.exe`；隔离依赖在仓库 `.phase2_deps`、`.phase3_deps`。未安装新依赖或修改原环境。

## 命令

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONPATH="$PWD\.phase3_deps;$PWD\.phase2_deps"
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$python='<ERA5_SOURCE_ROOT>\.venv\Scripts\python.exe'
& $python tools/probability_phase5_sites.py --config configs/probability_phase5_sites.json
& $python tools/probability_phase5_report.py --config configs/probability_phase5_sites.json --application configs/probability_phase5_application.json
& $python -m pytest tests/test_probability_site_evidence.py -q
```

每步退出码应为0。提取前核验seal链、实际读取的120个预测文件、字典和原历史/空间清单；完成后再验实际输入未变化。代码不下载、不拟合、不触碰原预测。输出只写本阶段配置目录；重新运行会更新本阶段派生文件和时间戳，先保留本地提交/备份即可。若任一源哈希不符，应定位源版本差异，不能跳过检查或替换成新训练输出。

提取清单列出输入/输出SHA-256及软件版本；报告清单连接提取清单、配置与图卡哈希。时间戳每次运行会变化，因此重建验收针对表值、样本和报告内容，不能要求运行时间字段相同。原始概率CSV用17位精度保存；读取计算时推荐pandas的 `float_precision='round_trip'`。

## 预期结果与核验口径

| 产物 | 预期 |
|---|---:|
| 场所映射/证据卡 | 5 |
| 有效场所/不同有效主格 | 4 / 2 |
| 历史格日/格年 | 4600 / 50 |
| 正式期不同物理格日 | 840 |
| scope×lag×model×year源文件 | 120 |
| 去重模型记录 | 15120 |
| 逐年分组指标/五年等权指标 | 1080 / 216 |
| 逐年十箱记录/冻结告警指标 | 10800 / 1080 |
| 条件影响路径/缺失字段/补证任务 | 15 / 65 / 15 |

每个有效格、scope、lag、模型的每年均84日；历史每年92日。strict只含杭州主格；采埃孚无strict行；特斯拉无概率或历史主格数值。新发生风险集包括00和01。D−2的D−1状态只用于评价分组，冻结的校准和告警结果必须逐条一致。

原始源标签按未取整HI≥40.6°C对照，含D−1状态；同格同日各模型与lag标签必须相同。验证同时覆盖空风险集、无事件AP/召回率NA、共有样本、输入截止、JSON null、数值引用及静态链接。不得把NA填零。

打开 [本地研究报告](../reports/probability_phase5/index.html) 可浏览四幅图和五张卡。图由matplotlib产生，没有地图底图、外部脚本或网络请求。源网页链接仅在各卡中列出以便人工追溯；原证据全文未复制到新增报告。

## 文件约定

- `predictions.csv`：唯一键scope/lag/model_id/year/primary_grid_id/target_date_bjt；不按site_id复制行。
- `site_mapping.csv`：场所至原主格关联；拒绝格索引为空，buffer严格评价不可用。
- `historical_daily.csv`、`historical_annual.csv`、`historical_grid_summary.json`：两个有效主格的旧证据派生副本；保留原典型/极端年。
- `annual_scores.csv`、`aggregate_scores.csv`：风险集all/onset/persistence_or_clearance，raw/cal均保留；BS、AP、梯形PR面积、截断log-loss分列。等权表的n/events是合计，其余为完整五年等权均值；缺任一年定义则NA，并保留可用年数。
- `reliability.csv`：年度固定十箱；空箱均值为空，不补零。
- `annual_alerts.csv`：原校准概率的primary/budget两规则；阈值、混淆矩阵、误报漏报等保留，没有单厂阈值优化。
- `site_cards.json`及 `reports/probability_phase5/cards/*.md`：五卡；未知现场字段为null。
- `impact_pathways.csv`、`data_gaps.csv`、`monitoring_queue.csv`：条件路径、资料缺口、待验证工作队列，不是风险排序。
- `source_catalog.json`：18条既有公开源元数据（URL、检索时间、哈希等）；未重新验证当前网页或经营状态。

实际执行与测试证据见[执行记录](PROBABILITY_PHASE5_EXECUTION.md)。
