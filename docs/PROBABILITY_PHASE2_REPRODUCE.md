# 第二阶段重现命令

> 公开版本说明：以下命令记录原本地归档流程，路径占位符不代表可直接执行的恢复命令。干净克隆请使用 [可移植复现入口](PROBABILITY_PUBLIC_REPRODUCE.md)。原 seal/清单保留为历史证据，公开脱敏文件使用独立 publication manifest。


在独立研究副本根目录使用 PowerShell。源项目只提供只读配置和原始数据；输出目录必须不存在，避免覆盖先前实验。所有年度CLI硬性限制为2000–2019，模型运行器只读取2000–2017。2018–2019不用于本轮拟合、选参、校准或评分。

```powershell
$python = '<ERA5_SOURCE_ROOT>\.venv\Scripts\python.exe'
$sourceRoot = '<ERA5_SOURCE_ROOT>'
$sourceConfig = Join-Path $sourceRoot 'configs/jja_2000_2024_bjt.yaml'

# 隔离依赖，不升级或修改源项目环境。
& $python -m pip install --target .phase2_deps --no-deps `
  scikit-learn==1.9.1 joblib==1.6.0 threadpoolctl==3.8.0 `
  cloudpickle==3.1.2 narwhals==2.28.0
$env:PYTHONPATH = "$PWD\.phase2_deps"
$env:OPENBLAS_NUM_THREADS = '2'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'

# 先执行2000，检查run.json中的耗时、内存、共同样本和SHA。
& $python -m tools.probability_phase2_features --config $sourceConfig `
  --source-project-root $sourceRoot --year 2000 `
  --output-dir data/probability_phase2/features/2000

# 每年独立进程；成功目录保留，可从首个未完成年度继续。
foreach ($featureYear in 2001..2019) {
  & $python -m tools.probability_phase2_features --config $sourceConfig `
    --source-project-root $sourceRoot --year $featureYear `
    --output-dir "data/probability_phase2/features/$featureYear"
  if ($LASTEXITCODE -ne 0) { throw "Feature extraction failed: $featureYear" }
}

# 固定普通时间与严格训练空间，18次有界LR拟合；输出目录须新建。
& $python -u -m tools.probability_phase2_baselines `
  --output-dir data/probability_phase2/experiments/run1

# 不重新拟合，只整理既有指标、输入清单及验证对照。
& $python -m tools.probability_phase2_report

# 最后生成包含报告/图/模型/大数组字节SHA的总目录；不会复制数组。
& $python -m tools.probability_phase2_archive
```

若仅需三个传统基线，可给模型CLI添加`--skip-lr`；这不与完整LR实验混用同一输出目录。`seasonal_pooled`额外报告区域季节先验，并不替代普通范围的格点收缩气候态。

预测采用`predictions/<scope>/lagN/<model>/<year>.npz`。每个文件保存原始概率、年度日索引、固定网格索引、目标标签和仅用于评价的前日标签。结合`prediction_schema.json`与冻结`grid_registry.csv`可确定唯一样本ID、经纬度、空间角色、起报时间、目标窗口及最大输入时间。`day_index=0`为6月9日；不得把数组行号当作跨年度独立ID。

可导出按原顺序排列的有限行明细（已保存20行D−2转移基线示例`data/probability_phase2/transition_d2_2015_preview.csv`）：

```powershell
& $python -m tools.probability_prediction_export `
  --experiment-dir data/probability_phase2/experiments/run1 `
  --scope temporal --lag 2 --model transition --year 2015 --limit 20 `
  --output new_prediction_preview.csv
```

年度`temporal.npz`和`strict.npz`各自同时保存`X1/X2`，对应同一共同键。列名和组别见`feature_registry.json`；所有模型，包括纯温度F0，均使用117列及两lag共同有限值筛选后的同一范围索引。严格邻域来源使用预冻结静态表；1299个训练查询格不意味着邻域来源也缩减为1299格。

LR先标准化全部训练列，再选取各臂列；这与各臂独立使用相同训练样本拟合其列均值/标准差等价。标准化不使用验证年份。每范围仅F4/D−1运行四个C，按三个验证年等权原始Brier选定，复用到其余臂及D−2。每次完整lag后持久化模型参数、逐年指标和预测；四个候选的选择记录逐次写出。中断后的成功年度特征可直接复用；模型中断需使用新实验目录重新执行，不能把不完整目录冒充完整结果。

原始ERA5的存储和既有公式计算精度沿用；NetCDF不额外降精度。事件标签直接由日HI按40.6°C判定，与供模型使用的float32特征存储独立。模型矩阵在LR计算时转为float64。缺任意固定来源或所需小时会被拒绝或标为缺测，绝不使用忽略NaN后的部分均值。

关键输出为`run.json`（命令、版本、输入SHA、资源）、`experiments.json`、`lr_selection_*.json`、`models/*.json`（可直接恢复线性概率计算）、`metrics_yearly.csv`和`reliability_yearly.csv`。模型原始概率是本阶段唯一概率产品，校准字段在schema中明确为null；0.5只是固定研究评分阈值。

针对性测试：

```powershell
& $python -m pytest tests/test_probability_samples.py `
  tests/test_probability_features.py tests/test_probability_baselines.py `
  tests/test_probability_complete_features.py tests/test_probability_metrics.py `
  tests/test_probability_phase2_runner.py -q `
  --basetemp <LOCAL_ARCHIVE_ROOT>\new-test-temp
```

Windows受限环境需选择新的可写`--basetemp`。测试使用合成数据，不需要打开真实最终测试年份。
