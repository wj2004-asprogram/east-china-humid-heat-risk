# 第四阶段执行与复现

> 公开版本说明：以下命令记录原本地归档流程，路径占位符不代表可直接执行的恢复命令。干净克隆请使用 [可移植复现入口](PROBABILITY_PUBLIC_REPRODUCE.md)。原 seal/清单保留为历史证据，公开脱敏文件使用独立 publication manifest。


在隔离仓库根目录运行。原始数据与源venv只读复用，不安装或改写源项目。以下命令按三个门依次执行；不要对已经完成的目录重复拟合，也不要通过删除seal来重开测试。此前各阶段成果及备份保留。

```powershell
$python = '<ERA5_SOURCE_ROOT>\.venv\Scripts\python.exe'
$sourceRoot = '<ERA5_SOURCE_ROOT>'
$env:PYTHONPATH = "$PWD\.phase3_deps;$PWD\.phase2_deps"
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
```

科学冻结之前，核对新执行器并运行针对性测试。`probability_phase4_restore_check`仅使用2000年训练样本，已经逐一验证104配置恢复后的概率与旧保存预测相符。它不读取校准或最终年份。

```powershell
& $python -m pytest tests/test_probability_phase4_data.py tests/test_probability_phase4_calibration.py tests/test_probability_phase4_evaluation.py tests/test_probability_phase4_runner.py -q -p no:cacheprovider
& $python -m tools.probability_phase4_seal --stage scientific
```

科学锁成功后，复用第二阶段既有2018—2019特征。年度batch在同一进程内处理104个固定配置，不重训模型；已有完整预测仅在锁链、输入、输出和源码哈希一致时复用。断裂产物拒绝覆盖，须保留错误记录后明确恢复。

```powershell
foreach ($calibrationYear in 2018..2019) {
  & $python -u -m tools.probability_phase4_predict --year $calibrationYear --batch
  if ($LASTEXITCODE -ne 0) { throw "Calibration inference failed: $calibrationYear" }
}
& $python -u -m tools.probability_phase4_calibrate
if ($LASTEXITCODE -ne 0) { throw 'Calibration failed' }
& $python -m tools.probability_phase4_seal --stage calibration
```

校准seal绑定实际映射及阈值后，逐年构造最终共同特征。气象配置使用本仓库已冻结的YAML，原始文件由只读sourceRoot提供。这一步仅检查完整性与共同索引，不报告事件率或成绩。每年写全域temporal及严格H两份数组；不重新构造校准资料。

```powershell
foreach ($finalYear in 2020..2024) {
  & $python -u -m tools.probability_phase4_features --year $finalYear `
    --config configs/jja_2000_2024_bjt.yaml --source-project-root $sourceRoot
  if ($LASTEXITCODE -ne 0) { throw "Final feature extraction failed: $finalYear" }
}
& $python -m tools.probability_phase4_seal --stage test_open
```

测试开放seal绑定五年两scope的实际共同索引与数组，随后执行一次冻结预测与正式评价。每个年度temporal与strict各52份预测，合计五年520份；校准为两年208份。评价沿用冻结映射，不能依据最终成绩重选阈值、候选、特征或子区。

```powershell
foreach ($finalYear in 2020..2024) {
  & $python -u -m tools.probability_phase4_predict --year $finalYear --batch
  if ($LASTEXITCODE -ne 0) { throw "Final inference failed: $finalYear" }
}
& $python -u -m tools.probability_phase4_evaluate
& $python -m tools.probability_phase4_report
```

所有命令的实际完成状态以 `data/probability_phase4/` 中seal、访问ledger和运行收据为准。完整序列不能仅凭本文假定已执行。统计解释见[协议](PROBABILITY_PHASE4_PROTOCOL.md)和[模型说明](PROBABILITY_MODEL_CARD.md)；最终报告还须明确数值失败、技术中断、未完成项及2017/2022既有接触历史。
