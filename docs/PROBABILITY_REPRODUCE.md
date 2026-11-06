# 概率阶段：本次执行命令与复现

> 公开版本说明：以下命令记录原本地归档流程，路径占位符不代表可直接执行的恢复命令。干净克隆请使用 [可移植复现入口](PROBABILITY_PUBLIC_REPRODUCE.md)。原 seal/清单保留为历史证据，公开脱敏文件使用独立 publication manifest。


从本独立研究副本根目录运行。使用既有科学环境，不需下载或安装包。将输出目录换成新的空路径；原始数据目录只读。下列为本次成功执行命令的等价变量形式。

```powershell
$python = '<ERA5_SOURCE_ROOT>\.venv\Scripts\python.exe'
$sourceProject = '<ERA5_SOURCE_ROOT>'
$acceptedInputs = '<LOCAL_EVIDENCE_ROOT>\phase2-production-20261029\input-bjt.json'

# 只读哈希/结构检查；数值验证限2000年6月。
& $python -B -m tools.probability_data_audit `
  --config configs/jja_2000_2024_bjt.yaml `
  --source-root $sourceProject --previous-raw-report $acceptedInputs `
  --output-dir data/probability_phase1/audit-r2

# 只读时间轴和静态mask，无最终测试标签统计。
& $python -B -m tools.probability_sample_manifest `
  --dates-netcdf "$sourceProject\data\processed\humid_heat_jja_2000_2024_bjt.nc" `
  --time-name date `
  --mask-netcdf "$sourceProject\data\processed\heat_risk_categories_jja_2000_2024_bjt.nc" `
  --mask-variable land_mask --output-dir data/probability_phase1/samples

& $python -B -m tools.probability_geometry `
  --grid-registry data/probability_phase1/samples/grid_registry.csv `
  --output-dir data/probability_phase1/geometry

# 仅一个训练夏季；输出7项每日数组与42行无目标标签的历史特征试算。
& $python -B -m tools.probability_feature_pilot `
  --config "$sourceProject\configs\jja_2000_2024_bjt.yaml" `
  --source-project-root $sourceProject --year 2000 `
  --output-dir data/probability_phase1/feature-pilot-2000

& $python -B -m pytest tests/test_probability_samples.py `
  tests/test_probability_features.py tests/test_probability_baselines.py `
  tests/test_indices.py tests/test_time_utils.py tests/test_land_sea_mask.py `
  -q -p no:cacheprovider --basetemp ..\pytest-probability-final `
  --junitxml data/probability_phase1/validation/targeted-tests.xml

# 本次固定目录的验收收据；先完成上面结果，再执行。
& $python -B data/probability_phase1/validation/make_receipt.py
```

本次实际测试使用相应绝对`--basetemp`路径，并将stdout/stderr保存到`validation/targeted-tests.log`。不要对一个正在运行的pytest复用同一临时目录。样本CLI会写入指定目录，复现时仍应使用新目录；其他数据工具拒绝覆盖现有输出。收据脚本绑定本次目录，重做时按新的运行目录留存单独收据，不覆盖历史验收。

初次审计因错误配置字段名失败；改正后`audit-r1`仅配置换行哈希FAIL；`audit-r2`采用原配置哈希及规范化文本双重核对后PASS。失败原因与处理保存在`validation/summary.json`，无需为了复现故意执行旧错误代码。

科学设计、样本清单、特征小试和合成基线测试分别完成，但这不等于已经拟合真实模型。下一步见[进度交接](PROBABILITY_PHASE1_PROGRESS.md)。
