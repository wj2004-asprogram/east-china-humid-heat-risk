# 公开版本：冻结模型复现

从仓库根目录使用 `tools/probability_reproduce.py`。它从已有原始小时数据
重建共同特征，恢复冻结模型和校准映射，并将预测与验收记录写入**新的输出目录**。
不重新训练、校准、选参或选择告警阈值，也不下载数据。

## 公开版本与原归档

原研究来源提交为 `fdc432eaa165eacc263d3b3d2f3d0e5af480fb4e`，保存在
原本地科学归档中。公开分支从既有公共仓库派生，未把该研究提交的未公开历史
一并发布。公开清单记录原文件哈希、发布文件哈希及脱敏或排除原因。

旧阶段的 seal、收据及执行文档是原实验的历史凭证。脱敏后的公开树不宣称通过
原归档的完整字节锁链。旧 `probability_phase2_*` 至 `probability_phase5_*`
执行命令不能作为干净克隆恢复入口；本页的新入口使用公开清单核验输入，
不会通过删除旧 seal 或覆盖原结果来继续旧实验。

恢复输入清单为 `publication/recovery_manifest.json`。它绑定公开版本中用于
推理的代码、模型、校准映射、网格和参考结果，与记录导出变换的发布清单分工不同。

## 环境

原实验使用 Python 3.13.1。建议创建独立环境；下面的 PowerShell 示例中
`python` 应指向该环境。在 macOS/Linux 使用相应的环境激活命令。

```powershell
python -m venv .venv-reproduce
.\.venv-reproduce\Scripts\Activate.ps1
python -m pip install -r requirements-reproduce.txt
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python -m tools.probability_reproduce --verify-only
```

依赖文件固定原科学锁记录的直接计算库版本。存储的LR系数可直接推理，
不需要安装scikit-learn。该文件不是完整的跨平台传递依赖锁；本轮没有完成
另一台机器上的全新环境安装验证。完整重训属于另一项工作，不能把本入口
称为训练流程复现，也不能据此再用2020—2024择优。

## 原始输入

数据是同一地区的ERA5小时 `t2m`、`d2m`，单位为K，UTC时间，0.25°网格；
范围为25—36°N、112—123°E。每个需要恢复的年份须有以下四份天气文件：

```text
era5_east_china_YYYY_06.nc
era5_east_china_YYYY_07.nc
era5_east_china_YYYY_08.nc
era5_east_china_YYYY_05_31_h16-23.nc
```

5月31日的8小时用于补齐6月1日北京时间日界。恢复器使用发布的固定陆地网格
及空间角色注册表，不重新选择H或buffer。已有数据可放在任意目录，
用 `--raw-dir` 指定；不必复制到仓库。

若没有数据，可使用已有CDS下载器，先配置个人CDS账号并接受数据条款。
凭证放在用户自己的CDS配置中，不能提交到仓库。下面的配置**仅含2018—2024**，
不会下载2000—2017；共21份月文件、7份日界文件及1份陆地掩膜。

```powershell
python -m pip install cdsapi==0.7.7 ecmwf-datastores-client==0.5.3 requests
python -c "from src.config import load_config; from src.download_era5 import run; run(load_config('configs/reproduce_2018_2024_bjt.yaml'), workers=1)"
```

此命令直接调用现有下载器，把输入放入 `data/raw/`，不启动后续处理或绘图，
也不运行旧的完整25年分析。只检查两个年份时，也可先用自己已有
的相应八份天气文件，不需要获取其余年份。数据来源和再分发说明见
[DATA_LICENSES.md](../DATA_LICENSES.md)。

## 先做小范围恢复

下面覆盖一个校准年份和一个正式年份、两种scope、两个lag及六个固定模型。
目录应为尚不存在的新目录；每次新恢复使用另一个目录。

```powershell
python -m tools.probability_reproduce --raw-dir data/raw --output-dir reproduction/check-2018-2020 --years 2018 2020 --models persistence seasonal seasonal_pooled transition logistic_F4 lightgbm_F4
```

2018年的strict样本来自训练空间T；2020年的strict样本来自留出区H。
每年共同目标窗口为北京时间6月9日至8月31日，共84日。
预期每年temporal为132,720格日；strict在2018—2019为109,116格日，
在2020—2024为13,860格日。这些是格日，不能当成独立天气样本。

## 恢复完整冻结矩阵

省略 `--years` 和 `--models` 时，恢复全部104个scope×lag×模型配置，
覆盖2018—2024七年，共728份预测；校准映射始终使用已保存的2018—2019参数。

```powershell
python -m tools.probability_reproduce --raw-dir data/raw --output-dir reproduction/all-2018-2024
```

输出包括 `predictions/{calibration|final_test}/{scope}/lagN/{model}/{year}.npz`、
`annual_scores.csv` 和 `reproduction_receipt.json`；只有添加 `--save-features`
才另外保存特征数组。验收记录应覆盖原始/校准概率、年度BS、样本键与告警标记，
不以仅有文件或仅有零退出码代替结果核验。

重新下载的NetCDF可能因服务端打包或元数据变化而与原归档字节哈希不同。
恢复器接受原文件SHA，或经K单位、时间、经纬度和 `t2m/d2m` 规范化内容指纹
确认相同的输入，并在收据中区分 `byte_match` 和 `content_match`。
科学数值不同的输入会被拒绝，没有跳过校验选项。结构、标签和告警要求精确
匹配；概率以原始指纹优先，或采用清单规定的12位小数规范指纹核验。
不要将内容相同但字节不同说成原始文件完全一致，也不能据差异修改模型。

本入口恢复冻结推理；公开报告、五厂表和原bootstrap区间仍是随仓库发布的
参考结果，不因运行一次恢复而变成新的独立研究。只有五个正式年份、一个H
留出区。五厂实际对应两个有效气候主格；杭州三厂共格、采埃孚仅有temporal、
特斯拉原主格被拒绝。复现结果仍是ERA5历史回报，不是实时预警、职业安全
阈值或企业损失估计。

## 本次公开版本验收

在不含本地特征/预测数组的干净Git克隆中，使用既有Windows环境和外部只读原始天气文件，完成104配置×7年、728份预测的完整恢复。全部数组参考核验通过；3,120条正式期年度分组分数的BS最大绝对差为0，样本及事件计数一致。8项新增针对性测试通过。

机器证据：[验收摘要](../publication/verification.json)、[完整重放收据](../publication/full_replay_receipt.json)。首次本机运行因依赖搜索路径遗漏而失败，补齐既有依赖路径后使用新目录完成恢复；没有改变模型、校准或阈值。未安装或升级依赖，未验证另一操作系统的全新环境安装。
