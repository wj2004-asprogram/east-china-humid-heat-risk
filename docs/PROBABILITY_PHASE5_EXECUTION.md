# 第五阶段执行与验证记录

隔离分支 `research/five-site-evidence-20261105`，起点为第四阶段完成提交 `4bb4028977ee3ad75d2035e60887491f53096a49`。阶段五遵循原审阅计划6—7节及新增转译协议，没有重新训练、校准、修改阈值或使用正式测试择优。身份/边界快照保持2026-10-29。

## 执行内容

1. 读取原协调交接、原ESG项目、现有五厂历史/空间结果、完整研究计划及第四阶段冻结评价。固定原五场所和主格，保留30候选未核验状态。
2. 建立config驱动的提取/描述模块：从120个enriched文件连接冻结字典，提取15120条模型记录，840个不同物理格日；完整JJA历史为4600格日、50格年。所有实际读取预测、字典、原历史/空间输出与清单哈希一致。
3. 逐格逐日核对未取整日HI的目标标签与D−1标签；各模型/lag共同样本一致，研究告警与既有概率/阈值一致。原有三重seal验证通过，既有校准参数没有改动。
4. 使用冻结计分函数生成1080逐年指标、216五年等权指标、10800十箱记录及1080告警记录。没有生成单厂置信区间或新显著性检验。
5. 交付五张Markdown/JSON证据卡、15条条件影响路径、65个显式缺失现场字段、15项待验证补证任务；生成四幅matplotlib图与独立本地HTML报告，原界面保持原样。

## 命令与环境

主要命令及复现条件见[复现入口](PROBABILITY_PHASE5_REPRODUCE.md)。本次解释器Python 3.13.1，numpy 2.5.4、pandas 3.0.6、matplotlib 3.11.2、pytest 9.1.1；未安装新依赖，未修改原venv。

实际提取及报告重建的完整命令、UTC时间、耗时、退出码和标准输出保存在 `data/probability_phase5/rebuild_verification.json`。提取约5.24秒，报告生成约2.62秒，均exit 0。8张核心数值表在重建前后逐字节一致，包含概率明细、映射、历史、分年/等权指标、可靠性及告警表。仅清单生成时间允许变化。

针对性测试命令：`python -m pytest tests/test_probability_site_evidence.py -q --junitxml=data/probability_phase5/pytest.xml`，最终12项通过，pytest报告耗时1.55秒。测试覆盖掩膜空值与不替换、三厂共格去重、buffer禁用strict、92/84窗口、D−1/D−2输入截止、标签/告警篡改拒绝、缺失日期/重复记录拒绝、逐年等权BS独立重算、新发生分母、空风险集NA、现场字段null、哈希、候选覆盖、来源与链接。最终结果见pytest.xml及validation_receipt.json。

本地浏览检查使用已有Playwright Core 1.62.1与Chromium headless shell，无安装、无服务器、无可见窗口：

```powershell
node tools/probability_phase5_verify_ui.cjs --playwright '<USER_HOME>\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\node_modules\playwright-core' --browser '<USER_HOME>\AppData\Local\ms-playwright\chromium_headless_shell-1234\chrome-headless-shell-win64\chrome-headless-shell.exe'
```

1280/768/390宽度的首屏及展开表格均无页面溢出；五卡、四图和本地下载路径存在；没有页面错误或外部网络请求。截图与实际浏览器版本位于 `reports/probability_phase5/qa/`。人工查看地图、逐年图与桌面/手机截图后，修正了历史段落中的Markdown符号直接显示及内部状态码文案；不涉及数据或模型。

Chromium底层记录了沙箱内DNS配置读取和Vulkan设备初始化诊断，保存在 `reports/probability_phase5/qa/chromium-debug.log`。浏览器使用默认设备完成截图，页面加载、图像和交互检查均成功；没有据此声称底层日志完全无警告。

## 故障与处理

数据提取、描述评价和报告生成均成功，没有训练/评价中断或数值重拟合。第一次沙箱内只读检查两个D盘原仓库的Git状态返回权限拒绝（exit 128）；核对既有用户授权后，按权限流程执行只读、关闭可选锁的同一检查，获批成功。没有绕过权限或修改原仓库。

最终原ESG仓库HEAD为 `3aa2b0872a5d9a11909776229054e0f944e21d5f`；原气象仓库HEAD为 `d5f50d029a7a292763ca94c7837adad253dd9c63`，两者status --porcelain均为空。隔离副本中第四阶段代码/配置/数据及旧界面无改写；归档时验证变更范围和字节哈希。

## 资源、备份与剩余事项

新增派生数据约8 MB，另有少量图片和截图；未复制原小时NetCDF、大模型数组或原证据全文。重建时C盘仍有约62 GB可用空间。轻量Git bundle保留代码、表格、报告和图片，冻结数组仍本地保留；不会用小备份宣称包含全部天气数据。

没有剩余技术执行阻塞。未完成的是新增资料的实地取得：当前主体/边界及运营确认、现场观测、岗位班次、实际冷却/能耗/质量/停机记录、独立气象观测和真实起报预报。本轮没有对外联系、部署、推送、合并或创建未来任务。

这些缺口不会妨碍本阶段研究交付，但使结果仍限于历史格点危害筛查，不能升级为实时预警、职业安全或企业损失结论。任何后续模型或场所方法变更需新验证设计，保留已接触测试历史。
