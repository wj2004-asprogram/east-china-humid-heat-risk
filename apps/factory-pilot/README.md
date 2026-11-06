# 制造业场所历史湿热环境研究工作台

直接用浏览器打开 [dist/index.html](dist/index.html)。无需服务端、API 令牌、在线地图或前端构建依赖；本原型不保存用户输入，所选结果的 CSV 导出在浏览器内生成。宽表格在表格区域内横向滚动。

研究工作台 v0.3（界面修订 2026-11-05）提供四个数据视图，并增加[首批五场所研究报告](dist/study/index.html)：固定研究口径、407段阈值连续过程、实际多边形交集与20种空间情景。报告附SVG/PNG图件及完整资料ZIP；本轮复现流程见[阶段1—3结果](../../docs/ESG_FIVE_SITE_RESULTS.md)。原有视图仍包含样本与空间匹配、年度与逐日数据、网格敏感性、方法与证据。

公开证据仍为 2026-10-29 快照：35 个地点中有 5 个公开资料已核验场所、30 个待核验候选。34 个地点具有有效主气候结果，按 `grid_id` 去重后只有 **4 个有效主网格、100 个网格—年份组合**；原 850 条场所年度展开记录包含共同网格重复。不同网格也不自动构成统计独立样本。默认只显示已核验场所，其有效主网格为 2 个。

气候时期为 2000—2024 年 JJA，变量是北京时间日最高热指数（HI），阈值为 ≥40.6°C。结果描述当前映射位置的历史网格环境，不用于推断工厂历史存续、员工实际暴露或企业 ESG 评级。方法、数据字典与敏感性解释见 [研究工作台说明](../../docs/ESG_RESEARCH_WORKBENCH.md)；首批核验和采样边界见 [场所核验记录](../../docs/ESG_FACTORY_VERIFICATION.md)。

## 重建

在仓库根目录运行，使用现有科学Python环境。以下路径为本机已核实位置；不重跑原始气候处理。

```powershell
$pilotPython = '<ERA5_SOURCE_ROOT>\.venv\Scripts\python.exe'
$pilotEvidence = '<LOCAL_EVIDENCE_ROOT>\esg-public-pilot-20261029\factory_evidence'
$pilotScreening = '<LOCAL_EVIDENCE_ROOT>\esg-public-pilot-20261029\factory_screening'
$pilotClimate = '<ERA5_SOURCE_ROOT>\data\processed'

& $pilotPython -m tools.esg_build_catalog --review data/site_inputs/factory_review.json --evidence $pilotEvidence --output data/site_inputs
& $pilotPython -m tools.esg_site_screen --sites data/site_inputs/factory_sites.csv --heat-index "$pilotClimate\humid_heat_jja_2000_2024_bjt.nc" --risk "$pilotClimate\heat_risk_categories_jja_2000_2024_bjt.nc" --output $pilotScreening
& $pilotPython -m tools.esg_build_dashboard --catalog data/site_inputs/factory_sites.json --sites data/site_inputs/factory_sites.csv --screening $pilotScreening --heat-index "$pilotClimate\humid_heat_jja_2000_2024_bjt.nc" --output apps/factory-pilot/dist
& $pilotPython -m tools.esg_build_research_data --catalog data/site_inputs/factory_sites.json --boundaries data/site_inputs/factory_boundaries.geojson --screening $pilotScreening --heat-index "$pilotClimate\humid_heat_jja_2000_2024_bjt.nc" --risk "$pilotClimate\heat_risk_categories_jja_2000_2024_bjt.nc" --output apps/factory-pilot/dist
```

`esg_build_catalog` 从审定的源对象和原始响应重建坐标/来源；`esg_build_dashboard` 核对输入与输出哈希、场所 ID、状态和坐标一致性后打包。最后的 `esg_build_research_data` 读取同一批现成 NetCDF，按主网格和包围盒候选网格提取年度/逐日值，逐项核对原 850 条场所年度结果，并保留边界、来源属性和输入哈希。若目录、筛查结果与边界未改变，只需运行最后两条打包命令。

两个打包器的覆盖范围不同：`esg_build_dashboard` 重写 `pilot-data.js`、四个基线下载文件及 `package_manifest.json`，不覆盖页面源码或 `research-data.js`、`research_*` 文件；`esg_build_research_data` 重写研究数据文件及 `downloads/research_manifest.json`。基线发生变化时须按上述顺序重新生成两包，避免页面同时加载新基线和旧研究结果。`package_manifest.json` 仅涵盖基线数据，研究数据哈希另见 `downloads/research_manifest.json`；两者都不等于整个界面源码的校验清单。

旧快照不可覆盖成新证据；重新获取时使用新的证据 ID 和审定记录。研究包的 `generated_at_utc` 是构建时间，不会更新证据快照日期。

由哈希清单引用的生成数据通过 `.gitattributes` 关闭换行自动转换，避免 Windows 检出改变文件字节。

`tools/esg_fetch_evidence.py` 只取得命令行明确指定的单个公开URL；不自动发现链接、不重试、不处理验证码。已存档的同一ID校验URL与内容哈希后直接复用。

## 验证

```powershell
& $pilotPython -m pytest tests/test_esg_site_screen.py tests/test_esg_catalog.py tests/test_esg_research_data.py -q
& $pilotPython -m tools.esg_validate_pilot --catalog data/site_inputs/factory_sites.json --boundaries data/site_inputs/factory_boundaries.geojson --evidence $pilotEvidence --screening $pilotScreening --heat-index "$pilotClimate\humid_heat_jja_2000_2024_bjt.nc" --risk "$pilotClimate\heat_risk_categories_jja_2000_2024_bjt.nc" --first-five '<LOCAL_EVIDENCE_ROOT>\esg-public-pilot-20261029\first_five_screening' --output docs/validation/ESG_FACTORY_PILOT_20261029.json

# 可使用已安装的 Playwright；本机使用 Codex 捆绑依赖。
$env:NODE_PATH = '<USER_HOME>\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\node_modules'
node apps/factory-pilot/verify-ui.cjs '<LOCAL_EVIDENCE_ROOT>\esg-research-workbench-20261105\ui_qa'
```

2026-11-05 研究工作台自审运行通过 24 项 Python 测试及 14 项浏览器检查，汇总记录见 [研究工作台检查记录](../../docs/validation/ESG_RESEARCH_WORKBENCH_20261105.json)。研究数据构建器另核对原 850 条场所年度结果，记录位于 `downloads/research_manifest.json` 的 `validation`；浏览器检查记录及截图写入命令指定目录。2026-10-29 基线核验另保留在 `docs/validation/ESG_FACTORY_PILOT_20261029.json`，不以旧界面检查替代本版检查。上述核对属于作者自审，未代表独立审阅或现场核验。

原始公开文档受原权利约束，只保存在本地证据目录；界面数据保留出处、获取时间与哈希。OSM 派生坐标及边界署名和许可见 [第三方数据说明](../../data/site_inputs/THIRD_PARTY_DATA.md)。
