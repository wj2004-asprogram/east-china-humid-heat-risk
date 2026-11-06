# 首批五场所：阶段1—3结果与复现

2026-11-05完成。对应协议EHHR-FIVE-SITE-01 v1.0，证据快照仍为2026-10-29；本轮没有新增场所。阅读入口：[研究报告](../apps/factory-pilot/dist/study/index.html)。方案、数据字典和方法核查分别见 [研究方案](ESG_FIVE_SITE_PROTOCOL.md)、[数据字典](ESG_FIVE_SITE_DATA_DICTIONARY.md)、[方法核查](ESG_FIVE_SITE_METHOD_AUDIT.md)。

## 时间分析

五个已核验场所中四个有有效主气候结果，只对应两个不同有效网格：50个完整网格年、4600个网格日。407段阈值连续过程按网格去重，不表示407个独立天气事件；杭州三厂共享结果。特斯拉主格仍为空。

| 主网格 | 年阈值天数 min / P25 / 中位 / P75 / max | 均值 | 典型年 | 阈值天数最多年 | 全期最长连续过程 |
| --- | --- | ---: | --- | --- | --- |
| 杭州30.25,120.5 | 24 / 36 / 43 / 55 / 64 | 44.12天/夏季 | 2008，43天（2008/2010并列取最早） | 2022，64天 | 2022-07-19—08-26，39天 |
| 采埃孚31,121.5 | 13 / 24 / 30 / 39 / 56 | 32.24天/夏季 | 2005，30天（2005/2006/2019并列取最早） | 2022，56天；当年最长30天 | 2017-07-04—08-08，36天 |

本轮固定使用日最高HI≥40.6°C的项目操作阈值，典型年与极端年按方案规则选择。最多阈值天数与最长连续过程不是同一指标；没有添加趋势显著性、因果归因或职业暴露结论。

## 空间核查

5个存档OSM厂区多边形共产生6个正面积交集。特斯拉原主格分区覆盖70.05325509%，北侧有效格分区覆盖29.94674491%；其余四厂各完全落在一个主格分区。比例是地图几何与匹配分区的关系，不是厂区真实海陆比例。

20行定位情景中，原代表点、真实面积质心、多边形内部点和最大覆盖格均没有改变所选主格。特斯拉仍不能用北侧有效格替代主结果。福特真实面积质心在厂内，与原先位于厂外的包围盒中心不同。

面积采用逐厂LAEA等积投影，检查半步加密、WGS84测地面积及面积闭合。最大闭合残差约1.17×10⁻⁷ m²；这些小残差衡量数值一致性，不代表地图边界达到相同测量精度。

## 方法核查

已核对K/°C/°F转换、温度与露点推导RH、HI计算分支、北京时间日界和缺小时/NaN处理、项目0.5掩膜规则。官方NWS资料存在不同分级边界；40.6°C接近105°F的一位小数近似，本轮保留操作阈值，不称为统一官方或职业标准。源文件继承的温度/K属性仍保留，派生日HI未再次换算。详细源码与官方链接见方法核查记录。

## 复现

在科学项目根目录运行。使用既有Python环境；空间部分的Shapely 2.1.2隔离安装在本轮产物目录，未修改主环境。依赖wheel的来源、SHA-256及GEOS/PROJ版本见空间清单。

```powershell
$pilotPython = '<ERA5_SOURCE_ROOT>\.venv\Scripts\python.exe'
$pilotClimate = '<ERA5_SOURCE_ROOT>\data\processed'
$env:PYTHONPATH = '<LOCAL_EVIDENCE_ROOT>\esg-five-site-study-20261105\deps'
$env:MPLCONFIGDIR = '<LOCAL_EVIDENCE_ROOT>\esg-five-site-study-20261105\mplconfig'

& $pilotPython -m tools.esg_case_temporal --research apps/factory-pilot/dist/research-data.js --catalog data/site_inputs/factory_sites.json --protocol data/research_phase1/protocol.json --output data/research_phase1/temporal

& $pilotPython -m tools.esg_case_spatial --catalog data/site_inputs/factory_sites.json --boundaries data/site_inputs/factory_boundaries.geojson --research apps/factory-pilot/dist/research-data.js --research-manifest apps/factory-pilot/dist/downloads/research_manifest.json --risk "$pilotClimate\heat_risk_categories_jja_2000_2024_bjt.nc" --protocol data/research_phase1/protocol.json --output data/research_phase1/spatial --dependency-wheel '<LOCAL_EVIDENCE_ROOT>\esg-five-site-study-20261105\wheels\shapely-2.1.2-cp313-cp313-win_amd64.whl'

& $pilotPython -m tools.esg_build_case_report --temporal data/research_phase1/temporal --spatial data/research_phase1/spatial --research apps/factory-pilot/dist/research-data.js --catalog data/site_inputs/factory_sites.json --protocol data/research_phase1/protocol.json --output apps/factory-pilot/dist/study
```

其他机器可在隔离环境安装同版Shapely并替换环境路径；wheel文件仅为当前Windows/CPython3.13平台。空间面积结果按公开投影定义复现；图件字体在本机采用Microsoft YaHei，数值不依赖字体。

产物目录 `data/research_phase1/temporal/`、`spatial/` 保存完整表、JSON和清单；报告的 `downloads/` 保留同字节副本、方案和脚本，另含ZIP资料包。清单校验必须按各自目录中的相对文件名执行。报告生成器只核查既有分析产物，不重新分析小时数据。修改方案应另存版本并保留本版，不能直接覆盖固定规则。

## 验证

74项针对性Python测试通过，包括原筛查/目录/研究包、时间与空间模块，以及HI、日界、掩膜测试；4条既有NetCDF/NumPy依赖兼容性或弃用警告保留在测试日志，未通过升级依赖掩盖。新增报告9项浏览器检查通过，覆盖过程筛选、原精度导出、空筛选、全部本地链接、图件与平板/手机布局；三组图件及报告截图已目视检查。

这些是作者自审。未重新运行25个夏季的完整小时链、未现场核验厂区、未将ERA5与场站观测独立验证。测试与哈希汇总见 [本轮验证记录](validation/ESG_FIVE_SITE_STUDY_20261105.json)，本机完整日志和截图在 `<LOCAL_EVIDENCE_ROOT>\esg-five-site-study-20261105`。
