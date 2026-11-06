# 首批5场所研究数据字典

适用版本：`EHHR-FIVE-SITE-01` v1.0；证据快照2026-10-29。以下路径相对 `data/research_phase1/`。场所为身份与地图对象，时间分析以不同主匹配网格为单元；当前5个场所关联3个主格，其中2个有效主格形成50个网格年和407条阈值过程记录。

## 通用编码与关联键

| 项目 / 字段 | 定义与使用方式 |
| --- | --- |
| `site_id` | 场所唯一键，例如 `factory_002`；关联原目录 `factory_sites.json`、时间 `site_links` 与全部空间表。5个已核验场所不表示5条不同气候序列。 |
| `grid_id`、`primary_grid_id` | 网格中心键，格式为 `纬度,经度`，各4位小数，例如 `30.2500,120.5000`。主格通过 `site_links.primary_grid_id = annual.grid_id` 等关联；保持字符串，不将逗号误作列分隔。 |
| `episode_id` | 一个网格内的阈值过程唯一键，格式 `grid_id:year:年内序号`；不按场所复制。 |
| CSV / JSON | CSV为UTF-8带BOM、逗号分隔，含逗号的字段带引号；JSON为UTF-8。用CSV解析器读取，不用简单字符串切分。 |
| 空值 | JSON为 `null`，CSV为空字段；表示不可计算、被规则保留为空或该情景不适用，须结合状态字段判读。空值不是0。 |
| 布尔值 / 列表 | JSON为 `true/false`、数组；CSV为 `True/False`，一般列表用竖线 `\|` 连接。`selection_detail_json` 是完整JSON字符串，须先解析CSV再解析该字段。 |
| 日期与坐标 | `YYYY-MM-DD`日标签为北京时间UTC+8；不是带时区的瞬时观测。场所及GeoJSON用WGS84；GeoJSON坐标顺序为 `[经度, 纬度]`，与 `grid_id` 顺序不同。 |
| 数值精度 | CSV/JSON保留计算浮点值；图表显示可舍入。阈值判断使用未舍入日值。坐标或面积的小数位数不代表现场测量精度。 |

关键关系：`site_links` → `primary_grid_id` → `annual` / `episodes` / `grid_summary`；`site_id` → `site_spatial_audit` → `polygon_grid_intersections` / `representative_scenarios`。连接后应保留网格去重语义，不能把杭州3厂关联出的重复行视为独立气候样本。

## 协议、清单和时间字段

| 文件 / 字段 | 定义 |
| --- | --- |
| `protocol.json` | 冻结的研究对象、期间、阈值、完整性、过程提取、分位数、选年及空间情景规则；不是结果表。 |
| `protocol_id`、`version` / `protocol_version` | 分析协议标识及版本。`schema_version` 是文件结构版本，两者含义不同。 |
| `primary_site_ids`、`period`、`threshold_c` | 首批场所白名单；2000—2024年各年JJA、92天、UTC+8；继承项目阈值40.6°C。后者是热指数阈值，不是气温阈值或职业限值。 |
| `episode`、`descriptive_statistics`、`year_selection`、`spatial` | 过程定义与边界规则、线性分位数、典型/极端年决策顺序、边界相交及代表点情景。修改规则应升级协议并重新计算。 |
| `rules_fixed_at_utc` | 内部规则固定时间；既有汇总已知，因此不声称前瞻预注册。 |
| `evidence_snapshot_date` | 公开场所目录的核验快照日期2026-10-29，不随计算或页面重建更新。 |
| `generated_at_utc` | 本份数据包的实际生成时间，不是原文日期、现场核验日期或数据对应年份。 |
| `osm_timestamp` | 存档OSM对象版本时间，不表示建厂或持续经营起始时间。 |
| 原目录 `evidence_date_note` / 来源 `retrieved_at_utc` | 分别说明原文证据涉及日期及实际获取时间；时间与空间案例表通过 `site_id` 回查，不在本轮重写。 |
| `temporal/manifest.json`、`spatial/spatial_manifest.json` | 输入/输出SHA-256、方法和检查记录。`input_sha256.protocol` 关联本次确切协议字节；`output_sha256` 键为对应目录中的文件名。 |
| `source_input_sha256` | 时间包转存的上游研究输入哈希；不同于时间模块直接读取文件的 `input_sha256`。 |
| 空间清单 `source_code_sha256`、`software`、`methods` | 空间脚本哈希、Python/GEOS/Shapely/PROJ等版本、投影/加密/数值容差；`reference_urls` 为方法参考链接。`isolated_dependency_wheel`（如有）记录隔离依赖来源与哈希。 |
| `validation` | 程序一致性、计数及数值检查，不代表独立现场或科学模型验证。 |

## 时间结果：`temporal/`

`case_temporal.json` 汇集 `site_links`、`grids`、规则和哈希。每个 `grids` 对象包含本格的 `annual`、`episodes`、`distribution`、`selection`；下列CSV是其可分析表格形式，不能把JSON与CSV重复相加。Tesla的掩膜主格保留在关联与汇总表中，年度/过程数组为空，不以邻格代替。

### `annual.csv`：每个主网格—年份一行，当前50行

| 字段 | 单位 / 定义 |
| --- | --- |
| `grid_id`、`year`、`threshold_c` | 复合主键 `(grid_id, year)`；年份及本次HI阈值。 |
| `expected_days`、`valid_days` | 天；预期JJA日期数92，以及有有限日值的日期数。 |
| `missing_date_days`、`nonfinite_days` | 天；前者为时间轴缺少的预期日期，后者为已有日期上的NaN/无穷值，不能混作同一原因。 |
| `complete_summer` | 日期齐全、无重复且所有日值有限。为假时本年阈值指标及过程提取结果保留为空。 |
| `days_ge_threshold` | 天；当年JJA日最高HI ≥40.6°C的日期总数。 |
| `max_consecutive_days` | 天；当年满足阈值的最长连续日数，不跨冬季、缺日期或缺值连接。 |
| `episode_count` | 条；当年阈值过程数，包括单日过程，不表示统计独立事件数。 |
| `season_edge_episode_count` | 条；当年触及6月1日或8月31日的过程数。同一过程同时触及两端时只计1条。 |
| `annual_peak_hi_c`、`annual_peak_date` | °C / 日期；整个完整JJA内的最高日HI及最早出现日期。不是逐小时峰值，也不是普通气温。 |

### `episodes.csv`：每个不同网格的阈值过程一行，当前407行

| 字段 | 单位 / 定义 |
| --- | --- |
| `episode_id`、`grid_id`、`year`、`threshold_c` | 过程键、网格、所属夏季及阈值。 |
| `start_date`、`end_date`、`duration_days` | 起止日期均包含，持续天数为两端之间的连续观测日期数。1天过程保留。 |
| `peak_hi_c`、`peak_date` | °C / 日期；本过程内最高日HI及最早峰值日期。 |
| `mean_hi_c` | °C；过程内每日最高HI的算术平均，不是全过程逐小时HI均值或人员暴露均值。 |
| `left_season_censored`、`right_season_censored` | 是否触及6月1日、8月31日。标记意味着过程可能延伸到观察季之外，不能确认季外何时开始/结束；`duration_days` 始终只计JJA内观察到的天数。 |

### 汇总、选年与场所关联

| 文件 / 字段 | 单位 / 定义 |
| --- | --- |
| `grid_summary.csv` | 3个主格各一行，含1个掩膜主格。`verified_site_ids` 列出共享本格的首批已核验场所。 |
| `land_mask`、`screening_status` | 掩膜1为本次陆地筛查接受、0为拒绝；`ok` 为完整可用，`non_land_grid` 为主格被掩膜拒绝，`incomplete_climate_data` 为年度不完整。身份核验是另一个字段。 |
| `complete_summers`、`total_summers` | 个夏季；汇总分母及具有年度诊断记录的夏季数。掩膜主格为空。 |
| `mean_days_ge_threshold_per_complete_summer` | 天/完整夏季；年度阈值天数的算术平均。 |
| 汇总表 `episode_count` | 所有完整夏季的过程总数；与年度表同名字段的“当年条数”区分。 |
| `days_ge_threshold_*`、`max_consecutive_days_*`、`episode_count_*`、`annual_peak_hi_c_*` | 后缀为 `min/p25/median/p75/max`，分别表示对应年度指标的经验分布。单位依次为天、天、条、°C；采用线性分位数，不是置信区间。 |
| `year_selection.csv` | 每个有效主格各有 `typical`、`extreme` 两行，当前4行；键为 `(grid_id, selection_role)`，`selected_year` 是最终所选年份。 |
| `selection_detail_json` / JSON `selection` | 典型年保留 `median_days_ge_threshold`、`absolute_distance_days`、全部 `candidate_years`，距中位数最近后并列取最早年。极端年保留 `candidate_years_max_days`、`longest_run_among_max_days`、`candidate_years_max_days_and_run`；先最大阈值天数、再最长连续天数、再最早年。 |
| `site_links.csv` | 首批5个场所各一行：`site_name/entity/production_address/site_verification`来自原目录；`primary_grid_id` 是关联键。 |
| `shared_verified_site_count`、`case_role` | 首批已核验场所共享主格的数量；角色为 `verified_primary` 或 `masked_primary_no_temporal_result`。不是全35个地点的关联数。 |
| `typical_year`、`extreme_year` | 回填对应主格的选年；掩膜主格为空。极端年指年度阈值天数的规则选择，不保证也是全期最高HI或最长过程所在年份。 |

## 空间结果：`spatial/`

`spatial_audit.json` 汇集 `site_audits`、`grid_intersections`、`representative_scenarios`；三者分别对应下列三张CSV。`factory_grid_intersections.geojson` 保存非空相交几何及相同交集属性，供地图复核；原厂区边界仍以存档目录为准。

### `site_spatial_audit.csv`：每个场所一行，当前5行

| 字段 | 单位 / 定义 |
| --- | --- |
| `site_id/name/primary_grid_id`、`osm_id/osm_timestamp/source_url` | 场所、既定主格及原地图对象来源。`geometry_valid` 是计算几何有效性，不证明边界为地籍或现场实测。 |
| `area_m2`、`coarse_area_m2` | m²；原多边形在本地LAEA等面积投影下的最终加密面积、较粗加密面积。此表为整个厂区几何面积。 |
| `refinement_absolute_delta_m2` | m²；两级边段加密面积的绝对差，用于数值收敛检查。 |
| `geodesic_check_area_m2`、`geodesic_check_absolute_delta_m2` | m²；WGS84椭球面积交叉计算及其与最终投影面积的绝对差。不是坐标误差估计。 |
| `projection` | 该厂区的LAEA投影定义，中心取包围盒中点，单位米。不能直接合并不同投影坐标作为同一平面。 |
| `intersection_area_sum_m2`、`area_conservation_residual_m2` | m²；分区相交面积之和，以及该和减去厂区面积的有符号残差。 |
| `area_fraction_sum` | 无量纲；所有候选分区的厂区面积份额之和。保留浮点闭合残差，不强行改成1。 |
| `bbox_candidate_grid_ids`、`positive_area_grid_ids`、`boundary_touch_grid_ids`、`bbox_without_positive_area_grid_ids` | 分别为包围盒候选、真实多边形具有正面积交集、仅边界接触、无正面积交集的候选格列表。最后一类可能含边界接触及不相交。 |
| `land_mask_accepted_area_fraction`、`land_mask_rejected_area_fraction` | 无量纲；映射多边形落入“网格掩膜接受/拒绝的最近中心分区”的面积份额。**不是厂区真实陆地/水域比例、工人分布、暴露比例或位置置信度。** 不将接受部分重新归一化为整厂。 |
| `primary_grid_area_fraction` | 无量纲；映射多边形落入既定主格分区的面积/整个多边形面积。 |
| `scenario_count`、`changed_grid_scenario_count`、`outside_centroid` | 情景数；相对主格发生网格变化的情景数；面积质心是否落在原边界覆盖范围之外。 |

### `polygon_grid_intersections.csv`：每个场所—候选网格一行

| 字段 | 单位 / 定义 |
| --- | --- |
| `site_id`、`grid_id` | 复合主键；同一网格可以与多个场所关联，不因此产生额外气候观测。 |
| `grid_latitude/grid_longitude`、`region_bounds_wsen` | 度；网格中心及最近中心分区的西、南、东、北边界，后者顺序为 `[西界经度,南界纬度,东界经度,北界纬度]`。 |
| `relation` | `positive_area`：真实多边形有正面积交集；`boundary_touch`：仅线/点接触；`disjoint`：包围盒候选但真实多边形不相交。 |
| 面积及交叉核查字段 | `area_m2`、`coarse_area_m2`、`refinement_absolute_delta_m2`、`geodesic_check_area_m2`、`geodesic_check_absolute_delta_m2`：定义同上，但对象是当前交集；接触或不相交面积为0。 |
| `factory_area_fraction` | 当前交集面积/整个厂区多边形面积，无量纲；百分比展示时乘100。不是网格中厂区所占比例。 |
| `land_mask`、`is_primary_grid`、`research_analysis_role` | 当前格的二值掩膜；是否为该场所既定主格；上游35地点研究包中该格角色 `primary/sensitivity_only`。该上游角色不使其自动进入本轮首批5场所的主时间分析。 |

### `representative_scenarios.csv`：每个场所4种情景，当前20行

| 字段 | 单位 / 定义 |
| --- | --- |
| `site_id`、`scenario` | 复合主键；`accepted_primary` 为既定点，`laea_area_centroid` 为等面积质心，`laea_point_on_surface` 为面上代表点，`largest_polygon_overlap_grid` 为正面积交集最大的网格。 |
| `scenario_type` | 前三种为 `representative_point`；最大交集为 `grid_coverage_scenario`，它没有新厂区点坐标。 |
| `longitude/latitude`、`inside_boundary`、`covered_by_boundary` | WGS84点坐标；严格位于内部与位于闭合边界覆盖范围（含边界）的布尔判断。网格覆盖情景这四字段为空。质心若在边界外仍保留并标记，不替换原主点。 |
| `distance_to_selected_grid_km` | km；情景点至所选网格中心的大圆距离，不是位置误差或置信区间；网格覆盖情景为空。 |
| `primary_grid_id`、`grid_id`、`changes_primary_grid` | 固定主格、情景所选格，以及二者是否不同。仅报告差异，不修改原主格。 |
| `screening_status`、`land_mask` | 情景格的气候状态及掩膜值。 |
| `mean_days_ge_threshold_per_complete_summer`、`max_consecutive_days` | 天/完整夏季、天；情景格25年汇总，后者为完整夏季最长过程的全期最大值。不是单年值或平均最长过程。 |
| `mean_days_difference_from_primary`、`difference_status` | 天/完整夏季；情景均值减主格均值，仅两者均为 `ok` 时计算，状态为 `computed`。否则差值为空且状态为 `unavailable_primary_or_scenario`。 |

掩膜主格即使存在有效相交邻格，也不能获得一个替代主结果或可计算的主格差值。空间份额与气候序列是不同量，当前未计算面积加权“整厂HI”或人员暴露。
