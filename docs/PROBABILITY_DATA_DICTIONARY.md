# 概率历史回报：数据字典与输入契约

本阶段只读输入根目录：`<ERA5_SOURCE_ROOT>`。配置为 `configs/jja_2000_2024_bjt.yaml`。审计见 `data/probability_phase1/audit-r2/data_audit.json`；实验设计见 [实验协议](PROBABILITY_EXPERIMENT_PROTOCOL.md)。

## 现有字段

| 文件 / 字段 | 实际含义与单位 | 使用约束 |
| --- | --- | --- |
| `data/raw/era5_east_china_YYYY_MM.nc`: `t2m`, `d2m` | UTC逐小时2米温度、露点，均为K；45×45规则经纬网 | Celsius = K − 273.15；只能按允许的历史小时取值 |
| `era5_east_china_YYYY_05_31_h16-23.nc` | 每夏北京6月1日最初8小时对应的UTC边界 | 只覆盖这一边界，不能当作完整五月历史 |
| `era5_land_sea_mask_36N_112E_25N_123E.nc`: `lsm` | 静态格点陆地比例，无量纲 | 项目规则≥0.5，得到1,580格；不是厂区实测海陆面积 |
| `daily_tmax_jja_2000_2024_bjt.nc`: `t2m` | 北京日最高温，°C | 已转换，禁止再次减273.15 |
| 同文件 `d2m` | **北京日最高露点，K** | 文件名和原变量名未体现此聚合；可转°C作为最高露点，不能当平均露点 |
| `humid_heat_jja_2000_2024_bjt.nc`: `daily_max_heat_index_c` | 北京日内小时HI的最大值，°C | `HI >= 40.6`为固定标签；继承的`GRIB_units=K`/气温属性与派生定义冲突，保留原文件并采用明确派生定义 |
| 同文件 `daily_mean_rh` | 先逐小时从T/Td推导RH，再对完整北京日取平均，% | 不是从日平均T/Td重新计算的RH |
| `heat_risk_categories_jja_2000_2024_bjt.nc`: `land_mask` | 静态0/1掩膜 | 本阶段清单只读取它和坐标，不读取最终年份标签数值 |
| `heat_index_climatology_jja_2000_2024_bjt.nc` | 按年×365日历中的JJA日期存储HI | 是逐年源数据，不是已可用于模型的全期气候态；拟合时只能选训练年份 |

当前NetCDF完整日序列为每年6月1日至8月31日，共2,300日。日坐标是无时区的**北京日历标签**，原始小时坐标为UTC；不得把日标签解析为UTC午夜。北京日D所需小时为UTC D−1日16时至D日15时，共24整点。

## 因子化样本清单

位置：`data/probability_phase1/samples/`。网格表与日期表的笛卡尔积定义候选样本，避免写出重复的331.8万行CSV。

| 字段 | 含义 |
| --- | --- |
| `grid_registry.csv`: `grid_id` | 精确经纬度浮点十六进制标识；与经纬轴排序无关。经纬度另有可读列，不按场所名重复建样本 |
| `latitude`, `longitude` | 度，ERA5格点中心；不是工厂边界点 |
| `spatial_role` | `train`为留出区及缓冲之外，`holdout`为闭矩形118–121°E、29–32°N，`buffer`为向外0.5°闭矩形扣除核心区。分别1,302、165、113个陆地格 |
| `date_manifest.csv`: `date`, `year`, `split` | 目标北京日期、年份、时间阶段；`split`与`spatial_role`为两个正交字段 |
| `issue_utc` / `target_start_utc` | D日00时BJT，即D−1日16时UTC |
| `target_end_utc_exclusive` | 次日00时BJT对应UTC，不含右端点 |
| `lag1_feature_start/end` | 七个完整日D−7至D−1，日期两端均含 |
| `lag2_feature_start/end` | 七个完整日D−8至D−2，日期两端均含 |
| `lag*_source_utc_end_exclusive` | 该臂最后历史日结束的UTC边界，不含右端点 |
| `lag*_source_utc_last_hour` | 最后允许使用的历史整点，比上述边界早一小时 |
| `lag*_eligible` | 该臂七日历史均真实存在、同年同一JJA；不代表数值有限或全部特征已提取 |
| `common_eligible` | 两臂资格交集，每夏6月9日至8月31日，共84日；正式横向对比必须筛选为真 |
| `yearly_counts.csv` | 按年份汇总日期和格日数；空间角色计数用于独立空间实验。常规跨年实验可用全1,580格，严格空间训练只能用`train`角色 |

`sample_id = grid_id + '|' + date`，同一真实样本不因模型、特征组、滞后臂或关联工厂而获得新的统计样本身份。未来预测明细还必须带`experiment_id`、模式、概率、目标标签及来源哈希。

## 完整特征与标签约束

`common_finite_mask`接收冻结的**全部特征组并集**和有限目标HI，并在所有历史分量上检查有限值；消融组也使用相同样本。不能让较复杂模型或较简单模型各自丢弃不同样本后直接比较。`schema.json`明确标为`STRUCTURAL_ONLY_NOT_FEATURE_COMPLETE`，全期夜间和平均露点完成后才可生成真正的训练样本清单。

缺失/无穷HI标签为缺失，不能编码为0。事后分组依据D−1和D：`new_onset`=0→1，`persistent`=1→1，`resolved`=1→0，`non_event`=0→0；缺前日或目标值时分组缺失。D−2实验可以在评价时用D−1做分层，但不能将此字段放入特征、条件转移预测或预先告警规则。

平均露点、最低温和完整夜间数据需从原始小时重新提取。预定夜间为D−2日20:00（含）至D−1日08:00（不含），共12整点；D−2信息臂再提前一天。定义见协议，禁止用尚未结束的D日清晨。小规模训练年提取只是实现核查，不证明全期特征完备。

## 不可直接解释的量

同一格的多个工厂不是独立气象样本；原五场所只有两个有效主气候网格。掩膜拒绝的特斯拉主格仍为空。HI及阈值是网格危害筛查，不能替代WBGT、职业安全、停工限值、企业财务损失或评级。ERA5历史回报不能称实时预警。
