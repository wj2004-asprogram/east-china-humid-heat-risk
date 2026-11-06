# 首批五场所：日最高热指数方法核查

核查日期：2026-11-05。对应[首批五场所研究方案 v1.0](ESG_FIVE_SITE_PROTOCOL.md)。本记录检查现有代码、配置、单元测试与官方方法资料，不改变方案中的阈值、选年、连续过程或空间情景规则。

本轮主案例有5个场所、3个主匹配网格，其中2格具有有效主气候序列，对应50个不同网格年。这里核查的是这些日值所依据的方法链，不把同格场所当作重复气候观测。

## 计算链与源码定位

| 环节 | 实际实现 | 源码 |
| --- | --- | --- |
| 原变量与单位 | 读取2米温度、2米露点温度；各减273.15，将K转为°C | [process_humid_heat.py，第64行](../src/process_humid_heat.py#L64)；[indices.py，第18行](../src/indices.py#L18) |
| 相对湿度RH | 采用水面饱和水汽压近似 `es(T)=6.112 exp(17.67 T/(T+243.5))`，计算 `100 es(Td)/es(T)`；结果截断至[0,100] | [indices.py，第37行](../src/indices.py#L37) |
| 小时HI | °C转°F；计算简单公式；用简单公式与气温的平均值判断是否达到80°F，再选择Rothfusz回归或简单公式；最后转回°C | [indices.py，第105行](../src/indices.py#L105) |
| 回归调整 | 回归分支中，RH<13%且80≤T≤112°F采用低湿修正；RH>85%且80≤T≤87°F采用高湿修正 | [indices.py，第72行](../src/indices.py#L72) |
| 日最高HI | 小时结果先按UTC+8日界归组，再取每日最大值；不是由日最高气温与日平均RH重新组合HI | [process_humid_heat.py，第119行](../src/process_humid_heat.py#L119)；[time_utils.py，第37行](../src/time_utils.py#L37) |
| 陆地筛选 | 原陆地比例满足 `lsm >= 0.5` 时输出掩膜1；场所筛查只接受1，不用其他陆地格替换被拒绝主格 | [land_sea_mask.py，第17行](../src/land_sea_mask.py#L17)；[esg_site_screen.py，第104行](../tools/esg_site_screen.py#L104) |
| 完整夏季 | JJA预期92日，日期齐全且日值全部有限才计算阈值天数和最长连续日数；判定为 `HI >= threshold_c` | [esg_site_screen.py，第42行](../tools/esg_site_screen.py#L42) |

ECMWF资料列明2米温度和露点的单位为K，并说明近地面RH可由 `100 es(Td)/es(T)` 推导。项目采用的是明确系数的饱和水汽压近似，不是直接读取ERA5近地面RH，也不宣称完整复现IFS湿度参数化。[ECMWF ERA5当前技术文档](https://confluence.ecmwf.int/pages/viewpage.action?pageId=76414402)

NWS官方说明提供Rothfusz回归、两项RH调整及约80°F处分支判断，并指出回归不适用于超出Steadman基础数据范围的极端温湿条件。项目低值分支返回简单公式本身；与气温的平均值用于分支判断。这个实现细节已在源码和测试中记录，不能仅以“使用NOAA公式”省略其定义。[NWS Tampa公式说明](https://www.weather.gov/tbw/heatindex)

## 日界、缺小时与缺值

配置明确采用UTC+8：北京日D由D−1日16:00 UTC至D日15:00 UTC的24个小时组成。每年6月1日需要5月31日16:00—23:00 UTC的边界小时。配置出处为 [jja_2000_2024_bjt.yaml，第27行](../configs/jja_2000_2024_bjt.yaml#L27)。

[local_daily实现](../src/time_utils.py#L72) 区分两类问题：

- **小时记录缺失**：目标日没有完整24个不同整点时，函数报错；重复时间或非整点时间也报错。目标期之外的边缘不完整日可以丢弃。
- **记录存在但数值缺失**：采用 `skipna=False`，任何小时为NaN使该格当日HI为NaN，不对剩余小时取最大值补齐。

进入场所研究后，年度检查还要求92个日值全部有限；不完整夏季的年度指标为空。连续过程遵循研究方案，不跨缺日期、缺值或季节连接。

## 40.6°C阈值的来源和表述

主阈值继续固定为 **HI≥40.6°C**，与研究方案一致。它继承 [risk.py，第29行](../src/risk.py#L29) 的固定摄氏分界，由 [esg_site_screen.py，第130行](../tools/esg_site_screen.py#L130) 作为默认值读取。原代码在 [risk.py，第7行](../src/risk.py#L7) 将分级称为“基于NOAA热指数阈值”，项目主README也列出 [40.6—54.4°C的Danger分级](../README.md#L94)。

官方发布材料中存在不同的分级边界，不能把其中一页概括为所有NOAA/NWS产品共同采用的唯一标准：

| 官方资料 | Danger区间 | Extreme Danger起点 |
| --- | --- | --- |
| [NWS Tulsa热指数页面](https://www.weather.gov/tsa/heatindex) | 105—129°F | 130°F |
| [NWS Amarillo热指数页面](https://www.weather.gov/ama/heatindex) | 103—124°F | 125°F |

105°F精确换算为40.5556°C，一位小数为40.6°C；40.6°C本身对应105.08°F，二者并不完全相等。103°F则为39.4444°C。因此，本轮采用的准确表述是：**继承项目固定的40.6°C操作阈值，接近105°F口径的一位小数近似；本轮不更换阈值，也不将其称为统一官方分级或职业限值。** “阈值连续过程”按方案定义，不直接称为正式热浪事件。

## 掩膜、空间产品与元数据

0.5是本项目配置中的陆地比例筛选阈值，见 [jja_2000_2024_bjt.yaml，第42行](../configs/jja_2000_2024_bjt.yaml#L42)；[classify_heat_risk.py，第159行](../src/classify_heat_risk.py#L159) 将实际规则写入输出属性。它不是对厂区实际海陆位置的判定，也不应被称为唯一的ERA5官方陆地标准。被拒绝的主格继续为空。

CDS提供的0.25°规则经纬度资料是从ERA5原生分辨率再网格化的产品。本研究按经纬轴最近中心构造的分区是**匹配分区**；多边形相交面积描述地图厂区落入各匹配分区的比例，不能解释为厂内热暴露实测面积。[CDS ERA5单层小时数据说明](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview)

既有日HI变量同时保留派生的 `units=Celsius` 与继承的 `GRIB_units=K`、气温名称等属性。当前源码在 [process_humid_heat.py，第90行](../src/process_humid_heat.py#L90) 和 [第130行](../src/process_humid_heat.py#L130) 明确计算并标记摄氏HI。因此研究包采用派生变量定义，保留冲突属性供审查，不对现成HI日值再次减273.15，也不修改原NetCDF。

HI使用温度和湿度，适用解释以阴影、弱风条件为基础；WBGT还考虑风、太阳辐射等参数。当前网格HI不等于WBGT，不能单独支持厂内作业暴露或职业合规结论。[NWS热环境工具说明](https://www.weather.gov/safety/heat-index)

## 验证记录与未验证范围

2026-11-05在既有科学Python环境运行以下针对性测试，结果为 **30项通过，4条警告**：

```powershell
python -m pytest tests/test_indices.py tests/test_time_utils.py tests/test_land_sea_mask.py -q -p no:cacheprovider
```

- [test_indices.py](../tests/test_indices.py) 检查单位、RH、NWS图表样例、与MetPy的选定热/冷条件比较，以及RH与分支边界。MetPy比较在本次环境实际执行，未跳过；它不代表所有温湿组合下各实现完全相同。
- [test_time_utils.py](../tests/test_time_utils.py) 检查北京日界、跨月、重复/非整点、缺小时及NaN传播。
- [test_land_sea_mask.py](../tests/test_land_sea_mask.py) 检查0.5边界、其他阈值与坐标不匹配。

4条警告均发生在掩膜文件测试的NetCDF依赖路径：1条 `numpy.ndarray size changed` 运行时兼容性提示，3条NumPy数组shape设置的弃用警告。测试成功；未据此认定已发生数值错误，本轮也未升级依赖。

本次确认的是**当前实现、配置与已保存数据定义的可追溯一致性**。没有重跑25个夏季的原始小时处理，没有检查所有原始小时温湿值是否处于回归适用范围，也没有用场站观测独立验证ERA5。因此不将此次检查称为对源NetCDF的独立再现或对现场热环境的验证。上述官方资料于2026-11-05查阅；场所身份与地址证据快照仍保持2026-10-29。
