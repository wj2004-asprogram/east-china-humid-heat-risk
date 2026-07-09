# Project Summary: Humid Heat Risk in East China from ERA5 Reanalysis

**Date:** 2026-07-09
**Status:** Workflow test (7-day small sample)

---

## 1. Background

Humid heat — the combined effect of high temperature and high humidity — reduces the human body's ability to cool itself through perspiration, leading to elevated risks of heat-related illness and mortality. East China, with its dense population and hot, humid summers influenced by the East Asian Monsoon, is a region of particular concern.

The **heat index** (HI) is a widely used metric that combines air temperature and relative humidity into a single "apparent temperature" value. When humidity is high, the heat index can substantially exceed the actual air temperature, capturing the physiological heat stress that dry-bulb temperature alone would miss.

This project develops a reproducible pipeline to compute humid-heat risk metrics from ERA5 reanalysis data over East China, classify risk levels, and visualise the spatial and temporal patterns.

> ⚠️ **Important:** The current analysis is a **small-sample workflow test** using only 7 days of hourly data (2022-07-01 to 2022-07-07). All results are illustrative and do **not** represent long-term climate trends or climatological norms.

## 2. Data and Study Area

### 2.1 Data Source

- **Product:** ERA5 hourly reanalysis on single levels
- **Provider:** Copernicus Climate Data Store (CDS), ECMWF
- **Horizontal resolution:** 0.25° × 0.25° (~28 km at 30°N)
- **Temporal resolution:** 1 hour

### 2.2 Study Area

| Parameter | Value |
|---|---|
| North boundary | 36°N |
| South boundary | 25°N |
| West boundary | 112°E |
| East boundary | 123°E |
| Approximate coverage | Jiangsu, Anhui, Shanghai, Zhejiang, and parts of surrounding provinces |

### 2.3 Period

- **Current test:** 2022-07-01 00:00 UTC to 2022-07-07 23:00 UTC (168 hourly time steps)
- **Planned expansion:** Full June–August (JJA), multi-year (e.g., 1991–2020)

### 2.4 Variables

| Variable | ERA5 Short Name | Native Unit |
|---|---|---|
| 2 m air temperature | `t2m` | Kelvin |
| 2 m dewpoint temperature | `d2m` | Kelvin |

### 2.5 Reference Cities

Four cities spanning the latitudinal extent of the domain were selected for point-based analysis:

| City | Latitude | Longitude |
|---|---|---|
| Nanjing | 32.06°N | 118.79°E |
| Shanghai | 31.23°N | 121.47°E |
| Hangzhou | 30.27°N | 120.15°E |
| Huai'an | 33.61°N | 119.02°E |

## 3. Methodology

### 3.1 Processing Pipeline

The analysis follows a sequential processing chain:

1. **Download** ERA5 hourly single-level data via CDS API.
2. **Unit conversion:** 2 m temperature and dewpoint from Kelvin to Celsius.
3. **Relative humidity:** Computed from temperature and dewpoint using the August–Roche–Magnus approximation.
4. **Hourly heat index:** Computed using the NOAA / Rothfusz formula.
5. **Daily aggregation:** Hourly heat index resampled to daily maximum.
6. **Risk classification:** Daily maximum heat index classified into five NOAA risk categories.
7. **Area fraction calculation:** Latitude-weighted (cos φ) area fractions computed for each risk category per day.
8. **Visualisation:** Regional mean time series, city comparisons, spatial maps, and stacked bar charts.

### 3.2 Formula Summary

#### 3.2.1 Celsius Conversion

$$T_{°C} = T_{K} - 273.15$$

#### 3.2.2 Relative Humidity (August–Roche–Magnus)

The saturation vapour pressure $e_s$ and actual vapour pressure $e$ are computed as:

$$e_s = 6.112 \cdot \exp\left(\frac{17.67 \cdot T}{T + 243.5}\right) \quad \text{[hPa]}$$

$$e = 6.112 \cdot \exp\left(\frac{17.67 \cdot T_d}{T_d + 243.5}\right) \quad \text{[hPa]}$$

Relative humidity is then:

$$RH = 100 \cdot \frac{e}{e_s} \quad \text{[%]}$$

Values are clipped to the physically plausible range [0, 100]%.

#### 3.2.3 Heat Index (NOAA / Rothfusz)

The heat index calculation follows the US National Weather Service procedure (Rothfusz 1990; Steadman 1979):

**Step 1 — Convert to Fahrenheit:**

$$T_F = \frac{9}{5} T_C + 32$$

**Step 2 — Rothfusz regression** (applied when $T_F \geq 80$ °F and $RH \geq 40$%):

$$\begin{aligned}
HI_F = &-42.379 \\
       &+ 2.04901523 \cdot T_F \\
       &+ 10.14333127 \cdot RH \\
       &- 0.22475541 \cdot T_F \cdot RH \\
       &- 0.00683783 \cdot T_F^2 \\
       &- 0.05481717 \cdot RH^2 \\
       &+ 0.00122874 \cdot T_F^2 \cdot RH \\
       &+ 0.00085282 \cdot T_F \cdot RH^2 \\
       &- 0.00000199 \cdot T_F^2 \cdot RH^2
\end{aligned}$$

**Step 3 — NOAA adjustments:**

- If $RH < 13$% and $80 < T_F < 112$ °F:

  $$HI_F \leftarrow HI_F - \frac{13 - RH}{4} \cdot \sqrt{\frac{17 - |T_F - 95|}{17}}$$

- If $RH > 85$% and $80 < T_F < 87$ °F:

  $$HI_F \leftarrow HI_F + \frac{RH - 85}{10} \cdot \frac{87 - T_F}{5}$$

**Step 4 — Simple formula** (used when $T_F < 80$ °F or $RH < 40$%):

$$HI_{simple} = 0.5 \cdot \left(T_F + 61.0 + (T_F - 68.0) \cdot 1.2 + RH \cdot 0.094\right)$$

$$HI_F = \frac{HI_{simple} + T_F}{2}$$

**Step 5 — Convert back to Celsius:**

$$HI_C = \frac{5}{9} (HI_F - 32)$$

#### 3.2.4 Heat Risk Classification

Daily maximum heat index values are classified into five categories following NOAA guidance:

| Category | Label | Threshold (°C) |
|---|---|---|
| 0 | No Risk | $HI < 26.7$ |
| 1 | Caution | $26.7 \leq HI < 32.2$ |
| 2 | Extreme Caution | $32.2 \leq HI < 40.6$ |
| 3 | Danger | $40.6 \leq HI < 54.4$ |
| 4 | Extreme Danger | $HI \geq 54.4$ |

#### 3.2.5 Latitude Area Weighting

Because the ERA5 grid uses a regular latitude–longitude projection, grid cells at higher latitudes cover a smaller surface area than those near the equator. To avoid over-weighting high-latitude cells, area fractions are computed using $\cos(\varphi)$ weighting:

$$w_j = \cos(\varphi_j)$$

where $\varphi_j$ is the latitude (in radians) of grid row $j$. The area fraction for category $k$ on a given day is:

$$f_k = \frac{\sum_{i,j} w_j \cdot \mathbf{1}[\text{category}_{i,j} = k]}{\sum_{i,j} w_j} \times 100\%$$

## 4. Results

### 4.1 Regional-Mean Time Series

The regional-mean daily maximum temperature over the 7-day period shows day-to-day variability of approximately 4–6 °C, typical of mid-summer synoptic fluctuations. The regional-mean daily maximum heat index is consistently higher than the corresponding air temperature (by approximately 2–5 °C on humid days), reflecting the moisture contribution to apparent temperature.

### 4.2 City-Level Comparison

Among the four reference cities, Nanjing, Shanghai, and Hangzhou track each other closely in both daily maximum temperature and daily maximum heat index. Huai'an, at ~33.6°N, is consistently 1–3 °C cooler. The ranking of peak heat index generally follows latitude: Hangzhou ≥ Shanghai ≈ Nanjing > Huai'an.

### 4.3 Spatial Distribution

On the day with the largest "Danger + Extreme Danger" area fraction (the "hottest risk day"), the daily maximum heat index map shows an inland maximum with values exceeding 45 °C over portions of southern Jiangsu and northern Zhejiang. Coastal grid cells near Shanghai and the East China Sea show somewhat lower values, likely due to the maritime moderating effect.

The risk category map on the same day shows "Danger" (category 3) covering substantial inland areas, "Extreme Caution" (category 2) over much of the domain, and "Caution" (category 1) limited to the northernmost and coastal fringes. "Extreme Danger" (category 4) was not reached in this 7-day sample.

### 4.4 Area Fraction Evolution

The stacked bar chart of daily risk-area fractions reveals that the fraction of the domain in the "Danger" category varies from near-zero to a substantial portion (exact values depend on the day). Days with widespread "Danger" conditions coincide with a reduced area of "No Risk," while "Extreme Caution" typically occupies the largest share of the domain on warm, humid days.

## 5. Interpretation

The 7-day test sample demonstrates that the ERA5-based humid-heat pipeline functions as intended: it produces physically plausible heat index values, risk classifications, and spatial patterns consistent with the known summer climate of East China (hot and humid inland, somewhat moderated near the coast).

The key insight from the comparison of air temperature and heat index is that **humidity contributes materially to heat stress in this region**. The difference between daily maximum temperature and daily maximum heat index ranges from ~2 °C to over 5 °C, meaning that analyses based on temperature alone would substantially underestimate physiological heat stress.

The dominance of the "Extreme Caution" and "Danger" categories on peak days is noteworthy even in this short sample, and suggests that East China regularly experiences conditions warranting heat advisories during mid-summer.

## 6. Limitations

This analysis has several important limitations:

1. **Sample size:** Only 7 days of data were analysed. No climatological conclusions can or should be drawn from this sample. Synoptic variability over a single week can easily produce conditions that are not representative of the broader summer climate.

2. **Single year:** July 2022 may have been anomalously hot or cool relative to the long-term mean. Multi-year data are required to distinguish typical conditions from anomalies.

3. **Reanalysis uncertainty:** ERA5 is a model-based reanalysis product, not direct observations. While it is among the best-validated reanalyses available, it may contain biases, particularly for humidity variables near the surface.

4. **Heat index limitations:** The NOAA heat index was developed for shaded, light-wind conditions and assumes a standard human physiology. It does not account for solar radiation, wind speed, clothing, activity level, or acclimatisation. In direct sunlight, the apparent temperature can be 8–9 °C higher.

5. **Resolution:** At 0.25° (~28 km), ERA5 cannot resolve urban heat island effects, which can add several degrees to local heat stress in cities like Shanghai and Nanjing.

6. **No exposure analysis:** Risk categories are mapped over land area but not yet weighted by population exposure.

## 7. Next Steps

1. **Expand temporal coverage:** Download and process full JJA seasons for 1991–2020 to establish a robust climatology of humid-heat risk in East China.

2. **Long-term trends:** Analyse interannual variability and long-term trends in heat index and the frequency of "Danger" days, linking to large-scale climate modes (e.g., El Niño–Southern Oscillation, West Pacific Subtropical High).

3. **Diurnal analysis:** Examine the diurnal cycle of heat stress — in particular, whether nighttime heat index remains elevated (a key factor in heat-wave mortality).

4. **Multi-index comparison:** Compare the NOAA heat index with Humidex, Wet-Bulb Globe Temperature (WBGT), Apparent Temperature, and Universal Thermal Climate Index (UTCI) to assess sensitivity to index choice.

5. **Population exposure:** Overlay gridded population data (e.g., GPWv4, GHS-POP) to estimate person-days of exposure to each risk category.

6. **Synoptic compositing:** Identify the weather patterns associated with the most severe humid-heat days and examine their frequency trends.

7. **Future projections:** Apply the same methodology to CMIP6 downscaled projections to assess how humid-heat risk may change under different warming scenarios.

---

*This summary accompanies the East China Humid Heat Risk project repository. See `README.md` for technical setup and usage instructions.*
