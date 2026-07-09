#!/usr/bin/env bash
set -e

echo "=== Step 1/7: Processing temperature data ==="
python src/process_temperature.py

echo "=== Step 2/7: Plotting figures ==="
python src/plot_figures.py

echo "=== Step 3/7: Plotting city Tmax ==="
python src/plot_city_tmax.py

echo "=== Step 4/7: Processing humid heat ==="
python src/process_humid_heat.py

echo "=== Step 5/7: Plotting heat index figures ==="
python src/plot_heat_index_figures.py

echo "=== Step 6/7: Classifying heat risk ==="
python src/classify_heat_risk.py

echo "=== Step 7/7: Plotting heat risk maps ==="
python src/plot_heat_risk_maps.py

echo "=== Sample workflow completed. ==="
