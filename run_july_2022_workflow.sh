#!/usr/bin/env bash

set -e

echo "Running July 2022 ERA5 humid heat risk workflow..."

python src/process_temperature_july_2022.py
python src/plot_figures_july_2022.py
python src/plot_city_tmax_july_2022.py
python src/process_humid_heat_july_2022.py
python src/plot_heat_index_figures_july_2022.py
python src/classify_heat_risk_july_2022.py
python src/plot_heat_risk_maps_july_2022.py

echo "July 2022 workflow completed."
