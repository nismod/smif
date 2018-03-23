#!/bin/bash
# This script takes screenshot from the GUI and removes the whitespace

declare -A screenshots
screenshots['welcome']="http://localhost:5000/"
screenshots['configure']="http://localhost:5000/configure"
screenshots['configure-sos-model-run']="http://localhost:5000/configure/sos-model-run/20170918_energy_water"
screenshots['configure-sos-models']="http://localhost:5000/configure/sos-models/energy_waste"
screenshots['configure-sector-models']="http://localhost:5000/configure/sector-models/water_supply"
screenshots['configure-scenario-set']="http://localhost:5000/configure/scenario-set/population"
screenshots['configure-narrative-set']="http://localhost:5000/configure/narrative-set/technology"
screenshots['configure-narratives']="http://localhost:5000/configure/narratives/High%20Tech%20Demand%20Side%20Management"

for screenshot in ${!screenshots[@]}
do
	echo ${screenshot} ${screenshots[${screenshot}]}
	cutycapt --url=${screenshots[${screenshot}]} --out=_${screenshot}.png --min-width=1920 --min-height=1200 --delay=100
	convert _${screenshot}.png -trim ${screenshot}.png
	convert -border 20 -bordercolor white ${screenshot}.png ${screenshot}.png
	rm _${screenshot}.png
done

exit 0
