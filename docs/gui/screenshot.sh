#!/bin/bash
# This script takes screenshot from the GUI, removes the whitespace and adds labels

gui_host="http://localhost:8080/"

declare -A screenshots
screenshots['welcome']=$gui_host
screenshots['jobs']=$gui_host"jobs"
screenshots['jobs-runner']=$gui_host"jobs/runner/20170918_energy_water_short"
screenshots['configure']=$gui_host"configure/sos-model-run"
screenshots['configure-sos-model-run']=$gui_host"configure/sos-model-run/20170918_energy_water"
screenshots['configure-sos-models']=$gui_host"configure/sos-models/energy_waste"
screenshots['configure-sector-models']=$gui_host"configure/sector-models/water_supply"
screenshots['configure-scenario-set']=$gui_host"configure/scenario-set/population"
screenshots['configure-narrative-set']=$gui_host"configure/narrative-set/technology"
screenshots['configure-narratives']=$gui_host"configure/narratives/High%20Tech%20Demand%20Side%20Management"

# If argument provided, only process the key supplied
if [ $# -ne 0 ]
then
	for screenshot in ${!screenshots[@]}
	do

		if [ ${screenshot} != $1 ]
		then
			unset screenshots[${screenshot}]
		fi
	done
fi

# Remove whitespace
for screenshot in ${!screenshots[@]}
do
	echo "Taking screenshot for: "${screenshot} ${screenshots[${screenshot}]}
	cutycapt --url=${screenshots[${screenshot}]} --out=_${screenshot}.png --min-width=1920 --min-height=1200 --delay=500
	convert _${screenshot}.png -trim ${screenshot}.png
	convert -border 20 +repage -bordercolor white ${screenshot}.png ${screenshot}.png
	rm _${screenshot}.png
	screenshots[${screenshot}]="${screenshot}.png"
done

# Add numbers
add_label () {
	convert -draw 'text '$1','$2'"'$3'" ' -fill '#0008' -font Ubuntu -undercolor grey -stroke black -strokewidth 1 -pointsize $4 $5 $5
}

for screenshot in ${!screenshots[@]}
do
	if [ "${screenshot}" == "welcome" ]
	then
		echo 'Add labels to: '${screenshots[${screenshot}]}
		add_label 320 130 "A" 35 ${screenshots[${screenshot}]}
		add_label 320 210 "B" 35 ${screenshots[${screenshot}]}
		add_label 320 245 "C" 35 ${screenshots[${screenshot}]}
		add_label 320 285 "D" 35 ${screenshots[${screenshot}]}
		add_label 320 325 "E" 35 ${screenshots[${screenshot}]}
	fi

	if [ "${screenshot}" == "jobs" ]
	then
		echo 'Add labels to: '${screenshots[${screenshot}]}
		add_label 220 170 "A" 35 ${screenshots[${screenshot}]}
		add_label 220 205 "B" 35 ${screenshots[${screenshot}]}
		add_label 220 240 "C" 35 ${screenshots[${screenshot}]}
		add_label 330 365 "D" 50 ${screenshots[${screenshot}]}
	fi

	if [ "${screenshot}" == "jobs-runner" ]
	then
		echo 'Add labels to: '${screenshots[${screenshot}]}
		add_label 344 80 "1" 50 ${screenshots[${screenshot}]}
		add_label 344 200 "2" 50 ${screenshots[${screenshot}]}
		add_label 344 595 "3" 50 ${screenshots[${screenshot}]}
		add_label 344 970 "4" 50 ${screenshots[${screenshot}]}

		add_label 600 595 "A" 50 ${screenshots[${screenshot}]}
		add_label 600 825 "B" 50 ${screenshots[${screenshot}]}
		add_label 1350 970 "C" 50 ${screenshots[${screenshot}]}
		add_label 1510 970 "D" 50 ${screenshots[${screenshot}]}
	fi

	if [ "${screenshot}" == "configure" ]
	then
		echo 'Add labels to: '${screenshots[${screenshot}]}
		add_label 610 225 "A" 50 ${screenshots[${screenshot}]}
		add_label 344 380 "B" 50 ${screenshots[${screenshot}]}
		add_label 1500 380 "C" 50 ${screenshots[${screenshot}]}
	fi

	if [ "${screenshot}" == "configure-sos-model-run" ]
	then
		echo 'Add labels to: '${screenshots[${screenshot}]}
		add_label 344 140  "1" 50 ${screenshots[${screenshot}]}
		add_label 344 190  "2" 50 ${screenshots[${screenshot}]}
		add_label 344 350  "3" 50 ${screenshots[${screenshot}]}
		add_label 344 520  "4" 50 ${screenshots[${screenshot}]}
		add_label 344 680  "5" 50 ${screenshots[${screenshot}]}
		add_label 344 1000 "6" 50 ${screenshots[${screenshot}]}
		add_label 344 1190 "7" 50 ${screenshots[${screenshot}]}
		add_label 344 1240 "8" 50 ${screenshots[${screenshot}]}
		add_label 344 1290 "9" 50 ${screenshots[${screenshot}]}

		add_label 530 1380 "A" 50 ${screenshots[${screenshot}]}
	fi

	if [ "${screenshot}" == "configure-sos-models" ]
	then
		echo 'Add labels to: '${screenshots[${screenshot}]}
		add_label 344 140  "1" 50 ${screenshots[${screenshot}]}
		add_label 344 190  "2" 50 ${screenshots[${screenshot}]}
		add_label 344 440  "3" 50 ${screenshots[${screenshot}]}
		add_label 344 560  "4" 50 ${screenshots[${screenshot}]}
		add_label 344 670  "5" 50 ${screenshots[${screenshot}]}
		add_label 344 870  "6" 50 ${screenshots[${screenshot}]}
		add_label 344 1440 "7" 50 ${screenshots[${screenshot}]}
		add_label 344 1500 "8" 50 ${screenshots[${screenshot}]}
		add_label 344 1560 "9" 50 ${screenshots[${screenshot}]}

		add_label 560 1290 "A" 50 ${screenshots[${screenshot}]}
		add_label 530 1670 "B" 50 ${screenshots[${screenshot}]}
	fi

	if [ "${screenshot}" == "configure-sector-models" ]
	then
		echo 'Add labels to: '${screenshots[${screenshot}]}
		add_label 344 140  "1" 50 ${screenshots[${screenshot}]}
		add_label 344 190  "2" 50 ${screenshots[${screenshot}]}
		add_label 344 440  "3" 50 ${screenshots[${screenshot}]}
		add_label 344 490  "4" 50 ${screenshots[${screenshot}]}
		add_label 344 660  "5" 50 ${screenshots[${screenshot}]}
		add_label 344 1080 "6" 50 ${screenshots[${screenshot}]}
		add_label 344 1510 "7" 50 ${screenshots[${screenshot}]}

		add_label 530 930  "A" 50 ${screenshots[${screenshot}]}
		add_label 530 1350 "B" 50 ${screenshots[${screenshot}]}
		add_label 530 1720 "C" 50 ${screenshots[${screenshot}]}
		add_label 530 1810 "D" 50 ${screenshots[${screenshot}]}
	fi

	if [ "${screenshot}" == "configure-scenario-set" ]
	then
		echo 'Add labels to: '${screenshots[${screenshot}]}
		add_label 344 140 "1" 50 ${screenshots[${screenshot}]}
		add_label 344 190 "2" 50 ${screenshots[${screenshot}]}
		add_label 344 450 "3" 50 ${screenshots[${screenshot}]}
		add_label 344 750 "4" 50 ${screenshots[${screenshot}]}

		add_label 530 585  "A" 50 ${screenshots[${screenshot}]}
		add_label 530 1020  "B" 50 ${screenshots[${screenshot}]}
		add_label 530 1110  "C" 50 ${screenshots[${screenshot}]}
	fi

	if [ "${screenshot}" == "configure-narrative-set" ]
	then
		echo 'Add labels to: '${screenshots[${screenshot}]}
		add_label 344 140 "1" 50 ${screenshots[${screenshot}]}
		add_label 344 190 "2" 50 ${screenshots[${screenshot}]}

		add_label 530 380  "A" 50 ${screenshots[${screenshot}]}
	fi

	if [ "${screenshot}" == "configure-narratives" ]
	then
		echo 'Add labels to: '${screenshots[${screenshot}]}
		add_label 344 140  "1" 50 ${screenshots[${screenshot}]}
		add_label 344 190  "2" 50 ${screenshots[${screenshot}]}
		add_label 344 440  "3" 50 ${screenshots[${screenshot}]}

		add_label 530 780  "A" 50 ${screenshots[${screenshot}]}
	fi

done


exit 0
