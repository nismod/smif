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

# Remove whitespace
for screenshot in ${!screenshots[@]}
do
	echo ${screenshot} ${screenshots[${screenshot}]}
	cutycapt --url=${screenshots[${screenshot}]} --out=_${screenshot}.png --min-width=1920 --min-height=1200 --delay=100
	convert _${screenshot}.png -trim ${screenshot}.png
	convert -border 20 +repage -bordercolor white ${screenshot}.png ${screenshot}.png
	rm _${screenshot}.png
	screenshots[${screenshot}]="${screenshot}.png"
done

# Add numbers
for screenshot in ${!screenshots[@]}
do
	if [ "${screenshot}" == "welcome" ]
	then
		convert -draw 'text 10,355  "A" ' -fill red -font Ubuntu -undercolor grey -stroke black -strokewidth 2 -pointsize 50 ${screenshots[${screenshot}]} ${screenshots[${screenshot}]}
	fi

	if [ "${screenshot}" == "configure" ]
	then
		convert -draw 'text 4,150  "1" ' -fill red -font Ubuntu -undercolor grey -stroke black -strokewidth 2 -pointsize 50 ${screenshots[${screenshot}]} ${screenshots[${screenshot}]}
		convert -draw 'text 4,350  "2" ' -fill red -font Ubuntu -undercolor grey -stroke black -strokewidth 2 -pointsize 50 ${screenshots[${screenshot}]} ${screenshots[${screenshot}]}
		convert -draw 'text 4,750  "3" ' -fill red -font Ubuntu -undercolor grey -stroke black -strokewidth 2 -pointsize 50 ${screenshots[${screenshot}]} ${screenshots[${screenshot}]}
		convert -draw 'text 4,1135  "4" ' -fill red -font Ubuntu -undercolor grey -stroke black -strokewidth 2 -pointsize 50 ${screenshots[${screenshot}]} ${screenshots[${screenshot}]}
		convert -draw 'text 4,1515  "5" ' -fill red -font Ubuntu -undercolor grey -stroke black -strokewidth 2 -pointsize 50 ${screenshots[${screenshot}]} ${screenshots[${screenshot}]}
		convert -draw 'text 4,1890  "6" ' -fill red -font Ubuntu -undercolor grey -stroke black -strokewidth 2 -pointsize 50 ${screenshots[${screenshot}]} ${screenshots[${screenshot}]}
		convert -draw 'text 4,2200  "7" ' -fill red -font Ubuntu -undercolor grey -stroke black -strokewidth 2 -pointsize 50 ${screenshots[${screenshot}]} ${screenshots[${screenshot}]}

		convert -draw 'text 400,570  "A" ' -fill red -font Ubuntu -undercolor grey -stroke black -strokewidth 2 -pointsize 50 ${screenshots[${screenshot}]} ${screenshots[${screenshot}]}
		convert -draw 'text 850,410  "B" ' -fill red -font Ubuntu -undercolor grey -stroke black -strokewidth 2 -pointsize 50 ${screenshots[${screenshot}]} ${screenshots[${screenshot}]}
		convert -draw 'text 950,410  "C" ' -fill red -font Ubuntu -undercolor grey -stroke black -strokewidth 2 -pointsize 50 ${screenshots[${screenshot}]} ${screenshots[${screenshot}]}
	fi

	if [ "${screenshot}" == "configure-sos-model-run" ]
	then
		convert -draw 'text 4,210  "1" ' -fill red -font Ubuntu -undercolor grey -stroke black -strokewidth 2 -pointsize 50 ${screenshots[${screenshot}]} ${screenshots[${screenshot}]}
		convert -draw 'text 4,280  "2" ' -fill red -font Ubuntu -undercolor grey -stroke black -strokewidth 2 -pointsize 50 ${screenshots[${screenshot}]} ${screenshots[${screenshot}]}
		convert -draw 'text 4,430  "3" ' -fill red -font Ubuntu -undercolor grey -stroke black -strokewidth 2 -pointsize 50 ${screenshots[${screenshot}]} ${screenshots[${screenshot}]}
		convert -draw 'text 4,610  "4" ' -fill red -font Ubuntu -undercolor grey -stroke black -strokewidth 2 -pointsize 50 ${screenshots[${screenshot}]} ${screenshots[${screenshot}]}
		convert -draw 'text 4,800  "5" ' -fill red -font Ubuntu -undercolor grey -stroke black -strokewidth 2 -pointsize 50 ${screenshots[${screenshot}]} ${screenshots[${screenshot}]}
		convert -draw 'text 4,1120  "6" ' -fill red -font Ubuntu -undercolor grey -stroke black -strokewidth 2 -pointsize 50 ${screenshots[${screenshot}]} ${screenshots[${screenshot}]}
		convert -draw 'text 4,1310  "7" ' -fill red -font Ubuntu -undercolor grey -stroke black -strokewidth 2 -pointsize 50 ${screenshots[${screenshot}]} ${screenshots[${screenshot}]}
		convert -draw 'text 4,1360  "8" ' -fill red -font Ubuntu -undercolor grey -stroke black -strokewidth 2 -pointsize 50 ${screenshots[${screenshot}]} ${screenshots[${screenshot}]}
		convert -draw 'text 4,1410  "9" ' -fill red -font Ubuntu -undercolor grey -stroke black -strokewidth 2 -pointsize 50 ${screenshots[${screenshot}]} ${screenshots[${screenshot}]}

		convert -draw 'text 440,1510  "A" ' -fill red -font Ubuntu -undercolor grey -stroke black -strokewidth 2 -pointsize 50 ${screenshots[${screenshot}]} ${screenshots[${screenshot}]}
		convert -draw 'text 440,1560  "B" ' -fill red -font Ubuntu -undercolor grey -stroke black -strokewidth 2 -pointsize 50 ${screenshots[${screenshot}]} ${screenshots[${screenshot}]}
	fi

	if [ "${screenshot}" == "configure-sos-models" ]
	then
		convert -draw 'text 4,220  "1" ' -fill red -font Ubuntu -undercolor grey -stroke black -strokewidth 2 -pointsize 50 ${screenshots[${screenshot}]} ${screenshots[${screenshot}]}
		convert -draw 'text 4,280  "2" ' -fill red -font Ubuntu -undercolor grey -stroke black -strokewidth 2 -pointsize 50 ${screenshots[${screenshot}]} ${screenshots[${screenshot}]}
		convert -draw 'text 4,550  "3" ' -fill red -font Ubuntu -undercolor grey -stroke black -strokewidth 2 -pointsize 50 ${screenshots[${screenshot}]} ${screenshots[${screenshot}]}
		convert -draw 'text 4,650  "4" ' -fill red -font Ubuntu -undercolor grey -stroke black -strokewidth 2 -pointsize 50 ${screenshots[${screenshot}]} ${screenshots[${screenshot}]}
		convert -draw 'text 4,760  "5" ' -fill red -font Ubuntu -undercolor grey -stroke black -strokewidth 2 -pointsize 50 ${screenshots[${screenshot}]} ${screenshots[${screenshot}]}
		convert -draw 'text 4,950  "6" ' -fill red -font Ubuntu -undercolor grey -stroke black -strokewidth 2 -pointsize 50 ${screenshots[${screenshot}]} ${screenshots[${screenshot}]}
		convert -draw 'text 4,1500  "7" ' -fill red -font Ubuntu -undercolor grey -stroke black -strokewidth 2 -pointsize 50 ${screenshots[${screenshot}]} ${screenshots[${screenshot}]}
		convert -draw 'text 4,1570  "8" ' -fill red -font Ubuntu -undercolor grey -stroke black -strokewidth 2 -pointsize 50 ${screenshots[${screenshot}]} ${screenshots[${screenshot}]}
		convert -draw 'text 4,1680  "9" ' -fill red -font Ubuntu -undercolor grey -stroke black -strokewidth 2 -pointsize 50 ${screenshots[${screenshot}]} ${screenshots[${screenshot}]}

		convert -draw 'text 440,1320  "A" ' -fill red -font Ubuntu -undercolor grey -stroke black -strokewidth 2 -pointsize 50 ${screenshots[${screenshot}]} ${screenshots[${screenshot}]}
		convert -draw 'text 440,1810  "B" ' -fill red -font Ubuntu -undercolor grey -stroke black -strokewidth 2 -pointsize 50 ${screenshots[${screenshot}]} ${screenshots[${screenshot}]}
		convert -draw 'text 440,1860  "C" ' -fill red -font Ubuntu -undercolor grey -stroke black -strokewidth 2 -pointsize 50 ${screenshots[${screenshot}]} ${screenshots[${screenshot}]}
	fi

	if [ "${screenshot}" == "configure-sector-models" ]
	then
		convert -draw 'text 4,220  "1" ' -fill red -font Ubuntu -undercolor grey -stroke black -strokewidth 2 -pointsize 50 ${screenshots[${screenshot}]} ${screenshots[${screenshot}]}
		convert -draw 'text 4,280  "2" ' -fill red -font Ubuntu -undercolor grey -stroke black -strokewidth 2 -pointsize 50 ${screenshots[${screenshot}]} ${screenshots[${screenshot}]}
		convert -draw 'text 4,540  "3" ' -fill red -font Ubuntu -undercolor grey -stroke black -strokewidth 2 -pointsize 50 ${screenshots[${screenshot}]} ${screenshots[${screenshot}]}
		convert -draw 'text 4,600  "4" ' -fill red -font Ubuntu -undercolor grey -stroke black -strokewidth 2 -pointsize 50 ${screenshots[${screenshot}]} ${screenshots[${screenshot}]}
		convert -draw 'text 4,760  "5" ' -fill red -font Ubuntu -undercolor grey -stroke black -strokewidth 2 -pointsize 50 ${screenshots[${screenshot}]} ${screenshots[${screenshot}]}
		convert -draw 'text 4,1180  "6" ' -fill red -font Ubuntu -undercolor grey -stroke black -strokewidth 2 -pointsize 50 ${screenshots[${screenshot}]} ${screenshots[${screenshot}]}
		convert -draw 'text 4,1600  "7" ' -fill red -font Ubuntu -undercolor grey -stroke black -strokewidth 2 -pointsize 50 ${screenshots[${screenshot}]} ${screenshots[${screenshot}]}

		convert -draw 'text 440,1010  "A" ' -fill red -font Ubuntu -undercolor grey -stroke black -strokewidth 2 -pointsize 50 ${screenshots[${screenshot}]} ${screenshots[${screenshot}]}
		convert -draw 'text 440,1430  "B" ' -fill red -font Ubuntu -undercolor grey -stroke black -strokewidth 2 -pointsize 50 ${screenshots[${screenshot}]} ${screenshots[${screenshot}]}
		convert -draw 'text 440,1895  "C" ' -fill red -font Ubuntu -undercolor grey -stroke black -strokewidth 2 -pointsize 50 ${screenshots[${screenshot}]} ${screenshots[${screenshot}]}
		convert -draw 'text 440,1995  "D" ' -fill red -font Ubuntu -undercolor grey -stroke black -strokewidth 2 -pointsize 50 ${screenshots[${screenshot}]} ${screenshots[${screenshot}]}
		convert -draw 'text 440,2045  "E" ' -fill red -font Ubuntu -undercolor grey -stroke black -strokewidth 2 -pointsize 50 ${screenshots[${screenshot}]} ${screenshots[${screenshot}]}
	fi

	if [ "${screenshot}" == "configure-scenario-set" ]
	then
		convert -draw 'text 4,220  "1" ' -fill red -font Ubuntu -undercolor grey -stroke black -strokewidth 2 -pointsize 50 ${screenshots[${screenshot}]} ${screenshots[${screenshot}]}
		convert -draw 'text 4,280  "2" ' -fill red -font Ubuntu -undercolor grey -stroke black -strokewidth 2 -pointsize 50 ${screenshots[${screenshot}]} ${screenshots[${screenshot}]}
		convert -draw 'text 4,540  "3" ' -fill red -font Ubuntu -undercolor grey -stroke black -strokewidth 2 -pointsize 50 ${screenshots[${screenshot}]} ${screenshots[${screenshot}]}
		convert -draw 'text 4,830  "4" ' -fill red -font Ubuntu -undercolor grey -stroke black -strokewidth 2 -pointsize 50 ${screenshots[${screenshot}]} ${screenshots[${screenshot}]}
	fi

	if [ "${screenshot}" == "configure-narrative-set" ]
	then
		convert -draw 'text 4,220  "1" ' -fill red -font Ubuntu -undercolor grey -stroke black -strokewidth 2 -pointsize 50 ${screenshots[${screenshot}]} ${screenshots[${screenshot}]}
		convert -draw 'text 4,280  "2" ' -fill red -font Ubuntu -undercolor grey -stroke black -strokewidth 2 -pointsize 50 ${screenshots[${screenshot}]} ${screenshots[${screenshot}]}
	fi

	if [ "${screenshot}" == "configure-narratives" ]
	then
		convert -draw 'text 4,220  "1" ' -fill red -font Ubuntu -undercolor grey -stroke black -strokewidth 2 -pointsize 50 ${screenshots[${screenshot}]} ${screenshots[${screenshot}]}
		convert -draw 'text 4,280  "2" ' -fill red -font Ubuntu -undercolor grey -stroke black -strokewidth 2 -pointsize 50 ${screenshots[${screenshot}]} ${screenshots[${screenshot}]}
		convert -draw 'text 4,540  "3" ' -fill red -font Ubuntu -undercolor grey -stroke black -strokewidth 2 -pointsize 50 ${screenshots[${screenshot}]} ${screenshots[${screenshot}]}
		convert -draw 'text 4,790  "4" ' -fill red -font Ubuntu -undercolor grey -stroke black -strokewidth 2 -pointsize 50 ${screenshots[${screenshot}]} ${screenshots[${screenshot}]}
	fi
	
done


exit 0
