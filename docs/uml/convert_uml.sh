# Generate UML images

# PlantUML converts plain text diagram notation to images
#
# Requires plantuml jar from http://plantuml.com/download to provide plantuml
# command. 
#
# Depends on GraphViz from http://www.graphviz.org/Download..php to create
# diagrams. E.g. set environment variable
# - export GRAPHVIZ_DOT=~/bin/graphviz-2.38/release/bin/dot.exe
#
# Docs:
# UML class notation: http://plantuml.com/class-diagram
# Colour and style parameters: http://plantuml.com/skinparam
# Syntax highlight plugin (for VS Code): Yog PlantUML Highlight
# Preview plugin (for VS Code): PlantUML by jebbs
if [ $# -eq 0 ]
  then
    echo "Please provide path to plantuml.jar"
    exit 1
fi

for image in $(ls ./*uml); do
  echo "Converting $image"
  java -jar $1 $image
done;
