# Generate UML images

# pyreverse deduces class and package structure from code
#
# Command is bundled with pylint
# - pip install pylint
pyreverse smif -p smif -o png

# PlantUML converts plain text diagram notation to images
#
# Requires plantuml jar from http://plantuml.com/download to provide plantuml
# command. E.g. create bash function:
#   plantuml()
#   {
#     java -jar ~/bin/plantuml-jar-mit-1.2017.14/plantuml.jar $1
#   }
#   export -f plantuml
#
# Depends on GraphViz from http://www.graphviz.org/Download..php to create
# diagrams. E.g. set environment variable
# - export GRAPHVIZ_DOT=~/bin/graphviz-2.38/release/bin/dot.exe
#
# Docs:
# UML class notation: http://plantuml.com/class-diagram
# Colour and style parameters: http://plantuml.com/skinparam
# Syntax highlight plugin (for VS Code): Yog PlantUML Highlight
run_plantuml ./*.uml
