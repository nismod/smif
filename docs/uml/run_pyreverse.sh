# pyreverse deduces class and package structure from code
#
# Command is bundled with pylint
# - pip install pylint
echo "Running pyreverse"
pyreverse smif -p smif -o png
