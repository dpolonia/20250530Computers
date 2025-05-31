#!/bin/bash

# Script to run paper_revision.py in final mode with Google model
# with interactive wait points enabled - fixed after token limit fixes

echo -e "\033[1;36m========================================================\033[0m"
echo -e "\033[1;36mPAPER REVISION TOOL - FINAL MODE WITH INTERACTIVE STOPS\033[0m"
echo -e "\033[1;36m========================================================\033[0m"

# Create a temporary file with inputs for the interactive wait points
cat > temp_inputs.txt << EOF
y
EOF

# Run paper_revision.py with explicit provider and model
echo -e "\033[1;34m[INFO]\033[0m Running paper_revision.py in final mode with Google model..."
python paper_revision.py --mode final --provider google --model "gemini-1.5-pro" < temp_inputs.txt

# Clean up
rm temp_inputs.txt

echo -e "\033[1;32m[COMPLETE]\033[0m Final mode execution completed!"