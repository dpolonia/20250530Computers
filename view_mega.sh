#!/bin/bash

# Script to view the mega-result with interactive wait points

echo -e "\033[1;36m========================================================\033[0m"
echo -e "\033[1;36mMEGA-RESULT VIEWER WITH INTERACTIVE STOPS\033[0m"
echo -e "\033[1;36m========================================================\033[0m"

# The run ID of the mega-result
RUN_ID="20250531082844"

# Create a temporary file with inputs for the interactive wait points
cat > temp_inputs.txt << EOF
y
EOF

# Run paper_revision.py with view-mega
echo -e "\033[1;34m[INFO]\033[0m Viewing mega-result with run ID: $RUN_ID"
python paper_revision.py --view-mega $RUN_ID < temp_inputs.txt

# Clean up
rm temp_inputs.txt

echo -e "\033[1;32m[COMPLETE]\033[0m Mega-result viewing completed!"