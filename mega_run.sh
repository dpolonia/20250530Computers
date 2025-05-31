#!/bin/bash

# Mega-Run Shell Script
# This script runs the paper_revision.py with mega-result creation and automatically
# generates the mega-result directory with all output files.

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Print header
echo -e "${CYAN}=====================================================================${NC}"
echo -e "${CYAN}MEGA-RESULT AUTOMATIC RUNNER${NC}"
echo -e "${CYAN}=====================================================================${NC}"

# STEP 1: Run paper_revision.py to create mega-result entry
echo -e "\n${BLUE}[STEP 1]${NC} Running paper_revision.py to create mega-result entry..."

# Use expect-like behavior with a here-document to provide all inputs needed
# This will:
# 1. Select mode 3 (final mode)
# 2. Accept the recommended model (Y)
# 3. Choose Google as provider (3)
# 4. Accept the recommended model for Google (Y)

output=$(cat <<EOF | python paper_revision.py
3
Y
3
Y
EOF
)

echo "$output"

# Extract run ID using grep and awk
run_id=$(echo "$output" | grep -o "run ID: [0-9]*" | awk '{print $3}')

if [ -z "$run_id" ]; then
    echo -e "\n${RED}[ERROR]${NC} Could not extract run ID from output"
    exit 1
fi

echo -e "\n${GREEN}[SUCCESS]${NC} Created mega-result with run ID: $run_id"

# STEP 2: Generate mega-result directory
echo -e "\n${BLUE}[STEP 2]${NC} Generating mega-result directory..."

# Allow a short delay to ensure database writes are complete
sleep 1

# Run the generate_mega_dir.py script
python generate_mega_dir.py $run_id

# STEP 3: View the mega-result
echo -e "\n${BLUE}[STEP 3]${NC} Viewing mega-result..."
python paper_revision.py --view-mega $run_id

echo -e "\n${GREEN}[COMPLETE]${NC} Mega-result creation and directory generation complete!"
echo -e "You can find all files in: ./tobe/MEGA/$run_id"