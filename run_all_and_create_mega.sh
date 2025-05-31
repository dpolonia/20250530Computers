#!/bin/bash

# Script to run paper_revision.py in final mode with all three models
# and create a mega-result

echo -e "\033[1;36m========================================================\033[0m"
echo -e "\033[1;36mPAPER REVISION TOOL - ALL MODELS AND MEGA-RESULT\033[0m"
echo -e "\033[1;36m========================================================\033[0m"

# Make scripts executable
chmod +x run_final_mode.sh
chmod +x run_final_mode_anthropic.sh
chmod +x run_final_mode_openai.sh

# Run each model - starting with Google which seems most reliable
echo -e "\033[1;34m[STEP 1/4]\033[0m Running Google model..."
./run_final_mode.sh

# Run Anthropic model
echo -e "\033[1;34m[STEP 2/4]\033[0m Running Anthropic model..."
./run_final_mode_anthropic.sh

# Run OpenAI model
echo -e "\033[1;34m[STEP 3/4]\033[0m Running OpenAI model..."
./run_final_mode_openai.sh

# Create mega-result
echo -e "\033[1;34m[STEP 4/4]\033[0m Creating mega-result..."
python create_mega.py

echo -e "\033[1;32m[COMPLETE]\033[0m All models run and mega-result created!"