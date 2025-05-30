#!/bin/bash
# Run paper_revision.py and only capture output for Claude in case of errors
# For successful runs, analyze logs and cost reports for improvement opportunities

# Get current timestamp
TIMESTAMP=$(date +%Y%m%d%H%M%S)
ERROR_OUTPUT_FILE="claude_error_$TIMESTAMP.txt"
TEMP_OUTPUT_FILE="temp_output_$TIMESTAMP.txt"

# Check if Python virtual environment is active, activate if not
if [[ -z "${VIRTUAL_ENV}" ]]; then
    if [[ -d "./venv" ]]; then
        echo "Activating virtual environment..."
        source ./venv/bin/activate
        ACTIVATED_VENV=1
    fi
fi

# Run the paper_revision.py script with all arguments passed to this script
echo "Running paper_revision.py with arguments: $@"

# First run the command and capture output to a temporary file
python paper_revision.py "$@" 2>&1 | tee "$TEMP_OUTPUT_FILE"

# Get the exit code
EXIT_CODE=${PIPESTATUS[0]}

# Check if the run was successful
if [ $EXIT_CODE -ne 0 ] || grep -q "Paper revision failed\|Error in paper revision process\|ERROR" "$TEMP_OUTPUT_FILE"; then
    # Error occurred - prepare output for Claude
    echo "Error detected. Preparing output for Claude..."
    
    # Use the claude_output.py script to format the output
    cat "$TEMP_OUTPUT_FILE" > "$ERROR_OUTPUT_FILE"
    
    # Add a header
    ERROR_HEADER="ERROR REPORT - $(date '+%Y-%m-%d %H:%M:%S')\n"
    ERROR_HEADER+="Command: python paper_revision.py $@\n"
    ERROR_HEADER+="Exit code: $EXIT_CODE\n"
    ERROR_HEADER+="----------------------------------------\n\n"
    
    # Add header to the file
    echo -e "$ERROR_HEADER$(cat "$ERROR_OUTPUT_FILE")" > "$ERROR_OUTPUT_FILE"
    
    # Show error output location
    echo -e "\n----------------------------------------"
    echo "Error output ready for Claude at: $ERROR_OUTPUT_FILE"
    echo "You can send this file to Claude with: cat $ERROR_OUTPUT_FILE | claude"
    echo -e "----------------------------------------\n"
else
    # Successful run - analyze logs and cost reports
    echo "Run completed successfully. Analyzing logs and cost reports..."
    
    # Extract paths to log and cost files
    LOG_FILE=$(grep -o "Log file: [^ ]*" "$TEMP_OUTPUT_FILE" | head -1 | cut -d' ' -f3)
    COST_REPORT=$(grep -o "Cost report: [^ ]*" "$TEMP_OUTPUT_FILE" | head -1 | cut -d' ' -f3)
    REVISION_REPORT=$(grep -o "Revision report: [^ ]*" "$TEMP_OUTPUT_FILE" | head -1 | cut -d' ' -f3)
    
    # Print summary of run
    echo -e "\n----------------------------------------"
    echo "PAPER REVISION SUMMARY"
    echo "----------------------------------------"
    
    # Extract and display model information
    MODEL_INFO=$(grep -o "Selected model: [^)]*)" "$TEMP_OUTPUT_FILE" | head -1)
    echo "Model: $MODEL_INFO"
    
    # Extract and display operation mode
    OP_MODE=$(grep -o "OPERATION MODE: [A-Z]*" "$TEMP_OUTPUT_FILE" | head -1)
    echo "$OP_MODE"
    
    # Show output files
    echo -e "\nOutput files:"
    grep "^- " "$TEMP_OUTPUT_FILE" | grep -v "error\|trash"
    
    # Analyze cost report if available
    if [ -n "$COST_REPORT" ] && [ -f "$COST_REPORT" ]; then
        echo -e "\nCost Analysis:"
        # Extract total cost and token usage
        TOTAL_COST=$(grep "Total cost:" "$COST_REPORT" | head -1)
        TOKENS_USED=$(grep "Tokens used:" "$COST_REPORT" | head -1)
        echo "$TOKENS_USED"
        echo "$TOTAL_COST"
        
        # Check for optimization recommendations
        echo -e "\nOptimization Recommendations:"
        RECOMMENDATIONS=$(sed -n '/OPTIMIZATION RECOMMENDATIONS/,/^$/p' "$COST_REPORT" | grep "^-")
        if [ -n "$RECOMMENDATIONS" ]; then
            echo "$RECOMMENDATIONS"
        else
            echo "No specific optimization recommendations found."
        fi
    fi
    
    # Check logs for warnings or areas of improvement
    if [ -n "$LOG_FILE" ] && [ -f "$LOG_FILE" ]; then
        WARNING_COUNT=$(grep -c "WARNING" "$LOG_FILE")
        if [ $WARNING_COUNT -gt 0 ]; then
            echo -e "\nWarnings detected ($WARNING_COUNT):"
            grep "WARNING" "$LOG_FILE" | head -3
            if [ $WARNING_COUNT -gt 3 ]; then
                echo "... and $(($WARNING_COUNT - 3)) more warnings. See $LOG_FILE for details."
            fi
        fi
    fi
    
    echo -e "\nAll outputs are available in the directories shown above."
    echo "----------------------------------------"
fi

# Clean up temporary file
rm -f "$TEMP_OUTPUT_FILE"

# Deactivate virtual environment if we activated it
if [[ -n "${ACTIVATED_VENV}" ]]; then
    deactivate
fi
