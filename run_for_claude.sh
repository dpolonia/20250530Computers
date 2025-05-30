#\!/bin/bash
# Run paper_revision.py and capture output for Claude

# Get current timestamp
TIMESTAMP=$(date +%Y%m%d%H%M%S)
OUTPUT_FILE="claude_output_$TIMESTAMP.txt"

# Check if Python virtual environment is active, activate if not
if [[ -z "${VIRTUAL_ENV}" ]]; then
    if [[ -d "./venv" ]]; then
        echo "Activating virtual environment..."
        source ./venv/bin/activate
    fi
fi

# Run the paper_revision.py script with all arguments passed to this script
echo "Running paper_revision.py with arguments: $@"
echo "Output will be saved to: $OUTPUT_FILE"

# Use the claude_output.py script to capture output
python claude_output.py --file "$OUTPUT_FILE" python paper_revision.py "$@"

# Show file location
echo "Output ready for Claude at: $OUTPUT_FILE"
echo "You can send this file to Claude with: cat $OUTPUT_FILE | claude"

# Deactivate virtual environment if we activated it
if [[ -n "${ACTIVATED_VENV}" ]]; then
    deactivate
fi
