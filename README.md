# Paper Revision Tool

This project automates parts of the paper revision workflow for the *Computers* journal.

## Usage

### Basic Usage

```bash
python paper_revision.py --mode [training|finetuning|final]
```

### Operation Modes

- **Training**: Uses cheapest models for workflow activation (minimal cost)
- **Finetuning**: Uses mid-range models for workflow optimization (balanced)
- **Final**: Uses best available models for final revisions (highest quality)

### Advanced Usage

```bash
python paper_revision.py --provider [anthropic|openai|google] --model [model_name]
```

### Output Files

All output files are stored in the `tobe` directory using this structure:
- `tobe/[ModelCode]_[ModelName]/[Timestamp]/`

Where:
- `ModelCode`: A standardized code (e.g., A01, B01, C01) that identifies the provider and model
- `ModelName`: The name of the model used
- `Timestamp`: The timestamp when the revision was started

## Integration with Claude

The tool includes intelligent integration with Claude that only activates when errors occur:

```bash
./run_for_claude.sh --mode [training|finetuning|final]
```

For successful runs:
1. The script will analyze logs and cost reports
2. Provide a summary of the run with model information
3. Extract cost analysis and token usage statistics
4. Show optimization recommendations
5. Report any warnings or issues detected in the logs

For error cases only:
1. The script will automatically capture the error output
2. Format it for easy sharing with Claude
3. Save it to a file named `claude_error_TIMESTAMP.txt`
4. Provide the command to send the file to Claude

This ensures Claude is only used when needed, while still providing useful analytics for successful runs.

## Error Handling

If the paper revision process fails, files will be automatically copied to the trash directory:
- `tobe/_trash/[ModelCode]_[ModelName]/[Timestamp]/`

This directory will contain:
- All generated files from the failed run
- A detailed failure report
- Error logs with diagnostic information
