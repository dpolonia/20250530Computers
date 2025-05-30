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

To capture output and send it directly to Claude Code, use the following command:

```bash
./run_for_claude.sh --mode [training|finetuning|final]
```

This will:
1. Run the paper revision tool with the specified arguments
2. Capture all output to a file
3. Format the output for easy sharing with Claude
4. Provide the file path that can be used with the Claude CLI

## Error Handling

If the paper revision process fails, files will be automatically copied to the trash directory:
- `tobe/_trash/[ModelCode]_[ModelName]/[Timestamp]/`

This directory will contain:
- All generated files from the failed run
- A detailed failure report
- Error logs with diagnostic information
