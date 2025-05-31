# Paper Revision Tool Documentation

This directory contains comprehensive documentation for the Paper Revision Tool project.

## Documentation Structure

- **[style_guide.md](style_guide.md)**: Coding and documentation standards
- **[architecture.md](architecture.md)**: Architectural overview and component relationships
- **[architecture_diagram.txt](architecture_diagram.txt)**: ASCII architecture diagram
- **[module_documentation_template.py](module_documentation_template.py)**: Template for module documentation
- **[adr/](adr/)**: Architectural Decision Records

## Architectural Decision Records (ADRs)

ADRs document significant architectural decisions made in the project:

1. [ADR-001: Layered Architecture](adr/001-layered-architecture.md)
2. [ADR-002: Interface-Based Design](adr/002-interface-based-design.md)

To create a new ADR, copy the [ADR template](adr/adr_template.md) and follow the format.

## Documentation Standards

All code in the project should follow the documentation standards defined in the [Style Guide](style_guide.md). The key points are:

- Use Google-style docstrings
- Document all modules, classes, methods, and functions
- Include types for parameters and return values
- Document exceptions that may be raised

## Scripts

The project includes scripts to help maintain documentation quality:

- **[update_docstrings.py](../scripts/update_docstrings.py)**: Analyzes and updates module docstrings

To generate a docstring coverage report:

```bash
python scripts/update_docstrings.py --report
```

To update missing module docstrings:

```bash
python scripts/update_docstrings.py --update
```

## Keeping Documentation Up to Date

Documentation should be treated as part of the codebase and kept up to date:

1. Update module docstrings when adding or modifying modules
2. Create ADRs for significant architectural decisions
3. Update architectural documentation when the architecture changes
4. Run the docstring update script regularly to maintain documentation quality