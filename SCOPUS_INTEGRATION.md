# Scopus API Integration

This document describes the integration of the Elsevier Scopus API with the Paper Revision Tool. The integration enhances the reference validation and management capabilities, allowing for more accurate bibliographic data and citation analysis.

## Configuration

The Scopus API integration uses the following API key:

- API Key: 794f87fe4933b144dd95702b217fcb50
- Label: 20250531
- CORS Domain: https://dev.elsevier.com

You can override this key by setting the `SCOPUS_API_KEY` environment variable.

## Features

The Scopus API integration provides the following features:

### 1. Enhanced Reference Validation

- Validates DOIs using Scopus in addition to DOI.org
- Attempts to fix invalid references by searching for matching papers in Scopus
- Enhances existing references with additional metadata from Scopus

### 2. Paper Search and Discovery

- Search for papers by title, author, or DOI
- Find similar papers based on title and abstract
- Retrieve citation information and impact metrics

### 3. Citation Analysis

- Generate citation reports for papers
- Analyze citation patterns and impact
- Identify top citing countries and journals

### 4. Journal Metrics

- Retrieve journal impact metrics
- Analyze journal quartiles and rankings
- Identify subject areas and categories

## Usage

### Basic Usage

```python
from src.utils.reference_validator import ReferenceValidator

# Initialize with a BibTeX file
validator = ReferenceValidator("references.bib", use_scopus=True)

# Validate references (will attempt to fix invalid references)
valid_refs, invalid_refs = validator.validate_references()

# Find similar papers to a reference
similar_papers = validator.find_similar_papers("ref_id", count=5)

# Add a reference by DOI
ref_id = validator.add_reference_from_doi("10.1016/j.artint.2022.103756")

# Search and add references
added_refs = validator.search_and_add_references("Natural Language Processing", count=3)

# Get citation report for a reference
citation_report = validator.get_citation_report("ref_id")

# Save the updated references
validator.save_references()
```

### Direct Scopus API Access

```python
from src.utils.scopus_client import get_scopus_client

# Create a Scopus client
client = get_scopus_client()

# Search for papers by title
papers = client.search_by_title("Machine Learning")

# Get paper by DOI
paper = client.search_by_doi("10.1016/j.artint.2022.103756")

# Get citations for a paper
citations = client.get_citations("10.1016/j.artint.2022.103756")

# Get references cited by a paper
references = client.get_references("10.1016/j.artint.2022.103756")

# Generate a citation report
report = client.generate_citation_report("10.1016/j.artint.2022.103756")
```

## Testing

A test script is provided to verify the Scopus integration:

```bash
python test_scopus.py
```

This script tests both the Scopus client and its integration with the reference validator.

## Cache System

The Scopus client includes a caching system to reduce API calls:

- Responses are cached to disk in `.cache/scopus/`
- Default cache TTL is 24 hours
- Cache can be disabled by setting `use_cache=False`

## Dependencies

The Scopus integration requires the following dependencies:

- requests
- bibtexparser
- urllib3
- pybliometrics (optional)

These dependencies are included in the `requirements.txt` file.

## Error Handling

The Scopus integration includes comprehensive error handling:

- Falls back to DOI.org validation if Scopus validation fails
- Logs errors with appropriate severity levels
- Returns empty results instead of raising exceptions
- Maintains a graceful degradation pattern

## Future Enhancements

Planned future enhancements include:

1. Integration with pybliometrics for additional Scopus features
2. Support for Scopus Abstract Retrieval API
3. Enhanced author disambiguation and affiliation information
4. Citation network visualization
5. Journal recommendation based on paper content