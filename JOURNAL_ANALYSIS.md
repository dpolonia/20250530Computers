# Journal Analysis System

This document describes the Journal Analysis System integrated with the Paper Revision Tool. The system uses the Scopus API to retrieve detailed information about academic journals, analyze their content, and provide insights to enhance the paper revision process.

## Overview

The Journal Analysis System enhances the paper revision process by:

1. Retrieving comprehensive information about target journals
2. Analyzing highly-cited papers in the journal
3. Finding papers similar to the manuscript being revised
4. Identifying journal-specific preferences and requirements
5. Providing guidance for manuscript revisions based on journal patterns
6. Enhancing the disclosure section in editor letters with journal information

## Components

### 1. Scopus API Client (`scopus_client.py`)

The Scopus API client provides direct access to Elsevier's Scopus database with features including:

- Journal information retrieval (metrics, scope, aims)
- Paper search by title, DOI, or author
- Citation analysis and reports
- Journal impact analysis
- Similar paper recommendations
- Journal reviewer preference extraction

### 2. Journal Analyzer (`journal_analyzer.py`)

The Journal Analyzer orchestrates the journal analysis process by:

- Retrieving journal information from Scopus
- Storing journal data in the local database
- Finding similar papers in the target journal
- Generating journal-specific guidance for revisions
- Analyzing journal reviewer preferences

### 3. Database Integration (`workflow_db.py`)

The database integration provides persistent storage of journal information:

- Journal metadata (title, publisher, ISSN, etc.)
- Journal metrics (impact factor, SJR, SNIP, etc.)
- Top-cited papers in the journal
- Journal reviewer preferences
- Similar papers for each manuscript revision
- Submission guidelines

## Usage

### Analyzing a Journal

```python
from src.utils.journal_analyzer import get_journal_analyzer

# Create analyzer
analyzer = get_journal_analyzer()

# Analyze by ISSN
journal = analyzer.analyze_journal_by_issn("0747-5632")  # Computers in Human Behavior

# Analyze by title
journal = analyzer.analyze_journal_by_title("Computers in Human Behavior")

# Get journal summary from database
journal_summary = analyzer.db.get_journal_summary(journal_id)
```

### Finding Similar Papers

```python
# Find papers in the journal similar to your manuscript
similar_papers = analyzer.find_similar_papers(
    run_id="20250531",
    journal_id=journal_id,
    title="Machine Learning Applications in Human-Computer Interaction",
    abstract="This paper explores the application of machine learning..."
)

# Get journal guidance for your manuscript
guidance = analyzer.get_journal_guidance(
    journal_id=journal_id,
    manuscript_title="Machine Learning Applications in Human-Computer Interaction",
    manuscript_abstract="This paper explores the application of machine learning..."
)
```

### Integration with Review Process

The Journal Analysis System integrates with the paper revision process by:

1. Analyzing the target journal at the beginning of the workflow
2. Storing journal information in the database
3. Using journal patterns to guide reviewer personas
4. Including journal-specific metrics and requirements in review criteria
5. Enhancing the final editor letter with journal-specific information
6. Adding journal information to the process disclosure

## Database Schema

### Journal Tables

- `journals`: Basic journal information (title, ISSN, publisher, etc.)
- `journal_metrics`: Journal impact metrics by year
- `journal_top_papers`: Highly-cited papers in the journal
- `journal_preferences`: Extracted reviewer preferences
- `journal_guidelines`: Specific submission guidelines
- `similar_journal_papers`: Papers similar to the manuscript being revised

## Configuration

The system uses the following Scopus API credentials:

- API Key: 794f87fe4933b144dd95702b217fcb50
- Label: 20250531
- CORS Domain: https://dev.elsevier.com

You can override the API key by setting the `SCOPUS_API_KEY` environment variable.

## Testing

A test script is provided to verify the Journal Analysis System functionality:

```bash
python test_journal_analyzer.py
```

## Caching

The system implements caching at multiple levels:

1. Scopus API responses are cached to minimize API calls
2. Journal information is stored persistently in the database
3. Analysis results are cached for each run

## Future Enhancements

Planned future enhancements include:

1. Advanced NLP for more sophisticated journal pattern analysis
2. Citation network analysis to identify key research clusters
3. Trend analysis for identifying emerging topics in the journal
4. Reviewer recommendation based on publication patterns
5. Automated formatting according to journal-specific guidelines
6. Journal comparison for optimal submission targeting