# Academic API Integration Guide

This document describes the integration of academic database APIs with the Paper Revision Tool. The integration provides comprehensive citation analysis, journal information retrieval, and reference validation capabilities.

## Overview

The Paper Revision Tool currently integrates with:

1. **Scopus API (ACTIVE)**: Provides comprehensive access to Elsevier's database of academic literature, including citation data, journal metrics, and article metadata.

2. **Web of Science API (INACTIVE)**: Integration is implemented but currently inactive. When activated, it will provide access to Clarivate's Web of Science database, offering complementary citation data and impact metrics.

The multi-database integration allows for:
- Cross-validation of citation counts
- More comprehensive literature searches
- Better journal impact analysis
- Enhanced reference validation
- Improved citation network analysis

## API Configuration

### Scopus API (ACTIVE)

- **API Key**: 794f87fe4933b144dd95702b217fcb50
- **Label**: 20250531
- **CORS Domains**: https://dev.elsevier.com

### Web of Science API (INACTIVE)

- **Application ID**: lazyresearcher
- **Application Name**: Lazy Researcher - Using vibe coding and API´s to optimize the paper creation and review automatized.
- **Client Type**: Public: Single Page Application (browser based app)
- **Status**: Integration implemented but currently disabled. No client secret has been configured.

## Components

### 1. Scopus Client (`scopus_client.py`)

The Scopus client provides access to the Elsevier Scopus API with features including:

- Journal information and metrics retrieval
- Paper search by title, DOI, or author
- Citation analysis and reports
- Reference validation
- Similar paper recommendations

### 2. Web of Science Client (`wos_client.py`)

The Web of Science client provides access to the Clarivate Web of Science API with features including:

- Paper search by title, DOI, or author
- Citation retrieval and analysis
- Journal impact metrics
- Reference and citing paper retrieval

### 3. Citation Analyzer (`citation_analyzer.py`)

The Citation Analyzer integrates data from both APIs to provide comprehensive citation analysis:

- Cross-database citation counts
- Citation overlap analysis
- Citation velocity metrics
- Multi-source paper recommendations
- Comprehensive journal impact metrics

## Usage

### Basic Usage

```python
from src.utils.citation_analyzer import get_citation_analyzer

# Create analyzer with Scopus API credentials only
analyzer = get_citation_analyzer(
    scopus_api_key="YOUR_SCOPUS_API_KEY"
)

# When WoS integration is activated, you can use:
# analyzer = get_citation_analyzer(
#     scopus_api_key="YOUR_SCOPUS_API_KEY",
#     wos_client_id="YOUR_WOS_CLIENT_ID",
#     wos_client_secret="YOUR_WOS_CLIENT_SECRET"
# )

# Analyze paper citations across databases
citation_data = analyzer.analyze_paper_citations("10.1016/j.example.2022.123456")

# Find papers across multiple databases
similar_papers = analyzer.find_cross_database_papers(
    title="Machine Learning for Academic Research",
    abstract="This paper explores the application of machine learning..."
)

# Analyze citation overlap between papers
overlap_data = analyzer.get_citation_overlap([
    "10.1016/j.example1.2022.123456",
    "10.1016/j.example2.2022.789012"
])
```

### Integration with Paper Revision Process

The API integration enhances the paper revision process in several ways:

1. **Reference Validation**: Validates and enhances references with data from multiple sources
2. **Journal Analysis**: Provides comprehensive journal metrics and impact factors
3. **Similar Paper Identification**: Finds relevant papers across multiple databases
4. **Citation Analysis**: Analyzes the paper's citation network and impact
5. **Reviewer Persona Enhancement**: Uses journal preferences derived from multiple sources

## Performance Optimizations

The implementation includes several performance optimizations:

1. **Robust Caching**:
   - All API responses are cached to disk
   - Cache TTL set to 7 days by default
   - Automatic cache pruning to maintain manageable size

2. **Retry Logic**:
   - Exponential backoff for failed requests
   - Automatic retries for transient errors
   - Token refresh for authentication failures

3. **Rate Limiting**:
   - Enforced minimum spacing between requests
   - Automatic handling of API rate limits
   - Graceful degradation when limits are reached

4. **Error Handling**:
   - Comprehensive error detection and reporting
   - Fallback to alternative sources when one fails
   - Detailed logging for troubleshooting

## Security Considerations

1. **API Key Management**:
   - API keys are stored securely using environment variables
   - Keys are never logged or exposed in error messages
   - Limited display of key fragments in logs

2. **Access Control**:
   - Minimal set of API permissions requested
   - No write operations to external APIs
   - Read-only access to academic databases

3. **Data Handling**:
   - Sensitive data is not persisted unnecessarily
   - Local database is properly secured
   - No unnecessary data collection

## Monitoring and Usage Statistics

The implementation includes comprehensive monitoring capabilities:

1. **API Usage Tracking**:
   - Request counts by endpoint
   - Cache hit rates
   - Error rates and types

2. **Performance Metrics**:
   - Response times
   - Cache size and efficiency
   - Resource utilization

3. **Usage Reporting**:
   - Detailed usage statistics available through `get_usage_statistics()`
   - Export of usage data for analysis
   - Rate limit monitoring

## Future Enhancements

Planned future enhancements include:

1. **Additional API Integrations**:
   - Google Scholar integration
   - Microsoft Academic Graph integration
   - Semantic Scholar API integration

2. **Advanced Analysis**:
   - Citation network visualization
   - Author collaboration network analysis
   - Topic modeling across databases
   - Trend analysis in research fields

3. **Performance Improvements**:
   - Parallel API queries
   - Distributed caching
   - Background data prefetching
   - Incremental updates

4. **Enhanced User Experience**:
   - Interactive citation graphs
   - Personalized recommendations
   - Custom citation alerts
   - Advanced filtering options

## Testing

A test script is provided to verify the multi-database integration:

```bash
python test_citation_analyzer.py
```

This script demonstrates the key functionality and exports results to a JSON file for review.