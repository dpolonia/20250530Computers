"""Scopus API Client for scientific paper metadata retrieval.

This module provides a client for accessing the Scopus API to retrieve
information about scientific papers, citations, and authors. It enhances
the paper revision process by providing accurate bibliographic data and
citation analysis.
"""

import os
import json
import time
import logging
import requests
from typing import Dict, List, Any, Optional, Tuple, Set
from urllib.parse import quote
import datetime
import functools

# Configure logging
logger = logging.getLogger(__name__)

# Cache configuration
SCOPUS_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".cache", "scopus")
CACHE_TTL_DAYS = 7  # Cache time-to-live in days
CACHE_PRUNE_THRESHOLD = 500  # Number of files before pruning
CACHE_PRUNE_TARGET = 400  # Target number of files after pruning

# Create cache directory
os.makedirs(SCOPUS_CACHE_DIR, exist_ok=True)

# Function to prune old cache files
def prune_scopus_cache():
    """Prune old cache files to keep the cache size manageable."""
    try:
        cache_files = [os.path.join(SCOPUS_CACHE_DIR, f) for f in os.listdir(SCOPUS_CACHE_DIR) if f.endswith('.json')]
        
        # Check if pruning is needed
        if len(cache_files) <= CACHE_PRUNE_THRESHOLD:
            return
            
        # Sort by modification time (oldest first)
        cache_files.sort(key=os.path.getmtime)
        
        # Calculate how many files to delete
        files_to_delete = len(cache_files) - CACHE_PRUNE_TARGET
        
        # Delete oldest files
        for f in cache_files[:files_to_delete]:
            try:
                os.remove(f)
                logger.debug(f"Pruned cache file: {os.path.basename(f)}")
            except OSError as e:
                logger.warning(f"Error pruning cache file {f}: {e}")
    except Exception as e:
        logger.warning(f"Error during cache pruning: {e}")

# Initial cache pruning
prune_scopus_cache()

class ScopusClient:
    """Client for interacting with the Scopus API."""
    
    def __init__(self, api_key: str = None, use_cache: bool = True, cache_ttl: int = None):
        """Initialize the Scopus API client.
        
        Args:
            api_key: Scopus API key. If None, will attempt to read from environment.
            use_cache: Whether to cache API responses (default: True)
            cache_ttl: Cache time-to-live in seconds (default: CACHE_TTL_DAYS * 86400)
        """
        self.api_key = api_key or os.environ.get("SCOPUS_API_KEY", "794f87fe4933b144dd95702b217fcb50")
        self.base_url = "https://api.elsevier.com"
        self.use_cache = use_cache
        self.cache_ttl = cache_ttl or (CACHE_TTL_DAYS * 86400)  # Convert days to seconds
        
        # Set up headers with API key and additional parameters for better responses
        self.headers = {
            "X-ELS-APIKey": self.api_key,
            "Accept": "application/json",
            "User-Agent": "PaperRevisionTool/1.0 (Academic Research)",
            "X-ELS-ResourceVersion": "XOCS"  # Request consistent response format
        }
        
        # Add Content-Type for some endpoints that require it
        self.content_headers = self.headers.copy()
        self.content_headers["Content-Type"] = "application/json"
        
        # Track API usage
        self.request_count = 0
        self.cache_hit_count = 0
        self.last_request_time = 0
        
        # Add minimum request spacing (rate limiting)
        self.request_spacing = 0.5  # seconds between requests
        
        # Verify API key
        self._verify_api_key()
    
    def _verify_api_key(self):
        """Verify that the API key is valid by making a simple request."""
        try:
            response = self._make_request("/content/search/scopus", {"query": "all(test)", "count": 1})
            if "service-error" in response:
                logger.error(f"Scopus API key validation failed: {response.get('service-error', {}).get('status', {}).get('statusText', 'Unknown error')}")
            else:
                logger.info("Scopus API key validated successfully")
        except Exception as e:
            logger.error(f"Error validating Scopus API key: {e}")
    
    def _get_cache_key(self, endpoint: str, params: Dict[str, Any]) -> str:
        """Generate a cache key for the request.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            Cache key string
        """
        params_str = json.dumps(params, sort_keys=True)
        return f"{endpoint}_{hash(params_str)}.json"
    
    def _get_cached_response(self, endpoint: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Retrieve a cached response if available and not expired.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            Cached response or None if not available/valid
        """
        if not self.use_cache:
            return None
            
        cache_key = self._get_cache_key(endpoint, params)
        cache_path = os.path.join(SCOPUS_CACHE_DIR, cache_key)
        
        if not os.path.exists(cache_path):
            return None
            
        # Check if cache is expired
        cache_time = os.path.getmtime(cache_path)
        if time.time() - cache_time > self.cache_ttl:
            # Cache expired
            try:
                os.remove(cache_path)  # Clean up expired cache file
                logger.debug(f"Removed expired cache file: {cache_key}")
            except OSError:
                pass  # Ignore errors removing cache files
            return None
            
        try:
            with open(cache_path, 'r') as f:
                cached_data = json.load(f)
                self.cache_hit_count += 1
                logger.debug(f"Cache hit for: {endpoint}")
                
                # Handle both old and new cache format
                if isinstance(cached_data, dict) and "data" in cached_data:
                    return cached_data["data"]
                else:
                    return cached_data
        except json.JSONDecodeError:
            # Remove corrupt cache file
            try:
                os.remove(cache_path)
                logger.warning(f"Removed corrupt cache file: {cache_key}")
            except OSError:
                pass
            return None
        except Exception as e:
            logger.warning(f"Error reading cache file {cache_path}: {e}")
            return None
    
    def _save_to_cache(self, endpoint: str, params: Dict[str, Any], response: Dict[str, Any]):
        """Save an API response to the cache.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            response: API response to cache
        """
        if not self.use_cache:
            return
            
        # Skip caching error responses
        if "service-error" in response:
            return
            
        cache_key = self._get_cache_key(endpoint, params)
        cache_path = os.path.join(SCOPUS_CACHE_DIR, cache_key)
        
        try:
            # Create a copy of the response to avoid modifying the original
            cache_data = {
                "data": response,
                "cached_at": time.time(),
                "endpoint": endpoint,
                "params": str(params)  # Convert to string to ensure JSON serialization
            }
            
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f)
                
            logger.debug(f"Saved response to cache: {cache_key}")
            
            # Check if cache pruning is needed
            cache_files = os.listdir(SCOPUS_CACHE_DIR)
            if len(cache_files) > CACHE_PRUNE_THRESHOLD:
                # Schedule pruning on next request
                prune_scopus_cache()
                
        except Exception as e:
            logger.warning(f"Error saving to cache file {cache_path}: {e}")
    
    def _make_request(self, endpoint: str, params: Dict[str, Any], retry: int = 3, backoff: float = 1.5) -> Dict[str, Any]:
        """Make a request to the Scopus API with caching and retry logic.
        
        Args:
            endpoint: API endpoint (starting with /)
            params: Query parameters
            retry: Number of retries for failed requests
            backoff: Backoff multiplier for retries
            
        Returns:
            API response as dictionary
        """
        # Check cache first
        cached_response = self._get_cached_response(endpoint, params)
        if cached_response:
            logger.debug(f"Using cached response for {endpoint}")
            return cached_response
        
        # Make the request with retry logic
        url = f"{self.base_url}{endpoint}"
        current_retry = 0
        wait_time = 1.0  # Initial wait time in seconds
        
        while current_retry <= retry:
            try:
                # Add View parameter for better response formatting
                if "view" not in params:
                    params["view"] = "COMPLETE"
                    
                # Add standard parameters for reliable results
                if "count" not in params and "query" in params:
                    params["count"] = 25  # Default count for search queries
                
                # Make the request
                response = requests.get(url, headers=self.headers, params=params, timeout=30)
                
                # Handle rate limiting (429 status code)
                if response.status_code == 429:
                    if current_retry < retry:
                        logger.warning(f"Rate limited by Scopus API, retrying in {wait_time:.1f} seconds")
                        time.sleep(wait_time)
                        wait_time *= backoff
                        current_retry += 1
                        continue
                    else:
                        logger.error("Exceeded maximum retries for rate limit")
                        return {"service-error": {"status": {"statusCode": 429, "statusText": "Rate limit exceeded after multiple retries"}}}
                
                # Handle other response codes
                if response.status_code == 200:
                    try:
                        data = response.json()
                        # Save to cache
                        self._save_to_cache(endpoint, params, data)
                        return data
                    except json.JSONDecodeError as e:
                        logger.error(f"Error decoding JSON response: {e}")
                        return {"service-error": {"status": {"statusCode": -2, "statusText": f"JSON decode error: {str(e)}"}}}
                else:
                    error_msg = f"Scopus API error: {response.status_code} - {response.text[:500]}"
                    
                    # Retry for server errors (5xx) and some client errors
                    if (response.status_code >= 500 or response.status_code in [408, 429]) and current_retry < retry:
                        logger.warning(f"{error_msg}, retrying in {wait_time:.1f} seconds")
                        time.sleep(wait_time)
                        wait_time *= backoff
                        current_retry += 1
                        continue
                    
                    logger.error(error_msg)
                    return {"service-error": {"status": {"statusCode": response.status_code, "statusText": response.text[:500]}}}
            
            except requests.exceptions.Timeout:
                if current_retry < retry:
                    logger.warning(f"Request timeout, retrying in {wait_time:.1f} seconds")
                    time.sleep(wait_time)
                    wait_time *= backoff
                    current_retry += 1
                    continue
                else:
                    logger.error("Exceeded maximum retries for timeout")
                    return {"service-error": {"status": {"statusCode": -3, "statusText": "Request timeout after multiple retries"}}}
            
            except Exception as e:
                logger.error(f"Error making request to Scopus API: {e}")
                return {"service-error": {"status": {"statusCode": -1, "statusText": str(e)}}}
                
        # This should never be reached with the return statements above
        return {"service-error": {"status": {"statusCode": -4, "statusText": "Unknown error in request handling"}}}
    
    def search_by_title(self, title: str, count: int = 5) -> List[Dict[str, Any]]:
        """Search for papers by title.
        
        Args:
            title: Paper title to search for
            count: Maximum number of results to return
            
        Returns:
            List of paper metadata dictionaries
        """
        params = {
            "query": f"TITLE(\"{title}\")",
            "count": count,
            "field": "title,authors,publicationName,coverDate,doi,citedby-count,description"
        }
        
        response = self._make_request("/content/search/scopus", params)
        
        if "search-results" in response and "entry" in response["search-results"]:
            return response["search-results"]["entry"]
        else:
            logger.warning(f"No results found for title: {title}")
            return []
    
    def search_by_doi(self, doi: str) -> Optional[Dict[str, Any]]:
        """Retrieve paper metadata by DOI.
        
        Args:
            doi: DOI of the paper
            
        Returns:
            Paper metadata dictionary or None if not found
        """
        # Clean the DOI
        doi = doi.strip()
        if doi.lower().startswith('doi:'):
            doi = doi[4:].strip()
        if doi.lower().startswith('https://doi.org/'):
            doi = doi[16:].strip()
        
        params = {
            "query": f"DOI(\"{doi}\")",
            "field": "title,authors,publicationName,coverDate,doi,citedby-count,description,abstract"
        }
        
        response = self._make_request("/content/search/scopus", params)
        
        if "search-results" in response and "entry" in response["search-results"] and len(response["search-results"]["entry"]) > 0:
            return response["search-results"]["entry"][0]
        else:
            logger.warning(f"No results found for DOI: {doi}")
            return None
    
    def get_citations(self, doi: str, count: int = 10) -> List[Dict[str, Any]]:
        """Retrieve papers that cite the given DOI.
        
        Args:
            doi: DOI of the paper
            count: Maximum number of citations to return
            
        Returns:
            List of citing paper metadata dictionaries
        """
        # Clean the DOI
        doi = doi.strip()
        if doi.lower().startswith('doi:'):
            doi = doi[4:].strip()
        if doi.lower().startswith('https://doi.org/'):
            doi = doi[16:].strip()
        
        params = {
            "query": f"REFSRCTITLE(\"{doi}\")",
            "count": count,
            "field": "title,authors,publicationName,coverDate,doi,citedby-count",
            "sort": "citedby-count desc"
        }
        
        response = self._make_request("/content/search/scopus", params)
        
        if "search-results" in response and "entry" in response["search-results"]:
            return response["search-results"]["entry"]
        else:
            logger.warning(f"No citations found for DOI: {doi}")
            return []
    
    def get_references(self, doi: str) -> List[Dict[str, Any]]:
        """Retrieve references cited by the given paper.
        
        Args:
            doi: DOI of the paper
            
        Returns:
            List of reference metadata dictionaries
        """
        # Clean the DOI
        doi = doi.strip()
        if doi.lower().startswith('doi:'):
            doi = doi[4:].strip()
        if doi.lower().startswith('https://doi.org/'):
            doi = doi[16:].strip()
        
        # Use the References API endpoint
        params = {
            "doi": doi,
            "view": "REF"
        }
        
        response = self._make_request(f"/content/abstract/doi/{quote(doi)}", params)
        
        if "abstracts-retrieval-response" in response and "references" in response["abstracts-retrieval-response"]:
            return response["abstracts-retrieval-response"]["references"]["reference"]
        else:
            logger.warning(f"No references found for DOI: {doi}")
            return []
    
    def search_by_author(self, author_name: str, count: int = 10) -> List[Dict[str, Any]]:
        """Search for papers by author name.
        
        Args:
            author_name: Author name to search for
            count: Maximum number of results to return
            
        Returns:
            List of paper metadata dictionaries
        """
        params = {
            "query": f"AUTHOR-NAME(\"{author_name}\")",
            "count": count,
            "field": "title,authors,publicationName,coverDate,doi,citedby-count",
            "sort": "citedby-count desc"
        }
        
        response = self._make_request("/content/search/scopus", params)
        
        if "search-results" in response and "entry" in response["search-results"]:
            return response["search-results"]["entry"]
        else:
            logger.warning(f"No results found for author: {author_name}")
            return []
    
    def get_author_details(self, author_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve detailed information about an author by Scopus Author ID.
        
        Args:
            author_id: Scopus Author ID
            
        Returns:
            Author metadata dictionary or None if not found
        """
        params = {
            "field": "given-name,surname,affiliation-current,document-count,h-index,coauthor-count"
        }
        
        response = self._make_request(f"/author/author_id/{author_id}", params)
        
        if "author-retrieval-response" in response:
            return response["author-retrieval-response"][0]
        else:
            logger.warning(f"No author details found for ID: {author_id}")
            return None
    
    def get_journal_metrics(self, issn: str) -> Optional[Dict[str, Any]]:
        """Retrieve metrics for a journal by ISSN.
        
        Args:
            issn: Journal ISSN
            
        Returns:
            Journal metrics dictionary or None if not found
        """
        params = {
            "issn": issn
        }
        
        response = self._make_request(f"/serial/title/issn/{issn}", params)
        
        if "serial-metadata-response" in response:
            return response["serial-metadata-response"]["entry"][0]
        else:
            logger.warning(f"No journal metrics found for ISSN: {issn}")
            return None
            
    def get_journal_info(self, issn: str) -> Dict[str, Any]:
        """Get comprehensive information about a journal.
        
        Args:
            issn: Journal ISSN
            
        Returns:
            Dictionary with comprehensive journal information
        """
        # Get basic journal information and metrics
        journal_metrics = self.get_journal_metrics(issn)
        
        if not journal_metrics:
            return {
                "issn": issn,
                "title": "Unknown Journal",
                "publisher": "",
                "subject_areas": [],
                "metrics": {},
                "aims_scope": "",
                "top_papers": []
            }
        
        # Extract basic information
        title = journal_metrics.get("dc:title", "")
        publisher = journal_metrics.get("dc:publisher", "")
        
        # Extract subject areas
        subject_areas = []
        if "subject-area" in journal_metrics:
            subject_areas = [
                area.get("$", "")
                for area in journal_metrics["subject-area"]
            ]
        
        # Extract metrics
        metrics = {
            "sjr": journal_metrics.get("SJR", ""),
            "snip": journal_metrics.get("SNIP", ""),
            "impact_factor": journal_metrics.get("citeScore", {}).get("$", ""),
            "h_index": journal_metrics.get("h-index", ""),
            "total_docs": journal_metrics.get("article-count", ""),
            "quartile": journal_metrics.get("citeScore", {}).get("quartile", "")
        }
        
        # Attempt to get journal description and aims/scope
        aims_scope = ""
        try:
            # Use the general search to find more information about the journal
            search_params = {
                "query": f"ISSN({issn}) OR EXACTSRCTITLE(\"{title}\")",
                "field": "description,website,publisher,openaccess"
            }
            
            search_response = self._make_request("/content/search/scopus", search_params)
            
            if "search-results" in search_response and "entry" in search_response["search-results"]:
                for entry in search_response["search-results"]["entry"]:
                    if "dc:description" in entry:
                        aims_scope = entry["dc:description"]
                        break
        except Exception as e:
            logger.warning(f"Error getting journal description: {e}")
        
        # Build the complete journal information
        journal_info = {
            "issn": issn,
            "e_issn": journal_metrics.get("prism:eIssn", ""),
            "title": title,
            "publisher": publisher,
            "subject_areas": ", ".join(subject_areas),
            "description": aims_scope,
            "aims_scope": aims_scope,
            "website": journal_metrics.get("website", ""),
            "open_access": "Open Access" in journal_metrics.get("openaccess", ""),
            "publication_frequency": journal_metrics.get("publicationFrequency", ""),
            "first_indexed_year": journal_metrics.get("coverageStartYear", ""),
            "metrics": metrics
        }
        
        # Get top cited papers in the journal
        top_papers = self.get_journal_top_papers(issn, title)
        journal_info["top_papers"] = top_papers
        
        return journal_info
    
    def get_journal_top_papers(self, issn: str = None, journal_title: str = None, count: int = 10) -> List[Dict[str, Any]]:
        """Get the top cited papers in a journal.
        
        Args:
            issn: Journal ISSN (optional if title is provided)
            journal_title: Journal title (optional if ISSN is provided)
            count: Maximum number of papers to return
            
        Returns:
            List of paper dictionaries
        """
        if not issn and not journal_title:
            logger.error("Either ISSN or journal title must be provided")
            return []
        
        # Build the query
        if issn:
            query = f"ISSN({issn})"
        else:
            query = f"EXACTSRCTITLE(\"{journal_title}\")"
            
        params = {
            "query": query,
            "count": count,
            "field": "title,authors,publicationName,coverDate,doi,citedby-count,description,volume,issue,pageRange",
            "sort": "citedby-count desc"
        }
        
        response = self._make_request("/content/search/scopus", params)
        
        if "search-results" in response and "entry" in response["search-results"]:
            return response["search-results"]["entry"]
        else:
            logger.warning(f"No top papers found for journal: {issn or journal_title}")
            return []
    
    def find_similar_papers_in_journal(self, title: str, abstract: str, issn: str = None, journal_title: str = None, count: int = 5) -> List[Dict[str, Any]]:
        """Find papers in a specific journal that are similar to the given paper.
        
        Args:
            title: Paper title
            abstract: Paper abstract
            issn: Journal ISSN (optional if title is provided)
            journal_title: Journal title (optional if ISSN is provided)
            count: Maximum number of papers to return
            
        Returns:
            List of similar paper dictionaries
        """
        if not issn and not journal_title:
            logger.error("Either ISSN or journal title must be provided")
            return []
        
        # Extract key terms from title and abstract
        key_terms = self._extract_key_terms(title, abstract)
        
        # Build journal constraint
        if issn:
            journal_constraint = f"AND ISSN({issn})"
        else:
            journal_constraint = f"AND EXACTSRCTITLE(\"{journal_title}\")"
            
        # Build a query with key terms
        query_terms = " OR ".join([f"KEY({term})" for term in key_terms[:5]])
        query = f"({query_terms}) {journal_constraint}"
        
        params = {
            "query": query,
            "count": count,
            "field": "title,authors,publicationName,coverDate,doi,citedby-count,description,volume,issue,pageRange",
            "sort": "relevancy desc"
        }
        
        response = self._make_request("/content/search/scopus", params)
        
        if "search-results" in response and "entry" in response["search-results"]:
            results = response["search-results"]["entry"]
            
            # Calculate similarity scores
            for i, paper in enumerate(results):
                # Simple scoring based on term overlap and citation count
                # In a real implementation, this would use more sophisticated NLP techniques
                title_overlap = sum(1 for term in key_terms if term.lower() in paper.get("dc:title", "").lower())
                abstract_overlap = sum(1 for term in key_terms if "dc:description" in paper and term.lower() in paper.get("dc:description", "").lower())
                
                # Combine with citation count for a weighted score
                citation_weight = min(int(paper.get("citedby-count", "0")) / 100, 1.0)
                similarity_score = (title_overlap * 0.5 + abstract_overlap * 0.3 + citation_weight * 0.2)
                
                # Add the score to the paper data
                paper["similarity_score"] = similarity_score
                paper["similarity_reason"] = f"Term overlap in title ({title_overlap}) and abstract ({abstract_overlap}), citation impact ({citation_weight:.2f})"
            
            # Sort by similarity score
            results.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
            return results
        else:
            logger.warning(f"No similar papers found in journal: {issn or journal_title}")
            return []
    
    def get_journal_reviewer_preferences(self, issn: str = None, journal_title: str = None) -> List[Dict[str, Any]]:
        """Extract typical reviewer preferences for a journal based on top papers.
        
        Args:
            issn: Journal ISSN (optional if title is provided)
            journal_title: Journal title (optional if ISSN is provided)
            
        Returns:
            List of reviewer preference dictionaries
        """
        # Get top papers in the journal
        top_papers = self.get_journal_top_papers(issn, journal_title, count=20)
        
        if not top_papers:
            return []
        
        # Extract keywords and phrases from abstracts
        all_abstracts = " ".join([paper.get("dc:description", "") for paper in top_papers if "dc:description" in paper])
        
        # Simple keyword extraction (in a real implementation, use NLP techniques)
        common_topics = self._extract_key_terms(all_abstracts, "", 20)
        
        # Extract methodologies
        methodologies = []
        methodology_patterns = [
            "method", "approach", "technique", "algorithm", "framework", 
            "model", "analysis", "evaluation", "experiment", "study"
        ]
        
        for paper in top_papers:
            if "dc:description" in paper:
                abstract = paper.get("dc:description", "").lower()
                for pattern in methodology_patterns:
                    if pattern in abstract:
                        # Extract phrases containing methodology terms
                        sentences = abstract.split(".")
                        for sentence in sentences:
                            if pattern in sentence:
                                methodologies.append(sentence.strip())
        
        # Create reviewer preferences
        preferences = []
        
        # Topic preferences
        for i, topic in enumerate(common_topics[:10], 1):
            preferences.append({
                "preference_type": "topic",
                "preference_value": topic,
                "importance": 10 - i + 1,  # Higher importance for more common topics
                "description": f"Papers on {topic} appear frequently in the journal's top-cited articles"
            })
        
        # Methodology preferences
        unique_methodologies = list(set(methodologies))[:10]
        for i, methodology in enumerate(unique_methodologies, 1):
            preferences.append({
                "preference_type": "methodology",
                "preference_value": methodology[:50],  # Limit length
                "importance": 10 - i + 1,
                "description": f"This methodological approach is common in the journal's top papers"
            })
        
        return preferences
    
    def format_bibtex_entry(self, paper_data: Dict[str, Any]) -> Dict[str, str]:
        """Convert Scopus paper data to BibTeX entry format.
        
        Args:
            paper_data: Paper metadata from Scopus API
            
        Returns:
            Dictionary in BibTeX entry format
        """
        # Extract publication year
        year = "2025"  # Default fallback
        if "prism:coverDate" in paper_data:
            try:
                year = paper_data["prism:coverDate"].split("-")[0]
            except:
                pass
        
        # Extract authors
        authors = ""
        if "dc:creator" in paper_data:
            authors = paper_data["dc:creator"]
        elif "authors" in paper_data and "author" in paper_data["authors"]:
            authors = " and ".join([
                f"{author.get('given-name', '')} {author.get('surname', '')}"
                for author in paper_data["authors"]["author"]
            ])
        
        # Generate citation key
        first_author = authors.split(" and ")[0].split()[-1] if authors else "Author"
        citation_key = f"{first_author}{year}"
        
        # Create BibTeX entry
        entry = {
            "ID": citation_key,
            "ENTRYTYPE": "article",
            "title": paper_data.get("dc:title", ""),
            "author": authors,
            "year": year,
            "journal": paper_data.get("prism:publicationName", ""),
            "volume": paper_data.get("prism:volume", ""),
            "number": paper_data.get("prism:issueIdentifier", ""),
            "pages": paper_data.get("prism:pageRange", ""),
            "doi": paper_data.get("prism:doi", ""),
            "url": f"https://doi.org/{paper_data.get('prism:doi', '')}" if "prism:doi" in paper_data else "",
            "abstract": paper_data.get("dc:description", "")
        }
        
        # Remove empty fields
        return {k: v for k, v in entry.items() if v}
    
    def recommend_similar_papers(self, 
                                title: str, 
                                abstract: str, 
                                count: int = 5) -> List[Dict[str, Any]]:
        """Find papers similar to the given title and abstract.
        
        Args:
            title: Paper title
            abstract: Paper abstract
            count: Maximum number of recommendations to return
            
        Returns:
            List of recommended paper metadata dictionaries
        """
        # Extract key terms from title and abstract
        key_terms = self._extract_key_terms(title, abstract)
        
        # Build a query with key terms
        query_terms = " OR ".join([f"KEY({term})" for term in key_terms[:5]])
        
        params = {
            "query": query_terms,
            "count": count,
            "field": "title,authors,publicationName,coverDate,doi,citedby-count,description",
            "sort": "relevancy desc"
        }
        
        response = self._make_request("/content/search/scopus", params)
        
        if "search-results" in response and "entry" in response["search-results"]:
            return response["search-results"]["entry"]
        else:
            logger.warning(f"No similar papers found for terms: {key_terms}")
            return []
    
    def _extract_key_terms(self, title: str, abstract: str) -> List[str]:
        """Extract key terms from title and abstract.
        
        Args:
            title: Paper title
            abstract: Paper abstract
            
        Returns:
            List of key terms
        """
        # Simple implementation - extract words longer than 4 characters
        # In a real system, this would use NLP techniques like TF-IDF or entity extraction
        combined_text = f"{title} {abstract}".lower()
        words = combined_text.split()
        
        # Remove common stop words and short words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                      'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'as', 'of', 
                      'from', 'that', 'this', 'these', 'those', 'they', 'them', 'their'}
        
        key_terms = [word for word in words if len(word) > 4 and word not in stop_words]
        
        # Return unique terms
        return list(set(key_terms))[:10]  # Limit to top 10
    
    def get_citation_overview(self, doi: str) -> Dict[str, Any]:
        """Get citation overview for a paper by DOI.
        
        Args:
            doi: DOI of the paper
            
        Returns:
            Dictionary with citation metrics
        """
        # First get the paper details to obtain the Scopus ID
        paper = self.search_by_doi(doi)
        if not paper or "dc:identifier" not in paper:
            logger.warning(f"Could not find Scopus ID for DOI: {doi}")
            return {
                "total_citations": 0,
                "recent_citations": 0,
                "citation_history": [],
                "top_citing_countries": [],
                "top_citing_journals": []
            }
        
        # Extract Scopus ID
        scopus_id = paper.get("dc:identifier", "").replace("SCOPUS_ID:", "")
        
        # Get citation overview
        params = {
            "scopus_id": scopus_id,
            "date": "2020-2025"  # Last 5 years
        }
        
        response = self._make_request(f"/content/abstract/citations/{scopus_id}", params)
        
        # Extract citation information
        citation_info = {
            "total_citations": int(paper.get("citedby-count", 0)),
            "recent_citations": 0,
            "citation_history": [],
            "top_citing_countries": [],
            "top_citing_journals": []
        }
        
        if "abstract-citations-response" in response:
            data = response["abstract-citations-response"]
            
            # Extract citation history
            if "citeColumnTotalXML" in data:
                citation_info["citation_history"] = [
                    {"year": item.get("year", ""), "count": int(item.get("columnTotal", 0))}
                    for item in data["citeColumnTotalXML"]
                ]
                
                # Calculate recent citations (last 2 years)
                current_year = datetime.datetime.now().year
                citation_info["recent_citations"] = sum(
                    int(item.get("columnTotal", 0))
                    for item in data["citeColumnTotalXML"]
                    if item.get("year", 0) and int(item.get("year", 0)) >= current_year - 2
                )
            
            # Extract top citing countries
            if "citeCountryXML" in data:
                citation_info["top_citing_countries"] = [
                    {"country": item.get("name", ""), "count": int(item.get("citationCount", 0))}
                    for item in data["citeCountryXML"]
                ]
            
            # Extract top citing journals
            if "citeSourceXML" in data:
                citation_info["top_citing_journals"] = [
                    {"journal": item.get("sourcetitle", ""), "count": int(item.get("citationCount", 0))}
                    for item in data["citeSourceXML"]
                ]
        
        return citation_info

    def analyze_journal_impact(self, issn: str) -> Dict[str, Any]:
        """Analyze journal impact metrics.
        
        Args:
            issn: Journal ISSN
            
        Returns:
            Dictionary with journal impact metrics
        """
        journal_metrics = self.get_journal_metrics(issn)
        
        if not journal_metrics:
            return {
                "title": "Unknown Journal",
                "sjr": None,
                "snip": None,
                "impact_factor": None,
                "quartile": None,
                "ranking": None,
                "subject_areas": []
            }
        
        # Extract metrics from the response
        metrics = {
            "title": journal_metrics.get("dc:title", "Unknown Journal"),
            "sjr": None,
            "snip": None,
            "impact_factor": None,
            "quartile": None,
            "ranking": None,
            "subject_areas": []
        }
        
        # Extract SJR and SNIP metrics
        if "SJR" in journal_metrics:
            metrics["sjr"] = journal_metrics["SJR"]
        if "SNIP" in journal_metrics:
            metrics["snip"] = journal_metrics["SNIP"]
        
        # Extract subject areas
        if "subject-area" in journal_metrics:
            metrics["subject_areas"] = [
                area.get("$", "")
                for area in journal_metrics["subject-area"]
            ]
        
        # Extract quartile information
        if "citeScore" in journal_metrics and "quartile" in journal_metrics["citeScore"]:
            metrics["quartile"] = journal_metrics["citeScore"]["quartile"]
        
        # Extract ranking information
        if "citeScore" in journal_metrics and "rank" in journal_metrics["citeScore"]:
            metrics["ranking"] = journal_metrics["citeScore"]["rank"]
        
        return metrics

    def generate_citation_report(self, doi: str) -> Dict[str, Any]:
        """Generate a comprehensive citation report for a paper.
        
        Args:
            doi: DOI of the paper
            
        Returns:
            Dictionary with citation report data
        """
        # Get paper details
        paper = self.search_by_doi(doi)
        if not paper:
            return {
                "status": "error",
                "message": f"Paper with DOI {doi} not found"
            }
        
        # Get citation overview
        citation_overview = self.get_citation_overview(doi)
        
        # Get recent citations
        recent_citations = self.get_citations(doi, count=5)
        
        # Get journal metrics if available
        journal_metrics = {}
        if "prism:issn" in paper:
            journal_metrics = self.analyze_journal_impact(paper["prism:issn"])
        
        # Compile the report
        report = {
            "paper": {
                "title": paper.get("dc:title", ""),
                "authors": paper.get("dc:creator", ""),
                "journal": paper.get("prism:publicationName", ""),
                "year": paper.get("prism:coverDate", "").split("-")[0] if "prism:coverDate" in paper else "",
                "doi": doi
            },
            "citations": {
                "total": citation_overview["total_citations"],
                "recent": citation_overview["recent_citations"],
                "history": citation_overview["citation_history"],
                "recent_examples": [
                    {
                        "title": cite.get("dc:title", ""),
                        "authors": cite.get("dc:creator", ""),
                        "journal": cite.get("prism:publicationName", ""),
                        "year": cite.get("prism:coverDate", "").split("-")[0] if "prism:coverDate" in cite else ""
                    }
                    for cite in recent_citations
                ]
            },
            "journal_metrics": journal_metrics,
            "geographical_impact": citation_overview["top_citing_countries"],
            "field_impact": citation_overview["top_citing_journals"],
            "generation_date": datetime.datetime.now().strftime("%Y-%m-%d")
        }
        
        return report


    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get usage statistics for the Scopus API client.
        
        Returns:
            Dictionary with usage statistics
        """
        cache_files = [f for f in os.listdir(SCOPUS_CACHE_DIR) if f.endswith('.json')]
        
        return {
            "api_key": self.api_key[-8:],  # Last 8 characters for security
            "cache_enabled": self.use_cache,
            "cache_ttl_days": self.cache_ttl / 86400,  # Convert seconds to days
            "cache_entries": len(cache_files),
            "cache_size_mb": sum(os.path.getsize(os.path.join(SCOPUS_CACHE_DIR, f)) for f in cache_files) / (1024 * 1024),
            "api_requests": self.request_count,
            "cache_hits": self.cache_hit_count,
            "cache_hit_rate": (self.cache_hit_count / max(1, self.request_count + self.cache_hit_count)) * 100,
            "rate_limit_spacing": self.request_spacing
        }

def get_scopus_client(api_key: str = None, use_cache: bool = True, cache_ttl: int = None) -> ScopusClient:
    """Factory function to get a configured Scopus client.
    
    Args:
        api_key: Scopus API key (optional)
        use_cache: Whether to use caching (default: True)
        cache_ttl: Cache time-to-live in seconds (optional)
        
    Returns:
        Configured ScopusClient instance
    """
    return ScopusClient(api_key, use_cache, cache_ttl)


# Usage example (when run as script)
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create a client
    client = get_scopus_client()
    
    # Example searches
    papers = client.search_by_title("Machine Learning")
    for paper in papers[:3]:
        print(f"Title: {paper.get('dc:title', '')}")
        print(f"Authors: {paper.get('dc:creator', '')}")
        print(f"Journal: {paper.get('prism:publicationName', '')}")
        print(f"Citations: {paper.get('citedby-count', '0')}")
        print()