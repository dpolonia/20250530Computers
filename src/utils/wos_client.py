"""Web of Science API Client for scientific paper metadata retrieval.

This module provides a client for accessing the Web of Science API to retrieve
information about scientific papers, citations, and journals. It complements
the Scopus API integration to provide a more comprehensive view of academic literature.
"""

import os
import json
import time
import logging
import requests
from typing import Dict, List, Any, Optional, Tuple
from urllib.parse import quote
import datetime
import functools
import base64

# Configure logging
logger = logging.getLogger(__name__)

# Cache configuration
WOS_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".cache", "wos")
CACHE_TTL_DAYS = 7  # Cache time-to-live in days
CACHE_PRUNE_THRESHOLD = 500  # Number of files before pruning
CACHE_PRUNE_TARGET = 400  # Target number of files after pruning

# Create cache directory
os.makedirs(WOS_CACHE_DIR, exist_ok=True)

# Function to prune old cache files
def prune_wos_cache():
    """Prune old cache files to keep the cache size manageable."""
    try:
        cache_files = [os.path.join(WOS_CACHE_DIR, f) for f in os.listdir(WOS_CACHE_DIR) if f.endswith('.json')]
        
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
prune_wos_cache()

class WoSClient:
    """Client for interacting with the Web of Science API."""
    
    def __init__(self, 
                 client_id: str = None, 
                 client_secret: str = None, 
                 use_cache: bool = True, 
                 cache_ttl: int = None):
        """Initialize the Web of Science API client.
        
        Args:
            client_id: Web of Science API client ID
            client_secret: Web of Science API client secret
            use_cache: Whether to cache API responses (default: True)
            cache_ttl: Cache time-to-live in seconds (default: CACHE_TTL_DAYS * 86400)
        """
        # Get credentials from environment variables if not provided
        self.client_id = client_id or os.environ.get("WOS_CLIENT_ID", "lazyresearcher")
        self.client_secret = client_secret or os.environ.get("WOS_CLIENT_SECRET", "")
        
        # API settings
        self.base_url = "https://api.clarivate.com/apis/wos-starter/v1"
        self.auth_url = "https://api.clarivate.com/apis/wos-starter/auth/token"
        self.use_cache = use_cache
        self.cache_ttl = cache_ttl or (CACHE_TTL_DAYS * 86400)  # Convert days to seconds
        
        # Auth token
        self.auth_token = None
        self.token_expiry = 0
        
        # Set up base headers
        self.headers = {
            "Accept": "application/json",
            "User-Agent": "PaperRevisionTool/1.0 (Academic Research)"
        }
        
        # Track API usage
        self.request_count = 0
        self.cache_hit_count = 0
        self.last_request_time = 0
        
        # Add minimum request spacing (rate limiting)
        self.request_spacing = 1.0  # seconds between requests
        
        # Initial auth
        self._get_auth_token()
    
    def _get_auth_token(self, force_refresh: bool = False) -> str:
        """Get or refresh the authentication token.
        
        Args:
            force_refresh: Whether to force token refresh
            
        Returns:
            Authentication token
        """
        # Check if token is still valid
        if not force_refresh and self.auth_token and time.time() < self.token_expiry - 60:
            return self.auth_token
            
        # Prepare auth request
        auth_string = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()
        headers = {
            "Authorization": f"Basic {auth_string}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        data = {
            "grant_type": "client_credentials"
        }
        
        try:
            response = requests.post(self.auth_url, headers=headers, data=data)
            
            if response.status_code == 200:
                auth_data = response.json()
                self.auth_token = auth_data.get("access_token")
                expires_in = auth_data.get("expires_in", 3600)  # Default to 1 hour
                self.token_expiry = time.time() + expires_in
                
                # Update headers with token
                self.headers["Authorization"] = f"Bearer {self.auth_token}"
                
                logger.info("Successfully authenticated with Web of Science API")
                return self.auth_token
            else:
                logger.error(f"Authentication failed: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error during authentication: {e}")
            return None
    
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
        cache_path = os.path.join(WOS_CACHE_DIR, cache_key)
        
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
        if "error" in response:
            return
            
        cache_key = self._get_cache_key(endpoint, params)
        cache_path = os.path.join(WOS_CACHE_DIR, cache_key)
        
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
            cache_files = os.listdir(WOS_CACHE_DIR)
            if len(cache_files) > CACHE_PRUNE_THRESHOLD:
                # Schedule pruning on next request
                prune_wos_cache()
                
        except Exception as e:
            logger.warning(f"Error saving to cache file {cache_path}: {e}")
    
    def _make_request(self, endpoint: str, params: Dict[str, Any], retry: int = 3, backoff: float = 1.5) -> Dict[str, Any]:
        """Make a request to the Web of Science API with caching and retry logic.
        
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
        
        # Ensure we have a valid token
        if not self._get_auth_token():
            return {"error": "Authentication failed"}
        
        # Make the request with retry logic
        url = f"{self.base_url}{endpoint}"
        current_retry = 0
        wait_time = 1.0  # Initial wait time in seconds
        
        while current_retry <= retry:
            try:
                # Respect rate limits
                now = time.time()
                if self.last_request_time > 0:
                    elapsed = now - self.last_request_time
                    if elapsed < self.request_spacing:
                        time.sleep(self.request_spacing - elapsed)
                
                # Make the request
                response = requests.get(url, headers=self.headers, params=params, timeout=30)
                self.last_request_time = time.time()
                self.request_count += 1
                
                # Handle rate limiting (429 status code)
                if response.status_code == 429:
                    if current_retry < retry:
                        logger.warning(f"Rate limited by WoS API, retrying in {wait_time:.1f} seconds")
                        time.sleep(wait_time)
                        wait_time *= backoff
                        current_retry += 1
                        continue
                    else:
                        logger.error("Exceeded maximum retries for rate limit")
                        return {"error": "Rate limit exceeded after multiple retries"}
                
                # Handle token expiration (401 status code)
                if response.status_code == 401:
                    if current_retry < retry:
                        logger.warning("Token expired, refreshing and retrying")
                        self._get_auth_token(force_refresh=True)
                        current_retry += 1
                        continue
                    else:
                        logger.error("Authentication failed after token refresh")
                        return {"error": "Authentication failed after token refresh"}
                
                # Handle other response codes
                if response.status_code == 200:
                    try:
                        data = response.json()
                        # Save to cache
                        self._save_to_cache(endpoint, params, data)
                        return data
                    except json.JSONDecodeError as e:
                        logger.error(f"Error decoding JSON response: {e}")
                        return {"error": f"JSON decode error: {str(e)}"}
                else:
                    error_msg = f"WoS API error: {response.status_code} - {response.text[:500]}"
                    
                    # Retry for server errors (5xx) and some client errors
                    if (response.status_code >= 500 or response.status_code in [408, 429]) and current_retry < retry:
                        logger.warning(f"{error_msg}, retrying in {wait_time:.1f} seconds")
                        time.sleep(wait_time)
                        wait_time *= backoff
                        current_retry += 1
                        continue
                    
                    logger.error(error_msg)
                    return {"error": response.text[:500]}
            
            except requests.exceptions.Timeout:
                if current_retry < retry:
                    logger.warning(f"Request timeout, retrying in {wait_time:.1f} seconds")
                    time.sleep(wait_time)
                    wait_time *= backoff
                    current_retry += 1
                    continue
                else:
                    logger.error("Exceeded maximum retries for timeout")
                    return {"error": "Request timeout after multiple retries"}
            
            except Exception as e:
                logger.error(f"Error making request to WoS API: {e}")
                return {"error": str(e)}
                
        # This should never be reached with the return statements above
        return {"error": "Unknown error in request handling"}
    
    def search_by_title(self, title: str, count: int = 5) -> List[Dict[str, Any]]:
        """Search for papers by title.
        
        Args:
            title: Paper title to search for
            count: Maximum number of results to return
            
        Returns:
            List of paper metadata dictionaries
        """
        params = {
            "q": f"TI=({title})",
            "limit": count,
            "orderBy": "TR desc"  # Sort by times cited (descending)
        }
        
        response = self._make_request("/search", params)
        
        if "error" in response:
            logger.warning(f"Error searching for title: {title}")
            return []
            
        if "Data" in response and "Records" in response["Data"]:
            return response["Data"]["Records"]
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
            "q": f"DO=({doi})",
            "limit": 1
        }
        
        response = self._make_request("/search", params)
        
        if "error" in response:
            logger.warning(f"Error searching for DOI: {doi}")
            return None
            
        if "Data" in response and "Records" in response["Data"] and len(response["Data"]["Records"]) > 0:
            return response["Data"]["Records"][0]
        else:
            logger.warning(f"No results found for DOI: {doi}")
            return None
    
    def get_cited_references(self, uid: str, count: int = 50) -> List[Dict[str, Any]]:
        """Get references cited by a paper.
        
        Args:
            uid: Web of Science UID of the paper
            count: Maximum number of references to return
            
        Returns:
            List of reference dictionaries
        """
        params = {
            "limit": count
        }
        
        response = self._make_request(f"/documents/{uid}/references", params)
        
        if "error" in response:
            logger.warning(f"Error getting cited references for UID: {uid}")
            return []
            
        if "Data" in response and "References" in response["Data"]:
            return response["Data"]["References"]
        else:
            logger.warning(f"No cited references found for UID: {uid}")
            return []
    
    def get_citing_papers(self, uid: str, count: int = 50) -> List[Dict[str, Any]]:
        """Get papers that cite the given paper.
        
        Args:
            uid: Web of Science UID of the paper
            count: Maximum number of citing papers to return
            
        Returns:
            List of citing paper dictionaries
        """
        params = {
            "limit": count
        }
        
        response = self._make_request(f"/documents/{uid}/citations", params)
        
        if "error" in response:
            logger.warning(f"Error getting citing papers for UID: {uid}")
            return []
            
        if "Data" in response and "Citations" in response["Data"]:
            return response["Data"]["Citations"]
        else:
            logger.warning(f"No citing papers found for UID: {uid}")
            return []
    
    def search_journal(self, journal_name: str, count: int = 5) -> List[Dict[str, Any]]:
        """Search for journals by name.
        
        Args:
            journal_name: Journal name to search for
            count: Maximum number of results to return
            
        Returns:
            List of journal dictionaries
        """
        params = {
            "q": f"SO=({journal_name})",
            "limit": count
        }
        
        response = self._make_request("/search", params)
        
        if "error" in response:
            logger.warning(f"Error searching for journal: {journal_name}")
            return []
            
        if "Data" in response and "Records" in response["Data"]:
            return response["Data"]["Records"]
        else:
            logger.warning(f"No results found for journal: {journal_name}")
            return []
    
    def get_journal_impact(self, journal_name: str) -> Dict[str, Any]:
        """Get impact metrics for a journal.
        
        Args:
            journal_name: Journal name
            
        Returns:
            Dictionary with journal impact metrics
        """
        # Search for the journal
        journals = self.search_journal(journal_name, count=1)
        
        if not journals:
            return {
                "journal_name": journal_name,
                "impact_factor": None,
                "jcr_category": None,
                "quartile": None,
                "papers_count": None
            }
        
        # Extract journal information
        journal = journals[0]
        
        # Prepare the result
        result = {
            "journal_name": journal.get("SourceTitle", journal_name),
            "impact_factor": None,
            "jcr_category": None,
            "quartile": None,
            "papers_count": None
        }
        
        # Extract journal metrics if available
        if "JCR" in journal:
            jcr = journal["JCR"]
            result["impact_factor"] = jcr.get("ImpactFactor")
            result["jcr_category"] = jcr.get("Category")
            result["quartile"] = jcr.get("Quartile")
        
        # Extract publication count
        if "PublicationCount" in journal:
            result["papers_count"] = journal["PublicationCount"]
        
        return result
    
    def find_similar_papers(self, title: str, abstract: str, count: int = 5) -> List[Dict[str, Any]]:
        """Find papers similar to the given title and abstract.
        
        Args:
            title: Paper title
            abstract: Paper abstract
            count: Maximum number of papers to return
            
        Returns:
            List of similar paper dictionaries
        """
        # Extract key terms from title and abstract
        key_terms = self._extract_key_terms(title, abstract)
        
        # Build a query with key terms
        query = " OR ".join([f"TS=({term})" for term in key_terms[:5]])
        
        params = {
            "q": query,
            "limit": count,
            "orderBy": "TR desc"  # Sort by times cited (descending)
        }
        
        response = self._make_request("/search", params)
        
        if "error" in response:
            logger.warning(f"Error searching for similar papers")
            return []
            
        if "Data" in response and "Records" in response["Data"]:
            return response["Data"]["Records"]
        else:
            logger.warning(f"No similar papers found")
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
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get usage statistics for the Web of Science API client.
        
        Returns:
            Dictionary with usage statistics
        """
        cache_files = [f for f in os.listdir(WOS_CACHE_DIR) if f.endswith('.json')]
        
        return {
            "client_id": self.client_id,
            "token_valid": self.auth_token is not None and time.time() < self.token_expiry,
            "token_expires_in": max(0, self.token_expiry - time.time()) if self.token_expiry else 0,
            "cache_enabled": self.use_cache,
            "cache_ttl_days": self.cache_ttl / 86400,  # Convert seconds to days
            "cache_entries": len(cache_files),
            "cache_size_mb": sum(os.path.getsize(os.path.join(WOS_CACHE_DIR, f)) for f in cache_files) / (1024 * 1024) if cache_files else 0,
            "api_requests": self.request_count,
            "cache_hits": self.cache_hit_count,
            "cache_hit_rate": (self.cache_hit_count / max(1, self.request_count + self.cache_hit_count)) * 100,
            "rate_limit_spacing": self.request_spacing
        }
    
    def format_bibtex_entry(self, paper_data: Dict[str, Any]) -> Dict[str, str]:
        """Convert Web of Science paper data to BibTeX entry format.
        
        Args:
            paper_data: Paper metadata from Web of Science API
            
        Returns:
            Dictionary in BibTeX entry format
        """
        # Extract publication year
        year = "2025"  # Default fallback
        if "Year" in paper_data:
            year = paper_data["Year"]
        
        # Extract authors
        authors = ""
        if "Authors" in paper_data:
            authors = " and ".join(paper_data["Authors"])
        
        # Generate citation key
        first_author = authors.split(" and ")[0].split()[-1] if authors else "Author"
        citation_key = f"{first_author}{year}"
        
        # Extract DOI
        doi = ""
        if "DOI" in paper_data:
            doi = paper_data["DOI"]
        
        # Create BibTeX entry
        entry = {
            "ID": citation_key,
            "ENTRYTYPE": "article",
            "title": paper_data.get("Title", ""),
            "author": authors,
            "year": year,
            "journal": paper_data.get("SourceTitle", ""),
            "volume": paper_data.get("Volume", ""),
            "number": paper_data.get("Issue", ""),
            "pages": paper_data.get("Pages", ""),
            "doi": doi,
            "url": f"https://doi.org/{doi}" if doi else "",
            "abstract": paper_data.get("Abstract", "")
        }
        
        # Remove empty fields
        return {k: v for k, v in entry.items() if v}


def get_wos_client(client_id: str = None, client_secret: str = None, use_cache: bool = True, cache_ttl: int = None) -> WoSClient:
    """Factory function to get a configured Web of Science client.
    
    Args:
        client_id: Web of Science API client ID (optional)
        client_secret: Web of Science API client secret (optional)
        use_cache: Whether to use caching (default: True)
        cache_ttl: Cache time-to-live in seconds (optional)
        
    Returns:
        Configured WoSClient instance
    """
    return WoSClient(client_id, client_secret, use_cache, cache_ttl)


# Usage example (when run as script)
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create a client
    client = get_wos_client()
    
    # Example searches
    papers = client.search_by_title("Machine Learning")
    for paper in papers[:3]:
        print(f"Title: {paper.get('Title', '')}")
        print(f"Authors: {', '.join(paper.get('Authors', []))}")
        print(f"Journal: {paper.get('SourceTitle', '')}")
        print(f"Citations: {paper.get('CitationCount', '0')}")
        print()