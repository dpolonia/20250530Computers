"""Utility modules for paper revision.

This package contains utility modules for file handling, text processing,
validation, and other common operations.
"""

# Import utilities to make them available at the package level
from src.utils.path_utils import (
    ensure_directory_exists,
    ensure_parent_directory_exists,
    get_current_timestamp,
    normalize_path,
    get_filename,
    get_extension,
    is_valid_file,
    is_valid_directory,
    construct_output_path,
    list_files_with_extension,
    create_temp_file,
    create_temp_directory,
    split_path,
    join_path,
    expand_path,
    resolve_relative_path
)

from src.utils.text_utils import (
    estimate_tokens,
    split_text_into_chunks,
    clean_whitespace,
    normalize_newlines,
    extract_sections,
    extract_json_from_text,
    clean_doi,
    extract_citations,
    highlight_changes,
    remove_urls,
    remove_html_tags,
    count_words
)

from src.utils.validation_utils import (
    validate_file_exists,
    validate_directory_exists,
    validate_file_extension,
    validate_api_key,
    validate_api_key_format,
    validate_model_name,
    validate_date_format,
    validate_email,
    validate_url,
    validate_doi,
    validate_integer_range,
    validate_float_range,
    validate_string_length,
    validate_json_schema,
    validate_reference_format,
    validate_all
)