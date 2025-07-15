"""
Input validation for the multi-agent support system.
"""

import re
import logging
from typing import Optional, Dict, Any
from pydantic import BaseModel

from .config import Config


class ValidationResult(BaseModel):
    """Result of input validation."""
    is_valid: bool
    error_message: Optional[str] = None
    sanitized_input: Optional[str] = None


class InputValidator:
    """Validates and sanitizes user input."""

    def __init__(self, config: Config, logger: logging.Logger):
        """Initialize input validator."""
        self.config = config
        self.logger = logger
        self.max_length = config.validation.max_query_length
        self.min_length = config.validation.min_query_length
        self.allowed_chars = config.validation.allowed_characters

    def validate_query(self, query: str) -> ValidationResult:
        """Validate and sanitize user query."""
        try:
            # Check if query is None or empty
            if not query:
                return ValidationResult(
                    is_valid=False,
                    error_message="Query cannot be empty"
                )

            # Check length constraints
            if len(query) < self.min_length:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Query must be at least {self.min_length} characters long"
                )

            if len(query) > self.max_length:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Query cannot exceed {self.max_length} characters"
                )

            # Sanitize input by removing/replacing dangerous characters
            sanitized = self._sanitize_input(query)

            # Check if sanitized input is still valid
            if not sanitized.strip():
                return ValidationResult(
                    is_valid=False,
                    error_message="Query contains only invalid characters"
                )

            # Check for suspicious patterns
            if self._contains_suspicious_patterns(sanitized):
                return ValidationResult(
                    is_valid=False,
                    error_message="Query contains potentially harmful content"
                )

            self.logger.info(f"Query validation successful: {len(sanitized)} characters")
            return ValidationResult(
                is_valid=True,
                sanitized_input=sanitized
            )

        except Exception as e:
            self.logger.error(f"Error during query validation: {str(e)}")
            return ValidationResult(
                is_valid=False,
                error_message="Failed to validate query"
            )

    def _sanitize_input(self, query: str) -> str:
        """Sanitize user input by removing/replacing dangerous characters."""
        # Remove null bytes and control characters
        sanitized = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]', '', query)

        # Keep only allowed characters
        sanitized = ''.join(char for char in sanitized if char in self.allowed_chars)

        # Normalize whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()

        return sanitized

    def _contains_suspicious_patterns(self, query: str) -> bool:
        """Check for suspicious patterns that might indicate malicious input."""
        suspicious_patterns = [
            r'<script[^>]*>.*?</script>',  # Script tags
            r'javascript:',                # JavaScript URLs
            r'data:text/html',            # Data URLs
            r'vbscript:',                 # VBScript
            r'onload\s*=',                # Event handlers
            r'onerror\s*=',
            r'onclick\s*=',
            r'eval\s*\(',                 # Code execution
            r'exec\s*\(',
            r'system\s*\(',
            r'import\s+os',               # Potentially dangerous imports
            r'import\s+subprocess',
            r'__import__',
            r'\.\./',                     # Path traversal
            r'\.\.\\',
        ]

        query_lower = query.lower()
        for pattern in suspicious_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                self.logger.warning(f"Suspicious pattern detected: {pattern}")
                return True

        return False

    def validate_file_path(self, file_path: str) -> ValidationResult:
        """Validate file path input."""
        try:
            if not file_path:
                return ValidationResult(
                    is_valid=False,
                    error_message="File path cannot be empty"
                )

            # Check for path traversal attempts
            if '..' in file_path or file_path.startswith('/') or ':' in file_path:
                return ValidationResult(
                    is_valid=False,
                    error_message="Invalid file path format"
                )

            # Sanitize file path
            sanitized = re.sub(r'[^\w\-_./]', '', file_path)

            if not sanitized:
                return ValidationResult(
                    is_valid=False,
                    error_message="File path contains only invalid characters"
                )

            return ValidationResult(
                is_valid=True,
                sanitized_input=sanitized
            )

        except Exception as e:
            self.logger.error(f"Error during file path validation: {str(e)}")
            return ValidationResult(
                is_valid=False,
                error_message="Failed to validate file path"
            )
