"""
Test validation for the multi-agent support system.
"""

import pytest
import logging
from unittest.mock import Mock

from hierarchical_multi_agent_support.validation import InputValidator, ValidationResult
from hierarchical_multi_agent_support.config import Config


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    config = Mock()

    # Validation configuration
    config.validation = Mock()
    config.validation.max_query_length = 1000
    config.validation.min_query_length = 3
    config.validation.allowed_characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .!?@#$%^&*()_+-=[]{}|;':\",./<>?`~"

    return config


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    return Mock(spec=logging.Logger)


@pytest.fixture
def validator(mock_config, mock_logger):
    """Create a validator instance for testing."""
    return InputValidator(mock_config, mock_logger)


class TestInputValidator:
    """Test input validation functionality."""

    def test_validate_query_success(self, validator):
        """Test successful query validation."""
        result = validator.validate_query("How do I reset my password?")

        assert result.is_valid is True
        assert result.error_message is None
        assert result.sanitized_input == "How do I reset my password?"

    def test_validate_query_empty(self, validator):
        """Test validation with empty query."""
        result = validator.validate_query("")

        assert result.is_valid is False
        assert "empty" in result.error_message.lower()

    def test_validate_query_none(self, validator):
        """Test validation with None query."""
        result = validator.validate_query(None)

        assert result.is_valid is False
        assert "empty" in result.error_message.lower()

    def test_validate_query_too_short(self, validator):
        """Test validation with too short query."""
        result = validator.validate_query("Hi")

        assert result.is_valid is False
        assert "at least" in result.error_message.lower()

    def test_validate_query_too_long(self, validator):
        """Test validation with too long query."""
        long_query = "a" * 1001
        result = validator.validate_query(long_query)

        assert result.is_valid is False
        assert "cannot exceed" in result.error_message.lower()

    def test_validate_query_sanitization(self, validator):
        """Test query sanitization."""
        # Query with extra whitespace and control characters
        dirty_query = "  How do I   reset my password?  "
        result = validator.validate_query(dirty_query)

        assert result.is_valid is True
        # The sanitizer normalizes multiple spaces to single spaces
        assert result.sanitized_input == "How do I reset my password?"

    def test_validate_query_suspicious_script(self, validator):
        """Test validation with suspicious script content."""
        malicious_query = "How do I <script>alert('xss')</script> reset my password?"
        result = validator.validate_query(malicious_query)

        assert result.is_valid is False
        assert "harmful" in result.error_message.lower()

    def test_validate_query_suspicious_javascript(self, validator):
        """Test validation with suspicious JavaScript content."""
        malicious_query = "javascript:alert('xss')"
        result = validator.validate_query(malicious_query)

        assert result.is_valid is False
        assert "harmful" in result.error_message.lower()

    def test_validate_query_path_traversal(self, validator):
        """Test validation with path traversal attempt."""
        malicious_query = "Show me ../../../etc/passwd file"
        result = validator.validate_query(malicious_query)

        assert result.is_valid is False
        assert "harmful" in result.error_message.lower()

    def test_validate_file_path_success(self, validator):
        """Test successful file path validation."""
        result = validator.validate_file_path("documents/readme.txt")

        assert result.is_valid is True
        assert result.sanitized_input == "documents/readme.txt"

    def test_validate_file_path_empty(self, validator):
        """Test file path validation with empty path."""
        result = validator.validate_file_path("")

        assert result.is_valid is False
        assert "empty" in result.error_message.lower()

    def test_validate_file_path_traversal(self, validator):
        """Test file path validation with path traversal."""
        result = validator.validate_file_path("../../../etc/passwd")

        assert result.is_valid is False
        assert "invalid" in result.error_message.lower()

    def test_validate_file_path_absolute(self, validator):
        """Test file path validation with absolute path."""
        result = validator.validate_file_path("/etc/passwd")

        assert result.is_valid is False
        assert "invalid" in result.error_message.lower()

    def test_validate_file_path_with_colon(self, validator):
        """Test file path validation with colon (Windows drive)."""
        result = validator.validate_file_path("C:/Windows/System32")

        assert result.is_valid is False
        assert "invalid" in result.error_message.lower()

    def test_validate_file_path_sanitization(self, validator):
        """Test file path sanitization."""
        result = validator.validate_file_path("documents/file@#$.txt")

        assert result.is_valid is True
        assert result.sanitized_input == "documents/file.txt"
