"""
Test tools for the multi-agent support system.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import logging

from hierarchical_multi_agent_support.tools import (
    WebSearchTool, ReadFileTool, ToolRegistry, ToolResult
)
from hierarchical_multi_agent_support.config import Config


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    config = Mock()

    # AWS configuration
    config.aws = Mock()
    config.aws.region = "us-west-2"
    config.aws.access_key_id = "test-key"
    config.aws.secret_access_key = "test-secret"

    # Tools configuration
    config.tools = Mock()
    config.tools.web_search = Mock()
    config.tools.web_search.enabled = True
    config.tools.web_search.timeout = 30
    config.tools.web_search.max_results = 5
    config.tools.file_reader = Mock()
    config.tools.file_reader.enabled = True
    config.tools.file_reader.max_file_size = 1048576
    config.tools.file_reader.allowed_extensions = [".txt", ".md"]

    # Documents configuration
    config.documents = Mock()
    config.documents.it_docs_path = "test_docs/it"
    config.documents.finance_docs_path = "test_docs/finance"

    return config


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    return Mock(spec=logging.Logger)


@pytest.fixture
def temp_doc_structure():
    """Create temporary document structure for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create document directories
        it_docs = temp_path / "it"
        finance_docs = temp_path / "finance"
        it_docs.mkdir(parents=True)
        finance_docs.mkdir(parents=True)

        # Create test files
        (it_docs / "test.txt").write_text("IT test content")
        (finance_docs / "test.md").write_text("Finance test content")

        yield temp_path


class TestWebSearchTool:
    """Test web search tool functionality."""

    @pytest.mark.asyncio
    async def test_web_search_success(self, mock_config, mock_logger):
        """Test successful web search."""
        tool = WebSearchTool(mock_config, mock_logger)

        result = await tool.execute(query="test query")

        assert result.success is True
        assert isinstance(result.data, list)
        assert len(result.data) <= 5
        assert all("title" in item and "url" in item and "snippet" in item for item in result.data)

    @pytest.mark.asyncio
    async def test_web_search_empty_query(self, mock_config, mock_logger):
        """Test web search with empty query."""
        tool = WebSearchTool(mock_config, mock_logger)

        result = await tool.execute(query="")

        assert result.success is False
        assert "empty" in result.error.lower()

    @pytest.mark.asyncio
    async def test_web_search_disabled(self, mock_config, mock_logger):
        """Test web search when disabled."""
        mock_config.tools.web_search.enabled = False
        tool = WebSearchTool(mock_config, mock_logger)

        result = await tool.execute(query="test query")

        assert result.success is False
        assert "disabled" in result.error.lower()


class TestReadFileTool:
    """Test file reading tool functionality."""

    @pytest.mark.asyncio
    async def test_read_file_success(self, mock_config, mock_logger, temp_doc_structure):
        """Test that read file tool redirects to RAG search."""
        mock_config.documents.it_docs_path = str(temp_doc_structure / "it")
        mock_config.documents.finance_docs_path = str(temp_doc_structure / "finance")

        tool = ReadFileTool(mock_config, mock_logger)

        result = await tool.execute(file_path="test.txt", domain="it")

        assert result.success is False
        assert "RAG search" in result.error
        assert "better results" in result.error

    @pytest.mark.asyncio
    async def test_read_file_not_found(self, mock_config, mock_logger, temp_doc_structure):
        """Test reading non-existent file redirects to RAG search."""
        mock_config.documents.it_docs_path = str(temp_doc_structure / "it")

        tool = ReadFileTool(mock_config, mock_logger)

        result = await tool.execute(file_path="nonexistent.txt", domain="it")

        assert result.success is False
        assert "RAG search" in result.error

    @pytest.mark.asyncio
    async def test_read_file_invalid_domain(self, mock_config, mock_logger):
        """Test reading file with invalid domain redirects to RAG search."""
        tool = ReadFileTool(mock_config, mock_logger)

        result = await tool.execute(file_path="test.txt", domain="invalid")

        assert result.success is False
        assert "RAG search" in result.error

    @pytest.mark.asyncio
    async def test_read_file_path_traversal(self, mock_config, mock_logger, temp_doc_structure):
        """Test path traversal redirects to RAG search."""
        mock_config.documents.it_docs_path = str(temp_doc_structure / "it")

        tool = ReadFileTool(mock_config, mock_logger)

        result = await tool.execute(file_path="../../../etc/passwd", domain="it")

        assert result.success is False
        assert "RAG search" in result.error

    @pytest.mark.asyncio
    async def test_read_file_invalid_extension(self, mock_config, mock_logger, temp_doc_structure):
        """Test reading file with invalid extension redirects to RAG search."""
        mock_config.documents.it_docs_path = str(temp_doc_structure / "it")

        # Create file with invalid extension
        (Path(temp_doc_structure) / "it" / "test.exe").write_text("executable content")

        tool = ReadFileTool(mock_config, mock_logger)

        result = await tool.execute(file_path="test.exe", domain="it")

        assert result.success is False
        assert "RAG search" in result.error

    @pytest.mark.asyncio
    async def test_read_file_disabled(self, mock_config, mock_logger):
        """Test file reading when disabled."""
        mock_config.tools.file_reader.enabled = False
        tool = ReadFileTool(mock_config, mock_logger)

        result = await tool.execute(file_path="test.txt", domain="it")

        assert result.success is False
        assert "disabled" in result.error.lower()


class TestToolRegistry:
    """Test tool registry functionality."""

    def test_tool_registry_initialization(self, mock_config, mock_logger):
        """Test tool registry initialization."""
        registry = ToolRegistry(mock_config, mock_logger)

        assert "web_search" in registry.tools
        assert "read_file" in registry.tools
        assert "rag_search" in registry.tools
        assert len(registry.list_tools()) == 3  # Updated to expect 3 tools

    def test_get_tool_existing(self, mock_config, mock_logger):
        """Test getting existing tool."""
        registry = ToolRegistry(mock_config, mock_logger)

        tool = registry.get_tool("web_search")

        assert tool is not None
        assert isinstance(tool, WebSearchTool)

    def test_get_tool_nonexistent(self, mock_config, mock_logger):
        """Test getting non-existent tool."""
        registry = ToolRegistry(mock_config, mock_logger)

        tool = registry.get_tool("nonexistent")

        assert tool is None

    @pytest.mark.asyncio
    async def test_execute_tool_success(self, mock_config, mock_logger):
        """Test successful tool execution."""
        registry = ToolRegistry(mock_config, mock_logger)

        result = await registry.execute_tool("web_search", query="test")

        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self, mock_config, mock_logger):
        """Test executing non-existent tool."""
        registry = ToolRegistry(mock_config, mock_logger)

        result = await registry.execute_tool("nonexistent", query="test")

        assert result.success is False
        assert "not found" in result.error.lower()
