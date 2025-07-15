"""
Test agents for the multi-agent support system.
"""

import pytest
import logging
from unittest.mock import Mock, AsyncMock, patch

from hierarchical_multi_agent_support.agents import (
    SupervisorAgent, ITAgent, FinanceAgent, AgentResponse
)
from hierarchical_multi_agent_support.config import Config
from hierarchical_multi_agent_support.tools import ToolRegistry
from hierarchical_multi_agent_support.models import ToolResult


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    config = Mock()
    # AWS configuration (updated to match current structure)
    config.aws = Mock()
    config.aws.region = "us-west-2"
    config.aws.access_key_id = "test-key"
    config.aws.secret_access_key = "test-secret"
    config.aws.model = "anthropic.claude-3-sonnet-20240229-v1:0"
    config.aws.temperature = 0.1
    config.aws.max_tokens = 1000

    config.agents = Mock()
    config.agents.supervisor = Mock()
    config.agents.supervisor.name = "Test Supervisor"
    config.agents.it_agent = Mock()
    config.agents.it_agent.name = "Test IT Agent"
    config.agents.finance_agent = Mock()
    config.agents.finance_agent.name = "Test Finance Agent"

    config.documents = Mock()
    config.documents.it_docs_path = "test_docs/it"
    config.documents.finance_docs_path = "test_docs/finance"

    # Add tools configuration
    config.tools = Mock()
    config.tools.web_search = Mock()
    config.tools.web_search.enabled = True
    config.tools.web_search.timeout = 30
    config.tools.web_search.max_results = 5
    config.tools.file_reader = Mock()
    config.tools.file_reader.enabled = True
    config.tools.file_reader.max_file_size = 1048576
    config.tools.file_reader.allowed_extensions = [".txt", ".md"]

    # Add validation configuration
    config.validation = Mock()
    config.validation.max_query_length = 1000

    return config


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    return Mock(spec=logging.Logger)


@pytest.fixture
def mock_tool_registry():
    """Create a mock tool registry for testing."""
    registry = Mock(spec=ToolRegistry)

    # Mock successful RAG search result
    rag_result = ToolResult(
        success=True,
        data="Mock RAG search result with relevant context",
        metadata={
            "sources": ["test_doc.pdf"],
            "chunks_found": 3,
            "similarity_scores": [0.8, 0.7, 0.6]
        }
    )

    # Mock successful web search result
    web_result = ToolResult(
        success=True,
        data=[
            {"title": "Test Result", "url": "https://example.com", "snippet": "Test snippet"}
        ],
        metadata={"query": "test", "results_count": 1}
    )

    # Configure mock to return these results
    registry.execute_tool = AsyncMock()
    registry.execute_tool.side_effect = lambda tool_name, **kwargs: (
        rag_result if tool_name == "rag_search" else web_result
    )

    return registry


class TestSupervisorAgent:
    """Test supervisor agent functionality."""

    @pytest.mark.asyncio
    async def test_supervisor_process_it_query(self, mock_config, mock_tool_registry, mock_logger):
        """Test supervisor routing IT query."""
        with patch('hierarchical_multi_agent_support.agents.ChatBedrock') as mock_bedrock:
            mock_llm = Mock()
            mock_bedrock.return_value = mock_llm

            supervisor = SupervisorAgent(mock_config, mock_tool_registry, mock_logger)

            # Mock the LLM response
            with patch.object(supervisor, '_call_llm', return_value="IT - This is a technical query about network issues"):
                result = await supervisor.process_query("My computer won't connect to the network")

                assert result.success is True
                assert result.routing_decision == "IT"
                assert result.agent_name == "Supervisor"

    @pytest.mark.asyncio
    async def test_supervisor_process_finance_query(self, mock_config, mock_tool_registry, mock_logger):
        """Test supervisor routing finance query."""
        with patch('hierarchical_multi_agent_support.agents.ChatBedrock') as mock_bedrock:
            mock_llm = Mock()
            mock_bedrock.return_value = mock_llm

            supervisor = SupervisorAgent(mock_config, mock_tool_registry, mock_logger)

            # Mock the LLM response
            with patch.object(supervisor, '_call_llm', return_value="Finance - This is about expense reporting"):
                result = await supervisor.process_query("How do I submit an expense report?")

                assert result.success is True
                assert result.routing_decision == "Finance"
                assert result.agent_name == "Supervisor"

    @pytest.mark.asyncio
    async def test_supervisor_process_unclear_query(self, mock_config, mock_tool_registry, mock_logger):
        """Test supervisor handling unclear query."""
        with patch('hierarchical_multi_agent_support.agents.ChatBedrock') as mock_bedrock:
            mock_llm = Mock()
            mock_bedrock.return_value = mock_llm

            supervisor = SupervisorAgent(mock_config, mock_tool_registry, mock_logger)

            # Mock the LLM response
            with patch.object(supervisor, '_call_llm', return_value="Unclear - This query is ambiguous"):
                result = await supervisor.process_query("Hello, how are you?")

                assert result.success is False
                assert result.routing_decision == "Unclear"
                assert "IT or Finance" in result.message

    @pytest.mark.asyncio
    async def test_supervisor_llm_error(self, mock_config, mock_tool_registry, mock_logger):
        """Test supervisor handling LLM error."""
        with patch('hierarchical_multi_agent_support.agents.ChatBedrock') as mock_bedrock:
            mock_llm = Mock()
            mock_bedrock.return_value = mock_llm

            supervisor = SupervisorAgent(mock_config, mock_tool_registry, mock_logger)

            # Mock the LLM to raise an exception
            with patch.object(supervisor, '_call_llm', side_effect=Exception("API Error")):
                result = await supervisor.process_query("Test query")

                assert result.success is False
                assert "technical difficulties" in result.message.lower()


class TestITAgent:
    """Test IT agent functionality."""

    @pytest.mark.asyncio
    async def test_it_agent_process_query(self, mock_config, mock_tool_registry, mock_logger):
        """Test IT agent processing query."""
        with patch('hierarchical_multi_agent_support.agents.ChatBedrock') as mock_bedrock:
            mock_llm = Mock()
            mock_bedrock.return_value = mock_llm

            it_agent = ITAgent(mock_config, mock_tool_registry, mock_logger)

            # Mock the LLM response
            with patch.object(it_agent, '_call_llm', return_value="Here's how to fix your network issue..."):
                result = await it_agent.process_query("My network is down")

                assert result.success is True
                assert result.agent_name == "IT Agent"
                assert result.metadata["domain"] == "IT"
                assert len(result.tool_calls) > 0

    @pytest.mark.asyncio
    async def test_it_agent_rag_search_failure(self, mock_config, mock_tool_registry, mock_logger):
        """Test IT agent handling RAG search failure."""
        with patch('hierarchical_multi_agent_support.agents.ChatBedrock') as mock_bedrock:
            mock_llm = Mock()
            mock_bedrock.return_value = mock_llm

            # Mock RAG search failure
            failed_result = ToolResult(success=False, error="RAG search failed")
            mock_tool_registry.execute_tool.side_effect = lambda tool_name, **kwargs: (
                failed_result if tool_name == "rag_search" else
                ToolResult(success=True, data=[])
            )

            it_agent = ITAgent(mock_config, mock_tool_registry, mock_logger)

            with patch.object(it_agent, '_call_llm', return_value="Based on general knowledge..."):
                result = await it_agent.process_query("Network troubleshooting")

                assert result.success is True
                assert result.agent_name == "IT Agent"
                # Should still have tool calls even if RAG search fails
                assert any(call["tool"] == "rag_search" and not call["success"] for call in result.tool_calls)


class TestFinanceAgent:
    """Test Finance agent functionality."""

    @pytest.mark.asyncio
    async def test_finance_agent_process_query(self, mock_config, mock_tool_registry, mock_logger):
        """Test Finance agent processing query."""
        with patch('hierarchical_multi_agent_support.agents.ChatBedrock') as mock_bedrock:
            mock_llm = Mock()
            mock_bedrock.return_value = mock_llm

            finance_agent = FinanceAgent(mock_config, mock_tool_registry, mock_logger)

            # Mock the LLM response
            with patch.object(finance_agent, '_call_llm', return_value="According to policy document..."):
                result = await finance_agent.process_query("What is the expense policy?")

                assert result.success is True
                assert result.agent_name == "Finance Agent"
                assert result.metadata["domain"] == "Finance"
                assert len(result.tool_calls) > 0

    @pytest.mark.asyncio
    async def test_finance_agent_processing_error(self, mock_config, mock_tool_registry, mock_logger):
        """Test Finance agent handling processing error."""
        with patch('hierarchical_multi_agent_support.agents.ChatBedrock') as mock_bedrock:
            mock_llm = Mock()
            mock_bedrock.return_value = mock_llm

            finance_agent = FinanceAgent(mock_config, mock_tool_registry, mock_logger)

            # Mock tool registry to raise exception
            mock_tool_registry.execute_tool.side_effect = Exception("Tool execution failed")

            result = await finance_agent.process_query("Expense policy question")

            assert result.success is False
            assert "technical difficulties" in result.message.lower()
            assert result.agent_name == "Finance Agent"


class TestAgentResponse:
    """Test AgentResponse model."""

    def test_agent_response_creation(self):
        """Test creating an AgentResponse."""
        response = AgentResponse(
            success=True,
            message="Test message",
            agent_name="Test Agent",
            routing_decision="IT",
            tool_calls=[{"tool": "test", "result": "success"}],
            metadata={"domain": "IT"}
        )

        assert response.success is True
        assert response.message == "Test message"
        assert response.agent_name == "Test Agent"
        assert response.routing_decision == "IT"
        assert len(response.tool_calls) == 1
        assert response.metadata["domain"] == "IT"

    def test_agent_response_defaults(self):
        """Test AgentResponse with default values."""
        response = AgentResponse(
            success=False,
            message="Error message",
            agent_name="Test Agent"
        )

        assert response.success is False
        assert response.message == "Error message"
        assert response.agent_name == "Test Agent"
        assert response.routing_decision is None
        assert response.tool_calls == []
        assert response.metadata == {}
