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


@pytest.fixture
def mock_bedrock_llm():
    """Create a mock ChatBedrock LLM."""
    mock_llm = Mock()
    mock_llm.ainvoke = AsyncMock()
    return mock_llm


class TestSupervisorAgent:
    """Test supervisor agent functionality."""

    @pytest.mark.asyncio
    async def test_supervisor_process_it_query(self, mock_config, mock_tool_registry, mock_logger):
        """Test supervisor routing IT query."""
        with patch('hierarchical_multi_agent_support.agents.ChatBedrock') as mock_bedrock:
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.content = "IT - This is a technical query about network issues"
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_bedrock.return_value = mock_llm

            supervisor = SupervisorAgent(mock_config, mock_tool_registry, mock_logger)

            result = await supervisor.process_query("My computer won't connect to the network")

            assert result.success is True
            assert result.routing_decision == "IT"
            assert result.agent_name == "Supervisor"

    @pytest.mark.asyncio
    async def test_supervisor_process_finance_query(self, mock_config, mock_tool_registry, mock_logger):
        """Test supervisor routing finance query."""
        with patch('hierarchical_multi_agent_support.agents.ChatBedrock') as mock_bedrock:
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.content = "Finance - This is about expense reporting"
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_bedrock.return_value = mock_llm

            supervisor = SupervisorAgent(mock_config, mock_tool_registry, mock_logger)

            result = await supervisor.process_query("How do I submit an expense report?")

            assert result.success is True
            assert result.routing_decision == "Finance"
            assert result.agent_name == "Supervisor"

    @pytest.mark.asyncio
    async def test_supervisor_process_unclear_query(self, mock_config, mock_tool_registry, mock_logger):
        """Test supervisor handling unclear query."""
        with patch('hierarchical_multi_agent_support.agents.ChatBedrock') as mock_bedrock:
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.content = "Unclear - This query is ambiguous"
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_bedrock.return_value = mock_llm

            supervisor = SupervisorAgent(mock_config, mock_tool_registry, mock_logger)

            result = await supervisor.process_query("Hello, how are you?")

            assert result.success is False
            assert result.routing_decision == "Unclear"

    @pytest.mark.asyncio
    async def test_supervisor_process_both_query(self, mock_config, mock_tool_registry, mock_logger):
        """Test supervisor routing query to both domains."""
        with patch('hierarchical_multi_agent_support.agents.ChatBedrock') as mock_bedrock:
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.content = "Both - This requires both IT and Finance expertise"
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_bedrock.return_value = mock_llm

            supervisor = SupervisorAgent(mock_config, mock_tool_registry, mock_logger)

            result = await supervisor.process_query("My computer broke and I need to submit an expense report for a new one")

            assert result.success is True
            assert result.routing_decision == "Both"
            assert result.agent_name == "Supervisor"

    @pytest.mark.asyncio
    async def test_supervisor_evaluate_response(self, mock_config, mock_tool_registry, mock_logger):
        """Test supervisor evaluation capability."""
        with patch('hierarchical_multi_agent_support.agents.ChatBedrock') as mock_bedrock:
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.content = "Refined response based on specialist input"
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_bedrock.return_value = mock_llm

            supervisor = SupervisorAgent(mock_config, mock_tool_registry, mock_logger)

            # Create mock specialist response
            specialist_response = AgentResponse(
                success=True,
                message="Original specialist response",
                agent_name="IT Agent",
                tool_calls=[],
                metadata={}
            )

            result = await supervisor.evaluate_response(
                original_query="Test query",
                specialist_responses=[specialist_response],
                routing_decision="IT"
            )

            assert result.success is True
            assert result.agent_name == "Supervisor"
            assert result.metadata["evaluated"] is True

    def test_supervisor_parse_routing_decision(self, mock_config, mock_tool_registry, mock_logger):
        """Test routing decision parsing."""
        with patch('hierarchical_multi_agent_support.agents.ChatBedrock'):
            supervisor = SupervisorAgent(mock_config, mock_tool_registry, mock_logger)

            # Test various routing decision formats
            assert supervisor._parse_routing_decision("Finance - This is about expenses") == "Finance"
            assert supervisor._parse_routing_decision("IT - This is technical") == "IT"
            assert supervisor._parse_routing_decision("Both - This needs both domains") == "Both"
            assert supervisor._parse_routing_decision("Unclear - Cannot determine") == "Unclear"
            assert supervisor._parse_routing_decision("Random text") == "Unclear"


class TestITAgent:
    """Test IT agent functionality."""

    @pytest.mark.asyncio
    async def test_it_agent_process_query(self, mock_config, mock_tool_registry, mock_logger):
        """Test IT agent query processing."""
        with patch('hierarchical_multi_agent_support.agents.ChatBedrock') as mock_bedrock:
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.content = "Here's how to reset your password..."
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_bedrock.return_value = mock_llm

            it_agent = ITAgent(mock_config, mock_tool_registry, mock_logger)

            result = await it_agent.process_query("How do I reset my password?")

            assert result.success is True
            assert result.agent_name == "IT Agent"
            assert len(result.tool_calls) >= 1  # Should have used tools
            assert result.metadata["domain"] == "IT"

    @pytest.mark.asyncio
    async def test_it_agent_tool_failure(self, mock_config, mock_tool_registry, mock_logger):
        """Test IT agent handling tool failures."""
        with patch('hierarchical_multi_agent_support.agents.ChatBedrock') as mock_bedrock:
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.content = "Response despite tool failure"
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_bedrock.return_value = mock_llm

            # Mock tool failure
            failed_result = ToolResult(
                success=False,
                data="",
                error="Tool execution failed",
                metadata={}
            )
            mock_tool_registry.execute_tool = AsyncMock(return_value=failed_result)

            it_agent = ITAgent(mock_config, mock_tool_registry, mock_logger)

            result = await it_agent.process_query("Test query")

            assert result.success is True  # Should still succeed despite tool failure
            assert result.agent_name == "IT Agent"


class TestFinanceAgent:
    """Test Finance agent functionality."""

    @pytest.mark.asyncio
    async def test_finance_agent_process_query(self, mock_config, mock_tool_registry, mock_logger):
        """Test Finance agent query processing."""
        with patch('hierarchical_multi_agent_support.agents.ChatBedrock') as mock_bedrock:
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.content = "Here's how to submit an expense report..."
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_bedrock.return_value = mock_llm

            finance_agent = FinanceAgent(mock_config, mock_tool_registry, mock_logger)

            result = await finance_agent.process_query("How do I submit an expense report?")

            assert result.success is True
            assert result.agent_name == "Finance Agent"
            assert len(result.tool_calls) >= 1  # Should have used tools
            assert result.metadata["domain"] == "Finance"

    @pytest.mark.asyncio
    async def test_finance_agent_tool_failure(self, mock_config, mock_tool_registry, mock_logger):
        """Test Finance agent handling tool failures."""
        with patch('hierarchical_multi_agent_support.agents.ChatBedrock') as mock_bedrock:
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.content = "Response despite tool failure"
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_bedrock.return_value = mock_llm

            # Mock tool failure
            failed_result = ToolResult(
                success=False,
                data="",
                error="Tool execution failed",
                metadata={}
            )
            mock_tool_registry.execute_tool = AsyncMock(return_value=failed_result)

            finance_agent = FinanceAgent(mock_config, mock_tool_registry, mock_logger)

            result = await finance_agent.process_query("Test query")

            assert result.success is True  # Should still succeed despite tool failure
            assert result.agent_name == "Finance Agent"


class TestAgentResponse:
    """Test AgentResponse model."""

    def test_agent_response_creation(self):
        """Test creating an AgentResponse."""
        response = AgentResponse(
            success=True,
            message="Test response",
            agent_name="Test Agent",
            routing_decision="IT",
            tool_calls=[{"tool": "test", "result": "success"}],
            metadata={"test": "value"}
        )

        assert response.success is True
        assert response.message == "Test response"
        assert response.agent_name == "Test Agent"
        assert response.routing_decision == "IT"
        assert len(response.tool_calls) == 1
        assert response.metadata["test"] == "value"
