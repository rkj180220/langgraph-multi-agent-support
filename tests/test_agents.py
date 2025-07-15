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


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    config = Mock()
    # AWS Bedrock configuration (not OpenAI)
    config.aws = Mock()
    config.aws.region = "us-west-2"
    config.aws.access_key_id = "test-key"
    config.aws.secret_access_key = "test-secret"

    config.bedrock = Mock()
    config.bedrock.model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
    config.bedrock.temperature = 0.1
    config.bedrock.max_tokens = 1000

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
    return Mock(spec=ToolRegistry)


class TestSupervisorAgent:
    """Test supervisor agent functionality."""

    @pytest.mark.asyncio
    async def test_supervisor_process_it_query(self, mock_config, mock_tool_registry, mock_logger):
        """Test supervisor routing IT query."""
        # Mock ChatBedrock to avoid initialization issues
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

            # Mock RAG search and web search tools properly
            rag_result = Mock(
                success=True,
                data="Internal IT documentation about network troubleshooting",
                metadata={'sources': ['troubleshooting.md'], 'chunks_found': 2}
            )
            web_result = Mock(
                success=True,
                data=[{"title": "Network Troubleshooting Guide", "url": "http://example.com", "snippet": "Check cables and connections"}],
                metadata={}
            )

            # Mock tool registry to return appropriate results for different tools
            async def mock_execute_tool(tool_name, **kwargs):
                if tool_name == "rag_search":
                    return rag_result
                elif tool_name == "web_search":
                    return web_result
                return Mock(success=False, data=None)

            mock_tool_registry.execute_tool = AsyncMock(side_effect=mock_execute_tool)

            # Mock the LLM response
            with patch.object(it_agent, '_call_llm', return_value="Here's how to fix your network issue: 1. Check cables..."):
                result = await it_agent.process_query("My network is down")

                assert result.success is True
                assert result.agent_name == "IT Agent"
                assert "network issue" in result.message.lower()
                assert len(result.tool_calls) > 0

    @pytest.mark.asyncio
    async def test_it_agent_internal_docs_found(self, mock_config, mock_tool_registry, mock_logger):
        """Test IT agent finding internal documentation."""
        with patch('hierarchical_multi_agent_support.agents.ChatBedrock') as mock_bedrock:
            mock_llm = Mock()
            mock_bedrock.return_value = mock_llm

            it_agent = ITAgent(mock_config, mock_tool_registry, mock_logger)

            # Mock successful RAG search and web search results
            rag_result = Mock(
                success=True,
                data="Password reset procedure from internal docs",
                metadata={'sources': ['security-policies.md'], 'chunks_found': 1, 'similarity_scores': [0.8]}
            )
            web_result = Mock(
                success=True,
                data=[{"title": "Password Reset Guide", "url": "http://example.com", "snippet": "How to reset passwords"}],
                metadata={}
            )

            # Mock execute_tool to return different results based on tool name
            async def mock_execute_tool(tool_name, **kwargs):
                if tool_name == "rag_search":
                    return rag_result
                elif tool_name == "web_search":
                    return web_result
                return Mock(success=False, data=None, error="Unknown tool")

            mock_tool_registry.execute_tool = AsyncMock(side_effect=mock_execute_tool)

            with patch.object(it_agent, '_call_llm', return_value="Based on internal docs: To reset password, go to..."):
                result = await it_agent.process_query("Password reset procedure")

                assert result.success is True
                assert len(result.tool_calls) >= 1

    @pytest.mark.asyncio
    async def test_it_agent_error_handling(self, mock_config, mock_tool_registry, mock_logger):
        """Test IT agent error handling."""
        with patch('hierarchical_multi_agent_support.agents.ChatBedrock') as mock_bedrock:
            mock_llm = Mock()
            mock_bedrock.return_value = mock_llm

            it_agent = ITAgent(mock_config, mock_tool_registry, mock_logger)

            # Mock tool to raise exception
            mock_tool_registry.execute_tool = AsyncMock(side_effect=Exception("Tool error"))

            result = await it_agent.process_query("Test query")

            assert result.success is False
            assert "technical difficulties" in result.message.lower()


class TestFinanceAgent:
    """Test Finance agent functionality."""

    @pytest.mark.asyncio
    async def test_finance_agent_process_query(self, mock_config, mock_tool_registry, mock_logger):
        """Test Finance agent processing query."""
        with patch('hierarchical_multi_agent_support.agents.ChatBedrock') as mock_bedrock:
            mock_llm = Mock()
            mock_bedrock.return_value = mock_llm

            finance_agent = FinanceAgent(mock_config, mock_tool_registry, mock_logger)

            # Mock RAG search and web search results
            rag_result = Mock(
                success=True,
                data="Expense submission guidelines from finance policies",
                metadata={'sources': ['expense-policy.pdf'], 'chunks_found': 3, 'similarity_scores': [0.9, 0.8, 0.7]}
            )
            web_result = Mock(
                success=True,
                data=[{"title": "Expense Guide", "url": "http://finance.com", "snippet": "Submit expense reports"}],
                metadata={}
            )

            # Mock execute_tool to return different results based on tool name
            async def mock_execute_tool(tool_name, **kwargs):
                if tool_name == "rag_search":
                    return rag_result
                elif tool_name == "web_search":
                    return web_result
                return Mock(success=False, data=None, error="Unknown tool")

            mock_tool_registry.execute_tool = AsyncMock(side_effect=mock_execute_tool)

            # Mock the LLM response
            with patch.object(finance_agent, '_call_llm', return_value="Here's how to submit expenses: 1. Gather receipts..."):
                result = await finance_agent.process_query("How do I submit expenses?")

                assert result.success is True
                assert result.agent_name == "Finance Agent"
                assert "expenses" in result.message.lower()
                assert len(result.tool_calls) > 0

    @pytest.mark.asyncio
    async def test_finance_agent_internal_docs_found(self, mock_config, mock_tool_registry, mock_logger):
        """Test Finance agent finding internal documentation."""
        with patch('hierarchical_multi_agent_support.agents.ChatBedrock') as mock_bedrock:
            mock_llm = Mock()
            mock_bedrock.return_value = mock_llm

            finance_agent = FinanceAgent(mock_config, mock_tool_registry, mock_logger)

            # Mock successful RAG search and web search results
            rag_result = Mock(
                success=True,
                data="Budget approval process from finance guidelines",
                metadata={'sources': ['budget-approval.pdf'], 'chunks_found': 2, 'similarity_scores': [0.9, 0.8]}
            )
            web_result = Mock(
                success=True,
                data=[{"title": "Budget Process", "url": "http://finance.com", "snippet": "Approval workflow"}],
                metadata={}
            )

            # Mock execute_tool to return different results based on tool name
            async def mock_execute_tool(tool_name, **kwargs):
                if tool_name == "rag_search":
                    return rag_result
                elif tool_name == "web_search":
                    return web_result
                return Mock(success=False, data=None, error="Unknown tool")

            mock_tool_registry.execute_tool = AsyncMock(side_effect=mock_execute_tool)

            with patch.object(finance_agent, '_call_llm', return_value="Based on our policies: Budget approval requires..."):
                result = await finance_agent.process_query("Budget approval process")

                assert result.success is True
                assert len(result.tool_calls) >= 1

    @pytest.mark.asyncio
    async def test_finance_agent_error_handling(self, mock_config, mock_tool_registry, mock_logger):
        """Test Finance agent error handling."""
        with patch('hierarchical_multi_agent_support.agents.ChatBedrock') as mock_bedrock:
            mock_llm = Mock()
            mock_bedrock.return_value = mock_llm

            finance_agent = FinanceAgent(mock_config, mock_tool_registry, mock_logger)

            # Mock tool to raise exception
            mock_tool_registry.execute_tool = AsyncMock(side_effect=Exception("Tool error"))

            result = await finance_agent.process_query("Test query")

            assert result.success is False
            assert "technical difficulties" in result.message.lower()
