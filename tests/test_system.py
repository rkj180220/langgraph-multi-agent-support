"""
Test system for the multi-agent support system.
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

from hierarchical_multi_agent_support.system import MultiAgentSupportSystem, SystemState


@pytest.fixture
def temp_config_file():
    """Create a temporary config file for testing."""
    config_content = """
aws:
  region: "us-west-2"
  access_key_id: "test-access-key"
  secret_access_key: "test-secret-key"

agents:
  supervisor:
    name: "Test Supervisor"
    description: "Test supervisor agent"
  it_agent:
    name: "Test IT Agent"
    description: "Test IT agent"
  finance_agent:
    name: "Test Finance Agent"
    description: "Test finance agent"

tools:
  web_search:
    enabled: true
    timeout: 30
    max_results: 5
  file_reader:
    enabled: true
    max_file_size: 1048576
    allowed_extensions: [".txt", ".md"]

documents:
  it_docs_path: "test_docs/it"
  finance_docs_path: "test_docs/finance"

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "test_logs/test.log"

validation:
  max_query_length: 1000
  min_query_length: 3
  allowed_characters: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .!?@#$%^&*()_+-=[]{}|;':\",./<>?`~"
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        f.flush()
        yield f.name

    os.unlink(f.name)


@pytest.fixture
def system(temp_config_file):
    """Create a system instance for testing."""
    # Mock AWS Bedrock instead of ChatOpenAI
    with patch('hierarchical_multi_agent_support.agents.ChatBedrock') as mock_bedrock:
        mock_llm = Mock()
        mock_bedrock.return_value = mock_llm

        # Mock boto3.client for RAG search
        with patch('boto3.client') as mock_boto:
            mock_bedrock_client = Mock()
            mock_boto.return_value = mock_bedrock_client

            system = MultiAgentSupportSystem(temp_config_file)
            yield system


class TestMultiAgentSupportSystem:
    """Test multi-agent support system functionality."""

    def test_system_initialization(self, system):
        """Test system initialization."""
        assert system.config is not None
        assert system.logger is not None
        assert system.tool_registry is not None
        assert system.validator is not None
        assert system.supervisor is not None
        assert system.it_agent is not None
        assert system.finance_agent is not None
        assert system.workflow is not None

    def test_get_system_info(self, system):
        """Test getting system information."""
        info = system.get_system_info()

        assert "agents" in info
        assert "tools" in info
        assert "config" in info
        assert info["agents"]["supervisor"] == "Test Supervisor"
        assert info["agents"]["it_agent"] == "Test IT Agent"
        assert info["agents"]["finance_agent"] == "Test Finance Agent"
        assert "web_search" in info["tools"]
        assert "read_file" in info["tools"]

    @pytest.mark.asyncio
    async def test_process_query_success_it(self, system):
        """Test successful IT query processing."""
        # Mock the agents' responses
        with patch.object(system.supervisor, 'process_query') as mock_supervisor:
            mock_supervisor.return_value = AsyncMock(
                success=True,
                routing_decision="IT",
                message="Routed to IT"
            )

            with patch.object(system.it_agent, 'process_query') as mock_it:
                mock_it.return_value = AsyncMock(
                    success=True,
                    message="Here's how to fix your network issue",
                    agent_name="IT Agent",
                    tool_calls=[]
                )

                result = await system.process_query("My network is down")

                assert result["success"] is True
                assert "network issue" in result["message"].lower()
                assert result["metadata"]["agent_used"] == "IT Agent"

    @pytest.mark.asyncio
    async def test_process_query_success_finance(self, system):
        """Test successful Finance query processing."""
        # Mock the agents' responses
        with patch.object(system.supervisor, 'process_query') as mock_supervisor:
            mock_supervisor.return_value = AsyncMock(
                success=True,
                routing_decision="Finance",
                message="Routed to Finance"
            )

            with patch.object(system.finance_agent, 'process_query') as mock_finance:
                mock_finance.return_value = AsyncMock(
                    success=True,
                    message="Here's how to submit expenses",
                    agent_name="Finance Agent",
                    tool_calls=[]
                )

                result = await system.process_query("How do I submit expenses?")

                assert result["success"] is True
                assert "expenses" in result["message"].lower()
                assert result["metadata"]["agent_used"] == "Finance Agent"

    @pytest.mark.asyncio
    async def test_process_query_validation_error(self, system):
        """Test query processing with validation error."""
        result = await system.process_query("")  # Empty query should fail validation

        assert result["success"] is False
        assert result["error"] is not None
        assert "empty" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_process_query_supervisor_error(self, system):
        """Test query processing with supervisor error."""
        with patch.object(system.supervisor, 'process_query') as mock_supervisor:
            mock_supervisor.return_value = AsyncMock(
                success=False,
                message="Cannot route this query"
            )

            result = await system.process_query("Valid query that supervisor cannot route")

            assert result["success"] is False
            assert "Cannot route" in result["message"]

    @pytest.mark.asyncio
    async def test_process_query_specialist_error(self, system):
        """Test query processing with specialist error."""
        with patch.object(system.supervisor, 'process_query') as mock_supervisor:
            mock_supervisor.return_value = AsyncMock(
                success=True,
                routing_decision="IT",
                message="Routed to IT"
            )

            with patch.object(system.it_agent, 'process_query') as mock_it:
                mock_it.return_value = AsyncMock(
                    success=False,
                    message="IT agent encountered an error"
                )

                result = await system.process_query("Network issue")

                assert result["success"] is False
                assert "IT agent encountered an error" in result["message"]

    @pytest.mark.asyncio
    async def test_process_query_unexpected_error(self, system):
        """Test query processing with unexpected error."""
        with patch.object(system.validator, 'validate_query', side_effect=Exception("Unexpected error")):
            result = await system.process_query("Test query")

            assert result["success"] is False
            assert "unexpected error" in result["message"].lower()


class TestSystemState:
    """Test system state functionality."""

    def test_system_state_creation(self):
        """Test system state creation."""
        state = SystemState(query="Test query")

        assert state.query == "Test query"
        assert state.validation_result is None
        assert state.supervisor_response is None
        assert state.specialist_response is None
        assert state.final_response == ""
        assert state.error is None
        assert state.metadata == {}

    def test_system_state_with_data(self):
        """Test system state with data."""
        state = SystemState(
            query="Test query",
            final_response="Test response",
            metadata={"test": "data"}
        )

        assert state.query == "Test query"
        assert state.final_response == "Test response"
        assert state.metadata == {"test": "data"}
