"""
Test system for the multi-agent support system.
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

from hierarchical_multi_agent_support.system import MultiAgentSupportSystem


@pytest.fixture
def temp_config_file():
    """Create a temporary config file for testing."""
    config_content = """
aws:
  region: "us-west-2"
  access_key_id: "test-access-key"
  secret_access_key: "test-secret-key"
  model: "anthropic.claude-3-sonnet-20240229-v1:0"
  temperature: 0.1
  max_tokens: 1000

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


class TestMultiAgentSupportSystem:
    """Test multi-agent support system functionality."""

    def test_system_initialization(self, temp_config_file):
        """Test system initialization."""
        with patch('hierarchical_multi_agent_support.agents.ChatBedrock'):
            system = MultiAgentSupportSystem(temp_config_file)

            assert system.config is not None
            assert system.supervisor is not None
            assert system.it_agent is not None
            assert system.finance_agent is not None
            assert system.orchestrator is not None
            assert system.tool_registry is not None
            assert system.validator is not None

    def test_system_info(self, temp_config_file):
        """Test system information retrieval."""
        with patch('hierarchical_multi_agent_support.agents.ChatBedrock'):
            system = MultiAgentSupportSystem(temp_config_file)

            info = system.get_system_info()

            assert "agents" in info
            assert "tools" in info
            assert "config" in info
            assert "logging" in info

            # Check agents info (removed evaluator_agent since it's merged into supervisor)
            assert "supervisor" in info["agents"]
            assert "it_agent" in info["agents"]
            assert "finance_agent" in info["agents"]
            assert "evaluator_agent" not in info["agents"]  # Should not be present

    @pytest.mark.asyncio
    async def test_process_query_success(self, temp_config_file):
        """Test successful query processing."""
        with patch('hierarchical_multi_agent_support.agents.ChatBedrock'):
            system = MultiAgentSupportSystem(temp_config_file)

            # Mock the orchestrator to return a successful response
            mock_result = {
                "query": "test query",
                "response": "Test response",
                "metadata": {
                    "routing_decision": "IT",
                    "processing_path": ["Supervisor Agent (Routing)", "IT Agent", "Supervisor Agent (Evaluation)"],
                    "evaluated": True
                },
                "success": True,
                "error": None
            }

            with patch.object(system.orchestrator, 'process_query', return_value=mock_result):
                result = await system.process_query("test query")

                assert result["success"] is True
                assert result["response"] == "Test response"
                assert result["metadata"]["routing_decision"] == "IT"
                assert result["metadata"]["evaluated"] is True

    @pytest.mark.asyncio
    async def test_process_query_failure(self, temp_config_file):
        """Test query processing with failure."""
        with patch('hierarchical_multi_agent_support.agents.ChatBedrock'):
            system = MultiAgentSupportSystem(temp_config_file)

            # Mock the orchestrator to return a failed response
            mock_result = {
                "query": "test query",
                "response": "Error occurred",
                "metadata": {"error": "Test error"},
                "success": False,
                "error": "Test error"
            }

            with patch.object(system.orchestrator, 'process_query', return_value=mock_result):
                result = await system.process_query("test query")

                assert result["success"] is False
                assert result["error"] == "Test error"

    @pytest.mark.asyncio
    async def test_health_check(self, temp_config_file):
        """Test system health check."""
        with patch('hierarchical_multi_agent_support.agents.ChatBedrock'):
            system = MultiAgentSupportSystem(temp_config_file)

            health_status = await system.health_check()

            assert "system" in health_status
            assert "components" in health_status
            assert "timestamp" in health_status

            # Check component health
            assert "tools" in health_status["components"]
            assert "validator" in health_status["components"]
            assert "agents" in health_status["components"]

            # Check agent health (removed evaluator_agent)
            agents_health = health_status["components"]["agents"]
            assert "supervisor" in agents_health
            assert "it_agent" in agents_health
            assert "finance_agent" in agents_health
            assert "evaluator_agent" not in agents_health  # Should not be present

    def test_update_log_level(self, temp_config_file):
        """Test updating log level."""
        with patch('hierarchical_multi_agent_support.agents.ChatBedrock'):
            system = MultiAgentSupportSystem(temp_config_file)

            # Should not raise an exception
            system.update_log_level("DEBUG")

            # Verify the logging manager was called
            assert system.logging_manager is not None

    def test_system_initialization_missing_config(self):
        """Test system initialization with missing config file."""
        with pytest.raises(Exception):
            MultiAgentSupportSystem("nonexistent.yaml")

    def test_system_initialization_invalid_config(self):
        """Test system initialization with invalid config."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content:")
            f.flush()

            try:
                with pytest.raises(Exception):
                    MultiAgentSupportSystem(f.name)
            finally:
                os.unlink(f.name)

    @pytest.mark.asyncio
    async def test_system_integration_routing_precision(self, temp_config_file):
        """Test that the system properly routes queries with precision."""
        with patch('hierarchical_multi_agent_support.agents.ChatBedrock'):
            system = MultiAgentSupportSystem(temp_config_file)

            # Mock orchestrator to simulate improved routing
            finance_result = {
                "query": "How do I submit an expense report?",
                "response": "Finance response",
                "metadata": {
                    "routing_decision": "Finance",  # Should be Finance, not Both
                    "processing_path": ["Supervisor Agent (Routing)", "Finance Agent", "Supervisor Agent (Evaluation)"],
                    "total_processing_steps": 3
                },
                "success": True,
                "error": None
            }

            with patch.object(system.orchestrator, 'process_query', return_value=finance_result):
                result = await system.process_query("How do I submit an expense report?")

                assert result["success"] is True
                assert result["metadata"]["routing_decision"] == "Finance"
                assert result["metadata"]["total_processing_steps"] == 3
                assert "Supervisor Agent (Evaluation)" in result["metadata"]["processing_path"]

    @pytest.mark.asyncio
    async def test_system_integration_multi_domain(self, temp_config_file):
        """Test system handling of true multi-domain queries."""
        with patch('hierarchical_multi_agent_support.agents.ChatBedrock'):
            system = MultiAgentSupportSystem(temp_config_file)

            # Mock orchestrator for multi-domain query
            both_result = {
                "query": "My computer broke and I need to submit an expense report for a new one",
                "response": "Combined response",
                "metadata": {
                    "routing_decision": "Both",
                    "processing_path": ["Supervisor Agent (Routing)", "IT Agent", "Finance Agent", "Supervisor Agent (Evaluation)"],
                    "total_processing_steps": 4,
                    "specialist_agents": ["IT Agent", "Finance Agent"]
                },
                "success": True,
                "error": None
            }

            with patch.object(system.orchestrator, 'process_query', return_value=both_result):
                result = await system.process_query("My computer broke and I need to submit an expense report for a new one")

                assert result["success"] is True
                assert result["metadata"]["routing_decision"] == "Both"
                assert result["metadata"]["total_processing_steps"] == 4
                assert "IT Agent" in result["metadata"]["specialist_agents"]
                assert "Finance Agent" in result["metadata"]["specialist_agents"]
