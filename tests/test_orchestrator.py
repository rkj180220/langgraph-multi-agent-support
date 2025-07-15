"""
Test the workflow orchestrator for the multi-agent support system.
"""

import pytest
import logging
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from hierarchical_multi_agent_support.orchestrator import WorkflowOrchestrator
from hierarchical_multi_agent_support.agents import SupervisorAgent, ITAgent, FinanceAgent, AgentResponse
from hierarchical_multi_agent_support.validation import InputValidator, ValidationResult
from hierarchical_multi_agent_support.state import SystemState


@pytest.fixture
def mock_agents():
    """Create mock agents for testing."""
    supervisor = Mock(spec=SupervisorAgent)
    it_agent = Mock(spec=ITAgent)
    finance_agent = Mock(spec=FinanceAgent)

    # Configure supervisor mock
    supervisor.process_query = AsyncMock(return_value=AgentResponse(
        success=True,
        message="Query routed to IT specialist",
        agent_name="Supervisor",
        routing_decision="IT"
    ))

    # Configure IT agent mock
    it_agent.process_query = AsyncMock(return_value=AgentResponse(
        success=True,
        message="IT support response",
        agent_name="IT Agent",
        tool_calls=[{"tool": "rag_search", "success": True}],
        metadata={"domain": "IT"}
    ))

    # Configure Finance agent mock
    finance_agent.process_query = AsyncMock(return_value=AgentResponse(
        success=True,
        message="Finance support response",
        agent_name="Finance Agent",
        tool_calls=[{"tool": "rag_search", "success": True}],
        metadata={"domain": "Finance"}
    ))

    return supervisor, it_agent, finance_agent


@pytest.fixture
def mock_validator():
    """Create mock validator for testing."""
    validator = Mock(spec=InputValidator)
    validator.validate_query = Mock(return_value=ValidationResult(
        is_valid=True,
        sanitized_input="test query"
    ))
    return validator


@pytest.fixture
def mock_logger():
    """Create mock logger for testing."""
    return Mock(spec=logging.Logger)


@pytest.fixture
def orchestrator(mock_agents, mock_validator, mock_logger):
    """Create orchestrator instance for testing."""
    supervisor, it_agent, finance_agent = mock_agents
    return WorkflowOrchestrator(
        supervisor=supervisor,
        it_agent=it_agent,
        finance_agent=finance_agent,
        validator=mock_validator,
        logger=mock_logger
    )


class TestWorkflowOrchestrator:
    """Test workflow orchestrator functionality."""

    @pytest.mark.asyncio
    async def test_successful_it_query_flow(self, orchestrator, mock_agents):
        """Test successful IT query processing through the entire workflow."""
        supervisor, it_agent, finance_agent = mock_agents

        # Set up supervisor to route to IT
        supervisor.process_query.return_value = AgentResponse(
            success=True,
            message="Query routed to IT specialist",
            agent_name="Supervisor",
            routing_decision="IT"
        )

        result = await orchestrator.process_query("My computer won't start")

        assert result["success"] is True
        assert "IT support response" in result["message"]
        assert result["metadata"]["agent_used"] == "IT Agent"
        assert result["metadata"]["routing_decision"] == "IT"

        # Verify the workflow called the right methods
        supervisor.process_query.assert_called_once()
        it_agent.process_query.assert_called_once()
        finance_agent.process_query.assert_not_called()

    @pytest.mark.asyncio
    async def test_successful_finance_query_flow(self, orchestrator, mock_agents):
        """Test successful Finance query processing through the entire workflow."""
        supervisor, it_agent, finance_agent = mock_agents

        # Set up supervisor to route to Finance
        supervisor.process_query.return_value = AgentResponse(
            success=True,
            message="Query routed to Finance specialist",
            agent_name="Supervisor",
            routing_decision="Finance"
        )

        result = await orchestrator.process_query("How do I submit an expense report?")

        assert result["success"] is True
        assert "Finance support response" in result["message"]
        assert result["metadata"]["agent_used"] == "Finance Agent"
        assert result["metadata"]["routing_decision"] == "Finance"

        # Verify the workflow called the right methods
        supervisor.process_query.assert_called_once()
        finance_agent.process_query.assert_called_once()
        it_agent.process_query.assert_not_called()

    @pytest.mark.asyncio
    async def test_validation_failure(self, orchestrator, mock_validator):
        """Test handling of validation failures."""
        mock_validator.validate_query.return_value = ValidationResult(
            is_valid=False,
            error_message="Query too short"
        )

        result = await orchestrator.process_query("Hi")

        assert result["success"] is False
        assert "Query too short" in result["message"]
        assert result["error"] == "Query too short"

    @pytest.mark.asyncio
    async def test_supervisor_routing_failure(self, orchestrator, mock_agents):
        """Test handling of supervisor routing failures."""
        supervisor, it_agent, finance_agent = mock_agents

        # Set up supervisor to return unclear routing
        supervisor.process_query.return_value = AgentResponse(
            success=False,
            message="Query is unclear",
            agent_name="Supervisor",
            routing_decision="Unclear"
        )

        result = await orchestrator.process_query("Hello, how are you?")

        assert result["success"] is False
        assert "IT" in result["message"] and "Finance" in result["message"]

        # Verify no specialist agents were called
        it_agent.process_query.assert_not_called()
        finance_agent.process_query.assert_not_called()

    @pytest.mark.asyncio
    async def test_agent_processing_failure(self, orchestrator, mock_agents):
        """Test handling of agent processing failures."""
        supervisor, it_agent, finance_agent = mock_agents

        # Set up supervisor to route to IT
        supervisor.process_query.return_value = AgentResponse(
            success=True,
            message="Query routed to IT specialist",
            agent_name="Supervisor",
            routing_decision="IT"
        )

        # Set up IT agent to fail
        it_agent.process_query.return_value = AgentResponse(
            success=False,
            message="IT processing failed",
            agent_name="IT Agent"
        )

        result = await orchestrator.process_query("Network issue")

        assert result["success"] is False
        assert "Failed to get valid response from specialist" in result["message"]

    @pytest.mark.asyncio
    async def test_unexpected_error_handling(self, orchestrator, mock_agents):
        """Test handling of unexpected errors in the workflow."""
        supervisor, it_agent, finance_agent = mock_agents

        # Set up supervisor to raise an exception
        supervisor.process_query.side_effect = Exception("Unexpected error")

        result = await orchestrator.process_query("Test query")

        assert result["success"] is False
        assert "unexpected error" in result["message"].lower()
        assert result["error"] == "Unexpected error"

    @pytest.mark.asyncio
    async def test_metadata_population(self, orchestrator, mock_agents):
        """Test that metadata is properly populated in successful responses."""
        supervisor, it_agent, finance_agent = mock_agents

        # Set up a successful IT workflow
        supervisor.process_query.return_value = AgentResponse(
            success=True,
            message="Query routed to IT specialist",
            agent_name="Supervisor",
            routing_decision="IT"
        )

        it_agent.process_query.return_value = AgentResponse(
            success=True,
            message="IT support response",
            agent_name="IT Agent",
            tool_calls=[
                {"tool": "rag_search", "success": True},
                {"tool": "web_search", "success": True}
            ],
            metadata={"domain": "IT"}
        )

        result = await orchestrator.process_query("Network troubleshooting")

        assert result["success"] is True
        assert result["metadata"]["agent_used"] == "IT Agent"
        assert result["metadata"]["tools_used"] == 2
        assert result["metadata"]["routing_decision"] == "IT"


class TestWorkflowErrorHandling:
    """Test comprehensive error handling in the workflow."""

    @pytest.mark.asyncio
    async def test_validation_exception_handling(self, orchestrator, mock_validator):
        """Test handling of validation exceptions."""
        mock_validator.validate_query.side_effect = Exception("Validation error")

        result = await orchestrator.process_query("test")

        assert result["success"] is False
        assert "Failed to validate input" in result["message"]

    @pytest.mark.asyncio
    async def test_supervisor_exception_handling(self, orchestrator, mock_agents):
        """Test handling of supervisor exceptions."""
        supervisor, it_agent, finance_agent = mock_agents
        supervisor.process_query.side_effect = Exception("Supervisor error")

        result = await orchestrator.process_query("test")

        assert result["success"] is False
        assert "Failed to route query" in result["message"]

    @pytest.mark.asyncio
    async def test_it_agent_exception_handling(self, orchestrator, mock_agents):
        """Test handling of IT agent exceptions."""
        supervisor, it_agent, finance_agent = mock_agents

        supervisor.process_query.return_value = AgentResponse(
            success=True,
            message="Query routed to IT specialist",
            agent_name="Supervisor",
            routing_decision="IT"
        )

        it_agent.process_query.side_effect = Exception("IT agent error")

        result = await orchestrator.process_query("test")

        assert result["success"] is False
        assert "Failed to process IT query" in result["message"]

    @pytest.mark.asyncio
    async def test_finance_agent_exception_handling(self, orchestrator, mock_agents):
        """Test handling of Finance agent exceptions."""
        supervisor, it_agent, finance_agent = mock_agents

        supervisor.process_query.return_value = AgentResponse(
            success=True,
            message="Query routed to Finance specialist",
            agent_name="Supervisor",
            routing_decision="Finance"
        )

        finance_agent.process_query.side_effect = Exception("Finance agent error")

        result = await orchestrator.process_query("test")

        assert result["success"] is False
        assert "Failed to process Finance query" in result["message"]
