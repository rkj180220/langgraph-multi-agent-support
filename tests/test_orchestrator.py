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

    # Configure supervisor mock for routing
    supervisor.process_query = AsyncMock(return_value=AgentResponse(
        success=True,
        message="Query routed to IT specialist",
        agent_name="Supervisor",
        routing_decision="IT"
    ))

    # Configure supervisor mock for evaluation
    supervisor.evaluate_response = AsyncMock(return_value=AgentResponse(
        success=True,
        message="Evaluated response",
        agent_name="Supervisor",
        metadata={"evaluated": True, "evaluation_success": True}
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
        sanitized_input="test query",
        error_message=None
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

    def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert orchestrator.supervisor is not None
        assert orchestrator.it_agent is not None
        assert orchestrator.finance_agent is not None
        assert orchestrator.validator is not None
        assert orchestrator.logger is not None
        assert orchestrator.workflow is not None

    @pytest.mark.asyncio
    async def test_process_query_it_route(self, orchestrator, mock_agents):
        """Test processing query routed to IT agent."""
        supervisor, it_agent, finance_agent = mock_agents

        # Configure supervisor to route to IT
        supervisor.process_query.return_value = AgentResponse(
            success=True,
            message="Query routed to IT specialist",
            agent_name="Supervisor",
            routing_decision="IT"
        )

        result = await orchestrator.process_query("How do I reset my password?")

        assert result["success"] is True
        assert "response" in result
        assert result["metadata"]["routing_decision"] == "IT"

        # Verify IT agent was called
        it_agent.process_query.assert_called_once()
        # Verify Finance agent was not called
        finance_agent.process_query.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_query_finance_route(self, orchestrator, mock_agents):
        """Test processing query routed to Finance agent."""
        supervisor, it_agent, finance_agent = mock_agents

        # Configure supervisor to route to Finance
        supervisor.process_query.return_value = AgentResponse(
            success=True,
            message="Query routed to Finance specialist",
            agent_name="Supervisor",
            routing_decision="Finance"
        )

        result = await orchestrator.process_query("How do I submit an expense report?")

        assert result["success"] is True
        assert "response" in result
        assert result["metadata"]["routing_decision"] == "Finance"

        # Verify Finance agent was called
        finance_agent.process_query.assert_called_once()
        # Verify IT agent was not called
        it_agent.process_query.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_query_both_route(self, orchestrator, mock_agents):
        """Test processing query routed to both agents."""
        supervisor, it_agent, finance_agent = mock_agents

        # Configure supervisor to route to both
        supervisor.process_query.return_value = AgentResponse(
            success=True,
            message="Query routed to both specialists",
            agent_name="Supervisor",
            routing_decision="Both"
        )

        result = await orchestrator.process_query("My computer broke and I need to submit an expense report")

        assert result["success"] is True
        assert "response" in result
        assert result["metadata"]["routing_decision"] == "Both"

        # Verify both agents were called
        it_agent.process_query.assert_called_once()
        finance_agent.process_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_query_validation_failure(self, orchestrator, mock_validator):
        """Test processing query with validation failure."""
        # Configure validator to fail
        mock_validator.validate_query.return_value = ValidationResult(
            is_valid=False,
            sanitized_input="",
            error_message="Query is invalid"
        )

        result = await orchestrator.process_query("invalid query")

        assert result["success"] is False
        assert "invalid" in result["response"].lower()

    @pytest.mark.asyncio
    async def test_process_query_supervisor_failure(self, orchestrator, mock_agents):
        """Test processing query with supervisor failure."""
        supervisor, it_agent, finance_agent = mock_agents

        # Configure supervisor to fail
        supervisor.process_query.return_value = AgentResponse(
            success=False,
            message="Routing failed",
            agent_name="Supervisor"
        )

        result = await orchestrator.process_query("test query")

        assert result["success"] is False
        assert "routing failed" in result["response"].lower()

    @pytest.mark.asyncio
    async def test_process_query_unclear_route(self, orchestrator, mock_agents):
        """Test processing query with unclear routing."""
        supervisor, it_agent, finance_agent = mock_agents

        # Configure supervisor to return unclear
        supervisor.process_query.return_value = AgentResponse(
            success=False,
            message="I can only help with IT or Finance related queries",
            agent_name="Supervisor",
            routing_decision="Unclear"
        )

        result = await orchestrator.process_query("What's the weather today?")

        assert result["success"] is False
        assert "IT or Finance" in result["response"]

    @pytest.mark.asyncio
    async def test_supervisor_evaluation_called(self, orchestrator, mock_agents):
        """Test that supervisor evaluation is called."""
        supervisor, it_agent, finance_agent = mock_agents

        # Configure supervisor to route to IT
        supervisor.process_query.return_value = AgentResponse(
            success=True,
            message="Query routed to IT specialist",
            agent_name="Supervisor",
            routing_decision="IT"
        )

        await orchestrator.process_query("How do I reset my password?")

        # Verify supervisor evaluation was called
        supervisor.evaluate_response.assert_called_once()

    @pytest.mark.asyncio
    async def test_processing_path_tracking(self, orchestrator, mock_agents):
        """Test that processing path is correctly tracked."""
        supervisor, it_agent, finance_agent = mock_agents

        # Configure supervisor to route to Finance
        supervisor.process_query.return_value = AgentResponse(
            success=True,
            message="Query routed to Finance specialist",
            agent_name="Supervisor",
            routing_decision="Finance"
        )

        result = await orchestrator.process_query("How do I submit an expense report?")

        assert result["success"] is True
        assert "processing_path" in result["metadata"]

        # Expected processing path for Finance query
        expected_path = [
            "Supervisor Agent (Routing)",
            "Finance Agent",
            "Supervisor Agent (Evaluation)"
        ]
        assert result["metadata"]["processing_path"] == expected_path

    @pytest.mark.asyncio
    async def test_error_handling_workflow(self, orchestrator, mock_agents):
        """Test error handling in workflow."""
        supervisor, it_agent, finance_agent = mock_agents

        # Simulate an exception during processing
        supervisor.process_query.side_effect = Exception("Test error")

        result = await orchestrator.process_query("test query")

        assert result["success"] is False
        assert "technical difficulties" in result["response"]
        assert result["error"] == "Test error"
