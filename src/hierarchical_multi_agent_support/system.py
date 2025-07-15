"""
Main multi-agent support system orchestrator using LangGraph.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel

from .config import Config, ConfigManager
from .agents import SupervisorAgent, ITAgent, FinanceAgent, AgentResponse
from .tools import ToolRegistry
from .validation import InputValidator, ValidationResult


class SystemState(BaseModel):
    """State of the multi-agent system."""
    query: str
    validation_result: Optional[ValidationResult] = None
    supervisor_response: Optional[AgentResponse] = None
    specialist_response: Optional[AgentResponse] = None
    final_response: str = ""
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}


class MultiAgentSupportSystem:
    """Main orchestrator for the multi-agent support system."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the multi-agent support system."""
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.load_config()
        self.config_manager.validate_config(self.config)

        # Setup logging
        self.logger = self._setup_logging()

        # Initialize components
        self.tool_registry = ToolRegistry(self.config, self.logger)
        self.validator = InputValidator(self.config, self.logger)

        # Initialize agents
        self.supervisor = SupervisorAgent(self.config, self.tool_registry, self.logger)
        self.it_agent = ITAgent(self.config, self.tool_registry, self.logger)
        self.finance_agent = FinanceAgent(self.config, self.tool_registry, self.logger)

        # Build LangGraph workflow
        self.workflow = self._build_workflow()

        self.logger.info("Multi-agent support system initialized successfully")

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger("MultiAgentSupport")
        logger.setLevel(getattr(logging, self.config.logging.level))

        # Create logs directory if it doesn't exist
        log_file = Path(self.config.logging.file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, self.config.logging.level))

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(self.config.logging.format)
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(SystemState)

        # Add nodes
        workflow.add_node("validate_input", self._validate_input)
        workflow.add_node("supervisor_route", self._supervisor_route)
        workflow.add_node("it_specialist", self._it_specialist)
        workflow.add_node("finance_specialist", self._finance_specialist)
        workflow.add_node("format_response", self._format_response)
        workflow.add_node("handle_error", self._handle_error)

        # Add edges
        workflow.add_edge(START, "validate_input")

        # From validation
        workflow.add_conditional_edges(
            "validate_input",
            self._validation_router,
            {
                "valid": "supervisor_route",
                "invalid": "handle_error"
            }
        )

        # From supervisor
        workflow.add_conditional_edges(
            "supervisor_route",
            self._supervisor_router,
            {
                "IT": "it_specialist",
                "Finance": "finance_specialist",
                "error": "handle_error"
            }
        )

        # From specialists to response formatting
        workflow.add_edge("it_specialist", "format_response")
        workflow.add_edge("finance_specialist", "format_response")

        # Final edges
        workflow.add_edge("format_response", END)
        workflow.add_edge("handle_error", END)

        return workflow.compile()

    async def _validate_input(self, state: SystemState) -> SystemState:
        """Validate user input."""
        try:
            validation_result = self.validator.validate_query(state.query)
            state.validation_result = validation_result

            if validation_result.is_valid:
                state.query = validation_result.sanitized_input
                self.logger.info(f"Input validation successful")
            else:
                state.error = validation_result.error_message
                self.logger.warning(f"Input validation failed: {validation_result.error_message}")

            return state

        except Exception as e:
            self.logger.error(f"Error in input validation: {str(e)}")
            state.error = "Failed to validate input"
            return state

    async def _supervisor_route(self, state: SystemState) -> SystemState:
        """Route query using supervisor agent."""
        try:
            supervisor_response = await self.supervisor.process_query(state.query)
            state.supervisor_response = supervisor_response

            if not supervisor_response.success:
                state.error = supervisor_response.message

            return state

        except Exception as e:
            self.logger.error(f"Error in supervisor routing: {str(e)}")
            state.error = "Failed to route query"
            return state

    async def _it_specialist(self, state: SystemState) -> SystemState:
        """Process query with IT specialist."""
        try:
            it_response = await self.it_agent.process_query(state.query)
            state.specialist_response = it_response

            if not it_response.success:
                state.error = it_response.message

            return state

        except Exception as e:
            self.logger.error(f"Error in IT specialist processing: {str(e)}")
            state.error = "Failed to process IT query"
            return state

    async def _finance_specialist(self, state: SystemState) -> SystemState:
        """Process query with Finance specialist."""
        try:
            finance_response = await self.finance_agent.process_query(state.query)
            state.specialist_response = finance_response

            if not finance_response.success:
                state.error = finance_response.message

            return state

        except Exception as e:
            self.logger.error(f"Error in Finance specialist processing: {str(e)}")
            state.error = "Failed to process Finance query"
            return state

    async def _format_response(self, state: SystemState) -> SystemState:
        """Format the final response."""
        try:
            if state.specialist_response and state.specialist_response.success:
                state.final_response = state.specialist_response.message

                # Add metadata
                state.metadata = {
                    "agent_used": state.specialist_response.agent_name,
                    "tools_used": len(state.specialist_response.tool_calls),
                    "routing_decision": state.supervisor_response.routing_decision if state.supervisor_response else "Unknown"
                }

                self.logger.info(f"Response formatted successfully by {state.specialist_response.agent_name}")
            else:
                state.error = "Failed to get valid response from specialist"

            return state

        except Exception as e:
            self.logger.error(f"Error in response formatting: {str(e)}")
            state.error = "Failed to format response"
            return state

    async def _handle_error(self, state: SystemState) -> SystemState:
        """Handle errors and create user-friendly error response."""
        error_message = state.error or "An unknown error occurred"

        # Check if this is a supervisor routing issue (unclear query)
        if state.supervisor_response and not state.supervisor_response.success:
            # This is an unclear query - provide helpful guidance
            state.final_response = f"""🤔 I'm not sure how to help with that query.

I specialize in **IT** and **Finance** support. Here are some examples of what I can help with:

**🔧 IT Support:**
• Password resets and account issues
• Computer and network troubleshooting
• Software installation and updates
• Email configuration problems
• Security-related questions

**💰 Finance Support:**
• Expense report submissions
• Budget and payment processes
• Financial policies and procedures
• Vendor payments and approvals
• Accounting questions

**💡 Try asking something like:**
• "How do I reset my password?"
• "My computer won't start - what should I do?"
• "How do I submit an expense report?"
• "What's the budget approval process?"

Please rephrase your question to be more specific about whether it's an IT or Finance issue."""
        else:
            # Other types of errors
            state.final_response = f"I apologize, but I encountered an error: {error_message}"

        self.logger.error(f"Error handled: {error_message}")
        return state

    def _validation_router(self, state: SystemState) -> str:
        """Router for validation results."""
        if state.validation_result and state.validation_result.is_valid:
            return "valid"
        else:
            return "invalid"

    def _supervisor_router(self, state: SystemState) -> str:
        """Router for supervisor decisions."""
        if state.error:
            return "error"
        elif state.supervisor_response and state.supervisor_response.success:
            routing_decision = state.supervisor_response.routing_decision
            if routing_decision in ["IT", "Finance"]:
                return routing_decision
            else:
                return "error"
        else:
            return "error"

    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process a user query through the multi-agent system."""
        try:
            self.logger.info(f"Processing query: {query}")

            # Create initial state
            initial_state = SystemState(query=query)

            # Run the workflow
            result = await self.workflow.ainvoke(initial_state)

            # Format the response - result is a dictionary-like object from LangGraph
            response = {
                "success": not bool(result.get("error")),
                "message": result.get("final_response", ""),
                "metadata": result.get("metadata", {}),
                "error": result.get("error")
            }

            self.logger.info(f"Query processing completed: {response['success']}")
            return response

        except Exception as e:
            self.logger.error(f"Error in query processing: {str(e)}")
            return {
                "success": False,
                "message": "I'm sorry, but I encountered an unexpected error while processing your query.",
                "metadata": {},
                "error": str(e)
            }

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            "agents": {
                "supervisor": self.config.agents.supervisor.name,
                "it_agent": self.config.agents.it_agent.name,
                "finance_agent": self.config.agents.finance_agent.name
            },
            "tools": self.tool_registry.list_tools(),
            "config": {
                "model": self.config.aws.model,
                "temperature": self.config.aws.temperature,
                "max_tokens": self.config.aws.max_tokens,
                "region": self.config.aws.region
            }
        }
