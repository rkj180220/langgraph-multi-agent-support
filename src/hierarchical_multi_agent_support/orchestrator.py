"""
Workflow orchestrator for the multi-agent support system.
Separated from the main system class for better modularity.
"""

import logging
from typing import Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from .agents import SupervisorAgent, ITAgent, FinanceAgent, AgentResponse
from .state import SystemState
from .validation import InputValidator


class WorkflowOrchestrator:
    """Handles the LangGraph workflow orchestration."""

    def __init__(self, supervisor: SupervisorAgent, it_agent: ITAgent, finance_agent: FinanceAgent,
                 validator: InputValidator, logger: logging.Logger):
        """Initialize the workflow orchestrator."""
        self.supervisor = supervisor
        self.it_agent = it_agent
        self.finance_agent = finance_agent
        self.validator = validator
        self.logger = logger

        # Build workflow
        self.workflow = self._build_workflow()

    def _build_workflow(self):
        """Build the LangGraph workflow."""
        workflow = StateGraph(SystemState)

        # Add nodes
        workflow.add_node("validate_input", self._validate_input)
        workflow.add_node("supervisor_route", self._supervisor_route)
        workflow.add_node("it_specialist", self._it_specialist)
        workflow.add_node("finance_specialist", self._finance_specialist)
        workflow.add_node("both_specialists", self._both_specialists)  # New node for handling both domains
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
                "Both": "both_specialists",  # New routing for both domains
                "error": "handle_error"
            }
        )

        # From specialists to response formatting
        workflow.add_edge("it_specialist", "format_response")
        workflow.add_edge("finance_specialist", "format_response")
        workflow.add_edge("both_specialists", "format_response")  # Both specialists also go to formatting

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

    async def _both_specialists(self, state: SystemState) -> SystemState:
        """Process query with both IT and Finance specialists."""
        try:
            self.logger.info("Processing query with both IT and Finance specialists")

            # Process with both agents concurrently
            it_response = await self.it_agent.process_query(state.query)
            finance_response = await self.finance_agent.process_query(state.query)

            # Store individual responses for evaluator
            state.individual_responses = [it_response, finance_response]

            # Combine responses from both specialists
            combined_success = it_response.success and finance_response.success

            # Create a comprehensive response combining both domains
            if combined_success:
                combined_message = f"""## 🔧 IT Support Response:
{it_response.message}

---

## 💰 Finance Support Response:
{finance_response.message}

---

**Note:** This response covers both IT and Finance aspects of your query. If you need more specific help with either domain, please feel free to ask focused questions."""
            else:
                # Handle partial failures
                combined_message = "I encountered some issues processing your multi-domain query:\n\n"

                if it_response.success:
                    combined_message += f"## 🔧 IT Support Response:\n{it_response.message}\n\n"
                else:
                    combined_message += f"## ❌ IT Support Issue:\n{it_response.message}\n\n"

                if finance_response.success:
                    combined_message += f"## 💰 Finance Support Response:\n{finance_response.message}\n\n"
                else:
                    combined_message += f"## ❌ Finance Support Issue:\n{finance_response.message}\n\n"

            # Create a combined agent response
            combined_tool_calls = it_response.tool_calls + finance_response.tool_calls

            state.specialist_response = AgentResponse(
                success=combined_success,
                message=combined_message,
                agent_name="IT & Finance Agents",
                tool_calls=combined_tool_calls,
                metadata={
                    "domains": ["IT", "Finance"],
                    "it_success": it_response.success,
                    "finance_success": finance_response.success,
                    "total_tools_used": len(combined_tool_calls)
                }
            )

            if not combined_success:
                state.error = "Partial success in multi-domain processing"

            self.logger.info(f"Both specialists processing completed - IT: {it_response.success}, Finance: {finance_response.success}")
            return state

        except Exception as e:
            self.logger.error(f"Error in processing with both specialists: {str(e)}")
            state.error = "Failed to process query with both specialists"
            return state

    async def _format_response(self, state: SystemState) -> SystemState:
        """Format the final response using the supervisor agent as evaluator."""
        try:
            if state.specialist_response and state.specialist_response.success:
                # Use supervisor to evaluate and refine the response
                routing_decision = state.supervisor_response.routing_decision if state.supervisor_response else "Unknown"

                # For multi-domain queries, use individual responses if available
                if hasattr(state, 'individual_responses') and state.individual_responses:
                    specialist_responses = state.individual_responses
                else:
                    # Single specialist response
                    specialist_responses = [state.specialist_response]

                # Evaluate and refine the response using supervisor
                evaluated_response = await self.supervisor.evaluate_response(
                    original_query=state.query,
                    specialist_responses=specialist_responses,
                    routing_decision=routing_decision
                )

                # Build the processing path - now simpler since supervisor handles both routing and evaluation
                processing_path = ["Supervisor Agent (Routing)"]
                if routing_decision == "Finance":
                    processing_path.append("Finance Agent")
                elif routing_decision == "IT":
                    processing_path.append("IT Agent")
                elif routing_decision == "Both":
                    processing_path.extend(["IT Agent", "Finance Agent"])
                processing_path.append("Supervisor Agent (Evaluation)")

                # Update the final response with evaluated content
                state.final_response = evaluated_response.message

                # Add comprehensive metadata with processing path
                state.metadata = {
                    "processing_path": processing_path,
                    "routing_decision": routing_decision,
                    "specialist_agents": evaluated_response.metadata.get("original_specialists", []),
                    "tools_used": len(evaluated_response.tool_calls),
                    "evaluated": evaluated_response.metadata.get("evaluated", False),
                    "evaluation_success": evaluated_response.metadata.get("evaluation_success", False),
                    "total_processing_steps": len(processing_path)
                }

                self.logger.info(f"Response evaluated and formatted successfully by supervisor")
            else:
                # Handle case where specialist response failed
                state.final_response = state.specialist_response.message if state.specialist_response else "No response generated"

                # Build processing path for failed queries
                processing_path = ["Supervisor Agent (Routing)"]
                if state.supervisor_response and state.supervisor_response.routing_decision:
                    routing_decision = state.supervisor_response.routing_decision
                    if routing_decision == "Finance":
                        processing_path.append("Finance Agent")
                    elif routing_decision == "IT":
                        processing_path.append("IT Agent")
                    elif routing_decision == "Both":
                        processing_path.extend(["IT Agent", "Finance Agent"])
                processing_path.append("Error Handler")

                state.metadata = {
                    "processing_path": processing_path,
                    "routing_decision": "Unknown",
                    "specialist_agents": [],
                    "tools_used": 0,
                    "evaluated": False,
                    "evaluation_success": False,
                    "total_processing_steps": len(processing_path),
                    "error": state.error
                }

            return state

        except Exception as e:
            self.logger.error(f"Error in response formatting: {str(e)}")
            state.error = "Failed to format response"
            return state

    async def _handle_error(self, state: SystemState) -> SystemState:
        """Handle errors in the workflow."""
        try:
            error_message = state.error or "An unexpected error occurred"
            state.final_response = f"I apologize, but I encountered an issue: {error_message}. Please try again or contact support if the problem persists."
            state.metadata = {
                "agent_used": "Error Handler",
                "tools_used": 0,
                "routing_decision": "Error",
                "evaluated": False,
                "evaluation_success": False,
                "error": error_message
            }

            self.logger.error(f"Workflow error handled: {error_message}")
            return state

        except Exception as e:
            self.logger.error(f"Error in error handling: {str(e)}")
            state.final_response = "I apologize, but I'm experiencing technical difficulties. Please try again later."
            state.metadata = {"error": "Critical error in error handling"}
            return state

    def _validation_router(self, state: SystemState) -> str:
        """Route based on validation result."""
        if state.validation_result and state.validation_result.is_valid:
            return "valid"
        else:
            return "invalid"

    def _supervisor_router(self, state: SystemState) -> str:
        """Route based on supervisor decision."""
        if state.supervisor_response and state.supervisor_response.success:
            routing_decision = state.supervisor_response.routing_decision
            if routing_decision in ["IT", "Finance", "Both"]:
                return routing_decision
            else:
                return "error"
        else:
            return "error"

    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process a query through the workflow."""
        try:
            # Create initial state
            initial_state = SystemState(
                query=query,
                validation_result=None,
                supervisor_response=None,
                specialist_response=None,
                final_response="",
                metadata={},
                error=None
            )

            # Run workflow
            result = await self.workflow.ainvoke(initial_state)

            # Extract final response - handle both dict and object results
            if isinstance(result, dict):
                final_response = result.get("final_response", "")
                metadata = result.get("metadata", {})
                error = result.get("error")
            else:
                final_response = getattr(result, "final_response", "")
                metadata = getattr(result, "metadata", {})
                error = getattr(result, "error", None)

            return {
                "query": query,
                "response": final_response,
                "metadata": metadata,
                "success": not bool(error),
                "error": error
            }

        except Exception as e:
            self.logger.error(f"Error in workflow processing: {str(e)}")
            return {
                "query": query,
                "response": "I apologize, but I'm experiencing technical difficulties. Please try again later.",
                "metadata": {"error": "Workflow processing failed"},
                "success": False,
                "error": str(e)
            }
