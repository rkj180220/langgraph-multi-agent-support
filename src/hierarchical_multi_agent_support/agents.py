"""
Agents for the multi-agent support system.
"""

import logging
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from pydantic import BaseModel
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage

from .config import Config
from .tools import ToolRegistry
from .exceptions import AgentError, LLMError


class AgentResponse(BaseModel):
    """Response from an agent."""
    success: bool
    message: str
    agent_name: str
    routing_decision: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}


class BaseAgent(ABC):
    """Base class for all agents."""

    def __init__(self, name: str, config: Config, tool_registry: ToolRegistry, logger: logging.Logger):
        """Initialize base agent."""
        self.name = name
        self.config = config
        self.tool_registry = tool_registry
        self.logger = logger

        # Initialize LLM with AWS Bedrock
        try:
            self.llm = ChatBedrock(
                model_id=config.aws.model,
                region_name=config.aws.region,
                model_kwargs={
                    "temperature": config.aws.temperature,
                    "max_tokens": config.aws.max_tokens,
                }
            )
        except Exception as e:
            raise AgentError(f"Failed to initialize LLM for {name}", self.name, "LLM_INIT_ERROR", {"original_error": str(e)})

    @abstractmethod
    async def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """Process a user query."""
        pass

    def _create_response(self, success: bool, message: str, **kwargs) -> AgentResponse:
        """Create a standardized agent response."""
        return AgentResponse(
            success=success,
            message=message,
            agent_name=self.name,
            **kwargs
        )

    async def _call_llm(self, messages: List[Any]) -> str:
        """Call the LLM with improved error handling."""
        try:
            response = await self.llm.ainvoke(messages)
            return response.content
        except Exception as e:
            self.logger.error(f"LLM call failed for {self.name}: {str(e)}")
            raise LLMError(f"LLM call failed for {self.name}", "LLM_CALL_ERROR", {"original_error": str(e)})

    def _handle_tool_error(self, tool_name: str, error: Exception) -> Dict[str, Any]:
        """Handle tool errors consistently."""
        self.logger.error(f"Tool {tool_name} failed: {str(error)}")
        return {
            "tool": tool_name,
            "success": False,
            "result": f"Tool execution failed: {str(error)}"
        }


class SupervisorAgent(BaseAgent):
    """Supervisor agent that routes queries to appropriate specialists and evaluates responses."""

    def __init__(self, config: Config, tool_registry: ToolRegistry, logger: logging.Logger):
        """Initialize supervisor agent."""
        super().__init__("Supervisor", config, tool_registry, logger)
        self.routing_prompt = self._create_routing_prompt()
        self.evaluation_prompt = self._create_evaluation_prompt()

    def _create_routing_prompt(self) -> str:
        """Create system prompt for routing decisions."""
        return """
        You are a Supervisor Agent in a multi-agent support system. Your role is to analyze user queries and route them to the appropriate specialist agent(s).
        
        Available specialist agents:
        - IT Agent: Handles technical issues, software problems, hardware troubleshooting, network issues, security concerns, and system administration
        - Finance Agent: Handles financial queries, accounting questions, budget analysis, expense reports, financial calculations, and payment processing
        
        CRITICAL ROUTING RULES:
        1. Analyze the user's query carefully
        2. Choose ONLY ONE domain unless the query explicitly needs BOTH
        3. "Both" should be used RARELY - only when the query contains distinct IT AND Finance elements
        4. If the query is completely unrelated to IT or Finance, respond with "Unclear"
        5. Respond with EXACTLY one of: "IT", "Finance", "Both", or "Unclear"
        
        Finance-only Examples:
        - "How do I submit an expense report?" → "Finance"
        - "What's my expense limit?" → "Finance"
        - "Budget approval process?" → "Finance"
        - "Payroll questions?" → "Finance"
        - "How do I file a reimbursement?" → "Finance"
        - "Travel policy questions?" → "Finance"
        
        IT-only Examples:
        - "How do I reset my password?" → "IT"
        - "My computer won't start" → "IT"
        - "How to install software?" → "IT"
        - "Network connection issues?" → "IT"
        - "VPN setup problems?" → "IT"
        
        Both Examples (RARE):
        - "I need help with my salary AND installing new software" → "Both"
        - "Issues with both my payroll and VPN setup" → "Both"
        - "My computer broke and I need to submit an expense report for a new one" → "Both"
        
        Unclear Examples:
        - "What's the weather today?" → "Unclear"
        - "How do I get to the office?" → "Unclear"
        
        IMPORTANT: Start your response with EXACTLY one word: IT, Finance, Both, or Unclear. Then provide a brief explanation.
        """

    def _create_evaluation_prompt(self) -> str:
        """Create system prompt for response evaluation."""
        return """
        You are a Supervisor Agent responsible for evaluating and refining responses from specialist agents before delivery to users.

        Your role is to:
        1. Evaluate if the response directly addresses the user's original question
        2. Combine multiple specialist responses intelligently when needed
        3. Remove irrelevant information and focus on what was specifically asked
        4. Ensure the response is clear, concise, and actionable
        5. Maintain the helpful and professional tone

        Guidelines:
        - If the response is from a single specialist, ensure it's relevant and well-structured
        - If combining multiple specialist responses, create a cohesive answer that flows naturally
        - Remove any redundant or off-topic information
        - Keep technical details when they're relevant to the question
        - Provide a direct answer to what was asked, not generic information
        - If the specialists provided partial answers, acknowledge limitations clearly

        Always maintain accuracy while improving clarity and relevance.
        """

    async def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """Process query and route to appropriate agent."""
        try:
            self.logger.info(f"Supervisor processing query: {query}")

            # Create messages for routing decision
            messages = [
                SystemMessage(content=self.routing_prompt),
                HumanMessage(content=f"Route this query: {query}")
            ]

            # Get routing decision from LLM
            response = await self._call_llm(messages)

            # Parse routing decision
            routing_decision = self._parse_routing_decision(response)

            if routing_decision == "Unclear":
                return self._create_response(
                    success=False,
                    message="I can only help with IT or Finance related queries. Please rephrase your question to be more specific about the domain.",
                    routing_decision=routing_decision
                )

            self.logger.info(f"Query routed to: {routing_decision}")
            return self._create_response(
                success=True,
                message=f"Query routed to {routing_decision} specialist",
                routing_decision=routing_decision,
                metadata={"original_query": query}
            )

        except LLMError as e:
            self.logger.error(f"LLM error in supervisor: {str(e)}")
            return self._create_response(
                success=False,
                message="I'm experiencing technical difficulties with the routing system. Please try again later.",
                metadata={"error_type": "LLM_ERROR", "error_code": e.error_code}
            )
        except Exception as e:
            self.logger.error(f"Unexpected error in supervisor processing: {str(e)}")
            return self._create_response(
                success=False,
                message="I'm experiencing technical difficulties. Please try again later.",
                metadata={"error_type": "UNEXPECTED_ERROR"}
            )

    async def evaluate_response(self, original_query: str, specialist_responses: List[AgentResponse], routing_decision: str) -> AgentResponse:
        """Evaluate and refine the specialist responses."""
        try:
            self.logger.info(f"Supervisor evaluating {len(specialist_responses)} responses for query: {original_query}")

            # Prepare context for evaluation
            if len(specialist_responses) == 1:
                # Single specialist response
                specialist_response = specialist_responses[0]
                context = f"""
Original User Query: {original_query}
Routing Decision: {routing_decision}

Specialist Response from {specialist_response.agent_name}:
{specialist_response.message}

Tools Used: {len(specialist_response.tool_calls)}
Response Success: {specialist_response.success}
"""
            else:
                # Multiple specialist responses
                context = f"""
Original User Query: {original_query}
Routing Decision: {routing_decision}

"""
                for i, response in enumerate(specialist_responses, 1):
                    context += f"""
Specialist Response {i} from {response.agent_name}:
{response.message}

Tools Used: {len(response.tool_calls)}
Response Success: {response.success}
---
"""

            # Create evaluation prompt
            messages = [
                SystemMessage(content=self.evaluation_prompt),
                HumanMessage(content=f"""{context}

Please evaluate the above specialist response(s) and provide a refined, focused answer that directly addresses the user's original query: "{original_query}"

Requirements:
- Address only what the user specifically asked about
- Remove any irrelevant information
- If multiple responses are provided, combine them intelligently
- Keep the response concise but complete
- Maintain a helpful, professional tone
- If information is missing or unclear, acknowledge it appropriately""")
            ]

            # Get refined response from evaluator
            refined_response = await self._call_llm(messages)

            # Combine tool calls from all specialists
            all_tool_calls = []
            for response in specialist_responses:
                all_tool_calls.extend(response.tool_calls)

            # Create comprehensive metadata
            metadata = {
                "evaluated": True,
                "original_specialists": [r.agent_name for r in specialist_responses],
                "routing_decision": routing_decision,
                "total_tools_used": len(all_tool_calls),
                "evaluation_success": True
            }

            self.logger.info(f"Supervisor completed response evaluation")
            return self._create_response(
                success=True,
                message=refined_response,
                tool_calls=all_tool_calls,
                metadata=metadata
            )

        except LLMError as e:
            self.logger.error(f"LLM error in supervisor evaluation: {str(e)}")
            # Fallback to original response if evaluation fails
            return self._fallback_to_original(specialist_responses, routing_decision, e)
        except Exception as e:
            self.logger.error(f"Unexpected error in supervisor evaluation: {str(e)}")
            return self._fallback_to_original(specialist_responses, routing_decision, e)

    def _fallback_to_original(self, specialist_responses: List[AgentResponse], routing_decision: str, error: Exception) -> AgentResponse:
        """Fallback to original response if evaluation fails."""
        if len(specialist_responses) == 1:
            response = specialist_responses[0]
            response.metadata["evaluation_failed"] = True
            response.metadata["evaluation_error"] = str(error)
            return response
        else:
            # Simple combination fallback
            combined_message = "\n\n".join([f"**{r.agent_name}:** {r.message}" for r in specialist_responses])
            all_tool_calls = []
            for response in specialist_responses:
                all_tool_calls.extend(response.tool_calls)

            return self._create_response(
                success=True,
                message=combined_message,
                tool_calls=all_tool_calls,
                metadata={
                    "evaluated": False,
                    "evaluation_failed": True,
                    "evaluation_error": str(error),
                    "routing_decision": routing_decision
                }
            )

    def _parse_routing_decision(self, response: str) -> str:
        """Parse the routing decision from LLM response with improved precision."""
        # First, try to extract the exact word at the beginning of the response
        response_words = response.strip().split()
        if response_words:
            first_word = response_words[0].lower()

            # Check for exact matches first
            if first_word in ["finance", "it", "both", "unclear"]:
                return first_word.title() if first_word != "it" else "IT"

        # Fallback to pattern matching but be more precise
        response_lower = response.lower()

        # Look for standalone words using word boundaries
        import re

        # Check for Finance first (since it's more specific)
        if re.search(r'\bfinance\b', response_lower):
            return "Finance"

        # Check for IT as a standalone word
        if re.search(r'\bit\b', response_lower):
            return "IT"

        # Check for Both
        if re.search(r'\bboth\b', response_lower):
            return "Both"

        # Check for Unclear
        if re.search(r'\bunclear\b', response_lower):
            return "Unclear"

        # Default fallback
        self.logger.warning(f"Could not parse routing decision from: {response}")
        return "Unclear"


class ITAgent(BaseAgent):
    """IT specialist agent with RAG-enhanced document search."""

    def __init__(self, config: Config, tool_registry: ToolRegistry, logger: logging.Logger):
        """Initialize IT agent."""
        super().__init__("IT Agent", config, tool_registry, logger)
        self.system_prompt = self._create_system_prompt()

    def _create_system_prompt(self) -> str:
        """Create system prompt for IT agent."""
        return """
        You are an IT Support Agent specializing in technical issues, software problems, hardware troubleshooting, network issues, security concerns, and system administration.
        
        Available tools:
        - rag_search: Search through internal IT documentation using semantic similarity
        - web_search: Search the web for additional information
        
        Instructions:
        1. Analyze the user's IT-related query
        2. Use rag_search to find relevant information from internal IT documents (troubleshooting guides, policies, etc.)
        3. If needed, supplement with web search for additional information
        4. Provide clear, actionable solutions based on the retrieved documents
        5. Include step-by-step instructions when appropriate
        6. Always be helpful and professional
        7. Reference specific documents when citing policies or procedures
        8. For ServiceNow-related queries, provide the exact URL and search terms
        
        Focus on providing practical, technical solutions for IT problems based on internal documentation.
        """

    async def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """Process IT-related query using RAG search."""
        tool_calls = []  # Initialize tool_calls at the start

        try:
            self.logger.info(f"IT Agent processing query: {query}")

            additional_context = ""

            # Use RAG search to find relevant IT documents
            try:
                rag_result = await self.tool_registry.execute_tool("rag_search", query=query, domain="it")
                if rag_result.success:
                    additional_context += f"Internal IT Documentation:\n{rag_result.data}\n\n"
                    sources = rag_result.metadata.get('sources', [])
                    chunks_found = rag_result.metadata.get('chunks_found', 0)

                    tool_calls.append({
                        "tool": "rag_search",
                        "success": True,
                        "result": f"Found {chunks_found} relevant sections from {len(sources)} documents",
                        "sources": sources,
                        "similarity_scores": rag_result.metadata.get('similarity_scores', [])
                    })
                else:
                    tool_calls.append({
                        "tool": "rag_search",
                        "success": False,
                        "result": f"RAG search failed: {rag_result.error}"
                    })
            except Exception as e:
                tool_calls.append(self._handle_tool_error("rag_search", e))

            # Perform web search for additional information
            try:
                web_result = await self.tool_registry.execute_tool("web_search", query=query)
                if web_result.success:
                    web_info = self._format_web_results(web_result.data)
                    additional_context += f"External Resources:\n{web_info}\n\n"
                    tool_calls.append({
                        "tool": "web_search",
                        "success": True,
                        "result": f"Found {len(web_result.data)} relevant web results"
                    })
                else:
                    tool_calls.append({
                        "tool": "web_search",
                        "success": False,
                        "result": f"Web search failed: {web_result.error}"
                    })
            except Exception as e:
                tool_calls.append(self._handle_tool_error("web_search", e))

            # Generate response using LLM with enhanced context
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=f"""User Query: {query}

Relevant Context from Internal Documents:
{additional_context}

Please provide a comprehensive IT support response based on the retrieved documents. 
If you reference specific procedures or policies, mention the source document name.
For ServiceNow requests, provide the exact URL and search instructions.""")
            ]

            response = await self._call_llm(messages)

            self.logger.info(f"IT Agent completed processing query")
            return self._create_response(
                success=True,
                message=response,
                tool_calls=tool_calls,
                metadata={
                    "domain": "IT",
                    "tools_used": len(tool_calls),
                    "successful_tools": len([call for call in tool_calls if call.get("success", False)]),
                    "context_length": len(additional_context)
                }
            )

        except LLMError as e:
            self.logger.error(f"LLM error in IT agent: {str(e)}")
            return self._create_response(
                success=False,
                message="I'm experiencing technical difficulties with generating the response. Please try again later.",
                tool_calls=tool_calls,
                metadata={"error_type": "LLM_ERROR", "error_code": e.error_code}
            )
        except Exception as e:
            self.logger.error(f"Unexpected error in IT Agent processing: {str(e)}")
            return self._create_response(
                success=False,
                message="I'm experiencing technical difficulties with processing your IT query. Please try again later.",
                tool_calls=tool_calls,
                metadata={"error_type": "UNEXPECTED_ERROR"}
            )

    def _format_web_results(self, results: List[Dict[str, str]]) -> str:
        """Format web search results for LLM context."""
        formatted = ""
        for i, result in enumerate(results, 1):
            formatted += f"{i}. {result['title']}\n   {result['snippet']}\n   URL: {result['url']}\n\n"
        return formatted


class FinanceAgent(BaseAgent):
    """Finance specialist agent with RAG-enhanced document search."""

    def __init__(self, config: Config, tool_registry: ToolRegistry, logger: logging.Logger):
        """Initialize Finance agent."""
        super().__init__("Finance Agent", config, tool_registry, logger)
        self.system_prompt = self._create_system_prompt()

    def _create_system_prompt(self) -> str:
        """Create system prompt for Finance agent."""
        return """
        You are a Finance Support Agent specializing in financial queries, accounting questions, budget analysis, expense reports, financial calculations, and payment processing.
        
        Available tools:
        - rag_search: Search through internal finance PDF documents using semantic similarity
        - web_search: Search the web for additional information
        
        Instructions:
        1. Analyze the user's finance-related query
        2. Use rag_search to find relevant information from internal finance PDFs (Employee Expense Guidelines, Travel Policies, etc.)
        3. If needed, supplement with web search for additional information
        4. Provide clear, accurate financial guidance based on the retrieved documents
        5. Include specific policy references and page numbers when available
        6. Always be helpful and professional
        7. When dealing with financial calculations, show your work
        8. Cite the specific documents you're referencing
        
        Focus on providing accurate, policy-compliant financial support and guidance.
        """

    async def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """Process Finance-related query using RAG search."""
        tool_calls = []  # Initialize tool_calls at the start

        try:
            self.logger.info(f"Finance Agent processing query: {query}")

            additional_context = ""

            # Use RAG search to find relevant finance documents
            try:
                rag_result = await self.tool_registry.execute_tool("rag_search", query=query, domain="finance")
                if rag_result.success:
                    additional_context += f"Internal Finance Documents:\n{rag_result.data}\n\n"
                    sources = rag_result.metadata.get('sources', [])
                    chunks_found = rag_result.metadata.get('chunks_found', 0)

                    tool_calls.append({
                        "tool": "rag_search",
                        "success": True,
                        "result": f"Found {chunks_found} relevant sections from {len(sources)} documents",
                        "sources": sources,
                        "similarity_scores": rag_result.metadata.get('similarity_scores', [])
                    })
                else:
                    tool_calls.append({
                        "tool": "rag_search",
                        "success": False,
                        "result": f"RAG search failed: {rag_result.error}"
                    })
            except Exception as e:
                tool_calls.append(self._handle_tool_error("rag_search", e))

            # Perform web search for additional context if needed
            try:
                web_result = await self.tool_registry.execute_tool("web_search", query=query)
                if web_result.success:
                    web_info = self._format_web_results(web_result.data)
                    additional_context += f"External Resources:\n{web_info}\n\n"
                    tool_calls.append({
                        "tool": "web_search",
                        "success": True,
                        "result": f"Found {len(web_result.data)} relevant web results"
                    })
                else:
                    tool_calls.append({
                        "tool": "web_search",
                        "success": False,
                        "result": f"Web search failed: {web_result.error}"
                    })
            except Exception as e:
                tool_calls.append(self._handle_tool_error("web_search", e))

            # Generate response using LLM with enhanced context
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=f"""User Query: {query}

Relevant Context from Internal Documents:
{additional_context}

Please provide a comprehensive finance support response based on the retrieved documents. 
If you reference specific policies or procedures, mention the source document name.
Be specific about requirements, deadlines, and approval processes.""")
            ]

            response = await self._call_llm(messages)

            self.logger.info(f"Finance Agent completed processing query")
            return self._create_response(
                success=True,
                message=response,
                tool_calls=tool_calls,
                metadata={
                    "domain": "Finance",
                    "tools_used": len(tool_calls),
                    "successful_tools": len([call for call in tool_calls if call.get("success", False)]),
                    "context_length": len(additional_context)
                }
            )

        except LLMError as e:
            self.logger.error(f"LLM error in Finance agent: {str(e)}")
            return self._create_response(
                success=False,
                message="I'm experiencing technical difficulties with generating the response. Please try again later.",
                tool_calls=tool_calls,
                metadata={"error_type": "LLM_ERROR", "error_code": e.error_code}
            )
        except Exception as e:
            self.logger.error(f"Unexpected error in Finance Agent processing: {str(e)}")
            return self._create_response(
                success=False,
                message="I'm experiencing technical difficulties with processing your finance query. Please try again later.",
                tool_calls=tool_calls,
                metadata={"error_type": "UNEXPECTED_ERROR"}
            )

    def _format_web_results(self, results: List[Dict[str, str]]) -> str:
        """Format web search results for LLM context."""
        formatted = ""
        for i, result in enumerate(results, 1):
            formatted += f"{i}. {result['title']}\n   {result['snippet']}\n   URL: {result['url']}\n\n"
        return formatted


class EvaluatorAgent(BaseAgent):
    """Final evaluator agent that assesses and refines responses before delivery."""

    def __init__(self, config: Config, tool_registry: ToolRegistry, logger: logging.Logger):
        """Initialize evaluator agent."""
        super().__init__("Evaluator Agent", config, tool_registry, logger)
        self.system_prompt = self._create_system_prompt()

    def _create_system_prompt(self) -> str:
        """Create system prompt for the evaluator."""
        return """
        You are an Evaluator Agent responsible for assessing and refining responses from specialist agents before delivery to users.

        Your role is to:
        1. Evaluate if the response directly addresses the user's original question
        2. Combine multiple specialist responses intelligently when needed
        3. Remove irrelevant information and focus on what was specifically asked
        4. Ensure the response is clear, concise, and actionable
        5. Maintain the helpful and professional tone

        Guidelines:
        - If the response is from a single specialist, ensure it's relevant and well-structured
        - If combining multiple specialist responses, create a cohesive answer that flows naturally
        - Remove any redundant or off-topic information
        - Keep technical details when they're relevant to the question
        - Provide a direct answer to what was asked, not generic information
        - If the specialists provided partial answers, acknowledge limitations clearly

        Always maintain accuracy while improving clarity and relevance.
        """

    async def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """This agent does not process queries directly. Use evaluate_response instead."""
        self.logger.warning("process_query called on EvaluatorAgent, which is not its intended use.")
        return self._create_response(
            success=False,
            message="EvaluatorAgent does not process queries directly.",
            metadata={"error_type": "NOT_SUPPORTED"}
        )

    async def evaluate_response(self, original_query: str, specialist_responses: List[AgentResponse], routing_decision: str) -> AgentResponse:
        """Evaluate and refine the specialist responses."""
        try:
            self.logger.info(f"Evaluator processing {len(specialist_responses)} responses for query: {original_query}")

            # Prepare context for evaluation
            if len(specialist_responses) == 1:
                # Single specialist response
                specialist_response = specialist_responses[0]
                context = f"""
Original User Query: {original_query}
Routing Decision: {routing_decision}

Specialist Response from {specialist_response.agent_name}:
{specialist_response.message}

Tools Used: {len(specialist_response.tool_calls)}
Response Success: {specialist_response.success}
"""
            else:
                # Multiple specialist responses
                context = f"""
Original User Query: {original_query}
Routing Decision: {routing_decision}

"""
                for i, response in enumerate(specialist_responses, 1):
                    context += f"""
Specialist Response {i} from {response.agent_name}:
{response.message}

Tools Used: {len(response.tool_calls)}
Response Success: {response.success}
---
"""

            # Create evaluation prompt
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=f"""{context}

Please evaluate the above specialist response(s) and provide a refined, focused answer that directly addresses the user's original query: "{original_query}"

Requirements:
- Address only what the user specifically asked about
- Remove any irrelevant information
- If multiple responses are provided, combine them intelligently
- Keep the response concise but complete
- Maintain a helpful, professional tone
- If information is missing or unclear, acknowledge it appropriately""")
            ]

            # Get refined response from evaluator
            refined_response = await self._call_llm(messages)

            # Combine tool calls from all specialists
            all_tool_calls = []
            for response in specialist_responses:
                all_tool_calls.extend(response.tool_calls)

            # Create comprehensive metadata
            metadata = {
                "evaluated": True,
                "original_specialists": [r.agent_name for r in specialist_responses],
                "routing_decision": routing_decision,
                "total_tools_used": len(all_tool_calls),
                "evaluation_success": True
            }

            self.logger.info(f"Evaluator completed response refinement")
            return self._create_response(
                success=True,
                message=refined_response,
                tool_calls=all_tool_calls,
                metadata=metadata
            )

        except LLMError as e:
            self.logger.error(f"LLM error in evaluator: {str(e)}")
            # Fallback to original response if evaluation fails
            return self._fallback_to_original(specialist_responses, routing_decision, e)
        except Exception as e:
            self.logger.error(f"Unexpected error in evaluator: {str(e)}")
            return self._fallback_to_original(specialist_responses, routing_decision, e)

    def _fallback_to_original(self, specialist_responses: List[AgentResponse], routing_decision: str, error: Exception) -> AgentResponse:
        """Fallback to original response if evaluation fails."""
        if len(specialist_responses) == 1:
            response = specialist_responses[0]
            response.metadata["evaluation_failed"] = True
            response.metadata["evaluation_error"] = str(error)
            return response
        else:
            # Simple combination fallback
            combined_message = "\n\n".join([f"**{r.agent_name}:** {r.message}" for r in specialist_responses])
            all_tool_calls = []
            for response in specialist_responses:
                all_tool_calls.extend(response.tool_calls)

            return self._create_response(
                success=True,
                message=combined_message,
                tool_calls=all_tool_calls,
                metadata={
                    "evaluated": False,
                    "evaluation_failed": True,
                    "evaluation_error": str(error),
                    "routing_decision": routing_decision
                }
            )
