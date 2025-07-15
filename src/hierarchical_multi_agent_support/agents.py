"""
Agents for the multi-agent support system.
"""

import logging
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from pydantic import BaseModel
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from .config import Config
from .models import ToolResult
from .tools import ToolRegistry


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
        self.llm = ChatBedrock(
            model_id=config.aws.model,
            region_name=config.aws.region,
            credentials_profile_name=None,  # Use default credentials
            model_kwargs={
                "temperature": config.aws.temperature,
                "max_tokens": config.aws.max_tokens,
            }
        )

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
        """Call the LLM with error handling."""
        try:
            response = await self.llm.ainvoke(messages)
            return response.content
        except Exception as e:
            self.logger.error(f"LLM call failed for {self.name}: {str(e)}")
            return "I apologize, but I'm experiencing technical difficulties. Please try again later."


class SupervisorAgent(BaseAgent):
    """Supervisor agent that routes queries to appropriate specialists."""

    def __init__(self, config: Config, tool_registry: ToolRegistry, logger: logging.Logger):
        """Initialize supervisor agent."""
        super().__init__("Supervisor", config, tool_registry, logger)
        self.system_prompt = self._create_system_prompt()

    def _create_system_prompt(self) -> str:
        """Create system prompt for the supervisor."""
        return """
        You are a Supervisor Agent in a multi-agent support system. Your role is to analyze user queries and route them to the appropriate specialist agent.
        
        Available specialist agents:
        - IT Agent: Handles technical issues, software problems, hardware troubleshooting, network issues, security concerns, and system administration
        - Finance Agent: Handles financial queries, accounting questions, budget analysis, expense reports, financial calculations, and payment processing
        
        Instructions:
        1. Analyze the user's query carefully
        2. Determine which specialist agent is best suited to handle the query
        3. If the query is ambiguous or could relate to both domains, choose the most likely primary domain
        4. If the query is completely unrelated to IT or Finance, politely explain that you can only help with IT or Finance queries
        5. Respond with exactly one of: "IT", "Finance", or "Unclear"
        
        Respond with only the routing decision (IT/Finance/Unclear) and a brief explanation.
        """

    async def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """Process query and route to appropriate agent."""
        try:
            self.logger.info(f"Supervisor processing query: {query}")

            # Create messages for routing decision
            messages = [
                SystemMessage(content=self.system_prompt),
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

        except Exception as e:
            self.logger.error(f"Error in supervisor processing: {str(e)}")
            return self._create_response(
                success=False,
                message="I'm experiencing technical difficulties. Please try again later."
            )

    def _parse_routing_decision(self, response: str) -> str:
        """Parse the routing decision from LLM response."""
        response_lower = response.lower()

        if "it" in response_lower and "finance" not in response_lower:
            return "IT"
        elif "finance" in response_lower and "it" not in response_lower:
            return "Finance"
        else:
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
        try:
            self.logger.info(f"IT Agent processing query: {query}")

            tool_calls = []
            additional_context = ""

            # Use RAG search to find relevant IT documents
            rag_result = await self.tool_registry.execute_tool("rag_search", query=query, domain="it")
            if rag_result.success:
                additional_context += f"Internal IT Documentation:\n{rag_result.data}\n\n"

                # Add source information to tool calls
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
                # If RAG search fails, note it but continue
                tool_calls.append({
                    "tool": "rag_search",
                    "success": False,
                    "result": f"RAG search failed: {rag_result.error}"
                })

            # Perform web search for additional information
            web_result = await self.tool_registry.execute_tool("web_search", query=query)
            if web_result.success:
                web_info = self._format_web_results(web_result.data)
                additional_context += f"External Resources:\n{web_info}\n\n"
                tool_calls.append({
                    "tool": "web_search",
                    "success": True,
                    "result": f"Found {len(web_result.data)} relevant web results"
                })

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
                    "rag_sources": rag_result.metadata.get('sources', []) if rag_result.success else [],
                    "chunks_processed": rag_result.metadata.get('chunks_found', 0) if rag_result.success else 0
                }
            )

        except Exception as e:
            self.logger.error(f"Error in IT Agent processing: {str(e)}")
            return self._create_response(
                success=False,
                message="I'm experiencing technical difficulties with processing your IT query. Please try again later."
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
        try:
            self.logger.info(f"Finance Agent processing query: {query}")

            tool_calls = []
            additional_context = ""

            # Use RAG search to find relevant finance documents
            rag_result = await self.tool_registry.execute_tool("rag_search", query=query, domain="finance")
            if rag_result.success:
                additional_context += f"Internal Finance Documents:\n{rag_result.data}\n\n"

                # Add source information to tool calls
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
                # If RAG search fails, note it but continue
                tool_calls.append({
                    "tool": "rag_search",
                    "success": False,
                    "result": f"RAG search failed: {rag_result.error}"
                })

            # Perform web search for additional context if needed
            web_result = await self.tool_registry.execute_tool("web_search", query=query)
            if web_result.success:
                web_info = self._format_web_results(web_result.data)
                additional_context += f"External Resources:\n{web_info}\n\n"
                tool_calls.append({
                    "tool": "web_search",
                    "success": True,
                    "result": f"Found {len(web_result.data)} relevant web results"
                })

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
                    "rag_sources": rag_result.metadata.get('sources', []) if rag_result.success else [],
                    "chunks_processed": rag_result.metadata.get('chunks_found', 0) if rag_result.success else 0
                }
            )

        except Exception as e:
            self.logger.error(f"Error in Finance Agent processing: {str(e)}")
            return self._create_response(
                success=False,
                message="I'm experiencing technical difficulties with processing your finance query. Please try again later."
            )

    def _format_web_results(self, results: List[Dict[str, str]]) -> str:
        """Format web search results for LLM context."""
        formatted = ""
        for i, result in enumerate(results, 1):
            formatted += f"{i}. {result['title']}\n   {result['snippet']}\n   URL: {result['url']}\n\n"
        return formatted
