"""
Tools for the multi-agent support system.
"""

import os
import logging
import requests
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod

from .config import Config
from .models import ToolResult
from .rag_search import RAGDocumentSearch


class BaseTool(ABC):
    """Base class for all tools."""

    def __init__(self, config: Config, logger: logging.Logger):
        """Initialize tool with configuration and logger."""
        self.config = config
        self.logger = logger

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool."""
        pass

    def _handle_error(self, error: Exception, operation: str) -> ToolResult:
        """Handle tool errors consistently."""
        error_msg = f"Error in {operation}: {str(error)}"
        self.logger.error(error_msg)
        return ToolResult(success=False, error=error_msg)


class WebSearchTool(BaseTool):
    """Tool for web search functionality."""

    def __init__(self, config: Config, logger: logging.Logger):
        """Initialize web search tool."""
        super().__init__(config, logger)
        self.timeout = config.tools.web_search.timeout
        self.max_results = config.tools.web_search.max_results
        self.enabled = config.tools.web_search.enabled

    async def execute(self, query: str) -> ToolResult:
        """Execute web search."""
        if not self.enabled:
            return ToolResult(
                success=False,
                error="Web search tool is disabled in configuration"
            )

        try:
            self.logger.info(f"Executing web search for query: {query}")

            # Validate input
            if not query or len(query.strip()) == 0:
                return ToolResult(
                    success=False,
                    error="Search query cannot be empty"
                )

            # Try to use real web search first, fallback to mock if unavailable
            search_results = await self._perform_web_search(query)

            self.logger.info(f"Web search completed successfully for query: {query}")
            return ToolResult(
                success=True,
                data=search_results,
                metadata={"query": query, "results_count": len(search_results)}
            )

        except requests.RequestException as e:
            return self._handle_error(e, "web search request")
        except Exception as e:
            return self._handle_error(e, "web search")

    async def _perform_web_search(self, query: str) -> List[Dict[str, str]]:
        """Perform actual web search with fallback to mock."""
        try:
            # Try DuckDuckGo search first (no API key needed)
            return await self._duckduckgo_search(query)
        except Exception as e:
            self.logger.warning(f"Real web search failed: {str(e)}, falling back to mock results")
            return self._simulate_web_search(query)

    async def _duckduckgo_search(self, query: str) -> List[Dict[str, str]]:
        """Perform web search using DuckDuckGo."""
        try:
            # Use requests to query DuckDuckGo's instant answer API
            search_url = "https://api.duckduckgo.com/"
            params = {
                'q': query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }

            response = requests.get(search_url, params=params, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()
            results = []

            # Extract results from DuckDuckGo response
            if data.get('AbstractText'):
                results.append({
                    'title': data.get('Heading', 'DuckDuckGo Result'),
                    'snippet': data.get('AbstractText', ''),
                    'url': data.get('AbstractURL', f'https://duckduckgo.com/?q={query}')
                })

            # Add related topics
            for topic in data.get('RelatedTopics', [])[:3]:
                if isinstance(topic, dict) and 'Text' in topic:
                    results.append({
                        'title': topic.get('Text', '').split(' - ')[0],
                        'snippet': topic.get('Text', ''),
                        'url': topic.get('FirstURL', f'https://duckduckgo.com/?q={query}')
                    })

            # If no results, try a simpler approach
            if not results:
                results = await self._simple_web_search(query)

            return results[:self.max_results]

        except Exception as e:
            self.logger.error(f"DuckDuckGo search failed: {str(e)}")
            raise

    async def _simple_web_search(self, query: str) -> List[Dict[str, str]]:
        """Simple web search approach for common IT/Finance queries."""
        # For demonstration, provide enhanced mock results based on query domain
        if any(keyword in query.lower() for keyword in ['password', 'login', 'vpn', 'network', 'computer']):
            return [
                {
                    'title': f'IT Support: {query} - Official Documentation',
                    'snippet': f'Official documentation and troubleshooting guide for {query}. Step-by-step instructions for resolving common issues.',
                    'url': f'https://docs.example.com/it/{query.replace(" ", "-")}'
                },
                {
                    'title': f'How to fix {query} - Tech Support',
                    'snippet': f'Comprehensive guide to resolving {query} issues. Includes common causes and solutions.',
                    'url': f'https://support.example.com/troubleshooting/{query.replace(" ", "-")}'
                }
            ]
        elif any(keyword in query.lower() for keyword in ['expense', 'budget', 'finance', 'payment', 'invoice']):
            return [
                {
                    'title': f'Finance Policy: {query} - Company Guidelines',
                    'snippet': f'Official company policy and procedures for {query}. Includes approval workflows and requirements.',
                    'url': f'https://finance.example.com/policies/{query.replace(" ", "-")}'
                },
                {
                    'title': f'{query} Best Practices - Finance Department',
                    'snippet': f'Best practices and guidelines for {query} management. Compliance and regulatory information.',
                    'url': f'https://finance.example.com/best-practices/{query.replace(" ", "-")}'
                }
            ]
        else:
            return self._simulate_web_search(query)


class RAGSearchTool(BaseTool):
    """Tool for RAG-based document search using semantic similarity."""

    def __init__(self, config: Config, logger: logging.Logger):
        """Initialize RAG search tool."""
        super().__init__(config, logger)
        self.enabled = config.tools.file_reader.enabled
        self.rag_search = RAGDocumentSearch(config, logger)

    async def execute(self, query: str, domain: str = "finance") -> ToolResult:
        """Execute semantic search through documents."""
        if not self.enabled:
            return ToolResult(
                success=False,
                error="RAG search tool is disabled in configuration"
            )

        try:
            self.logger.info(f"Executing RAG search for query: {query} in domain: {domain}")

            # Support both IT and Finance domains
            if domain not in ["finance", "it"]:
                return ToolResult(
                    success=False,
                    error="RAG search supports 'finance' and 'it' domains only"
                )

            # Get relevant context using RAG
            context = await self.rag_search.get_context_for_query(query, domain)

            if not context:
                return ToolResult(
                    success=False,
                    error="No relevant documents found for the query"
                )

            # Get the actual matching chunks for metadata
            relevant_chunks = await self.rag_search.search_documents(query, domain, top_k=5)

            metadata = {
                "query": query,
                "domain": domain,
                "chunks_found": len(relevant_chunks),
                "sources": list(set([chunk.source for chunk in relevant_chunks])),
                "similarity_scores": [chunk.metadata.get('similarity_score', 0.0) for chunk in relevant_chunks]
            }

            self.logger.info(f"RAG search found {len(relevant_chunks)} relevant chunks")
            return ToolResult(
                success=True,
                data=context,
                metadata=metadata
            )

        except Exception as e:
            return self._handle_error(e, "RAG search")


class ToolRegistry:
    """Registry for managing and executing tools."""

    def __init__(self, config: Config, logger: logging.Logger):
        """Initialize tool registry."""
        self.config = config
        self.logger = logger
        self.tools = {}

        # Register tools
        self._register_tools()

    def _register_tools(self) -> None:
        """Register all available tools."""
        self.tools["web_search"] = WebSearchTool(self.config, self.logger)
        self.tools["rag_search"] = RAGSearchTool(self.config, self.logger)

    async def execute_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool by name."""
        if tool_name not in self.tools:
            return ToolResult(
                success=False,
                error=f"Tool '{tool_name}' not found"
            )

        try:
            return await self.tools[tool_name].execute(**kwargs)
        except Exception as e:
            self.logger.error(f"Error executing tool '{tool_name}': {str(e)}")
            return ToolResult(
                success=False,
                error=f"Tool execution failed: {str(e)}"
            )

    def list_tools(self) -> List[str]:
        """List available tools."""
        return list(self.tools.keys())

    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Get a tool instance by name."""
        return self.tools.get(tool_name)

    def get_tool_info(self, tool_name: str) -> Dict[str, Any]:
        """Get information about a tool."""
        if tool_name not in self.tools:
            return {"error": f"Tool '{tool_name}' not found"}

        tool = self.tools[tool_name]
        return {
            "name": tool_name,
            "class": tool.__class__.__name__,
            "enabled": getattr(tool, 'enabled', True)
        }
