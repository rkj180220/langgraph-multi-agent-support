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

            # Simulate web search (in a real implementation, you'd use a search API)
            # For demonstration, we'll return mock results
            search_results = self._simulate_web_search(query)

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

    def _simulate_web_search(self, query: str) -> List[Dict[str, str]]:
        """Simulate web search results."""
        # In a real implementation, this would use a search API like Google Custom Search
        mock_results = [
            {
                "title": f"Search result for '{query}' - Documentation",
                "url": f"https://docs.example.com/search?q={query}",
                "snippet": f"This is a mock search result for the query '{query}'. It provides relevant information about the topic."
            },
            {
                "title": f"Best practices for {query}",
                "url": f"https://bestpractices.example.com/{query}",
                "snippet": f"Learn about best practices and solutions related to {query}."
            },
            {
                "title": f"Common issues with {query}",
                "url": f"https://support.example.com/issues/{query}",
                "snippet": f"Troubleshooting guide for common issues related to {query}."
            }
        ]

        return mock_results[:self.max_results]


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
