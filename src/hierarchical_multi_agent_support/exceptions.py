"""
Custom exceptions for the multi-agent support system.
Provides specific error types for better error handling and debugging.
"""

from typing import Optional, Dict, Any


class MultiAgentError(Exception):
    """Base exception for all multi-agent system errors."""

    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """Initialize the exception."""
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class ConfigurationError(MultiAgentError):
    """Raised when there's a configuration error."""
    pass


class ValidationError(MultiAgentError):
    """Raised when input validation fails."""
    pass


class AgentError(MultiAgentError):
    """Raised when an agent encounters an error."""

    def __init__(self, message: str, agent_name: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """Initialize the agent error."""
        super().__init__(message, error_code, details)
        self.agent_name = agent_name


class ToolError(MultiAgentError):
    """Raised when a tool encounters an error."""

    def __init__(self, message: str, tool_name: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """Initialize the tool error."""
        super().__init__(message, error_code, details)
        self.tool_name = tool_name


class WorkflowError(MultiAgentError):
    """Raised when the workflow encounters an error."""
    pass


class RAGSearchError(MultiAgentError):
    """Raised when RAG search encounters an error."""
    pass


class LLMError(MultiAgentError):
    """Raised when LLM calls fail."""
    pass


class WebSearchError(ToolError):
    """Raised when web search fails."""

    def __init__(self, message: str, query: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """Initialize the web search error."""
        super().__init__(message, "web_search", error_code, details)
        self.query = query


class DocumentProcessingError(MultiAgentError):
    """Raised when document processing fails."""

    def __init__(self, message: str, document_path: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """Initialize the document processing error."""
        super().__init__(message, error_code, details)
        self.document_path = document_path
