"""
Common data models used across the multi-agent support system.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel


class ToolResult(BaseModel):
    """Result of a tool execution."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}
