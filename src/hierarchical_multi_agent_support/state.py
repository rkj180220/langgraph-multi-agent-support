"""
State models for the multi-agent support system workflow.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel

from .validation import ValidationResult
from .agents import AgentResponse


class SystemState(BaseModel):
    """State of the multi-agent system."""
    query: str
    validation_result: Optional[ValidationResult] = None
    supervisor_response: Optional[AgentResponse] = None
    specialist_response: Optional[AgentResponse] = None
    final_response: str = ""
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}

    class Config:
        arbitrary_types_allowed = True
