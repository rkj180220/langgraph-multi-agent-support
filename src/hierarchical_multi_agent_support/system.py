"""
Main multi-agent support system orchestrator using LangGraph.
Refactored for better modularity and separation of concerns.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from .config import Config, ConfigManager
from .agents import SupervisorAgent, ITAgent, FinanceAgent
from .tools import ToolRegistry
from .validation import InputValidator
from .orchestrator import WorkflowOrchestrator
from .logging_manager import LoggingManager
from .state import SystemState


class MultiAgentSupportSystem:
    """Main entry point for the multi-agent support system."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the multi-agent support system."""
        # Load and validate configuration
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.load_config()
        self.config_manager.validate_config(self.config)

        # Setup logging
        self.logging_manager = LoggingManager(self.config)
        self.logger = self.logging_manager.get_logger("MultiAgentSupport")

        # Initialize core components
        self._initialize_components()

        # Create workflow orchestrator
        self.orchestrator = WorkflowOrchestrator(
            supervisor=self.supervisor,
            it_agent=self.it_agent,
            finance_agent=self.finance_agent,
            validator=self.validator,
            logger=self.logger
        )

        self.logger.info("Multi-agent support system initialized successfully")

    def _initialize_components(self) -> None:
        """Initialize all system components."""
        # Create shared logger for components
        component_logger = self.logging_manager.get_logger("Components")

        # Initialize tools and validation
        self.tool_registry = ToolRegistry(self.config, component_logger)
        self.validator = InputValidator(self.config, component_logger)

        # Initialize agents
        self.supervisor = SupervisorAgent(self.config, self.tool_registry, component_logger)
        self.it_agent = ITAgent(self.config, self.tool_registry, component_logger)
        self.finance_agent = FinanceAgent(self.config, self.tool_registry, component_logger)

    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process a user query through the multi-agent system."""
        return await self.orchestrator.process_query(query)

    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        return {
            "agents": {
                "supervisor": self.config.agents.supervisor.name,
                "it_agent": self.config.agents.it_agent.name,
                "finance_agent": self.config.agents.finance_agent.name
            },
            "tools": self.tool_registry.list_tools(),
            "config": {
                "model": self.config.aws.model,
                "region": self.config.aws.region,
                "temperature": self.config.aws.temperature,
                "max_tokens": self.config.aws.max_tokens
            },
            "logging": self.logging_manager.get_system_info()
        }

    def update_log_level(self, level: str) -> None:
        """Update the logging level for all components."""
        self.logging_manager.update_log_level(level)
        self.logger.info(f"Log level updated to {level}")

    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check of all system components."""
        health_status = {
            "system": "healthy",
            "components": {},
            "timestamp": asyncio.get_event_loop().time()
        }

        try:
            # Check tool registry
            health_status["components"]["tools"] = {
                "status": "healthy",
                "available_tools": len(self.tool_registry.list_tools())
            }

            # Check validator
            test_validation = self.validator.validate_query("test query")
            health_status["components"]["validator"] = {
                "status": "healthy" if test_validation.is_valid else "warning",
                "last_validation": test_validation.is_valid
            }

            # Check agents (basic initialization check)
            health_status["components"]["agents"] = {
                "supervisor": "healthy",
                "it_agent": "healthy",
                "finance_agent": "healthy"
            }

        except Exception as e:
            health_status["system"] = "unhealthy"
            health_status["error"] = str(e)
            self.logger.error(f"Health check failed: {str(e)}")

        return health_status
