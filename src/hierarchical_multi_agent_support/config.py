"""
Configuration management for the multi-agent support system.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv


class AWSConfig(BaseModel):
    """AWS Bedrock configuration."""
    access_key_id: str
    secret_access_key: str
    region: str = "us-east-1"
    model: str = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    temperature: float = 0.1
    max_tokens: int = 1000


class AgentConfig(BaseModel):
    """Individual agent configuration."""
    name: str
    description: str


class AgentsConfig(BaseModel):
    """Configuration for all agents."""
    supervisor: AgentConfig
    it_agent: AgentConfig
    finance_agent: AgentConfig
    evaluator_agent: AgentConfig


class ToolConfig(BaseModel):
    """Tool configuration."""
    enabled: bool = True
    timeout: int = 30
    max_results: int = 5


class FileReaderConfig(BaseModel):
    """File reader tool configuration."""
    enabled: bool = True
    max_file_size: int = 1048576  # 1MB
    allowed_extensions: list[str] = [".txt", ".md", ".pdf", ".doc", ".docx"]


class ToolsConfig(BaseModel):
    """Configuration for all tools."""
    web_search: ToolConfig
    file_reader: FileReaderConfig


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "logs/agent_system.log"


class ValidationConfig(BaseModel):
    """Input validation configuration."""
    max_query_length: int = 1000
    min_query_length: int = 5
    allowed_characters: str = Field(
        default="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .!?@#$%^&*()_+-=[]{}|;':\",./<>?`~"
    )


class DocumentsConfig(BaseModel):
    """Document paths configuration."""
    it_docs_path: str = "docs/it"
    finance_docs_path: str = "docs/finance"


class Config(BaseModel):
    """Main configuration class."""
    aws: AWSConfig
    agents: AgentsConfig
    tools: ToolsConfig
    logging: LoggingConfig
    validation: ValidationConfig
    documents: DocumentsConfig


class ConfigManager:
    """Configuration manager for the multi-agent support system."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager."""
        self.config_path = config_path or "config.yaml"
        load_dotenv()  # Load environment variables

    def load_config(self) -> Config:
        """Load configuration from YAML file."""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

            with open(config_file, 'r', encoding='utf-8') as file:
                config_data = yaml.safe_load(file)

            # Substitute environment variables
            config_data = self._substitute_env_vars(config_data)

            return Config(**config_data)

        except Exception as e:
            raise RuntimeError(f"Failed to load configuration: {str(e)}")

    def _substitute_env_vars(self, data: Any) -> Any:
        """Recursively substitute environment variables in configuration."""
        if isinstance(data, dict):
            return {key: self._substitute_env_vars(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._substitute_env_vars(item) for item in data]
        elif isinstance(data, str) and data.startswith("${") and data.endswith("}"):
            env_var = data[2:-1]
            # Handle default values like ${AWS_REGION:-us-east-1}
            if ":-" in env_var:
                var_name, default_value = env_var.split(":-", 1)
                return os.getenv(var_name, default_value)
            else:
                return os.getenv(env_var, "")
        else:
            return data

    def validate_config(self, config: Config) -> None:
        """Validate configuration values."""
        # Validate AWS configuration with better error messages
        if not config.aws.access_key_id or config.aws.access_key_id.strip() == "":
            raise ValueError("AWS access key ID is required. Please set AWS_ACCESS_KEY_ID environment variable.")

        if not config.aws.secret_access_key or config.aws.secret_access_key.strip() == "":
            raise ValueError("AWS secret access key is required. Please set AWS_SECRET_ACCESS_KEY environment variable.")

        if not config.aws.region or config.aws.region.strip() == "":
            raise ValueError("AWS region is required. Please set AWS_REGION environment variable or use default 'us-east-1'.")

        # Validate temperature and token limits
        if config.aws.temperature < 0 or config.aws.temperature > 1:
            raise ValueError("Temperature must be between 0 and 1")

        if config.aws.max_tokens < 1:
            raise ValueError("Max tokens must be positive")

        # Create necessary directories
        logs_dir = Path(config.logging.file).parent
        logs_dir.mkdir(parents=True, exist_ok=True)

        docs_it_dir = Path(config.documents.it_docs_path)
        docs_finance_dir = Path(config.documents.finance_docs_path)
        docs_it_dir.mkdir(parents=True, exist_ok=True)
        docs_finance_dir.mkdir(parents=True, exist_ok=True)
