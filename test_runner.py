#!/usr/bin/env python3
"""
Simple test runner for the multi-agent support system.
This bypasses pytest to avoid hanging issues.
"""

import sys
import os
import tempfile
from unittest.mock import Mock, patch

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_config_loading():
    """Test configuration loading."""
    print("Testing configuration loading...")

    try:
        from hierarchical_multi_agent_support.config import Config, ConfigManager

        # Create a temporary config file
        config_content = """
aws:
  region: "us-west-2"
  model: "anthropic.claude-3-sonnet-20240229-v1:0"
  temperature: 0.1
  max_tokens: 1000

agents:
  supervisor:
    name: "Test Supervisor"
  it_agent:
    name: "Test IT Agent"
  finance_agent:
    name: "Test Finance Agent"

tools:
  web_search:
    enabled: true
    timeout: 30
    max_results: 5

documents:
  it_docs_path: "docs/it"
  finance_docs_path: "docs/finance"

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "test_logs/test.log"

validation:
  max_query_length: 1000
  min_query_length: 3
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            f.flush()

            # Test loading
            config_manager = ConfigManager(f.name)
            config = config_manager.load_config()

            # Validate structure
            assert isinstance(config, Config)
            assert config.aws.model == "anthropic.claude-3-sonnet-20240229-v1:0"
            assert config.agents.supervisor.name == "Test Supervisor"
            assert config.tools.web_search.enabled is True

            print("✅ Configuration loading test passed")

            # Cleanup
            os.unlink(f.name)
            return True

    except Exception as e:
        print(f"❌ Configuration loading test failed: {e}")
        return False

def test_agent_response_model():
    """Test AgentResponse model."""
    print("Testing AgentResponse model...")

    try:
        from hierarchical_multi_agent_support.agents import AgentResponse

        # Create an AgentResponse
        response = AgentResponse(
            success=True,
            message="Test response",
            agent_name="Test Agent",
            routing_decision="IT",
            tool_calls=[{"tool": "test", "result": "success"}],
            metadata={"test": "value"}
        )

        # Validate fields
        assert response.success is True
        assert response.message == "Test response"
        assert response.agent_name == "Test Agent"
        assert response.routing_decision == "IT"
        assert len(response.tool_calls) == 1
        assert response.metadata["test"] == "value"

        print("✅ AgentResponse model test passed")
        return True

    except Exception as e:
        print(f"❌ AgentResponse model test failed: {e}")
        return False

def test_supervisor_routing_logic():
    """Test supervisor routing decision parsing."""
    print("Testing supervisor routing logic...")

    try:
        from hierarchical_multi_agent_support.agents import SupervisorAgent

        # Mock configuration
        config = Mock()
        config.aws = Mock()
        config.aws.model = "test-model"
        config.aws.region = "us-west-2"
        config.aws.temperature = 0.1
        config.aws.max_tokens = 1000

        tool_registry = Mock()
        logger = Mock()

        # Mock ChatBedrock to prevent AWS connection
        with patch('hierarchical_multi_agent_support.agents.ChatBedrock') as mock_bedrock:
            mock_llm = Mock()
            mock_bedrock.return_value = mock_llm

            supervisor = SupervisorAgent(config, tool_registry, logger)

            # Test routing decision parsing
            assert supervisor._parse_routing_decision("Finance - This is about expenses") == "Finance"
            assert supervisor._parse_routing_decision("IT - This is technical") == "IT"
            assert supervisor._parse_routing_decision("Both - This needs both domains") == "Both"
            assert supervisor._parse_routing_decision("Unclear - Cannot determine") == "Unclear"
            assert supervisor._parse_routing_decision("Random text") == "Unclear"

            print("✅ Supervisor routing logic test passed")
            return True

    except Exception as e:
        print(f"❌ Supervisor routing logic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tool_result_model():
    """Test ToolResult model."""
    print("Testing ToolResult model...")

    try:
        from hierarchical_multi_agent_support.models import ToolResult

        # Create a successful ToolResult
        result = ToolResult(
            success=True,
            data="Test data",
            metadata={"source": "test"}
        )

        assert result.success is True
        assert result.data == "Test data"
        assert result.metadata["source"] == "test"
        assert result.error is None

        # Create a failed ToolResult
        failed_result = ToolResult(
            success=False,
            data="",
            error="Test error"
        )

        assert failed_result.success is False
        assert failed_result.error == "Test error"

        print("✅ ToolResult model test passed")
        return True

    except Exception as e:
        print(f"❌ ToolResult model test failed: {e}")
        return False

def test_validation_logic():
    """Test input validation logic."""
    print("Testing validation logic...")

    try:
        from hierarchical_multi_agent_support.validation import InputValidator, ValidationResult

        # Mock configuration
        config = Mock()
        config.validation = Mock()
        config.validation.max_query_length = 1000
        config.validation.min_query_length = 3

        logger = Mock()

        validator = InputValidator(config, logger)

        # Test valid query
        result = validator.validate_query("How do I reset my password?")
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert result.sanitized_input == "How do I reset my password?"

        # Test empty query
        result = validator.validate_query("")
        assert result.is_valid is False
        assert result.error_message is not None

        print("✅ Validation logic test passed")
        return True

    except Exception as e:
        print(f"❌ Validation logic test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Running unit tests for multi-agent support system...")
    print("=" * 60)

    test_functions = [
        test_config_loading,
        test_agent_response_model,
        test_supervisor_routing_logic,
        test_tool_result_model,
        test_validation_logic,
    ]

    results = []

    for test_func in test_functions:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            results.append(False)
        print()

    # Summary
    passed = sum(results)
    total = len(results)

    print("=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All unit tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
