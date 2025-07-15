#!/usr/bin/env python3
"""
Simple test to identify the issue with test execution.
"""

import sys
import os
import asyncio
from unittest.mock import Mock, patch

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_imports():
    """Test basic imports work."""
    try:
        from hierarchical_multi_agent_support.config import Config
        print("✅ Config import successful")

        from hierarchical_multi_agent_support.agents import AgentResponse
        print("✅ AgentResponse import successful")

        from hierarchical_multi_agent_support.models import ToolResult
        print("✅ ToolResult import successful")

        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_agent_response_creation():
    """Test creating an AgentResponse."""
    try:
        from hierarchical_multi_agent_support.agents import AgentResponse

        response = AgentResponse(
            success=True,
            message="Test response",
            agent_name="Test Agent"
        )

        assert response.success is True
        assert response.message == "Test response"
        assert response.agent_name == "Test Agent"
        print("✅ AgentResponse creation successful")
        return True
    except Exception as e:
        print(f"❌ AgentResponse creation failed: {e}")
        return False

def test_supervisor_agent_creation():
    """Test creating a SupervisorAgent with proper mocking."""
    try:
        from hierarchical_multi_agent_support.agents import SupervisorAgent
        from hierarchical_multi_agent_support.config import Config
        from hierarchical_multi_agent_support.tools import ToolRegistry
        import logging

        # Mock configuration
        config = Mock()
        config.aws = Mock()
        config.aws.model = "test-model"
        config.aws.region = "us-west-2"
        config.aws.temperature = 0.1
        config.aws.max_tokens = 1000

        # Mock tool registry and logger
        tool_registry = Mock()
        logger = Mock(spec=logging.Logger)

        # Mock ChatBedrock to prevent AWS connection
        with patch('hierarchical_multi_agent_support.agents.ChatBedrock') as mock_bedrock:
            mock_llm = Mock()
            mock_bedrock.return_value = mock_llm

            # This should not hang if mocking is correct
            supervisor = SupervisorAgent(config, tool_registry, logger)

            print("✅ SupervisorAgent creation successful")
            return True
    except Exception as e:
        print(f"❌ SupervisorAgent creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_async_functionality():
    """Test async functionality."""
    try:
        print("✅ Async test running")
        await asyncio.sleep(0.1)  # Small delay to test async
        print("✅ Async test completed")
        return True
    except Exception as e:
        print(f"❌ Async test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Running simple diagnostic tests...")

    results = []

    # Test basic imports
    results.append(test_basic_imports())

    # Test AgentResponse creation
    results.append(test_agent_response_creation())

    # Test SupervisorAgent creation
    results.append(test_supervisor_agent_creation())

    # Test async functionality
    try:
        result = asyncio.run(test_async_functionality())
        results.append(result)
    except Exception as e:
        print(f"❌ Async test execution failed: {e}")
        results.append(False)

    # Summary
    passed = sum(results)
    total = len(results)
    print(f"\n📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
