#!/usr/bin/env python3
"""
Demo script for the Multi-Agent Support System with AWS Bedrock/Claude
This script demonstrates the system functionality without requiring real AWS credentials.
"""

import asyncio
import logging
from unittest.mock import Mock, AsyncMock

# Mock AWS Bedrock for demo purposes
class MockChatBedrock:
    """Mock ChatBedrock for demonstration purposes."""

    def __init__(self, model_id, region_name, credentials_profile_name, model_kwargs):
        self.model_id = model_id
        self.region_name = region_name
        self.model_kwargs = model_kwargs

    async def ainvoke(self, messages):
        """Mock LLM response based on message content."""
        # Extract the last human message
        last_message = messages[-1].content if messages else ""

        # Simple routing logic for demo
        if "route this query" in last_message.lower():
            if any(keyword in last_message.lower() for keyword in ["password", "network", "computer", "email", "software"]):
                return Mock(content="IT - This is a technical support query that should be handled by the IT specialist.")
            elif any(keyword in last_message.lower() for keyword in ["expense", "budget", "payment", "finance", "money"]):
                return Mock(content="Finance - This is a financial query that should be handled by the Finance specialist.")
            else:
                return Mock(content="Unclear - This query is not clearly related to IT or Finance domains.")

        # IT Agent responses
        elif "it support agent" in last_message.lower():
            return Mock(content="""Based on your query, here are the recommended steps:

1. **First, check the basics:**
   - Verify all cable connections are secure
   - Ensure the device is powered on
   - Check for any error messages or indicators

2. **Troubleshooting steps:**
   - Restart the affected system/application
   - Check system logs for any errors
   - Verify network connectivity if applicable

3. **If the issue persists:**
   - Contact your system administrator
   - Document any error messages
   - Note when the issue first occurred

4. **For password-related issues:**
   - Use the self-service password reset portal
   - Contact IT help desk with proper identification
   - Follow company security policies

This solution is based on our internal IT documentation and best practices.""")

        # Finance Agent responses
        elif "finance support agent" in last_message.lower():
            return Mock(content="""Here's the guidance for your financial query:

1. **For expense reporting:**
   - Submit reports within 30 days of incurring expenses
   - Attach original receipts for expenses over $25
   - Use approved expense categories
   - Obtain manager approval for expenses over $500

2. **Budget-related queries:**
   - Review budget monthly with department heads
   - Submit variance reports for deviations >10%
   - Plan next fiscal year budget 6 months in advance

3. **Payment processing:**
   - Net 30 payment terms for approved vendors
   - Require purchase orders for expenses >$1000
   - Process payments twice weekly

4. **Authorization limits:**
   - Department heads: up to $5,000
   - Directors: up to $25,000
   - VP Finance: up to $100,000
   - CFO: above $100,000

This information is based on our internal finance policies and procedures.""")

        else:
            return Mock(content="I'm here to help with your query. Could you please provide more details about your specific issue?")


async def demo_system():
    """Demonstrate the multi-agent system functionality."""
    print("🚀 Multi-Agent Support System Demo")
    print("=" * 60)
    print("Using AWS Bedrock with Claude (Mock Implementation)")
    print("=" * 60)

    # Mock the ChatBedrock import for demo
    import sys
    from unittest.mock import patch

    # Sample queries to demonstrate routing
    demo_queries = [
        "How do I reset my password?",
        "My computer won't connect to the network",
        "How do I submit an expense report?",
        "What's the budget approval process?",
        "I'm having trouble with my email",
        "How do I request a new software license?"
    ]

    print("\n🎯 Demonstrating Query Routing:")
    print("-" * 40)

    for i, query in enumerate(demo_queries, 1):
        print(f"\n{i}. Query: '{query}'")

        # Simulate routing decision
        if any(keyword in query.lower() for keyword in ["password", "network", "computer", "email", "software"]):
            routing = "IT"
            agent_name = "IT Support Agent"
        elif any(keyword in query.lower() for keyword in ["expense", "budget", "payment", "finance", "money"]):
            routing = "Finance"
            agent_name = "Finance Support Agent"
        else:
            routing = "Unclear"
            agent_name = "Supervisor"

        print(f"   → Routed to: {routing} Agent")
        print(f"   → Handler: {agent_name}")

        # Simulate response generation
        if routing == "IT":
            print("   → Response: Providing IT troubleshooting steps and technical guidance...")
        elif routing == "Finance":
            print("   → Response: Providing financial procedures and policy information...")
        else:
            print("   → Response: Requesting clarification on domain (IT vs Finance)...")

    print("\n" + "=" * 60)
    print("🔧 System Architecture:")
    print("-" * 40)
    print("✅ Configuration: AWS Bedrock with Claude")
    print("✅ Model: anthropic.claude-3-sonnet-20240229-v1:0")
    print("✅ Region: us-east-1")
    print("✅ Agents: Supervisor, IT Support, Finance Support")
    print("✅ Tools: File Reader, Web Search")
    print("✅ Workflow: LangGraph-based orchestration")
    print("✅ Validation: Input sanitization and security checks")
    print("✅ Logging: Comprehensive system monitoring")

    print("\n🌟 Key Features:")
    print("-" * 40)
    print("• Hierarchical agent routing")
    print("• AWS Bedrock integration with Claude")
    print("• Comprehensive error handling")
    print("• Input validation and sanitization")
    print("• Tool-based information retrieval")
    print("• Configurable via YAML")
    print("• Full test coverage")
    print("• Production-ready logging")

    print("\n🚀 To run the actual system:")
    print("-" * 40)
    print("1. Set up AWS credentials in .env file")
    print("2. Configure AWS Bedrock access")
    print("3. Run: python main.py")
    print("4. Or try: python main.py --demo")

    print("\n✨ Demo completed successfully!")


if __name__ == "__main__":
    asyncio.run(demo_system())
