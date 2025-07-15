# Testing Guide for Multi-Agent Support System

## Overview
This guide provides comprehensive testing strategies for the hierarchical multi-agent support system using AWS Bedrock/Claude with the new streamlined architecture.

## Architecture Changes Reflected in Tests

### 🏗️ **Current Architecture (Updated)**
The system now uses a streamlined 3-agent architecture:
- **Supervisor Agent**: Handles both routing AND evaluation (dual responsibility)
- **IT Agent**: Processes IT-related queries with RAG search
- **Finance Agent**: Processes finance queries with PDF-based RAG search
- **Removed**: Separate EvaluatorAgent (functionality merged into Supervisor)

### 📊 **Key Testing Areas**
1. **Routing Precision**: Ensure single-domain queries don't route to "Both"
2. **Supervisor Dual Role**: Test both routing and evaluation functions
3. **Processing Path Tracking**: Verify correct workflow visualization
4. **Performance Optimization**: Validate 3-step vs 4-step processing

## Test Structure

### 1. **Unit Tests** (`tests/`)
- `test_agents.py` - Tests for SupervisorAgent (routing + evaluation), ITAgent, FinanceAgent
- `test_config.py` - Tests for configuration management
- `test_system.py` - Tests for the main system orchestration
- `test_tools.py` - Tests for web search, file reading, and RAG tools
- `test_validation.py` - Tests for input validation
- `test_orchestrator.py` - Tests for LangGraph workflow orchestration

### 2. **Integration Tests**
- End-to-end query processing with new architecture
- Vector store initialization and caching
- RAG document search functionality
- Processing path validation

### 3. **Routing Tests (Critical)**
- Single-domain classification accuracy
- Multi-domain detection (rare cases)
- Word boundary matching precision
- Fallback handling for unclear queries

### 4. **Manual Testing**
- Interactive CLI testing with processing path visualization
- Real AWS Bedrock integration testing
- Performance comparison (3-step vs 4-step)

## Running Tests

### Basic Test Commands

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_agents.py -v

# Run specific test class
uv run pytest tests/test_agents.py::TestSupervisorAgent -v

# Run specific test method
uv run pytest tests/test_agents.py::TestSupervisorAgent::test_supervisor_routing_precision -v

# Run tests with coverage
uv run pytest tests/ --cov=src/hierarchical_multi_agent_support --cov-report=html

# Run tests in parallel (faster)
uv run pytest tests/ -n auto

# Run only failed tests from last run
uv run pytest --lf
```

### Performance Testing

```bash
# Test routing performance
uv run pytest tests/test_agents.py::TestSupervisorAgent::test_routing_performance -v

# Test end-to-end processing time
uv run pytest tests/test_system.py::TestSystemPerformance -v

# Memory usage testing
uv run pytest tests/ --memray
```

## Key Test Scenarios

### 🎯 **Routing Precision Tests**

```python
# Test cases for improved routing
test_cases = [
    ("How do I submit an expense report?", "Finance"),  # Should NOT route to "Both"
    ("What's the travel policy?", "Finance"),
    ("How do I reset my password?", "IT"),
    ("My computer won't start", "IT"),
    ("My computer broke and I need to submit an expense report for a new one", "Both"),  # True multi-domain
    ("Issues with both my payroll and VPN setup", "Both"),  # True multi-domain
]
```

### 🔄 **Supervisor Dual Role Tests**

```python
# Test supervisor routing capability
def test_supervisor_routing():
    # Test routing decision making
    
# Test supervisor evaluation capability  
def test_supervisor_evaluation():
    # Test response refinement
    
# Test supervisor handles both roles efficiently
def test_supervisor_dual_role_efficiency():
    # Test processing time and accuracy
```

### 📊 **Processing Path Tests**

```python
# Test processing path tracking
def test_processing_path_single_domain():
    # Expected: Supervisor (Routing) → Finance Agent → Supervisor (Evaluation)
    
def test_processing_path_multi_domain():
    # Expected: Supervisor (Routing) → IT Agent → Finance Agent → Supervisor (Evaluation)
    
def test_processing_steps_count():
    # Single domain: 3 steps
    # Multi domain: 4 steps
```

## Test Data & Fixtures

### 🗂️ **Test Queries by Domain**

```python
# Finance domain test queries
FINANCE_QUERIES = [
    "How do I submit an expense report?",
    "What's my expense limit?",
    "Budget approval process?",
    "How do I file a reimbursement?",
    "Travel policy questions?",
    "Payroll procedures?",
]

# IT domain test queries
IT_QUERIES = [
    "How do I reset my password?",
    "My computer won't start",
    "VPN connection issues",
    "How to install software?",
    "Network troubleshooting",
    "Email setup problems",
]

# Multi-domain test queries (rare)
MULTI_DOMAIN_QUERIES = [
    "My computer broke and I need to submit an expense report for a new one",
    "Issues with both my payroll and VPN setup",
    "I need help with my salary AND installing new software",
]

# Unclear queries
UNCLEAR_QUERIES = [
    "What's the weather today?",
    "How do I get to the office?",
    "What's for lunch?",
]
```

### 🔧 **Mock Objects**

```python
# Mock supervisor agent for testing
@pytest.fixture
def mock_supervisor():
    return MockSupervisorAgent()

# Mock specialist agents
@pytest.fixture
def mock_it_agent():
    return MockITAgent()

@pytest.fixture
def mock_finance_agent():
    return MockFinanceAgent()

# Mock orchestrator with new architecture
@pytest.fixture
def mock_orchestrator():
    return MockWorkflowOrchestrator()
```

## Performance Benchmarks

### ⚡ **Processing Time Tests**

```python
# Test processing time improvement
def test_processing_time_single_domain():
    # Should complete in 3 steps
    assert processing_steps == 3
    assert processing_time < SINGLE_DOMAIN_THRESHOLD

def test_processing_time_multi_domain():
    # Should complete in 4 steps
    assert processing_steps == 4
    assert processing_time < MULTI_DOMAIN_THRESHOLD

# Compare with old architecture
def test_performance_improvement():
    # New architecture should be faster for single-domain queries
    assert new_processing_time < old_processing_time
```

### 🎯 **Accuracy Tests**

```python
# Test routing accuracy
def test_routing_accuracy():
    correct_routes = 0
    total_queries = len(test_queries)
    
    for query, expected_domain in test_queries:
        actual_domain = supervisor.route_query(query)
        if actual_domain == expected_domain:
            correct_routes += 1
    
    accuracy = correct_routes / total_queries
    assert accuracy > 0.95  # 95% accuracy threshold

# Test evaluation quality
def test_evaluation_quality():
    # Test response refinement effectiveness
    pass
```

## Integration Testing

### 🔗 **End-to-End Testing**

```bash
# Test complete workflow
uv run pytest tests/test_integration.py::test_complete_workflow -v

# Test with real AWS Bedrock
uv run pytest tests/test_integration.py::test_real_bedrock_integration -v

# Test RAG search functionality
uv run pytest tests/test_integration.py::test_rag_search_integration -v
```

### 📝 **Test Scenarios**

```python
# Test complete finance query processing
def test_finance_query_end_to_end():
    query = "How do I submit an expense report?"
    result = system.process_query(query)
    
    # Verify routing decision
    assert result.metadata["routing_decision"] == "Finance"
    
    # Verify processing path
    expected_path = [
        "Supervisor Agent (Routing)",
        "Finance Agent", 
        "Supervisor Agent (Evaluation)"
    ]
    assert result.metadata["processing_path"] == expected_path
    
    # Verify response quality
    assert result.success == True
    assert "expense report" in result.response.lower()
    assert result.metadata["evaluation_success"] == True

# Test complete IT query processing
def test_it_query_end_to_end():
    query = "How do I reset my password?"
    result = system.process_query(query)
    
    # Verify routing decision
    assert result.metadata["routing_decision"] == "IT"
    
    # Verify processing path
    expected_path = [
        "Supervisor Agent (Routing)",
        "IT Agent",
        "Supervisor Agent (Evaluation)"
    ]
    assert result.metadata["processing_path"] == expected_path
    
    # Verify response quality
    assert result.success == True
    assert "password" in result.response.lower()
```

## Manual Testing Procedures

### 🖥️ **Interactive CLI Testing**

1. **Start the system:**
   ```bash
   uv run python main.py
   ```

2. **Test routing precision:**
   ```
   💬 Your query: How do I submit an expense report?
   
   Expected Result:
   - Routing Decision: Finance (NOT Both)
   - Processing Path: Supervisor Agent (Routing) → Finance Agent → Supervisor Agent (Evaluation)
   - Total Processing Steps: 3 steps
   ```

3. **Test processing path visualization:**
   ```
   💬 Your query: How do I reset my password?
   
   Expected Result:
   - Processing Path: Supervisor Agent (Routing) → IT Agent → Supervisor Agent (Evaluation)
   - Specialist Agents: IT Agent (NOT both)
   - Tools Used: 2
   ```

4. **Test multi-domain queries:**
   ```
   💬 Your query: My computer broke and I need to submit an expense report for a new one
   
   Expected Result:
   - Routing Decision: Both
   - Processing Path: Supervisor Agent (Routing) → IT Agent → Finance Agent → Supervisor Agent (Evaluation)
   - Total Processing Steps: 4 steps
   ```

### 🔍 **Debugging Tests**

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with detailed output
uv run python main.py --verbose

# Check processing path in logs
tail -f logs/agent_system.log | grep "Processing Path"
```

## Test Coverage Requirements

### 📊 **Coverage Targets**
- **Overall Coverage**: > 90%
- **Routing Logic**: > 95% (critical functionality)
- **Evaluation Logic**: > 90%
- **Error Handling**: > 85%
- **Integration Tests**: > 80%

### 🎯 **Critical Test Areas**
1. **Supervisor Agent Routing**: Must achieve >95% accuracy
2. **Processing Path Tracking**: Must be 100% accurate
3. **Performance Optimization**: Must show improvement over old architecture
4. **Error Handling**: Must gracefully handle all edge cases

## Continuous Integration

### 🔄 **CI/CD Pipeline**

```yaml
# .github/workflows/test.yml
name: Test Suite
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.12'
      - name: Install uv
        run: pip install uv
      - name: Install dependencies
        run: uv sync
      - name: Run tests
        run: uv run pytest tests/ --cov=src/hierarchical_multi_agent_support --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v1
```

### 📈 **Performance Monitoring**

```python
# Monitor processing time trends
def test_performance_regression():
    # Ensure new changes don't slow down the system
    current_time = measure_processing_time()
    baseline_time = load_baseline_time()
    
    assert current_time <= baseline_time * 1.1  # Allow 10% variance
```

## Troubleshooting Test Issues

### 🔧 **Common Test Failures**

**Problem:** Routing tests failing with "Both" instead of single domain
**Solution:** Check the routing prompt and word boundary matching in `_parse_routing_decision()`

**Problem:** Processing path tests failing
**Solution:** Verify orchestrator workflow states and metadata tracking

**Problem:** Performance tests timing out
**Solution:** Check if vector store cache is working and optimize RAG search

**Problem:** Mock objects not behaving correctly
**Solution:** Update mock objects to match new Supervisor Agent dual-role behavior

### 🔍 **Debug Commands**

```bash
# Run single test with detailed output
uv run pytest tests/test_agents.py::TestSupervisorAgent::test_routing_precision -v -s

# Run with pdb debugging
uv run pytest tests/test_agents.py::TestSupervisorAgent::test_routing_precision --pdb

# Run with coverage and detailed report
uv run pytest tests/ --cov=src/hierarchical_multi_agent_support --cov-report=html --cov-report=term-missing
```

This updated testing guide now reflects the current streamlined architecture with the Supervisor Agent handling both routing and evaluation, improved routing precision, and the new processing path visualization features.
