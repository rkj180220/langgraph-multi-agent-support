# Testing Guide for Multi-Agent Support System

## Overview
This guide provides comprehensive testing strategies for your multi-agent support system using AWS Bedrock/Claude.

## Current Test Issues Fixed

The original tests were failing because they expected OpenAI configuration but your system uses AWS Bedrock/Claude. I've updated the test fixtures to match your current architecture.

## Test Structure

### 1. **Unit Tests** (`tests/`)
- `test_agents.py` - Tests for SupervisorAgent, ITAgent, FinanceAgent
- `test_config.py` - Tests for configuration management
- `test_system.py` - Tests for the main system orchestration
- `test_tools.py` - Tests for web search, file reading, and RAG tools
- `test_validation.py` - Tests for input validation

### 2. **Integration Tests**
- End-to-end query processing
- Vector store initialization
- RAG document search functionality

### 3. **Manual Testing**
- Interactive CLI testing
- Real AWS Bedrock integration testing

## Running Tests

### Basic Test Commands

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_agents.py -v

# Run specific test class
python -m pytest tests/test_agents.py::TestSupervisorAgent -v

# Run specific test method
python -m pytest tests/test_agents.py::TestSupervisorAgent::test_supervisor_process_it_query -v

# Run tests with coverage
python -m pytest tests/ --cov=src/hierarchical_multi_agent_support --cov-report=html

# Run tests in parallel (faster)
python -m pytest tests/ -n auto

# Run only failed tests from last run
python -m pytest --lf
```

### Test Categories

```bash
# Run only unit tests
python -m pytest tests/ -m "not integration"

# Run only integration tests
python -m pytest tests/ -m integration

# Run only async tests
python -m pytest tests/ -k "async"
```

## Test Types

### 1. **Unit Tests**
Test individual components in isolation:
- Agent routing logic
- Configuration loading
- Tool execution
- Input validation

### 2. **Integration Tests**
Test component interactions:
- Full query processing pipeline
- Vector store with real documents
- AWS Bedrock integration (with mocking)

### 3. **End-to-End Tests**
Test complete user workflows:
- CLI interaction simulation
- Real document processing
- Complete query-to-response flow

## Mock Strategy

### AWS Bedrock Mocking
```python
# Mock AWS Bedrock client
with patch('boto3.client') as mock_boto:
    mock_bedrock = Mock()
    mock_boto.return_value = mock_bedrock
    
    # Mock Claude response
    mock_bedrock.invoke_model.return_value = {
        'body': Mock(read=Mock(return_value=json.dumps({
            'completion': 'Test response from Claude'
        })))
    }
```

### Vector Store Mocking
```python
# Mock FAISS vector store
with patch('faiss.IndexFlatIP') as mock_faiss:
    mock_index = Mock()
    mock_faiss.return_value = mock_index
    mock_index.search.return_value = (
        np.array([[0.9, 0.8, 0.7]]),  # similarities
        np.array([[0, 1, 2]])         # indices
    )
```

## Testing Best Practices

### 1. **Test Data Management**
```python
@pytest.fixture
def sample_pdf_content():
    """Create sample PDF content for testing."""
    return "This is a test document about expense policies..."

@pytest.fixture
def temp_finance_docs():
    """Create temporary finance documents for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test PDF
        pdf_path = Path(tmpdir) / "test_policy.pdf"
        # ... create test PDF content
        yield tmpdir
```

### 2. **Async Testing**
```python
@pytest.mark.asyncio
async def test_async_function():
    """Test async functions properly."""
    result = await some_async_function()
    assert result.success is True
```

### 3. **Error Handling Tests**
```python
def test_error_handling():
    """Test error conditions."""
    with pytest.raises(SpecificException):
        problematic_function()
```

## Manual Testing Guide

### 1. **Interactive CLI Testing**
```bash
# Start the application
python main.py

# Test different query types
# IT queries: "My computer won't start"
# Finance queries: "How do I submit expenses?"
# Unclear queries: "Hello, how are you?"
```

### 2. **Command Line Testing**
```bash
# Test vector store initialization
python main.py init-vector-store --domain finance

# Test demo mode
python main.py --demo

# Test batch mode
python main.py --batch "IT query" "Finance query"
```

### 3. **Document Processing Testing**
```bash
# Add test documents to docs/finance/ and docs/it/
# Run vector store initialization
# Test queries related to your documents
```

## Environment Setup for Testing

### 1. **Test Configuration**
Create `config_test.yaml`:
```yaml
aws:
  region: "us-west-2"
  access_key_id: "${AWS_ACCESS_KEY_ID}"
  secret_access_key: "${AWS_SECRET_ACCESS_KEY}"

bedrock:
  model_id: "anthropic.claude-3-sonnet-20240229-v1:0"
  temperature: 0.1
  max_tokens: 1000

# ... rest of config
```

### 2. **Test Environment Variables**
```bash
export AWS_ACCESS_KEY_ID="test-key"
export AWS_SECRET_ACCESS_KEY="test-secret"
export PYTEST_CURRENT_TEST="true"
```

## Performance Testing

### 1. **Load Testing**
```python
@pytest.mark.performance
def test_concurrent_queries():
    """Test system under concurrent load."""
    import asyncio
    
    async def run_concurrent_queries():
        tasks = []
        for i in range(10):
            task = system.process_query(f"Test query {i}")
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        assert all(r['success'] for r in results)
```

### 2. **Memory Usage Testing**
```python
import psutil
import gc

def test_memory_usage():
    """Test memory usage during processing."""
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    
    # Process many queries
    for i in range(100):
        system.process_query(f"Query {i}")
        gc.collect()
    
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    
    # Assert memory increase is reasonable
    assert memory_increase < 100 * 1024 * 1024  # Less than 100MB
```

## Troubleshooting Test Issues

### 1. **Common Test Failures**
- **Import errors**: Check PYTHONPATH and module imports
- **Configuration errors**: Verify test config matches your system
- **Async errors**: Use `@pytest.mark.asyncio` for async tests
- **Mock errors**: Ensure mocks match actual API signatures

### 2. **Debug Test Failures**
```bash
# Run with verbose output
python -m pytest tests/ -v -s

# Run with debugging
python -m pytest tests/ --pdb

# Run with trace
python -m pytest tests/ --trace
```

### 3. **Test Isolation Issues**
```python
@pytest.fixture(autouse=True)
def cleanup():
    """Clean up after each test."""
    yield
    # Clean up resources
    # Clear caches
    # Reset global state
```

## Continuous Integration

### 1. **GitHub Actions Example**
```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.12
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-asyncio pytest-cov
    - name: Run tests
      run: pytest tests/ --cov=src --cov-report=xml
```

## Test Coverage Goals

- **Unit Tests**: 90%+ coverage
- **Integration Tests**: Cover all major workflows
- **Error Handling**: Test all error paths
- **Edge Cases**: Test boundary conditions

## Running Specific Test Scenarios

### 1. **Test RAG Functionality**
```bash
# Ensure you have documents in docs/finance/
python -m pytest tests/test_tools.py::TestRAGSearchTool -v
```

### 2. **Test Agent Routing**
```bash
python -m pytest tests/test_agents.py::TestSupervisorAgent -v
```

### 3. **Test Configuration**
```bash
python -m pytest tests/test_config.py -v
```

This comprehensive testing approach will help you maintain code quality and catch issues early in development.
