# Hierarchical Multi-Agent Support System

## Overview
A CLI-based multi-agent support system for IT and Finance queries, featuring:
- **Intelligent Multi-agent Orchestration** with Supervisor-led routing and evaluation
- **Streamlined 3-Step Processing** for efficient query handling
- **Enhanced RAG (Retrieval-Augmented Generation)** for both IT and Finance domains
- **Precise Domain Classification** to avoid unnecessary processing
- **Titan Embeddings (AWS Bedrock)** for semantic search
- **Claude 3 (Sonnet/Haiku/Opus)** via AWS Bedrock for LLM responses
- **User-friendly CLI** with rich output and processing path visualization

## Key Features

### 🚀 **Streamlined Architecture**
- **Supervisor Agent**: Handles both intelligent routing AND response evaluation
- **Optimized Processing**: 3-step workflow instead of 4-step (faster response times)
- **Precise Classification**: Improved routing logic prevents unnecessary dual-domain processing
- **Processing Path Visualization**: See exactly how your query flows through the system

### 🧠 **Enhanced Intelligence**
- **Smart Routing**: Routes queries to single domains unless explicitly multi-domain
- **Response Evaluation**: Supervisor refines specialist responses for clarity and relevance
- **Context-Aware Processing**: Uses RAG search for both IT and Finance domains
- **Fallback Mechanisms**: Graceful error handling with informative feedback

### 📊 **Processing Details**
The system now shows comprehensive processing metadata:
- **Processing Path**: Visual flow (e.g., `Supervisor Agent (Routing) → Finance Agent → Supervisor Agent (Evaluation)`)
- **Routing Decision**: Clear domain classification (IT/Finance/Both/Unclear)
- **Specialist Agents**: Which agents actually processed your query
- **Tools Used**: Number of tools utilized
- **Evaluation Success**: Whether response was successfully refined

## Configuration
- **Model**: Claude 3 (configurable via environment, e.g., `anthropic.claude-3-sonnet-20240229-v1:0`)
- **Region**: AWS region (e.g., `us-west-2`)
- **Temperature**: 0.1 (default, configurable)
- **Max Tokens**: 1000 (default, configurable)
- **Finance Knowledge Base**: All PDF files in `docs/finance/` are indexed and used for RAG search
- **IT Knowledge Base**: Markdown files in `docs/it/`

## Quick Start

### Prerequisites
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- AWS credentials with Bedrock access

### Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd Hierarchical-Multi-Agent-Support
   ```

2. **Install dependencies using uv:**
   ```bash
   uv sync
   ```

3. **Configure AWS credentials:**
   - Set your AWS credentials and region in your environment:
   ```bash
   export AWS_ACCESS_KEY_ID=your_access_key
   export AWS_SECRET_ACCESS_KEY=your_secret_key
   export AWS_REGION=us-east-1
   ```
   - Or use AWS CLI configuration: `aws configure`

4. **Configure the system:**
   - Edit `config.yaml` to set your desired model, region, and other parameters
   - Ensure you have access to AWS Bedrock and Titan Embeddings

5. **Add your internal documents:**
   - Place finance PDFs in `docs/finance/`
   - Place IT markdown files in `docs/it/`

## Usage

### Run the CLI Application
```bash
uv run python main.py
```

### Advanced Usage Options
- **Demo mode:**
  ```bash
  uv run python main.py --demo
  ```
- **Batch processing:**
  ```bash
  uv run python main.py --batch "How do I reset my password?" "How do I submit an expense report?"
  ```
- **Initialize vector stores manually:**
  ```bash
  uv run python main.py init-vector-store --domain finance
  uv run python main.py init-vector-store --domain it
  ```

### Interactive Commands
Once running, you can use these commands:
- `help` - Show available commands
- `info` - Display system information
- `quit`, `exit`, `bye` - Exit the application

### Development Commands
- **Run tests:**
  ```bash
  uv run pytest
  ```
- **Format code:**
  ```bash
  uv run black src/ tests/
  ```
- **Type checking:**
  ```bash
  uv run mypy src/
  ```

## Supported Domains & Routing

### 🔧 **IT Domain**
**Handles:** Technical issues, software problems, hardware troubleshooting, network issues, security concerns, system administration

**Example Queries:**
- "How do I reset my password?"
- "My computer won't start"
- "VPN connection issues"
- "How to install software?"
- "Network troubleshooting"

### 💰 **Finance Domain**
**Handles:** Financial queries, accounting questions, budget analysis, expense reports, financial calculations, payment processing

**Example Queries:**
- "How do I submit an expense report?"
- "What's the travel policy?"
- "Budget approval process"
- "Reimbursement procedures"
- "Payroll questions"

### 🔄 **Multi-Domain (Rare)**
**Only when explicitly needing BOTH domains:**
- "My computer broke and I need to submit an expense report for a new one"
- "Issues with both my payroll and VPN setup"
- "I need help with my salary AND installing new software"

## Features

### 🎯 **Intelligent Routing**
- **Precise Classification**: Improved algorithm prevents false "Both" routing
- **Single-Domain Focus**: Routes to one domain unless explicitly multi-domain
- **Word Boundary Matching**: Avoids false matches (e.g., "IT" in "submit")
- **Priority-Based Parsing**: Finance-first matching for better accuracy

### 📈 **Enhanced Processing**
- **Supervisor-Led Evaluation**: Same agent handles routing and evaluation for efficiency
- **Response Refinement**: Specialist responses are refined for clarity and relevance
- **Processing Path Tracking**: Full visibility into query flow
- **Fallback Mechanisms**: Graceful error handling with informative feedback

### 🔍 **Advanced RAG Search**
- **Semantic Search**: Uses Titan Embeddings for document similarity
- **Context-Aware Responses**: Retrieves relevant information from internal docs
- **Source Attribution**: Shows which documents were used
- **Vector Store Caching**: Improved performance with persistent embeddings

### 🎨 **Rich CLI Experience**
- **Processing Path Visualization**: See exactly how your query flows
- **Color-Coded Output**: Easy-to-read responses with visual formatting
- **Progress Indicators**: Real-time feedback during processing
- **Metadata Display**: Comprehensive processing details

## Architecture

### 🏗️ **Current Architecture (Optimized)**
The system now uses a streamlined 3-agent architecture:

1. **Supervisor Agent** (Dual Role):
   - **Routing**: Analyzes queries and routes to appropriate specialists
   - **Evaluation**: Refines specialist responses for clarity and relevance

2. **IT Agent**: Handles IT-related queries with RAG search through internal documentation

3. **Finance Agent**: Handles finance queries with PDF-based RAG search

4. **Tools & Orchestration**:
   - **Web Search**: External information retrieval
   - **RAG Search**: Internal document search for both domains
   - **LangGraph**: Orchestrates the workflow between agents

### 📊 **Processing Flow**
```
Query Input
    ↓
Supervisor Agent (Routing)
    ↓
[IT Agent] OR [Finance Agent] OR [Both Agents]
    ↓
Specialist Processing (RAG + Web Search)
    ↓
Supervisor Agent (Evaluation)
    ↓
Refined Response Output
```

### 🔄 **Workflow States**
- **Validation**: Input sanitization and validation
- **Routing**: Intelligent domain classification
- **Specialist Processing**: Domain-specific query handling
- **Evaluation**: Response refinement and quality assurance
- **Error Handling**: Graceful fallback mechanisms

## Example Outputs

### Finance Query (Optimized)
**Query:** "How do I submit an expense report?"

**Processing Details:**
```
Processing Path: Supervisor Agent (Routing) → Finance Agent → Supervisor Agent (Evaluation)
Routing Decision: Finance
Specialist Agents: Finance Agent
Tools Used: 2
Total Processing Steps: 3 steps
```

### IT Query (Optimized)
**Query:** "How do I reset my password?"

**Processing Details:**
```
Processing Path: Supervisor Agent (Routing) → IT Agent → Supervisor Agent (Evaluation)
Routing Decision: IT
Specialist Agents: IT Agent
Tools Used: 2
Total Processing Steps: 3 steps
```

### Multi-Domain Query (Rare)
**Query:** "My computer broke and I need to submit an expense report for a new one"

**Processing Details:**
```
Processing Path: Supervisor Agent (Routing) → IT Agent → Finance Agent → Supervisor Agent (Evaluation)
Routing Decision: Both
Specialist Agents: IT Agent, Finance Agent
Tools Used: 4
Total Processing Steps: 4 steps
```

## Project Structure
```
├── src/hierarchical_multi_agent_support/
│   ├── agents.py          # Agent implementations (Supervisor, IT, Finance)
│   ├── config.py          # Configuration management
│   ├── models.py          # Data models
│   ├── orchestrator.py    # LangGraph workflow orchestration
│   ├── rag_search.py      # RAG search implementation
│   ├── state.py           # Workflow state management
│   ├── system.py          # Main system orchestrator
│   ├── tools.py           # Tool implementations
│   └── validation.py      # Input validation
├── tests/                 # Test suite
├── docs/                  # Documentation storage
│   ├── finance/          # Finance PDF documents
│   └── it/               # IT documentation
├── cache/                # Vector store cache
├── config.yaml           # Configuration file
└── main.py              # CLI entry point
```

## Performance Improvements

### ⚡ **Faster Processing**
- **3-Step Workflow**: Reduced from 4 steps to 3 for most queries
- **Precise Routing**: No unnecessary dual-domain processing
- **Supervisor Efficiency**: Single agent handles both routing and evaluation

### 🎯 **Better Accuracy**
- **Enhanced Routing Logic**: Improved classification prevents false matches
- **Word Boundary Matching**: Precise pattern matching
- **Priority-Based Parsing**: Finance-first matching for better accuracy

### 📈 **Improved User Experience**
- **Processing Path Visualization**: Clear visibility into query flow
- **Comprehensive Metadata**: Full processing details
- **Rich CLI Output**: Color-coded, formatted responses
- **Real-time Progress**: Processing indicators and feedback

## Contributing

When contributing to this project, please:
1. Follow the existing code structure
2. Update tests for any new functionality
3. Ensure routing logic maintains precision
4. Update documentation for new features
5. Test with both single and multi-domain queries

## Troubleshooting

### Common Issues

**Problem:** Query routed to "Both" when it should be single-domain
**Solution:** Check the routing prompt in `SupervisorAgent._create_routing_prompt()` and ensure word boundary matching in `_parse_routing_decision()`

**Problem:** Slow processing times
**Solution:** Verify vector store cache is working and check if unnecessary dual-domain processing is occurring

**Problem:** Inaccurate responses
**Solution:** Check if the Supervisor Agent evaluation is working correctly and if RAG search is finding relevant documents
