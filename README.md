# Hierarchical Multi-Agent Support System

## Overview
A CLI-based multi-agent support system for IT and Finance queries, featuring:
- Multi-agent orchestration (Supervisor, IT, Finance)
- RAG (Retrieval-Augmented Generation) for Finance using PDF documents
- Titan Embeddings (AWS Bedrock) for semantic search
- Claude 3 (Sonnet/Haiku/Opus) via AWS Bedrock for LLM responses
- User-friendly CLI with rich output

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

## Supported Domains
- **IT**: General IT troubleshooting, how-tos, and support
- **Finance**: Only queries based on internal PDF documents (policies, guidelines, etc.)

## Features
- CLI with rich formatting (tables, colors, etc.)
- Query validation and domain routing
- RAG search for finance: extracts relevant context from PDFs using Titan Embeddings
- LLM response generation via Claude 3 on AWS Bedrock
- Semantic search through internal documentation
- Vector store caching for improved performance

## Example Prompts & Answers

### IT Domain
**Prompt:**
```
How do I reset my VPN password?
```
**Answer:**
```
To reset your VPN password, please visit the IT self-service portal and select 'Reset VPN Password'. If you encounter issues, contact the IT helpdesk at it-support@example.com.
```

### Finance Domain (RAG from PDF)
**Prompt:**
```
What is the per diem allowance for international travel?
```
**Answer:**
```
According to the 'Presidio travel-policy-International.pdf', the per diem allowance for international travel is $150 per day, covering meals and incidentals. Please refer to the policy document in docs/finance/ for detailed breakdowns by country and exceptions.
```

## Architecture

The system uses a hierarchical multi-agent architecture:
1. **Supervisor Agent**: Routes queries to appropriate specialists
2. **IT Agent**: Handles IT-related queries with RAG search
3. **Finance Agent**: Handles finance queries with PDF-based RAG search
4. **Tools**: Web search and RAG-based document search
5. **LangGraph**: Orchestrates the workflow between agents

## Project Structure
```
├── src/hierarchical_multi_agent_support/
│   ├── agents.py          # Agent implementations
│   ├── config.py          # Configuration management
│   ├── models.py          # Data models
│   ├── rag_search.py      # RAG search implementation
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

## Troubleshooting

### Common Issues
- **Vector store not initialized**: Ensure you have valid documents in `docs/finance/` or `docs/it/` and run the manual initialization command
- **AWS/Bedrock errors**: Check your credentials and ensure you have proper permissions for Bedrock and Titan Embeddings
- **Module not found**: Make sure you're using `uv run` to execute Python commands
- **Empty responses**: Check that your documents are properly placed in the docs directories

### Performance Tips
- Vector stores are cached in `cache/` directory for faster subsequent runs
- Use `uv run python main.py init-vector-store --domain both` to pre-build all vector stores
- Monitor the logs in `logs/agent_system.log` for debugging

## Configuration Reference

The `config.yaml` file supports the following sections:

- **aws**: AWS Bedrock configuration
- **agents**: Agent-specific settings
- **tools**: Tool configuration and timeouts
- **logging**: Logging levels and output
- **validation**: Input validation rules
- **documents**: Document storage paths

Environment variables can be used with the `${VAR_NAME}` or `${VAR_NAME:-default_value}` syntax.

## Contributing

1. Install development dependencies: `uv sync --dev`
2. Run tests: `uv run pytest`
3. Format code: `uv run black src/ tests/`
4. Type check: `uv run mypy src/`
5. Follow the existing code structure and patterns
