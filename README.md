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

## Usage
Run the main CLI:
```bash
python main.py
```

### Supported Domains
- **IT**: General IT troubleshooting, how-tos, and support
- **Finance**: Only queries based on internal PDF documents (policies, guidelines, etc.)

### Features
- CLI with rich formatting (tables, colors, etc.)
- Query validation and domain routing
- RAG search for finance: extracts relevant context from PDFs using Titan Embeddings
- LLM response generation via Claude 3 on AWS Bedrock

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

## Initial Setup

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd Hierarchical-Multi-Agent-Support
   ```
2. **Create a virtual environment and activate it:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Configure AWS credentials:**
   - Set your AWS credentials and region in your environment or via `~/.aws/credentials` and `~/.aws/config`.
   - Ensure you have access to Bedrock and Titan Embeddings.
5. **Edit `config.yaml` as needed:**
   - Set your desired model, region, and other parameters.
6. **Add your internal documents:**
   - Place finance PDFs in `docs/finance/`.
   - Place IT markdown files in `docs/it/`.
7. **(Optional) Manually initialize vector stores:**
   ```bash
   python main.py init-vector-store --domain finance
   python main.py init-vector-store --domain it
   ```
   This will pre-build the semantic search index for each domain.

## Advanced Usage
- Run in demo mode:
  ```bash
  python main.py --demo
  ```
- Run in batch mode:
  ```bash
  python main.py --batch "How do I reset my password?" "How do I submit an expense report?"
  ```

## Troubleshooting
- If you see `Vector store not initialized for domain: finance`, ensure you have valid PDFs in `docs/finance/` and run the manual initialization command above.
- For AWS/Bedrock errors, check your credentials and permissions.

---

For more details, see the code and comments in each module.
