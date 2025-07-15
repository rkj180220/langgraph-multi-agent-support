"""
RAG (Retrieval-Augmented Generation) system for document search.
Uses AWS Titan embeddings for semantic search through internal documents.
"""

import os
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import faiss
import tiktoken
import boto3
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel
import pickle
import hashlib

from .config import Config
from .models import ToolResult


class DocumentChunk(BaseModel):
    """Represents a chunk of document text with metadata."""
    text: str
    source: str
    page: Optional[int] = None
    chunk_id: str
    metadata: Dict[str, Any] = {}


class RAGDocumentSearch:
    """RAG-based document search using AWS Titan embeddings."""

    def __init__(self, config: Config, logger: logging.Logger):
        """Initialize the RAG document search system."""
        self.config = config
        self.logger = logger

        # Initialize AWS Bedrock client for embeddings with better error handling
        try:
            self.logger.info(f"Initializing AWS Bedrock client with region: {config.aws.region}")
            self.bedrock_client = boto3.client(
                'bedrock-runtime',
                region_name=config.aws.region,
                aws_access_key_id=config.aws.access_key_id,
                aws_secret_access_key=config.aws.secret_access_key
            )
            self.logger.info("AWS Bedrock client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize AWS Bedrock client: {str(e)}")
            # Create a mock client that will handle errors gracefully
            self.bedrock_client = None

        # Text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        # Initialize tokenizer for text processing
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # Vector store components for different domains
        self.finance_vector_store: Optional[faiss.IndexFlatIP] = None
        self.it_vector_store: Optional[faiss.IndexFlatIP] = None
        self.finance_chunks: List[DocumentChunk] = []
        self.it_chunks: List[DocumentChunk] = []

        # Cache files for different domains
        self.finance_cache_file = Path("cache/finance_embeddings_cache.pkl")
        self.it_cache_file = Path("cache/it_embeddings_cache.pkl")

        # Initialization flags
        self._finance_initialized = False
        self._it_initialized = False
        self._initialization_lock = asyncio.Lock()

    async def _ensure_domain_initialized(self, domain: str) -> None:
        """Ensure the specified domain is initialized."""
        async with self._initialization_lock:
            if domain == "finance" and not self._finance_initialized:
                await self._initialize_finance_vector_store()
                self._finance_initialized = True
            elif domain == "it" and not self._it_initialized:
                await self._initialize_it_vector_store()
                self._it_initialized = True

    async def _initialize_finance_vector_store(self) -> None:
        """Initialize the finance vector store."""
        try:
            if self.finance_cache_file.exists():
                await self._load_finance_vector_store()
            else:
                await self._build_finance_vector_store()
        except Exception as e:
            self.logger.error(f"Failed to initialize finance vector store: {str(e)}")

    async def _initialize_it_vector_store(self) -> None:
        """Initialize the IT vector store."""
        try:
            if self.it_cache_file.exists():
                await self._load_it_vector_store()
            else:
                await self._build_it_vector_store()
        except Exception as e:
            self.logger.error(f"Failed to initialize IT vector store: {str(e)}")

    async def _load_finance_vector_store(self) -> None:
        """Load existing finance vector store from cache."""
        try:
            with open(self.finance_cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.finance_vector_store = cache_data['vector_store']
                self.finance_chunks = cache_data['document_chunks']

            self.logger.info(f"Loaded finance vector store with {len(self.finance_chunks)} chunks")
        except Exception as e:
            self.logger.error(f"Failed to load finance vector store: {str(e)}")
            await self._build_finance_vector_store()

    async def _load_it_vector_store(self) -> None:
        """Load existing IT vector store from cache."""
        try:
            with open(self.it_cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.it_vector_store = cache_data['vector_store']
                self.it_chunks = cache_data['document_chunks']

            self.logger.info(f"Loaded IT vector store with {len(self.it_chunks)} chunks")
        except Exception as e:
            self.logger.error(f"Failed to load IT vector store: {str(e)}")
            await self._build_it_vector_store()

    async def _build_finance_vector_store(self) -> None:
        """Build the finance vector store from PDF documents."""
        try:
            self.logger.info("Building finance vector store from PDF documents...")

            # Get all PDF files from finance docs
            finance_docs_path = Path(self.config.documents.finance_docs_path)
            pdf_files = list(finance_docs_path.glob("*.pdf"))

            if not pdf_files:
                self.logger.warning("No PDF files found in finance documents directory")
                return

            # Process each PDF file
            all_chunks = []
            for pdf_file in pdf_files:
                try:
                    chunks = await self._process_pdf_file(pdf_file)
                    all_chunks.extend(chunks)
                    self.logger.info(f"Processed {pdf_file.name}: {len(chunks)} chunks")
                except Exception as e:
                    self.logger.error(f"Error processing {pdf_file.name}: {str(e)}")

            if not all_chunks:
                self.logger.warning("No text chunks extracted from finance PDF files")
                return

            # Generate embeddings and build vector store
            await self._build_vector_store_from_chunks(all_chunks, "finance")

        except Exception as e:
            self.logger.error(f"Failed to build finance vector store: {str(e)}")

    async def _build_it_vector_store(self) -> None:
        """Build the IT vector store from markdown and other documents."""
        try:
            self.logger.info("Building IT vector store from documents...")

            # Get all supported files from IT docs
            it_docs_path = Path(self.config.documents.it_docs_path)
            supported_extensions = ['.md', '.txt', '.pdf', '.docx']

            all_files = []
            for ext in supported_extensions:
                all_files.extend(list(it_docs_path.glob(f"*{ext}")))

            if not all_files:
                self.logger.warning("No supported files found in IT documents directory")
                return

            # Process each file
            all_chunks = []
            for doc_file in all_files:
                try:
                    if doc_file.suffix.lower() == '.pdf':
                        chunks = await self._process_pdf_file(doc_file)
                    else:
                        chunks = await self._process_text_file(doc_file)
                    all_chunks.extend(chunks)
                    self.logger.info(f"Processed {doc_file.name}: {len(chunks)} chunks")
                except Exception as e:
                    self.logger.error(f"Error processing {doc_file.name}: {str(e)}")

            if not all_chunks:
                self.logger.warning("No text chunks extracted from IT documents")
                return

            # Generate embeddings and build vector store
            await self._build_vector_store_from_chunks(all_chunks, "it")

        except Exception as e:
            self.logger.error(f"Failed to build IT vector store: {str(e)}")

    async def _build_vector_store_from_chunks(self, chunks: List[DocumentChunk], domain: str) -> None:
        """Build vector store from document chunks for a specific domain."""
        # Generate embeddings for all chunks
        embeddings = await self._generate_embeddings([chunk.text for chunk in chunks])

        # Create FAISS index
        dimension = len(embeddings[0])
        vector_store = faiss.IndexFlatIP(dimension)

        # Add embeddings to index
        embeddings_array = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings_array)  # Normalize for cosine similarity
        vector_store.add(embeddings_array)

        # Store in appropriate domain
        if domain == "finance":
            self.finance_vector_store = vector_store
            self.finance_chunks = chunks
            await self._save_finance_vector_store()
        elif domain == "it":
            self.it_vector_store = vector_store
            self.it_chunks = chunks
            await self._save_it_vector_store()

        self.logger.info(f"Built {domain} vector store with {len(chunks)} chunks")

    async def _process_pdf_file(self, pdf_file: Path) -> List[DocumentChunk]:
        """Process a PDF file and extract text chunks."""
        try:
            import pypdf

            # Extract text from PDF
            text_content = ""
            with open(pdf_file, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                for page in pdf_reader.pages:
                    text_content += page.extract_text() + "\n"

            if not text_content.strip():
                self.logger.warning(f"No text extracted from PDF: {pdf_file.name}")
                return []

            return self._create_chunks_from_text(text_content, pdf_file.name)

        except Exception as e:
            self.logger.error(f"Error processing PDF file {pdf_file.name}: {str(e)}")
            return []

    async def _process_text_file(self, text_file: Path) -> List[DocumentChunk]:
        """Process a text/markdown file and extract text chunks."""
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                text_content = f.read()

            if not text_content.strip():
                return []

            return self._create_chunks_from_text(text_content, text_file.name)

        except Exception as e:
            self.logger.error(f"Error reading text file {text_file}: {str(e)}")
            return []

    def _create_chunks_from_text(self, text_content: str, source_name: str) -> List[DocumentChunk]:
        """Create document chunks from text content."""
        # Split text into chunks
        text_chunks = self.text_splitter.split_text(text_content)

        # Create document chunks
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            if len(chunk_text.strip()) < 50:  # Skip very short chunks
                continue

            chunk_id = hashlib.md5(f"{source_name}_{i}_{chunk_text[:100]}".encode()).hexdigest()

            chunk = DocumentChunk(
                text=chunk_text.strip(),
                source=source_name,
                chunk_id=chunk_id,
                metadata={
                    "file_path": source_name,
                    "chunk_index": i,
                    "token_count": len(self.tokenizer.encode(chunk_text))
                }
            )
            chunks.append(chunk)

        return chunks

    async def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using AWS Titan embeddings."""
        embeddings = []

        # Process texts in batches to avoid API limits
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = await self._get_batch_embeddings(batch)
            embeddings.extend(batch_embeddings)

        return embeddings

    async def _get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a batch of texts."""
        if not self.bedrock_client:
            self.logger.warning("AWS Bedrock client not available, using dummy embeddings")
            # Return dummy embeddings as fallback
            return [[0.0] * 1536 for _ in texts]

        try:
            embeddings = []

            for text in texts:
                # Prepare request for Titan embeddings - correct format
                request_body = {
                    "inputText": text[:8000]  # Titan has input limits
                }

                response = self.bedrock_client.invoke_model(
                    modelId="amazon.titan-embed-text-v1",
                    body=json.dumps(request_body)
                )

                response_body = json.loads(response['body'].read())
                embedding = response_body['embedding']
                embeddings.append(embedding)

                # Small delay to avoid rate limiting
                await asyncio.sleep(0.1)

            return embeddings

        except Exception as e:
            self.logger.error(f"Error generating embeddings: {str(e)}")
            # Return dummy embeddings as fallback
            return [[0.0] * 1536 for _ in texts]

    async def _save_finance_vector_store(self) -> None:
        """Save finance vector store to cache."""
        try:
            cache_data = {
                'vector_store': self.finance_vector_store,
                'document_chunks': self.finance_chunks
            }

            with open(self.finance_cache_file, 'wb') as f:
                pickle.dump(cache_data, f)

            self.logger.info("Finance vector store saved to cache")

        except Exception as e:
            self.logger.error(f"Failed to save finance vector store: {str(e)}")

    async def _save_it_vector_store(self) -> None:
        """Save IT vector store to cache."""
        try:
            cache_data = {
                'vector_store': self.it_vector_store,
                'document_chunks': self.it_chunks
            }

            with open(self.it_cache_file, 'wb') as f:
                pickle.dump(cache_data, f)

            self.logger.info("IT vector store saved to cache")

        except Exception as e:
            self.logger.error(f"Failed to save IT vector store: {str(e)}")

    async def search_documents(self, query: str, domain: str = "finance", top_k: int = 5) -> List[DocumentChunk]:
        """Search documents using semantic similarity for a specific domain."""
        if domain == "finance":
            vector_store = self.finance_vector_store
            chunks = self.finance_chunks
        elif domain == "it":
            vector_store = self.it_vector_store
            chunks = self.it_chunks
        else:
            self.logger.error(f"Unsupported domain: {domain}")
            return []

        if not vector_store or not chunks:
            self.logger.warning(f"Vector store not initialized for domain: {domain}")
            return []

        try:
            # Generate query embedding
            query_embeddings = await self._generate_embeddings([query])
            if not query_embeddings:
                return []

            query_embedding = np.array(query_embeddings[0]).astype('float32').reshape(1, -1)
            faiss.normalize_L2(query_embedding)

            # Search vector store
            similarities, indices = vector_store.search(query_embedding, top_k)

            # Retrieve matching chunks
            results = []
            for i, (similarity, index) in enumerate(zip(similarities[0], indices[0])):
                if index < len(chunks):
                    chunk = chunks[index]
                    chunk.metadata['similarity_score'] = float(similarity)
                    results.append(chunk)

            self.logger.info(f"Found {len(results)} relevant chunks for {domain} query: {query}")
            return results

        except Exception as e:
            self.logger.error(f"Error searching {domain} documents: {str(e)}")
            return []

    async def get_context_for_query(self, query: str, domain: str = "finance", max_context_length: int = 4000) -> str:
        """Get relevant context for a query with length limit."""
        relevant_chunks = await self.search_documents(query, domain, top_k=10)

        if not relevant_chunks:
            return ""

        # Build context from most relevant chunks
        context_parts = []
        current_length = 0

        for chunk in relevant_chunks:
            chunk_text = f"[From {chunk.source}]\n{chunk.text}\n"
            chunk_length = len(self.tokenizer.encode(chunk_text))

            if current_length + chunk_length > max_context_length:
                break

            context_parts.append(chunk_text)
            current_length += chunk_length

        context = "\n".join(context_parts)

        self.logger.info(f"Generated {domain} context with {len(context_parts)} chunks, {current_length} tokens")
        return context

    async def refresh_vector_store(self, domain: str = "both") -> None:
        """Refresh the vector store by rebuilding from current files."""
        try:
            if domain in ["finance", "both"]:
                # Remove finance cache file
                if self.finance_cache_file.exists():
                    self.finance_cache_file.unlink()
                # Rebuild finance vector store
                await self._build_finance_vector_store()

            if domain in ["it", "both"]:
                # Remove IT cache file
                if self.it_cache_file.exists():
                    self.it_cache_file.unlink()
                # Rebuild IT vector store
                await self._build_it_vector_store()

            self.logger.info(f"Vector store refreshed successfully for domain: {domain}")

        except Exception as e:
            self.logger.error(f"Failed to refresh vector store for domain {domain}: {str(e)}")

    async def initialize_all_vector_stores(self) -> None:
        """Initialize all vector stores for supported domains if not already initialized."""
        await self._ensure_domain_initialized("finance")
        await self._ensure_domain_initialized("it")
        self.logger.info("All vector stores initialized (if available)")
