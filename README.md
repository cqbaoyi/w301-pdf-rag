# w301-pdf-rag

A RAG (Retrieval Augmented Generation) system for querying PDF documents using hybrid search, query fusion, and LLM-based response generation.

## Overview

This system enables you to:
- **Index PDFs**: Extract text, tables, and images from PDF files, generate embeddings, and store them in ElasticSearch
- **Query Documents**: Ask questions and get answers with citations using advanced RAG techniques including query fusion and hybrid search

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start ElasticSearch**:
   ```bash
   cd elastic-start-local
   ./start.sh
   ```

3. **Set required environment variables** (add to `~/.bashrc` or export in your shell):
   ```bash
   export EMBEDDING_URL=http://your-service:9800/v1/emb
   export RERANK_URL=http://your-service:2260/rerank
   export IMAGE_MODEL_URL=http://your-service:23333/v1
   export OPENAI_API_KEY=your-api-key-here
   ```

4. **Index a PDF**:
   ```bash
   python main.py index path/to/document.pdf
   ```

5. **Query the system**:
   ```bash
   python main.py query "What is the main topic of the document?"
   ```

## Configuration

### Required Environment Variables

These must be set before running:

```bash
# Service endpoints
export EMBEDDING_URL=http://your-service:9800/v1/emb      # Embedding service
export RERANK_URL=http://your-service:2260/rerank         # Reranking service
export IMAGE_MODEL_URL=http://your-service:23333/v1       # Image captioning service
export OPENAI_API_KEY=your-api-key-here                   # OpenAI API key
```

### Optional Configuration

Set in `config/config.yaml` or via environment variables:

- ElasticSearch settings (host, port, credentials, index name)
- Chunking parameters (chunk size, overlap, table handling)
- Search weights (dense vs sparse)
- LLM settings (model, temperature, max tokens)

See `config/config.yaml` for all available options.

## Usage

### Indexing

Index a single PDF:
```bash
python main.py index document.pdf
```

Index all PDFs in a directory:
```bash
python main.py index /path/to/pdf/directory/
```

### Querying

Ask a question:
```bash
python main.py query "What are the key findings in the document?"
```

## How It Works

### Indexing Pipeline

1. **Extract**: PDF processor extracts text, tables, and images
2. **Chunk**: Content is split into retrievable chunks with overlap
3. **Caption**: Images are captioned using vision-language model
4. **Embed**: All chunks are converted to vector embeddings
5. **Store**: Chunks and embeddings are stored in ElasticSearch with metadata

### Query Pipeline

1. **Query Fusion**: Generate multiple query variations from user query (RAG Fusion technique)
2. **Embed Queries**: Convert all query variations to embeddings
3. **Hybrid Search**: For each query variation, perform:
   - Dense search (vector similarity)
   - Sparse search (BM25)
   - Combine with weighted scores
4. **Result Fusion**: Merge results from all queries using Reciprocal Rank Fusion (RRF)
5. **Rerank**: Use reranking model to score and reorder results
6. **Generate**: LLM generates final answer with citations from top-ranked chunks

## Project Structure

```
w301-pdf-rag/
├── main.py                          # CLI entry point
├── requirements.txt                 # Python dependencies
├── config/
│   └── config.yaml                  # Configuration file
├── src/
│   ├── __init__.py
│   ├── config.py                    # Configuration management
│   ├── pdf_processor.py             # PDF extraction (text, tables, images)
│   ├── chunker.py                   # Content chunking
│   ├── image_captioner.py           # Image captioning
│   ├── embeddings.py                # Embedding generation
│   ├── elasticsearch_client.py      # ElasticSearch operations
│   ├── indexing_pipeline.py         # PDF indexing pipeline
│   ├── query_fusion.py              # Query variation generation
│   ├── retriever.py                 # Hybrid search retriever
│   ├── result_fusion.py             # Result fusion (RRF)
│   ├── reranker.py                  # Result reranking
│   ├── generator.py                 # Response generation
│   └── pipeline.py                  # Query processing pipeline
├── elastic-start-local/             # ElasticSearch local setup
│   ├── docker-compose.yml
│   ├── start.sh
│   ├── stop.sh
│   └── config/
└── tests/                           # Test files
    ├── test_embedding_api.py
    ├── test_image_captioning_api.py
    ├── test_reranking_api.py
    └── *.pdf                        # Test PDF files
```

## Requirements

- Python 3.8+
- ElasticSearch 8.x (via docker-compose in `elastic-start-local/`)
- External service endpoints for:
  - Embedding service
  - Reranking service
  - Image captioning service
  - OpenAI API (or compatible LLM endpoint)

## Architecture

### Core Components

- **PDF Processor**: Extracts structured content from PDFs
- **Chunker**: Splits documents into searchable chunks
- **Embedding Service**: Generates vector representations
- **ElasticSearch**: Stores documents and enables hybrid search
- **Query Fusion**: Expands queries for better retrieval
- **Hybrid Retriever**: Combines dense and sparse search
- **Reranker**: Refines search results
- **Response Generator**: Creates answers with citations

## License

[Add your license here]
