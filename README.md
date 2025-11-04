# w301-pdf-rag

A RAG (Retrieval Augmented Generation) system for querying PDF documents. Features hybrid search (vector + BM25), query fusion, and LLM-based response generation with citations.

## Features

- **Multi-modal PDF processing**: Extracts text, tables, and images (with automatic captioning)
- **Hybrid search**: Combines dense vector search with sparse BM25 keyword search
- **Query fusion**: Generates query variations for better retrieval coverage
- **Result fusion**: Merges results using Reciprocal Rank Fusion (RRF)
- **Reranking**: Uses neural reranking to improve final results
- **Citation support**: Generates answers with source citations

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
   # Quiet mode (default)
   python main.py query "What is the main topic of the document?"
   
   # Verbose mode (with --verbose flag)
   python main.py query --verbose "What is the main topic of the document?"
   ```

## Configuration

### Required Environment Variables

```bash
export EMBEDDING_URL=http://your-service:9800/v1/emb      # Embedding service endpoint
export RERANK_URL=http://your-service:2260/rerank         # Reranking service endpoint
export IMAGE_MODEL_URL=http://your-service:23333/v1       # Image captioning service endpoint
export OPENAI_API_KEY=your-api-key-here                   # API key for LLM services
```

### Configuration File

Edit `config/config.yaml` to customize:

- **ElasticSearch**: Host, port, credentials, index name
- **Chunking**: Text chunk size (default: 512), overlap (default: 50), table handling
- **Search**: Dense/sparse weights (default: 0.6/0.4), top-k parameters
- **Query Fusion**: Max variations, similarity thresholds
- **LLM**: Model names, temperature, max tokens

Environment variables override config file values. See `config/config.yaml` for defaults.

## Usage

### Indexing

```bash
# Index single PDF
python main.py index document.pdf

# Index all PDFs in directory
python main.py index /path/to/pdf/directory/

# Custom config file
python main.py --config custom_config.yaml index document.pdf
```

### Querying

```bash
# Default: Quiet mode (minimal output, no timestamps)
python main.py query "What are the key findings in the document?"

# Verbose mode: Full logging with timestamps
python main.py query --verbose "What are the key findings in the document?"
```

By default, query uses quiet mode showing only essential progress and the final response. Use `--verbose` to enable detailed logging with timestamps.

## How It Works

### Indexing Pipeline

1. **Extract**: Extracts text, tables, and images from PDF (filters small images)
2. **Chunk**: Splits content into overlapping chunks (sentence-aware for text, per-table for tables)
3. **Caption**: Generates captions for images using vision-language model
4. **Embed**: Converts all chunks to vector embeddings (batch processing)
5. **Store**: Indexes chunks, embeddings, and metadata in ElasticSearch

### Query Pipeline

1. **Query Fusion**: Generates query variations using LLM (skips for very short queries)
2. **Embed Queries**: Converts all variations to embeddings
3. **Hybrid Search**: For each variation, ElasticSearch performs:
   - Dense search (kNN vector similarity)
   - Sparse search (BM25 keyword matching)
   - Native score combination
4. **Result Fusion**: Merges results using Reciprocal Rank Fusion (RRF)
5. **Rerank**: Neural reranking model scores and reorders final candidates
6. **Generate**: LLM generates answer with citations from top-ranked chunks

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
│   ├── hybrid_retriever.py          # Hybrid search retriever
│   ├── result_fusion.py             # Result fusion (RRF)
│   ├── reranker.py                  # Result reranking
│   ├── response_generator.py        # Response generation
│   └── query_pipeline.py            # Query processing pipeline
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
- ElasticSearch 8.x+ (native hybrid search support via `knn` + `query`)
- External services (OpenAI-compatible APIs):
  - Embedding service (e.g., qwen3-embedding-0.6b)
  - Reranking service (e.g., qwen3-reranker-0.6b)
  - Image captioning service (e.g., internvl-internlm2)
  - LLM service (for query fusion and response generation)

## Potential Improvements

- **Coreference Resolution**: Add support for resolving pronouns in consecutive queries. For example, if a user asks "What did Thomas Jefferson say about the law?" followed by "What else did he say?", the system could automatically expand "he" to "Thomas Jefferson" in the second query. This would improve the natural flow of conversational interactions with the system.

- **Image and Table Query Stability**: Currently, queries specifically targeting images or tables can produce unstable or undesirable results. This is due to limitations in how image captions and table content are indexed and retrieved. Potential improvements include:
  - Better integration of image captions with surrounding text context
  - Improved table structure preservation during indexing and retrieval
  - Enhanced formatting of table content in responses for better LLM comprehension
  - More robust matching between queries and table/image content

