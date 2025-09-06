# WET Pipeline

A web content processing pipeline that extracts, filters, and embeds web pages from Common Crawl's WET (Web Extracted Text) files. The pipeline includes English language filtering and local embedding generation for cost-effective semantic search.

## Features

- **English Language Filtering**: Automatically detects and processes only English content
- **Local Embeddings**: Uses sentence-transformers for free, local embedding generation
- **Smart Chunking**: Sentence-based chunking with optimized overlap control
- **Vector Search**: PostgreSQL with pgvector for semantic similarity search
- **Cost Effective**: No cloud API costs for embeddings

## Prerequisites

- Linux or macOS (or any OS with Docker)
- Python 3.10+
- PostgreSQL 16+ with pgvector extension

## Quick Start

### 1) Start PostgreSQL (Docker)
```bash
docker run --name wet-pg \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=wet_pipeline \
  -p 5432:5432 -d postgres:16
```

### 2) Create Environment File
Create a `.env` file in the project root:
```env
PGHOST=localhost
PGPORT=5432
PGDATABASE=wet_pipeline
PGUSER=postgres
PGPASSWORD=postgres
```

### 3) Setup Python Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4) Initialize Database
```bash
python3 scripts/db.py
```

### 5) Download WET File Paths
Streams Common Crawl's wet.paths.gz and inserts paths into the database:
```bash
python3 scripts/filePathDownload.py
```

### 6) Process WET Files (English Only)
Downloads WET files, extracts pages, and filters for English content:
```bash
python3 scripts/individualFileDownload.py
```

### 7) Generate Embeddings
Creates embeddings for all pending pages using local sentence-transformers:
```bash
python3 scripts/embedPages.py
```

## Pipeline Architecture

```
WET Files → Language Detection → English Pages → Chunking → Local Embeddings → Vector Storage
```

### Database Schema

#### `wet_paths` Table
- Tracks WET file paths and processing status
- Status: `pending`, `processing`, `done`, `error`

#### `pages` Table
- Stores individual web pages (English only)
- Includes language detection results
- Status: `pending`, `processing`, `done`, `error`

#### `page_chunks` Table
- Text chunks with 384-dimensional embeddings
- Optimized for semantic search
- Model: `all-MiniLM-L6-v2`

## Configuration

### Chunking Settings
```python
CHUNK_TARGET_TOKENS = 250        # Target tokens per chunk
CHUNK_OVERLAP_RATIO = 0.2        # 20% overlap between chunks (max 50 tokens)
USE_CHARACTER_CHUNKING = False   # Use sentence-based chunking (proven stable)
```

### Language Detection
```python
MIN_CONFIDENCE = 0.6              # Minimum confidence for English detection
SAMPLE_SIZE = 1000                # Characters used for language detection
```

## Monitoring

### Check Processing Status
```sql
-- View WET file processing status
SELECT status, COUNT(*) FROM wet_paths GROUP BY status;

-- View page processing status
SELECT embedding_status, COUNT(*) FROM pages GROUP BY embedding_status;

-- View language distribution
SELECT language, COUNT(*) FROM pages GROUP BY language;
```

### Monitor Progress
```sql
-- Recent pages with language info
SELECT id, url, language, language_confidence, 
       LEFT(content, 100) as preview
FROM pages 
ORDER BY id DESC 
LIMIT 10;

-- Chunk statistics
SELECT COUNT(*) as total_chunks, 
       AVG(LENGTH(original_text)) as avg_chunk_length
FROM page_chunks;
```

## Performance

### Expected Processing Times
- **Language Detection**: ~10-50ms per page
- **Local Embeddings**: ~50-200ms per chunk (MPS on Mac)
- **Sentence Chunking**: ~1-5ms per page

### Resource Usage
- **Memory**: ~500MB for sentence-transformers model
- **Storage**: ~1-2MB per 1000 pages
- **CPU**: Moderate usage for embeddings

## Troubleshooting

### Common Issues

**Connection refused**
```bash
# Ensure PostgreSQL is running
docker ps | grep wet-pg
```

**Module import errors**
```bash
# Run from project root
cd /path/to/wet_pipeline
python3 scripts/script_name.py
```

**Language detection failures**
```bash
# Check langdetect installation
pip install langdetect
```

**Embedding model download**
```bash
# First run will download ~500MB model
# Ensure stable internet connection
```

### Reset Pipeline
```sql
-- Clear all data and restart
DELETE FROM page_chunks;
DELETE FROM pages;
UPDATE wet_paths SET status = 'pending' WHERE status = 'done';
```

## Development

### Running Both Scripts Simultaneously
```bash
# Terminal 1: Data ingestion
python3 scripts/individualFileDownload.py

# Terminal 2: Embedding processing
python3 scripts/embedPages.py
```

### Database GUI
Recommended tools for database management:
- **DBeaver** (Free): `brew install --cask dbeaver-community`
- **TablePlus** (macOS): `brew install --cask tableplus`

## Cost Analysis

### Local vs Cloud Embeddings
| Method | Cost | Speed | Quality | Privacy |
|--------|------|-------|---------|---------|
| **Local (CPU)** | Free | Medium | Good | ✅ |
| **Local (GPU/MPS)** | Free | Fast | Good | ✅ |
| **Google Cloud** | $9.60/8K pages | Fast | High | ❌ |

### Storage Requirements
- **Pages**: ~1-2MB per 1000 pages
- **Chunks**: ~5-10MB per 1000 pages
- **Embeddings**: ~1.5MB per 1000 chunks (384 dimensions)

## Notes

- The pipeline processes pages in batches of 5 by default
- Sentence-based chunking with limited overlap (max 50 tokens) for optimal performance
- Language detection uses the first 1000 characters of each page
- All processing is local - no external API calls required
- Proven to handle 10,000+ pages successfully
- Data directory is git-ignored for security

## License

This project is for educational and research purposes.

