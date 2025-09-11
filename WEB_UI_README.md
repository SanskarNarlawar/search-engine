# WET Pipeline Web UI

A modern web interface for semantic search on the WET pipeline database.

## Features

- 🔍 **Semantic Search**: Natural language queries powered by AI embeddings
- ⚡ **Fast Performance**: HNSW vector index for lightning-fast similarity search
- 📊 **Real-time Statistics**: Database stats and content analysis
- 🎨 **Modern UI**: Beautiful, responsive design with Bootstrap 5
- 🔧 **Configurable**: Adjustable similarity thresholds and result limits
- 📱 **Mobile Friendly**: Works on desktop, tablet, and mobile devices

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Web UI
```bash
python3 start_web_ui.py
```

### 3. Open in Browser
Navigate to: http://localhost:5001

## Usage

### Search Interface
- Enter your search query in natural language
- Adjust similarity threshold (0.1 = more similar, 1.0 = less similar)
- Set maximum number of results (10-100)
- Click "Search" to find similar content

### Example Queries
- "machine learning algorithms"
- "climate change research"
- "web development frameworks"
- "artificial intelligence applications"

### Results Display
- **Similarity Score**: How similar the result is to your query
- **Source URL**: Original webpage where content was found
- **Language**: Detected language and confidence
- **Content Preview**: First 500 characters of the matching text
- **Metadata**: Chunk ID, page ID, and creation date

## API Usage

### Search Endpoint
```bash
curl -X POST http://localhost:5001/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "search_text": "machine learning",
    "threshold": 0.5,
    "limit": 20
  }'
```

### Response Format
```json
{
  "results": [
    {
      "id": 123,
      "page_id": 456,
      "chunk_index": 2,
      "original_text": "Machine learning is a subset of...",
      "url": "https://example.com/article",
      "language": "en",
      "language_confidence": 0.95,
      "cosine_distance": 0.234,
      "created_at": "2024-01-15T10:30:00"
    }
  ],
  "query_time": 0.123,
  "total_time": 0.456,
  "result_count": 15,
  "search_text": "machine learning",
  "threshold": 0.5
}
```

## Configuration

### Environment Variables
Create a `.env` file in the project root:
```env
PGHOST=localhost
PGPORT=5432
PGDATABASE=wet_pipeline
PGUSER=postgres
PGPASSWORD=postgres
```

### Flask Configuration
The app runs on:
- **Host**: 0.0.0.0 (accessible from any IP)
- **Port**: 5000
- **Debug Mode**: Enabled (auto-reload on changes)

## File Structure

```
wet_pipeline/
├── app.py                 # Main Flask application
├── start_web_ui.py       # Startup script with checks
├── templates/            # HTML templates
│   ├── base.html         # Base template with styling
│   ├── index.html        # Search interface
│   ├── search_results.html # Results display
│   └── stats.html        # Database statistics
├── static/               # Static assets (CSS, JS)
└── scripts/              # Backend scripts
    ├── db.py            # Database functions
    ├── semanticSearch.py # Search logic
    └── embedPages.py    # Embedding generation
```

## Troubleshooting

### Common Issues

**"No embeddings found"**
- Run `python3 scripts/embedPages.py` to generate embeddings
- Ensure you have processed some pages first

**"Database connection failed"**
- Check PostgreSQL is running
- Verify connection settings in `.env` file
- Run `python3 scripts/db.py` to initialize database

**"Model loading failed"**
- Ensure sentence-transformers is installed
- Check internet connection (first run downloads model)
- Verify sufficient disk space (~500MB for model)

**"Port 5000 already in use"**
- Kill existing process: `lsof -ti:5000 | xargs kill`
- Or change port in `app.py`: `app.run(port=5001)`

### Performance Tips

1. **Use HNSW Index**: Ensure vector index is created for fast searches
2. **Adjust Threshold**: Lower values (0.3-0.5) for more similar results
3. **Limit Results**: Use smaller limits (10-20) for faster queries
4. **Batch Processing**: Process embeddings in batches for better performance

## Development

### Adding New Features
1. Modify `app.py` for new routes
2. Update templates in `templates/` directory
3. Add static assets to `static/` directory
4. Test with `python3 start_web_ui.py`

### Debugging
- Enable debug logging: Set `logging.basicConfig(level=logging.DEBUG)`
- Check browser console for JavaScript errors
- Monitor Flask logs for backend issues
- Use `EXPLAIN ANALYZE` for query optimization

## Security Notes

- Change `app.secret_key` in production
- Use environment variables for sensitive data
- Consider authentication for production use
- Validate and sanitize user inputs
- Use HTTPS in production

## License

This project is for educational and research purposes.
