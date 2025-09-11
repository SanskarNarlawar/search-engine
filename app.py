#!/usr/bin/env python3
"""
WET Pipeline Web UI

A Flask web application for semantic search on the WET pipeline database.
Allows users to search for similar content using natural language queries.
"""

import logging
import time
from pathlib import Path
from typing import List, Tuple, Optional

import psycopg
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from sentence_transformers import SentenceTransformer

# Import from scripts
import sys
sys.path.append(str(Path(__file__).parent / "scripts"))
from db import get_conn, init_db

# Flask app configuration
app = Flask(__name__)
app.secret_key = 'wet_pipeline_secret_key_2024'

# Global variables for model and configuration
model = None
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('web_ui.log')
        ]
    )

def load_model():
    """Load the sentence transformer model."""
    global model
    if model is None:
        logging.info(f"Loading model: {EMBED_MODEL}")
        start_time = time.time()
        try:
            model = SentenceTransformer(EMBED_MODEL)
            load_time = time.time() - start_time
            logging.info(f"Model loaded successfully in {load_time:.2f} seconds")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise
    return model

def generate_embedding(text: str) -> List[float]:
    """Generate embedding for the given text."""
    model = load_model()
    try:
        embedding = model.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    except Exception as e:
        logging.error(f"Failed to generate embedding: {e}")
        raise

def search_similar_chunks(
    search_text: str, 
    threshold: float = 0.5, 
    limit: int = 20
) -> Tuple[List[Tuple], float, float]:
    """
    Search for similar chunks using semantic similarity.
    
    Returns:
        Tuple of (results, query_time, total_time)
    """
    search_start_time = time.time()
    
    # Generate embedding for search text
    search_embedding = generate_embedding(search_text)
    embedding_str = '[' + ','.join(map(str, search_embedding)) + ']'
    
    # Execute semantic search query
    query_start_time = time.time()
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT 
                pc.id,
                pc.page_id,
                pc.chunk_index,
                pc.original_text,
                pc.enriched_text,
                p.url,
                p.language,
                p.language_confidence,
                pc.embedding <=> %s as cosine_distance,
                pc.created_at
            FROM page_chunks pc
            JOIN pages p ON pc.page_id = p.id
            WHERE pc.embedding <=> %s < %s
            ORDER BY pc.embedding <=> %s
            LIMIT %s
            """,
            (embedding_str, embedding_str, threshold, embedding_str, limit)
        )
        results = cur.fetchall()
        query_time = time.time() - query_start_time
        total_time = time.time() - search_start_time
        
        logging.info(f"Found {len(results)} similar chunks in {total_time:.3f} seconds")
        return results, query_time, total_time

def get_database_stats():
    """Get database statistics."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT 
                COUNT(*) as total_chunks,
                COUNT(CASE WHEN embedding IS NOT NULL THEN 1 END) as chunks_with_embeddings,
                COUNT(DISTINCT page_id) as unique_pages,
                COUNT(DISTINCT p.url) as unique_urls
            FROM page_chunks pc
            LEFT JOIN pages p ON pc.page_id = p.id;
        """)
        stats = cur.fetchone()
        
        cur.execute("""
            SELECT 
                language,
                COUNT(*) as count
            FROM pages 
            WHERE language IS NOT NULL
            GROUP BY language
            ORDER BY count DESC
            LIMIT 10;
        """)
        language_stats = cur.fetchall()
        
        return stats, language_stats

@app.route('/')
def index():
    """Main search page."""
    try:
        stats, language_stats = get_database_stats()
        return render_template('index.html', 
                             stats=stats, 
                             language_stats=language_stats)
    except Exception as e:
        logging.error(f"Error loading index: {e}")
        flash(f"Error loading page: {str(e)}", 'error')
        return render_template('index.html', 
                             stats=(0, 0, 0, 0), 
                             language_stats=[])

@app.route('/search', methods=['POST'])
def search():
    """Handle search requests."""
    try:
        search_text = request.form.get('search_text', '').strip()
        threshold = float(request.form.get('threshold', 0.5))
        limit = int(request.form.get('limit', 20))
        
        if not search_text:
            flash('Please enter a search query.', 'warning')
            return redirect(url_for('index'))
        
        if limit > 100:
            limit = 100  # Cap at 100 results
        
        # Perform search
        results, query_time, total_time = search_similar_chunks(
            search_text, threshold, limit
        )
        
        # Format results for display
        formatted_results = []
        for row in results:
            formatted_results.append({
                'id': row[0],
                'page_id': row[1],
                'chunk_index': row[2],
                'original_text': row[3],
                'enriched_text': row[4],
                'url': row[5],
                'language': row[6],
                'language_confidence': row[7],
                'cosine_distance': row[8],
                'created_at': row[9]
            })
        
        # Calculate average similarity
        avg_distance = sum(r['cosine_distance'] for r in formatted_results) / len(formatted_results) if formatted_results else 0
        
        return render_template('search_results.html',
                             search_text=search_text,
                             threshold=threshold,
                             limit=limit,
                             results=formatted_results,
                             query_time=query_time,
                             total_time=total_time,
                             avg_distance=avg_distance,
                             result_count=len(formatted_results))
        
    except Exception as e:
        logging.error(f"Search error: {e}")
        flash(f"Search failed: {str(e)}", 'error')
        return redirect(url_for('index'))

@app.route('/api/search', methods=['POST'])
def api_search():
    """API endpoint for search requests."""
    try:
        data = request.get_json()
        search_text = data.get('search_text', '').strip()
        threshold = float(data.get('threshold', 0.5))
        limit = int(data.get('limit', 20))
        
        if not search_text:
            return jsonify({'error': 'Search text is required'}), 400
        
        if limit > 100:
            limit = 100
        
        # Perform search
        results, query_time, total_time = search_similar_chunks(
            search_text, threshold, limit
        )
        
        # Format results for JSON response
        formatted_results = []
        for row in results:
            formatted_results.append({
                'id': row[0],
                'page_id': row[1],
                'chunk_index': row[2],
                'original_text': row[3][:500] + '...' if len(row[3]) > 500 else row[3],
                'url': row[5],
                'language': row[6],
                'language_confidence': row[7],
                'cosine_distance': round(row[8], 4),
                'created_at': row[9].isoformat() if row[9] else None
            })
        
        return jsonify({
            'results': formatted_results,
            'query_time': round(query_time, 3),
            'total_time': round(total_time, 3),
            'result_count': len(formatted_results),
            'search_text': search_text,
            'threshold': threshold
        })
        
    except Exception as e:
        logging.error(f"API search error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/stats')
def stats():
    """Database statistics page."""
    try:
        stats, language_stats = get_database_stats()
        return render_template('stats.html', 
                             stats=stats, 
                             language_stats=language_stats)
    except Exception as e:
        logging.error(f"Error loading stats: {e}")
        flash(f"Error loading statistics: {str(e)}", 'error')
        return redirect(url_for('index'))

if __name__ == '__main__':
    setup_logging()
    logging.info("Starting WET Pipeline Web UI...")
    
    # Initialize database
    init_db()
    
    # Load model
    load_model()
    
    print("\n🚀 WET Pipeline Web UI is starting...")
    print("📊 Database initialized")
    print("🤖 AI model loaded")
    print("🌐 Web interface ready")
    print("\nAccess the UI at: http://localhost:5001")
    print("Press Ctrl+C to stop the server\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)
