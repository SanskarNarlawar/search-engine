#!/usr/bin/env python3
"""
Semantic Search Script for WET Pipeline

This script performs semantic similarity search on the page_chunks table
using the same all-MiniLM-L6-v2 model used for generating embeddings.
"""

import logging
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional

import psycopg
from sentence_transformers import SentenceTransformer

from db import get_conn, init_db

# Use the same model as the embedding pipeline
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('semantic_search.log')
        ]
    )


def load_model() -> SentenceTransformer:
    """Load the sentence transformer model."""
    logging.info(f"Loading model: {EMBED_MODEL}")
    start_time = time.time()
    try:
        model = SentenceTransformer(EMBED_MODEL)
        load_time = time.time() - start_time
        logging.info(f"Model loaded successfully in {load_time:.2f} seconds. Embedding dimension: {model.get_sentence_embedding_dimension()}")
        return model
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise


def generate_embedding(model: SentenceTransformer, text: str) -> List[float]:
    """Generate embedding for the given text."""
    logging.info(f"Generating embedding for text (length: {len(text)})")
    start_time = time.time()
    try:
        embedding = model.encode(text, convert_to_tensor=False)
        embed_time = time.time() - start_time
        logging.info(f"Embedding generated successfully in {embed_time:.3f} seconds (dimension: {len(embedding)})")
        return embedding.tolist()
    except Exception as e:
        logging.error(f"Failed to generate embedding: {e}")
        raise


def check_vector_index():
    """Check if vector index exists and get index information."""
    with get_conn() as conn, conn.cursor() as cur:
        try:
            # Check for HNSW index
            cur.execute("""
                SELECT indexname, indexdef 
                FROM pg_indexes 
                WHERE tablename = 'page_chunks' 
                AND indexname LIKE '%embedding%'
                ORDER BY indexname;
            """)
            indexes = cur.fetchall()
            
            if indexes:
                logging.info("Vector indexes found:")
                for index_name, index_def in indexes:
                    logging.info(f"  - {index_name}: {index_def}")
                return True
            else:
                logging.warning("No vector indexes found on page_chunks table")
                return False
        except Exception as e:
            logging.error(f"Failed to check indexes: {e}")
            return False

def search_similar_chunks(
    search_text: str, 
    model: SentenceTransformer, 
    threshold: float = 0.5, 
    limit: int = 20
) -> List[Tuple]:
    """
    Search for similar chunks using semantic similarity with HNSW index.
    
    Args:
        search_text: Text to search for
        model: Sentence transformer model
        threshold: Cosine distance threshold (lower = more similar)
        limit: Maximum number of results to return
    
    Returns:
        List of tuples containing search results (may include multiple chunks per page)
        Note: The web app deduplicates results to show only the best chunk per page
    """
    logging.info(f"Searching for similar chunks (threshold: {threshold}, limit: {limit})")
    search_start_time = time.time()
    
    # Check if vector index exists
    has_index = check_vector_index()
    if not has_index:
        logging.warning("No vector index found. Query may be slow.")
    
    # Generate embedding for search text
    search_embedding = generate_embedding(model, search_text)
    embedding_str = '[' + ','.join(map(str, search_embedding)) + ']'
    
    # Execute optimized semantic search query (uses HNSW index automatically)
    query_start_time = time.time()
    with get_conn() as conn, conn.cursor() as cur:
        # Use EXPLAIN to verify index usage (optional, for debugging)
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            cur.execute("""
                EXPLAIN (ANALYZE, BUFFERS) 
                SELECT pc.id, pc.embedding <=> %s as distance
                FROM page_chunks pc
                WHERE pc.embedding <=> %s < %s
                ORDER BY pc.embedding <=> %s
                LIMIT %s
            """, (embedding_str, embedding_str, threshold, embedding_str, limit))
            explain_result = cur.fetchall()
            logging.debug("Query execution plan:")
            for row in explain_result:
                logging.debug(f"  {row[0]}")
        
        # Main similarity search query
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
        
        logging.info(f"Database query completed in {query_time:.3f} seconds")
        logging.info(f"Found {len(results)} similar chunks in {total_time:.3f} seconds total")
        
        # Log performance metrics
        if results:
            avg_distance = sum(row[8] for row in results) / len(results)
            logging.info(f"Average similarity distance: {avg_distance:.4f}")
        
        return results


def search_text_chunks(
    search_terms: List[str], 
    limit: int = 20
) -> List[Tuple]:
    """
    Search for chunks using text matching (ILIKE).
    
    Args:
        search_terms: List of terms to search for
        limit: Maximum number of results to return
    
    Returns:
        List of tuples containing search results
    """
    logging.info(f"Searching for text chunks with terms: {search_terms}")
    search_start_time = time.time()
    
    # Build ILIKE conditions
    conditions = " OR ".join([f"pc.original_text ILIKE %s" for _ in search_terms])
    search_patterns = [f"%{term}%" for term in search_terms]
    
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT 
                pc.id,
                pc.page_id,
                pc.chunk_index,
                pc.original_text,
                pc.enriched_text,
                p.url,
                p.language,
                p.language_confidence,
                0.0 as cosine_distance,
                pc.created_at
            FROM page_chunks pc
            JOIN pages p ON pc.page_id = p.id
            WHERE {conditions}
            ORDER BY pc.created_at DESC
            LIMIT %s
            """,
            search_patterns + [limit]
        )
        results = cur.fetchall()
        search_time = time.time() - search_start_time
        logging.info(f"Found {len(results)} text matches in {search_time:.3f} seconds")
        return results


def display_results(results: List[Tuple], search_type: str = "semantic"):
    """Display search results in a formatted way."""
    if not results:
        print(f"No {search_type} search results found.")
        return
    
    print(f"\n=== {search_type.upper()} SEARCH RESULTS ===")
    print(f"Found {len(results)} results\n")
    
    for i, row in enumerate(results, 1):
        print(f"--- Result {i} ---")
        print(f"Chunk ID: {row[0]}")
        print(f"Page ID: {row[1]}")
        print(f"Chunk Index: {row[2]}")
        print(f"URL: {row[5]}")
        print(f"Language: {row[6]} (confidence: {row[7]:.2f})")
        if search_type == "semantic":
            print(f"Cosine Distance: {row[8]:.4f}")
        print(f"Created: {row[9]}")
        print(f"Text Preview: {row[3][:300]}...")
        print("-" * 80)


def test_index_performance():
    """Test the performance of the vector index."""
    logging.info("Testing vector index performance...")
    
    with get_conn() as conn, conn.cursor() as cur:
        try:
            # Get some basic stats
            cur.execute("""
                SELECT 
                    COUNT(*) as total_chunks,
                    COUNT(CASE WHEN embedding IS NOT NULL THEN 1 END) as chunks_with_embeddings
                FROM page_chunks;
            """)
            stats = cur.fetchone()
            total_chunks, chunks_with_embeddings = stats
            
            logging.info(f"Database stats: {total_chunks} total chunks, {chunks_with_embeddings} with embeddings")
            
            if chunks_with_embeddings == 0:
                logging.warning("No embeddings found. Run embedPages.py first.")
                return False
            
            # Test a simple similarity query
            test_start = time.time()
            cur.execute("""
                SELECT COUNT(*) 
                FROM page_chunks 
                WHERE embedding IS NOT NULL 
                LIMIT 1;
            """)
            test_time = time.time() - test_start
            logging.info(f"Index test query completed in {test_time:.3f} seconds")
            
            return True
            
        except Exception as e:
            logging.error(f"Index performance test failed: {e}")
            return False

def main():
    """Main function to run semantic search."""
    setup_logging()
    logging.info("Starting semantic search...")
    total_start_time = time.time()
    
    # Initialize database
    db_start_time = time.time()
    init_db()
    db_time = time.time() - db_start_time
    logging.info(f"Database initialization completed in {db_time:.3f} seconds")
    
    # Test index performance
    index_ok = test_index_performance()
    if not index_ok:
        print("⚠️  No embeddings found. Run embedPages.py first to generate embeddings.")
        return
    
    # Load model
    model_start_time = time.time()
    model = load_model()
    model_time = time.time() - model_start_time
    
    # Example search text (you can modify this)
    search_text = """Sri Sathya Sai Baba Website - Teachings, Experiences, Miracles
Home | Thought for the Day | Sai Inspires
Articles | Avatar | Bhajans | Experiences | Messages | Miracles | Prayers | Quotes | Stories | Service | Teachings
SITE CONTENT Today is
SAI WEBSITES
Articles
Avatar
Bhajans
Discourses
Download
Experiences
Messages
Miracles
Pictures
Prayers
Quotes
Reports
Service Activities
Stories
Teachings
Videoclips
Darshan News
A monthly e-journal
Sanathana Sarathi
Subscribe online
Sri Sathya Sai Speaks"""
    
    # Check if custom search text provided as command line argument
    if len(sys.argv) > 1:
        search_text = " ".join(sys.argv[1:])
        logging.info(f"Using custom search text: {search_text[:100]}...")
    
    try:
        # Perform semantic search
        print("Performing semantic similarity search...")
        search_start_time = time.time()
        semantic_results = search_similar_chunks(
            search_text=search_text,
            model=model,
            threshold=0.5,  # Adjust this value (lower = more similar)
            limit=20
        )
        search_time = time.time() - search_start_time
        
        display_results(semantic_results, "semantic")
        
        # Also perform text search for comparison
        # print("\nPerforming text-based search...")
        # text_terms = ["sai baba", "sathya sai", "teachings", "miracles", "bhajans"]
        # text_results = search_text_chunks(text_terms, limit=10)
        # display_results(text_results, "text")
        
        # Display total execution time
        total_time = time.time() - total_start_time
        print(f"\n=== TIMING SUMMARY ===")
        print(f"Database initialization: {db_time:.3f} seconds")
        print(f"Model loading: {model_time:.3f} seconds")
        print(f"Search execution: {search_time:.3f} seconds")
        print(f"Total execution time: {total_time:.3f} seconds")
        print(f"Results found: {len(semantic_results)}")
        
    except Exception as e:
        logging.error(f"Search failed: {e}")
        raise


if __name__ == "__main__":
    main()
