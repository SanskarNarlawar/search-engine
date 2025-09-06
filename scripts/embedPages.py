import logging
import os
import time
from pathlib import Path
from typing import List

import psycopg
import spacy
from spacy.lang.xx import MultiLanguage

from db import get_conn, init_db

# Local embedding model
from sentence_transformers import SentenceTransformer


CHUNK_TARGET_TOKENS = 250
CHUNK_OVERLAP_RATIO = 0.2
EMBED_MODEL = "all-MiniLM-L6-v2"  # Lightweight model (384 dimensions)
MAX_PAGE_CHARS = 50000
EMBEDDING_DIM = 384  # Dimension for all-MiniLM-L6-v2
USE_CHARACTER_CHUNKING = False  # Use sentence-based chunking (reverted)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('embed_pages.log')
        ]
    )


def ensure_spacy_sentencizer():
    logging.info("Initializing spaCy sentencizer...")
    # Use a lightweight blank multilingual pipeline with sentencizer
    nlp = spacy.blank("xx")
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    logging.info("spaCy sentencizer ready")
    return nlp


def sentence_split(nlp, text: str) -> List[str]:
    #logging.debug(f"Splitting text into sentences (length: {len(text)})")
    doc = nlp(text)
    sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
    #logging.debug(f"Split into {len(sentences)} sentences")
    return sentences


def window_sentences(sentences: List[str], target_tokens: int, overlap_ratio: float) -> List[str]:
    #logging.debug(f"Creating chunks from {len(sentences)} sentences (target: {target_tokens} tokens, overlap: {overlap_ratio})")
    chunks: List[str] = []
    current: List[str] = []
    approx_tokens = 0
    token_per_char = 1 / 4  # crude heuristic ~4 chars/token
    max_tokens = target_tokens
    # Limit overlap to reasonable amount (max 50 tokens or 20% of target, whichever is smaller)
    overlap_tokens = min(50, int(target_tokens * overlap_ratio))

    for sent in sentences:
        sent_tokens = max(1, int(len(sent) * token_per_char))
        if approx_tokens + sent_tokens > max_tokens and current:
            chunks.append(" ".join(current).strip())
            # Build overlap context from end of current
            overlap: List[str] = []
            t = 0
            for s in reversed(current):
                t += max(1, int(len(s) * token_per_char))
                overlap.append(s)
                if t >= overlap_tokens:
                    break
            overlap.reverse()
            current = overlap + [sent]
            approx_tokens = sum(max(1, int(len(s) * token_per_char)) for s in current)
        else:
            current.append(sent)
            approx_tokens += sent_tokens
    if current:
        chunks.append(" ".join(current).strip())
    #logging.debug(f"Created {len(chunks)} chunks")
    return chunks


def character_chunk(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Character-based chunking for better web content handling.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
        start = end - overlap
        if start >= len(text):
            break
    return chunks


def init_local_model():
    logging.info("Initializing local sentence-transformers model...")
    try:
        # Load model (will use MPS if available on Mac)
        model = SentenceTransformer(EMBED_MODEL)
        logging.info(f"Model '{EMBED_MODEL}' loaded successfully")
        logging.info(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")
        logging.info("Model loaded successfully")
        return model
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise


def embed_text(model, text: str) -> list[float]:
    #logging.debug(f"Embedding text (length: {len(text)})")
    try:
        # Generate embedding using local model
        embedding = model.encode(text, convert_to_tensor=False)
        #logging.debug(f"Embedding successful (dimension: {len(embedding)})")
        return embedding.tolist()
    except Exception as e:
        #logging.error(f"Embedding failed: {e}")
        raise RuntimeError(f"Failed to embed text: {e}")


def load_pending_pages(limit: int) -> list[tuple[int, str, str]]:
    logging.info(f"Loading up to {limit} pending pages...")
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE pages
            SET embedding_status = 'processing', embedding_error = NULL
            WHERE id IN (
                SELECT id FROM pages
                WHERE embedding_status = 'pending'
                ORDER BY id
                LIMIT %s
            )
            RETURNING id, url, content;
            """,
            (limit,),
        )
        pages = cur.fetchall()
        logging.info(f"Loaded {len(pages)} pages for processing")
        return pages


def mark_done(page_id: int) -> None:
    #logging.info(f"Marking page {page_id} as done")
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE pages SET embedding_status='done' WHERE id=%s",
            (page_id,),
        )
        conn.commit()


def mark_error(page_id: int, msg: str) -> None:
    logging.error(f"Marking page {page_id} as error: {msg}")
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE pages SET embedding_status='error', embedding_error=%s WHERE id=%s",
            (msg[:1000], page_id),
        )
        conn.commit()


def process_page(nlp, model, page_id: int, url: str, content: str) -> None:
    #logging.info(f"Processing page {page_id}: {url[:100]}...")
    if not content or not content.strip():
        logging.warning(f"Page {page_id} has empty content, marking as error")
        mark_error(page_id, "empty content")
        return

    text = content
    #logging.info(f"Processing {len(text)} characters (truncated from {len(content)})")
    
    if USE_CHARACTER_CHUNKING:
        # Use character-based chunking for better web content handling
        chunk_size = CHUNK_TARGET_TOKENS * 4  # ~1000 characters per chunk
        overlap_size = int(chunk_size * CHUNK_OVERLAP_RATIO)  # ~200 characters overlap
        chunks = character_chunk(text, chunk_size, overlap_size)
        #logging.info(f"Created {len(chunks)} chunks using character-based chunking for page {page_id}")
    else:
        # Use sentence-based chunking
        sents = sentence_split(nlp, text)
        chunks = window_sentences(sents, CHUNK_TARGET_TOKENS, CHUNK_OVERLAP_RATIO)
        #logging.info(f"Created {len(chunks)} chunks using sentence-based chunking for page {page_id}")

    with get_conn() as conn, conn.cursor() as cur:
        chunk_idx = 0
        for i, chunk in enumerate(chunks):
            #logging.debug(f"Processing chunk {i+1}/{len(chunks)} for page {page_id}")
            # Use original chunk text directly - no enrichment
            emb = embed_text(model, chunk)
            cur.execute(
                """
                INSERT INTO page_chunks (page_id, chunk_index, original_text, enriched_text, embedding, model)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (page_id, chunk_index) DO NOTHING
                """,
                (page_id, chunk_idx, chunk, chunk, emb, EMBED_MODEL),  # enriched_text = original_text
            )
            chunk_idx += 1
        conn.commit()
    #logging.info(f"Successfully processed page {page_id} with {len(chunks)} chunks")
    mark_done(page_id)


def main() -> None:
    setup_logging()
    logging.info("Starting local embedding process...")
    init_db()
    model = init_local_model()
    nlp = ensure_spacy_sentencizer()

    pages = load_pending_pages(limit=5)  # Back to original batch size
    if not pages:
        logging.info("No pending pages found")
        return
    
    for pid, url, content in pages:
        try:
            #logging.info(f"Starting processing of page {pid}")
            process_page(nlp, model, pid, url, content)
            #logging.info(f"Completed processing of page {pid}")
        except Exception as e:
            logging.error(f"Failed to process page {pid}: {e}", exc_info=True)
            mark_error(pid, str(e))
    
    logging.info(f"Embedding process completed. Processed {len(pages)} pages.")


if __name__ == "__main__":
    main()