import gzip
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Iterable, Iterator, Tuple
from urllib.parse import urljoin
from urllib.request import urlopen

# Language detection
from langdetect import detect, DetectorFactory

# Set seed for consistent language detection
DetectorFactory.seed = 0

BASE_URL = "https://data.commoncrawl.org/"

# Resolve repo root: parent of this script's directory
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
from db import get_conn, init_db

RECORDS_DIR = DATA_DIR / "records"  # no longer used for new output; kept if needed
WET_PATHS_FILE = DATA_DIR / "wet.paths"  # only used to bootstrap DB; DB is source of truth now
PROCESSED_LIST_FILE = DATA_DIR / "processed_wet.txt"  # deprecated; DB status used now


def iter_wet_paths(file_path: Path) -> Iterable[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            path = line.strip()
            if not path:
                continue
            if path.startswith("#"):
                continue
            yield path


def open_wet_gzip_stream(url: str) -> io.BufferedReader:
    """Open a remote .gz file as a binary stream, decompressed on the fly."""
    resp = urlopen(url)
    # Wrap response in a GzipFile, return a buffered reader interface
    return gzip.GzipFile(fileobj=resp)  # type: ignore[arg-type]


def parse_warc_records(stream: io.BufferedReader) -> Iterator[Tuple[str, bytes]]:
    """Yield (uri, payload_bytes) for each WARC record in a decompressed WET binary stream."""
    while True:
        line = stream.readline()
        if not line:
            break
        if not line.startswith(b"WARC/"):
            # Skip until the start of a record header
            continue

        headers: dict[str, str] = {}
        # Read header lines until blank line
        while True:
            hline = stream.readline()
            if not hline:
                break
            # Blank line indicates end of headers
            if hline in (b"\n", b"\r\n"):
                break
            try:
                key, value = hline.split(b":", 1)
                headers[key.decode("utf-8", "replace").strip()] = value.decode("utf-8", "replace").strip()
            except ValueError:
                # Malformed header line; skip
                continue

        uri = headers.get("WARC-Target-URI")
        content_length_str = headers.get("Content-Length", "0")
        try:
            content_length = int(content_length_str)
        except ValueError:
            content_length = 0

        payload = stream.read(content_length) if content_length > 0 else b""

        # Consume possible trailing newlines between records
        _ = stream.readline()

        if uri is not None:
            yield uri, payload


def hashed_record_path(url: str) -> Path:
    """Return a stable hashed path under data/records for the given URL."""
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()
    subdir = RECORDS_DIR / h[:2] / h[2:4]
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir / f"{h}.json"


def load_batch_of_pending(limit: int) -> list[tuple[int, str]]:
    """Fetch a batch of pending wet_paths and mark them processing."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, wet_path
            FROM wet_paths
            WHERE status = 'pending'
            ORDER BY id
            LIMIT %s
            FOR UPDATE SKIP LOCKED
            """,
            (limit,),
        )
        rows = cur.fetchall()
        ids = [r[0] for r in rows]
        if ids:
            cur.execute(
                """
                UPDATE wet_paths
                SET status = 'processing', updated_at = NOW()
                WHERE id = ANY(%s)
                """,
                (ids,),
            )
        conn.commit()
        return [(r[0], r[1]) for r in rows]


def detect_language(text: str) -> tuple[str, float]:
    """
    Detect the language of the given text.
    Returns (language_code, confidence) tuple.
    """
    try:
        # Use first 1000 characters for language detection (faster and more reliable)
        sample_text = text[:1000].strip()
        if len(sample_text) < 10:  # Too short for reliable detection
            return 'unknown', 0.0
        
        # Detect language
        detected_lang = detect(sample_text)
        
        # Get confidence (langdetect doesn't provide confidence directly)
        # We'll use a simple heuristic based on text length and character diversity
        confidence = min(0.9, len(sample_text) / 1000.0)
        
        return detected_lang, confidence
    except Exception as e:
        #print(f"[warning] Language detection failed: {e}")
        return 'unknown', 0.0


def is_english(text: str, min_confidence: float = 0.6) -> bool:
    """
    Check if the text is English with sufficient confidence.
    """
    language, confidence = detect_language(text)
    return language == 'en' and confidence >= min_confidence


def main() -> None:
    # Ensure DB and tables
    init_db()

    # Hardcoded number of WET files to process
    limit = 100

    processed_files = 0
    batch = load_batch_of_pending(limit)
    for wet_path_id, wet_path in batch:
        full_url = urljoin(BASE_URL, wet_path.lstrip("/"))
        print(f"[process] {full_url}")
        try:
            with open_wet_gzip_stream(full_url) as stream, get_conn() as conn, conn.cursor() as cur:
                english_pages = 0
                total_pages = 0
                
                for uri, payload in parse_warc_records(stream):
                    text = payload.decode("utf-8", errors="replace")
                    total_pages += 1
                    
                    # Detect language and only process English pages
                    if is_english(text):
                        language, confidence = detect_language(text)
                        # Insert English page row with language info
                        cur.execute(
                            """
                            INSERT INTO pages (wet_path_id, url, content, language, language_confidence)
                            VALUES (%s, %s, %s, %s, %s)
                            """,
                            (wet_path_id, uri, text, language, confidence),
                        )
                        english_pages += 1
                    #else:
                        # Skip non-English pages (don't insert into pages table)
                        #language, confidence = detect_language(text)
                        #print(f"[skip] Non-English page: {uri[:100]}... (lang: {language}, conf: {confidence:.2f})")
                
                print(f"[process] {full_url} - Processed {english_pages}/{total_pages} English pages")
                # Mark wet_path as done
                cur.execute(
                    """
                    UPDATE wet_paths
                    SET status = 'done', updated_at = NOW(), error = NULL
                    WHERE id = %s
                    """,
                    (wet_path_id,),
                )
                conn.commit()
        except Exception as e:
            print(f"[error] Failed processing {full_url}: {e}", file=sys.stderr)
            with get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE wet_paths
                    SET status = 'error', updated_at = NOW(), error = %s
                    WHERE id = %s
                    """,
                    (str(e)[:1000], wet_path_id),
                )
                conn.commit()

        processed_files += 1

    print(f"Done. Processed {processed_files} WET files into DB.")


if __name__ == "__main__":
    main()


