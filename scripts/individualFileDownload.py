import gzip
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Iterable, Iterator, Tuple
from urllib.parse import urljoin
from urllib.request import urlopen


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


def main() -> None:
    # Ensure DB and tables
    init_db()

    # Hardcoded number of WET files to process
    limit = 5

    processed_files = 0
    batch = load_batch_of_pending(limit)
    for wet_path_id, wet_path in batch:
        full_url = urljoin(BASE_URL, wet_path.lstrip("/"))
        print(f"[process] {full_url}")
        try:
            with open_wet_gzip_stream(full_url) as stream, get_conn() as conn, conn.cursor() as cur:
                for uri, payload in parse_warc_records(stream):
                    text = payload.decode("utf-8", errors="replace")
                    # Insert page row
                    cur.execute(
                        """
                        INSERT INTO pages (wet_path_id, url, content)
                        VALUES (%s, %s, %s)
                        """,
                        (wet_path_id, uri, text),
                    )
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


