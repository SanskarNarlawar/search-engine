import gzip
import shutil
import io
from pathlib import Path
from urllib.request import urlopen

from db import get_conn, init_db


COMMONCRAWL_WET_PATHS_URL = "https://data.commoncrawl.org/crawl-data/CC-MAIN-2024-18/wet.paths.gz"
# Always place outputs in repo root's `data/` (parent of `scripts/`)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DOWNLOAD_FILENAME = DATA_DIR / "wet.paths.gz"
OUTPUT_FILENAME = DATA_DIR / "wet.paths"


def download_file_streaming(url: str, destination: Path) -> None:
    """Download a file from url to destination using streaming to avoid high memory usage."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response, open(destination, "wb") as out_file:
        shutil.copyfileobj(response, out_file)


def gunzip_file(source_gz: Path, destination: Path) -> None:
    """Decompress a .gz file into destination path."""
    with gzip.open(source_gz, "rb") as f_in, open(destination, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


def iter_wet_paths_streaming(url: str):
    """Yield wet paths by streaming and decompressing the remote gz file."""
    with urlopen(url) as response:
        with gzip.GzipFile(fileobj=io.BufferedReader(response)) as gz:
            for raw in gz:
                line = raw.decode("utf-8", errors="replace").strip()
                if line:
                    yield line


def main() -> None:
    # Ensure DB exists
    init_db()

    # Stream, decompress, and insert wet paths directly without saving to disk
    print("Streaming wet.paths.gz and inserting into database ...")
    rows = 0
    with get_conn() as conn, conn.cursor() as cur:
        for wet_path in iter_wet_paths_streaming(COMMONCRAWL_WET_PATHS_URL):
            cur.execute(
                """
                INSERT INTO wet_paths (wet_path, status)
                VALUES (%s, 'pending')
                ON CONFLICT (wet_path) DO NOTHING
                """,
                (wet_path,),
            )
            rows += cur.rowcount
        conn.commit()
    print(f"Inserted {rows} new wet paths.")

    print("Done.")


if __name__ == "__main__":
    main()