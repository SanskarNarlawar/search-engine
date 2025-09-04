Wet pipeline

Prerequisites
- Linux or macOS (or any OS with Docker)
- Python 3.10+

1) Start Postgres (Docker)
Run once to start a local Postgres:
```bash
docker run --name wet-pg \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=wet_pipeline \
  -p 5432:5432 -d postgres:16
```

2) Create .env
Create a file `.env` in the project root:
```env
PGHOST=localhost
PGPORT=5432
PGDATABASE=wet_pipeline
PGUSER=postgres
PGPASSWORD=postgres
```

3) Setup Python env
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

4) Insert wet paths into DB
Streams the Common Crawl wet.paths.gz and inserts into `wet_paths` table.
```bash
python3 scripts/filePathDownload.py
```

5) Process WET files into pages
Fetches up to 5 pending WET paths, streams each `.gz`, and writes pages to `pages` table.
```bash
python3 scripts/individualFileDownload.py
```

Inspect the database (psql)
```bash
psql "postgresql://postgres:postgres@localhost:5432/wet_pipeline"
-- list tables
\dt
-- view recent paths
SELECT id, wet_path, status, updated_at FROM wet_paths ORDER BY id DESC LIMIT 10;
-- view pages sample
SELECT id, LEFT(url, 80) url FROM pages ORDER BY id DESC LIMIT 10;
```

Notes
- The `individualFileDownload.py` script marks each `wet_paths` row as `processing` and then `done` or `error`.
- IDs may have gaps (Postgres sequences aren’t transactional).
- To stop a running script, press Ctrl+C.

Troubleshooting
- Connection refused: ensure Postgres is running (see step 1).
- Module import errors: run from project root: `python3 scripts/...`.
- Data directory is git-ignored; if already committed, run: `git rm -r --cached data`.

