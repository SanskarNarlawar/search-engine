from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import psycopg


# Resolve repo root and .env location
REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = REPO_ROOT / ".env"


def load_env_if_present() -> None:
    if ENV_FILE.exists():
        try:
            for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                os.environ.setdefault(key, value)
        except Exception:
            # Ignore malformed .env
            pass


def get_conn() -> psycopg.Connection:
    load_env_if_present()
    host = os.getenv("PGHOST", "localhost")
    port = int(os.getenv("PGPORT", "5432"))
    dbname = os.getenv("PGDATABASE", "wet_pipeline")
    user = os.getenv("PGUSER", "postgres")
    password = os.getenv("PGPASSWORD", "postgres")

    return psycopg.connect(host=host, port=port, dbname=dbname, user=user, password=password)


def init_db() -> None:
    with get_conn() as conn, conn.cursor() as cur:
        # Enable pgvector (ignore if not permitted)
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        except Exception:
            pass
        # wet_paths table: tracks each WET path and processing status
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS wet_paths (
                id SERIAL PRIMARY KEY,
                wet_path TEXT UNIQUE NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','processing','done','error')),
                error TEXT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )

        # pages table: one row per extracted page
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS pages (
                id BIGSERIAL PRIMARY KEY,
                wet_path_id INTEGER NULL REFERENCES wet_paths(id) ON DELETE SET NULL,
                url TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )
        # Add embedding_status and embedding_error columns if missing
        cur.execute(
            """
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'pages' AND column_name = 'embedding_status'
                ) THEN
                    ALTER TABLE pages
                    ADD COLUMN embedding_status TEXT NOT NULL DEFAULT 'pending'
                    CHECK (embedding_status IN ('pending','processing','done','error'));
                END IF;
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'pages' AND column_name = 'embedding_error'
                ) THEN
                    ALTER TABLE pages ADD COLUMN embedding_error TEXT NULL;
                END IF;
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'pages' AND column_name = 'language'
                ) THEN
                    ALTER TABLE pages ADD COLUMN language VARCHAR(10) DEFAULT 'unknown';
                END IF;
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'pages' AND column_name = 'language_confidence'
                ) THEN
                    ALTER TABLE pages ADD COLUMN language_confidence FLOAT DEFAULT 0.0;
                END IF;
            END$$;
            """
        )

        # page_chunks table with pgvector embedding
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS page_chunks (
                id BIGSERIAL PRIMARY KEY,
                page_id BIGINT NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                original_text TEXT NOT NULL,
                enriched_text TEXT NOT NULL,
                embedding vector(384) NOT NULL,
                model TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(page_id, chunk_index)
            );
            """
        )

        conn.commit()


