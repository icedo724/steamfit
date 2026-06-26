"""DB 연결 추상화 — 로컬 DuckDB 파일 / 클라우드 MotherDuck 동일 코드.

MOTHERDUCK_TOKEN 환경변수가 있으면 클라우드(MotherDuck), 없으면 로컬 파일.
→ 로컬에서 전부 개발·검증 후, 토큰만 주면 코드 변경 없이 클라우드로 전환.
"""
from __future__ import annotations

import os
from pathlib import Path

import duckdb

try:
    from dotenv import load_dotenv
    load_dotenv()  # 로컬 .env (GH Actions에선 파일 없음 → no-op, secrets 사용)
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = (Path(__file__).resolve().parent / "schema.sql").read_text(encoding="utf-8")
LOCAL_PATH = os.getenv("DUCKDB_PATH", str(ROOT / "data" / "steam.duckdb"))
MD_DATABASE = os.getenv("MD_DATABASE", "steam")


def connect():
    """DuckDB 커넥션 반환 (MotherDuck 또는 로컬). 스키마 보장."""
    token = os.getenv("MOTHERDUCK_TOKEN")
    if token:
        con = duckdb.connect("md:")  # MOTHERDUCK_TOKEN 자동 사용
        con.execute(f"CREATE DATABASE IF NOT EXISTS {MD_DATABASE}")
        con.execute(f"USE {MD_DATABASE}")
        where = f"MotherDuck:{MD_DATABASE}"
    else:
        Path(LOCAL_PATH).parent.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect(LOCAL_PATH)
        where = f"local:{LOCAL_PATH}"
    con.execute(SCHEMA)
    return con, where


def get_meta(con, key, default=None):
    row = con.execute("SELECT value FROM meta WHERE key=?", [key]).fetchone()
    return row[0] if row else default


def set_meta(con, key, value):
    con.execute(
        "INSERT INTO meta(key,value) VALUES(?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        [key, str(value)],
    )
