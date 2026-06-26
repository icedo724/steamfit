"""수집 진행상태(SQLite) + 리뷰 parquet 샤드 저장.

설계 의도(대용량·재개 대비):
  - progress.db (SQLite): 가벼운 것 — applist / 게임 메타 / 진행 체크포인트
  - data/reviews/part-*.parquet: 무거운 것 — 리뷰 본문+interaction. 샤드로 누적.
  며칠에 걸쳐 끊겨도 details_status / reviews_status 의 체크포인트로 이어받는다.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

import pandas as pd

from config import DATA, RAW

PROGRESS_DB = DATA / "progress.db"
REVIEWS_DIR = DATA / "reviews"

SCHEMA = """
CREATE TABLE IF NOT EXISTS applist (
    appid INTEGER PRIMARY KEY, name TEXT, last_modified INTEGER
);
CREATE TABLE IF NOT EXISTS games (
    appid INTEGER PRIMARY KEY, name TEXT, type TEXT, is_free INTEGER,
    short_description TEXT, detailed_description TEXT,
    genres TEXT, categories TEXT, release_date TEXT,
    price_cents INTEGER, recommendations_total INTEGER, fetched_at TEXT
);
CREATE TABLE IF NOT EXISTS details_status (
    appid INTEGER PRIMARY KEY, status TEXT, fetched_at TEXT  -- ok | no_data | fail
);
CREATE TABLE IF NOT EXISTS reviews_status (
    appid INTEGER PRIMARY KEY, total_reviews INTEGER, cursor TEXT,
    collected INTEGER DEFAULT 0, done INTEGER DEFAULT 0, updated_at TEXT
);
"""


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def init() -> None:
    for d in (DATA, RAW, REVIEWS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    with get_conn() as c:
        c.executescript(SCHEMA)


@contextmanager
def get_conn():
    c = sqlite3.connect(PROGRESS_DB, timeout=60)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    try:
        yield c
        c.commit()
    except Exception:
        c.rollback()
        raise
    finally:
        c.close()


# ── applist ───────────────────────────────────────────────────────
def save_applist(rows: list[dict]) -> None:
    with get_conn() as c:
        c.executemany(
            "INSERT OR REPLACE INTO applist(appid,name,last_modified) VALUES(?,?,?)",
            [(r["appid"], r.get("name", ""), r.get("last_modified")) for r in rows],
        )


def applist_count() -> int:
    with get_conn() as c:
        return c.execute("SELECT COUNT(*) FROM applist").fetchone()[0]


# ── appdetails ────────────────────────────────────────────────────
def pending_detail_appids(limit: int | None = None) -> list[int]:
    """applist에 있으나 아직 details를 안 받은 appid."""
    q = """
        SELECT a.appid FROM applist a
        LEFT JOIN details_status d ON d.appid = a.appid
        WHERE d.appid IS NULL
        ORDER BY a.appid
    """
    if limit:
        q += f" LIMIT {int(limit)}"
    with get_conn() as c:
        return [r[0] for r in c.execute(q).fetchall()]


def save_game(record: dict, status: str = "ok") -> None:
    with get_conn() as c:
        if record:
            c.execute(
                """INSERT OR REPLACE INTO games(
                    appid,name,type,is_free,short_description,detailed_description,
                    genres,categories,release_date,price_cents,recommendations_total,fetched_at)
                   VALUES(:appid,:name,:type,:is_free,:short_description,:detailed_description,
                    :genres,:categories,:release_date,:price_cents,:recommendations_total,:fetched_at)""",
                {**record, "fetched_at": now()},
            )
        c.execute(
            "INSERT OR REPLACE INTO details_status(appid,status,fetched_at) VALUES(?,?,?)",
            (record.get("appid") if record else None, status, now()),
        )


def mark_detail_status(appid: int, status: str) -> None:
    with get_conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO details_status(appid,status,fetched_at) VALUES(?,?,?)",
            (appid, status, now()),
        )


# ── reviews ───────────────────────────────────────────────────────
def review_targets(min_recommendations: int = 1) -> list[int]:
    """리뷰 수집 대상: 메타 확보된 game 중 인기순(recommendations_total desc),
    아직 done 안 된 것. 인기 게임부터 단계적으로 받기 위함."""
    with get_conn() as c:
        return [r[0] for r in c.execute(
            """
            SELECT g.appid FROM games g
            LEFT JOIN reviews_status rs ON rs.appid = g.appid
            WHERE g.type = 'game'
              AND COALESCE(g.recommendations_total,0) >= ?
              AND COALESCE(rs.done,0) = 0
            ORDER BY COALESCE(g.recommendations_total,0) DESC
            """,
            (min_recommendations,),
        ).fetchall()]


def get_review_status(appid: int) -> sqlite3.Row | None:
    with get_conn() as c:
        return c.execute("SELECT * FROM reviews_status WHERE appid=?", (appid,)).fetchone()


def update_review_status(appid: int, total: int, cursor: str,
                         collected: int, done: int) -> None:
    with get_conn() as c:
        c.execute(
            """INSERT OR REPLACE INTO reviews_status(
                appid,total_reviews,cursor,collected,done,updated_at)
               VALUES(?,?,?,?,?,?)""",
            (appid, total, cursor, collected, done, now()),
        )


def save_review_statuses(rows: list[dict]) -> None:
    """여러 게임의 진행상태를 한 트랜잭션으로 저장 (flush 시점 체크포인트).
    rows: [{appid,total_reviews,cursor,collected,done}, ...]"""
    if not rows:
        return
    with get_conn() as c:
        c.executemany(
            """INSERT OR REPLACE INTO reviews_status(
                appid,total_reviews,cursor,collected,done,updated_at)
               VALUES(:appid,:total_reviews,:cursor,:collected,:done,:updated_at)""",
            [{**r, "updated_at": now()} for r in rows],
        )


class ReviewShardWriter:
    """리뷰 레코드를 버퍼링하다가 일정량마다 parquet 샤드로 flush.

    샤드 인덱스는 기존 part-*.parquet 개수에서 이어받아 재시작에도 안전.
    flush 직후 on_flush 콜백을 호출 → "디스크에 안착한 지점"에서만 진행상태를
    커밋하게 해서, 강제 종료 시에도 SQLite 커서가 parquet보다 앞서지 않게 한다(갭 방지).
    """

    def __init__(self, flush_size: int = 50_000, on_flush=None):
        self.flush_size = flush_size
        self.on_flush = on_flush
        self.buf: list[dict] = []
        existing = sorted(REVIEWS_DIR.glob("part-*.parquet"))
        self.idx = (
            int(existing[-1].stem.split("-")[1]) + 1 if existing else 0
        )

    def add(self, rows: list[dict]) -> None:
        self.buf.extend(rows)
        if len(self.buf) >= self.flush_size:
            self.flush()

    def flush(self) -> None:
        if not self.buf:
            return
        path = REVIEWS_DIR / f"part-{self.idx:05d}.parquet"
        pd.DataFrame(self.buf).to_parquet(path, index=False)   # 1) 먼저 디스크에
        self.idx += 1
        self.buf.clear()
        if self.on_flush:
            self.on_flush()                                    # 2) 그 다음 체크포인트 커밋
