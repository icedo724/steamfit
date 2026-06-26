"""원시 수집물(data/) → 학습용 데이터셋(dataset/) 가공.

읽기 전용으로 data/를 참조하고, 산출물은 dataset/에만 쓴다(원시 데이터 분리·보호).

산출:
  dataset/games.parquet        : 게임 메타 + 콘텐츠 텍스트(doc)  [type='game']
  dataset/interactions.parquet : (steamid, appid, voted_up, playtime) 중복제거
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROGRESS_DB = ROOT / "data" / "progress.db"
REVIEWS_GLOB = str(ROOT / "data" / "reviews" / "*.parquet")
OUT = ROOT / "dataset"


def build_games() -> int:
    """games 테이블 → 콘텐츠 텍스트 doc 포함 parquet."""
    con = sqlite3.connect(PROGRESS_DB)
    df = pd.read_sql_query(
        """SELECT appid, name, genres, categories, short_description,
                  release_date, price_cents, recommendations_total
           FROM games WHERE type='game'""",
        con,
    )
    con.close()

    def to_list(s):
        try:
            return json.loads(s) if s else []
        except json.JSONDecodeError:
            return []

    df["genres_list"] = df["genres"].map(to_list)
    df["categories_list"] = df["categories"].map(to_list)

    def make_doc(r):
        # 콘텐츠 임베딩 입력: 이름 + 장르 + 카테고리 + 짧은설명
        parts = [str(r["name"] or "")]
        parts += r["genres_list"]
        parts += r["categories_list"]
        parts.append(str(r["short_description"] or ""))
        return " | ".join(p for p in parts if p).strip()

    df["doc"] = df.apply(make_doc, axis=1)
    df["recommendations_total"] = df["recommendations_total"].fillna(0).astype(int)

    out = df[["appid", "name", "genres", "categories", "short_description",
              "release_date", "price_cents", "recommendations_total", "doc"]]
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT / "games.parquet", index=False)
    return len(out)


def build_interactions() -> int:
    """리뷰 parquet → (steamid, appid) 중복제거 인터랙션."""
    con = duckdb.connect()
    OUT.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"""
        COPY (
            SELECT
                steamid,
                appid,
                MAX(CAST(voted_up AS INT))        AS voted_up,
                MAX(COALESCE(playtime_at_review,0)) AS playtime
            FROM read_parquet('{REVIEWS_GLOB}')
            WHERE steamid IS NOT NULL
            GROUP BY steamid, appid
        ) TO '{(OUT / "interactions.parquet").as_posix()}' (FORMAT PARQUET)
        """
    )
    n = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{(OUT / 'interactions.parquet').as_posix()}')"
    ).fetchone()[0]
    con.close()
    return n


def main() -> None:
    print("[games] 빌드 중...")
    ng = build_games()
    print(f"[games] 완료 — {ng:,}개 → dataset/games.parquet")
    print("[interactions] 빌드 중...")
    ni = build_interactions()
    print(f"[interactions] 완료 — {ni:,}행 → dataset/interactions.parquet")


if __name__ == "__main__":
    main()
