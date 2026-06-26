"""기존 로컬 수집물 → 새 스키마(클라우드/로컬 DB) 1회 시드 이관.

data/ 는 읽기 전용으로 참조(원시 보호). 기존 1,021만 인터랙션을 클라우드로 옮겨
처음부터 재수집하지 않게 한다. EN 메타는 즉시 시드, KO는 이후 details 패스로 백필.

실행(.env에 MOTHERDUCK_TOKEN 있으면 클라우드로):
  python -m pipeline.seed
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.db import connect

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
GAMES = (ROOT / "dataset" / "games.parquet").as_posix()
REVIEWS = (ROOT / "data" / "reviews" / "*.parquet").as_posix()


def main():
    con, where = connect()
    print(f"DB: {where}")

    print("[games] EN 메타 시드...")
    con.execute(f"""
        INSERT INTO games(appid,name,short_desc_en,genres_en,categories,
                          release_date,price_cents,recommendations_total,type,updated_at)
        SELECT appid, name, short_description, genres, categories,
               release_date, price_cents, recommendations_total, 'game', now()
        FROM read_parquet('{GAMES}')
        ON CONFLICT(appid) DO NOTHING
    """)
    print("  games:", con.execute("SELECT COUNT(*) FROM games").fetchone()[0])

    print("[interactions] 1,021만 이관(리뷰→중복제거)... (네트워크 업로드, 수 분 소요 가능)")
    con.execute(f"""
        INSERT INTO interactions(steamid,appid,voted_up,playtime,ts_created)
        SELECT CAST(steamid AS BIGINT), appid,
               bool_or(voted_up), MAX(COALESCE(playtime_at_review,0)), MAX(timestamp_created)
        FROM read_parquet('{REVIEWS}')
        WHERE steamid IS NOT NULL
        GROUP BY CAST(steamid AS BIGINT), appid
        ON CONFLICT(steamid,appid) DO NOTHING
    """)
    print("  interactions:", con.execute("SELECT COUNT(*) FROM interactions").fetchone()[0])

    print("[reviews] 리뷰 본문 이관(텍스트 포함)... (대용량 업로드, 시간 소요)")
    con.execute(f"""
        INSERT INTO reviews(recommendationid,appid,steamid,language,voted_up,
                            votes_up,weighted_vote_score,playtime_at_review,ts_created,review)
        SELECT CAST(recommendationid AS BIGINT), appid, CAST(steamid AS BIGINT), language,
               voted_up, COALESCE(votes_up,0), CAST(weighted_vote_score AS DOUBLE),
               COALESCE(playtime_at_review,0), timestamp_created, review
        FROM read_parquet('{REVIEWS}')
        WHERE recommendationid IS NOT NULL AND steamid IS NOT NULL
        QUALIFY row_number() OVER (PARTITION BY CAST(recommendationid AS BIGINT)
                                   ORDER BY timestamp_created DESC) = 1
        ON CONFLICT(recommendationid) DO NOTHING
    """)
    print("  reviews:", con.execute("SELECT COUNT(*) FROM reviews").fetchone()[0])

    print("[collection_state] 워터마크 시드(게임별 마지막 리뷰 시각)...")
    con.execute("""
        INSERT INTO collection_state(appid,details_done,last_review_ts,reviews_seeded,updated_at)
        SELECT g.appid, TRUE, COALESCE(m.mx,0), TRUE, now()
        FROM games g
        LEFT JOIN (SELECT appid, MAX(ts_created) AS mx FROM interactions GROUP BY appid) m
               ON m.appid = g.appid
        ON CONFLICT(appid) DO UPDATE SET details_done=TRUE,
               last_review_ts=excluded.last_review_ts, reviews_seeded=TRUE
    """)
    print("  collection_state:", con.execute("SELECT COUNT(*) FROM collection_state").fetchone()[0])
    print("시드 완료. 이후 GH Actions가 증분만 갱신.")
    con.close()


if __name__ == "__main__":
    main()
