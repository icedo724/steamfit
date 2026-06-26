"""대조학습용 양성쌍(positive pairs) 추출.

양성쌍 = 같은 유저가 둘 다 추천(voted_up=1)한 두 게임.
이 쌍으로 "같이 즐겨지는 게임은 임베딩이 가까워지도록" 대조학습한다.

효율적 샘플링: 유저별로 랜덤 정렬 후 인접 아이템끼리 짝지어(floor(n/2)쌍) 선형 추출.
  - 파워유저 편향 방지: 게임 2~100개 유저만, 유저당 쌍 수 제한.
  - 최종 MAX_PAIRS개로 다운샘플.
산출: dataset/pairs.parquet (appid_a, appid_b)  — 학습 시 games.doc과 조인.
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
INTER = (ROOT / "dataset" / "interactions.parquet").as_posix()
OUT = (ROOT / "dataset" / "pairs.parquet").as_posix()

MAX_USER_GAMES = 50    # 이보다 많이 리뷰한 파워유저 제외(자기조인 폭발 방지)
MAX_PAIRS = 4_000_000  # 최종 양성쌍 수 (조밀하게)


def main():
    con = duckdb.connect()
    # 유저당 전체 공동출현 쌍(조합)을 만들고 reservoir 샘플로 MAX_PAIRS 추출.
    # 인접 짝짓기(희소) 대신 전체 조합 → cooc 신호를 더 충실히 반영.
    con.execute(
        f"""
        COPY (
            WITH pos AS (
                SELECT steamid, appid FROM read_parquet('{INTER}') WHERE voted_up = 1
            ),
            deg AS (SELECT steamid, COUNT(*) AS n FROM pos GROUP BY steamid),
            filt AS (
                SELECT p.steamid, p.appid
                FROM pos p JOIN deg d USING(steamid)
                WHERE d.n BETWEEN 2 AND {MAX_USER_GAMES}
            )
            SELECT appid_a, appid_b FROM (
                SELECT a.appid AS appid_a, b.appid AS appid_b
                FROM filt a JOIN filt b
                  ON a.steamid = b.steamid AND a.appid < b.appid
            ) USING SAMPLE {MAX_PAIRS} ROWS (reservoir, 42)
        ) TO '{OUT}' (FORMAT PARQUET)
        """
    )
    n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{OUT}')").fetchone()[0]
    uniq = con.execute(
        f"SELECT COUNT(*) FROM (SELECT DISTINCT appid_a, appid_b FROM read_parquet('{OUT}'))"
    ).fetchone()[0]
    con.close()
    print(f"[pairs] 완료 — {n:,}쌍 (고유 {uniq:,}) → dataset/pairs.parquet")


if __name__ == "__main__":
    main()
