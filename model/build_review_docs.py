"""게임별 다국어 리뷰 문서 생성 — A(리뷰 enrich 재학습)용 데이터 단계.

로컬 data/reviews/*.parquet(1,022만)에서 게임·언어별 상위(추천수) 리뷰를 추려
짧게 잘라 결합. 한국어를 먼저 배치(256토큰 윈도우에서 한국어 어휘 보존 — A의 핵심 목표).
산출: dataset/review_docs.parquet (appid, review_text).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import duckdb
import pandas as pd

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
REVIEWS = (ROOT / "data" / "reviews" / "*.parquet").as_posix()
OUT = ROOT / "dataset" / "review_docs.parquet"

# 언어별 게임당 리뷰 수 (한국어 우대 — 희소·고가치 신호)
K = {"koreana": 6, "english": 5, "schinese": 3}
ORDER = ["koreana", "english", "schinese"]   # doc 결합 순서(한국어 먼저)
SNIPPET = 220       # 리뷰당 최대 글자
MINLEN, MAXLEN = 20, 600


def main():
    t0 = time.time()
    langs = "', '".join(K)
    # 게임·언어별 추천수 상위 리뷰 (길이 필터)
    q = f"""
    WITH r AS (
        SELECT appid, language, votes_up,
               substr(replace(replace(review, chr(10), ' '), chr(13), ' '), 1, {SNIPPET}) AS snip,
               ROW_NUMBER() OVER (PARTITION BY appid, language
                                  ORDER BY votes_up DESC, length(review)) AS rn
        FROM read_parquet('{REVIEWS}')
        WHERE language IN ('{langs}') AND voted_up
              AND length(review) BETWEEN {MINLEN} AND {MAXLEN}
    )
    SELECT appid, language, snip, rn FROM r
    WHERE (language='koreana'  AND rn <= {K['koreana']})
       OR (language='english'  AND rn <= {K['english']})
       OR (language='schinese' AND rn <= {K['schinese']})
    """
    df = duckdb.query(q).df()
    print(f"추린 리뷰 행: {len(df):,}  ({time.time()-t0:.0f}s)")

    lang_rank = {l: i for i, l in enumerate(ORDER)}
    df["lr"] = df["language"].map(lang_rank)
    df = df.sort_values(["appid", "lr", "rn"])
    docs = (df.groupby("appid")["snip"]
              .apply(lambda s: "  ·  ".join(x for x in s if x))
              .reset_index().rename(columns={"snip": "review_text"}))
    docs.to_parquet(OUT, index=False)
    n_ko = df[df.language == "koreana"]["appid"].nunique()
    print(f"게임 {len(docs):,}개 리뷰문서 저장 → {OUT.name}  (한국어 포함 게임 {n_ko:,})")
    print(f"총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
