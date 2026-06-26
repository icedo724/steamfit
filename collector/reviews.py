"""3단계 — 리뷰 수집 (store appreviews, 키 불필요).

인기순(recommendations_total desc)으로 게임을 돌며 커서 페이징.
리뷰 본문 + 작성자(steamid·playtime)·추천여부를 parquet 샤드에 누적.

재개·크래시 안전 설계:
  - 진행상태(cursor/collected/done)는 메모리 pending에 모았다가, parquet **flush 직후에만**
    한 번에 커밋한다(ReviewShardWriter.on_flush).
  - 따라서 강제 종료/정전이 나도 SQLite 커서가 parquet보다 앞서지 않음 → **갭 없음**.
    마지막 flush 지점부터 재수집하며, 겹치는 부분은 recommendationid로 나중에 dedup.

max_per_game: 게임당 최대 리뷰 수(단계적 수집용). None이면 전체.
"""
import time

from config import APPREVIEWS_URL, REVIEWS_PER_PAGE, REQUEST_DELAY
from . import http, storage

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **k):
        return x


def _map_review(appid: int, rv: dict) -> dict:
    a = rv.get("author", {})
    return {
        "appid": appid,
        "recommendationid": rv.get("recommendationid"),
        "steamid": a.get("steamid"),
        "language": rv.get("language"),
        "voted_up": rv.get("voted_up"),
        "votes_up": rv.get("votes_up"),
        "votes_funny": rv.get("votes_funny"),
        # Steam은 문자열로 줌 → parquet 타입 일관성 위해 float 캐스팅
        "weighted_vote_score": float(rv.get("weighted_vote_score") or 0.0),
        "comment_count": rv.get("comment_count"),
        "steam_purchase": rv.get("steam_purchase"),
        "received_for_free": rv.get("received_for_free"),
        "written_during_early_access": rv.get("written_during_early_access"),
        "playtime_at_review": a.get("playtime_at_review"),
        "playtime_forever": a.get("playtime_forever"),
        "num_games_owned": a.get("num_games_owned"),
        "num_reviews": a.get("num_reviews"),
        "timestamp_created": rv.get("timestamp_created"),
        "timestamp_updated": rv.get("timestamp_updated"),
        "review": rv.get("review"),
    }


def _collect_one(appid, writer, pending, max_per_game, languages) -> int:
    """게임 1개 리뷰 수집. pending[appid]에 진행상태를 갱신(커밋은 flush 시점).
    반환: 이번 실행에서 수집한 리뷰 수."""
    st = storage.get_review_status(appid)
    cursor = st["cursor"] if st and st["cursor"] else "*"
    collected = st["collected"] if st else 0
    total = st["total_reviews"] if st else 0
    start = collected

    def mark(done: int):
        pending[appid] = {"appid": appid, "total_reviews": total,
                          "cursor": cursor, "collected": collected, "done": done}

    while True:
        if max_per_game and collected >= max_per_game:
            mark(1)
            return collected - start
        r = http.get(
            APPREVIEWS_URL.format(appid=appid),
            params={"json": 1, "num_per_page": REVIEWS_PER_PAGE, "filter": "recent",
                    "language": languages, "cursor": cursor},
        )
        if r is None:
            mark(0)
            return collected - start
        try:
            j = r.json()
        except ValueError:
            mark(0)
            return collected - start

        reviews = j.get("reviews", [])
        if total == 0:
            total = j.get("query_summary", {}).get("total_reviews", 0)
        if not reviews:
            mark(1)
            return collected - start

        rows = [_map_review(appid, rv) for rv in reviews]
        collected += len(reviews)
        next_cursor = j.get("cursor", cursor)
        done = next_cursor == cursor or len(reviews) < REVIEWS_PER_PAGE
        cursor = next_cursor
        mark(int(done))            # 먼저 pending 갱신(커서·진행상태)
        writer.add(rows)           # 그 다음 버퍼 추가 — flush 나면 pending이 함께 커밋됨
        if done:
            return collected - start
        time.sleep(REQUEST_DELAY)


def collect(max_per_game: int | None = None, max_games: int | None = None,
            languages: str = "all", min_recommendations: int = 1) -> dict:
    storage.init()
    targets = storage.review_targets(min_recommendations)
    if max_games:
        targets = targets[:max_games]
    print(f"[reviews] 대상 게임 {len(targets):,}개 (게임당 최대={max_per_game or '전체'})")

    pending: dict[int, dict] = {}

    def persist():
        # flush로 디스크에 안착한 만큼만 진행상태 커밋
        if pending:
            storage.save_review_statuses(list(pending.values()))
            pending.clear()

    writer = storage.ReviewShardWriter(on_flush=persist)
    total_reviews = 0
    try:
        for appid in tqdm(targets, desc="reviews"):
            total_reviews += _collect_one(appid, writer, pending, max_per_game, languages)
    finally:
        writer.flush()   # 남은 버퍼 디스크 기록(+ pending 커밋)
        persist()        # 버퍼가 비어도 남은 진행상태(예: 무리뷰 게임) 커밋
    print(f"[reviews] 완료 — 이번 실행 수집 {total_reviews:,}건")
    return {"reviews": total_reviews, "games": len(targets)}
