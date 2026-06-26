"""수집 실행기.

  python run.py applist                          # 1단계: 전체 게임 목록
  python run.py details                          # 2단계: 메타데이터(전체)
  python run.py details --limit 500              #        일부만(테스트)
  python run.py reviews --max-per-game 200       # 3단계: 인기순 리뷰(게임당 200)
  python run.py reviews --max-games 100 --max-per-game 50   # 소규모 테스트
  python run.py status                           # 진행 현황

모든 단계는 재개 가능 — 중단 후 같은 명령을 다시 실행하면 이어서 수집한다.
"""
import argparse
import sys

# Windows 콘솔(cp949)에서 em-dash 등 유니코드 출력 깨짐/크래시 방지
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from collector import applist, appdetails, reviews, storage


def show_status() -> None:
    storage.init()
    with storage.get_conn() as c:
        applist_n = c.execute("SELECT COUNT(*) FROM applist").fetchone()[0]
        details_n = c.execute("SELECT COUNT(*) FROM details_status").fetchone()[0]
        games_n = c.execute("SELECT COUNT(*) FROM games WHERE type='game'").fetchone()[0]
        rv_done = c.execute("SELECT COUNT(*) FROM reviews_status WHERE done=1").fetchone()[0]
        rv_collected = c.execute("SELECT COALESCE(SUM(collected),0) FROM reviews_status").fetchone()[0]
    print("=== 수집 현황 ===")
    print(f"  applist  : {applist_n:,}")
    print(f"  details  : {details_n:,} 처리 / 게임 {games_n:,}")
    print(f"  reviews  : {rv_done:,} 게임 완료 / 리뷰 {rv_collected:,}건 수집")


def main() -> None:
    p = argparse.ArgumentParser(description="Steam 데이터 수집기")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("applist")

    pd_ = sub.add_parser("details")
    pd_.add_argument("--limit", type=int, default=None)
    pd_.add_argument("--lang", default="english")

    pr = sub.add_parser("reviews")
    pr.add_argument("--max-per-game", type=int, default=None)
    pr.add_argument("--max-games", type=int, default=None)
    pr.add_argument("--lang", default="all", help="리뷰 언어 필터 (all|english|koreana...)")
    pr.add_argument("--min-recommendations", type=int, default=1)

    sub.add_parser("status")

    args = p.parse_args()
    if args.cmd == "applist":
        applist.collect()
    elif args.cmd == "details":
        appdetails.collect(limit=args.limit, lang=args.lang)
    elif args.cmd == "reviews":
        reviews.collect(max_per_game=args.max_per_game, max_games=args.max_games,
                        languages=args.lang, min_recommendations=args.min_recommendations)
    elif args.cmd == "status":
        show_status()


if __name__ == "__main__":
    main()
