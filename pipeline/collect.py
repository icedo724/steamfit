"""증분 수집기 (클라우드/로컬 공통).

핵심: 매일 전체 재수집이 아니라 **추가분만** 감지해 적재.
  - 새 게임:   GetAppList vs DB → 신규 appid만 details 수집 (EN+KO)
  - 새 리뷰:   게임별 last_review_ts 워터마크 → 그 이후 작성된 리뷰만 수집
실행:
  python -m pipeline.collect applist
  python -m pipeline.collect details --limit 200
  python -m pipeline.collect reviews --max-games 500
  python -m pipeline.collect all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.db import connect, get_meta, set_meta

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

APPLIST_URL = "https://api.steampowered.com/IStoreService/GetAppList/v1/"
APPDETAILS_URL = "https://store.steampowered.com/api/appdetails"
APPREVIEWS_URL = "https://store.steampowered.com/appreviews/{appid}"
DELAY = 1.3
UA = {"User-Agent": "Mozilla/5.0 (steam-recsys; research)"}

FIRST_CAP = 500      # 신규 게임 첫 시드 시 게임당 최대 리뷰
PER_RUN_CAP = 200    # 이후 증분 실행 시 게임당 상한(보통 그보다 훨씬 적게 들어옴)

_session = requests.Session()
_session.headers.update(UA)
# 연령 게이트 우회(성인 게임 스토어 페이지 태그 접근용)
_session.cookies.update({"birthtime": "0", "wants_mature_content": "1",
                         "lastagecheckage": "1-0-1990", "mature_content": "1"})


def _fetch_tags(appid):
    """Steam 스토어 페이지에서 user 태그 추출 (JSON 배열명, 최대 25개)."""
    import re
    try:
        r = _session.get(f"https://store.steampowered.com/app/{appid}/", timeout=20)
        m = re.search(r"InitAppTagModal\(\s*\d+\s*,\s*(\[.*?\])\s*,", r.text, re.S)
        if m:
            return [t["name"] for t in json.loads(m.group(1))][:25]
    except Exception:
        pass
    return []


def _get(url, params=None, retries=4):
    for a in range(retries):
        try:
            r = _session.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r
            if r.status_code in (429, 403):
                time.sleep(30 * (a + 1)); continue
            time.sleep(5)
        except requests.RequestException:
            time.sleep(5)
    return None


# ── 1) applist 증분: 신규 게임 감지 ───────────────────────────────
def sync_applist(con):
    key = os.getenv("STEAM_API_KEY", "")
    if not key:
        raise SystemExit("STEAM_API_KEY 필요(.env)")
    known = {r[0] for r in con.execute("SELECT appid FROM collection_state").fetchall()}
    last, new = 0, 0
    while True:
        p = {"key": key, "include_games": "true", "include_dlc": "false",
             "include_software": "false", "include_videos": "false",
             "include_hardware": "false", "max_results": 50000}
        if last:
            p["last_appid"] = last
        r = _get(APPLIST_URL, p)
        if not r:
            break
        resp = r.json().get("response", {})
        apps = resp.get("apps", [])
        if not apps:
            break
        fresh = [(a["appid"],) for a in apps if a["appid"] not in known]
        if fresh:
            con.executemany(
                "INSERT INTO collection_state(appid) VALUES(?) ON CONFLICT(appid) DO NOTHING",
                fresh,
            )
            new += len(fresh)
        if resp.get("have_more_results"):
            last = resp["last_appid"]; time.sleep(DELAY)
        else:
            break
    set_meta(con, "last_applist_sync", int(time.time()))
    print(f"[applist] 신규 게임 {new:,}개 큐 적재")
    return new


# ── 2) details 증분: 신규 게임만 EN+KO 메타 ───────────────────────
def _appdetails(appid, lang):
    r = _get(APPDETAILS_URL, {"appids": appid, "l": lang})
    if not r:
        return None
    try:
        node = r.json().get(str(appid), {})
    except ValueError:
        return None
    return node.get("data") if node.get("success") else None


def collect_details(con, limit=None):
    q = "SELECT appid FROM collection_state WHERE details_done=FALSE ORDER BY appid"
    if limit:
        q += f" LIMIT {int(limit)}"
    appids = [r[0] for r in con.execute(q).fetchall()]
    print(f"[details] 신규 대상 {len(appids):,}개")
    ok = 0
    for appid in appids:
        en = _appdetails(appid, "english"); time.sleep(DELAY)
        ko = _appdetails(appid, "koreana"); time.sleep(DELAY)
        if en is None:  # 게임 아님/삭제됨
            con.execute("UPDATE collection_state SET details_done=TRUE, updated_at=now() WHERE appid=?", [appid])
            continue
        tags = _fetch_tags(appid); time.sleep(DELAY)
        g = (
            appid,
            en.get("name"), (ko or {}).get("name"),
            en.get("type"), bool(en.get("is_free")),
            en.get("short_description", ""), (ko or {}).get("short_description", ""),
            json.dumps([x["description"] for x in en.get("genres", [])], ensure_ascii=False),
            json.dumps([x["description"] for x in (ko or {}).get("genres", [])], ensure_ascii=False),
            json.dumps([x["description"] for x in en.get("categories", [])], ensure_ascii=False),
            json.dumps(tags, ensure_ascii=False),
            (en.get("release_date") or {}).get("date"),
            (en.get("price_overview") or {}).get("final"),
            (en.get("recommendations") or {}).get("total"),
        )
        con.execute(
            """INSERT INTO games(appid,name,name_ko,type,is_free,short_desc_en,short_desc_ko,
                  genres_en,genres_ko,categories,tags,release_date,price_cents,recommendations_total,updated_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,now())
               ON CONFLICT(appid) DO UPDATE SET name=excluded.name, name_ko=excluded.name_ko,
                  type=excluded.type, short_desc_en=excluded.short_desc_en, short_desc_ko=excluded.short_desc_ko,
                  genres_en=excluded.genres_en, genres_ko=excluded.genres_ko, tags=excluded.tags,
                  recommendations_total=excluded.recommendations_total, updated_at=now()""",
            g,
        )
        con.execute("UPDATE collection_state SET details_done=TRUE, updated_at=now() WHERE appid=?", [appid])
        ok += 1
    print(f"[details] 메타 수집 {ok:,}개")
    return ok


# ── 3) reviews 증분: 워터마크 이후만 ──────────────────────────────
def _refresh_one(con, appid):
    st = con.execute(
        "SELECT last_review_ts, reviews_seeded FROM collection_state WHERE appid=?", [appid]
    ).fetchone()
    last_ts = st[0] or 0
    cap = PER_RUN_CAP if (st and st[1]) else FIRST_CAP
    cursor, newest, rows, rev_rows = "*", last_ts, [], []
    while len(rows) < cap:
        r = _get(APPREVIEWS_URL.format(appid=appid),
                 {"json": 1, "num_per_page": 100, "filter": "recent",
                  "language": "all", "cursor": cursor})
        if not r:
            break
        try:
            j = r.json()
        except ValueError:
            break
        revs = j.get("reviews", [])
        if not revs:
            break
        stop = False
        for rv in revs:
            ts = rv.get("timestamp_created", 0)
            if ts <= last_ts:        # 이미 수집한 지점 → 증분 종료
                stop = True; break
            a = rv.get("author", {})
            rows.append((int(a.get("steamid")), appid, bool(rv.get("voted_up")),
                         a.get("playtime_at_review") or 0, ts))
            try:
                rev_rows.append((
                    int(rv.get("recommendationid")), appid, int(a.get("steamid")),
                    rv.get("language"), bool(rv.get("voted_up")), rv.get("votes_up") or 0,
                    float(rv.get("weighted_vote_score") or 0.0),
                    a.get("playtime_at_review") or 0, ts, rv.get("review") or "",
                ))
            except (TypeError, ValueError):
                pass
            newest = max(newest, ts)
        nc = j.get("cursor", cursor)
        if stop or len(revs) < 100 or nc == cursor:
            break
        cursor = nc
        time.sleep(DELAY)
    if rows:
        con.executemany(
            """INSERT INTO interactions(steamid,appid,voted_up,playtime,ts_created)
               VALUES(?,?,?,?,?)
               ON CONFLICT(steamid,appid) DO UPDATE SET voted_up=excluded.voted_up,
                  playtime=excluded.playtime""",
            rows,
        )
    if rev_rows:
        con.executemany(
            """INSERT INTO reviews(recommendationid,appid,steamid,language,voted_up,
                  votes_up,weighted_vote_score,playtime_at_review,ts_created,review)
               VALUES(?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(recommendationid) DO NOTHING""",
            rev_rows,
        )
    con.execute(
        "UPDATE collection_state SET last_review_ts=?, reviews_seeded=TRUE, updated_at=now() WHERE appid=?",
        [newest, appid],
    )
    return len(rows)


def refresh_reviews(con, max_games=None):
    q = """SELECT g.appid FROM games g
           JOIN collection_state s ON s.appid=g.appid
           WHERE g.type='game' AND COALESCE(g.recommendations_total,0) > 0
           ORDER BY COALESCE(g.recommendations_total,0) DESC"""
    if max_games:
        q += f" LIMIT {int(max_games)}"
    appids = [r[0] for r in con.execute(q).fetchall()]
    print(f"[reviews] 대상 게임 {len(appids):,}개 (증분)")
    total = 0
    for i, appid in enumerate(appids, 1):
        total += _refresh_one(con, appid)
        if i % 200 == 0:
            print(f"  {i:,}/{len(appids):,} · 누적 신규 리뷰 {total:,}")
    print(f"[reviews] 신규 인터랙션 {total:,}건")
    return total


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("applist")
    pd_ = sub.add_parser("details"); pd_.add_argument("--limit", type=int)
    pr = sub.add_parser("reviews"); pr.add_argument("--max-games", type=int)
    pa = sub.add_parser("all"); pa.add_argument("--max-games", type=int)
    a = p.parse_args()

    con, where = connect()
    print(f"DB: {where}")
    if a.cmd == "applist":
        sync_applist(con)
    elif a.cmd == "details":
        collect_details(con, a.limit)
    elif a.cmd == "reviews":
        refresh_reviews(con, a.max_games)
    elif a.cmd == "all":
        sync_applist(con)
        collect_details(con)
        refresh_reviews(con, a.max_games)
    con.close()


if __name__ == "__main__":
    main()
