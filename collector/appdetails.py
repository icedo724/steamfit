"""2단계 — 게임 메타데이터 수집 (store appdetails, 키 불필요).

applist의 미수집 appid에 대해 메타 파싱 → games 테이블 저장.
recommendations.total(인기도)도 함께 받아 3단계 리뷰 수집 우선순위로 사용.
재개: details_status에 기록된 건 건너뜀.
"""
import json
import time

from config import APPDETAILS_URL, REQUEST_DELAY
from . import http, storage

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **k):
        return x


def _parse(appid: int, data: dict) -> dict:
    """appdetails data → games 레코드."""
    return {
        "appid": appid,
        "name": data.get("name"),
        "type": data.get("type"),
        "is_free": int(bool(data.get("is_free"))),
        "short_description": data.get("short_description", ""),
        "detailed_description": data.get("detailed_description", ""),
        "genres": json.dumps(
            [g["description"] for g in data.get("genres", [])], ensure_ascii=False
        ),
        "categories": json.dumps(
            [c["description"] for c in data.get("categories", [])], ensure_ascii=False
        ),
        "release_date": (data.get("release_date") or {}).get("date"),
        "price_cents": (data.get("price_overview") or {}).get("final"),
        "recommendations_total": (data.get("recommendations") or {}).get("total"),
    }


def collect(limit: int | None = None, lang: str = "english") -> dict:
    storage.init()
    appids = storage.pending_detail_appids(limit)
    print(f"[details] 대상 {len(appids):,}개")
    stats = {"ok": 0, "no_data": 0, "fail": 0}

    for appid in tqdm(appids, desc="appdetails"):
        r = http.get(APPDETAILS_URL, params={"appids": appid, "l": lang})
        if r is None:
            storage.mark_detail_status(appid, "fail")
            stats["fail"] += 1
            time.sleep(REQUEST_DELAY)
            continue
        try:
            node = r.json().get(str(appid), {})
        except ValueError:
            storage.mark_detail_status(appid, "fail")
            stats["fail"] += 1
            time.sleep(REQUEST_DELAY)
            continue

        if node.get("success") and node.get("data"):
            storage.save_game(_parse(appid, node["data"]), status="ok")
            stats["ok"] += 1
        else:
            storage.mark_detail_status(appid, "no_data")
            stats["no_data"] += 1
        time.sleep(REQUEST_DELAY)

    print(f"[details] 완료 — {stats}")
    return stats
