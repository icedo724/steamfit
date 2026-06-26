"""1단계 — 전체 게임 목록 수집 (IStoreService/GetAppList, 키 필요).

게임만 필터(include_games), last_appid 페이징. progress.db의 applist에 저장.
"""
import time

from config import APPLIST_URL, REQUEST_DELAY, require_key
from . import http, storage


def collect() -> int:
    key = require_key()
    storage.init()
    last = 0
    total = 0
    while True:
        params = {
            "key": key,
            "include_games": "true",
            "include_dlc": "false",
            "include_software": "false",
            "include_videos": "false",
            "include_hardware": "false",
            "max_results": 50000,
        }
        if last:
            params["last_appid"] = last
        r = http.get(APPLIST_URL, params=params, timeout=60)
        if r is None:
            print("[applist] 요청 실패 — 중단")
            break
        resp = r.json().get("response", {})
        apps = resp.get("apps", [])
        if not apps:
            break
        storage.save_applist(apps)
        total += len(apps)
        print(f"[applist] 누적 {total:,} (last_appid={resp.get('last_appid')})")
        if resp.get("have_more_results"):
            last = resp["last_appid"]
            time.sleep(REQUEST_DELAY)
        else:
            break
    print(f"[applist] 완료 — 총 {storage.applist_count():,}개 게임")
    return total
