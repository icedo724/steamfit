"""재시도·백오프 내장 HTTP 게터 (rate-limit 대응)."""
import time

import requests

from config import USER_AGENT

_session = requests.Session()
_session.headers["User-Agent"] = USER_AGENT


def get(url, params=None, timeout=30, retries=4):
    """200이면 Response 반환, 실패하면 None.
    429/403(rate-limit)은 점증 백오프로 재시도."""
    for attempt in range(retries):
        try:
            r = _session.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code in (429, 403):
                time.sleep(30 * (attempt + 1))  # 30s, 60s, 90s...
                continue
            time.sleep(5)
        except requests.RequestException:
            time.sleep(5)
    return None
