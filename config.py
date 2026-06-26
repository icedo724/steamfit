"""전역 설정 — 경로, API 키, 수집 파라미터."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"

# Steam Web API 키 (GetAppList 등 키 필요한 호출용). 리뷰/메타는 키 불필요.
STEAM_API_KEY = os.getenv("STEAM_API_KEY", "")

# ── 엔드포인트 ────────────────────────────────────────────────────
APPLIST_URL    = "https://api.steampowered.com/IStoreService/GetAppList/v1/"
APPDETAILS_URL = "https://store.steampowered.com/api/appdetails"
APPREVIEWS_URL = "https://store.steampowered.com/appreviews/{appid}"

# ── 수집 파라미터 (rate-limit 준수) ───────────────────────────────
REQUEST_DELAY = 1.5          # 호출 간 대기(초)
REVIEWS_PER_PAGE = 100       # appreviews 최대
USER_AGENT = "Mozilla/5.0 (steam-recommender; personal research)"


def require_key() -> str:
    if not STEAM_API_KEY or STEAM_API_KEY.startswith("여기에"):
        raise SystemExit(
            "STEAM_API_KEY 미설정 — .env.example 을 .env 로 복사하고 키를 넣으세요."
        )
    return STEAM_API_KEY
