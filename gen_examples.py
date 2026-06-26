"""데모용 예시 추천 결과를 미리 계산해 examples.json 으로 저장 (정적 HTML 임베드용)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recommend import Recommender

rec = Recommender()

CASES = [
    {"title": "취향만 (의도 없음)", "intent_ko": "—", "liked": [400, 620], "intent": "", "wt": 1.0, "wi": 0.0},
    {"title": "+ 오픈월드 생존 크래프팅", "intent_ko": "open world survival crafting", "liked": [400, 620],
     "intent": "open world survival crafting", "wt": 0.5, "wi": 0.5},
    {"title": "+ 무서운 호러 분위기", "intent_ko": "scary horror atmospheric", "liked": [400, 620],
     "intent": "scary horror atmospheric", "wt": 0.5, "wi": 0.5},
    {"title": "+ 빠른 경쟁 멀티플레이 슈터", "intent_ko": "fast competitive multiplayer shooter", "liked": [400, 620],
     "intent": "fast competitive multiplayer shooter", "wt": 0.5, "wi": 0.5},
]

out = []
for c in CASES:
    recs = rec.recommend(c["liked"], c["intent"], c["wt"], c["wi"], topn=8)
    out.append({
        "title": c["title"],
        "intent_ko": c.get("intent_ko", c["intent"]),
        "liked": [rec.name.get(a, str(a)) for a in c["liked"]],
        "intent": c["intent"],
        "recs": [{"name": n, "score": round(s, 3)} for _, n, s in recs],
    })

Path("examples.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("saved examples.json:", len(out), "cases")
for o in out:
    print("-", o["title"], "->", o["recs"][0]["name"], "...")
