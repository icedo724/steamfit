"""steamfit 프로젝트 라이브 현황판 — MotherDuck를 폴링해 진행상황 표시.

  /            모바일 대시보드(HTML)
  /api/status  클라우드 DB 실시간 카운트(JSON)
실행: python dashboard/status_server.py   (포트 7870)
"""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.db import connect

SEED_GAMES = 169597
NEW_BATCH = 2647  # 이번 증분에서 감지한 신규 게임 수(진행 표시용)

_con, _where = connect()
_lock = threading.Lock()


def get_status():
    with _lock:
        g = _con.execute("SELECT COUNT(*) FROM games").fetchone()[0]
        inter = _con.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
        pend = _con.execute("SELECT COUNT(*) FROM collection_state WHERE details_done=FALSE").fetchone()[0]
        wm = _con.execute("SELECT COUNT(*) FROM collection_state WHERE last_review_ts>0").fetchone()[0]
    done = max(0, NEW_BATCH - pend)
    return {
        "where": _where, "games": g, "new_games": g - SEED_GAMES,
        "interactions": inter, "pending_details": pend, "watermarked": wm,
        "backfill_done": done, "backfill_total": NEW_BATCH,
    }


HTML = """<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>steamfit 현황판</title><style>
:root{--bg:#0b1622;--card:#16212e;--card2:#1d2a39;--line:#27384b;--txt:#e7eef6;--sub:#9fb2c6;--accent:#4f9bff;--good:#39d98a;--warn:#f5b54a;--mut:#5b6b7d}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--txt);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Noto Sans KR",sans-serif;line-height:1.5;padding:0 0 40px}
.wrap{max-width:560px;margin:0 auto;padding:0 16px}
header{padding:26px 16px 16px;background:linear-gradient(160deg,#13315c,#0b1622 80%)}
h1{font-size:1.4rem;margin:0 0 4px}.tag{font-size:.72rem;color:var(--accent);font-weight:700}
.live{display:inline-flex;align-items:center;gap:6px;font-size:.72rem;color:var(--good);margin-left:8px}
.dot{width:7px;height:7px;border-radius:50%;background:var(--good);animation:p 1.4s infinite}
@keyframes p{0%,100%{opacity:1}50%{opacity:.3}}
h2{font-size:1rem;margin:26px 0 10px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px;margin-bottom:12px}
.stats{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.stat{background:var(--card2);border-radius:10px;padding:12px}
.stat .v{font-size:1.35rem;font-weight:800;letter-spacing:-.02em}
.stat .l{font-size:.73rem;color:var(--sub);margin-top:2px}
.stat .d{font-size:.7rem;color:var(--good);margin-top:1px}
.bar{height:8px;background:var(--card2);border-radius:4px;overflow:hidden;margin-top:8px}
.bar>i{display:block;height:100%;background:var(--warn);width:0;transition:width .5s}
.ph{display:flex;gap:10px;padding:9px 0;border-top:1px solid var(--line)}
.ph:first-child{border-top:none}.ph .ic{flex:none;width:22px}.ph .t{flex:1;font-size:.9rem}
.ph .s{font-size:.72rem;color:var(--sub)}
.done .t{color:var(--good)}.doing .t{color:var(--warn)}.todo .t{color:var(--mut)}
.todo li{font-size:.88rem;color:var(--sub);margin:6px 0}
.upd{font-size:.7rem;color:var(--mut);text-align:center;margin-top:14px}
</style></head><body>
<header><div class="wrap"><div class="tag">PROJECT · AI 모델 개발/추천</div>
<h1>steamfit 현황판<span class="live"><span class="dot"></span>LIVE</span></h1>
<div style="color:var(--sub);font-size:.85rem">Steam 취향+의도 하이브리드 추천 · 클라우드 증분 파이프라인</div></div></header>
<div class="wrap">
<h2>☁️ 클라우드 수집 현황 <span style="font-size:.7rem;color:var(--sub)">(실시간)</span></h2>
<div class="card">
<div class="stats">
<div class="stat"><div class="v" id="games">—</div><div class="l">게임 (MotherDuck)</div><div class="d" id="newg"></div></div>
<div class="stat"><div class="v" id="inter">—</div><div class="l">인터랙션</div></div>
<div class="stat"><div class="v" id="wm">—</div><div class="l">워터마크 게임</div></div>
<div class="stat"><div class="v" id="pend">—</div><div class="l">신규 details 대기</div></div>
</div>
<div style="font-size:.78rem;color:var(--sub);margin-top:12px">신규 게임 EN+KO 백필 <span id="bftxt"></span></div>
<div class="bar"><i id="bf"></i></div>
</div>
<h2>🗺️ 로드맵</h2>
<div class="card">
<div class="ph done"><div class="ic">✅</div><div class="t">데이터 수집<div class="s">공식 Steam API · 리뷰 1,021만</div></div></div>
<div class="ph done"><div class="ic">✅</div><div class="t">모델링<div class="s">협업 임베딩 학습 · 하이브리드 베이스라인 +13%</div></div></div>
<div class="ph done"><div class="ic">✅</div><div class="t">추천 데모<div class="s">취향+의도 steering 작동 (Gradio)</div></div></div>
<div class="ph doing"><div class="ic">🔄</div><div class="t">클라우드 전환<div class="s">MotherDuck 적재 완료 · 증분 파이프라인 가동 중</div></div></div>
<div class="ph todo"><div class="ic">⏳</div><div class="t">자동화<div class="s">GitHub Actions 매일 cron (steamfit 레포)</div></div></div>
<div class="ph todo"><div class="ic">⏳</div><div class="t">다국어 재학습 + 영구 배포<div class="s">다국어 임베딩 · HF Spaces</div></div></div>
<div class="ph todo"><div class="ic">⏳</div><div class="t">추론 과정 시각화<div class="s">취향→의도→결합→근접검색 애니메이션 (explainability)</div></div></div>
</div>
<h2>✅ 다음 할 일</h2>
<div class="card"><ul class="todo" style="margin:0;padding-left:18px">
<li>steamfit GitHub 레포 푸시</li>
<li>Actions Secrets 등록 (STEAM_API_KEY · MOTHERDUCK_TOKEN) — 폰에서 GitHub 웹</li>
<li>다국어 임베딩 재학습 (한국어 의도 지원)</li>
<li>HF Spaces 영구 배포</li>
<li>추론 과정 시각화 (알고리즘 흐름 애니메이션)</li>
</ul></div>
<div class="upd" id="upd"></div>
</div>
<script>
function f(n){return n.toLocaleString()}
async function tick(){
 try{const r=await fetch('/api/status',{cache:'no-store'});const d=await r.json();
  document.getElementById('games').textContent=f(d.games);
  document.getElementById('newg').textContent=d.new_games>0?('+'+f(d.new_games)+' 신규'):'';
  document.getElementById('inter').textContent=f(d.interactions);
  document.getElementById('wm').textContent=f(d.watermarked);
  document.getElementById('pend').textContent=f(d.pending_details);
  const pct=d.backfill_total?Math.round(d.backfill_done/d.backfill_total*100):0;
  document.getElementById('bf').style.width=pct+'%';
  document.getElementById('bftxt').textContent=f(d.backfill_done)+' / '+f(d.backfill_total)+' ('+pct+'%)';
  document.getElementById('upd').textContent='업데이트: '+new Date().toLocaleTimeString('ko-KR');
 }catch(e){document.getElementById('upd').textContent='연결 재시도 중...';}
}
tick();setInterval(tick,8000);
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/api/status"):
            try:
                body = json.dumps(get_status()).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
            except Exception as e:  # noqa: BLE001
                body = json.dumps({"error": str(e)}).encode()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML.encode("utf-8"))

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print("status board on http://127.0.0.1:7870  (DB:", _where, ")")
    HTTPServer(("127.0.0.1", 7870), H).serve_forever()
