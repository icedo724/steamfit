"""SteamFit — 취향+의도 하이브리드 게임 추천 (HF Spaces). 라이트 테마.

탭: 추천(취향+의도 + 추론 흐름 + 2D맵) / 학습 과정(에폭 슬라이더 + 구조도).
모델 인코더는 별도 Hub 저장소(mininiming/steamfit-encoder)에서 로드.
"""
import json
from collections import Counter
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent

# ── 색상 팔레트 (warm neutrals + 가독용 다크 텍스트) ───────────────
BG = "#f5ebe0"        # linen (페이지 배경)
CARD = "#e3d5ca"      # powder-petal (카드/표면)
PARCH = "#edede9"     # parchment (입력칸/보조 표면)
ACCENT = "#d5bdaf"    # almond-silk (버튼/강조)
MUTE = "#d6ccc2"      # dust-grey (테두리/구름)
TEXT = "#3a332c"      # 다크 에스프레소 (주요 글자)
TEXT2 = "#6f6253"     # 미디엄 브라운 (보조 글자)
HILITE = "#bc6c25"    # 강조 포인트(추천/칩) — 밝은 배경 대비용
LIKED = "#5a4632"     # 즐긴 게임 마커(다크 브라운)
# 데이터 시각화용 장르 색(밝은 배경에서 구분되는 어스톤)
GENRE_PAL = ["#9c6644", "#6b705c", "#bc6c25", "#7f5539", "#a5a58d", "#5c6b73",
             "#8a5a44", "#606c38", "#9d8189", "#7d6b5d", "#937341"]

games = pd.read_parquet(ROOT / "games_lookup.parquet")
NAME = dict(zip(games["appid"], games["name"]))
GENRE = dict(zip(games["appid"], games["genres"]))
POP = dict(zip(games["appid"], games["recommendations_total"]))

_map = pd.read_parquet(ROOT / "map2d.parquet")
MX = dict(zip(_map["appid"], _map["x"]))
MY = dict(zip(_map["appid"], _map["y"]))

collab = np.load(ROOT / "item2vec_emb.npy")
cids = pd.read_csv(ROOT / "item2vec_appids.csv")["appid"].tolist()
c_row = {a: i for i, a in enumerate(cids)}
content = np.load(ROOT / "game_emb.npy")
tids = pd.read_csv(ROOT / "game_emb_appids.csv").iloc[:, 0].tolist()
t_row = {a: i for i, a in enumerate(tids)}
cand = [a for a in cids if a in t_row]
collab_c = np.stack([collab[c_row[a]] for a in cand])
collab_n = collab_c / (np.linalg.norm(collab_c, axis=1, keepdims=True) + 1e-9)  # 코사인용 정규화
content_c = np.stack([content[t_row[a]] for a in cand])
cand_idx = {a: i for i, a in enumerate(cand)}

# 공동플레이(co-occurrence) top-K 이웃 — 하이브리드 취향(RRF)용
#   eval_taste.py 측정: RRF(item2vec+cooc) 취향 R@10 0.223 vs item2vec 단독 0.172
_ck = np.load(ROOT / "cooc_topk.npz")
_ck_appids = _ck["appids"].astype(np.int64)              # cooc-row → appid
_ck_nb_appid = _ck_appids[_ck["nb"]]                     # [rows × K] 이웃 appid
_ck_wt = _ck["wt"]                                       # [rows × K] 공동플레이 가중치
_cooc_row = {int(a): i for i, a in enumerate(_ck_appids)}


def _ranks(s):
    """점수 내림차순 순위(1=최고). RRF 융합용."""
    o = np.argsort(-s)
    r = np.empty(len(s), np.float32)
    r[o] = np.arange(1, len(s) + 1, dtype=np.float32)
    return r

# 학습 과정 데이터(에폭별 임베딩 스냅샷)
TRAIN = json.loads((ROOT / "training_frames.json").read_text(encoding="utf-8"))
N_EPOCHS = len(TRAIN["frames"]) - 1
_clusters = TRAIN.get("clusters") or [0] * len(TRAIN["names"])
_gcolor = [GENRE_PAL[c % len(GENRE_PAL)] for c in _clusters]   # 색 = 협업 '이웃 그룹'(KMeans)

_encoder = None


def encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer("mininiming/steamfit-encoder")
    return _encoder


def _norm(v):
    return (v - v.min()) / (v.max() - v.min() + 1e-9)


def _genre_list(a):
    try:
        return json.loads(GENRE.get(a) or "[]")
    except Exception:
        return []


def _genres(a):
    return ", ".join(_genre_list(a)[:3])


def _reason_badges(a, r, cooc_src, liked_genres, intent, intent_s, intent_hi):
    """추천 카드의 '왜 이 게임인지' 근거 칩 — 공동플레이·공유장르·의도부합."""
    bs = []
    src = cooc_src.get(r)
    if src:
        bs.append(f'<span class="rb play">🎮 {NAME.get(src[0], src[0])} 플레이어가 함께 즐김</span>')
    rg = _genre_list(a)
    shared = [g for g in rg if g in liked_genres][:2]
    if shared:
        bs.append(f'<span class="rb">🏷 {", ".join(shared)} 취향 일치</span>')
    elif not liked_genres and rg:
        bs.append(f'<span class="rb">🏷 {", ".join(rg[:2])}</span>')
    if intent and intent_s is not None and intent_hi is not None and intent_s[r] >= intent_hi:
        bs.append('<span class="rb intent">🧭 의도 부합</span>')
    if not bs:
        g = _genres(a)
        bs.append(f'<span class="rb">🏷 {g}</span>' if g else '<span class="rb">추천</span>')
    return "".join(bs[:3])


_namecount = Counter(NAME.get(a, "?") for a in cand)
_sorted = sorted(cand, key=lambda a: -(POP.get(a) or 0))
CHOICES = [
    (f"{NAME.get(a,a)}  ·  #{a}" if _namecount[NAME.get(a, '?')] > 1 else NAME.get(a, str(a)), a)
    for a in _sorted[:6000]
]


def _theme():
    t = gr.themes.Base()
    kw = dict(
        body_background_fill=BG, body_background_fill_dark=BG,
        block_background_fill=CARD, block_background_fill_dark=CARD,
        block_border_color=MUTE, block_border_color_dark=MUTE,
        border_color_primary="#c9b8a8", border_color_primary_dark="#c9b8a8",
        body_text_color=TEXT, body_text_color_dark=TEXT,
        body_text_color_subdued=TEXT2, body_text_color_subdued_dark=TEXT2,
        block_label_text_color=TEXT, block_label_text_color_dark=TEXT,
        block_title_text_color=TEXT, block_title_text_color_dark=TEXT,
        button_primary_background_fill=ACCENT, button_primary_background_fill_dark=ACCENT,
        button_primary_background_fill_hover="#c9a78f", button_primary_background_fill_hover_dark="#c9a78f",
        button_primary_text_color=TEXT, button_primary_text_color_dark=TEXT,
        input_background_fill=PARCH, input_background_fill_dark=PARCH,
    )
    try:
        return t.set(**kw)
    except Exception:
        return t


CSS = """
.reclist{display:flex;flex-direction:column;gap:8px;margin-top:6px;color:#3a332c}
.rc{display:flex;align-items:center;gap:12px;background:#e3d5ca;border:1px solid #c9b8a8;border-radius:10px;padding:11px 14px}
.rk{color:#6f6253;font-size:.82rem;width:22px;text-align:right;font-weight:700}
.rc-main{flex:1;display:flex;flex-direction:column;gap:5px;min-width:0}
.rn{color:#3a332c;font-weight:700;text-decoration:none;font-size:.95rem}
.rn:hover{text-decoration:underline;color:#bc6c25}
.rr{display:flex;flex-wrap:wrap;gap:5px}
.rb{font-size:.72rem;color:#6f6253;background:#edede9;border:1px solid #d6ccc2;border-radius:999px;padding:1px 8px;white-space:nowrap}
.rb.play{color:#7d5a3c;background:#f0e6dc;border-color:#d8c3ae}
.rb.intent{color:#9c5410;background:#f3e6d6;border-color:#e0c39c}
.rs{color:#bc6c25;font-size:.82rem;font-weight:700}
"""


def _empty_fig(msg="게임/의도를 입력하면 추론 과정이 여기 그려집니다"):
    f = go.Figure()
    f.update_layout(template="plotly_white", paper_bgcolor=BG, plot_bgcolor=PARCH,
                    height=460, margin=dict(l=10, r=10, t=10, b=10),
                    xaxis=dict(visible=False), yaxis=dict(visible=False),
                    annotations=[dict(text=msg, showarrow=False, font=dict(color=TEXT2))])
    return f


def _build_fig(liked_ap, rec_ap):
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=_map["x"], y=_map["y"], mode="markers",
                  marker=dict(size=3, color="rgba(140,120,100,0.20)"),
                  hoverinfo="skip", showlegend=False))
    rx = [(MX[a], MY[a], NAME.get(a, a)) for a in rec_ap if a in MX]
    if rx:
        fig.add_trace(go.Scattergl(x=[p[0] for p in rx], y=[p[1] for p in rx],
                      mode="markers+text", text=[p[2] for p in rx], textposition="top center",
                      marker=dict(size=11, color=HILITE, line=dict(width=1, color="#fff")),
                      textfont=dict(size=9, color=HILITE), name="추천"))
    lx = [(MX[a], MY[a], NAME.get(a, a)) for a in liked_ap if a in MX]
    if lx:
        fig.add_trace(go.Scattergl(x=[p[0] for p in lx], y=[p[1] for p in lx],
                      mode="markers+text", text=[p[2] for p in lx], textposition="bottom center",
                      marker=dict(size=16, color=LIKED, symbol="star", line=dict(width=1, color="#fff")),
                      textfont=dict(size=10, color=LIKED), name="즐긴 게임"))
    fig.update_layout(template="plotly_white", paper_bgcolor=BG, plot_bgcolor=PARCH,
                      height=460, margin=dict(l=10, r=10, t=34, b=10),
                      title=dict(text="🧭 임베딩 공간 — 취향(별)에서 추천이 나오는 과정",
                                 font=dict(size=13, color=TEXT)),
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      legend=dict(orientation="h", y=1.02, x=0, font=dict(color=TEXT)))
    return fig


FLOW_CSS = """
<style>
@keyframes flIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
@keyframes flFade{0%,100%{opacity:.3}50%{opacity:1}}
.fl{font-family:-apple-system,BlinkMacSystemFont,'Noto Sans KR',sans-serif;color:#3a332c;max-width:460px;margin:0 auto}
.fl .map{background:#edede9;border:1px solid #d6ccc2;border-radius:12px;display:block;width:100%;height:auto}
.fl .s{opacity:0;animation:flIn .55s ease forwards}
.fl text{font-family:inherit}
.fl .starlab{font-size:11px;fill:#5a4632;font-weight:700}
.fl .qlab{font-size:11px;fill:#bc6c25;font-weight:800}
.fl .cap{font-size:10px;fill:#8a7d6c}
.fl .ring{animation:flFade 1.5s ease-in-out infinite}
.fl ol.steps{list-style:none;padding:0;margin:8px 0 0}
.fl ol.steps li{opacity:0;animation:flIn .5s ease forwards;font-size:.9rem;line-height:1.5;margin:3px 0}
.fl ol.steps b{color:#bc6c25}
.fl .recs{margin-top:8px;display:flex;flex-wrap:wrap;gap:5px}
.fl .rchip{opacity:0;animation:flIn .45s ease forwards;background:#bc6c25;color:#fff;font-weight:700;border-radius:999px;padding:3px 11px;font-size:.78rem}
.fl .mut{color:#8a7d6c;font-weight:400}
</style>
"""


def _flow_html(liked_names, intent, w_intent, rec_names):
    """추론 과정 — 튜토리얼 스타일 '게임 지도' 애니메이션(자동 단계 재생, 배경은 일부 점만)."""
    intent = (intent or "").strip()
    has_l, has_i = bool(liked_names), bool(intent)
    wt, wi = round((1 - w_intent) * 100), round(w_intent * 100)

    def e(s):
        return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    itxt = (intent[:16] + "…") if len(intent) > 16 else intent

    # 배경 '게임 지도' — 전체가 아니라 일부 점만
    bgpts = [(205, 45), (255, 172), (180, 192), (392, 150), (120, 34), (60, 198),
             (416, 55), (345, 182), (270, 60), (150, 104), (232, 118)]
    bg = "".join(f'<circle cx="{x}" cy="{y}" r="4" fill="#c9b8a8"/>' for x, y in bgpts)

    # 즐긴 게임(별) + 취향 자리
    O = (155, 130)
    star_svg = ""
    if has_l:
        sp = [(95, 86), (82, 150)]
        for i, nm in enumerate(liked_names[:2]):
            x, y = sp[i]
            star_svg += (f'<g class="s" style="animation-delay:{0.55 + i * 0.2}s">'
                         f'<text x="{x + 11}" y="{y + 4}" class="starlab">⭐ {e(nm)}</text>'
                         f'<circle cx="{x}" cy="{y}" r="6" fill="#5a4632"/></g>')
        if len(liked_names) > 2:
            star_svg += f'<text x="95" y="178" class="cap">+ 외 {len(liked_names) - 2}개</text>'
        star_svg += ('<g class="s" style="animation-delay:1.2s">'
                     f'<circle cx="{O[0]}" cy="{O[1]}" r="9" fill="none" stroke="#5a4632" stroke-width="2.5"/>'
                     f'<text x="{O[0]}" y="{O[1] + 22}" class="cap" text-anchor="middle">취향 자리</text></g>')
    else:
        O = (100, 116)
        star_svg = (f'<g class="s" style="animation-delay:0.55s">'
                    f'<text x="{O[0]}" y="{O[1]}" class="starlab" text-anchor="middle">✍️ 의도</text></g>')

    # 화살표 → 쿼리 지점 + 검색 링
    Q = (300, 95)
    ql = f'✍️ &ldquo;{e(itxt)}&rdquo; 지점' if has_i else '🧭 취향 지점'
    arrow = (f'<g class="s" style="animation-delay:1.75s">'
             f'<line x1="{O[0] + 14}" y1="{O[1]}" x2="{Q[0] - 16}" y2="{Q[1]}" stroke="#bc6c25" stroke-width="2.5" stroke-dasharray="5 4"/>'
             f'<path d="M{Q[0] - 14},{Q[1]} l-10,-5 l3,5 l-3,5 z" fill="#bc6c25"/>'
             f'<circle class="ring" cx="{Q[0]}" cy="{Q[1]}" r="26" fill="none" stroke="#bc6c25" stroke-width="1.5"/>'
             f'<circle cx="{Q[0]}" cy="{Q[1]}" r="7" fill="#bc6c25"/>'
             f'<text x="{Q[0]}" y="{Q[1] - 30}" class="qlab" text-anchor="middle">{ql}</text></g>')

    # 추천 점(쿼리 근처)
    rp = [(332, 70), (322, 124), (356, 104)]
    recdots = ('<g class="s" style="animation-delay:2.3s">'
               + "".join(f'<circle cx="{x}" cy="{y}" r="7" fill="#606c38" stroke="#fff" stroke-width="1.5"/>'
                         for x, y in rp) + '</g>')

    svg = (f'<svg class="map" viewBox="0 0 440 210" role="img" xmlns="http://www.w3.org/2000/svg">'
           f'<title>추천이 만들어지는 과정</title>'
           f'<g class="s" style="animation-delay:.15s">{bg}</g>{star_svg}{arrow}{recdots}</svg>')

    # 단계 설명(순차 등장)
    steps = ["🎮 <b>즐긴 게임</b>을 &lsquo;게임 지도&rsquo;에 콕 찍어요" if has_l
             else "✍️ <b>의도</b>를 &lsquo;게임 지도&rsquo;의 한 지점으로 바꿔요"]
    if has_l:
        steps.append("🧭 그 위치들의 평균 = <b>당신의 취향 자리</b>")
    if has_l and has_i:
        steps.append(f"✍️ 의도 &lsquo;<b>{e(itxt)}</b>&rsquo;가 취향을 그쪽으로 옮겨요 "
                     f"<span class='mut'>(취향 {wt}%·의도 {wi}%)</span>")
    elif has_l and not has_i:
        steps.append("취향 자리 <b>그대로</b> 검색해요")
    steps.append("🎯 그 자리 <b>근처의 게임</b>을 골라 추천!")
    steps_html = "".join(f'<li style="animation-delay:{0.7 + i * 0.5:.2f}s">{s}</li>'
                         for i, s in enumerate(steps))

    rd = 0.7 + len(steps) * 0.5
    recs = "".join(f'<span class="rchip" style="animation-delay:{rd + i * 0.12:.2f}s">{e(n)}</span>'
                   for i, n in enumerate(rec_names[:5]))

    return (f'{FLOW_CSS}<div class="fl">{svg}'
            f'<ol class="steps">{steps_html}</ol>'
            f'<div class="recs">{recs}</div></div>')


# 한국어 게임 외래어 → 영어 Steam 태그/용어 사전.
#   진단: 게임은 보편적 영어 태그(Souls-like 등)를 가져 영어 검색은 정확하나,
#   한국어 외래어("소울라이크")가 그 태그에 안 붙음. 쿼리에 영어 표준어를 병기해
#   검증된 영어 경로로 결정론적 라우팅(무재학습).
GAMING_GLOSSARY = {
    # 세부 장르 외래어 (영어 태그로 직행)
    "소울라이크": "Souls-like", "소울라이트": "Souls-like", "소울류": "Souls-like",
    "메트로배니아": "Metroidvania", "메트로바니아": "Metroidvania",
    "로그라이크": "Roguelike", "로그라이트": "Roguelite", "로그라잇": "Roguelike",
    "핵앤슬래시": "Hack and Slash", "핵슬": "Hack and Slash", "디아블로류": "Action RPG Hack and Slash",
    "타워디펜스": "Tower Defense", "디펜스": "Tower Defense",
    "방치형": "Idle", "방치": "Idle", "클리커": "Clicker", "클릭커": "Clicker",
    "비주얼노벨": "Visual Novel", "비주얼 노벨": "Visual Novel", "미연시": "Dating Sim", "연애시뮬": "Dating Sim",
    "덱빌딩": "Deckbuilding", "덱빌더": "Deckbuilding", "카드게임": "Card Game",
    "탑다운": "Top-Down", "쿼터뷰": "Isometric", "아이소메트릭": "Isometric",
    "턴제": "Turn-Based", "실시간전략": "Real-Time Strategy", "알티에스": "Real-Time Strategy",
    "도트": "Pixel Graphics", "도트그래픽": "Pixel Graphics", "픽셀": "Pixel Graphics", "픽셀아트": "Pixel Graphics",
    "샌드박스": "Sandbox", "오픈월드": "Open World", "오픈 월드": "Open World",
    "생존공포": "Survival Horror", "생존": "Survival", "서바이벌": "Survival",
    "크래프팅": "Crafting", "제작": "Crafting", "도시건설": "City Builder", "건설": "Building",
    "공포": "Horror", "호러": "Horror", "좀비": "Zombies",
    "협동": "Co-op", "코옵": "Co-op", "코업": "Co-op",
    "멀티플레이": "Multiplayer", "싱글플레이": "Singleplayer",
    "배틀로얄": "Battle Royale", "배틀로열": "Battle Royale",
    "모바": "MOBA", "에이오에스": "MOBA",
    "에프피에스": "FPS", "1인칭슈팅": "FPS", "3인칭": "Third Person", "삼인칭": "Third Person",
    "슈팅": "Shooter", "슈터": "Shooter", "탄막": "Bullet Hell", "불릿헬": "Bullet Hell",
    "플랫포머": "Platformer", "플랫폼게임": "Platformer",
    "격투": "Fighting", "대전격투": "Fighting",
    "레이싱": "Racing", "리듬게임": "Rhythm", "리듬": "Rhythm",
    "퍼즐": "Puzzle", "방탈출": "Escape Room", "추리": "Detective Mystery",
    "잠입": "Stealth", "스텔스": "Stealth",
    "경영시뮬": "Management Simulation", "경영": "Management", "농장": "Farming Sim", "농사": "Farming Sim",
    "수집형": "Collectathon", "가챠": "Gacha",
    "오토배틀러": "Auto Battler", "오토체스": "Auto Battler", "자동전투": "Auto Battler",
    "비대칭": "Asymmetric",
    "제이알피지": "JRPG", "알피지": "RPG", "시뮬레이션": "Simulation", "시뮬": "Simulation",
    "어드벤처": "Adventure", "액션": "Action", "전략": "Strategy", "인디": "Indie", "캐주얼": "Casual",
    # 커뮤니티 표현
    "명작": "critically acclaimed", "띵작": "critically acclaimed", "갓겜": "critically acclaimed",
    "노가다": "Grinding", "그라인딩": "Grinding", "하드코어": "Hardcore Difficult", "고난도": "Difficult",
}


def _expand_intent(text):
    """의도 텍스트에 매칭된 외래어의 영어 표준어를 병기(중복 제거). 원문은 보존."""
    hits = []
    for ko, en in GAMING_GLOSSARY.items():
        if ko in text and en not in hits:
            hits.append(en)
    return f"{text}  {' '.join(hits)}" if hits else text


def recommend(liked, intent, w_intent, topn):
    liked = liked or []
    intent = (intent or "").strip()
    if not liked and not intent:
        return ("<p style='color:#6f6253'>게임을 선택하거나 의도를 입력하세요.</p>",
                "<p style='color:#6f6253'>추천을 실행하면 추론 과정이 애니메이션으로 재생됩니다.</p>",
                _empty_fig())
    n = len(cand)
    score = np.zeros(n, np.float32)
    rows = [cand_idx[a] for a in liked if a in cand_idx]
    w = float(w_intent)
    cooc_src = {}                       # 후보idx → (공동플레이 기여 1위 즐긴게임 appid, 가중치)
    liked_genres = set()
    for a in liked:
        if a in cand_idx:
            liked_genres |= set(_genre_list(a))
    if rows:
        # 취향 = item2vec 코사인 + 공동플레이(cooc)를 RRF(랭크 융합)로 결합 → 스케일에 강건
        emb_s = collab_n @ (collab_n[rows].mean(0) / (np.linalg.norm(collab_n[rows].mean(0)) + 1e-9))
        cooc_s = np.zeros(n, np.float32)
        for a in liked:
            ri = _cooc_row.get(int(a))
            if ri is None:
                continue
            for nbr, wv in zip(_ck_nb_appid[ri], _ck_wt[ri]):
                ci = cand_idx.get(int(nbr))
                if ci is not None:
                    cooc_s[ci] += wv
                    if wv > cooc_src.get(ci, (0, 0.0))[1]:
                        cooc_src[ci] = (int(a), float(wv))   # 근거 표시용 출처 추적
        taste = 1.0 / (60 + _ranks(emb_s)) + 1.0 / (60 + _ranks(cooc_s))
        score += (1 - w) * _norm(taste)
    intent_s = None
    if intent:
        qi = encoder().encode(_expand_intent(intent), normalize_embeddings=True)
        intent_s = content_c @ qi
        score += w * _norm(intent_s)
    intent_hi = float(np.quantile(intent_s, 0.75)) if intent_s is not None else None  # 상위25% 의도매칭만 '의도 부합'
    for r in rows:
        score[r] = -np.inf
    top = np.argsort(-score)[: int(topn)]
    rec_ap = [cand[r] for r in top]
    out = []
    for i, r in enumerate(top, 1):
        a = cand[r]; url = f"https://store.steampowered.com/app/{a}"
        reasons = _reason_badges(a, int(r), cooc_src, liked_genres, intent, intent_s, intent_hi)
        out.append(f'<div class="rc"><span class="rk">{i}</span>'
                   f'<div class="rc-main">'
                   f'<a class="rn" href="{url}" target="_blank">{NAME.get(a,a)}</a>'
                   f'<div class="rr">{reasons}</div></div>'
                   f'<span class="rs">{score[r]:.3f}</span></div>')
    html = f"<style>{CSS}</style><div class='reclist'>" + "".join(out) + "</div>"
    liked_in = [a for a in liked if a in cand_idx]
    flow = _flow_html([NAME.get(a, a) for a in liked_in], intent, float(w_intent),
                      [NAME.get(a, a) for a in rec_ap])
    fig = _build_fig(liked_in, rec_ap)
    return html, flow, fig


def epoch_fig(ep):
    """학습 진행 슬라이더 — 선별 스냅샷의 UMAP 궤적(게임이 '이웃'으로 뭉치는 실제 과정)."""
    ep = int(ep)
    f = TRAIN["frames"][ep]
    bnd = TRAIN["bounds"]
    loss = TRAIN["losses"][ep] if ep < len(TRAIN["losses"]) else None
    sub = "🎲 랜덤 초기화 (학습 전)" if loss is None else f"대조손실 {loss}"
    fig = go.Figure(go.Scattergl(
        x=[p[0] for p in f], y=[p[1] for p in f], mode="markers",
        marker=dict(size=6, color=_gcolor, line=dict(width=0)),
        text=TRAIN["names"], hoverinfo="text", showlegend=False))
    fig.update_layout(template="plotly_white", paper_bgcolor=BG, plot_bgcolor=PARCH, height=440,
                      margin=dict(l=10, r=10, t=40, b=10),
                      title=dict(text=f"학습 진행 {ep}/{N_EPOCHS} · {sub} — 게임들이 '이웃'으로 뭉치는 실제 궤적 (색=이웃 그룹)",
                                 font=dict(color=TEXT, size=12)),
                      xaxis=dict(visible=False, range=[bnd["xlo"], bnd["xhi"]]),
                      yaxis=dict(visible=False, range=[bnd["ylo"], bnd["yhi"]]))
    return fig


ARCH_HTML = """
<style>.arch{color:#3a332c;font-size:.86rem}.arch .row{display:flex;gap:6px;justify-content:center;flex-wrap:wrap;margin:5px 0}
.arch .nd{background:#edede9;border:1px solid #c9b8a8;border-radius:8px;padding:7px 11px;font-size:.8rem;text-align:center}
.arch .nd b{color:#3a332c}.arch .ar{color:#9c6644;font-weight:700}.arch .nt{color:#6f6253;font-size:.76rem;margin-top:8px}</style>
<div class="arch">
<div class="row"><div class="nd">게임 A<br><b>(앵커)</b></div><div class="nd">게임 B<br><b>(양성쌍)</b></div></div>
<div class="row"><span class="ar">↓ 임베딩 룩업</span></div>
<div class="row"><div class="nd"><b>임베딩 테이블</b> · 게임 N × 차원 D</div></div>
<div class="row"><span class="ar">↓ L2 정규화 → 유사도 행렬(배치 B×B)</span></div>
<div class="row"><div class="nd"><b>InfoNCE</b> · 양성쌍 가깝게 / 배치 내 음성 멀게</div></div>
<div class="nt">· 협업 임베딩(item2vec식): 게임ID→벡터, 공동플레이 쌍 학습 (위 그림)<br>
· 콘텐츠 임베딩: 텍스트(이름·태그·설명)→다국어 트랜스포머→벡터<br>
· 추천 = 협업(취향) + 콘텐츠(의도) 가중 결합 → 하이브리드</div></div>
"""

# 비전공자용 자동재생 애니메이션 — "AI가 비슷한 게임을 배우는 과정"(용어 없이 비유).
#   순수 CSS 키프레임(JS 불필요, 모바일 자동재생). 클래스는 la- 로 네임스페이스(Gradio 충돌 방지).
LEARN_ANIM = """
<div class="la-wrap">
  <h2 class="la-sr">AI가 게임 추천을 학습하는 과정: 같은 사람이 함께 즐긴 게임을 서로 가까이 끌어당겨 '동네'(군집)를 만들고, 내가 좋아한 게임의 동네에서 이웃 게임을 추천합니다.</h2>
  <div class="la-title">🧠 AI는 어떻게 <b>&ldquo;비슷한 게임&rdquo;</b>을 배울까?</div>
  <div class="la-scene">
    <svg viewBox="0 0 640 320" role="img" xmlns="http://www.w3.org/2000/svg">
      <title>AI가 게임을 군집으로 학습하는 과정</title>
      <g class="la-labels">
        <rect x="70" y="278" width="118" height="26" rx="13" fill="#9c6644"/>
        <text x="129" y="295" class="la-lbl">🎮 액션 동네</text>
        <rect x="270" y="30" width="120" height="26" rx="13" fill="#5c6b73"/>
        <text x="330" y="47" class="la-lbl">👻 공포 동네</text>
        <rect x="442" y="278" width="118" height="26" rx="13" fill="#606c38"/>
        <text x="501" y="295" class="la-lbl">🌱 농장 동네</text>
      </g>
      <g class="la-dot" style="--dx:-350px;--dy:160px"><circle cx="480" cy="90"  r="10" fill="#9c6644"/></g>
      <g class="la-dot" style="--dx:-140px;--dy:10px"><circle cx="300" cy="250" r="10" fill="#9c6644"/></g>
      <g class="la-dot" style="--dx:40px;--dy:180px"><circle cx="90"  cy="60"  r="10" fill="#9c6644"/></g>
      <g class="la-dot" style="--dx:-390px;--dy:-60px"><circle cx="560" cy="290" r="10" fill="#9c6644"/></g>
      <g class="la-dot" style="--dx:200px;--dy:-90px"><circle cx="120" cy="210" r="10" fill="#5c6b73"/></g>
      <g class="la-dot" style="--dx:-195px;--dy:-95px"><circle cx="540" cy="210" r="10" fill="#5c6b73"/></g>
      <g class="la-dot" style="--dx:75px;--dy:-190px"><circle cx="250" cy="300" r="10" fill="#5c6b73"/></g>
      <g class="la-dot" style="--dx:-80px;--dy:95px"><circle cx="420" cy="40"  r="10" fill="#5c6b73"/></g>
      <g class="la-dot" style="--dx:410px;--dy:-45px"><circle cx="80"  cy="290" r="10" fill="#606c38"/></g>
      <g class="la-dot" style="--dx:165px;--dy:190px"><circle cx="350" cy="70"  r="10" fill="#606c38"/></g>
      <g class="la-dot" style="--dx:300px;--dy:140px"><circle cx="200" cy="130" r="10" fill="#606c38"/></g>
      <g class="la-dot" style="--dx:-80px;--dy:125px"><circle cx="600" cy="110" r="10" fill="#606c38"/></g>
      <g class="la-pull">
        <line x1="480" y1="90" x2="300" y2="250" stroke="#bc6c25" stroke-width="2.5" stroke-dasharray="6 5"/>
        <rect x="330" y="150" width="150" height="28" rx="14" fill="#bc6c25"/>
        <text x="405" y="169" class="la-badge">👥 같이 즐긴 사람</text>
      </g>
      <g class="la-reco">
        <circle class="la-ring" cx="501" cy="250" r="18" fill="none" stroke="#bc6c25" stroke-width="2.5"/>
        <text x="501" y="256" class="la-star">⭐</text>
        <rect x="446" y="205" width="110" height="24" rx="12" fill="#3a332c"/>
        <text x="501" y="222" class="la-mine">내가 좋아한 게임</text>
        <line x1="501" y1="250" x2="470" y2="245" stroke="#bc6c25" stroke-width="3"/>
        <line x1="501" y1="250" x2="515" y2="260" stroke="#bc6c25" stroke-width="3"/>
        <rect x="548" y="238" width="78" height="24" rx="12" fill="#606c38"/>
        <text x="587" y="255" class="la-pick">추천 &check;</text>
      </g>
    </svg>
  </div>
  <div class="la-caps">
    <div class="la-cap la-c1">① 처음엔 AI도 몰라요 — 어떤 게임이 비슷한지 몰라 뒤죽박죽 흩어져 있죠.</div>
    <div class="la-cap la-c2">② &ldquo;같은 사람이 <b>둘 다</b> 즐겼네?&rdquo; — 이런 게임 쌍을 발견하면&hellip;</div>
    <div class="la-cap la-c3">③ 그 두 게임을 <b>서로 가까이 끌어당깁니다.</b> (상관없는 건 멀어지고요)</div>
    <div class="la-cap la-c4">④ 수백만 번 반복하면 — 비슷한 게임끼리 <b>&lsquo;동네&rsquo;</b>가 생겨요.</div>
    <div class="la-cap la-c5">⑤ 추천 = <b>내 게임의 &lsquo;동네&rsquo;</b>에서 이웃을 골라주는 것! 🎯</div>
  </div>
  <div class="la-hint">▶ 자동 반복 재생 — 사람이 함께 즐긴 게임을 모아 &lsquo;동네&rsquo;를 만드는 게 학습의 전부예요</div>
</div>
<style>
.la-sr{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0)}
.la-wrap{background:#f5ebe0;border:1px solid #d6ccc2;border-radius:16px;padding:16px 14px 12px;max-width:660px;margin:0 auto;font-family:-apple-system,BlinkMacSystemFont,'Noto Sans KR',sans-serif;color:#3a332c}
.la-title{text-align:center;font-size:1.12rem;font-weight:800;margin-bottom:6px;color:#3a332c}
.la-title b{color:#bc6c25}
.la-scene{background:#edede9;border:1px solid #d6ccc2;border-radius:12px}
.la-scene svg{display:block;width:100%;height:auto;animation:la_scene 16s linear infinite}
.la-dot{animation:la_gather 16s ease-in-out infinite}
.la-dot circle{filter:drop-shadow(0 1px 1px rgba(58,51,44,.18))}
.la-labels{opacity:0;animation:la_labels 16s ease infinite}
.la-lbl{fill:#fff;font-size:14px;font-weight:800;text-anchor:middle}
.la-pull{opacity:0;animation:la_pull 16s ease infinite}
.la-badge{fill:#fff;font-size:13px;font-weight:800;text-anchor:middle}
.la-reco{opacity:0;animation:la_reco 16s ease infinite}
.la-star{font-size:22px;text-anchor:middle}
.la-mine{fill:#fff;font-size:12px;font-weight:700;text-anchor:middle}
.la-pick{fill:#fff;font-size:13px;font-weight:800;text-anchor:middle}
.la-ring{animation:la_ring 1.3s ease-in-out infinite}
.la-caps{position:relative;height:54px;margin-top:10px}
.la-cap{position:absolute;left:0;right:0;text-align:center;font-size:.98rem;line-height:1.35;opacity:0;padding:0 6px;color:#3a332c}
.la-cap b{color:#bc6c25}
.la-c1{animation:la_cap1 16s ease infinite}
.la-c2{animation:la_cap2 16s ease infinite}
.la-c3{animation:la_cap3 16s ease infinite}
.la-c4{animation:la_cap4 16s ease infinite}
.la-c5{animation:la_cap5 16s ease infinite}
.la-hint{text-align:center;font-size:.76rem;color:#6f6253;margin-top:8px}
@keyframes la_scene{0%{opacity:0}3%{opacity:1}94%{opacity:1}99%{opacity:0}100%{opacity:0}}
@keyframes la_gather{0%,30%{transform:translate(0,0)}48%,100%{transform:translate(var(--dx),var(--dy))}}
@keyframes la_pull{0%,12%{opacity:0}16%,29%{opacity:1}33%,100%{opacity:0}}
@keyframes la_labels{0%,46%{opacity:0}54%,100%{opacity:1}}
@keyframes la_reco{0%,68%{opacity:0}75%,94%{opacity:1}99%,100%{opacity:0}}
@keyframes la_ring{0%,100%{opacity:.3;transform:scale(.9);transform-origin:501px 250px}50%{opacity:1;transform:scale(1.15);transform-origin:501px 250px}}
@keyframes la_cap1{0%,11%{opacity:1}15%,100%{opacity:0}}
@keyframes la_cap2{0%,13%{opacity:0}17%,28%{opacity:1}31%,100%{opacity:0}}
@keyframes la_cap3{0%,29%{opacity:0}34%,46%{opacity:1}49%,100%{opacity:0}}
@keyframes la_cap4{0%,47%{opacity:0}55%,67%{opacity:1}71%,100%{opacity:0}}
@keyframes la_cap5{0%,69%{opacity:0}76%,93%{opacity:1}97%,100%{opacity:0}}
</style>
"""

# 입력칸/드롭다운 가시성 (밝은 배경 + 다크 글자)
GLOBAL_CSS = """
input, textarea { color:#3a332c !important; background:#ffffff !important; }
input::placeholder, textarea::placeholder { color:#9a8d7c !important; }
ul.options { background:#ffffff !important; border:1px solid #c9b8a8 !important; }
ul.options li, li.item, .options .item { background:#ffffff !important; color:#3a332c !important; }
ul.options li:hover, li.item:hover, .item.selected, .item.active { background:#e3d5ca !important; color:#3a332c !important; }
.token, span.token { background:#d5bdaf !important; color:#3a332c !important; }
.token .token-remove, .token svg { color:#3a332c !important; fill:#3a332c !important; }
"""

with gr.Blocks(title="SteamFit", theme=_theme(), css=GLOBAL_CSS) as demo:
    gr.Markdown("# 🎮 SteamFit — Steam 게임 추천\n"
                "**취향**(즐긴 게임) + **의도**(원하는 특징 텍스트)를 결합한 하이브리드 추천. "
                "Steam 상점이 못 하는 *의도 반영*이 핵심. **한국어·영어 의도 모두 지원** 🇰🇷🇺🇸")
    with gr.Tabs():
        with gr.Tab("🎮 추천"):
            liked = gr.Dropdown(CHOICES, multiselect=True, label="🎮 즐겨한 게임 (검색해서 선택)", filterable=True)
            intent = gr.Textbox(label="✍️ 원하는 특징 / 의도",
                                placeholder="예: relaxing open world crafting / 협동 호러 / competitive multiplayer")
            with gr.Row():
                w_intent = gr.Slider(0, 1, value=0.4, step=0.1, label="의도 반영 비중 (0=취향만 · 1=의도만)")
                topn = gr.Slider(5, 20, value=10, step=1, label="추천 개수")
            btn = gr.Button("추천 받기", variant="primary")
            with gr.Row():
                with gr.Column(scale=1):
                    out = gr.HTML(label="추천 결과")
                with gr.Column(scale=1):
                    gr.Markdown("##### 🎬 추론 과정 (자동 재생)")
                    out_flow = gr.HTML()
            with gr.Accordion("🧭 임베딩 공간 2D 맵 (취향→추천)", open=False):
                out_plot = gr.Plot()
            btn.click(recommend, [liked, intent, w_intent, topn], [out, out_flow, out_plot])
        with gr.Tab("🎬 학습 과정"):
            gr.Markdown("### AI가 ‘비슷한 게임’을 배우는 과정\n"
                        "용어 없이 — 아래 애니메이션만 보면 학습 원리가 한눈에 들어옵니다. (자동 반복)")
            gr.HTML(LEARN_ANIM)
            with gr.Accordion("🔬 실제 학습 데이터로 보기 (자세히)", open=False):
                gr.Markdown("위 비유가 **실제로** 일어난 궤적입니다. **슬라이더를 왼→오른쪽으로 드래그**하면 "
                            "랜덤으로 흩어진 900개 게임이 학습을 거치며 **‘이웃 그룹’(색)으로 뭉치는 실제 과정**이 보입니다. "
                            "(협업 임베딩을 UMAP으로 투영 · 점 위에 마우스=게임명)")
                ep_slider = gr.Slider(0, N_EPOCHS, value=0, step=1,
                                      label="학습 진행 — 드래그 (0=랜덤 초기 → 끝=군집 형성)")
                train_plot = gr.Plot(value=epoch_fig(0))
                ep_slider.change(epoch_fig, ep_slider, train_plot)
                gr.Markdown("#### 🧩 모델 구조 (대조학습)")
                gr.HTML(ARCH_HTML)
    gr.Markdown("<small>협업 임베딩(item2vec식 직접 학습) + 콘텐츠 임베딩 · 공식 Steam API 데이터 1,021만 리뷰</small>")


if __name__ == "__main__":
    demo.launch()
