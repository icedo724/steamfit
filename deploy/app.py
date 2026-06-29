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
content_c = np.stack([content[t_row[a]] for a in cand])
cand_idx = {a: i for i, a in enumerate(cand)}

# 학습 과정 데이터(에폭별 임베딩 스냅샷)
TRAIN = json.loads((ROOT / "training_frames.json").read_text(encoding="utf-8"))
N_EPOCHS = len(TRAIN["frames"]) - 1
_gset = list(dict.fromkeys(TRAIN["genres"]))
_gcolor = [GENRE_PAL[_gset.index(g) % len(GENRE_PAL)] for g in TRAIN["genres"]]

_encoder = None


def encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer("mininiming/steamfit-encoder")
    return _encoder


def _norm(v):
    return (v - v.min()) / (v.max() - v.min() + 1e-9)


def _genres(a):
    try:
        return ", ".join(json.loads(GENRE.get(a) or "[]")[:3])
    except Exception:
        return ""


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
.rn{color:#3a332c;font-weight:700;text-decoration:none;flex:1;font-size:.95rem}
.rn:hover{text-decoration:underline;color:#bc6c25}
.rg{color:#6f6253;font-size:.78rem}
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
@keyframes fin{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:translateY(0)}}
@keyframes pulse{0%,100%{opacity:.4}50%{opacity:1}}
@keyframes dash{to{background-position:0 -28px}}
.flow{font-family:-apple-system,BlinkMacSystemFont,'Noto Sans KR',sans-serif;color:#3a332c;max-width:520px;margin:0 auto}
.flow .st{opacity:0;animation:fin .5s ease forwards}
.flow .box{background:#edede9;border:1px solid #c9b8a8;border-radius:12px;padding:11px 14px;text-align:center}
.flow .hd{font-size:.82rem;font-weight:700;margin-bottom:4px;color:#3a332c}
.flow .conn{width:3px;height:26px;margin:3px auto;border-radius:2px;
  background:repeating-linear-gradient(#b08968 0 7px,transparent 7px 14px);background-size:3px 28px;animation:dash .7s linear infinite}
.flow .chip{display:inline-block;background:#e3d5ca;border-radius:999px;padding:4px 10px;margin:3px;font-size:.78rem;color:#3a332c}
.flow .chip.rec{background:#bc6c25;border:1px solid #a85a18;color:#fff;font-weight:700;opacity:0;animation:fin .4s forwards}
.flow .vec{display:inline-flex;gap:3px;margin-top:6px;height:20px;align-items:flex-end}
.flow .vec i{width:6px;border-radius:2px;background:#9c6644;animation:pulse 1.1s infinite}
.flow .vec.i i{background:#6b705c}.flow .vec.q i{background:#bc6c25}
.flow .lbl{font-size:.72rem;color:#6f6253;margin:2px 0}
.flow .mix{display:flex;gap:8px}.flow .mix>div{flex:1}
.flow .mut{color:#8a7d6c}
</style>
"""


def _bars(color_cls, seed):
    import random
    rng = random.Random(seed)
    hs = [rng.randint(6, 20) for _ in range(10)]
    bs = "".join(f'<i style="height:{h}px;animation-delay:{i*.05}s"></i>' for i, h in enumerate(hs))
    return f'<div class="vec {color_cls}">{bs}</div>'


def _flow_html(liked_names, intent, w_intent, rec_names):
    wt, wi = round((1 - w_intent) * 100), round(w_intent * 100)
    lc = "".join(f'<span class="chip">{n}</span>' for n in liked_names[:5]) or '<span class="mut">없음</span>'
    rc = "".join(f'<span class="chip rec" style="animation-delay:{3.0+i*.18:.2f}s">{n}</span>'
                 for i, n in enumerate(rec_names[:6]))
    intent_box = (f'<div class="st box" style="animation-delay:1.4s"><div class="hd">✍️ 의도</div>'
                  f'<div class="lbl">"{intent}"</div>{_bars("i", 2)}</div>') if intent.strip() else \
                 '<div class="st box mut" style="animation-delay:1.4s">✍️ 의도 없음 (취향만)</div>'
    return f"""{FLOW_CSS}<div class="flow">
  <div class="st box" style="animation-delay:.1s"><div class="hd">🎮 즐긴 게임</div>{lc}</div>
  <div class="st conn" style="animation-delay:.7s"></div>
  <div class="st lbl" style="animation-delay:.7s">평균 풀링 ↓</div>
  <div class="st box" style="animation-delay:.9s"><div class="hd">🧭 취향 벡터</div>{_bars("", 1)}</div>
  <div class="st conn" style="animation-delay:1.3s"></div>
  <div class="mix">
    <div class="st box" style="animation-delay:.9s"><div class="hd">위 취향</div></div>
    {intent_box}
  </div>
  <div class="st conn" style="animation-delay:2.0s"></div>
  <div class="st lbl" style="animation-delay:2.0s">⊕ 가중 결합 (취향 {wt}% · 의도 {wi}%) ↓</div>
  <div class="st box" style="animation-delay:2.2s"><div class="hd">🎯 쿼리 벡터</div>{_bars("q", 3)}</div>
  <div class="st conn" style="animation-delay:2.6s"></div>
  <div class="st lbl" style="animation-delay:2.6s">🔍 12,000개 게임에서 벡터 검색 ↓</div>
  <div class="st box" style="animation-delay:2.9s"><div class="hd">✅ 추천</div><div>{rc}</div></div>
</div>"""


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
    if rows:
        q = collab_c[rows].mean(0); q /= np.linalg.norm(q) + 1e-9
        score += (1 - float(w_intent)) * _norm(collab_c @ q)
    if intent:
        qi = encoder().encode(intent, normalize_embeddings=True)
        score += float(w_intent) * _norm(content_c @ qi)
    for r in rows:
        score[r] = -np.inf
    top = np.argsort(-score)[: int(topn)]
    rec_ap = [cand[r] for r in top]
    out = []
    for i, r in enumerate(top, 1):
        a = cand[r]; url = f"https://store.steampowered.com/app/{a}"
        out.append(f'<div class="rc"><span class="rk">{i}</span>'
                   f'<a class="rn" href="{url}" target="_blank">{NAME.get(a,a)}</a>'
                   f'<span class="rg">{_genres(a)}</span>'
                   f'<span class="rs">{score[r]:.3f}</span></div>')
    html = f"<style>{CSS}</style><div class='reclist'>" + "".join(out) + "</div>"
    liked_in = [a for a in liked if a in cand_idx]
    flow = _flow_html([NAME.get(a, a) for a in liked_in], intent, float(w_intent),
                      [NAME.get(a, a) for a in rec_ap])
    fig = _build_fig(liked_in, rec_ap)
    return html, flow, fig


def epoch_fig(ep):
    """학습 과정 — 슬라이더가 가리키는 에폭의 임베딩 산점도 (모바일 친화: 슬라이더 드래그)."""
    ep = int(ep)
    f = TRAIN["frames"][ep]
    bnd = TRAIN["bounds"]
    loss = TRAIN["losses"][ep - 1] if ep > 0 else "—"
    fig = go.Figure(go.Scattergl(
        x=[p[0] for p in f], y=[p[1] for p in f], mode="markers",
        marker=dict(size=5, color=_gcolor), text=TRAIN["names"], hoverinfo="text", showlegend=False))
    fig.update_layout(template="plotly_white", paper_bgcolor=BG, plot_bgcolor=PARCH, height=440,
                      margin=dict(l=10, r=10, t=40, b=10),
                      title=dict(text=f"에폭 {ep} · 대조손실 {loss} — 게임 임베딩 군집화 (점=게임, 색=장르)",
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
            gr.Markdown("### 대조학습으로 임베딩이 군집을 형성하는 과정\n"
                        "**아래 슬라이더를 드래그**하면 에폭이 진행되며 무작위로 흩어진 게임들이 "
                        "군집으로 뭉칩니다. (모바일에서도 동작)")
            ep_slider = gr.Slider(0, N_EPOCHS, value=0, step=1, label="에폭 — 드래그해서 학습 진행 보기")
            train_plot = gr.Plot(value=epoch_fig(0))
            ep_slider.change(epoch_fig, ep_slider, train_plot)
            gr.Markdown("#### 🧩 모델 구조 (대조학습)")
            gr.HTML(ARCH_HTML)
    gr.Markdown("<small>협업 임베딩(item2vec식 직접 학습) + 콘텐츠 임베딩 · 공식 Steam API 데이터 1,021만 리뷰</small>")


if __name__ == "__main__":
    demo.launch()
