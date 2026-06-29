"""SteamFit — 취향+의도 하이브리드 게임 추천 (HF Spaces, 자체 포함).

취향(즐긴 게임 → 협업 임베딩) + 의도(텍스트 → 콘텐츠 임베딩) 결합.
모델/임베딩은 Space에 함께 업로드된 파일에서 로드.
"""
import json
from collections import Counter
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent

# 추론 시각화용 2D 맵 (협업 임베딩 UMAP 투영)
_map = pd.read_parquet(ROOT / "map2d.parquet")
MX = dict(zip(_map["appid"], _map["x"]))
MY = dict(zip(_map["appid"], _map["y"]))

games = pd.read_parquet(ROOT / "games_lookup.parquet")
NAME = dict(zip(games["appid"], games["name"]))
GENRE = dict(zip(games["appid"], games["genres"]))
POP = dict(zip(games["appid"], games["recommendations_total"]))

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

_encoder = None


def encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        # 모델은 별도 Hub 저장소에서 로드(Space 용량 절약)
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

CSS = """
.reclist{display:flex;flex-direction:column;gap:8px;margin-top:6px}
.rc{display:flex;align-items:center;gap:12px;background:#16212e;border:1px solid #27384b;border-radius:10px;padding:10px 14px}
.rk{color:#9fb2c6;font-size:.8rem;width:20px;text-align:right}
.rn{color:#4f9bff;font-weight:700;text-decoration:none;flex:1}
.rg{color:#9fb2c6;font-size:.78rem}.rs{color:#39d98a;font-size:.78rem}
"""


def _empty_fig(msg="게임/의도를 입력하면 추론 과정이 여기 그려집니다"):
    f = go.Figure()
    f.update_layout(template="plotly_dark", paper_bgcolor="#0b1622", plot_bgcolor="#0b1622",
                    height=460, margin=dict(l=10, r=10, t=10, b=10),
                    xaxis=dict(visible=False), yaxis=dict(visible=False),
                    annotations=[dict(text=msg, showarrow=False, font=dict(color="#9fb2c6"))])
    return f


def _build_fig(liked_ap, rec_ap):
    """임베딩 2D 맵: 전체(흐림) + 즐긴 게임(별) + 추천(초록)."""
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=_map["x"], y=_map["y"], mode="markers",
                  marker=dict(size=3, color="rgba(120,140,160,0.22)"),
                  hoverinfo="skip", showlegend=False))
    rx = [(MX[a], MY[a], NAME.get(a, a)) for a in rec_ap if a in MX]
    if rx:
        fig.add_trace(go.Scattergl(x=[p[0] for p in rx], y=[p[1] for p in rx],
                      mode="markers+text", text=[p[2] for p in rx], textposition="top center",
                      marker=dict(size=11, color="#39d98a", line=dict(width=1, color="#0b1622")),
                      textfont=dict(size=9, color="#39d98a"), name="추천"))
    lx = [(MX[a], MY[a], NAME.get(a, a)) for a in liked_ap if a in MX]
    if lx:
        fig.add_trace(go.Scattergl(x=[p[0] for p in lx], y=[p[1] for p in lx],
                      mode="markers+text", text=[p[2] for p in lx], textposition="bottom center",
                      marker=dict(size=16, color="#4f9bff", symbol="star", line=dict(width=1, color="#fff")),
                      textfont=dict(size=10, color="#7fb0ff"), name="즐긴 게임"))
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0b1622", plot_bgcolor="#0b1622",
                      height=460, margin=dict(l=10, r=10, t=34, b=10),
                      title=dict(text="🧭 임베딩 공간 — 취향(별)에서 추천(초록)이 나오는 과정", font=dict(size=13)),
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      legend=dict(orientation="h", y=1.02, x=0))
    return fig


FLOW_CSS = """
<style>
@keyframes fin{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:translateY(0)}}
@keyframes pulse{0%,100%{opacity:.35}50%{opacity:1}}
@keyframes grow{from{height:4px}to{}}
@keyframes dash{to{background-position:0 -28px}}
.flow{font-family:-apple-system,BlinkMacSystemFont,'Noto Sans KR',sans-serif;color:#e7eef6;max-width:520px;margin:0 auto}
.flow .st{opacity:0;animation:fin .5s ease forwards}
.flow .box{background:#16212e;border:1px solid #27384b;border-radius:12px;padding:11px 14px;text-align:center}
.flow .hd{font-size:.82rem;font-weight:700;margin-bottom:4px}
.flow .conn{width:3px;height:26px;margin:3px auto;border-radius:2px;
  background:repeating-linear-gradient(#4f9bff 0 7px,transparent 7px 14px);background-size:3px 28px;animation:dash .7s linear infinite}
.flow .chip{display:inline-block;background:#22344a;border-radius:999px;padding:4px 10px;margin:3px;font-size:.78rem}
.flow .chip.rec{background:#16352604;border:1px solid #2e6b4a;color:#39d98a;opacity:0;animation:fin .4s forwards}
.flow .vec{display:inline-flex;gap:3px;margin-top:6px;height:20px;align-items:flex-end}
.flow .vec i{width:6px;border-radius:2px;background:#4f9bff;animation:pulse 1.1s infinite}
.flow .vec.i i{background:#f5b54a}.flow .vec.q i{background:#39d98a}
.flow .lbl{font-size:.72rem;color:#9fb2c6;margin:2px 0}
.flow .mix{display:flex;gap:8px}.flow .mix>div{flex:1}
.flow .mut{color:#5b6b7d}
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
        return ("<p style='color:#9fb2c6'>게임을 선택하거나 의도를 입력하세요.</p>",
                "<p style='color:#9fb2c6'>추천을 실행하면 추론 과정이 애니메이션으로 재생됩니다.</p>",
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


with gr.Blocks(title="SteamFit") as demo:
    gr.Markdown("# 🎮 SteamFit — Steam 게임 추천\n"
                "**취향**(즐긴 게임) + **의도**(원하는 특징 텍스트)를 결합한 하이브리드 추천. "
                "Steam 상점이 못 하는 *의도 반영*이 핵심. **한국어·영어 의도 모두 지원** 🇰🇷🇺🇸")
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
    gr.Markdown("<small>협업 임베딩(item2vec식 직접 학습) + 콘텐츠 임베딩 · 공식 Steam API 데이터 1,021만 리뷰 · "
                "맵: 협업 임베딩 UMAP 2D (게임이 플레이 성향별로 군집)</small>")


if __name__ == "__main__":
    demo.launch()
