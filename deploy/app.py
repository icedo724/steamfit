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

ROOT = Path(__file__).resolve().parent

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
        _encoder = SentenceTransformer(str(ROOT / "steam-embed-model"))
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


def recommend(liked, intent, w_intent, topn):
    liked = liked or []
    intent = (intent or "").strip()
    if not liked and not intent:
        return "<p style='color:#9fb2c6'>게임을 선택하거나 의도를 입력하세요.</p>"
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
    out = []
    for i, r in enumerate(top, 1):
        a = cand[r]; url = f"https://store.steampowered.com/app/{a}"
        out.append(f'<div class="rc"><span class="rk">{i}</span>'
                   f'<a class="rn" href="{url}" target="_blank">{NAME.get(a,a)}</a>'
                   f'<span class="rg">{_genres(a)}</span>'
                   f'<span class="rs">{score[r]:.3f}</span></div>')
    return f"<style>{CSS}</style><div class='reclist'>" + "".join(out) + "</div>"


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
    out = gr.HTML()
    btn.click(recommend, [liked, intent, w_intent, topn], out)
    gr.Markdown("<small>협업 임베딩(item2vec식 직접 학습) + 콘텐츠 임베딩 · 공식 Steam API 데이터 1,021만 리뷰</small>")


if __name__ == "__main__":
    demo.launch()
