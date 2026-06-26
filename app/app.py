"""Steam 게임 추천기 — Gradio 데모 (로컬).

취향(즐긴 게임 → 협업 임베딩) + 의도(텍스트 → 콘텐츠 임베딩) 하이브리드 추천.
실행:  python app/app.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

import gradio as gr
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from recommend import Recommender

rec = Recommender()

# 표시용 메타(장르·인기도)
_g = pd.read_parquet(ROOT / "dataset" / "games.parquet",
                     columns=["appid", "genres", "recommendations_total"])
META = {r.appid: r for r in _g.itertuples()}

# 입력 드롭다운: 후보 게임을 인기순 정렬, 동명 게임은 appid로 구분 (UI 경량화 위해 상위 6000)
_cand = sorted(rec.cand, key=lambda a: -(getattr(META.get(a), "recommendations_total", 0) or 0))
_namecount = Counter(rec.name.get(a, "?") for a in _cand)


def _label(a):
    nm = rec.name.get(a, str(a))
    return f"{nm}  ·  #{a}" if _namecount[nm] > 1 else nm


CHOICES = [(_label(a), a) for a in _cand[:6000]]


def _genres(a):
    try:
        return ", ".join(json.loads(META[a].genres)[:3]) if a in META else ""
    except Exception:
        return ""


CARD_CSS = """
<style>
.reclist{display:flex;flex-direction:column;gap:8px;margin-top:6px}
.rc{display:flex;align-items:center;gap:12px;background:#16212e;border:1px solid #27384b;
    border-radius:10px;padding:10px 14px}
.rk{color:#9fb2c6;font-size:.8rem;width:20px;text-align:right}
.rn{color:#4f9bff;font-weight:700;text-decoration:none;flex:1}
.rn:hover{text-decoration:underline}
.rg{color:#9fb2c6;font-size:.78rem}
.rs{color:#39d98a;font-size:.78rem;font-variant-numeric:tabular-nums}
</style>
"""


def recommend(liked, intent, w_intent, topn):
    liked = liked or []
    intent = (intent or "").strip()
    if not liked and not intent:
        return "<p style='color:#9fb2c6'>게임을 선택하거나 의도를 입력하세요.</p>"
    recs = rec.recommend(liked, intent, w_taste=1 - float(w_intent),
                         w_intent=float(w_intent), topn=int(topn))
    rows = []
    for i, (appid, nm, s) in enumerate(recs, 1):
        url = f"https://store.steampowered.com/app/{appid}"
        rows.append(
            f'<div class="rc"><span class="rk">{i}</span>'
            f'<a class="rn" href="{url}" target="_blank">{nm}</a>'
            f'<span class="rg">{_genres(appid)}</span>'
            f'<span class="rs">{s:.3f}</span></div>'
        )
    return CARD_CSS + '<div class="reclist">' + "".join(rows) + "</div>"


with gr.Blocks(title="Steam 게임 추천기", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🎮 Steam 게임 추천기\n"
        "**취향**(즐긴 게임) + **의도**(원하는 특징 텍스트)를 결합한 하이브리드 추천. "
        "Steam 상점이 못 하는 *의도 반영*이 핵심."
    )
    with gr.Row():
        liked = gr.Dropdown(CHOICES, multiselect=True, label="🎮 즐겨한 게임 (검색해서 선택)",
                            filterable=True)
    intent = gr.Textbox(
        label="✍️ 원하는 특징 / 의도",
        placeholder="예: relaxing open world crafting  /  scary horror  /  competitive multiplayer",
    )
    with gr.Row():
        w_intent = gr.Slider(0, 1, value=0.4, step=0.1, label="의도 반영 비중 (0=취향만 · 1=의도만)")
        topn = gr.Slider(5, 20, value=10, step=1, label="추천 개수")
    btn = gr.Button("추천 받기", variant="primary")
    out = gr.HTML()
    btn.click(recommend, [liked, intent, w_intent, topn], out)

    gr.Markdown(
        "<small>협업 임베딩(item2vec식, 직접 학습) + 콘텐츠 임베딩(대조학습) · "
        "공식 Steam API 수집 데이터 1,021만 리뷰 · 추천 클릭 시 Steam 상점으로 이동</small>"
    )


if __name__ == "__main__":
    import os
    share = os.getenv("SHARE", "0") == "1"   # SHARE=1 이면 임시 공개 링크(gradio.live)
    demo.launch(server_port=7860, share=share, inbrowser=False)
