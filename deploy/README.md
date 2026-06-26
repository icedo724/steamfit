---
title: SteamFit
emoji: 🎮
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# SteamFit — Steam 게임 추천 (취향 + 의도 하이브리드)

즐겨한 게임(취향) + 원하는 특징 텍스트(의도)를 결합해 추천. Steam 상점이 못 하는 **의도 반영**이 핵심.

- 협업 임베딩(item2vec식, 직접 대조학습) + 콘텐츠 임베딩
- 공식 Steam API 수집 데이터(리뷰 1,021만) 기반
- 하이브리드가 co-occurrence 베이스라인 대비 Recall@10 +13%
