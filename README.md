# Steam Game Recommender

> 즐겨한 게임 + 원하는 특징(텍스트) → 우선도순 게임 추천.
> Steam 상점 추천이 "취향"만 보는 한계를, **취향 + 의도(텍스트)** 하이브리드 임베딩으로 해결한다.
>
> 핵심 역량: **임베딩을 직접 학습(대조학습)** 하고 BM25 → 사전학습 임베딩 → 파인튜닝 →
> 리랭킹으로 비교한 뒤 라이브 데모로 배포.

---

## 데이터 (공식 Steam API, 직접 수집)

| 소스 | 키 | 내용 |
|------|----|------|
| `IStoreService/GetAppList` | 필요 | 전체 게임 목록 (~17만) |
| `store/appdetails` | 불필요 | 메타: 이름·설명·장르·카테고리·가격·**recommendations(인기도)** |
| `store/appreviews` | 불필요 | 리뷰 본문 + **작성자 steamid·playtime·추천여부** (커서 페이징) |

> 배포물은 **모델 가중치/인덱스**라 원시 데이터 재배포 이슈 없음. 데이터는 로컬 개인 학습용.

### 저장 구조 (대용량·재개 대비)

```
data/
├── progress.db            # SQLite: 가벼운 것 — applist / 게임메타 / 진행 체크포인트
│   ├── applist            # 전체 게임 목록
│   ├── games              # 메타데이터 (인기도 포함)
│   ├── details_status     # appdetails 수집 체크포인트
│   └── reviews_status     # appid별 cursor/collected/done (재개용)
└── reviews/part-*.parquet # 무거운 것 — 리뷰 본문 + interaction, 5만건 단위 샤드
```

- **interaction(steamid×appid×voted_up×playtime)** = 대조학습 양성쌍("같은 유저가 즐긴 게임")의 핵심 신호.
- 분석은 DuckDB로 parquet 직접 쿼리.

---

## 수집 실행 (전부 재개 가능 — 끊겨도 같은 명령 재실행)

```bash
pip install -r requirements.txt
copy .env.example .env        # STEAM_API_KEY 입력

python run.py applist                       # 1) 전체 목록 (~분)
python run.py details                       # 2) 메타+인기도+게임필터 (~3일, 재개가능)
python run.py reviews --max-per-game 1000   # 3) 인기순 리뷰 (단계적, 재개가능)
python run.py status                        # 진행 현황
```

### 단계적 전체(우선순위) 수집 전략
- 2단계에서 `recommendations_total`로 게임을 필터·정렬 → 3단계 리뷰를 **인기 게임부터**.
- `--max-per-game`을 점점 늘려가며 커버리지 확장 (예: 200 → 1000 → 무제한).
- `--min-recommendations`로 롱테일 쉐어웨어 제외.

---

## 데이터셋 가공 (수집 후)

원시 수집물(`data/`)은 읽기 전용으로 참조하고, 가공 산출물은 `dataset/`에 분리 저장.

```bash
python prep/build_dataset.py    # games.parquet(콘텐츠 doc) + interactions.parquet(중복제거)
python prep/build_pairs.py      # pairs.parquet — 대조학습 양성쌍(공동 추천 게임) 100만
```

| 산출물 | 내용 |
|--------|------|
| `dataset/games.parquet` | 16.96만 게임, 콘텐츠 텍스트 `doc`(이름·장르·카테고리·설명) |
| `dataset/interactions.parquet` | 1,021만 (steamid×appid, voted_up, playtime) 중복제거 |
| `dataset/pairs.parquet` | 양성쌍 100만 — 같은 유저가 둘 다 추천한 게임 |

## 모델 — 알고리즘 사다리 (평가: leave-one-out, Recall@k·nDCG·MRR)

```bash
python model/eval_baseline.py            # 티어0~1: 인기도 + 공동출현(협업)
# 티어2~3은 Kaggle GPU:  model/train_embeddings_kaggle.py
```

실행:
```bash
python model/eval_baseline.py        # 티어0~1: 인기도 + 공동출현
python model/train_embeddings_kaggle.py  # 티어2~3a: 콘텐츠 임베딩(사전학습→대조학습)  [GPU]
python model/train_item2vec.py       # 티어3b: 협업 임베딩 직접 학습  [GPU]
python model/eval_hybrid.py          # 티어4: 임베딩+cooc 앙상블
```

| 티어 | 방식 | Recall@10 | Recall@50 | nDCG@10 | MRR |
|------|------|-----------|-----------|---------|-----|
| 0 | Popularity | 0.0016 | 0.009 | 0.0007 | 0.0007 |
| 2 | 콘텐츠 임베딩 (사전학습) | 0.032 | 0.062 | 0.020 | 0.018 |
| 3a | 콘텐츠 임베딩 (대조학습 파인튜닝) | 0.068 | 0.161 | 0.039 | 0.034 |
| 3b | **협업 임베딩 (직접 학습, item2vec식)** | 0.164 | 0.327 | 0.100 | 0.088 |
| 1 | **Co-occurrence (협업 베이스라인)** | 0.196 | 0.368 | 0.124 | 0.110 |
| 4 | **하이브리드 (학습 임베딩 + cooc, α≈0.8)** | **0.221** | **0.411** | **0.140** | **0.123** ✅ |

> 핵심: 학습 임베딩 단독은 강한 cooc 베이스라인에 근접(0.164), **앙상블 시 베이스라인 초과**(R@10 +13%).
> 학습 임베딩은 밀집·일반화 표현이라 cooc이 못 하는 것(텍스트 의도 쿼리·콜드스타트·2D 맵·고속 ANN)을 가능케 함 — 이게 제품의 본질.

### 남은 단계
- 하이브리드 **취향+의도** 추천(즐긴 게임 임베딩 평균 + 텍스트 의도 임베딩 결합)
- (선택) LLM 리랭킹 + 추천 이유 생성
- HF Spaces 라이브 데모 + 임베딩 2D 맵

배포: HF Spaces 라이브 데모(취향+의도 하이브리드 쿼리) + 임베딩 2D 맵.

### 입력→출력 (데모)
즐겨한 게임(취향 벡터=게임 임베딩 평균) + 원하는 특징 텍스트(의도 벡터) → 가중 결합 →
벡터검색 → 이미 한 게임 제외 → 우선도순 추천 + 추천 이유.
