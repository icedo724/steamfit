"""베이스라인 추천 + leave-one-out 평가.

평가 프로토콜:
  - 긍정 인터랙션(voted_up=1)만 사용.
  - 게임 5개 이상 즐긴 유저 중 5,000명 샘플 → 각자 게임 1개를 정답으로 가림(holdout).
  - 나머지를 프로필로 추천 점수 산출 → 가린 정답이 상위 k에 오는지 측정.
  지표: Recall@k, nDCG@k, MRR (k=10, 50).

베이스라인 2종:
  1) Popularity — 전역 인기순 (개인화 없음, 하한선)
  2) Item co-occurrence — 같이 즐겨진 게임 (협업 신호). 임베딩 모델의 비교 기준.

후보 아이템은 인기 상위 12,000개로 제한(롱테일 노이즈 차단·연산 절감).
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import duckdb
import numpy as np
from scipy.sparse import csr_matrix

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
INTER = (ROOT / "dataset" / "interactions.parquet").as_posix()

TOP_ITEMS = 12_000
N_EVAL = 5_000
MIN_ITEMS = 5
KS = (10, 50)
SEED = 0


def load():
    df = duckdb.query(
        f"SELECT steamid, appid FROM read_parquet('{INTER}') WHERE voted_up=1"
    ).df()
    pop = df.groupby("appid").size().sort_values(ascending=False)
    top = pop.head(TOP_ITEMS).index
    df = df[df["appid"].isin(set(top))]

    items = list(top)
    iidx = {a: i for i, a in enumerate(items)}
    users = df["steamid"].unique()
    uidx = {u: i for i, u in enumerate(users)}
    df["u"] = df["steamid"].map(uidx).astype(np.int32)
    df["i"] = df["appid"].map(iidx).astype(np.int32)
    item_pop = np.asarray(
        df.groupby("i").size().reindex(range(len(items)), fill_value=0)
    )
    return df, len(users), len(items), item_pop


def make_eval(df, n_users):
    rng = random.Random(SEED)
    by_user = df.groupby("u")["i"].apply(list)
    pool = [u for u, its in by_user.items() if len(its) >= MIN_ITEMS]
    rng.shuffle(pool)
    eval_users = pool[:N_EVAL]
    holdout = {u: rng.choice(by_user[u]) for u in eval_users}
    profiles = {u: [i for i in by_user[u] if i != holdout[u]] for u in eval_users}
    return eval_users, holdout, profiles


def metrics_at(ranked, target, ks):
    out = {}
    pos = ranked.index(target) if target in ranked else None
    for k in ks:
        hit = pos is not None and pos < k
        out[f"recall@{k}"] = 1.0 if hit else 0.0
        out[f"ndcg@{k}"] = (1.0 / math.log2(pos + 2)) if hit else 0.0
    out["mrr"] = (1.0 / (pos + 1)) if pos is not None else 0.0
    return out


def evaluate(score_fn, eval_users, holdout, profiles, n_items, maxk):
    agg = {}
    for u in eval_users:
        scores = score_fn(profiles[u])
        scores[profiles[u]] = -np.inf  # 이미 가진 게임 제외
        topk = np.argpartition(-scores, maxk)[:maxk]
        topk = topk[np.argsort(-scores[topk])]
        m = metrics_at(list(topk), holdout[u], KS)
        for key, v in m.items():
            agg[key] = agg.get(key, 0.0) + v
    return {k: round(v / len(eval_users), 4) for k, v in agg.items()}


def main():
    print("[load] 인터랙션 로드·인코딩...")
    df, n_users, n_items, item_pop = load()
    print(f"  유저 {n_users:,} · 아이템 {n_items:,} · 인터랙션 {len(df):,}")

    eval_users, holdout, profiles = make_eval(df, n_users)
    print(f"  평가 유저 {len(eval_users):,} (게임 {MIN_ITEMS}개+, 1개 holdout)")

    # 아이템 공동출현 행렬 C = X^T X
    print("[cooc] item-item 공동출현 행렬 구축...")
    X = csr_matrix((np.ones(len(df), np.float32), (df["u"], df["i"])),
                   shape=(n_users, n_items))
    C = (X.T @ X).tocsr()
    C.setdiag(0)
    C.eliminate_zeros()

    pop_vec = item_pop.astype(np.float32)

    def score_pop(profile):
        return pop_vec.copy()

    def score_cooc(profile):
        return np.asarray(C[profile].sum(axis=0)).ravel()

    maxk = max(KS)
    print("\n=== 베이스라인 평가 ===")
    print("Popularity   :", evaluate(score_pop, eval_users, holdout, profiles, n_items, maxk))
    print("Co-occurrence:", evaluate(score_cooc, eval_users, holdout, profiles, n_items, maxk))


if __name__ == "__main__":
    main()
