"""A — 리뷰 enrich 다국어 콘텐츠 임베딩 재학습. [GPU, ST5 Trainer + bf16]

train_multilingual.py를 확장: 게임 doc에 다국어 리뷰 스니펫(dataset/review_docs.parquet)을
결합 → 실제 한국어/중국어 사용자 어휘로 의도매칭·한국어 외래어(소울라이크 등) 보강.
기존 콘텐츠 티어(0.079)와 동일 leave-one-out으로 비교. 검증 전까진 별도 저장(프로덕션 미덮어쓰기).

속도: ST5 SentenceTransformerTrainer + bf16 + dataloader 멀티워커(fp32 단일스레드 대비 대폭↑).
환경변수: REV_MAXSTEPS(프로브-스루풋만 측정), REV_SMALL, REV_EPOCHS, REV_BATCH, REV_MAXSEQ, REV_PAIRS.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                   SentenceTransformerTrainingArguments, losses)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.train_multilingual import jl, build_eval, encode, evaluate, ko_sanity  # noqa: E402
from pipeline.db import connect  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

PAIRS = (ROOT / "dataset" / "pairs.parquet").as_posix()
REVIEW_DOCS = ROOT / "dataset" / "review_docs.parquet"
OUT = ROOT / "models"
CKPT = ROOT / "models" / "_rv_ckpt"
BASE = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SMALL = bool(os.getenv("REV_SMALL"))
MAXSTEPS = int(os.getenv("REV_MAXSTEPS", "0"))   # >0 이면 프로브(스루풋만)
EPOCHS = int(os.getenv("REV_EPOCHS", "1"))
BATCH = int(os.getenv("REV_BATCH", "64"))
MAXSEQ = int(os.getenv("REV_MAXSEQ", "192"))
N_PAIRS = int(os.getenv("REV_PAIRS", "80000" if SMALL else "1500000"))


def load_docs_with_reviews():
    """MotherDuck 메타(이름+태그+장르+EN/KO설명) + 로컬 리뷰 스니펫 결합."""
    con, where = connect()
    print("base docs from:", where)
    df = con.execute("""
        SELECT appid, name, name_ko, tags, genres_en, short_desc_en, short_desc_ko
        FROM games WHERE type='game' AND COALESCE(recommendations_total,0) > 0
    """).df()
    con.close()
    rev = {}
    if REVIEW_DOCS.exists():
        rd = pd.read_parquet(REVIEW_DOCS)
        rev = dict(zip(rd["appid"], rd["review_text"]))
    doc, n_rev = {}, 0
    for r in df.itertuples():
        parts = [str(r.name or "")]
        parts += jl(r.tags)
        parts += jl(r.genres_en)
        if r.short_desc_en:
            parts.append(str(r.short_desc_en))
        if r.short_desc_ko:
            parts.append(str(r.short_desc_ko))
        base = " | ".join(p for p in parts if p).strip()
        rv = rev.get(r.appid)
        if rv:
            base = f"{base}  ‖ 사용자 리뷰: {rv}"
            n_rev += 1
        doc[r.appid] = base
    print(f"docs: {len(doc):,}  (리뷰 포함 {n_rev:,})")
    return doc


def make_dataset(doc):
    """parquet로 써서 memory-map 로드 — 1.5M쌍 from_dict 핑거프린트 OOM 회피."""
    pairs = pd.read_parquet(PAIRS)
    if len(pairs) > N_PAIRS:
        pairs = pairs.sample(N_PAIRS, random_state=0)
    doc_s = pd.Series(doc)                                  # appid → doc (벡터 매핑용)
    keep = pairs["appid_a"].isin(doc_s.index) & pairs["appid_b"].isin(doc_s.index)
    pairs = pairs[keep]
    df = pd.DataFrame({"anchor": pairs["appid_a"].map(doc_s).values,
                       "positive": pairs["appid_b"].map(doc_s).values})
    print(f"학습쌍: {len(df):,}")
    CKPT.mkdir(parents=True, exist_ok=True)
    tmp = CKPT / "_pairs.parquet"
    df.to_parquet(tmp, index=False)
    del df, pairs, doc_s
    return Dataset.from_parquet(str(tmp))


def main():
    probe = MAXSTEPS > 0
    mode = "프로브(스루풋)" if probe else ("캘리브(소규모)" if SMALL else "전체")
    print(f"=== A 리뷰재학습 [{mode}] bf16 ep={EPOCHS} batch={BATCH} maxseq={MAXSEQ} "
          f"pairs≤{N_PAIRS:,} maxsteps={MAXSTEPS} ===")
    doc = load_docs_with_reviews()
    names = {a: d.split(" | ")[0] for a, d in doc.items()}

    model = SentenceTransformer(BASE, device=DEVICE)
    model.max_seq_length = MAXSEQ
    loss = losses.MultipleNegativesRankingLoss(model)
    train_ds = make_dataset(doc)

    if not probe:
        top, holdout, prof, eu = build_eval()
        if SMALL:
            eu = eu[:2000]; holdout = {u: holdout[u] for u in eu}; prof = {u: prof[u] for u in eu}
        emb0, a2r = encode(model, top, doc)
        print("리뷰enrich 사전학습:", evaluate(emb0, a2r, holdout, prof, eu))

    args = SentenceTransformerTrainingArguments(
        output_dir=str(CKPT), num_train_epochs=EPOCHS, max_steps=(MAXSTEPS if probe else -1),
        per_device_train_batch_size=BATCH, bf16=True, warmup_ratio=0.1,
        dataloader_num_workers=4, logging_steps=200, report_to=[],
        save_strategy=("no" if probe else "epoch"), save_total_limit=1,  # 에폭별 체크포인트(중단 대비)
        disable_tqdm=bool(os.getenv("REV_NOTQDM")),
    )
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=train_ds, loss=loss)
    t0 = time.time()
    trainer.train()
    dt = time.time() - t0
    ran = MAXSTEPS if probe else (len(train_ds) // BATCH) * EPOCHS
    ips = ran / max(dt, 1)
    print(f"학습 {dt:.0f}s  ({ran} steps, {ips:.2f} it/s, {ips*BATCH:.0f} samples/s)")
    if probe:
        full_steps = (1_500_000 // BATCH)
        print(f"[ETA] 전체 1.5M쌍 1에폭 ≈ {full_steps/ips/3600:.2f}h "
              f"(2에폭 ≈ {2*full_steps/ips/3600:.2f}h)")
        return

    emb1, a2r = encode(model, top, doc)
    print("리뷰enrich 대조학습:", evaluate(emb1, a2r, holdout, prof, eu))
    print("한국어 의도 sanity:")
    ko_sanity(model, emb1, a2r, doc, names,
              ["협동 공포 게임", "여유로운 농장 경영", "어려운 소울라이크", "도트 로그라이크"])
    print("영어 의도 sanity:")
    ko_sanity(model, emb1, a2r, doc, names, ["co-op horror", "relaxing farming sim", "hard soulslike"])

    if SMALL:
        print("캘리브 종료 — 저장 생략.")
        return
    OUT.mkdir(exist_ok=True)
    model.save(str(OUT / "reviews-content-model"))
    np.save(OUT / "game_emb_rv.npy", emb1)
    pd.Series(list(a2r.keys()), name="appid").to_csv(OUT / "game_emb_rv_appids.csv", index=False)
    print("저장: models/reviews-content-model, game_emb_rv.npy (검증 후 프로덕션 반영)")


if __name__ == "__main__":
    main()
