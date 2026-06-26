-- Steam 추천 클라우드 파이프라인 스키마 (DuckDB / MotherDuck 공통)
-- 저장 범위: interactions + EN/KO 게임 메타 + 증분 수집 상태. (리뷰 본문 코퍼스는 미저장)

CREATE TABLE IF NOT EXISTS games (
    appid                BIGINT PRIMARY KEY,
    name                 TEXT,          -- 기본(영문) 이름
    name_ko              TEXT,          -- 한국어 이름(있으면)
    type                 TEXT,
    is_free              BOOLEAN,
    short_desc_en        TEXT,
    short_desc_ko        TEXT,
    genres_en            TEXT,          -- JSON 배열 문자열
    genres_ko            TEXT,
    categories           TEXT,          -- JSON 배열 문자열(언어 무관 코드성)
    release_date         TEXT,
    price_cents          INTEGER,
    recommendations_total INTEGER,
    updated_at           TIMESTAMP DEFAULT now()
);

-- 행동 신호(협업 임베딩의 핵심). 유저는 게임당 1리뷰 → (steamid,appid) 유일.
CREATE TABLE IF NOT EXISTS interactions (
    steamid     BIGINT,
    appid       BIGINT,
    voted_up    BOOLEAN,
    playtime    INTEGER,
    ts_created  BIGINT,                 -- 리뷰 작성 시각(unix) — 증분 워터마크 기준
    PRIMARY KEY (steamid, appid)
);

-- 게임별 증분 수집 상태(워터마크).
CREATE TABLE IF NOT EXISTS collection_state (
    appid           BIGINT PRIMARY KEY,
    details_done    BOOLEAN DEFAULT FALSE,
    last_review_ts  BIGINT  DEFAULT 0,  -- 이 시각 이후 리뷰만 새로 수집
    reviews_seeded  BOOLEAN DEFAULT FALSE,
    updated_at      TIMESTAMP DEFAULT now()
);

-- 전역 메타(applist 워터마크 등 key-value).
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
