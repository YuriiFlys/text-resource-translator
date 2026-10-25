# План реалізації: автоматизована локалізація програмного інтерфейсу за допомогою LLM

Варіант на **одній СУБД — PostgreSQL + розширення `pgvector`**. Продуктові таблиці `i18n_*` і RAG‑шар (повні документи, фрагменти, вектори) живуть в одній базі, у різних схемах.

## 1. Мета та межі проєкту

Побудувати Python‑сервіс, який:

1. за вимогою (CLI/HTTP) або періодично (планувальник) витягує тексти інтерфейсу та допоміжні документи з джерел;
2. зберігає оригінальні артефакти (файли ресурсів, PDF, експорти) на диску без змін;
3. читає вміст і витягує з нього саме текст (ключ → рядок, текст PDF/DOCX/MD);
4. зберігає повний текстовий вміст у PostgreSQL з індексами для швидкого доступу щонайменше за ключем;
5. засобами LangChain розбиває документи на фрагменти (chunks);
6. для кожного фрагмента обчислює вектор‑вкладення (embedding) обраною моделлю;
7. зберігає фрагменти з векторами в окрему таблицю з індексом `pgvector` (HNSW) для пошуку за схожістю;
8. надає функцію kNN‑пошуку: за вектором повертає N найближчих фрагментів — їх текст, `chunk_id` та `document_id`;
9. **(прикладна частина теми)** використовує знайдені фрагменти як «пам'ять перекладів» (translation memory) і глосарій у промпті LLM, щоб перекладати нові/застарілі рядки інтерфейсу консистентно, і записує результат у `i18n_translation`.

Пункти 1–8 — інфраструктура RAG. Пункт 9 — те, що робить її «локалізацією за допомогою LLM».

## 2. Джерела даних

| № | Джерело | Що беремо | Як | Режим |
|---|---------|-----------|-----|-------|
| S1 | Таблиці `i18n_locale`, `i18n_text`, `i18n_translation` | усі тексти та наявні переклади | SQLAlchemy + psycopg, інкрементально за `date_updated` | періодично (кожні N хв) + за вимогою |
| S2 | Файли ресурсів UI (`.properties`, `.json` i18next, `.po/.pot`, `.resx`, `.strings`, `.xliff`) | ключ → рядок, коментарі розробника | завантаження по HTTP/git/локальна тека; парсери на кожен формат | за вимогою + за розкладом |
| S3 | Довідкові документи (глосарій, style guide, довідка продукту) у PDF/DOCX/MD/HTML | повний текст | `pypdf`/`langchain_community` loaders (`PyPDFLoader`, `Docx2txtLoader`, `UnstructuredMarkdownLoader`) | за вимогою |

S1 — основне джерело. S2/S3 потрібні, щоб продемонструвати «закачані файли», витягування тексту з PDF та змістовне чанкування довгих текстів (рядки UI самі по собі короткі).

## 3. Архітектура та структура проєкту

```
text-resource-translator/
├── pyproject.toml              # залежності, entrypoint `trt`
├── .env.example                # PG_DSN, EMBEDDING_MODEL, EMBEDDING_DIM, LLM_MODEL, RAW_DIR ...
├── docker-compose.yml          # postgres (pgvector/pgvector:pg16), app
├── alembic/                    # міграції
│   └── versions/
│       ├── 001_i18n_schema.py      # таблиці з завдання
│       └── 002_rag_schema.py       # схема rag: documents, chunks, sync_state, ...
├── src/trt/
│   ├── config.py               # pydantic-settings
│   ├── cli.py                  # Typer: fetch / ingest / embed / search / translate / schedule / run-all
│   ├── scheduler.py            # APScheduler (interval/cron jobs)
│   ├── db.py                   # engine, session, ensure_extension("vector")
│   ├── models.py               # SQLAlchemy ORM (i18n_* + rag.*), тип Vector з pgvector.sqlalchemy
│   ├── sources/                # S1–S3: витягування даних
│   │   ├── base.py             # Source(Protocol): fetch() -> Iterable[RawItem]
│   │   ├── postgres_i18n.py
│   │   ├── resource_files.py   # http/git/fs завантаження
│   │   └── documents.py        # PDF/DOCX/MD
│   ├── storage/
│   │   └── raw_store.py        # збереження оригіналів на диску (content-addressed)
│   ├── extract/                # «сирий файл → текст»
│   │   ├── resource_parsers.py # .properties/.json/.po/.resx/.xliff
│   │   └── document_text.py    # PDF/DOCX/MD → text
│   ├── ingest/
│   │   └── documents.py        # upsert повних документів у rag.document
│   ├── chunking.py             # LangChain text splitters
│   ├── embeddings.py           # фабрика моделей вкладень (LangChain Embeddings)
│   ├── vector_store.py         # запис чанків + kNN через pgvector
│   ├── translate/
│   │   ├── prompt.py           # промпт з TM + глосарієм
│   │   ├── translator.py       # LLM через LangChain, structured output
│   │   └── writeback.py        # upsert у i18n_translation
│   └── schemas.py              # pydantic‑схеми для CLI/HTTP
└── tests/
```

Потік даних (усе в одній базі):

```
Sources ──fetch──▶ RawStore (диск) ──extract──▶ rag.document
                                                   │ chunk (LangChain)
                                                   ▼
                                     embed ──▶ rag.chunk (vector + HNSW index)
                                                   │ kNN: ORDER BY embedding <=> :q
                                                   ▼
                         Translator (LLM + TM/glossary) ──▶ public.i18n_translation
```

## 4. Етап 1 — витягування даних (fetch)

### 4.1 Таблиці i18n (S1)

- Запит з інкрементом за `max(date_updated)` (курсор у `rag.sync_state`):

  ```sql
  SELECT t.text_id, t.text_code, t.owner_kind,
         tr.translation_id, tr.locale_code, tr.translated_text, tr.date_updated
  FROM i18n_text t
  LEFT JOIN i18n_translation tr ON tr.text_id = t.text_id
  WHERE tr.date_updated > :since OR tr.date_updated IS NULL
  ORDER BY tr.date_updated NULLS FIRST;
  ```

- Вибірка також дампиться на диск як JSONL‑знімок (`raw/pg/<ts>.jsonl`) — «оригінальні дані» для цього джерела, корисні для відтворюваності.
- Список локалей береться з `i18n_locale WHERE enabled`. Вихідна локаль (`SOURCE_LOCALE`, напр. `en`) — у конфігурації.

### 4.2 Файли ресурсів та документи (S2, S3)

- Конфігурація джерел у `sources.yaml`: URL/шлях, формат, локаль, `owner_kind`.
- Завантаження через `httpx` з ETag/Last‑Modified, щоб не тягнути незмінені файли.
- Кожен файл зберігається в `RawStore`.

### 4.3 RawStore (збереження оригіналів)

- Шлях: `RAW_DIR/<source>/<sha256[:2]>/<sha256>.<ext>` + сайдкар `.meta.json` (URL, час, розмір, mime, оригінальна назва).
- Content‑addressed: однаковий файл не дублюється; у `rag.document` записується `raw_path` і `sha256`.

## 5. Етап 2 — витягування тексту (extract)

| Формат | Інструмент | Результат |
|--------|-----------|-----------|
| `.properties` | `jproperties` або власний парсер | `{key: value}` |
| `.json` (i18next, плоский і вкладений) | `json` + flatten `a.b.c` | `{key: value}` |
| `.po/.pot` | `polib` | `msgid → msgstr`, коментарі |
| `.resx`, `.xliff` | `lxml` | `{key: value}` |
| PDF | `pypdf` (`PyPDFLoader`), для сканів — опційно OCR | текст по сторінках |
| DOCX / MD / HTML | `docx2txt`, `markdown`, `BeautifulSoup` | plain text |

Текст нормалізується (NFC, trim), плейсхолдери (`{0}`, `%s`, `{{name}}`, `<b>`) **не чіпаються** — вони потрібні LLM.

## 6. Етап 3 — повні документи в PostgreSQL (схема `rag`)

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS rag;

CREATE TABLE rag.document (
    document_id        bigserial PRIMARY KEY,
    doc_key            varchar(512) NOT NULL UNIQUE,      -- "pg:text:12345:en", "file:<sha>:settings.save", "doc:<sha>"
    source             varchar(32)  NOT NULL,             -- postgres | resource_file | document
    kind               varchar(32)  NOT NULL,             -- ui_string | reference_doc
    text_id            bigint REFERENCES i18n_text ON DELETE SET NULL,
    text_code          varchar(256),
    owner_kind         varchar(20),
    locale_code        varchar(16) REFERENCES i18n_locale,
    content            text NOT NULL,                     -- ПОВНИЙ текст
    pages              jsonb,                             -- для PDF: [{page, text}]
    content_sha256     char(64) NOT NULL,
    raw_path           text,
    source_updated_at  timestamptz,
    ingested_at        timestamptz NOT NULL DEFAULT now(),
    chunked_sha256     char(64)                           -- sha, для якого вже побудовані чанки
);

CREATE INDEX ix_document_text_locale ON rag.document (text_id, locale_code);
CREATE INDEX ix_document_text_code   ON rag.document (text_code);
CREATE INDEX ix_document_sha         ON rag.document (content_sha256);
CREATE INDEX ix_document_source_upd  ON rag.document (source, source_updated_at DESC);
-- опційно повнотекстовий пошук як доповнення до векторного
CREATE INDEX ix_document_fts ON rag.document USING gin (to_tsvector('simple', content));
```

Upsert: `INSERT ... ON CONFLICT (doc_key) DO UPDATE SET content=EXCLUDED.content, content_sha256=..., ingested_at=now() WHERE rag.document.content_sha256 <> EXCLUDED.content_sha256`. Якщо `content_sha256` не змінився — чанки не перераховуються (`chunked_sha256 = content_sha256`).

Усе робиться в одній транзакції з читанням `i18n_*` — не потрібно узгоджувати два сховища.

## 7. Етап 4 — чанкування (LangChain)

- `ui_string` (короткі рядки): `RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=0)` — на практиці 1 рядок = 1 чанк, але довгі тексти (підказки, юридичні блоки) розіб'ються коректно.
- `reference_doc`: `RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120, separators=["\n\n","\n",". "," "])`; для Markdown — `MarkdownHeaderTextSplitter` → потім рекурсивний.
- Вхід — `langchain_core.documents.Document(page_content, metadata={document_id, text_id, locale_code, ...})`, вихід — список `Document` з `metadata.chunk_index`.
- Для UI‑рядків до тексту чанка додається контекст для кращих вкладень: `"{text_code} | {owner_kind}: {content}"` (оригінальний текст зберігається окремо в `chunk.text`, збагачений — в `chunk.embed_text`).

## 8. Етап 5 — вкладення (embeddings)

Фабрика `embeddings.py` повертає `langchain_core.embeddings.Embeddings` залежно від `EMBEDDING_PROVIDER`:

| Провайдер | Модель за замовчуванням | Розмірність | Коментар |
|-----------|------------------------|-------------|----------|
| `huggingface` (локально, безкоштовно) | `intfloat/multilingual-e5-base` | 768 | багатомовна, добре для uk/en/pl… |
| `openai` | `text-embedding-3-small` | 1536 | хмарна |

Розмірність фіксується у конфігурації (`EMBEDDING_DIM`) і підставляється в тип колонки `vector(N)` під час міграції. Батчування по 64 чанки, ретраї, кеш `rag.embedding_cache (sha256, model, embedding)`, щоб не платити двічі.

## 9. Етап 6 — таблиця фрагментів і векторний індекс (pgvector)

```sql
CREATE TABLE rag.chunk (
    chunk_id         bigserial PRIMARY KEY,
    document_id      bigint NOT NULL REFERENCES rag.document ON DELETE CASCADE,
    chunk_index      int    NOT NULL,
    text             text   NOT NULL,          -- оригінальний фрагмент
    embed_text       text   NOT NULL,          -- те, що подавалось у модель
    embedding        vector(768) NOT NULL,     -- розмірність = EMBEDDING_DIM
    embedding_model  varchar(128) NOT NULL,
    -- денормалізовані фільтри (щоб kNN не робив JOIN)
    kind             varchar(32)  NOT NULL,
    locale_code      varchar(16),
    text_id          bigint,
    owner_kind       varchar(20),
    created_at       timestamptz NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX ix_chunk_text_locale ON rag.chunk (text_id, locale_code);
CREATE INDEX ix_chunk_kind_locale ON rag.chunk (kind, locale_code);

-- векторний індекс: HNSW, косинусна відстань
CREATE INDEX ix_chunk_embedding_hnsw ON rag.chunk
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
```

Примітки:

- HNSW обрано замість IVFFlat, бо не потребує попереднього навчання на даних і стабільно працює при інкрементальному додаванні рядків.
- Вектори нормалізуються перед записом; тоді `<=>` (cosine) і `<#>` (inner product) еквівалентні, можна використати `vector_ip_ops` як трохи швидший варіант.
- При зміні моделі/розмірності — міграція `ALTER COLUMN embedding TYPE vector(N)` + `trt embed --reindex`.
- Для >1–2 млн чанків — партиціювання `rag.chunk` за `kind` або `locale_code` з окремим HNSW на кожну партицію.

## 10. Етап 7 — пошук N найближчих

```python
def search_similar(session, query_vector: list[float], n: int = 5,
                   locale: str | None = None, kind: str | None = None) -> list[Hit]:
    sql = text("""
        SELECT c.chunk_id, c.document_id, c.chunk_index, c.text,
               c.text_id, c.locale_code, d.doc_key,
               1 - (c.embedding <=> :q) AS score
        FROM rag.chunk c
        JOIN rag.document d USING (document_id)
        WHERE (:locale IS NULL OR c.locale_code = :locale)
          AND (:kind   IS NULL OR c.kind = :kind)
        ORDER BY c.embedding <=> :q
        LIMIT :n
    """).bindparams(bindparam("q", type_=Vector(settings.embedding_dim)))
    session.execute(text("SET LOCAL hnsw.ef_search = :ef"), {"ef": max(40, n * 10)})
    rows = session.execute(sql, {"q": query_vector, "n": n, "locale": locale, "kind": kind})
    return [Hit(**r._mapping) for r in rows]
```

- `hnsw.ef_search` піднімається пропорційно `n`, щоб фільтри `WHERE` не «з'їдали» результати.
- Обгортка `search_text(query: str, ...)` спочатку обчислює вектор запиту тією ж моделлю.
- Опційно гібридний пошук: об'єднання kNN з `ts_rank` по `ix_document_fts` через Reciprocal Rank Fusion — усе в одному SQL.
- CLI: `trt search "Save changes" -n 5 --locale uk`.

Інтеграція з LangChain за бажанням: `langchain_postgres.PGVector` як `VectorStore`/`Retriever` поверх тієї самої таблиці — зручно для `RunnableChain`, але власна таблиця дає контроль над схемою й фільтрами, тому в плані — власний SQL.

## 11. Етап 8 — переклад за допомогою LLM (RAG)

1. **Відбір роботи** одним запитом:

   ```sql
   SELECT t.text_id, t.text_code, t.owner_kind, src.translated_text AS source_text
   FROM i18n_text t
   JOIN i18n_translation src ON src.text_id = t.text_id AND src.locale_code = :source_locale
   LEFT JOIN i18n_translation tgt ON tgt.text_id = t.text_id AND tgt.locale_code = :target_locale
   WHERE tgt.translation_id IS NULL OR tgt.date_updated < src.date_updated
   LIMIT :batch;
   ```

2. **Контекст**: для вихідного рядка — `k` найближчих чанків серед `kind='ui_string' AND locale_code=:source_locale`, до них JOIN наявних перекладів у цільову локаль (пари «джерело → переклад» = translation memory) + `k` фрагментів із `kind='reference_doc'` (глосарій/стайлгайд). Один SQL із CTE на kNN і JOIN на `i18n_translation`.
3. **Промпт**: system — правила (зберігати плейсхолдери, HTML‑теги, довжину ±30 %, тон, термінологію з глосарію); user — JSON зі списком рядків (`text_code`, `owner_kind`, `source`), прикладами TM, глосарієм. Відповідь — structured output (pydantic) `[{text_code, translation}]`. Пакетами по 20–30 рядків.
4. **LLM**: через `langchain.chat_models.init_chat_model(settings.llm_model)` — провайдер змінний (за замовчуванням `claude-sonnet-5-5`, альтернатива — будь‑яка модель із підтримкою structured output).
5. **Валідація**: плейсхолдери з джерела присутні у перекладі в тій же кількості; не порожньо; не ідентично джерелу (крім винятків). Невалідні — повтор із поясненням помилки, потім у `rag.review_queue`.
6. **Запис**: `INSERT INTO i18n_translation ... ON CONFLICT (text_id, locale_code) DO UPDATE SET translated_text=..., date_updated=now()` + рядок у `rag.translation_audit (translation_id, model, prompt_sha256, tm_chunk_ids bigint[], created_at)` для простежуваності. Усе в одній транзакції.
7. Нові переклади самі стають частиною TM після наступного `ingest` → система «вчиться» на підтверджених перекладах.

## 12. Оркестрація: за вимогою та періодично

CLI (Typer) — кожен крок окремо та разом:

```
trt migrate                     # alembic upgrade head (розширення vector, схема rag, індекси)
trt fetch   [--source pg|files|docs] [--since ...]
trt ingest                      # extract + upsert у rag.document
trt embed   [--all|--new|--reindex]
trt search  "<text>" -n 5 --locale uk
trt translate --locale uk [--dry-run] [--limit 100]
trt run-all                     # fetch→ingest→embed→translate
trt schedule                    # APScheduler: run-all кожні N хв (INTERVAL_MINUTES) або cron
```

- Планувальник: `APScheduler` (BackgroundScheduler, `coalesce=True`, `max_instances=1`), стан останнього запуску в `rag.sync_state`; захист від паралельного запуску через `pg_try_advisory_lock`. У контейнері як альтернатива — системний cron, що викликає `trt run-all`.
- Опційно тонкий HTTP‑шар (FastAPI): `POST /jobs/run-all`, `GET /search?q=&n=`, `POST /translate` — «за вимогою» для зовнішніх систем.

## 13. Інфраструктура та залежності

- Python 3.12, `pyproject.toml` (uv/poetry).
- Бібліотеки: `langchain`, `langchain-core`, `langchain-community`, `langchain-text-splitters`, `langchain-huggingface` / `langchain-openai`, `langchain-anthropic`, `pgvector` (python‑адаптер для SQLAlchemy/psycopg), `psycopg[binary]`, `sqlalchemy`, `alembic`, `pydantic-settings`, `typer`, `apscheduler`, `httpx`, `pypdf`, `polib`, `lxml`, `python-docx`/`docx2txt`, `structlog`.
- `docker-compose.yml`: `pgvector/pgvector:pg16` (готовий образ PostgreSQL з розширенням), сервіс `app`.
- Секрети/налаштування тільки через `.env` (`.env.example` у репозиторії).

## 14. Тестування

- Unit: парсери ресурсів, нормалізація, чанкер (детерміновані межі), валідація плейсхолдерів, збірка промпту (snapshot).
- Integration (pytest + testcontainers з образом `pgvector/pgvector`): міграції, інкрементальний fetch, ідемпотентність upsert, kNN повертає посіяні чанки у правильному порядку з фільтрами.
- LLM та embeddings у тестах підміняються фейковими (`FakeEmbeddings`, `FakeListChatModel` з LangChain).
- Смоук‑сценарій: `docker compose up` → `trt migrate` → `trt run-all` → `trt search` → перевірка нових рядків у `i18n_translation`.

## 15. Етапи робіт (milestones)

| # | Етап | Результат |
|---|------|-----------|
| M0 | Скелет проєкту, конфіг, docker‑compose з pgvector, міграції `i18n_*` + seed | `trt --help`, база піднімається |
| M1 | Fetch S1 + RawStore + інкрементальний курсор | JSONL‑знімки на диску |
| M2 | Схема `rag`, таблиця `document` + індекси, ingest з `i18n_*` | документи з `doc_key` |
| M3 | Fetch/extract файлів ресурсів та PDF (S2/S3) | текст PDF у `rag.document` |
| M4 | Чанкування LangChain + embeddings + `rag.chunk` + HNSW‑індекс | `trt embed` |
| M5 | kNN‑пошук + CLI/HTTP | `trt search` повертає текст, `document_id`, `chunk_id`, score |
| M6 | LLM‑переклад з TM/глосарієм, валідація, writeback у `i18n_translation` | `trt translate` |
| M7 | Планувальник, логування, тести, README з інструкцією | `trt schedule`, зелений CI |

## 16. Ризики та рішення

- **Зміна моделі вкладень / розмірності** → поле `embedding_model` у чанках, міграція типу колонки, `trt embed --reindex`.
- **Фільтри у `WHERE` зменшують recall HNSW** → підняти `hnsw.ef_search`, або партиціювати за `kind`/`locale_code`; у pgvector ≥ 0.8 працює ітеративне сканування (`hnsw.iterative_scan = relaxed_order`).
- **Пошкодження плейсхолдерів LLM** → жорстка валідація + повторна спроба + ручна черга.
- **Вартість LLM** → батчування, кеш за хешем (джерело, локаль, контекст), `--dry-run`.
- **Короткі рядки дають шумні вкладення** → збагачення `embed_text` через `text_code`/`owner_kind`, фільтрація за `kind`/`locale`, гібрид з FTS.
- **Навантаження на продуктову базу** → схема `rag` може жити в окремій базі/інстансі з тим самим кодом (`RAG_PG_DSN`), а `i18n_*` читатись через FDW або окреме з'єднання.
