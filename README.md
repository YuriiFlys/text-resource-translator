# text-resource-translator

Автоматизована локалізація програмного інтерфейсу за допомогою LLM.
Конвеєр RAG на **PostgreSQL + pgvector + LangChain**: витягує рядки інтерфейсу та
довідкові документи, зберігає повні тексти й фрагменти з векторами в одній базі,
шукає найближчі фрагменти та перекладає нові рядки LLM, спираючись на пам'ять
перекладів і глосарій.

Детальний план — у [PLAN.md](PLAN.md).

## Стан реалізації

| Етап | Статус |
|------|--------|
| M0 Скелет проєкту, конфіг, docker-compose, міграції `i18n_*` + `rag`, CLI | готово |
| M1 Fetch з `i18n_*`, RawStore, інкрементальний курсор | — |
| M2 Ingest у `rag.document` | — |
| M3 Файли ресурсів та PDF | — |
| M4 Чанкування, вкладення, `rag.chunk` + HNSW | — |
| M5 kNN-пошук, CLI/HTTP | — |
| M6 LLM-переклад з translation memory | — |
| M7 Планувальник, тести, документація | — |

## Швидкий старт

```bash
cp .env.example .env
docker compose up -d db                 # PostgreSQL 16 з pgvector

uv venv && source .venv/bin/activate
uv pip install -e ".[dev,sources,embeddings-local,llm-anthropic]"

trt check                               # з'єднання та наявність pgvector
trt migrate                             # розширення vector, схема rag, індекси
trt seed                                # тестові дані в i18n_*
trt --help
```

## Структура

```
alembic/versions/0001_i18n_schema.py   # таблиці із завдання
alembic/versions/0002_rag_schema.py    # схема rag: document, chunk (vector + HNSW), службові таблиці
sql/seed_i18n.sql                      # тестові локалі, рядки, часткові переклади
src/trt/config.py                      # налаштування (pydantic-settings, .env)
src/trt/db.py                          # engine, session_scope, перевірка pgvector
src/trt/models.py                      # ORM i18n_* та rag.*
src/trt/cli.py                         # команди trt
tests/                                 # unit + інтеграційні (TEST_PG_DSN)
```

## Схема даних

- `public.i18n_locale`, `i18n_text`, `i18n_translation` — продуктові таблиці, джерело істини.
- `rag.document` — повний текст кожної одиниці (рядок UI у певній локалі або довідковий документ), унікальний `doc_key`, індекси по `text_id + locale_code`, `text_code`, `content_sha256`, GIN для FTS.
- `rag.chunk` — фрагменти з `embedding vector(EMBEDDING_DIM)` та індексом `hnsw (embedding vector_cosine_ops)`.
- `rag.embedding_cache`, `rag.sync_state`, `rag.translation_audit`, `rag.review_queue` — службові таблиці конвеєра.

Розмірність вектора задається `EMBEDDING_DIM` у `.env` **до** першого `trt migrate`.

## Тести

```bash
pytest                                  # unit-тести
TEST_PG_DSN=postgresql+psycopg://trt:trt@localhost:5432/trt pytest   # + інтеграційні
```
