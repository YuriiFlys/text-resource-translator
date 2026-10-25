# text-resource-translator

Автоматизована локалізація програмного інтерфейсу за допомогою LLM.
Конвеєр RAG на **PostgreSQL + pgvector + LangChain**: витягує рядки інтерфейсу та
довідкові документи, зберігає оригінали на диску й повні тексти в базі, розбиває їх на
фрагменти з векторами, шукає найближчі фрагменти та перекладає нові рядки LLM,
спираючись на пам'ять перекладів (translation memory) і глосарій.

Детальний план і обґрунтування рішень — у [PLAN.md](PLAN.md).

## Як це працює

```
sources.yaml / i18n_*  ──fetch──▶  RawStore (диск, content-addressed) + rag.raw_artifact
                                        │ ingest: parse .json/.properties/.po/.resx/.xliff, PDF/DOCX/MD/HTML
                                        ▼
                                  rag.document (повний текст, unique doc_key)
                                        │ embed: LangChain splitters → embeddings → нормалізовані вектори
                                        ▼
                                  rag.chunk (vector(N) + HNSW cosine)  ◀── search: ORDER BY embedding <=> :q
                                        │ translate: kNN → translation memory + глосарій → LLM → валідація
                                        ▼
                                  i18n_translation (upsert) + rag.translation_audit / rag.review_queue
```

1. **fetch** — інкрементально (курсор у `rag.sync_state`) вибирає переклади з `i18n_*` у JSONL‑знімок; завантажує файли ресурсів і документи з `sources.yaml` (HTTP з ETag/Last‑Modified або локальний шлях). Кожен оригінал лягає в `RAW_DIR/<source>/<sha256[:2]>/<sha256>.<ext>` із сайдкаром метаданих і реєструється в `rag.raw_artifact`.
2. **ingest** — витягує текст (ключ → рядок для ресурсів, текст сторінок для PDF) і робить upsert у `rag.document`; документ оновлюється лише коли змінився `content_sha256`.
3. **embed** — `RecursiveCharacterTextSplitter` / `MarkdownHeaderTextSplitter` → модель вкладень (HuggingFace, OpenAI або локальна hashing‑модель) з кешем у `rag.embedding_cache` → `rag.chunk` з HNSW‑індексом. Короткі UI‑рядки збагачуються `text_code | owner_kind`.
4. **search** — kNN через pgvector (`<=>`), фільтри за локаллю і типом, повертає текст, `chunk_id`, `document_id`, score.
5. **translate** — для кожної увімкненої локалі знаходить неперекладені або застарілі рядки, збирає translation memory (найближчі рядки вихідної локалі з наявними перекладами) та фрагменти глосарію, викликає LLM через LangChain (`init_chat_model`, провайдер за назвою моделі), валідує плейсхолдери й теги, повторює спробу з поясненням помилки, пише в `i18n_translation` з аудитом; невалідні потрапляють у `rag.review_queue`. Нові переклади на наступному циклі стають частиною translation memory.
6. **schedule** — APScheduler запускає `run-all` за інтервалом або cron; паралельні запуски блокуються advisory lock'ом.

## Швидкий старт

```bash
cp .env.example .env                    # заповнити PG_DSN, ANTHROPIC_API_KEY або OPENAI_API_KEY
docker compose up -d db                 # PostgreSQL 16 з pgvector

uv venv && source .venv/bin/activate
uv pip install -e ".[dev,sources,embeddings-local,llm-anthropic]"

trt check                               # з'єднання та наявність pgvector
trt migrate                             # розширення vector, схема rag, індекси
trt seed                                # тестові дані в i18n_* (en повна, uk часткова, pl порожня)

trt fetch                               # i18n_* → JSONL, sources.yaml → файли на диск
trt ingest                              # оригінали → rag.document
trt embed                               # фрагменти + вектори → rag.chunk
trt search "Save changes" -n 5 --locale en --kind ui_string
trt translate --locale uk --dry-run     # показати переклади без запису
trt translate                           # усі увімкнені локалі, запис у i18n_translation
trt run-all                             # fetch → ingest → embed → translate → (нові переклади у TM)
trt schedule                            # періодично, кожні INTERVAL_MINUTES; --cron "*/30 * * * *"
```

Увесь конвеєр у контейнері: `docker compose up` (сервіс `app` запускає `trt schedule`).

Без завантаження моделей і ключів можна перевірити все, крім LLM‑перекладу:
`EMBEDDING_PROVIDER=hashing` використовує детерміновану локальну модель вкладень.

## Налаштування (`.env`)

| Змінна | Призначення |
|--------|-------------|
| `PG_DSN` | SQLAlchemy DSN однієї бази для `i18n_*` і схеми `rag` |
| `SOURCE_LOCALE` | вихідна локаль для перекладу (типово `en`) |
| `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL` / `EMBEDDING_DIM` | `huggingface` (`intfloat/multilingual-e5-base`, 768), `openai` (`text-embedding-3-small`, 1536) або `hashing`; розмірність фіксується типом колонки при `trt migrate` |
| `LLM_MODEL` | модель чату для LangChain `init_chat_model`, напр. `claude-sonnet-5-5` |
| `RAW_DIR` | корінь RawStore на диску |
| `SOURCES_FILE` | шлях до `sources.yaml` |
| `INTERVAL_MINUTES` | період планувальника |

Джерела файлів описуються в [`sources.yaml`](sources.yaml); приклади у [`samples/`](samples/).

## Структура

```
alembic/versions/            0001 i18n_*, 0002 rag (document, chunk+HNSW, cache, state, audit, review), 0003 raw_artifact
sql/seed_i18n.sql            тестові локалі, рядки, часткові переклади
src/trt/
  cli.py                     команди trt
  pipeline.py                оркестрація кроків, run_all з advisory lock
  scheduler.py               APScheduler
  sources/                   postgres_i18n (S1), resource_files (S2/S3), base (реєстр артефактів, курсори)
  storage/raw_store.py       content-addressed сховище оригіналів
  extract/                   парсери ресурсів (.json .properties .po .resx .xliff), текст з PDF/DOCX/MD/HTML
  ingest/documents.py        upsert у rag.document
  chunking.py                LangChain splitters
  embeddings.py              фабрика моделей + кеш; HashingEmbeddings
  vector_store.py            rag.chunk, kNN-пошук
  translate/                 prompt, translator (LLM + TM), validation, writeback
tests/                       unit + інтеграційні (TEST_PG_DSN)
```

## Тести

```bash
pytest                                                                 # unit
TEST_PG_DSN=postgresql+psycopg://trt:trt@localhost:5432/trt pytest     # + інтеграційні: міграції, kNN,
                                                                       #   повний конвеєр із фейковою LLM
```
