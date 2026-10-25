# План реалізації: автоматизована локалізація програмного інтерфейсу за допомогою LLM

## 1. Мета та межі проєкту

Побудувати Python‑сервіс, який:

1. за вимогою (CLI/HTTP) або періодично (планувальник) витягує тексти інтерфейсу та допоміжні документи з джерел;
2. зберігає оригінальні артефакти (файли ресурсів, PDF, експорти) на диску без змін;
3. читає вміст і витягує з нього саме текст (ключ → рядок, текст PDF/DOCX/MD);
4. зберігає повний текстовий вміст у MongoDB з індексами для швидкого доступу щонайменше за ключем;
5. засобами LangChain розбиває документи на фрагменти (chunks);
6. для кожного фрагмента обчислює вектор‑вкладення (embedding) обраною моделлю;
7. зберігає фрагменти з векторами в окрему колекцію MongoDB з векторним індексом для пошуку за схожістю;
8. надає функцію kNN‑пошуку: за вектором повертає N найближчих фрагментів — їх текст, `chunk_id` та `document_id`;
9. **(прикладна частина теми)** використовує знайдені фрагменти як «пам'ять перекладів» (translation memory) і глосарій у промпті LLM, щоб перекладати нові/застарілі рядки інтерфейсу консистентно, і записує результат назад у `i18n_translation`.

Пункти 1–8 — обов'язкова інфраструктура RAG. Пункт 9 — те, що робить її «локалізацією за допомогою LLM».

## 2. Джерела даних

| № | Джерело | Що беремо | Як | Режим |
|---|---------|-----------|-----|-------|
| S1 | PostgreSQL, таблиці `i18n_locale`, `i18n_text`, `i18n_translation` | усі тексти та наявні переклади | SQLAlchemy + psycopg, інкрементально за `date_updated` | періодично (кожні N хв) + за вимогою |
| S2 | Файли ресурсів UI (`.properties`, `.json` i18next, `.po/.pot`, `.resx`, `.strings`, `.xliff`) | ключ → рядок, коментарі розробника | завантаження по HTTP/git/локальна тека; парсери на кожен формат | за вимогою + за розкладом |
| S3 | Довідкові документи (глосарій, style guide, довідка продукту) у PDF/DOCX/MD/HTML | повний текст | `pypdf`/`langchain_community` loaders (`PyPDFLoader`, `Docx2txtLoader`, `UnstructuredMarkdownLoader`) | за вимогою |

S1 — основне джерело. S2/S3 потрібні, щоб продемонструвати «закачані файли», витягування тексту з PDF та змістовне чанкування довгих текстів (рядки UI самі по собі короткі).

## 3. Архітектура та структура проєкту

```
text-resource-translator/
├── pyproject.toml              # залежності, entrypoint `trt`
├── .env.example                # PG_DSN, MONGO_URI, EMBEDDING_MODEL, LLM_MODEL, RAW_DIR ...
├── docker-compose.yml          # postgres, mongodb (atlas-local), app
├── sql/
│   └── 001_i18n_schema.sql     # схема з завдання + тестові дані
├── src/trt/
│   ├── config.py               # pydantic-settings
│   ├── cli.py                  # Typer: fetch / ingest / embed / search / translate / schedule / run-all
│   ├── scheduler.py            # APScheduler (interval/cron jobs)
│   ├── sources/                # S1–S3: витягування даних
│   │   ├── base.py             # Source(Protocol): fetch() -> Iterable[RawItem]
│   │   ├── postgres_i18n.py
│   │   ├── resource_files.py   # http/git/fs завантаження
│   │   └── documents.py        # PDF/DOCX/MD
│   ├── storage/
│   │   ├── raw_store.py        # збереження оригіналів на диску (content-addressed)
│   │   └── mongo.py            # клієнт, створення колекцій та індексів
│   ├── extract/                # «сирий файл → текст»
│   │   ├── resource_parsers.py # .properties/.json/.po/.resx/.xliff
│   │   └── document_text.py    # PDF/DOCX/MD → text
│   ├── ingest/
│   │   └── documents.py        # upsert повних документів у Mongo
│   ├── chunking.py             # LangChain text splitters
│   ├── embeddings.py           # фабрика моделей вкладень (LangChain Embeddings)
│   ├── vector_store.py         # запис чанків + $vectorSearch
│   ├── translate/
│   │   ├── prompt.py           # промпт з TM + глосарієм
│   │   ├── translator.py       # LLM через LangChain, structured output
│   │   └── writeback.py        # upsert у i18n_translation
│   └── models.py               # pydantic‑схеми документів Mongo
└── tests/
```

Потік даних:

```
Sources ──fetch──▶ RawStore (диск) ──extract──▶ Mongo.documents
                                                   │ chunk (LangChain)
                                                   ▼
                                     embed ──▶ Mongo.chunks (+ vector index)
                                                   │ $vectorSearch (kNN)
                                                   ▼
                         Translator (LLM + TM/glossary) ──▶ PostgreSQL.i18n_translation
```

## 4. Етап 1 — витягування даних (fetch)

### 4.1 PostgreSQL (S1)

- Запит з інкрементом за `max(date_updated)` (стан курсора зберігається в Mongo, колекція `sync_state`):

  ```sql
  SELECT t.text_id, t.text_code, t.owner_kind,
         tr.translation_id, tr.locale_code, tr.translated_text, tr.date_updated
  FROM i18n_text t
  LEFT JOIN i18n_translation tr ON tr.text_id = t.text_id
  WHERE tr.date_updated > :since OR tr.date_updated IS NULL
  ORDER BY tr.date_updated NULLS FIRST;
  ```

- Результат вибірки також дампиться на диск як JSONL‑знімок (`raw/pg/<ts>.jsonl`) — це і є «оригінальні дані» для цього джерела, корисні для відтворюваності.
- Список локалей береться з `i18n_locale WHERE enabled`. Вихідна локаль (`SOURCE_LOCALE`, напр. `en`) — у конфігурації.

### 4.2 Файли ресурсів та документи (S2, S3)

- Конфігурація джерел у `sources.yaml`: URL/шлях, формат, локаль, `owner_kind`.
- Завантаження через `httpx` з ETag/Last‑Modified, щоб не тягнути незмінені файли.
- Кожен файл зберігається в `RawStore`.

### 4.3 RawStore (збереження оригіналів)

- Шлях: `RAW_DIR/<source>/<sha256[:2]>/<sha256>.<ext>` + сайдкар `.meta.json` (URL, час, розмір, mime, оригінальна назва).
- Content‑addressed: однаковий файл не дублюється; у Mongo‑документ записується `raw_path` і `sha256`.

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

## 6. Етап 3 — повні документи в MongoDB

База `localization`, колекція **`documents`** — один документ на одиницю тексту:

```jsonc
{
  "_id": ObjectId,
  "doc_key": "pg:text:12345:en",          // унікальний ключ (джерело:тип:id:локаль)
  "source": "postgres" | "resource_file" | "document",
  "kind": "ui_string" | "reference_doc",
  "text_id": 12345,                        // з i18n_text (для ui_string)
  "text_code": "settings.profile.save",
  "owner_kind": "button",
  "locale_code": "en",
  "content": "Save changes",               // ПОВНИЙ текст
  "content_sha256": "...",
  "raw_path": "raw/pg/2026-10-25T10.jsonl",
  "source_updated_at": ISODate,
  "ingested_at": ISODate,
  "chunked": false
}
```

Для `reference_doc` поля `text_id/text_code` відсутні, а `content` — весь текст PDF (сторінки зберігаються також у масиві `pages[]`).

Індекси (створюються ідемпотентно при старті):

```python
documents.create_index("doc_key", unique=True)
documents.create_index([("text_id", 1), ("locale_code", 1)])
documents.create_index("text_code")
documents.create_index("content_sha256")
documents.create_index([("source", 1), ("source_updated_at", -1)])
```

Upsert за `doc_key`; якщо `content_sha256` не змінився — чанки не перераховуються.

## 7. Етап 4 — чанкування (LangChain)

- `ui_string` (короткі рядки): `RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=0)` — на практиці 1 рядок = 1 чанк, але довгі тексти (підказки, юридичні блоки) розіб'ються коректно.
- `reference_doc`: `RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120, separators=["\n\n","\n",". "," "])`; для Markdown — `MarkdownHeaderTextSplitter` → потім рекурсивний.
- Вхід — `langchain_core.documents.Document(page_content, metadata={document_id, text_id, locale_code, ...})`, вихід — список `Document` з `metadata.chunk_index`.
- Для UI‑рядків до тексту чанка додається контекст для кращих вкладень: `"{text_code} | {owner_kind}: {content}"` (оригінальний `content` зберігається окремо).

## 8. Етап 5 — вкладення (embeddings)

Фабрика `embeddings.py` повертає `langchain_core.embeddings.Embeddings` залежно від `EMBEDDING_PROVIDER`:

| Провайдер | Модель за замовчуванням | Розмірність | Коментар |
|-----------|------------------------|-------------|----------|
| `huggingface` (локально, безкоштовно) | `intfloat/multilingual-e5-base` | 768 | багатомовна, добре для uk/en/pl… |
| `openai` | `text-embedding-3-small` | 1536 | хмарна |

Розмірність фіксується у конфігурації і використовується при створенні векторного індексу. Батчування по 64 чанки, ретраї, кеш за `sha256(text+model)` у колекції `embedding_cache`, щоб не платити двічі.

## 9. Етап 6 — колекція фрагментів і векторний індекс

Колекція **`chunks`**:

```jsonc
{
  "_id": ObjectId,                 // chunk_id
  "document_id": ObjectId,         // → documents._id
  "doc_key": "pg:text:12345:en",
  "chunk_index": 0,
  "text": "Save changes",
  "embedding": [0.012, -0.08, ...],
  "embedding_model": "intfloat/multilingual-e5-base",
  "locale_code": "en",
  "text_id": 12345,
  "owner_kind": "button",
  "kind": "ui_string",
  "created_at": ISODate
}
```

Звичайні індекси: `{document_id:1, chunk_index:1}` (unique), `doc_key`, `{text_id:1, locale_code:1}`.

Векторний індекс (Atlas Vector Search; працює в Atlas, у Docker‑образі `mongodb/mongodb-atlas-local` та в MongoDB Community ≥ 8.2 з `mongot`):

```python
from pymongo.operations import SearchIndexModel
chunks.create_search_index(SearchIndexModel(
    name="chunks_vector_idx",
    type="vectorSearch",
    definition={
        "fields": [
            {"type": "vector", "path": "embedding",
             "numDimensions": settings.embedding_dim, "similarity": "cosine"},
            {"type": "filter", "path": "locale_code"},
            {"type": "filter", "path": "kind"},
            {"type": "filter", "path": "owner_kind"},
        ]
    },
))
```

Створення ідемпотентне (перевірка через `list_search_indexes()`); при зміні моделі/розмірності — переіндексація колекції.

## 10. Етап 7 — пошук N найближчих

```python
def search_similar(query_vector: list[float], n: int = 5,
                   locale: str | None = None, kind: str | None = None) -> list[Hit]:
    pre = {k: v for k, v in {"locale_code": locale, "kind": kind}.items() if v}
    pipeline = [
        {"$vectorSearch": {
            "index": "chunks_vector_idx", "path": "embedding",
            "queryVector": query_vector, "numCandidates": n * 20, "limit": n,
            **({"filter": pre} if pre else {}),
        }},
        {"$project": {"text": 1, "document_id": 1, "chunk_index": 1,
                      "doc_key": 1, "text_id": 1, "locale_code": 1,
                      "score": {"$meta": "vectorSearchScore"}}},
    ]
    return [Hit(chunk_id=d["_id"], document_id=d["document_id"], text=d["text"],
                score=d["score"], ...) for d in chunks.aggregate(pipeline)]
```

Обгортка `search_text(query: str, ...)` спочатку обчислює вектор запиту тією ж моделлю. CLI: `trt search "Save changes" -n 5 --locale uk`.

## 11. Етап 8 — переклад за допомогою LLM (RAG)

1. **Відбір роботи**: для кожної `enabled` локалі знайти `text_id`, у яких немає рядка в `i18n_translation` або `date_updated` перекладу старіший за дату вихідного тексту.
2. **Контекст**: для вихідного рядка знайти `k` найближчих чанків серед `kind=ui_string, locale=SOURCE_LOCALE`, підтягнути їхні наявні переклади у цільову локаль (пари «джерело → переклад» = translation memory) + `k` фрагментів із `reference_doc` (глосарій/стайлгайд).
3. **Промпт**: system — правила (зберігати плейсхолдери, HTML‑теги, довжину ±30 %, тон, термінологію з глосарію); user — JSON зі списком рядків (`text_code`, `owner_kind`, `source`), прикладами TM, глосарієм. Відповідь — structured output (pydantic) `[{text_code, translation}]`. Пакетами по 20–30 рядків.
4. **LLM**: через `langchain.chat_models.init_chat_model(settings.llm_model)` — провайдер змінний (за замовчуванням `claude-sonnet-5-5`, альтернатива — будь‑яка модель із підтримкою structured output).
5. **Валідація**: плейсхолдери з джерела присутні у перекладі в тій же кількості; не порожньо; не ідентично джерелу (крім винятків). Невалідні — повтор із поясненням помилки, потім у `review_queue`.
6. **Запис**: `INSERT ... ON CONFLICT (text_id, locale_code) DO UPDATE SET translated_text=..., date_updated=now()`. Опційно — окрема таблиця/колекція `translation_audit` (модель, промпт‑хеш, використані TM‑приклади) для простежуваності.
7. Нові переклади самі стають частиною TM після наступного `ingest` → система «вчиться» на підтверджених перекладах.

## 12. Оркестрація: за вимогою та періодично

CLI (Typer) — кожен крок окремо та разом:

```
trt init-db                     # колекції + індекси Mongo
trt fetch   [--source pg|files|docs] [--since ...]
trt ingest                      # extract + upsert у documents
trt embed   [--all|--new]       # chunk + embeddings + chunks
trt search  "<text>" -n 5 --locale uk
trt translate --locale uk [--dry-run] [--limit 100]
trt run-all                     # fetch→ingest→embed→translate
trt schedule                    # APScheduler: run-all кожні N хв (INTERVAL_MINUTES) або cron
```

- Планувальник: `APScheduler` (BackgroundScheduler, `coalesce=True`, `max_instances=1`), стан останнього запуску в `sync_state`. У контейнері як альтернатива — системний cron, що викликає `trt run-all`.
- Опційно тонкий HTTP‑шар (FastAPI): `POST /jobs/run-all`, `GET /search?q=&n=`, `POST /translate` — «за вимогою» для зовнішніх систем.

## 13. Інфраструктура та залежності

- Python 3.12, `pyproject.toml` (uv/poetry).
- Бібліотеки: `langchain`, `langchain-core`, `langchain-community`, `langchain-text-splitters`, `langchain-huggingface` / `langchain-openai`, `langchain-anthropic`, `pymongo`, `psycopg[binary]`, `sqlalchemy`, `pydantic-settings`, `typer`, `apscheduler`, `httpx`, `pypdf`, `polib`, `lxml`, `python-docx`/`docx2txt`, `structlog`.
- `docker-compose.yml`: `postgres:16` (init‑скрипт `sql/001_i18n_schema.sql`), `mongodb/mongodb-atlas-local` (підтримує `$vectorSearch` локально), сервіс `app`.
- Секрети/налаштування тільки через `.env` (`.env.example` у репозиторії).

## 14. Тестування

- Unit: парсери ресурсів, нормалізація, чанкер (детерміновані межі), валідація плейсхолдерів, збірка промпту (snapshot).
- Integration (pytest + testcontainers або compose): PG→Mongo інкремент, ідемпотентність upsert, створення індексів, `$vectorSearch` повертає посіяні чанки у правильному порядку.
- LLM та embeddings у тестах підміняються фейковими (`FakeEmbeddings`, `FakeListChatModel` з LangChain).
- Смоук‑сценарій: `docker compose up` → `trt init-db` → `trt run-all` → `trt search` → перевірка нових рядків у `i18n_translation`.

## 15. Етапи робіт (milestones)

| # | Етап | Результат |
|---|------|-----------|
| M0 | Скелет проєкту, конфіг, docker‑compose, SQL‑схема та seed | `trt --help`, бази піднімаються |
| M1 | Fetch S1 + RawStore + інкрементальний курсор | JSONL‑знімки на диску |
| M2 | Mongo `documents` + індекси, ingest з PG | документи з `doc_key` |
| M3 | Fetch/extract файлів ресурсів та PDF (S2/S3) | текст PDF у `documents` |
| M4 | Чанкування LangChain + embeddings + `chunks` + векторний індекс | `trt embed` |
| M5 | kNN‑пошук (`$vectorSearch`) + CLI/HTTP | `trt search` повертає текст, `document_id`, `chunk_id`, score |
| M6 | LLM‑переклад з TM/глосарієм, валідація, writeback у PG | `trt translate` |
| M7 | Планувальник, логування, тести, README з інструкцією | `trt schedule`, зелений CI |

## 16. Ризики та рішення

- **Векторний індекс недоступний у звичайному `mongo:*` образі** → використовувати `mongodb-atlas-local` або Atlas; у коді перевіряти наявність індексу і давати зрозумілу помилку.
- **Зміна моделі вкладень** → поле `embedding_model` у чанках, команда `trt embed --reindex`.
- **Пошкодження плейсхолдерів LLM** → жорстка валідація + повторна спроба + ручна черга.
- **Вартість LLM** → батчування, кеш за хешем (джерело, локаль, контекст), `--dry-run`.
- **Короткі рядки дають шумні вкладення** → збагачення чанка `text_code`/`owner_kind`, фільтрація за `kind`/`locale` у `$vectorSearch`.
