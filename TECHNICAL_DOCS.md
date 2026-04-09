# NexusIQ — Complete Technical Documentation

> End-to-end code reference, design decisions, data flows, and implementation notes for every component of the NexusIQ Enterprise Knowledge Base platform.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Repository Structure](#2-repository-structure)
3. [Environment & Configuration](#3-environment--configuration)
4. [bedrock_engine.py — AWS Abstraction Layer](#4-bedrock_enginepy--aws-abstraction-layer)
5. [app.py — Flask Application](#5-apppy--flask-application)
   - 5.1 [Global State & Session Model](#51-global-state--session-model)
   - 5.2 [Session Helpers](#52-session-helpers)
   - 5.3 [PDF Store](#53-pdf-store)
   - 5.4 [Engine Factory](#54-engine-factory)
   - 5.5 [Bedrock Direct Invoke](#55-bedrock-direct-invoke)
   - 5.6 [Text Extraction](#56-text-extraction)
   - 5.7 [AI Features — Question Generation](#57-ai-features--question-generation)
   - 5.8 [AI Features — Follow-up Generation](#58-ai-features--follow-up-generation)
   - 5.9 [Highlight Phrase Extraction](#59-highlight-phrase-extraction)
   - 5.10 [Page Number Extraction](#510-page-number-extraction)
   - 5.11 [PDF Phrase Search](#511-pdf-phrase-search)
   - 5.12 [PDF Renderer](#512-pdf-renderer)
   - 5.13 [Single Page Renderer](#513-single-page-renderer)
   - 5.14 [Routes Reference](#514-routes-reference)
6. [index.html — Frontend](#6-indexhtml--frontend)
   - 6.1 [CSS Design System](#61-css-design-system)
   - 6.2 [JavaScript State Model](#62-javascript-state-model)
   - 6.3 [Boot Sequence](#63-boot-sequence)
   - 6.4 [AWS Config Modal](#64-aws-config-modal)
   - 6.5 [Question Chips System](#65-question-chips-system)
   - 6.6 [Upload Pipeline](#66-upload-pipeline)
   - 6.7 [Ingestion Sync Polling](#67-ingestion-sync-polling)
   - 6.8 [Document List](#68-document-list)
   - 6.9 [Query & Answer Pipeline](#69-query--answer-pipeline)
   - 6.10 [PDF Viewer Panel](#610-pdf-viewer-panel)
   - 6.11 [Page Navigation](#611-page-navigation)
   - 6.12 [Utility Functions](#612-utility-functions)
7. [Data Flow Diagrams](#7-data-flow-diagrams)
8. [Design Decisions & Engineering Notes](#8-design-decisions--engineering-notes)
9. [Known Limitations & Future Work](#9-known-limitations--future-work)

---

## 1. System Overview

NexusIQ is a three-tier web application:

```
Tier 1  ─  Browser (index.html)
            Vanilla JS single-page app. No framework, no build step.
            Three-column layout: sidebar / chat / PDF viewer.

Tier 2  ─  Flask Backend (app.py)
            REST API. Manages sessions, orchestrates AI features,
            renders PDFs, proxies S3/Bedrock calls.

Tier 3  ─  AWS (bedrock_engine.py)
            S3 for document storage, Bedrock Knowledge Bases for
            vector search + RAG, Bedrock Runtime for direct
            Claude invocations (questions, follow-ups).
```

The defining design choice is that **every AI answer is grounded**. Claude is never asked to answer from its training data. All responses go through Bedrock's `retrieve_and_generate` API which injects retrieved document chunks into the context before generation.

---

## 2. Repository Structure

```
nexusiq/
├── app.py                  Main Flask application (~940 lines)
├── bedrock_engine.py       AWS abstraction layer (~230 lines)
├── templates/
│   └── index.html          Frontend SPA (~1,200 lines)
├── uploads/                Ephemeral temp dir for upload transit
│                           Files here exist only during the HTTP request
│                           and are deleted in the finally block.
├── previews/               Persistent local PDF cache
│                           PDFs here are re-downloaded from S3
│                           if missing after a server restart.
├── .env                    Runtime secrets (never commit)
├── env.example             Committed template with placeholder values
├── requirements.txt        Python package dependencies
└── __init__.py             Empty — makes directory importable as package
```

---

## 3. Environment & Configuration

### env.example

```env
FLASK_SECRET_KEY=change-this-to-a-random-secret
FLASK_ENV=development
PORT=8000
MAX_UPLOAD_MB=50

AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
AWS_SESSION_TOKEN=

S3_BUCKET_NAME=nexusiq-kb-docs
BEDROCK_KB_ID=4UVNU3EXAM
BEDROCK_DS_ID=YXCOS7EXAM
BEDROCK_MODEL_ARN=arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-5-sonnet-20241022-v2:0
```

All environment variables have in-code defaults. If `.env` is missing entirely, the app starts but shows "Not Connected" until credentials are entered via the modal. This allows deploying to EC2 with IAM roles and no `.env` at all.

**Load order:** `.env` file is read first by `python-dotenv`. Runtime values entered via `/api/configure` override the `.env` values for that session but do not modify the file.

---

## 4. bedrock_engine.py — AWS Abstraction Layer

### Class: `BedrockEngine`

All AWS communication is encapsulated in this class. `app.py` never calls boto3 directly except for the direct-invoke path (`_rt_client()` / `_invoke()`).

```python
DEFAULT_MODEL_ID = "anthropic.claude-3-5-sonnet-20241022-v2:0"
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx", ".doc", ".md", ".html", ".csv"}
```

#### `__init__(region, key_id, secret, token, bucket, kb_id, ds_id, model_arn)`

Constructs three boto3 clients from a single boto3 Session:
- `self.s3` — `boto3.client("s3")` — upload, download, delete, list
- `self.agent` — `boto3.client("bedrock-agent")` — ingestion job management
- `self.rt` — `boto3.client("bedrock-agent-runtime")` — retrieve_and_generate

If `key_id` and `secret` are empty strings, boto3 falls back to the standard credential chain (IAM role, `~/.aws/credentials`, etc.). This allows EC2 instance-role operation with no credentials in the environment.

`model_arn` defaults to a fully-qualified ARN constructed from `region` and `DEFAULT_MODEL_ID` if not provided.

---

#### `validate() -> dict`

Checks three things:
1. Are `bucket`, `kb_id`, and `ds_id` all non-empty? If not, returns `{"ok": False, "error": "Missing: ..."}`.
2. Calls `s3.head_bucket(Bucket=self.bucket)` — this proves both that the bucket exists and that the credentials have `s3:HeadBucket` permission.
3. Wraps all errors in `ClientError` / `Exception` catch blocks.

**Returns:** `{"ok": True}` or `{"ok": False, "error": "<message>"}`.

This is called on every app startup (`/api/status`) and after every credential save (`/api/configure`).

---

#### `upload(local_path, filename) -> dict`

1. Validates file extension against `ALLOWED_EXTENSIONS`.
2. Constructs `s3_key = "knowledge-base/<filename>"` — all documents live under this prefix in the bucket.
3. Guesses content-type via `mimetypes.guess_type`.
4. Calls `s3.upload_file(local_path, bucket, s3_key, ExtraArgs={"ContentType": ct})`.
5. Calls `self._sync()` to start a Bedrock ingestion job immediately.
6. Returns `{"s3_key": ..., "sync_job_id": ..., "filename": ...}`.

---

#### `_sync() -> str`

Starts a Bedrock ingestion job:
```python
r = self.agent.start_ingestion_job(
    knowledgeBaseId=self.kb_id,
    dataSourceId=self.ds_id
)
return r["ingestionJob"]["ingestionJobId"]
```

This triggers Bedrock to crawl the S3 prefix, chunk any new/changed documents, embed them, and upsert them into the OpenSearch Serverless vector store. The job runs asynchronously — the job ID is returned and polled separately via `ingestion_status()`.

---

#### `ingestion_status(job_id) -> dict`

Calls `agent.get_ingestion_job(...)` and extracts:
- `status` — `IN_PROGRESS`, `COMPLETE`, `FAILED`
- `indexed` — number of documents successfully indexed
- `failed` — number of documents that failed
- `scanned` — total documents scanned

---

#### `list_docs() -> list`

Lists all objects under `knowledge-base/` prefix in S3. Filters out the bare prefix key itself (`fname != ""`). Returns structured dicts with `s3_key`, `filename`, `size_kb`, `last_modified` (formatted as "Apr 09, 2026").

---

#### `delete(s3_key) -> bool`

Deletes the object from S3 and immediately calls `_sync()` to trigger KB re-indexing so the deleted document's chunks are removed from the vector store.

---

#### `query(question, mode, source_filename) -> dict`

This is the core RAG function. Here is the complete logic:

**Step 1 — Build vector search config:**
```python
vector_cfg = {"numberOfResults": 8}
```
`numberOfResults: 8` retrieves more chunks than the default 5, improving recall for long or complex questions.

If `source_filename` is provided:
```python
vector_cfg["filter"] = {
    "equals": {
        "key":   "x-amz-bedrock-kb-source-uri",
        "value": f"s3://{self.bucket}/knowledge-base/{source_filename}"
    }
}
```
This uses Bedrock's metadata filter to restrict vector search to only chunks from that specific S3 object. This prevents stale chunks from previously-indexed documents contaminating the answer.

**Step 2 — Call retrieve_and_generate:**

The full API call structure:
```python
self.rt.retrieve_and_generate(
    input={"text": question},
    retrieveAndGenerateConfiguration={
        "type": "KNOWLEDGE_BASE",
        "knowledgeBaseConfiguration": {
            "knowledgeBaseId": self.kb_id,
            "modelArn":        self.model_arn,
            "retrievalConfiguration": {
                "vectorSearchConfiguration": vector_cfg
            },
            "generationConfiguration": {
                "promptTemplate": {
                    "textPromptTemplate": self._prompt(mode)
                },
                "inferenceConfig": {
                    "textInferenceConfig": {
                        "temperature": 0.1 if mode == "precise" else 0.45,
                        "maxTokens":   2048,
                    }
                },
            },
        },
    },
)
```

**Step 3 — Filter fallback:**

If the API call raises a `ClientError` with `ValidationException` in the code or "filter" in the message, the metadata filter isn't supported by this KB version or region. Retry silently without the filter.

**Step 4 — Extract chunk texts:**

Iterates `resp["citations"]` → each citation has `retrievedReferences` → each reference has `content.text`. Also tries `ref["text"]` as a fallback for different Bedrock API versions.

**Step 5 — Zero-chunk fallback:**

If `chunks_used == 0` after a filtered query, Bedrock received an empty `$search_results$` context. Claude would respond with a refusal/apology regardless of the question. Instead of letting that happen, always retry without the filter:
```python
if source_filename and chunks_used == 0:
    resp2 = _call({"numberOfResults": 8})
    # use resp2 results instead
```

**Step 6 — Build sources list:**

`_sources(citations)` iterates citations, extracts `location.s3Location.uri`, parses the filename from the URI, extracts the relevance score from `metadata.score` (multiplied by 100 for a percentage), deduplicates by filename.

---

#### `_prompt(mode) -> str`

Returns the system prompt injected into Bedrock's `textPromptTemplate`. The template uses Bedrock's special variables:
- `$search_results$` — replaced by Bedrock with the retrieved chunk texts
- `$query$` — replaced by Bedrock with the user's question

The base prompt instructs Claude to:
1. Answer ONLY from the provided context
2. Always mention which document the answer comes from
3. Say clearly if the answer is not in the context

Mode-specific suffixes:
- `precise` → "Be concise and factual. Use bullet points." (temperature 0.1)
- `detailed` → "Be thorough and comprehensive with full context." (temperature 0.45)
- `summary` → "Summarize in 3-5 sentences only." (temperature 0.45)

---

## 5. app.py — Flask Application

### Application Setup

```python
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "nexusiq-change-this-in-production")
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
CORS(app)
```

`SESSION_COOKIE_SAMESITE = "Lax"` prevents CSRF while allowing normal navigation. `CORS(app)` allows all origins — restrict this in production to your specific domain.

Two directories are created at startup:
- `uploads/` — temporary staging for uploaded files (files deleted after each request)
- `previews/` — persistent PDF cache (survives server restarts; re-downloaded from S3 if missing)

---

### 5.1 Global State & Session Model

```python
_doc_texts: dict = {}        # sid -> {filename: str}
_engines: dict = {}          # engine_key -> BedrockEngine
_question_cache: dict = {}   # sid -> {filename: [str]}
_question_running: set = set()  # set of (sid, filename) tuples
```

All state is keyed by **session ID (`sid`)** — a UUID hex generated per browser session. This means multiple concurrent users get completely isolated state. There is no database — all state lives in process memory.

`_question_running` is a set (not a dict) — it holds `(sid, filename)` tuples. Before spawning a background question-generation thread, the code checks if the key is already in the set, preventing duplicate concurrent Bedrock calls for the same document.

`_engines` is keyed by `engine_key`, which is constructed as `cfg["key"][:8] + cfg["bucket"]`. This allows different sessions using the same credentials/bucket to share a cached engine instance while sessions with different credentials get their own.

---

### 5.2 Session Helpers

#### `_sid() -> str`

Reads or creates a session ID:
```python
if "sid" not in session:
    session["sid"] = uuid.uuid4().hex
return session["sid"]
```
Called at the start of any request that needs session-scoped state.

#### `_session_texts() -> dict`

Returns the `{filename: text}` dict for the current session. Creates it if missing.

#### `_session_qcache() -> dict`

Returns the `{filename: [questions]}` dict for the current session. The session-scoped cache is the fix for the bug where uploading a new document returned questions from the previous document (which shared the same filename in the old global cache).

---

### 5.3 PDF Store

#### `_pdf_path(filename) -> Path`

Returns `previews/<secure_filename>`. Uses `werkzeug.utils.secure_filename` to prevent path traversal attacks even though filenames come from S3 keys the user controls.

#### `_ensure_pdf_local(filename) -> bool`

Guarantees the PDF is on disk before rendering:
1. If already in `previews/` → return `True` immediately.
2. Otherwise, download from `s3://bucket/knowledge-base/<filename>`.
3. Returns `True` if the file exists after download, `False` on any error.

This means the preview panel works correctly even after:
- Server restarts (local cache is lost but S3 is authoritative)
- Cross-session uploads (session A uploads, session B queries the same file)

---

### 5.4 Engine Factory

#### `_make_engine() -> BedrockEngine`

Constructs a `BedrockEngine` from the session config merged with environment variable fallbacks. Tries to import `bedrock_engine` from the root directory first, falls back to `utils.bedrock_engine` for alternate project layouts.

#### `get_engine() -> BedrockEngine`

Returns a cached engine from `_engines`. Uses `session["engine_key"]` as the cache key. This avoids re-constructing boto3 clients on every request while still isolating users with different credentials.

`_engines.clear()` is called in `/api/configure` to force reconstruction with new credentials.

---

### 5.5 Bedrock Direct Invoke

These functions bypass `BedrockEngine` to call Claude directly (not through the KB RAG pipeline). Used for question generation and follow-up generation where you want Claude to process raw document text, not KB chunks.

#### `_rt_client() -> boto3.client`

Builds a `bedrock-runtime` boto3 client from session/env credentials. Note: this is separate from `BedrockEngine.rt` (which is `bedrock-agent-runtime`). `bedrock-runtime` supports `invoke_model` for direct completions; `bedrock-agent-runtime` supports `retrieve_and_generate` for RAG.

#### `_model_id() -> str`

Returns the model ARN/ID from session config or environment. Defaults to `anthropic.claude-3-5-sonnet-20241022-v2:0`.

#### `_invoke(prompt, max_tokens, temperature) -> str`

Calls Claude directly via `invoke_model`:
```python
body = json.dumps({
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": max_tokens,
    "temperature": temperature,
    "messages": [{"role": "user", "content": prompt}]
})
resp = _rt_client().invoke_model(modelId=mid, body=body)
return json.loads(resp["body"].read())["content"][0]["text"].strip()
```

#### `_parse_list(text) -> list`

Safe JSON array extractor. Finds the first `[` and last `]` in the text and attempts to parse just that slice. Handles cases where Claude adds preamble or explanation before/after the JSON.

---

### 5.6 Text Extraction

#### `extract_text(filepath, filename) -> str`

Extracts plain text from uploaded files for question generation.

**PDF (`.pdf`):**
1. Tries PyMuPDF (`fitz`): opens the file, extracts text from first 15 pages via `page.get_text()`. Best quality — preserves layout, handles columnar text.
2. Falls back to pypdf: pure Python, lower quality but no binary deps.
3. Returns empty string if both fail.

**DOCX/DOC:**
Uses `python-docx`: iterates `Document(filepath).paragraphs` and joins with newlines. Does not handle tables or text boxes.

**All other types (TXT, MD, HTML, CSV):**
Opens as plain text with `errors="ignore"`, reads first 50,000 characters.

---

### 5.7 AI Features — Question Generation

#### `generate_questions(text, filename) -> list`

Synchronous version. Checks session cache first. Calls `_invoke` with a prompt asking Claude to produce exactly 6 questions as a JSON array. Returns cached result on subsequent calls.

Only used in code paths where a synchronous result is acceptable. The primary path for question generation is the async background thread.

#### `_generate_questions_async(text, filename, cfg_snapshot, sid) -> None`

Background thread function. Runs outside the Flask request context, so it cannot use `session` or `app` context. Instead:
- It receives `cfg_snapshot` (a plain dict copy of the session config)
- It receives `sid` (the session ID string)
- It directly manipulates `_question_cache[sid]` (a module-level dict, safe across threads)
- It directly manipulates `_question_running` (a module-level set)

**Guard pattern:**
```python
key = (sid, filename)
if key in _question_running:
    return
_question_running.add(key)
try:
    # ... generate questions ...
    _question_cache.setdefault(sid, {})[filename] = qs[:6]
finally:
    _question_running.discard(key)
```

The `finally` ensures the running-key is always removed even if Bedrock throws an exception, so future poll requests can retry.

**Prompt design:** The prompt asks for exactly 6 questions that reference actual content (names, numbers, concepts, dates) with a max of 12 words each. This produces specific, answerable questions rather than generic ones like "What is this document about?". Using `max_tokens=300` and `temperature=0.2` keeps generation fast and deterministic.

---

### 5.8 AI Features — Follow-up Generation

#### `generate_followups(question, answer) -> list`

Called synchronously during the `/api/query` response. Takes the user's question and the RAG answer (first 400 chars), asks Claude to produce 3 natural follow-up questions as a JSON array.

`max_tokens=150` and `temperature=0.4` — fast, slightly varied. The follow-ups are generated after the RAG answer is received, adding minimal latency to the response (~300-600ms for Bedrock to generate 3 short questions).

---

### 5.9 Highlight Phrase Extraction

#### `extract_highlight_phrases(answer, chunk_texts) -> list`

Builds a list of up to 12 phrases that will be searched for inside the PDF to generate highlights. Six-tier priority system:

**Internal helper `_add(phrase) -> bool`:**
- Normalises whitespace
- Strips trailing source citations like `(Source 1)` that Bedrock sometimes appends
- Deduplicates by the first 45 lowercase characters
- Returns `False` (stop adding) once 12 phrases are collected

**Tier 1 — Verbatim chunk sentences:**
The raw chunk texts from Bedrock are the most reliable source. Sentences 30–100 chars are extracted. Min 30 chars prevents trivially short fragments from matching everywhere; max 100 chars keeps PyMuPDF's `search_for` fast.

Unlike the original implementation, Tier 1 does NOT early-return. Even if chunk sentences are found, lower tiers continue to supplement the list. The ubiquity filter in the renderer handles any footer/header noise.

**Tier 2 — Bold markdown headings:**
Regex `\*\*([^*]{3,60})\*\*` extracts text Claude wrapped in `**...**`. These are section headings from the document that Claude marked as key terms — PyMuPDF can find them exactly.

**Tier 3 — Quoted strings:**
Regex `"([^"]{8,90})"` extracts text Claude put in quotation marks — explicit citations of document text.

**Tier 4 — Number + meaningful words:**
Regex `\b\d+\.?\d*(?:\s+[A-Za-z][A-Za-z0-9\-]{1,25}){3,6}` finds things like "2.5 days per calendar month" or "28.4 BLEU on WMT 2014". Stop-word filter prevents matching "5 of the clauses" which would match TOC pages.

**Tier 5 — Title-Case 3+-word phrases:**
Regex `[A-Z][a-zA-Z]{2,}(?:\s+(?:[A-Z][a-zA-Z]*|\d+)){2,5}` matches "Earned Leave Policy", "Transformer Self Attention", "WMT 2014 English". Requires 3+ words and 12+ chars to avoid short 2-word headers like "HR Department" that appear in footers.

**Tier 6 — Full answer sentences:**
Last resort. Splits the cleaned answer on sentence-ending punctuation (with a lookbehind to avoid splitting "28.4" → "28" + "4"). Uses sentences in the 30–115 char sweet spot.

---

### 5.10 Page Number Extraction

#### `extract_page_numbers(answer) -> list`

Finds explicit page references in the answer using four regex patterns:
- `pages 3-7` / `pages 3 to 7` → adds both endpoint pages
- `page 5` / `pages 12`
- `p.12` / `p. 4`
- `(p.4)` / `(p 4)`

Returns a sorted, deduplicated list of integers in range 1–500. The 500 cap prevents runaway from false positives in very large documents.

Used as a fallback navigation target when no highlight phrase matches any page — in that case, the frontend jumps to the first page number the answer mentioned.

---

### 5.11 PDF Phrase Search

#### `_search_phrase(pg, phrase) -> list`

Takes a PyMuPDF page object and a phrase string. Tries progressively shorter search strategies until something matches:

**Strategy 1 — Full phrase:**
`pg.search_for(phrase, flags=fitz.TEXT_IGNORE_CASE)`. Case-insensitive full-text search. Returns a list of `Rect` objects (bounding boxes). Fast.

**Strategy 2 — First 52 chars, trimmed to word boundary:**
If the phrase is over 40 chars, try just the first 52 chars trimmed to the last complete word. Handles cases where Claude slightly reformulated a sentence.

**Strategy 3 — Sliding 4-word windows:**
Only used for phrases with 6+ words. Requires each 4-word window to either contain a digit or have 3+ capitalized words. This prevents stop-word windows like "in the of a" from matching on every page. Window size raised from 3 to 4 words compared to earlier versions to reduce false positives.

---

### 5.12 PDF Renderer

#### `render_pdf_pages(pdf_path, highlight_texts) -> dict`

The most complex function in the application. Renders only the pages that contain relevant content, with yellow highlights.

**Step 0 — Pre-scan all page texts:**
```python
page_texts = [doc[i].get_text() for i in range(total)]
```
All page texts are read once into memory. This single scan is reused for both the ubiquity filter and the highlight search pre-screen, avoiding re-opening the page multiple times.

**Step 0b — Ubiquity filter:**

For each phrase, count how many pages its first 24 chars appear on:
```python
match_count = sum(1 for pt in page_texts if probe in pt.lower())
```
Threshold: `max(3, min(int(total * 0.25), 12))`.

For a 208-page document: threshold = `min(52, 12) = 12`. A phrase appearing on 13+ pages is considered structural and discarded.
For a 10-page document: threshold = `max(3, 2) = 3`. A phrase appearing on 4+ of 10 pages is discarded.

This is the primary fix for the "footer highlighting" bug where phrases like "IIMA HR Policy Manual" (printed on every page footer) were being highlighted on random pages instead of relevant ones.

**Step 1 — Build highlight map:**

For each page and each filtered phrase:
1. Quick pre-screen: does the first 24 chars of the phrase appear in the page text?
2. If yes, call `_search_phrase()` for full search.
3. If match found, check `y_ratio = rects[0].y0 / pg_height`.
4. Footer heuristic: if `y_ratio > 0.90` (bottom 10% of page) AND `len(phrase) < 35`, skip — it's almost certainly a page footer.
5. Store in `hl_map[page_index]` with y_ratio, phrase, and rects.

**Step 2 — Select pages to render:**

If highlights found: render each highlighted page ±1 buffer page (so the reader sees context above and below the highlighted passage). On a 208-page doc with 3 highlighted pages, this renders at most 9 pages.

If no highlights: render first 3 pages (cover + intro as a document preview).

This selective rendering is critical for performance — rendering all 208 pages would cause a multi-second timeout.

**Step 3 — Render and annotate:**

For highlighted pages, scale = 1.5×. For buffer pages, scale = 1.2×.

Highlight annotation:
```python
annot = pg.add_highlight_annot(rect)
annot.set_colors(stroke=[1, 0.85, 0.0])  # yellow
annot.update()
```

Then render to PNG:
```python
pix = pg.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
data = base64.b64encode(pix.tobytes("png")).decode()
```

Each rendered page dict includes `highlight_y_ratio` — the Y position of the highlight as a fraction of page height. This is used by the frontend to scroll to the exact pixel position within the page image.

---

### 5.13 Single Page Renderer

#### `render_single_page(pdf_path, page_num, scale) -> str | None`

Renders a single page (1-based index) without any highlights. Used by the `/api/page/<filename>/<page_num>` route for on-demand page loading during navigation. Returns a base64 PNG string or `None` on error.

---

### 5.14 Routes Reference

#### `GET /`
Returns `render_template("index.html")`. Flask looks for `index.html` in the `templates/` subdirectory.

#### `GET /health`
Simple liveness probe. Returns `{"status": "ok"}`. Used by load balancers and deployment health checks.

#### `POST /api/configure`
Stores credentials in `session["cfg"]`. Constructs `engine_key` as `key[:8] + bucket`. Clears `_engines` to force reconstruction. Validates and returns stats.

#### `GET /api/status`
Validates current credentials and returns KB stats. Called once on page load to restore the connected state from a previous session.

#### `POST /api/upload`
Full upload pipeline:
1. Save to temp file in `uploads/`
2. `extract_text()` — local text extraction for question gen
3. Store text in `_session_texts()`
4. Set `session["active_file"]`
5. Copy PDF to `previews/` if applicable
6. `engine.upload()` → S3 + Bedrock ingestion start
7. Spawn `_generate_questions_async()` thread (fire-and-forget)
8. Return immediately with `questions: []` — frontend polls

Critical: the `finally` block deletes the temp file even on error:
```python
finally:
    if tmp.exists():
        tmp.unlink()
```

#### `GET /api/questions/<filename>`
Non-blocking poll endpoint. Checks session cache first for instant return. If not cached, ensures text is available (extracts from local PDF if needed), fires background thread if not already running, returns `{"questions": [], "ready": false}` immediately. Frontend polls every 2 seconds.

#### `POST /api/query`
Full RAG pipeline:
1. Get `active_file` from request body or session fallback
2. `engine.query()` with source scoping
3. `generate_followups()` — 3 follow-up questions
4. `extract_highlight_phrases()` — phrases to find in PDF
5. `extract_page_numbers()` — page refs from answer
6. Resolve `preview_file`: source → session → latest disk PDF
7. Return all results in one JSON response

#### `GET /api/preview/<filename>?highlights=...`
Parses `||`-delimited highlight phrases from query param. Calls `render_pdf_pages()`. Returns pages array with metadata including `first_hl_page` and `first_hl_y_ratio` for frontend scroll targeting.

#### `GET /api/page/<filename>/<page_num>`
On-demand single page renderer. Opens the PDF, gets total page count, renders requested page. Used for navigation to pages not in the initial preview render.

#### `POST /api/documents/delete`
Cleans up: session texts, session question cache, running-set, local PDF file, then calls `engine.delete()` for S3 + KB re-sync.

#### `GET /api/ingestion/<job_id>`
Proxies `engine.ingestion_status()`. Polled every 3 seconds by the frontend after upload.

#### `GET /api/pdf/<filename>`
Serves raw PDF binary via Flask's `send_file` with `conditional=True` (supports HTTP range requests and ETags for efficient re-fetching by the browser iframe).

#### `GET /api/debug/pdfs`
Lists PDFs in `previews/` with sizes. Not authenticated — disable or restrict in production.

---

## 6. index.html — Frontend

The entire frontend is a single HTML file (~1,200 lines) using vanilla JavaScript with no framework, no npm, no build step. This was an intentional choice: the application runs on a single EC2 instance with no static asset pipeline.

### Layout Structure

```html
<header>          Brand + connection pill + settings icon + new chat button
<div class="app">
  <aside class="sidebar">
    ├── AWS Connection status block
    ├── Upload dropzone + progress bar + ingestion card
    ├── Document list (scrollable)
    └── Stats bar (docs / size / region)
  <div class="chatpanel">
    ├── Toolbar (mode selector + KB tag)
    ├── Chat area (messages)
    ├── Chips area (question suggestions)
    └── Input area (textarea + send button)
  <div class="docpanel">
    ├── Panel header (filename + close)
    ├── Panel body (PDF page images)
    └── Nav bar (prev/next + page input + HL jump buttons)
<div class="mov">  AWS config modal
<div class="tstack">  Toast notifications
```

---

### 6.1 CSS Design System

The design is built entirely on CSS custom properties, allowing consistent theming across all 1,200+ lines.

```css
/* Background depth scale (darkest to lightest) */
--ink:  #03040a    /* body background */
--ink1: #080b14
--ink2: #0e1220    /* page cards, inputs */
--ink3: #151c2e    /* hover states */
--ink4: #1c253c
--ink5: #253048

/* Primary accent — indigo */
--indigo: #6366f1
--ih:     #818cf8   /* lighter indigo for text on dark */
--ilo:    rgba(99,102,241,.10)   /* indigo low opacity */
--imd:    rgba(99,102,241,.20)   /* indigo medium opacity */
--iglow:  rgba(99,102,241,.35)   /* indigo glow for shadows */

/* Secondary accent — cyan */
--cyan: #22d3ee
--clo:  rgba(34,211,238,.08)
--cmd:  rgba(34,211,238,.18)

/* Semantic colours */
--em:   #10b981    /* success / connected */
--rose: #f43f5e    /* error / delete */
--am:   #f59e0b    /* warning / highlight amber */
--vi:   #a78bfa    /* violet for gradients */

/* Text scale */
--t0: #fff
--t1: #e8edf5   /* primary body text */
--t2: #94a3b8   /* secondary text */
--t3: #4a5568   /* muted / labels */
--t4: #252f42   /* scrollbar thumbs */

/* Surface materials */
--glass:  rgba(14,18,32,.78)    /* chat message background */
--glass2: rgba(21,28,46,.65)    /* input shell */
--b0: rgba(255,255,255,.04)     /* subtle borders */
--b1: rgba(255,255,255,.08)
--b2: rgba(255,255,255,.14)
--bi: rgba(99,102,241,.25)      /* indigo border */
--bc: rgba(34,211,238,.2)       /* cyan border */

/* Radii */
--r:  10px
--rl: 16px    /* large — message bubbles */
--rx: 22px    /* extra large — modal */

/* Easing */
--spring: cubic-bezier(.34,1.56,.64,1)   /* overshoot spring */
--ease:   cubic-bezier(.4,0,.2,1)        /* material ease */
```

**Ambient background:** `.bg-grid` uses a repeating linear gradient to create a subtle grid pattern. `.glow.g1` and `.glow.g2` are fixed `filter:blur(130px)` divs in the corners that create ambient colour bleed. The originally-present `.glow.g3` (centre) was removed as it caused repaint performance issues during scroll and animation.

---

### 6.2 JavaScript State Model

```javascript
// Application state
const S = {
  connected:           false,  // true after successful /api/configure or /api/status
  loading:             false,  // true while /api/query is in flight
  docs:                [],     // array of doc objects from /api/documents
  jTimer:              null,   // setInterval ID for ingestion polling
  activeFile:          null,   // currently selected filename (string)
  conversationStarted: false   // set true on first sent message; hides chips
};

// UI state
let currentHighlights = [];   // phrases from last query response
let _panelFile        = null; // filename last loaded into panel (unused after refactor)
let _queryAbort       = null; // AbortController for in-flight fetch
let _pollingFor       = null; // filename currently being polled for questions

// PDF panel state
const _pdf = {
  file:      null,   // filename currently in panel
  total:     0,      // total pages in document
  current:   1,      // current page number (displayed in nav input)
  hlPages:   [],     // highlighted page numbers (1-based array)
  lastHlKey: ''      // fingerprint of last highlights set (prevents redundant re-renders)
};
```

`STORE = 'nxq5_cfg'` — localStorage key for persisting AWS credentials across page refreshes.

---

### 6.3 Boot Sequence

```javascript
document.addEventListener('DOMContentLoaded', () => {
  const saved = rStore();
  if (saved) prefill(saved);      // pre-fill modal with saved credentials
  $('qi').addEventListener('input', updBtn);
  checkStatus();                   // GET /api/status
});
```

`checkStatus()`:
- On success: calls `setOn(r)` to update UI state, then `loadDocs(true)` with `autoQ=true` to load the document list and start question generation for the most recently active file.
- On failure: calls `setOff()` to show the disconnected state.

`prefill(c)` fills the modal input fields with saved credentials. The modal does not auto-submit — the user must click "Connect" or the save button.

---

### 6.4 AWS Config Modal

`openModal()` / `closeModal()` toggle the `open` class on `.mov`. The modal uses CSS transitions for the fade-in and scale-up effect.

`saveConfig()`:
1. Reads all 8 input fields
2. Validates that required fields (key, secret, bucket, kb, ds) are non-empty
3. POSTs to `/api/configure`
4. On success: `wStore(c)` saves to localStorage, `setOn(r)`, `loadDocs(true)`
5. On failure: shows error in `.merr` div

`setOn(d)` / `setOff(msg)` update:
- Connection pill colour and pulse animation
- Sidebar status block text and colour
- AWS info grid (region, bucket, KB ID)
- Stats bar region display
- Send button enabled state

---

### 6.5 Question Chips System

Chips are AI-generated questions shown as clickable suggestion buttons before the first message in a conversation. They disappear permanently once the user sends their first message.

**Visibility rules:**
- Only shown when `!S.conversationStarted && S.docs.length > 0`
- Hidden by `hideChips()` when conversation starts or no docs exist
- Container (`.chips-area`) uses `display:none` / `display:flex` toggled by `.visible` class

**Loading state:**
`showChipsLoading(filename)` shows a spinning indicator and updates the label to "Generating questions from filename".

**Ready state:**
`updateChips(questions, filename)` renders question buttons with staggered fade-in animation (60ms delay per chip). Each chip uses `data-q` attribute for the question text and `onclick=fill(q)` to populate the textarea.

**Polling flow:**
1. Upload completes → `showChipsLoading(lastFile)` → `fetchAndShowQuestions(lastFile, 15)`
2. `fetchAndShowQuestions` polls `/api/questions/<filename>` every 2 seconds
3. Guard: `if (_pollingFor !== filename) return` — cancels stale polls when a new doc is selected
4. `retries=15` gives a 30-second window before giving up and hiding chips
5. On success: `updateChips(r.questions, filename)`, `_pollingFor = null`

**New chat behaviour:**
`newChat()` resets `S.conversationStarted = false` and `_pollingFor = null` then calls `fetchAndShowQuestions(S.activeFile, 15)` to re-show chips for the active document. Crucially, it does NOT call `closePanel()` — the PDF preview stays open.

---

### 6.6 Upload Pipeline

`doUp(files)`:
1. Shows upload progress bar
2. For each file: `showChipsLoading(filename)`, POST `multipart/form-data` to `/api/upload`
3. On success: `S.activeFile = lastFile`, `fetchAndShowQuestions(lastFile, 15)` (immediate start, no delay)
4. After all files: hide progress bar, `pollJob(job, lastFile)`, `loadDocs()`
5. If PDF: `setTimeout(() => openPanel(lastFile, []), 300)` — opens preview with 300ms delay to let the DOM settle

The key difference from the original implementation is that `fetchAndShowQuestions` is called immediately (no 3-second delay) and with `retries=15` (30-second window, 2-second intervals).

---

### 6.7 Ingestion Sync Polling

`pollJob(id, filename)`:
- Shows the ingestion card (`.icard`) with "SYNCING" badge
- Sets an `S.jTimer = setInterval(...)` polling `/api/ingestion/<id>` every 3 seconds
- Updates badge and text based on status: `IN_PROGRESS` → running dots; `COMPLETE` → green badge + refresh doc list + toast; `FAILED` → red badge + toast
- Hides the card 5 seconds after completion

The `clearInterval(S.jTimer)` at the start of `pollJob` prevents multiple concurrent timers if the user uploads multiple files.

---

### 6.8 Document List

`loadDocs(autoQ)`:
- GETs `/api/documents`
- Updates `S.docs`, calls `renderDocs()`
- Updates stats bar values
- If `autoQ=true` and not in conversation: selects the most recently active doc (or first doc), calls `fetchAndShowQuestions`

`renderDocs()` builds HTML for each document card. Uses `data-` attributes to avoid inline function call escaping issues. Shows active state (left border highlight + indigo background) for `S.activeFile`.

`selectDoc(name)`:
- Sets `S.activeFile = name`
- Sets `_pollingFor = name` (cancels any in-flight poll for the previous doc)
- Opens PDF panel if PDF
- Starts question chip polling for the newly selected doc

`delDoc(e, k, n)`:
- Confirms with `confirm()` dialog
- POSTs to `/api/documents/delete`
- Resets `_pollingFor` if deleting the currently polled doc
- Closes panel if deleting the active file

---

### 6.9 Query & Answer Pipeline

`sendMsg()`:
1. Guards: `!q || S.loading || !S.connected`
2. Aborts any in-flight query: `_queryAbort.abort()` — prevents ghost responses from previous queries
3. Hides chips (sets `S.conversationStarted = true`)
4. Removes welcome screen (`rmW()`)
5. Adds user message bubble (`addU(q)`)
6. Adds typing indicator (`addTyping()`)
7. Creates new `AbortController`, stores as `_queryAbort`
8. POSTs to `/api/query` with `{question, mode, active_file: S.activeFile}`
9. On `AbortError`: silent return (user sent new message)
10. On success: `addA(r, ms)`, open PDF panel with highlights

`addA(data, ms)` builds the answer message HTML:
- Renders answer markdown via `renderMD()`
- Sources section with clickable `.srcpill` elements (clicking opens PDF at that source)
- Follow-up questions as `.fuchip` chips
- Message footer with chunk count badge, "Amazon Bedrock" badge, elapsed time, "View in doc" button

`addErr(msg)` shows a red-bordered error bubble.

---

### 6.10 PDF Viewer Panel

`openPanel(filename, highlights, answerPages)`:

1. Shows panel, updates header (filename + extension tag)
2. Non-PDF files: shows "Only PDF files support in-app preview" empty state
3. Skip-render guard: `if (_pdf.file === filename && _pdf.lastHlKey === hlKey && !highlights.length) return` — avoids redundant re-renders for identical requests
4. Shows loading spinner while fetching
5. GETs `/api/preview/<filename>?highlights=...` (phrases URL-encoded, joined by `||`)
6. On success with pages: builds page card HTML, calls `_updateNav()`, scrolls to highlight

**Scroll logic:**
- If `hasHL && firstHL`: `_scrollToHighlight(body, firstHL, firstY)` — scrolls to exact Y position
- Else if `answerPages.length > 0`: `navToPage(targetPage)` — jumps to first mentioned page
- Else: `body.scrollTop = 0` — scroll to top

**`_scrollToHighlight(body, pageNum, yRatio)`:**
1. Finds the page element by `#dpp-<pageNum>`
2. Waits for the page image to load if not yet complete
3. Calculates `target = pageEl.offsetTop + (imgH * yRatio) - OFFSET`
4. `body.scrollTo({top: target, behavior: 'smooth'})`
5. Applies a gold glow animation to the page element for 2.5 seconds

**Fallback (no PyMuPDF or page error):** `_renderEmbedFallback()` creates an `<iframe>` pointing at `/api/pdf/<filename>` — shows the raw browser PDF viewer.

---

### 6.11 Page Navigation

`_updateNav(total, hlPages, answerPages)`:
- Shows the nav bar
- Creates gold highlight-jump buttons for up to 4 highlighted pages
- Creates indigo page-jump buttons for answer-referenced pages not already in HL list
- Updates page input max and current value

`navPage(delta)`: relative navigation, calls `navToPage(current + delta)`.

`navToPage(pageNum)`:
1. Checks if page element `#dpp-<pageNum>` is already rendered — if yes, smooth-scrolls to it with a brief indigo glow ring
2. If not rendered: creates a loading spinner, fetches `/api/page/<filename>/<pageNum>`, creates and appends a new `.dppage` element, smooth-scrolls to it

This lazy-loading pattern means only the pages the user actually navigates to are rendered. For a 208-page PDF, the initial load renders ~3-9 pages; browsing adds pages one at a time.

---

### 6.12 Utility Functions

`req(url, opts)`: Wrapper around `fetch` that always sets `Content-Type: application/json` and JSON-serialises the body. Returns parsed JSON or `null` on network error. Note: does NOT support AbortController (that's handled directly in `sendMsg` for the query path).

`toast(msg, type)`: Adds a toast notification to `.tstack`. Types: `ok`, `err`, `info`, `warn`. Auto-removes after 3.5 seconds with a slide-out animation.

`esc(s)`: HTML entity escaping for user-supplied strings inserted into innerHTML.

`renderMD(s)`: A minimal Markdown renderer handling: `###`/`##`/`#` headers, `**bold**`, `*italic*`, `` `code` ``, `---` horizontal rules, `> blockquotes`, `- * •` bullet lists, and double newline paragraph breaks. Not a full Markdown parser — intentionally minimal for performance.

`autoh(el)`: Auto-resize textarea on input. Sets height to `auto` then to `scrollHeight` (capped at 130px).

`rmW()`: Removes the welcome screen div from the chat area.

`updBtn()`: Enables/disables the send button based on `input value + S.connected + !S.loading`.

---

## 7. Data Flow Diagrams

### Upload Flow

```
User drags file
      │
      ▼
doUp() [browser]
  └── showChipsLoading(filename)
  └── POST /api/upload (multipart)
           │
           ▼
      upload() [Flask]
        ├── save to uploads/<uuid>_<filename>  (temp)
        ├── extract_text()  ──────────────────────── PyMuPDF → pypdf → plain text
        ├── _session_texts()[filename] = text
        ├── session["active_file"] = filename
        ├── shutil.copy2() to previews/           (PDFs only)
        ├── engine.upload()
        │     ├── s3.upload_file()               ──► S3 bucket
        │     └── agent.start_ingestion_job()    ──► Bedrock KB sync starts
        ├── Thread(_generate_questions_async, daemon=True).start()
        │     └── [runs in background]
        │           ├── client.invoke_model(prompt)   ──► Bedrock Runtime
        │           └── _question_cache[sid][filename] = questions
        └── return {success, questions:[], has_preview, s3_key, sync_job_id}
           │
           ▼
      doUp() resumes [browser]
        ├── S.activeFile = lastFile
        ├── fetchAndShowQuestions(lastFile, 15)  ─── polls every 2s
        ├── pollJob(sync_job_id)                 ─── polls every 3s
        ├── loadDocs()
        └── openPanel(lastFile, [])              ─── PDF preview (PDFs only)
```

### Query Flow

```
User types question + Enter
      │
      ▼
sendMsg() [browser]
  ├── _queryAbort.abort() if in-flight
  ├── addU(question)
  ├── addTyping()
  └── POST /api/query {question, mode, active_file}
           │
           ▼
      query() [Flask]
        ├── engine.query(question, mode, active_file)
        │     ├── retrieve_and_generate() ──────────►
        │     │    Bedrock KB: retrieve top-8 chunks (scoped to active_file if set)
        │     │    Claude 3.5: generate answer from chunks
        │     │    ◄────────────────────────────────
        │     ├── [if 0 chunks]: retry without filter
        │     └── return {answer, sources, chunks_used, chunk_texts}
        │
        ├── generate_followups(question, answer)
        │     └── invoke_model() ──────────────────►
        │          Claude 3.5: 3 follow-up questions
        │          ◄──────────────────────────────
        │
        ├── extract_highlight_phrases(answer, chunk_texts)
        │     └── 6-tier regex extraction → list of ≤12 phrases
        │
        ├── extract_page_numbers(answer)
        │     └── regex → list of page numbers
        │
        ├── resolve preview_file (sources → session → disk)
        │
        └── return {answer, sources, chunks_used, followups,
                    highlight_phrases, answer_pages, preview_file}
           │
           ▼
      sendMsg() resumes [browser]
        ├── rmTyping()
        ├── addA(result, ms)
        └── openPanel(pdf, highlights, answerPages)
               │
               ▼
          openPanel() [browser]
            └── GET /api/preview/<filename>?highlights=...
                     │
                     ▼
                preview() [Flask]
                  ├── _ensure_pdf_local()  ──► download from S3 if needed
                  └── render_pdf_pages(pdf_path, hl_list)
                        ├── Step 0: read all page texts
                        ├── Step 0b: ubiquity filter
                        ├── Step 1: build highlight map
                        ├── Step 2: select pages to render
                        └── Step 3: render to base64 PNG with fitz annotations
                     │
                     ▼
            openPanel() builds page HTML
            _scrollToHighlight(body, firstHL, firstY)
```

---

## 8. Design Decisions & Engineering Notes

### Why vanilla JS and no framework?
The entire frontend runs from a single HTML file served by Flask's `render_template`. No npm, no webpack, no build pipeline. This was an intentional simplicity tradeoff — the app runs on a single EC2 instance with `python app.py`. Adding a React/Vue build step would require a separate static asset server or a build pipeline. For a single-developer project the maintenance cost isn't worth it.

### Why not stream RAG responses?
Bedrock's `retrieve_and_generate` API does not support streaming responses. The entire answer arrives at once. A streaming experience would require switching to the two-step API: `retrieve()` first, then `invoke_model()` with streaming and manually constructing the RAG prompt. That was a future enhancement, not implemented here.

### Why per-session question cache instead of global?
The original implementation used `_question_cache: dict = {}` keyed only by `filename`. This caused a subtle bug: if two different users (or the same user after a page refresh) uploaded a file with the same name but different content, the second user would receive questions generated from the first user's document content. Per-session keying (`sid + filename`) ensures complete isolation.

### Why fire-and-forget for question generation?
The original implementation called `t.join(timeout=25)` — blocking the HTTP response for up to 25 seconds waiting for Bedrock. This made every upload feel like a hang. The fix: spawn the thread and return immediately. The frontend polls `/api/questions` every 2 seconds. Bedrock typically generates 6 short questions in 3-6 seconds, so the chips appear within one or two poll cycles.

### Why the ubiquity filter?
PyMuPDF's `search_for` is exact text matching, not semantic. Short phrases from the AI answer like "HR Department" appear in the footer of every page of an HR manual. Without filtering, the highlight system would match the very first page that contained the footer text — usually the cover or TOC — rather than the page with the relevant content. The ubiquity filter (>25% of pages) reliably identifies structural text.

### Why Y-ratio scrolling?
When the PDF is rendered to PNG images, we don't know the pixel position of a highlighted text within the image until the image is rendered. PyMuPDF returns the highlight rect's `y0` coordinate in PDF points. Dividing by `page.rect.height` gives a 0-1 ratio. On the frontend, this is multiplied by the image's rendered `clientHeight` to get the exact pixel offset to scroll to. This works regardless of window size or browser zoom level.

### Why AbortController on queries?
Without AbortController, if a user types a message, changes their mind, and types another message quickly, two queries are in flight simultaneously. The first query's answer arrives after the second query's answer, appearing as a second response at the wrong position in the chat. AbortController.abort() cancels the first fetch request, ensuring only the most recent query produces a response.

### Why selective page rendering (not all pages)?
A 208-page PDF rendered at 1.5× scale produces ~208 × 200KB = 41MB of PNG data. This would time out the `/api/preview` request and consume enormous memory. Rendering only ±1 pages around each highlighted passage (typically 3-9 pages total) keeps response size under 2MB.

### Why the source-scoping filter in Bedrock?
When a Bedrock Knowledge Base contains multiple documents, vector similarity search retrieves chunks from any document that is semantically similar to the question. If you indexed an HR policy and then an ML research paper, asking "what is attention?" might retrieve chunks from both. By filtering to the active document's S3 URI, queries are scoped to the document the user is currently working with. The fallback (retry without filter if 0 chunks) handles the case where the indexed URI doesn't exactly match (different regions, different sync states, etc.).

---

## 9. Known Limitations & Future Work

### Current Limitations

- **No persistent authentication:** Credentials are stored in Flask's server-side session (cookie-based) and localStorage. A server restart clears all sessions. For production, use a proper auth system (Cognito, OAuth).
- **Single-instance only:** `_doc_texts`, `_question_cache`, and `_engines` are in-process dicts. Multiple Gunicorn workers with `-w 4` will NOT share state. Run with `-w 1` or use a shared store (Redis) for multi-worker deployment.
- **No DOCX/TXT preview:** The right panel only supports PDFs. Other file types show an empty state.
- **No conversation history:** Each query is independent. The RAG system has no memory of previous turns. Claude's answer doesn't benefit from context built up in the conversation.
- **KB sync is not instant:** After upload, the Bedrock ingestion job typically takes 30-120 seconds. Queries made before the job completes may not include the newly uploaded document's content.
- **Text extraction is first 15 pages only:** `extract_text()` for question generation reads at most 15 pages. Questions about content deep in a long document may not be generated.

### Suggested Future Improvements

- **Multi-turn conversation:** Store `(role, content)` history in session and pass it to `retrieve_and_generate` for contextual follow-ups.
- **Redis session backend:** Replace in-process dicts with Redis for multi-worker Gunicorn compatibility.
- **Streaming answers:** Implement two-step RAG: `retrieve()` → `invoke_model(stream=True)` for progressive answer display.
- **DOCX preview:** Convert DOCX to PDF server-side (using LibreOffice headless) and serve through the existing PDF pipeline.
- **Persistent document metadata:** Store upload history in DynamoDB/SQLite so the document list survives server restarts without an S3 round-trip.
- **Authentication:** Add API key or Cognito auth to all `/api/` routes for multi-tenant deployments.
- **CDN for font assets:** Currently loading Syne, JetBrains Mono, and Figtree from Google Fonts on every page load. Self-host or use a CDN for better performance in restricted environments.
