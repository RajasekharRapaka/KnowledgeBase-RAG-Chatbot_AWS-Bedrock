<div align="center">

<img src="https://img.shields.io/badge/Amazon%20Bedrock-RAG-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white"/>
<img src="https://img.shields.io/badge/Claude%203.5%20Sonnet-Anthropic-6366f1?style=for-the-badge&logo=anthropic&logoColor=white"/>
<img src="https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask&logoColor=white"/>
<img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/AWS%20S3-Storage-569A31?style=for-the-badge&logo=amazons3&logoColor=white"/>

<br/><br/>

```
                              ███╗   ██╗███████╗██╗  ██╗██╗   ██╗███████╗  ██╗ ██████╗
                              ████╗  ██║██╔════╝╚██╗██╔╝██║   ██║██╔════╝  ██║██╔═══██╗
                              ██╔██╗ ██║█████╗   ╚███╔╝ ██║   ██║███████╗  ██║██║   ██║
                              ██║╚██╗██║██╔══╝   ██╔██╗ ██║   ██║╚════██║  ██║██║▄▄ ██║
                              ██║ ╚████║███████╗██╔╝ ██╗╚██████╔╝███████║  ██║╚██████╔╝
                              ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝  ╚═╝ ╚══▀▀═╝
```

### Enterprise Knowledge Base Intelligence Platform
<img width="1366" height="645" alt="Screenshot 2026-04-10 at 11 40 19 PM" src="https://github.com/user-attachments/assets/8077aea3-c3c8-437b-bfdd-ee4e8c026394" />


**Ask an LLM questions across your private documents — powered by Amazon Bedrock RAG + Claude 3.5 Sonnet. Zero hallucinations. Full source citations. Live PDF highlighting.**

<br/>

</div>

---

---

## What is NexusIQ?

NexusIQ is a **production-ready Enterprise Knowledge Base Q&A system** that lets you query your private company documents using natural language. It uses **Retrieval-Augmented Generation (RAG)** via Amazon Bedrock — meaning answers are always grounded in your actual documents, never hallucinated.

Upload any PDF, DOCX, TXT, Markdown, CSV or HTML file. NexusIQ pushes it to your S3 bucket, indexes it in a Bedrock Knowledge Base, and makes it instantly queryable. Every answer comes with source citations and a live PDF panel that scrolls to and highlights the exact passage the answer came from.

---

## Key Features

| Feature | Description |
|---|---|
| **RAG via Amazon Bedrock** | Retrieve-and-generate using your private KB. Answers are grounded in your documents. |
| **Live PDF Highlighting** | After every answer, the right panel opens the source PDF and highlights the exact retrieved passage with smart ubiquity filtering (ignores headers/footers). |
| **Multi-mode Answers** | Switch between Precise (bullet points), Detailed (full context), and Summary (3-5 sentences) modes. |
| **AI-generated Question Chips** | Claude reads each document and generates 6 contextual questions to get you started — fully async, never blocks the UI. |
| **Follow-up Questions** | Every answer is accompanied by 3 AI-generated follow-up questions as clickable chips. |
| **Per-session Document Isolation** | Question caches and document state are strictly per browser session — uploading a different document never serves stale questions. |
| **Source Scoping** | Queries can be filtered to a specific document's S3 URI via Bedrock metadata filter, preventing cross-document contamination. |
| **On-demand Page Navigation** | Browse any page of any PDF with prev/next nav and jump-to-page — pages rendered on demand to avoid timeout on large docs. |
| **Auto S3 Recovery** | If the server restarts, PDFs are re-downloaded from S3 automatically — preview panel always works. |
| **Drag & Drop Upload** | Multi-file drag-and-drop with real-time upload progress bar and Bedrock ingestion sync status polling. |
| **Dark Glass UI** | Premium dark theme with glassmorphism, spring-physics animations, and Syne + JetBrains Mono + Figtree fonts. |
| **AbortController Queries** | Sending a new message automatically cancels the previous in-flight query — no ghost responses or stacked answers. |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Browser (index.html)                      │
│                                                                   │
│  ┌──────────────┐  ┌──────────────────────┐  ┌───────────────┐  │
│  │  Left Panel  │  │     Chat Panel       │  │  PDF Viewer   │  │
│  │              │  │                      │  │               │  │
│  │ AWS status   │  │ User query input     │  │ PyMuPDF pages │  │
│  │ Doc list     │  │ RAG answers          │  │ Highlighted   │  │
│  │ Upload zone  │  │ Source pills         │  │ passages      │  │
│  │ KB stats     │  │ Follow-up chips      │  │ Page nav bar  │  │
│  └──────────────┘  └──────────────────────┘  └───────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP / REST
┌────────────────────────────▼────────────────────────────────────┐
│                     Flask (app.py)                               │
│                                                                   │
│  /api/configure   /api/upload     /api/query                     │
│  /api/questions   /api/preview    /api/page/<n>                  │
│  /api/documents   /api/ingestion  /api/pdf/<file>                │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   Core Functions                          │   │
│  │  extract_text()          extract_highlight_phrases()      │   │
│  │  render_pdf_pages()      render_single_page()             │   │
│  │  _search_phrase()        extract_page_numbers()           │   │
│  │  generate_followups()    _generate_questions_async()      │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────┬────────────────────────────┬────────────────────────┘
           │                            │
┌──────────▼──────────┐     ┌──────────▼──────────────────────────┐
│  bedrock_engine.py  │     │         AWS Services                  │
│                     │     │                                       │
│  BedrockEngine      │────▶│  S3 Bucket                           │
│  .validate()        │     │  ├── knowledge-base/<filename>       │
│  .upload()          │     │  └── (source of truth for KB)        │
│  .query()           │     │                                       │
│  .list_docs()       │     │  Bedrock Knowledge Base              │
│  .delete()          │     │  ├── Vector store (OpenSearch)       │
│  .ingestion_status()│     │  └── Chunked embeddings              │
│  .stats()           │     │                                       │
└─────────────────────┘     │  Bedrock Runtime (Claude 3.5)        │
                            │  ├── retrieve_and_generate()         │
                            │  └── invoke_model() (questions/FUs)  │
                            └──────────────────────────────────────┘
```

---

## Project Structure

```
nexusiq/
│
├── app.py                  # Flask application — all routes + core logic
├── bedrock_engine.py       # AWS Bedrock/S3 abstraction layer
├── templates/
│   └── index.html          # Single-page frontend (vanilla JS, no framework)
├── uploads/                # Temp dir for file transit (auto-created, auto-cleaned)
├── previews/               # Local PDF cache for preview panel (auto-created)
├── .env                    # Your secrets (never commit this)
├── env.example             # Safe template to commit
└── requirements.txt        # Python dependencies
```

---

## AWS Setup

You need four things in AWS before running NexusIQ:

### 1. S3 Bucket
Create a bucket (e.g. `nexusiq-kb-docs`) in your preferred region. No special settings required — just block public access.

### 2. Bedrock Knowledge Base
1. Go to **Amazon Bedrock → Knowledge Bases → Create knowledge base**
2. Choose **Amazon S3** as the data source
3. Point it at your bucket prefix: `knowledge-base/`
4. Choose an embeddings model (e.g. `amazon.titan-embed-text-v2:0`)
5. Let Bedrock create an OpenSearch Serverless vector store
6. Note the **Knowledge Base ID** and **Data Source ID** — you'll need both

### 3. Model Access
In **Bedrock → Model Access**, enable:
- `anthropic.claude-3-5-sonnet-20241022-v2:0` (for RAG generation + question/followup gen)
- Your chosen embeddings model

### 4. IAM User
Create an IAM user (`nexusiq-admin`) with these permissions:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject","s3:PutObject","s3:DeleteObject","s3:ListBucket","s3:HeadBucket"],
      "Resource": ["arn:aws:s3:::YOUR-BUCKET","arn:aws:s3:::YOUR-BUCKET/*"]
    },
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:RetrieveAndGenerate",
        "bedrock:Retrieve",
        "bedrock-agent:StartIngestionJob",
        "bedrock-agent:GetIngestionJob",
        "bedrock-agent:ListIngestionJobs"
      ],
      "Resource": "*"
    }
  ]
}
```
Generate an **Access Key + Secret** for this user.

---

## Installation & Running

### Prerequisites
- Python 3.10+
- pip

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/nexusiq.git
cd nexusiq

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp env.example .env
# Edit .env with your real values (see Configuration section below)

# 5. Run
python app.py
# → NexusIQ running at http://localhost:8000
```

---

## Configuration

Copy `env.example` to `.env` and fill in your values:

```env
# Flask
FLASK_SECRET_KEY=change-this-to-a-random-secret
FLASK_ENV=development          # set to 'production' to disable debug mode
PORT=8000

# AWS Credentials
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
AWS_SESSION_TOKEN=                        # optional, for temporary credentials

# S3
S3_BUCKET_NAME=nexusiq-kb-docs

# Bedrock Knowledge Base
BEDROCK_KB_ID=4UVNU3EXAM
BEDROCK_DS_ID=YXCOS7EXAM

# Model (optional — defaults to claude-3-5-sonnet-20241022-v2:0)
BEDROCK_MODEL_ARN=arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-5-sonnet-20241022-v2:0

# Upload
MAX_UPLOAD_MB=50
```

> **Alternatively**, you can enter all credentials through the in-app AWS Configuration modal (⚙ button in the header). This stores them in the browser session only — nothing is persisted to disk.

---

## API Reference

All endpoints return JSON. All POST bodies are `application/json` unless otherwise noted.

### `GET /health`
Health check. Returns `{"status": "ok"}`.

---

### `POST /api/configure`
Connect to AWS. Stores config in the Flask session.

**Body:**
```json
{
  "key":    "AWS_ACCESS_KEY_ID",
  "secret": "AWS_SECRET_ACCESS_KEY",
  "token":  "",
  "region": "us-east-1",
  "bucket": "nexusiq-kb-docs",
  "kb":     "4UVNU3EXAM",
  "ds":     "YXCOS7EXAM",
  "model":  "arn:aws:bedrock:..."
}
```

**Response:**
```json
{
  "ok": true,
  "total_docs": 3,
  "total_size_kb": 1024.5,
  "kb_id": "4UVNU3EXAM",
  "s3_bucket": "nexusiq-kb-docs",
  "region": "us-east-1"
}
```

---

### `GET /api/status`
Validate the current AWS connection and return KB stats. Same response shape as `/api/configure`.

---

### `POST /api/upload`
Upload a document. Saves locally → extracts text → copies to `previews/` (PDFs) → uploads to S3 → starts Bedrock ingestion sync → fires async question generation. Returns immediately without waiting for question gen.

**Body:** `multipart/form-data` with field `file`.

**Response:**
```json
{
  "success": true,
  "questions": [],
  "has_preview": true,
  "active_file": "report.pdf",
  "s3_key": "knowledge-base/report.pdf",
  "sync_job_id": "abc123",
  "filename": "report.pdf"
}
```

---

### `GET /api/questions/<filename>`
Poll for AI-generated questions for a specific document. Returns `ready: true` when complete, `ready: false` while generating (fires background thread if not already running).

**Response (generating):**
```json
{"questions": [], "ready": false}
```

**Response (ready):**
```json
{
  "questions": [
    "What is the maximum casual leave allowed annually?",
    "How is earned leave calculated?",
    ...
  ],
  "ready": true
}
```

---

### `POST /api/query`
Run a RAG query against the knowledge base.

**Body:**
```json
{
  "question":    "What are the types of leave available?",
  "mode":        "precise",
  "active_file": "HR_Policy.pdf"
}
```
`mode` can be `precise`, `detailed`, or `summary`.
`active_file` scopes the Bedrock vector search to that document's S3 URI (falls back to unfiltered if 0 chunks are found).

**Response:**
```json
{
  "answer": "According to the HR Policy Manual...",
  "sources": [
    {"filename": "HR_Policy.pdf", "uri": "s3://...", "relevance": 87.5}
  ],
  "chunks_used": 5,
  "chunk_texts": ["...raw retrieved chunk text..."],
  "followups": ["Can unused leave be carried forward?", ...],
  "highlight_phrases": ["types of leave available", ...],
  "answer_pages": [12, 15],
  "preview_file": "HR_Policy.pdf"
}
```

---

### `GET /api/preview/<filename>?highlights=phrase1||phrase2`
Render relevant PDF pages as base64 PNG images, with yellow highlights on matched passages. Uses ubiquity filter to avoid highlighting headers/footers.

**Query param:** `highlights` — `||`-delimited list of phrases to search for.

**Response:**
```json
{
  "pages": [
    {
      "page": 12,
      "total": 208,
      "data": "<base64 PNG>",
      "has_highlight": true,
      "highlight_y_ratio": 0.42
    }
  ],
  "total": 208,
  "filename": "HR_Policy.pdf",
  "first_hl_page": 12,
  "first_hl_y_ratio": 0.42,
  "highlighted_page_nums": [12, 15],
  "can_embed": true
}
```

---

### `GET /api/page/<filename>/<page_num>`
Render a single PDF page on demand (used by page navigation). Returns base64 PNG.

**Response:**
```json
{"page": 5, "total": 208, "data": "<base64 PNG>"}
```

---

### `GET /api/documents`
List all documents currently in the S3 knowledge base.

**Response:**
```json
{
  "documents": [
    {
      "s3_key": "knowledge-base/report.pdf",
      "filename": "report.pdf",
      "size_kb": 2163.3,
      "last_modified": "Apr 09, 2026"
    }
  ]
}
```

---

### `POST /api/documents/delete`
Delete a document from S3 and re-sync the KB. Also clears local PDF cache and session question cache.

**Body:** `{"s3_key": "knowledge-base/report.pdf"}`

**Response:** `{"success": true}`

---

### `GET /api/ingestion/<job_id>`
Poll the status of a Bedrock ingestion (sync) job.

**Response:**
```json
{
  "status": "COMPLETE",
  "indexed": 1,
  "failed": 0,
  "scanned": 1
}
```
`status` can be `IN_PROGRESS`, `COMPLETE`, `FAILED`, or `ERROR`.

---

### `GET /api/pdf/<filename>`
Serve the raw PDF binary for direct browser embedding (iframe fallback when PyMuPDF is unavailable).

---

### `GET /api/debug/pdfs`
List all PDFs currently cached in the local `previews/` directory with their sizes. Useful for debugging preview issues.

---

## Supported File Types

| Extension | Notes |
|---|---|
| `.pdf` | Full preview panel support. Text extracted via PyMuPDF (fitz) with pypdf fallback. |
| `.docx` / `.doc` | Text extracted via python-docx. No preview panel. |
| `.txt` | Read directly (up to 50,000 chars). No preview panel. |
| `.md` | Read directly. No preview panel. |
| `.html` | Read directly. No preview panel. |
| `.csv` | Read directly. No preview panel. |

Maximum upload size: **50 MB** (configurable via `MAX_UPLOAD_MB`).

---

## How RAG Works in NexusIQ

```
User Question
     │
     ▼
BedrockEngine.query()
     │
     ├── Build vector search config
     │     └── Optional: Bedrock metadata filter by S3 URI
     │           (restricts retrieval to one document's chunks)
     │
     ├── rt.retrieve_and_generate()
     │     ├── Bedrock retrieves top-8 most relevant chunks
     │     │     from the KB vector store (OpenSearch Serverless)
     │     └── Claude 3.5 Sonnet generates answer
     │           using retrieved chunks as context
     │
     ├── If 0 chunks returned with filter:
     │     └── Retry without filter (graceful fallback)
     │
     └── Return: answer + citations + chunk_texts
          │
          ▼
     app.py query()
          ├── generate_followups()   — 3 follow-up questions via Claude
          ├── extract_highlight_phrases()  — phrases to find in PDF
          ├── extract_page_numbers() — page refs mentioned in answer
          └── Resolve preview_file  — which PDF to open in panel
```

---

## PDF Highlighting Pipeline

The highlight system is one of the most complex parts of NexusIQ. Here's exactly how it works:

```
answer + chunk_texts
        │
        ▼
extract_highlight_phrases()     ← 6-tier phrase extraction
  Tier 1: Verbatim chunk sentences (30-100 chars)
  Tier 2: **Bold markdown** headings from answer
  Tier 3: "Quoted strings" in answer
  Tier 4: Number + 3+ meaningful context words
  Tier 5: Title-Case 3+-word phrases (12-65 chars)
  Tier 6: Full answer sentences (30-115 chars)
        │
        ▼
render_pdf_pages(pdf_path, phrases)
        │
        ├── Step 0: Pre-scan all page texts into memory
        │
        ├── Step 0b: UBIQUITY FILTER
        │     For each phrase, count how many pages it appears on.
        │     If > 25% of pages → it's a header/footer → discard it.
        │     This is the fix for "IIMA HR Policy Manual" appearing
        │     on every page being wrongly highlighted.
        │
        ├── Step 1: Build highlight map
        │     For each (filtered phrase, page) combination:
        │       - Quick probe: first 24 chars in page text?
        │       - _search_phrase(): full → truncated → 4-word windows
        │       - Footer heuristic: y_ratio > 0.90 + len < 35 → skip
        │     Result: {page_index: {y_ratio, phrase, rects}}
        │
        ├── Step 2: Select pages to render
        │     Highlighted pages ± 1 buffer page each side.
        │     No highlights → first 3 pages only.
        │
        └── Step 3: Render selected pages
              Highlighted pages at 1.5× scale.
              Apply fitz.add_highlight_annot(rect) in yellow.
              Encode as base64 PNG.
              Return with first_hl_page + first_hl_y_ratio
              so frontend can scroll to exact Y position.
```

---

## Frontend Architecture

The entire frontend is a single `index.html` file (~1,200 lines) — no framework, no build step, pure vanilla JS.

### State Management
```javascript
const S = {
  connected: false,          // AWS connection status
  loading: false,            // query in flight
  docs: [],                  // current S3 doc list
  jTimer: null,              // ingestion poll interval ID
  activeFile: null,          // currently selected document
  conversationStarted: false // hides question chips after first message
};
```

### Key JS Functions

| Function | Purpose |
|---|---|
| `sendMsg()` | Send query, AbortController cancels in-flight requests |
| `openPanel(filename, highlights, answerPages)` | Load PDF into right panel, scroll to highlight |
| `fetchAndShowQuestions(filename, retries)` | Poll `/api/questions` every 2s until ready |
| `_pollingFor` | Global guard — prevents stale polls from overwriting newer doc's chips |
| `pollJob(id, filename)` | Poll Bedrock ingestion status every 3s |
| `render_pdf_pages()` | Build HTML for PDF page cards from base64 data |
| `navToPage(pageNum)` | Fetch single page on demand if not rendered |
| `_scrollToHighlight(body, pageNum, yRatio)` | Smooth-scroll to exact Y position within a page image |
| `renderMD(s)` | Minimal Markdown renderer (headers, bold, italic, lists, code) |

### CSS Design System

The UI uses CSS custom properties for the entire design system:

```css
/* Palette */
--indigo: #6366f1        /* Primary accent */
--cyan: #22d3ee          /* Secondary accent */
--em: #10b981            /* Success/connected green */
--rose: #f43f5e          /* Error/delete red */
--am: #f59e0b            /* Warning/highlight amber */
--vi: #a78bfa            /* Violet for gradients */

/* Surfaces */
--ink: #03040a           /* Deepest background */
--glass: rgba(14,18,32,.78)   /* Glassmorphism panels */

/* Typography */
--syne: 'Syne'           /* Headers, brand */
--mono: 'JetBrains Mono' /* Labels, badges, code */
--fig: 'Figtree'         /* Body text, chat */
```

---

## Deployment

### EC2 (Recommended for production)

```bash
# On your EC2 instance (Amazon Linux 2 / Ubuntu)
git clone https://github.com/YOUR_USERNAME/nexusiq.git
cd nexusiq
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp env.example .env && nano .env   # fill in your values

# Run with Gunicorn (production WSGI server)
gunicorn -w 4 -b 0.0.0.0:8000 --timeout 120 app:app
```

For EC2, ensure your instance's security group allows inbound TCP on port 8000 (or 80 if behind a load balancer/nginx).

### Environment Variables (EC2 / Docker)
Prefer environment variables over `.env` file in production:
```bash
export AWS_REGION=us-east-1
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
# etc.
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

### IAM Roles (Best Practice)
If running on EC2, attach an IAM role with the required permissions to the instance instead of using access keys. Remove `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` from the environment — boto3 will use the instance metadata service automatically.

---

## Security Notes

- **Never commit `.env`** — it's in `.gitignore` for a reason
- **Rotate keys** if accidentally exposed — generate a new IAM access key immediately
- `FLASK_SECRET_KEY` should be a long random string in production (`python3 -c "import secrets; print(secrets.token_hex(32))"`)
- The in-app credential modal stores values only in the Flask session cookie — nothing is written to disk
- All file uploads use `werkzeug.utils.secure_filename` to prevent path traversal
- Temporary upload files are deleted in a `finally` block after each upload

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3.12, Flask 3.0, Flask-CORS |
| **AI / RAG** | Amazon Bedrock (Claude 3.5 Sonnet v2), Bedrock Knowledge Bases |
| **Vector Store** | Amazon OpenSearch Serverless (managed by Bedrock) |
| **Storage** | Amazon S3 |
| **PDF Processing** | PyMuPDF (fitz) with pypdf fallback |
| **Frontend** | Vanilla JS, CSS3, no framework, no build step |
| **Fonts** | Syne, JetBrains Mono, Figtree (Google Fonts) |
| **WSGI** | Gunicorn (production) |
| **Session** | Flask server-side sessions |

---

## License

Apache License 2.0 — see `LICENSE` for details.

---

<div align="center">

Built with Amazon Bedrock, Claude 3.5 Sonnet, and Flask.


</div>
