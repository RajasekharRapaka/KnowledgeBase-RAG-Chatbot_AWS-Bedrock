import os, uuid, json, base64, re, sys, threading
from pathlib import Path
from flask import Flask, request, jsonify, render_template, session, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()

_HERE = Path(__file__).parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "nexusiq-change-this-in-production")
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
CORS(app)

UPLOAD_DIR  = Path(__file__).parent / "uploads"
PREVIEW_DIR = Path(__file__).parent / "previews"
UPLOAD_DIR.mkdir(exist_ok=True)
PREVIEW_DIR.mkdir(exist_ok=True)

ALLOWED = {".pdf", ".txt", ".docx", ".doc", ".md", ".html", ".csv"}
MAX_MB  = int(os.getenv("MAX_UPLOAD_MB", 50))
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024

# Per-session state — never shared across users
_doc_texts: dict = {}   # sid -> {filename: text}
_engines:   dict = {}
# Question cache is now per-session: sid -> {filename: [questions]}
_question_cache: dict = {}   # sid -> {filename: [questions]}
_question_running: set = set()  # (sid, filename) currently being generated


# ── SESSION ────────────────────────────────────────────────────────────────────
def _sid() -> str:
    if "sid" not in session:
        session["sid"] = uuid.uuid4().hex
    return session["sid"]

def _session_texts() -> dict:
    sid = _sid()
    if sid not in _doc_texts:
        _doc_texts[sid] = {}
    return _doc_texts[sid]

def _session_qcache() -> dict:
    """Per-session question cache: {filename: [questions]}"""
    sid = _sid()
    if sid not in _question_cache:
        _question_cache[sid] = {}
    return _question_cache[sid]


# ── PDF STORE ──────────────────────────────────────────────────────────────────
def _pdf_path(filename: str) -> Path:
    return PREVIEW_DIR / secure_filename(filename)

def _pdf_exists(filename: str) -> bool:
    return _pdf_path(filename).exists()

def _ensure_pdf_local(filename: str) -> bool:
    """
    Guarantee the PDF is in PREVIEW_DIR.
    If already there → instant return True.
    If not → download from S3 (knowledge-base/<filename>).
    This means the panel works even after server restarts or cross-session uploads.
    """
    if _pdf_exists(filename):
        return True
    try:
        engine = get_engine()
        s3_key = f"knowledge-base/{filename}"
        dest   = _pdf_path(filename)
        print(f"[ensure_pdf] Downloading s3://{engine.bucket}/{s3_key} → {dest}")
        engine.s3.download_file(engine.bucket, s3_key, str(dest))
        return dest.exists()
    except Exception as e:
        print(f"[ensure_pdf] Failed to download '{filename}': {e}")
        return False


# ── ENGINE ─────────────────────────────────────────────────────────────────────
def _make_engine():
    cfg = session.get("cfg", {})
    try:
        from bedrock_engine import BedrockEngine
    except ImportError:
        from utils.bedrock_engine import BedrockEngine
    return BedrockEngine(
        region    = cfg.get("region")  or os.getenv("AWS_REGION", "us-east-1"),
        key_id    = cfg.get("key")     or os.getenv("AWS_ACCESS_KEY_ID", ""),
        secret    = cfg.get("secret")  or os.getenv("AWS_SECRET_ACCESS_KEY", ""),
        token     = cfg.get("token")   or os.getenv("AWS_SESSION_TOKEN", ""),
        bucket    = cfg.get("bucket")  or os.getenv("S3_BUCKET_NAME", ""),
        kb_id     = cfg.get("kb")      or os.getenv("BEDROCK_KB_ID", ""),
        ds_id     = cfg.get("ds")      or os.getenv("BEDROCK_DS_ID", ""),
        model_arn = cfg.get("model")   or os.getenv("BEDROCK_MODEL_ARN", ""),
    )

def get_engine():
    key = session.get("engine_key", "env")
    if key not in _engines:
        _engines[key] = _make_engine()
    return _engines[key]


# ── BEDROCK INVOKE ─────────────────────────────────────────────────────────────
def _rt_client():
    import boto3
    cfg    = session.get("cfg", {})
    region = cfg.get("region") or os.getenv("AWS_REGION", "us-east-1")
    key_id = cfg.get("key")    or os.getenv("AWS_ACCESS_KEY_ID", "")
    secret = cfg.get("secret") or os.getenv("AWS_SECRET_ACCESS_KEY", "")
    kw = {"region_name": region}
    if key_id and secret:
        kw["aws_access_key_id"]     = key_id
        kw["aws_secret_access_key"] = secret
    return boto3.Session(**kw).client("bedrock-runtime")

def _model_id() -> str:
    cfg = session.get("cfg", {})
    arn = cfg.get("model") or os.getenv("BEDROCK_MODEL_ARN", "")
    return arn if arn else "anthropic.claude-3-5-sonnet-20241022-v2:0"

def _invoke(prompt: str, max_tokens: int = 400, temperature: float = 0.3) -> str:
    mid  = _model_id()
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}]
    })
    resp = _rt_client().invoke_model(modelId=mid, body=body)
    return json.loads(resp["body"].read())["content"][0]["text"].strip()

def _parse_list(text: str) -> list:
    s = text.find("["); e = text.rfind("]") + 1
    if s == -1 or e <= s:
        return []
    try:
        r = json.loads(text[s:e])
        return r if isinstance(r, list) else []
    except Exception:
        return []


# ── TEXT EXTRACTION ────────────────────────────────────────────────────────────
def extract_text(filepath: str, filename: str) -> str:
    ext = Path(filename).suffix.lower()
    try:
        if ext == ".pdf":
            # Try PyMuPDF (fitz) first — best quality
            try:
                import fitz
                doc = fitz.open(filepath)
                pages = list(doc)[:15]
                text = "\n".join(p.get_text() for p in pages)
                doc.close()
                if text.strip():
                    return text
            except ImportError:
                print("[extract_text] PyMuPDF not installed, falling back to pypdf")
            except Exception as e:
                print(f"[extract_text] fitz error: {e}")
            # Fallback: pypdf (pure Python, no binary deps)
            try:
                from pypdf import PdfReader
                reader = PdfReader(filepath)
                texts = []
                for page in reader.pages[:15]:
                    t = page.extract_text()
                    if t:
                        texts.append(t)
                text = "\n".join(texts)
                if text.strip():
                    return text
            except ImportError:
                print("[extract_text] pypdf not installed either")
            except Exception as e:
                print(f"[extract_text] pypdf error: {e}")
            return ""
        elif ext in (".docx", ".doc"):
            from docx import Document
            return "\n".join(p.text for p in Document(filepath).paragraphs)
        else:
            with open(filepath, "r", errors="ignore") as f:
                return f.read(50000)
    except Exception as ex:
        print(f"[extract_text] {ex}")
        return ""


# ── AI FEATURES ────────────────────────────────────────────────────────────────
def generate_questions(text: str, filename: str) -> list:
    """Generate 6 document-specific questions. Session-scoped cache."""
    if not text.strip():
        return []
    qcache = _session_qcache()
    if filename in qcache:
        return qcache[filename]
    prompt = (
        f'Read this document excerpt from "{filename}" and write exactly 6 specific questions '
        f'a reader would naturally ask. Each question must reference actual content '
        f'(names, numbers, concepts, dates) from the text. Max 12 words each. '
        f'Return ONLY a valid JSON array of 6 strings, nothing else.\n\n'
        f'Document:\n{text[:3000]}\n\nJSON:'
    )
    try:
        out = _invoke(prompt, max_tokens=300, temperature=0.2)
        qs  = _parse_list(out)
        if len(qs) >= 3:
            result = qs[:6]
            qcache[filename] = result
            return result
    except Exception as e:
        print(f"[gen_q] {e}")
    return []

def _generate_questions_async(text: str, filename: str, cfg_snapshot: dict, sid: str) -> None:
    """
    Background thread: generate questions for (sid, filename) and store in
    per-session cache. Guards against duplicate concurrent calls via
    _question_running set.
    """
    key = (sid, filename)
    if key in _question_running:
        return
    _question_running.add(key)
    try:
        # Check session cache first (another thread may have finished)
        sess_cache = _question_cache.setdefault(sid, {})
        if filename in sess_cache:
            return

        import boto3
        region = cfg_snapshot.get("region") or os.getenv("AWS_REGION", "us-east-1")
        key_id = cfg_snapshot.get("key")    or os.getenv("AWS_ACCESS_KEY_ID", "")
        secret = cfg_snapshot.get("secret") or os.getenv("AWS_SECRET_ACCESS_KEY", "")
        arn    = cfg_snapshot.get("model")  or os.getenv("BEDROCK_MODEL_ARN", "") \
                 or "anthropic.claude-3-5-sonnet-20241022-v2:0"
        kw = {"region_name": region}
        if key_id and secret:
            kw["aws_access_key_id"]     = key_id
            kw["aws_secret_access_key"] = secret
        client = boto3.Session(**kw).client("bedrock-runtime")
        prompt = (
            f'Read this document excerpt from "{filename}" and write exactly 6 specific questions '
            f'a reader would naturally ask. Each question must reference actual content '
            f'(names, numbers, concepts, dates) from the text. Max 12 words each. '
            f'Return ONLY a valid JSON array of 6 strings, nothing else.\n\n'
            f'Document:\n{text[:3000]}\n\nJSON:'
        )
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 300, "temperature": 0.2,
            "messages": [{"role": "user", "content": prompt}]
        })
        resp = client.invoke_model(modelId=arn, body=body)
        out  = json.loads(resp["body"].read())["content"][0]["text"].strip()
        s = out.find("["); e = out.rfind("]") + 1
        if s != -1 and e > s:
            qs = json.loads(out[s:e])
            if isinstance(qs, list) and len(qs) >= 3:
                sess_cache[filename] = qs[:6]
                print(f"[gen_q_async] sid={sid[:8]} cached {len(qs)} Qs for {filename}")
    except Exception as ex:
        print(f"[gen_q_async] {ex}")
    finally:
        _question_running.discard(key)

def generate_followups(question: str, answer: str) -> list:
    prompt = (
        f"Given this Q&A, generate exactly 3 natural follow-up questions. "
        f"Max 10 words each. Return ONLY a valid JSON array of 3 strings.\n\n"
        f"Q: {question}\nA: {answer[:400]}\n\nJSON:"
    )
    try:
        out = _invoke(prompt, max_tokens=150, temperature=0.4)
        qs  = _parse_list(out)
        return qs[:3] if qs else []
    except Exception as e:
        print(f"[gen_fu] {e}")
    return []

def extract_highlight_phrases(answer: str, chunk_texts: list = None) -> list:
    """
    Build a list of phrases to search for inside the PDF.
    Strategy is ordered from most to least verbatim-reliable.

      Tier 1 — chunk_texts           Verbatim doc text from Bedrock (best)
      Tier 2 — Bold / heading text   **Earned Leave**, **Half Pay Leave**
                                     These are exact section titles → always match
      Tier 3 — Quoted strings        Text the AI explicitly quoted
      Tier 4 — Number + 3-5 words    "2.5 days per completed calendar month"
                                     Short numbers alone (2.5, 10) are BANNED —
                                     they fire on every contents/index page.
      Tier 5 — Title-Case phrases    "Earned Leave", "Leave Type 3", "WMT 2014"
      Tier 6 — Full answer sentences Last resort; _search_phrase handles fuzzy match
    """
    seen: set   = set()
    result: list = []

    def _add(phrase: str) -> bool:
        """Normalise, deduplicate and append. Returns True while we want more."""
        phrase = re.sub(r'\s+', ' ', phrase).strip().rstrip(':')
        # Strip trailing source citations that crept into the phrase
        phrase = re.sub(r'\s*\(Sources?\s*[\d,\s]+\)\s*$', '', phrase,
                        flags=re.IGNORECASE).strip()
        key = phrase[:45].lower()
        if key in seen or len(phrase) < 8:
            return True
        seen.add(key)
        result.append(phrase)
        return len(result) < 12

    # ── Tier 1: verbatim retrieved passages (ideal, often empty) ──────────────
    # NOTE: We do NOT early-return here anymore. Chunk texts can also contain
    # footer/header text that ubiquity-filter in render_pdf will handle, but we
    # want higher-quality phrases from later tiers to supplement.
    if chunk_texts:
        for chunk in chunk_texts[:4]:
            for sent in re.split(r'[.!?\n]', chunk):
                sent = sent.strip()
                # Min 30 chars: ensures distinctiveness; max 100: keeps search fast
                if 30 <= len(sent) <= 100:
                    if not _add(sent):
                        return result

    # ── Tier 2: bold / heading text from markdown ─────────────────────────────
    # **Earned Leave** → "Earned Leave"   **Half Pay Leave (Leave Type 3):** → …
    # These are section titles — PyMuPDF finds them exactly in the document body.
    for m in re.findall(r'\*\*([^*]{3,60})\*\*', answer):
        m = m.strip().rstrip(':').strip()
        if m:
            _add(m)

    # ── Tier 3: explicitly quoted strings ─────────────────────────────────────
    for q in re.findall(r'"([^"]{8,90})"', answer):
        _add(q)

    # ── Strip markdown for remaining tiers ────────────────────────────────────
    clean = re.sub(r'\(Sources?\s*[\d,\s]+\)', '', answer, flags=re.IGNORECASE)
    clean = re.sub(r'\*+|#+|`+|\[.*?\]', '', clean)

    # ── Tier 4: number + ≥3 meaningful context words (min 15 chars total) ─────
    # BANNED: bare numbers ("2.5", "10", "2023") — they match every page of a
    # table-of-contents or index.  We require 3+ words of context so the phrase
    # is distinctive enough to land on the RIGHT page.
    # Also BANNED: number immediately followed by stop-word prepositions like
    # "5 of the IIMA..." (chapter ref, not a measurement).
    _STOPS = {'of','in','on','the','by','a','an','to','is','at','as','and','or',
              'for','from','with','that','this','which','be','are','was','were'}
    for m in re.finditer(
        r'\b\d+\.?\d*(?:\s+[A-Za-z][A-Za-z0-9\-]{1,25}){3,6}',
        clean
    ):
        tok = m.group(0).strip()
        if len(tok) < 15:
            continue
        # First two words after the number must include at least one non-stop-word
        following = tok.split()[1:]
        if all(w.lower() in _STOPS for w in following[:2]):
            continue
        _add(tok)

    # ── Tier 5: Title-Case 3+-word phrases (section headings, named concepts) ─
    # Raised from 2+ to 3+ words and 12+ chars to avoid matching generic 2-word
    # headers like "HR Department" or "Leave Policy" that appear as footers.
    for m in re.finditer(
        r'[A-Z][a-zA-Z]{2,}(?:\s+(?:[A-Z][a-zA-Z]*|\d+)){2,5}',
        clean
    ):
        tok = m.group(0).strip()
        word_count = len(tok.split())
        if 12 <= len(tok) <= 65 and word_count >= 3:
            _add(tok)

    # ── Tier 6: full answer sentences (least reliable) ────────────────────────
    # Split on sentence-ending punctuation that is NOT between digits
    # (avoids splitting "28.4" → "28" + "4 BLEU...").
    for sent in re.split(r'(?<!\d)[.!?](?!\d)\s*|\n', clean):
        sent = sent.strip().lstrip('-•* ')
        # Sweet-spot: long enough to be specific, short enough for search_for
        if 30 < len(sent) < 115:
            if not _add(sent):
                return result

    return result


def extract_page_numbers(answer: str) -> list:
    """
    Extract page-number references from the AI answer.
    Handles patterns like: page 5, pages 3-7, p.12, (p. 4), Page 2 of …
    Returns a sorted, de-duplicated list of ints (capped at 500 to avoid noise).
    """
    pages: set = set()
    patterns = [
        r'\bpages?\s+(\d+)\s*[-–to]+\s*(\d+)',   # pages 3-7 / pages 3 to 7
        r'\bpages?\s+(\d+)',                       # page 5 / pages 12
        r'\bp\.\s*(\d+)',                          # p.12 / p. 4
        r'\(p\.?\s*(\d+)\)',                       # (p.4) / (p 4)
    ]
    for pat in patterns:
        for m in re.finditer(pat, answer, re.IGNORECASE):
            n = int(m.group(1))
            if 1 <= n <= 500:
                pages.add(n)
            if m.lastindex and m.lastindex >= 2:
                try:
                    n2 = int(m.group(2))
                    if 1 <= n2 <= 500:
                        pages.add(n2)
                except (IndexError, TypeError):
                    pass
    return sorted(pages)


def _search_phrase(pg, phrase: str) -> list:
    """
    Search for 'phrase' on PyMuPDF page 'pg' using progressively shorter
    strategies until something matches or we give up.

    Order:
      1. Full phrase (case-insensitive)
      2. First 50 characters, trimmed to last whole word
      3. Sliding 4-word windows (higher bar than before to avoid stop-word runs)
    """
    def _try(text):
        if len(text) < 6:
            return []
        try:
            r = pg.search_for(text, flags=fitz.TEXT_IGNORE_CASE)
        except Exception:
            try:
                r = pg.search_for(text)
            except Exception:
                r = []
        return r or []

    # Strategy 1 — full phrase
    rects = _try(phrase)
    if rects:
        return rects

    # Strategy 2 — first 50 chars (last whole word boundary)
    if len(phrase) > 40:
        snippet = phrase[:52].rsplit(' ', 1)[0]
        if len(snippet) >= 10:
            rects = _try(snippet)
            if rects:
                return rects

    # Strategy 3 — sliding 4-word windows (raised from 3 to reduce false positives)
    # Only windows containing a digit or 3+ cap words to stay specific.
    words = phrase.split()
    if len(words) >= 6:
        for i in range(len(words) - 3):
            window = ' '.join(words[i:i + 4])
            if len(window) >= 16:
                caps = sum(1 for w in words[i:i + 4] if w and w[0].isupper())
                has_digit = any(c.isdigit() for c in window)
                if has_digit or caps >= 3:
                    rects = _try(window)
                    if rects:
                        return rects

    return []


# ── PDF RENDERER ───────────────────────────────────────────────────────────────
def render_pdf_pages(pdf_path: str, highlight_texts: list = None) -> dict:
    """
    Smart PDF renderer — only renders the pages that matter:
      - If highlights: highlighted pages ± 1 buffer page each side
      - If no highlights: first 3 pages only (cover + intro)
    Never renders all pages — prevents timeout on large docs.
    Returns a dict with pages list + metadata.

    KEY FIX: Ubiquity filter — phrases matching >25% of pages are headers/footers
    and are discarded. Also rejects bottom-of-page short matches (footers).
    """
    try:
        import fitz
    except ImportError:
        print("[render_pdf] PyMuPDF not installed")
        return {"pages": [], "total": 0, "highlighted_page_nums": []}
    try:
        doc   = fitz.open(pdf_path)
        total = len(doc)

        # ── Step 0: Pre-scan page texts (cache for ubiquity check + search) ───
        page_texts = [doc[i].get_text() for i in range(total)]

        # ── Step 0b: Ubiquity filter ──────────────────────────────────────────
        # Phrases that appear on more than 25% of pages (min 3) are structural
        # elements like headers, footers, page numbers — skip them entirely.
        filtered_texts: list = []
        if highlight_texts:
            ubiquity_threshold = max(3, min(int(total * 0.25), 12))
            for phrase in highlight_texts:
                if not phrase or len(phrase) < 6:
                    continue
                phrase_norm = re.sub(r"\s+", " ", phrase).strip()
                probe = phrase_norm[:24].lower()
                match_count = sum(1 for pt in page_texts if probe in pt.lower())
                if match_count <= ubiquity_threshold:
                    filtered_texts.append(phrase_norm)
                else:
                    print(f"[render_pdf] ubiquity-drop ({match_count}/{total} pages): {phrase_norm[:55]}")
            if not filtered_texts and highlight_texts:
                # All phrases ubiquitous (edge case) — fall back to longest 2
                filtered_texts = sorted(
                    [re.sub(r"\s+", " ", p).strip() for p in highlight_texts if p and len(p) >= 6],
                    key=len, reverse=True
                )[:2]
        else:
            filtered_texts = []

        # ── Step 1: Find which pages have highlights ──────────────────────────
        # page_index → {y_ratio, phrase, rects}
        hl_map: dict = {}

        if filtered_texts:
            for i in range(total):
                pg        = doc[i]
                pg_text_l = page_texts[i].lower()
                pg_height = pg.rect.height or 1
                for phrase_norm in filtered_texts:
                    # Quick pre-screen
                    probe = phrase_norm[:24].lower()
                    if probe and probe not in pg_text_l:
                        continue
                    rects = _search_phrase(pg, phrase_norm)
                    if rects:
                        y_ratio = round(rects[0].y0 / pg_height, 4)
                        # Footer heuristic: very bottom of page + short phrase = footer
                        if y_ratio > 0.90 and len(phrase_norm) < 35:
                            print(f"[render_pdf] footer-skip p{i+1} y={y_ratio:.2f}: {phrase_norm[:40]}")
                            continue
                        hl_map[i] = {
                            "y_ratio": y_ratio,
                            "phrase":  phrase_norm,
                            "rects":   rects,
                        }
                        break   # one phrase per page is enough to mark it

        # ── Step 2: Decide which pages to render ─────────────────────────────
        pages_to_render: set = set()

        if hl_map:
            for pi in hl_map:
                # ±1 buffer so the reader sees context above and below
                for off in (-1, 0, 1):
                    idx = pi + off
                    if 0 <= idx < total:
                        pages_to_render.add(idx)
        else:
            # No highlights — show first 3 pages as document preview
            for i in range(min(3, total)):
                pages_to_render.add(i)

        # ── Step 3: Render only the selected pages ────────────────────────────
        rendered = []
        for i in sorted(pages_to_render):
            pg    = doc[i]
            is_hl = i in hl_map
            scale = 1.5 if is_hl else 1.2

            if is_hl:
                # Apply yellow highlight annotations using cached rects or re-search
                cached_rects = hl_map[i].get("rects")
                if cached_rects:
                    # Annotate the primary matched phrase
                    for rect in cached_rects:
                        try:
                            annot = pg.add_highlight_annot(rect)
                            annot.set_colors(stroke=[1, 0.85, 0.0])
                            annot.update()
                        except Exception:
                            pass
                # Also try every other filtered phrase on this page for completeness
                for phrase_norm in filtered_texts:
                    if not phrase_norm or phrase_norm == hl_map[i].get("phrase"):
                        continue   # already handled above
                    rects = _search_phrase(pg, phrase_norm)
                    for rect in rects:
                        try:
                            annot = pg.add_highlight_annot(rect)
                            annot.set_colors(stroke=[1, 0.85, 0.0])
                            annot.update()
                        except Exception:
                            pass

            pix = pg.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            rendered.append({
                "page":              i + 1,          # 1-based
                "total":             total,
                "data":              base64.b64encode(pix.tobytes("png")).decode(),
                "has_highlight":     is_hl,
                "highlight_y_ratio": hl_map[i]["y_ratio"] if is_hl else None,
            })

        doc.close()
        hl_page_nums = sorted(v + 1 for v in hl_map)
        print(f"[render_pdf] rendered {len(rendered)}/{total} pages, hl pages: {hl_page_nums}")
        return {
            "pages":                rendered,
            "total":                total,
            "highlighted_page_nums": hl_page_nums,
        }
    except Exception as e:
        print(f"[render_pdf] Error: {e}")
        return {"pages": [], "total": 0, "highlighted_page_nums": []}


def render_single_page(pdf_path: str, page_num: int, scale: float = 1.2) -> str | None:
    """Render a single page (1-based) and return base64 PNG, or None on error."""
    try:
        import fitz
        doc = fitz.open(pdf_path)
        if page_num < 1 or page_num > len(doc):
            doc.close()
            return None
        pix  = doc[page_num - 1].get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        data = base64.b64encode(pix.tobytes("png")).decode()
        doc.close()
        return data
    except Exception as e:
        print(f"[render_single] {e}")
        return None



# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/api/configure", methods=["POST"])
def configure():
    data = request.json or {}
    cfg  = {k: data.get(k, "").strip() for k in ("key","secret","token","region","bucket","kb","ds","model")}
    cfg["region"] = cfg["region"] or "us-east-1"
    session["cfg"] = cfg
    session["engine_key"] = cfg["key"][:8] + cfg["bucket"]
    _engines.clear()
    try:
        engine = get_engine()
        result = engine.validate()
        if not result["ok"]:
            return jsonify({"ok": False, "error": result["error"]})
        return jsonify({"ok": True, **engine.stats()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.route("/api/status")
def status():
    try:
        engine = get_engine()
        result = engine.validate()
        if not result["ok"]:
            return jsonify({"ok": False, "error": result["error"]})
        return jsonify({"ok": True, **engine.stats()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file attached"}), 400
    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400
    filename = secure_filename(file.filename)
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    tmp = UPLOAD_DIR / f"{uuid.uuid4().hex}_{filename}"
    try:
        file.save(str(tmp))

        # 1. Extract text (fast — first 10 pages only for speed)
        text = extract_text(str(tmp), filename)
        _session_texts()[filename] = text

        # 2. Track active file for this session
        session["active_file"] = filename
        session.modified = True

        # 3. Persist PDF immediately for preview
        if ext == ".pdf":
            import shutil
            dest = _pdf_path(filename)
            shutil.copy2(str(tmp), str(dest))
            print(f"[upload] PDF saved → {dest}")

        # 4. Upload to S3 + start KB sync (this is the slow part — unavoidable)
        engine = get_engine()
        result = engine.upload(str(tmp), filename)

        # 5. Fire question generation in background — DON'T BLOCK the response.
        #    Return instantly. Frontend polls /api/questions until ready.
        sid = _sid()
        qcache = _question_cache.setdefault(sid, {})
        questions = qcache.get(filename, [])

        if not questions and text:
            cfg_snap = dict(session.get("cfg", {}))
            t = threading.Thread(
                target=_generate_questions_async,
                args=(text, filename, cfg_snap, sid),
                daemon=True
            )
            t.start()
            # Don't join — return immediately so browser isn't blocked

        return jsonify({
            "success":     True,
            "questions":   questions,   # empty [] is fine — frontend will poll
            "has_preview": ext == ".pdf",
            "active_file": filename,
            **result
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if tmp.exists():
            tmp.unlink()

@app.route("/api/ingestion/<job_id>")
def ingestion(job_id):
    try:
        return jsonify(get_engine().ingestion_status(job_id))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/documents")
def documents():
    try:
        return jsonify({"documents": get_engine().list_docs()})
    except Exception as e:
        return jsonify({"error": str(e), "documents": []}), 500

@app.route("/api/documents/delete", methods=["POST"])
def delete_doc():
    s3_key = (request.json or {}).get("s3_key", "")
    if not s3_key:
        return jsonify({"error": "s3_key required"}), 400
    try:
        filename = s3_key.split("/")[-1]
        _session_texts().pop(filename, None)
        # Clean per-session question cache
        sid = _sid()
        _question_cache.get(sid, {}).pop(filename, None)
        _question_running.discard((sid, filename))
        if session.get("active_file") == filename:
            session.pop("active_file", None)
            session.modified = True
        pdf_p = _pdf_path(filename)
        if pdf_p.exists():
            pdf_p.unlink()
        ok = get_engine().delete(s3_key)
        return jsonify({"success": ok})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/query", methods=["POST"])
def query():
    data     = request.json or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Question is required"}), 400
    try:
        # active_file comes from the frontend (currently selected doc).
        # Passing it to the engine restricts vector search to that document's
        # chunks only, preventing stale/mixed results from old KB entries.
        active_file = (
            data.get("active_file", "")                # sent by frontend
            or session.get("active_file", "")          # session fallback
        ).strip()

        result   = get_engine().query(question, mode=data.get("mode", "precise"),
                                      source_filename=active_file)
        answer   = result.get("answer", "")
        result["followups"]         = generate_followups(question, answer)
        result["highlight_phrases"] = extract_highlight_phrases(answer, result.get("chunk_texts", []))
        result["answer_pages"]      = extract_page_numbers(answer)

        # Resolve preview_file — use _ensure_pdf_local so PDFs are auto-downloaded
        # from S3 if the server restarted and the local cache is empty.
        result["preview_file"] = None

        for src in result.get("sources", []):
            fname = src.get("filename", "")
            if fname and fname.lower().endswith(".pdf"):
                if _ensure_pdf_local(fname):
                    result["preview_file"] = fname
                    print(f"[query] preview ← source (ensured): {fname}")
                    break

        if not result["preview_file"]:
            active = session.get("active_file", "")
            if active and active.lower().endswith(".pdf") and _ensure_pdf_local(active):
                result["preview_file"] = active
                print(f"[query] preview ← session active_file: {active}")

        if not result["preview_file"]:
            # Last resort — check any PDF already on disk
            pdfs = sorted(PREVIEW_DIR.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
            if pdfs:
                result["preview_file"] = pdfs[0].name
                print(f"[query] preview ← disk: {result['preview_file']}")

        print(f"[query] preview_file={result['preview_file']}, hl_phrases={len(result['highlight_phrases'])}")
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/preview/<filename>")
def preview(filename):
    filename = secure_filename(filename)
    print(f"[preview] requested: {filename}")
    if not _ensure_pdf_local(filename):
        return jsonify({
            "error":     f"PDF '{filename}' not found.",
            "can_embed": False,
            "pages":     [],
        }), 404
    highlights = request.args.get("highlights", "")
    hl_list    = [h.strip() for h in highlights.split("||") if h.strip()] if highlights else []
    result     = render_pdf_pages(str(_pdf_path(filename)), hl_list)
    pages      = result["pages"]
    first_hl   = next((p for p in pages if p.get("has_highlight")), None)
    return jsonify({
        "pages":                 pages,
        "total":                 result["total"],
        "filename":              filename,
        "first_hl_page":         first_hl["page"]               if first_hl else None,
        "first_hl_y_ratio":      first_hl.get("highlight_y_ratio", 0) if first_hl else None,
        "highlighted_page_nums": result["highlighted_page_nums"],
        "can_embed":             True,
    })

@app.route("/api/page/<filename>/<int:page_num>")
def get_page(filename, page_num):
    """On-demand single page renderer — used by frontend page navigation."""
    filename = secure_filename(filename)
    if not _ensure_pdf_local(filename):
        return jsonify({"error": "PDF not found"}), 404
    data = render_single_page(str(_pdf_path(filename)), page_num)
    if data is None:
        return jsonify({"error": f"Page {page_num} not available"}), 404
    # Get total page count
    try:
        import fitz
        doc   = fitz.open(str(_pdf_path(filename)))
        total = len(doc)
        doc.close()
    except Exception:
        total = page_num
    return jsonify({"page": page_num, "total": total, "data": data})

@app.route("/api/questions/<filename>")
def get_questions(filename):
    filename = secure_filename(filename)
    sid = _sid()
    sess_cache = _question_cache.setdefault(sid, {})

    # Instant return if already cached for this session
    if filename in sess_cache:
        return jsonify({"questions": sess_cache[filename], "ready": True})

    # Ensure we have text to work with (extract if needed)
    text = _session_texts().get(filename, "")
    if not text:
        if filename.lower().endswith(".pdf") and _ensure_pdf_local(filename):
            text = extract_text(str(_pdf_path(filename)), filename)
            if text:
                _session_texts()[filename] = text

    if not text:
        return jsonify({"questions": [], "ready": False})

    # Fire background generation if not already running for this (sid, filename)
    key = (sid, filename)
    if key not in _question_running:
        cfg_snap = dict(session.get("cfg", {}))
        t = threading.Thread(
            target=_generate_questions_async,
            args=(text, filename, cfg_snap, sid),
            daemon=True,
        )
        t.start()

    # Return immediately — client will poll again in 2-3s
    return jsonify({"questions": [], "ready": False})

@app.route("/api/pdf/<filename>")
def serve_pdf(filename):
    """Serve the raw PDF file directly — used by the embedded viewer in the panel."""
    filename = secure_filename(filename)
    if not _ensure_pdf_local(filename):
        return jsonify({"error": f"PDF '{filename}' not available"}), 404
    return send_file(
        str(_pdf_path(filename)),
        mimetype="application/pdf",
        as_attachment=False,
        download_name=filename,
        conditional=True,
    )

@app.route("/api/debug/pdfs")
def debug_pdfs():
    pdfs = [{"name": p.name, "size_kb": round(p.stat().st_size/1024,1)} for p in PREVIEW_DIR.glob("*.pdf")]
    return jsonify({"pdfs": pdfs, "preview_dir": str(PREVIEW_DIR)})

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": f"File too large. Max {MAX_MB}MB"}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404

if __name__ == "__main__":
    port  = int(os.getenv("PORT", 8000))
    debug = os.getenv("FLASK_ENV", "production") == "development"
    print(f"\n  NexusIQ → http://localhost:{port}")
    print(f"  PDFs: {[p.name for p in PREVIEW_DIR.glob('*.pdf')]}\n")
    app.run(host="0.0.0.0", port=port, debug=debug)