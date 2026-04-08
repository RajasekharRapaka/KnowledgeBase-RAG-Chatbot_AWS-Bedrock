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
# Cache generated questions so /api/questions is instant on repeat calls
_question_cache: dict = {}  # filename -> [questions]


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
    """Generate 6 document-specific questions. Results are cached by filename."""
    if not text.strip():
        return []
    # Return cached result instantly if available
    if filename in _question_cache:
        return _question_cache[filename]
    prompt = (
        f'Read this document excerpt from "{filename}" and write exactly 6 specific questions '
        f'a reader would naturally ask. Each question must reference actual content '
        f'(names, numbers, concepts, dates) from the text. Max 12 words each. '
        f'Return ONLY a valid JSON array of 6 strings, nothing else.\n\n'
        f'Document:\n{text[:3000]}\n\nJSON:'
    )
    try:
        out = _invoke(prompt, max_tokens=400, temperature=0.2)
        qs  = _parse_list(out)
        if len(qs) >= 3:
            result = qs[:6]
            _question_cache[filename] = result
            return result
    except Exception as e:
        print(f"[gen_q] {e}")
    return []

def _generate_questions_async(text: str, filename: str, cfg_snapshot: dict) -> None:
    """Run question generation in a background thread and cache the result."""
    if filename in _question_cache:
        return
    # We need a fake app context to call _invoke; snapshot the config
    try:
        import boto3
        region = cfg_snapshot.get("region") or os.getenv("AWS_REGION", "us-east-1")
        key_id = cfg_snapshot.get("key")    or os.getenv("AWS_ACCESS_KEY_ID", "")
        secret = cfg_snapshot.get("secret") or os.getenv("AWS_SECRET_ACCESS_KEY", "")
        arn    = cfg_snapshot.get("model")  or os.getenv("BEDROCK_MODEL_ARN", "") or "anthropic.claude-3-5-sonnet-20241022-v2:0"
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
            "max_tokens": 400, "temperature": 0.2,
            "messages": [{"role": "user", "content": prompt}]
        })
        resp = client.invoke_model(modelId=arn, body=body)
        out  = json.loads(resp["body"].read())["content"][0]["text"].strip()
        s = out.find("["); e = out.rfind("]") + 1
        if s != -1 and e > s:
            qs = json.loads(out[s:e])
            if isinstance(qs, list) and len(qs) >= 3:
                _question_cache[filename] = qs[:6]
                print(f"[gen_q_async] cached {len(qs)} questions for {filename}")
    except Exception as ex:
        print(f"[gen_q_async] {ex}")

def generate_followups(question: str, answer: str) -> list:
    prompt = (
        f"Given this Q&A, generate exactly 3 natural follow-up questions. "
        f"Max 10 words each. Return ONLY a valid JSON array of 3 strings.\n\n"
        f"Q: {question}\nA: {answer[:500]}\n\nJSON:"
    )
    try:
        out = _invoke(prompt, max_tokens=200, temperature=0.4)
        qs  = _parse_list(out)
        return qs[:3] if qs else []
    except Exception as e:
        print(f"[gen_fu] {e}")
    return []

def extract_highlight_phrases(answer: str) -> list:
    """Extract short verbatim-matchable phrases from the answer for PDF highlighting."""
    clean = re.sub(r'\*+|#+|`+|\[.*?\]', '', answer)
    quoted = re.findall(r'"([^"]{15,90})"', answer)
    sentences = []
    for s in re.split(r'[.!?\n]', clean):
        s = s.strip()
        if 20 < len(s) < 100:
            sentences.append(s)
    seen, result = set(), []
    for p in quoted + sentences:
        key = p[:35].lower()
        if key not in seen:
            seen.add(key)
            result.append(p)
        if len(result) >= 10:
            break
    return result


# ── PDF RENDERER ───────────────────────────────────────────────────────────────
def render_pdf_pages(pdf_path: str, highlight_texts: list = None) -> list:
    """
    Fast PDF renderer using PyMuPDF.
    Returns [] if PyMuPDF is unavailable — caller should fall back to /api/pdf embed.
    """
    try:
        import fitz
    except ImportError:
        print("[render_pdf] PyMuPDF (fitz) not installed — install with: pip install PyMuPDF")
        return []
    try:
        doc = fitz.open(pdf_path)
        total = len(doc)
        pages = []
        highlighted_pages = set()

        if highlight_texts:
            for i in range(total):
                pg = doc[i]
                for phrase in highlight_texts:
                    if len(phrase) > 8 and pg.search_for(phrase[:80]):
                        highlighted_pages.add(i)
                        break

        for i in range(min(total, 30)):
            pg = doc[i]
            scale = 1.5 if (highlight_texts and i in highlighted_pages) else 1.2
            if highlight_texts and i in highlighted_pages:
                for phrase in highlight_texts:
                    if len(phrase) > 8:
                        for area in pg.search_for(phrase[:80]):
                            hl = pg.add_highlight_annot(area)
                            hl.set_colors(stroke=[1, 0.85, 0.0])
                            hl.update()
            pix = pg.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            pages.append({
                "page":          i + 1,
                "total":         total,
                "data":          base64.b64encode(pix.tobytes("png")).decode(),
                "has_highlight": i in highlighted_pages,
            })
        doc.close()
        return pages
    except Exception as e:
        print(f"[render_pdf] {e}")
        return []


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

        # 1. Extract text (fast — only first 15 pages)
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

        # 4. Upload to S3 + start KB sync
        engine = get_engine()
        result = engine.upload(str(tmp), filename)

        # 5. Check question cache first (instant if already generated)
        questions = _question_cache.get(filename, [])

        if not questions and text:
            # Try to generate synchronously but with a tight budget
            # If it takes too long the async background thread will cache it
            cfg_snap = dict(session.get("cfg", {}))
            if len(text) > 0:
                # Fire async generation so /api/questions returns fast later
                t = threading.Thread(
                    target=_generate_questions_async,
                    args=(text, filename, cfg_snap),
                    daemon=True
                )
                t.start()
                # Also try a quick synchronous attempt (will be fast if already cached)
                t.join(timeout=25)  # wait up to 25s — questions returned immediately if done
                questions = _question_cache.get(filename, [])

        return jsonify({
            "success":     True,
            "questions":   questions,
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
        _question_cache.pop(filename, None)
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
        result   = get_engine().query(question, mode=data.get("mode", "precise"))
        answer   = result.get("answer", "")
        result["followups"]         = generate_followups(question, answer)
        result["highlight_phrases"] = extract_highlight_phrases(answer)

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
            "error":       f"PDF '{filename}' not found locally or in S3.",
            "can_embed":   False,
            "pages":       [],
        }), 404
    highlights = request.args.get("highlights", "")
    hl_list    = [h.strip() for h in highlights.split("||") if h.strip()] if highlights else []
    pages      = render_pdf_pages(str(_pdf_path(filename)), hl_list)
    first_hl_page = next((p["page"] for p in pages if p.get("has_highlight")), None)
    # can_embed=True tells the frontend it can fall back to /api/pdf embed
    return jsonify({
        "pages":         pages,
        "filename":      filename,
        "first_hl_page": first_hl_page,
        "can_embed":     True,   # PDF is available at /api/pdf/<filename>
    })

@app.route("/api/questions/<filename>")
def get_questions(filename):
    filename = secure_filename(filename)
    # Instant return if already cached
    if filename in _question_cache:
        return jsonify({"questions": _question_cache[filename], "ready": True})

    # Ensure we have text to work with
    text = _session_texts().get(filename, "")
    if not text:
        if filename.lower().endswith(".pdf") and _ensure_pdf_local(filename):
            text = extract_text(str(_pdf_path(filename)), filename)
            if text:
                _session_texts()[filename] = text
                print(f"[questions] extracted {len(text)} chars from {filename}")

    if not text:
        print(f"[questions] no text available for {filename}")
        return jsonify({"questions": [], "ready": False})

    # Fire async generation if not already running
    cfg_snap = dict(session.get("cfg", {}))
    t = threading.Thread(
        target=_generate_questions_async,
        args=(text, filename, cfg_snap),
        daemon=True,
    )
    t.start()
    # Wait up to 20s — questions returned instantly if Bedrock responds quickly
    t.join(timeout=20)

    questions = _question_cache.get(filename, [])
    return jsonify({"questions": questions, "ready": len(questions) > 0})

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