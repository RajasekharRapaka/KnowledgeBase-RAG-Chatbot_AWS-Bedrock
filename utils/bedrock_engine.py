"""
NexusIQ — Amazon Bedrock RAG Engine
Handles: S3 upload · KB ingestion · retrieve-and-generate · document management
"""
import os, mimetypes
from pathlib import Path
import boto3
from botocore.exceptions import ClientError

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx", ".doc", ".md", ".html", ".csv"}
DEFAULT_MODEL_ID = "anthropic.claude-3-5-sonnet-20241022-v2:0"


class BedrockEngine:
    def __init__(self, region, key_id, secret, token, bucket, kb_id, ds_id, model_arn=""):
        self.region    = region    or "us-east-1"
        self.bucket    = bucket    or ""
        self.kb_id     = kb_id     or ""
        self.ds_id     = ds_id     or ""
        self.model_arn = model_arn or (
            f"arn:aws:bedrock:{self.region}::foundation-model/{DEFAULT_MODEL_ID}"
        )
        session_kwargs = {"region_name": self.region}
        if key_id and secret:
            session_kwargs["aws_access_key_id"]     = key_id
            session_kwargs["aws_secret_access_key"] = secret
            if token:
                session_kwargs["aws_session_token"] = token

        sess = boto3.Session(**session_kwargs)
        self.s3     = sess.client("s3")
        self.agent  = sess.client("bedrock-agent")
        self.rt     = sess.client("bedrock-agent-runtime")

    # ── CONFIG CHECK ──────────────────────────────────────────────────────────
    def validate(self) -> dict:
        missing = [k for k, v in {"S3 Bucket": self.bucket, "KB ID": self.kb_id, "Data Source ID": self.ds_id}.items() if not v]
        if missing:
            return {"ok": False, "error": f"Missing: {', '.join(missing)}"}
        try:
            self.s3.head_bucket(Bucket=self.bucket)
            return {"ok": True}
        except ClientError as e:
            return {"ok": False, "error": f"S3: {e.response['Error']['Message']}"}
        except Exception as e:
            return {"ok": False, "error": f"AWS connection failed: {str(e)[:200]}"}

    # ── UPLOAD TO S3 ──────────────────────────────────────────────────────────
    def upload(self, local_path: str, filename: str) -> dict:
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")
        s3_key = f"knowledge-base/{filename}"
        ct = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        self.s3.upload_file(local_path, self.bucket, s3_key, ExtraArgs={"ContentType": ct})
        job_id = self._sync()
        return {"s3_key": s3_key, "sync_job_id": job_id, "filename": filename}

    def _sync(self) -> str:
        try:
            r = self.agent.start_ingestion_job(knowledgeBaseId=self.kb_id, dataSourceId=self.ds_id)
            return r["ingestionJob"]["ingestionJobId"]
        except ClientError as e:
            raise RuntimeError(f"Ingestion failed: {e.response['Error']['Message']}")

    # ── INGESTION POLL ────────────────────────────────────────────────────────
    def ingestion_status(self, job_id: str) -> dict:
        try:
            r = self.agent.get_ingestion_job(
                knowledgeBaseId=self.kb_id, dataSourceId=self.ds_id, ingestionJobId=job_id
            )
            j = r["ingestionJob"]
            s = j.get("statistics", {})
            return {
                "status":    j["status"],
                "indexed":   s.get("numberOfDocumentsIndexed", 0),
                "failed":    s.get("numberOfDocumentsFailed", 0),
                "scanned":   s.get("numberOfDocumentsScanned", 0),
            }
        except ClientError as e:
            return {"status": "ERROR", "error": e.response["Error"]["Message"]}

    # ── LIST S3 DOCS ──────────────────────────────────────────────────────────
    def list_docs(self) -> list:
        try:
            r = self.s3.list_objects_v2(Bucket=self.bucket, Prefix="knowledge-base/")
            out = []
            for obj in r.get("Contents", []):
                fname = obj["Key"].replace("knowledge-base/", "")
                if fname:
                    out.append({
                        "s3_key":        obj["Key"],
                        "filename":      fname,
                        "size_kb":       round(obj["Size"] / 1024, 1),
                        "last_modified": obj["LastModified"].strftime("%b %d, %Y"),
                    })
            return out
        except ClientError:
            return []

    # ── DELETE FROM S3 ────────────────────────────────────────────────────────
    def delete(self, s3_key: str) -> bool:
        try:
            self.s3.delete_object(Bucket=self.bucket, Key=s3_key)
            self._sync()
            return True
        except ClientError:
            return False

    # ── RAG QUERY ─────────────────────────────────────────────────────────────
    def query(self, question: str, mode: str = "precise", source_filename: str = "") -> dict:
        """
        source_filename — when provided, adds a Bedrock metadata filter so
        *only* chunks from that specific S3 object are retrieved.
        This is the primary fix for stale-chunk contamination: if the KB still
        has old documents indexed (e.g. after a failed sync), those chunks are
        completely excluded when querying about a different file.
        Falls back to unfiltered search if the filter returns nothing.
        """
        # ── Build vector search config ────────────────────────────────────────
        vector_cfg: dict = {"numberOfResults": 8}   # more chunks = better recall

        if source_filename and self.bucket:
            s3_uri = f"s3://{self.bucket}/knowledge-base/{source_filename}"
            vector_cfg["filter"] = {
                "equals": {
                    "key":   "x-amz-bedrock-kb-source-uri",
                    "value": s3_uri,
                }
            }
            print(f"[query] KB filter → {s3_uri}")

        def _call(vcfg):
            return self.rt.retrieve_and_generate(
                input={"text": question},
                retrieveAndGenerateConfiguration={
                    "type": "KNOWLEDGE_BASE",
                    "knowledgeBaseConfiguration": {
                        "knowledgeBaseId": self.kb_id,
                        "modelArn":        self.model_arn,
                        "retrievalConfiguration": {
                            "vectorSearchConfiguration": vcfg
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

        try:
            resp = _call(vector_cfg)
        except ClientError as e:
            code = e.response["Error"]["Code"]
            msg  = e.response["Error"]["Message"]
            # Some regions / KB versions don't support the filter field —
            # silently retry without it rather than erroring out.
            if "filter" in vector_cfg and ("ValidationException" in code or "filter" in msg.lower()):
                print(f"[query] filter unsupported, retrying without it")
                fallback_cfg = {"numberOfResults": 8}
                try:
                    resp = _call(fallback_cfg)
                except ClientError as e2:
                    raise RuntimeError(f"Bedrock ({e2.response['Error']['Code']}): {e2.response['Error']['Message']}")
            else:
                raise RuntimeError(f"Bedrock ({code}): {msg}")

        answer    = resp["output"]["text"]
        citations = resp.get("citations", [])

        # ── Extract chunk texts ───────────────────────────────────────────────
        chunk_texts: list = []
        chunks_used: int  = 0
        try:
            for c in citations:
                refs = c.get("retrievedReferences", [])
                chunks_used += len(refs)
                for ref in refs:
                    text = (ref.get("content") or {}).get("text", "") or ref.get("text", "")
                    if text:
                        chunk_texts.append(text)
        except Exception as ex:
            print(f"[WARN] chunk extraction: {ex}")

        # ── If filtered query returned 0 chunks, ALWAYS retry without filter ──
        # Reason: when the filter finds no chunks, Bedrock receives an empty
        # $search_results$ context and responds with a safety/refusal message
        # ("Sorry, I am unable to assist…") regardless of what the question is.
        # Don't try to detect this by string-matching the answer — just always
        # fall back when no chunks were retrieved.
        if source_filename and chunks_used == 0:
            print("[query] 0 chunks with filter — retrying without filter (unscoped search)")
            try:
                fallback_cfg = {"numberOfResults": 8}
                resp2        = _call(fallback_cfg)
                answer2      = resp2["output"]["text"]
                citations2   = resp2.get("citations", [])
                chunks_used2 = sum(len(c.get("retrievedReferences", [])) for c in citations2)
                print(f"[query] unscoped retry — {chunks_used2} chunks")
                answer      = answer2
                citations   = citations2
                chunks_used = chunks_used2
                chunk_texts = []
                for c in citations2:
                    for ref in c.get("retrievedReferences", []):
                        text = (ref.get("content") or {}).get("text", "") or ref.get("text", "")
                        if text:
                            chunk_texts.append(text)
            except Exception as fe:
                print(f"[query] fallback error: {fe}")

        return {
            "answer":      answer,
            "sources":     self._sources(citations),
            "chunks_used": chunks_used,
            "chunk_texts": chunk_texts[:10],
        }

    # ── HELPERS ───────────────────────────────────────────────────────────────
    def _prompt(self, mode: str) -> str:
        style = {
            "precise":  "Be concise and factual. Use bullet points.",
            "detailed": "Be thorough and comprehensive with full context.",
            "summary":  "Summarize in 3-5 sentences only.",
        }.get(mode, "Be concise and factual.")
        return (
            "You are an Enterprise Knowledge Base Assistant. "
            "Answer ONLY using the provided context below. "
            "Always mention which document your answer comes from. "
            "If the answer is not in the context, say so clearly.\n\n"
            "$search_results$\n\n"
            f"Question: $query$\n\n{style}"
        )

    def _sources(self, citations: list) -> list:
        seen, out = set(), []
        for c in citations:
            for ref in c.get("retrievedReferences", []):
                uri   = ref.get("location", {}).get("s3Location", {}).get("uri", "")
                fname = uri.split("/")[-1] if uri else "Unknown source"
                score = round(ref.get("metadata", {}).get("score", 0) * 100, 1)
                if fname not in seen:
                    seen.add(fname)
                    out.append({"filename": fname, "uri": uri, "relevance": score})
        return out

    def stats(self) -> dict:
        docs = self.list_docs()
        return {
            "total_docs":    len(docs),
            "total_size_kb": round(sum(d["size_kb"] for d in docs), 1),
            "kb_id":         self.kb_id,
            "s3_bucket":     self.bucket,
            "region":        self.region,
        }