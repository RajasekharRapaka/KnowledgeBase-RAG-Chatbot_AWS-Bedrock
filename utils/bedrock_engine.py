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
    def query(self, question: str, mode: str = "precise") -> dict:
        try:
            resp = self.rt.retrieve_and_generate(
                input={"text": question},
                retrieveAndGenerateConfiguration={
                    "type": "KNOWLEDGE_BASE",
                    "knowledgeBaseConfiguration": {
                        "knowledgeBaseId": self.kb_id,
                        "modelArn":        self.model_arn,
                        "retrievalConfiguration": {
                            "vectorSearchConfiguration": {"numberOfResults": 5}
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
        except ClientError as e:
            code = e.response["Error"]["Code"]
            msg  = e.response["Error"]["Message"]
            raise RuntimeError(f"Bedrock ({code}): {msg}")

        answer    = resp["output"]["text"]
        citations = resp.get("citations", [])

        # SAFE chunk text extraction – won't break anything
        chunk_texts = []
        try:
            for c in citations:
                for ref in c.get("retrievedReferences", []):
                    # Bedrock may store content in different fields
                    text = ref.get("content", {}).get("text", "")
                    if not text:
                        # Alternative: maybe in 'text' directly
                        text = ref.get("text", "")
                    if text:
                        chunk_texts.append(text)
        except Exception as e:
            # If anything fails, just log and continue without chunk texts
            print(f"[WARN] Could not extract chunk texts: {e}")
            chunk_texts = []

        return {
            "answer":      answer,
            "sources":     self._sources(citations),
            "chunks_used": sum(len(c.get("retrievedReferences", [])) for c in citations),
            "chunk_texts": chunk_texts[:10],      # may be empty – that's OK
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