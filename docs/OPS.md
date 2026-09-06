# Operations & Observability Runbook (OPS)

**Project:** OCR Pipeline
**Version:** 0.1
**Status:** Draft
**Last reconciled:** N/A — not yet reconciled with prod
**Related docs:** [SDD.md](SDD.md) · [QAD.md](QAD.md) · [Build.md](Build.md)

---

> **Scope reality:** This is a **headless Python/FastAPI microservice** running on Docker Compose. There is no UI and no server-rendered frontend. The operational surface is: keeping the API and Celery workers healthy, monitoring the Ollama model service, and ensuring the two consuming services (SERMS and PRS) receive reliable callbacks. The primary operational risks are **bad deploys**, **Ollama OOM/timeout failures**, and **Celery queue blockage**.

---

## 1. SLOs & SLIs

| SLI (what you measure) | SLO (target) | Measured by | Breach action |
|---|---|---|---|
| API Availability (`/api/health`) | ≥ 99% uptime | Manual spot-checks (no formal uptime monitor yet) | Inspect container logs; restart failing service |
| OCR Job Processing Time | 95% of jobs complete in ≤ 10s | MongoDB `jobs` collection timestamps | Investigate Ollama latency or queue depth |
| OCR Job Processing Time (offline target) | Average receipt < 60 s on clean receipts; hard receipts may exceed | `scripts/benchmark.py` per-stage timing | Profile engine pool size; review fast-path escalation ratio |
| OCR Pool Throughput (worker) | Escalated pool ≤ 60s per hard receipt | `scripts/benchmark.py`; worker env must set `OMP_NUM_THREADS=1`/`OMP_THREAD_LIMIT=1` | Without the OMP caps, concurrent Tesseract subprocesses thrash the CPU — measured 258-309s for a grid that runs in ~4s capped (26x penalty). Post-fix benchmark: fast-path avg 1.4s, pooled avg 13.7s, max 22.6s across the 14-receipt corpus |
| Celery Task Success Rate | ≥ 97% (after 3 retries) | MongoDB job status field | Alert on consecutive failures; inspect worker logs |
| Callback Delivery | 100% on completed/failed jobs | Application logs (`app.core.callback`) | Re-trigger callback manually if SERMS/PRS reports gap |
| Composite Confidence Score | 95% of receipts score ≥ 0.75 | MongoDB `composite_confidence_score` field | Review Paddle/Tesseract cross-reference, preprocessing, or Ollama extraction quality |

> **Note:** No formal uptime monitoring is in place yet. The `/api/health` endpoint exists but its dependency checks (Ollama, Redis, MongoDB, PostgreSQL) are currently stub TODO implementations. These should be wired before production go-live.

---

## 2. Observability — Logs, Metrics, Traces

All observability is currently via **Docker container logs** and **MongoDB job records**.

| Pillar | Tool | What's captured | How to access |
|---|---|---|---|
| Logs | Docker Compose (`docker compose logs`) | FastAPI request logs, Celery task lifecycle, pipeline step errors | `docker compose logs -f api` / `docker compose logs -f worker` |
| Metrics | `/api/metrics` endpoint (stub) | Job counts, success rate, avg processing time, queue depth, avg confidence score — **currently returns zeros** | `GET /api/metrics` with valid API key |
| Job State | MongoDB `jobs` collection | Per-job status (`queued`, `processing`, `completed`, `failed`), raw OCR text, extracted data, BIR validation, confidence score | Query MongoDB directly or via `GET /api/jobs/{job_id}` |
| Traces | — | N/A (no distributed tracing configured) | — |

**No-PII rule:** No real receipt images or customer PII should appear in logs. The pipeline logs job IDs and statuses only — never raw extracted text at `INFO` level in production.

**Key external dependencies to watch:**
- **Ollama (`ocr_ollama` container):** LLM inference for structured data extraction. High memory consumer — monitor for OOM kills.
- **Redis (`ocr_redis`):** Celery broker. If Redis is down, all job submissions fail silently to the queue.
- **SERMS & PRS callback URLs:** If the downstream service is unreachable, the pipeline logs a failure and raises — the Celery task will retry up to 3 times.

---

## 3. Alerting & On-Call

> **No formal alerting is configured yet.** The following are the recommended alert definitions to implement.

| Alert | Condition | Severity | Recommended action |
|---|---|---|---|
| API container down | `ocr_api` container exits or stops responding | P0 | Restart immediately; inspect logs |
| Worker container down | `ocr_worker` container exits | P0 | Restart; check if Redis is reachable |
| Ollama OOM/crash | `ocr_ollama` container exits unexpectedly | P1 | Restart `ocr_ollama`; model re-loads on startup |
| Celery queue depth high | Redis queue length > 50 | P1 | Scale workers or investigate blocking tasks |
| High job failure rate | >3% of jobs in `failed` state in last hour | P1 | Inspect MongoDB `failed` jobs; check Ollama availability |
| Callback failures | Any `callback failed` in worker logs | P2 | Re-trigger manually; verify SERMS/PRS endpoint health |

**On-call model:** Best-effort by the development team. Monitor `docker compose ps` and MongoDB job statuses during active receipt processing windows.

---

## 4. Incident Response

**Severity ladder:**
- **P0:** API or worker completely down; no jobs can be processed.
- **P1:** Ollama unavailable; high failure rate; queue blocked.
- **P2:** Callback delivery failures; degraded confidence scores.
- **P3:** Metrics endpoint returning stubs; minor logging gaps.

**When an incident fires:**
1. **Acknowledge** — claim ownership in the dev channel.
2. **Assess** — run `docker compose ps` to check container health; check `docker compose logs -f` for the failing service.
3. **Mitigate first** — restart the failing container: `docker compose restart <service>`.
4. **Rollback strategy:** No CI/CD automated rollback is defined yet. Until formalised: `git revert` the offending commit, rebuild the image, and redeploy with `docker compose up --build -d`. Use the previous image tag if available.
5. **Resolve & verify** — submit a test receipt via `POST /api/ocr/process` and confirm a `completed` job status in MongoDB.
6. **Postmortem** — for any P0/P1, document in `docs/pm-NNN.md` within 48h and fold action items back into this runbook.

---

## 5. Routine Operations

- **Dependency updates:** Regularly run `pip list --outdated` inside the container and update `pyproject.toml`. Always re-run `docker compose up --build` and `pytest` after updates.
- **Alembic migrations:** Any schema change to the PostgreSQL `receipt_embeddings` table must be applied via Alembic before deploying new code: `alembic upgrade head`.
- **MongoDB storage:** Monitor the `ocr_mongo_data` volume growth; the `jobs` collection accumulates all OCR results.
- **Ollama model management:** Qwen2.5 1.5B is pulled automatically by `scripts/ollama-entrypoint.sh` on first start. To upgrade the model, update the entrypoint script and re-deploy.
- **API key rotation:** SERMS and PRS API keys are stored in `.env`. Rotate on a regular schedule and notify the consuming teams before rotation.

---

## Self-Check

- [ ] `/api/health` dependency checks are wired (Ollama, Redis, MongoDB, PostgreSQL stubs resolved).
- [ ] `/api/metrics` endpoint aggregates real data from MongoDB.
- [ ] Rollback procedure documented and rehearsed.
- [ ] Formal uptime monitoring configured for the API endpoint.
- [ ] Celery queue depth monitoring in place.

## 6. Financial semantics operations
Ollama is an optional assist for tax-basis and currency semantics. Timeouts, malformed JSON, low confidence, or ungrounded evidence do not fail a receipt; the deterministic baseline is retained and the audit record identifies unresolved/review state. Monitor Mongo `financial_semantics` and `reconciliation.needs_manual_review` when investigating disputed totals. Live Ollama tests are opt-in and must not be part of the default offline test gate.
