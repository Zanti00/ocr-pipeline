# Documentation Index: OCR Pipeline

**Project slug:** `ocr-pipeline`  
**Maintained by:** SERMS Core Team  
**Last updated:** 2026-07-11  

---

## 1. Document Suite

This index maps the canonical documentation files for the OCR Pipeline microservice. Use these links to navigate to specific sections:

| Document | File | Version | Status | Description |
|---|---|---|---|---|
| **PRD** (Product Requirements) | [PRD.md](PRD.md) | 1.0.0 | Approved | Focuses on what the product does, its target audience, and non-goals. |
| **SAD** (Software Architecture) | [SAD.md](SAD.md) | 1.0.0 | Approved | Focuses on high-level system components, the technology stack, and component communication. |
| **SDD** (System Design) | [SDD.md](SDD.md) | 1.0.0 | Approved | Details FastAPI API routes, Celery background worker tasks, and authentication logic. |
| **DSD** (Detailed System/Database Design) | [DSD.md](DSD.md) | 1.0.0 | Approved | Specifies PostgreSQL pgvector models, MongoDB job collection structures, and Redis caches. |
| **BUILD** (Build & Setup Guide) | [Build.md](Build.md) | 1.0.0 | Approved | Provides setup guides for running the microservice stack via Docker Compose and running test suites. |
| **AGENT-SETUP** (One-Shot Subagent Setup) | [agent-setup/SETUP-02-ocr-pipeline.md](agent-setup/SETUP-02-ocr-pipeline.md) | 1.1.0 | Approved | Executable setup runbook for AI subagents on bare machines (zero-state installer, `USER-INPUT-GATE` secret collection, verification matrix). |

---

## 2. Change Log

| Date | Change Description | Docs Affected |
|---|---|---|
| 2026-07-11 | Initialized standard document suite for the OCR Pipeline microservice. | All documents |
| 2026-07-11 | Documented PostgreSQL pgvector and MongoDB schemas under Detailed System Design. | DSD.md |
| 2026-07-11 | Structured API endpoints and Celery tasks under System Design. | SDD.md |
| 2026-07-11 | Created Docker Compose startup instructions and pytest guides under Build Guide. | Build.md |
| 2026-09-06 | Added one-shot AI subagent setup runbook (zero-state host installer, secret gates, verification matrix). | agent-setup/SETUP-02-ocr-pipeline.md |

---

## 3. Health Check

Quick triage list to ensure document alignment:
- [x] All document versions are unified at `1.0.0`.
- [x] Environment variable definitions in `Build.md` correspond to the keys in `.env.local`.
- [x] Schema maps in `DSD.md` reflect the current code models inside `app/db/models.py`.
- [x] Integration sequences in `SAD.md` accurately depict the webhook callbacks between FastAPI and Laravel.
