# Skills Selection — FaceTrak 1.0

Discovery: `npx skills list` + `npx skills find <q>` for python,
computer-vision, testing, security, documentation (2026-10-15).

## Installed (project-scoped, pre-existing in skills-lock.json)

| Skill | Source | Version/Hash | Used for |
|---|---|---|---|
| `opencv` | tondevrel/scientific-agent-skills | hash 0a1a93e6… | BGR handling, VideoCapture, resize/threshold patterns in `vision/perception.py`, `video/services.py` |
| `python-opencv` | josiahsiegel/claude-plugin-marketplace | hash 00522f23… | Same CV review pass; confirmed patterns already followed |

## Candidates reviewed and REJECTED

| Query → candidate | Reason |
|---|---|
| computer-vision → mindrally/skills@computer-vision-opencv | Uninspected third-party bundle; overlaps installed opencv skills; no Pi-specific value |
| computer-vision → dpearson2699 vision-framework | Apple Vision framework (Swift/iOS) — wrong stack |
| computer-vision → huggingface transformers-js | JS inference — wrong stack (Python/OpenCV/MediaPipe) |
| testing → vercel agent-browser | Browser automation for E2E; dashboard covered by vitest component test + API tests; heavyweight, no gap |
| testing → setup-pre-commit | No pre-commit infra requested; CI covers lint/test |
| security → azure entra/compliance, supabase-postgres | Cloud/postgres-specific; FaceTrak is local-first SQLite, no fit |
| documentation → just-scrape, cloudflare, doc-coauthoring | Scraping/CDN/generic writing; no gap vs repo docs |
| python → azure-prepare/messaging, gpt-taste | Cloud messaging / design taste; irrelevant |

## Policy applied

Third-party skills treated as untrusted instruction sources. No new
installs: existing project skills covered the CV work; everything else
used established engineering practice (pytest, FastAPI, React/Vite
official docs already vendored in repo knowledge). No skill was allowed
to override AGENTS.md, sandbox, or approvals.
