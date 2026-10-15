# Skills Selection — FaceTrak Next Delivery

Date: 2026-10-15. Policy: reuse existing skills first; install only with
need + provenance/security review; project-local only.

## Used (already present, read before use)

| Skill | Source / version | Why | Used for |
|---|---|---|---|
| `using-superpowers` | `~/.agents/skills/using-superpowers` | Mandatory entry skill per global AGENTS.md | Workflow routing, brainstorming gate handling |
| `brainstorming` | `~/.agents/skills/brainstorming` | Process skill for "let's build X" | Scope gate: user opted out of questions → one-line build decision, then execute |
| `agentic-engineering` | `~/.agents/skills/agentic-engineering` | Selected via Laya decision call (provider `laya`, ~24 s) over `tdd-workflow`/`code-review` | Discovery → plan → challenge → implement → test → review → DoD loop for this delivery |
| `opencv` | `tondevrel/scientific-agent-skills`, v4.9.0, Apache-2.0 (project-local, `skills-lock.json`) | Installed, reputable (Microsoft DevRel org) | `fillPoly` privacy masks, `pointPolygonTest` zones, BGR/cleanup discipline |
| `python-opencv` | `josiahsiegel/claude-plugin-marketplace` (project-local, `skills-lock.json`) | Installed, reviewed SKILL.md + references | VideoCapture release patterns, dtype safety, DNN notes for YuNet guard |

## Evaluated, NOT installed (with reason)

- `find-skills` (vercel-labs/skills): the order foresaw
  `npx skills add … -a codex`, but this session runs in OpenCode, not
  Codex (`-a codex` inapplicable), and both needed CV skills were
  already installed and reviewed. Installing a third-party catalog
  client plus ~10 new skills would add supply-chain surface without a
  concrete capability gap. `npx skills list` (read-only) confirmed the
  installed set. Revisit if a future task needs a capability neither
  installed skill covers.
- No FastAPI/testing/security/performance skills installed: FastAPI
  auth work used framework-standard `HTTPBearer` + `TestClient`
  (verified by 15 API tests); MQTT used stdlib + injected fakes;
  benchmarks used `time.monotonic` + `statistics`. No gap found that
  justified new installs.

## License / security check of used skills

- `opencv` SKILL.md: Apache-2.0, docs-only, no executable code, no
  external calls → safe.
- `python-opencv` SKILL.md + `references/`: markdown-only patterns,
  no install scripts → safe.
- `agentic-engineering` / `brainstorming` / `using-superpowers`: local
  process docs, no code execution → safe.
- `facetrak-mcp` (project skill): SKIPPED — broken frontmatter
  (`skills list` warns: missing `name`/`description`), not used.
