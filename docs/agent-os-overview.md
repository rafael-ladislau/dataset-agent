# Agent OS Overview

`agent-os/` contains the product context, engineering standards, and AI agent commands used to guide AI-assisted development of the Dataset Research Agent. It is a configuration layer for AI coding assistants (Cursor, Claude, etc.) rather than runtime application code.

---

## Folder Structure

```
agent-os/
├── config.yml          # Agent OS version and mode settings
├── product/            # Product definition (mission, roadmap, tech stack, decisions)
├── standards/          # Engineering standards referenced by the AI agent
│   ├── global/         # Cross-cutting standards (architecture, git, coding style, etc.)
│   ├── backend/        # Backend-specific standards
│   ├── frontend/       # Frontend-specific standards (n/a for this project)
│   └── testing/        # Test standards
├── commands/           # Reusable AI agent slash-command workflows
│   ├── create-spec/
│   ├── implement-spec/
│   ├── new-spec/
│   └── plan-product/
├── roles/              # Agent role definitions
└── specs/              # Feature specifications with implementation artifacts
    └── 2026-01-08-project-documentation-for-sow-response/
```

---

## Product Definition (`agent-os/product/`)

These files define the *what* and *why* of the project. They are the primary source of truth for the AI agent's understanding of scope.

| File | Purpose |
|---|---|
| `mission.md` | Product pitch, user personas, problem statement, key features |
| `roadmap.md` | Completed phases, current work, near-term enhancements |
| `tech-stack.md` | Technology choices, env variables, API endpoints |
| `decisions.md` | Architectural decisions and rationale |
| `mission-lite.md` | Condensed one-liner for quick agent context |

### Product Summary
The Dataset Research Agent automates metadata enrichment for datasets — extracting descriptions, aliases, organizations, access type, and URLs — using LLMs with web search. It ships as both a FastAPI service and a CLI.

See full product details:
- [docs/agent-os-overview.md](agent-os-overview.md) ← this file
- [agent-os/product/mission.md](../agent-os/product/mission.md)
- [agent-os/product/roadmap.md](../agent-os/product/roadmap.md)
- [agent-os/product/tech-stack.md](../agent-os/product/tech-stack.md)
- [agent-os/product/decisions.md](../agent-os/product/decisions.md)

---

## Engineering Standards (`agent-os/standards/`)

Referenced by the AI agent when writing or reviewing code. Humans can read these to understand project conventions.

### Global Standards
| File | Covers |
|---|---|
| `architecture.md` | Clean architecture layers (domain, adapters, utils), dependency rules |
| `coding-style.md` | Python style conventions for this project |
| `git-workflow.md` | Branch strategy, commit format, PR process |
| `conventions.md` | Naming conventions, file organization |
| `error-handling.md` | Exception handling patterns |
| `validation.md` | Input/output validation patterns |
| `commenting.md` | When and how to comment code |
| `tech-stack.md` | Detailed tech stack rationale and usage guidelines |

### Backend Standards
Covers FastAPI route patterns, SQLite access patterns, background task conventions, and logging.

### Testing Standards
Unit test conventions for domain logic and adapters.

---

## Feature Specifications (`agent-os/specs/`)

Each spec folder captures the full lifecycle of a feature: planning, implementation notes, and verification.

### Completed Specs
- **`2026-01-08-project-documentation-for-sow-response/`** — White paper and project documentation for a Statement of Work response. Produced [`docs/white-paper-dataset-research-agent.md`](white-paper-dataset-research-agent.md).

---

## AI Agent Commands (`agent-os/commands/`)

Slash-command workflows that AI coding assistants can invoke to perform structured tasks:
- `new-spec/` — Create a new feature spec from scratch
- `create-spec/` — Create a spec from an existing description
- `implement-spec/` — Implement a planned spec
- `plan-product/` — Update roadmap and product definition

---

## Configuration (`agent-os/config.yml`)

```yaml
version: 2.0.3
profile: default
multi_agent_mode: false
single_agent_mode: true
single_agent_tool: generic
```

Configures the Agent OS version and whether to use single vs multi-agent mode for AI-assisted tasks.

---

## Related Documentation

- [README.md](../README.md) — Project setup and quick start
- [docs/white-paper-dataset-research-agent.md](white-paper-dataset-research-agent.md) — Comprehensive project white paper
- [docs/lmstudio-integration-guide.md](lmstudio-integration-guide.md) — LMStudio LLM provider setup
- [docs/lmstudio-provider-spec.md](lmstudio-provider-spec.md) — LMStudio feature specification
- [docs/official-name-detection-summary.md](official-name-detection-summary.md) — Official name detection implementation
- [scripts/README.md](../scripts/README.md) — Batch processing scripts documentation
