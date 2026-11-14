# RepoRover: raw content for the submission template

Theme: Agentic AI for Cybersecurity & Digital Trust.

This file is source material to paste into the official template by hand. Every number below was
produced by a command in this repository and can be reproduced; the command is given next to it.
Section 6 lists what has **not** been verified, so nothing on a slide overstates the project.

---

## 1. Problem statement

- Applications are increasingly "vibe-coded": assembled by prompting an AI assistant, one feature at
  a time, by people who do not review every line.
- Each generated piece works in isolation. The defects are in what was never asked for: no
  ownership check on a record lookup, no rate limit, money stored as floats, business data kept
  only in the browser, a client-supplied price trusted by the server.
- These are logic and architecture flaws, not syntax or known-vulnerable-API patterns. A pattern
  scanner tuned for injection sinks has no rule for "this app has no idempotency key on checkout".
- Pointing a general LLM at the repository has the opposite failure: it produces confident findings
  about code it never read. A security report that invents findings destroys the trust it is meant
  to create.
- The gap: a reviewer that understands intent like an LLM, but can only assert what it can prove
  from the code.

## 2. Solution

RepoRover is a five-agent pipeline that analyses a local repository and produces a prioritised,
copy-paste remediation prompt in which every finding carries its evidence.

| Stage | What it does | Uses an LLM? |
|---|---|---|
| Fingerprint | Detects languages, frameworks and libraries from manifests, imports, script tags and config files | No |
| Detector engine | Runs 143 catalogue rules over the code; each hit has file, line and the verbatim text | No |
| Scout | States the business intent, each claim backed by an exact quote | Yes, checked |
| Inspector | Judges whether each detector hit applies to this product; cannot add findings | Yes, checked |
| Red Teamer | Proposes product-specific scenarios; may only store ones with verified evidence | Yes, checked |
| Optimizer | Finds algorithmic waste; every finding must quote code it read | Yes, checked |
| Synthesizer | Writes the wording; labels and layout are assigned by code | Yes, checked |

Output: one report where each item is labelled `[Verified]` (an exact quote matched),
`[Detector-Hit]` (a deterministic rule fired) or `[Hypothesis]` (no code evidence; shown only in a
separate final section), plus ready-to-run chaos experiments for the evidence-backed findings.

## 3. Novelty and USP

### 3.1 Evidence-first pipeline

- **Deterministic pre-pass before any LLM context.** The stack and the candidate findings are
  computed by code. The model receives `file:line | verbatim code` and is asked to judge it.
- **A read ledger.** Every line a tool serves to an agent is recorded with its file hash. A claim
  about a file that was not read, or a quote that does not match what was served, is rejected and
  the agent must retry.
- **The model cannot add or relabel findings.** The Inspector may only return verdicts on hit ids
  the engine produced. The Synthesizer may only write text for ids in the findings registry; the
  report is rendered by code from that registry.
- **Graceful degradation.** If an agent keeps failing validation, its unverified claims are
  removed, the run is marked degraded and database writes are blocked. The run still finishes.
- **Nothing unproven reaches the database.** A discovery is stored only with verified evidence, a
  confidence value and a stack taken from the deterministic fingerprint.

Measured (command: `reporover bench`):

| Metric | Value |
|---|---|
| Precision / Recall / F1, all rules | 90.0% / 85.1% / 0.875 |
| Precision / Recall / F1, code-level rules only | 89.1% / 87.5% / 0.883 |
| Hallucination rate | 0.0% (0 of 64 cited code locations failed re-verification) |
| Scope | 3 synthetic fixtures, 74 hand-labelled flaws, detector stage only |

On one recorded live run (model `gemini-3.5-flash-lite`, fixture `static_ledger`): 20 model calls,
3 agent outputs rejected by guardrails and retried, final status completed, 16 findings in the
report (2 Verified, 14 Detector-Hit, 0 Hypothesis), 0 failing independent re-verification.

### 3.2 IP security

- **The source code never has to leave the machine.** The model client talks to any
  OpenAI-compatible endpoint, set by `LLM_BASE_URL`. Pointing it at a local Ollama server
  (`http://localhost:11434/v1`) keeps inference on the developer's hardware.
- **Zero cloud egress is a configuration, not a default.** It needs three things: a local model
  endpoint, a local Postgres in `DATABASE_URL`, and telemetry off. RepoRover disables CrewAI's
  telemetry and trace-upload prompt by default.
- **Tool confinement.** Agents can only read inside the target directory. Paths that escape it,
  including through `..` or symlinks, are refused. Version-control folders, dependency folders and
  virtual environments are skipped. Secret files (`.env`, keys, certificates) are never read, so
  they cannot be sent to a model.
- **Bounded exposure.** At most 200 lines per read and 4,000 lines per run are served to agents.

## 4. Show stopper

**RepoRover is held to the standard it applies to others, and it proves findings instead of
asserting them.**

- **Dogfooding.** `reporover chaos self` injects faults into RepoRover's own model and database
  paths: HTTP 429 bursts, HTTP 503, a total model outage, added latency, malformed JSON
  completions, statement timeouts, an unreachable database, a connection lost after commit, a
  process crash mid-pipeline, an exhausted read budget and 63 oversized or malformed inputs.
  Result: 11 of 11 experiments pass, and the system returns to steady state after each rollback.
- **What makes it survive.** Exponential backoff with full jitter; a fallback chain to a smaller
  model; idempotent writes (`ON CONFLICT DO NOTHING`); checkpoints so a crashed run resumes; a
  deterministic report even when every model is down.
- **From finding to proof.** For each evidence-backed finding RepoRover generates an executable
  experiment: Playwright for client-state flaws (fill the storage quota, double-click submit, save
  from a stale tab, offline and slow network, malformed imports) and k6 for HTTP endpoints (spike,
  ramp, soak, oversized bodies). Each has a hypothesis, a steady-state check, abort conditions and
  a rollback.
- **Safe by construction.** Plans are generated, not run. Execution needs `--run`,
  `--i-own-this-target` and a localhost or local-sandbox URL; external URLs are refused, and each
  generated script refuses a non-local target on its own.
- **A learning loop.** If an experiment leaves a target in a failure state the catalogue does not
  describe, it is queued as a new discovery and can be promoted into the catalogue after a
  similarity check against existing entries.

## 5. Methodology and tech stack

- **Agents:** CrewAI, sequential process, five agents with per-task guardrails.
- **Model:** any OpenAI-compatible endpoint. Developed against Groq-hosted Llama 3.3 70B (with
  fallback to Llama 3.1 8B) and Google Gemini Flash-Lite; local models through Ollama by
  configuration.
- **Database:** Neon serverless Postgres. Tables for the failure catalogue, staged discoveries and
  analysis runs. `pg_trgm` trigram similarity decides whether a staged discovery is novel before
  promotion, with a Python `difflib` fallback.
- **Failure catalogue:** 143 entries in 13 categories, versioned as YAML and seeded idempotently;
  122 entries have a deterministic detector (92 regex, 27 absence, 13 manifest detectors in total).
- **Custom tool confinement:** list, read, search and outline tools restricted to the target
  directory, with a read ledger and hard budgets.
- **Chaos framework:** experiment model, three fault injectors (network, resource, data), a runner
  with a hard global timeout, seeded randomness and rollback in a `finally` block.
- **Engineering practice:** additive SQL migrations only; 100% of the test suite runs offline.

Numbers for this section (commands: `reporover catalog stats`, `reporover catalog validate`,
`uv run python -m unittest discover -s tests`): 143 catalogue entries, 216 detector self-test
samples checked on every validate, and 181 offline tests (180 pass; 1 is skipped on Windows because
it needs permission to create a symlink).

## 6. Verification status: read before presenting

| Claim | Status | How to check |
|---|---|---|
| Benchmark scores | Measured, on 3 small fixtures written and labelled by the authors | `reporover bench` |
| 0% hallucination rate | Measured for the detector stage: each cited location is re-opened and compared | `reporover bench` |
| Guardrails reject bad model output | Observed on one recorded live run (3 rejections) and in offline tests | `reporover demo` |
| Self-resilience, 11 of 11 | Measured against in-process stand-ins for the model and database, not live outages | `reporover chaos self` |
| Chaos scripts are generated | Done; the JavaScript parses under Node | `reporover demo`, then open `out/static_ledger/chaos` |
| Chaos scripts prove vulnerabilities | **Not yet run against an application.** k6 is not installed on the development machine | Run one against a local copy of the fixture before claiming a proven result |
| Local Ollama execution | **Supported by configuration, not tested.** | Set `LLM_BASE_URL` to an Ollama endpoint and run `reporover analyze` once |
| No cloud egress | True only with a local model **and** a local Postgres; the default setup uses hosted services | See 3.2 |
| Works on real repositories | One earlier live run on a real single-file app; no systematic evaluation | Say "tested on fixtures", not "validated in production" |
| "SAST misses these flaws" | An argument from how pattern scanners work; this project did not benchmark any SAST tool | Phrase as reasoning, do not quote a comparison figure |

Suggested honest one-liners:
- "Every finding in the report cites the line that proves it, or it is labelled a hypothesis."
- "On our fixtures: 90% precision, 85% recall, and no cited location that failed re-verification."
- "We ran chaos experiments on our own tool first: 11 of 11 passed."
