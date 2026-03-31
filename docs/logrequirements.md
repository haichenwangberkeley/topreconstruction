# log.md Requirements

This document defines the required structure and content for the project’s ongoing session log (`log.md`). The intent is to create a clear, human-readable history of what was attempted, what changed, what was tested, and where outputs can be found.

---

## 1) File-level rules

- **Single file:** Maintain one `log.md` at the repository/project root (unless the project explicitly requires another location).
- **Append-only:** Add new entries to the **top** of the file (reverse chronological order). Do not rewrite past entries except to fix obvious typos; if a correction is substantive, add a new note in a new entry.
- **Human-first:** Prefer concise, plain language. Avoid excessive jargon.
- **Linkable paths:** Whenever you mention a file, directory, or artifact, include a path relative to the repository root (or an absolute path if outside the repo).
- **Actionable detail:** Record enough detail that someone else (or you later) can reconstruct what happened and where to look.

---

## 2) Timestamp requirements (mandatory)

Every log entry **must begin** with a timestamp line that includes:
- Day of week
- Month name
- Day of month
- Year
- Local time (24-hour)
- Timezone

**Format (required):**
`YYYY-MM-DD (DOW) HH:MM TZ`

**Example:**
`2026-02-25 (Wed) 14:37 America/Los_Angeles`

If the session spans multiple short updates, each update still gets its own timestamp line.

---

## 3) Entry template (mandatory sections)

Each entry must include the sections below **in this order**. If a section has nothing to report, write `N/A` (do not omit the section).

### 3.1 Objective (conceptual, non-technical)
Describe the goal in conceptual terms—what you are trying to accomplish and why—without implementation detail.

**Guidelines:**
- Use “what/why” language, not “how.”
- Avoid naming libraries, frameworks, or algorithms unless essential to the concept.
- Keep it short (2–6 sentences).

**Examples:**
- “Improve the reliability of the training workflow so results are reproducible and easier to compare across runs.”
- “Make it easier to trace how datasets were produced and what assumptions were applied.”

---

### 3.2 Work summary
A short narrative of what you did in this session.

**Guidelines:**
- 3–10 bullet points.
- Focus on decisions, outcomes, and progress.
- Note any important constraints or assumptions introduced.

---

### 3.3 Changes to code and configuration
Document what was created, modified, renamed, moved, or removed.

#### A) Files created
List new files with:
- Path
- One-line purpose

#### B) Files modified
List changed files with:
- Path
- What changed (high level)
- Why it changed (one line)

#### C) Files renamed/moved/removed
List structural changes with:
- Old path → new path (or removed)
- Reason

**Rules:**
- Mention scripts explicitly (anything runnable: `.py`, `.sh`, `.ipynb`, workflow YAML, etc.).
- Mention configuration changes explicitly (YAML/JSON/TOML/INI, env files, CI configs).

---

### 3.4 Commands and runs executed
Record what you ran during the session.

Include:
- Command(s) (exact copy/paste when feasible)
- Working directory
- Key runtime parameters or inputs
- Approximate runtime (optional, if useful)

**Rule:** If a command is long, include it in a fenced code block.

---

### 3.5 Tests and validation
Describe what checks were performed and their outcomes.

Include:
- What was tested/validated
- Where results can be found
- Outcome summary (pass/fail/partial)
- Any known limitations or follow-ups

**Examples:**
- Unit/integration tests
- Sanity checks (plots, small-sample runs, schema checks)
- Smoke tests (CLI runs end-to-end)

---

### 3.6 Output artifacts
For any produced outputs (figures, tables, logs, models, checkpoints, reports), list:

- Artifact type (e.g., “plot”, “model checkpoint”, “parquet file”, “report”)
- Path(s)
- How it was produced (reference the command/run in §3.4)
- Notes on interpretation (1–3 bullets)

**Rules:**
- Prefer stable directories (e.g., `outputs/`, `artifacts/`, `reports/`) when possible.
- If outputs are outside the repo, state the absolute path and the storage context (local, cluster scratch, etc.).

---

### 3.7 Issues, surprises, and decisions
Capture anything that could matter later:

- Errors encountered (include the error message excerpt if short; otherwise point to a log file)
- Unexpected behavior
- Decisions made and rationale
- Risks or uncertainties

---

### 3.8 Next steps
List actionable next steps (3–8 bullets). Each bullet should be concrete and testable.

---

## 4) Recommended formatting conventions

- Use `- [ ]` checkboxes for next steps when helpful.
- Use code fences for commands and short snippets:
  ```bash
  # example
  python scripts/run_pipeline.py --config configs/dev.yaml