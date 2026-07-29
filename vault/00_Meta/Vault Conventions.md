# Vault Conventions

Last reviewed: 2026-07-26

## Authority

Use this order when sources disagree:

1. live code, Git state, tests, and regenerated results;
2. `01_Project/Current State.md` and
   `05_Handoff/Handoff - Latest.md`;
3. decision and algorithm notes;
4. dated historical notes.

Use evidence labels where ambiguity matters:

- **Confirmed**: supported by live evidence or a direct user statement;
- **Hypothesis**: plausible but not adequately verified;
- **Superseded**: retained because it explains history but no longer current.

## Note ownership

- `01_Project/Project Brief.md`: stable purpose, scope, workflow, and code map.
- `01_Project/Current State.md`: currently verified technical snapshot.
- `02_Algorithm/`: durable scientific and algorithmic interpretation.
- `03_Decisions/Decision Log.md`: durable decisions and rationale.
- `04_Sessions/`: new dated historical evidence when a session merits it.
- `05_Handoff/Handoff - Latest.md`: concise next actionable state, risks, and
  verification entry points, plus a structured `Portfolio impact` section.

Link to the note that owns a detail rather than repeating the same narrative.

## Cross-session synchronization

Project agents update this vault automatically after durable milestones. They
do not edit the central second brain.

At each meaningful checkpoint, refresh the latest handoff with:

- `Central update needed: Yes` or `No`;
- portfolio-level changes, limited to status, priority, deadline, next
  decision, or research significance;
- a one-to-three-sentence sync summary.

The central agent uses that section to reconcile project work asynchronously.
Skip vault updates for transient exploration, unchanged state, and explicitly
read-only work.

## Legacy material

This vault predates the shared architecture. Its dated Markdown notes and image
attachments remain at the vault root to preserve working links and history.
Treat those notes as session history. Do not append later work to them or use
their task lists as the current project plan.

Local Obsidian workspace state under `.obsidian/` is ignored. Project memory
and attachments remain version-controlled subject to the repository's normal
data and artifact policies.
