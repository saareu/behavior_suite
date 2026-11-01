# D0 — Shared Workspace Architecture

**Status:** **Shared workspace MVP closed.** D0-A, D0-B, D0-D1, and D0-D2 are **complete**. **D0-C is deferred** (not implemented). Ready for S10 gallery implementation against the shared analysis target.  
**Date:** 2026-11-01 (coding + MVP closure same date)  
**Baseline:** Behavior Suite **v0.1.1**; S1–S10 MVP lab-validated.  
**Parent roadmap:** [`post_v011_s10_review_and_visualization.md`](post_v011_s10_review_and_visualization.md)  
**D0-A/B code:** `src/workspace/` (+ `tests/workspace/`).  
**D0-D1 code:** `src/ui/workspace/` shell + MainWindow wiring (+ `tests/ui/test_workspace_gui.py`).  
**D0-D2 code:** page Browse write-back + `analysisTargetChanged` propagation (+ `tests/ui/test_workspace_d0_d2_integration.py`). No science schema changes, no automatic launch restore (W4(c)), no D0-C subsession science processing.

**Legend**

| Mark | Meaning |
| --- | --- |
| **Fact** | Observed in current repository / contracts / lab trees |
| **Recommendation** | Proposed design pending operator approval; not binding |
| **Operator decision** | Must not be silently decided by implementation |
| **Reuse** | Existing component to extend |
| **New** | Infrastructure that does not exist today |

---

## 0. Executive summary

Today a laboratory “project” and “session” are the **same directory** (must contain `preprocess/`). Shared GUI state is an in-memory `MainWindow._session_root` hint with a Browse write-back gap. Subsystem runs are discovered under fixed sibling folders; provenance lives in per-run manifests. There is no Project → Session nesting, no subsession model, and no durable workspace catalog.

D0 introduces a **catalog-referenced workspace model** and a **shared `WorkspaceContext`** so S1–S10, gallery, and PI report share one analysis target — without rewriting immutable science trees, without globalizing run selection, and without inventing scientific provenance.

Recommended increments:

```text
D0-A  Core schema + WorkspaceContext controller
D0-B  Existing-session discovery / registration / migration
D0-C  Optional subsession support
D0-D  S1–S10 GUI integration + laboratory acceptance
```

Gallery (A) and PI report (B) may proceed against an analysis-target root + explicit package/review pins; they must not embed GUI-only workspace state into published manifests.

---

## 1. Current-state audit (fact)

### 1.1 Identity today

| Concern | Behavior |
| --- | --- |
| Project | `ProjectService.create_project` / `open_project` → directory + `preprocess/` (`src/project/`) |
| Session | Same path as project; GUI calls it `session_root` |
| S1 | Singleton fixed files under `preprocess/` (overwrite-in-place on re-prepare) |
| S2–S10 | Versioned run dirs under subsystem dirnames |
| Shared GUI | `MainWindow._session_root` in-memory; S1 wizard + some shell Browse write it |
| Page Browse | S2/S4–S10 typically call page `set_session` **without** writing `_session_root` |
| Persistence | QSettings stores S2/S7 tool paths only — not last session |
| Run selection | Per-controller `selected_*_run_id` (scientifically correct) |
| Subsession | Does not exist |

### 1.2 Laboratory session tree (fact)

```text
{session_root}/
  preprocess/                    # S1 singleton
  pose_inference/*/              # S2  {model_id}__{timestamp}
  tracking_correction/*/         # S3  {profile_id}__{timestamp}
  pose_finalization/*/           # S4  pose_finalization__{timestamp}
  blob_features/*/               # S5  blob_features__{timestamp}
  combined_features/*/           # S6  combined_features__{timestamp}
  behavioral_inference/*/        # S7  behavioral_inference__{timestamp}
  behavioral_arbitration/*/      # S8  behavioral_arbitration__{timestamp}
  results_package/*/             # S9  results_package__{timestamp}
  behavioral_review/*/           # S10 behavioral_review__{timestamp}
  results_exports/*/             # S9 media/CSV jobs (optional)
```

Eligibility is digest/provenance-based from S4/S5 upward; S10 adds `CLASS_STALE_PARENT` when parent S9 pins fail. Historical `run_meta.status` is not rewritten by discovery.

### 1.3 Who owns session selection today (fact)

| Surface | Browse/Open | Updates `_session_root`? | Soft/hard lock notes |
| --- | --- | --- | --- |
| S1 wizard / `ProjectPage` | Create/Open | **Yes** (`session_changed`) | Wizard disables steps while preprocess runs |
| S2 `PoseInferencePage` | Browse → `set_session` | **No** | Hard-lock nav while S2 task runs |
| S3 shell | MainWindow browse | **Yes** | Hard-lock with S2/S4 |
| S4 page | Browse → `set_session` | **No** (shell browse yes) | Hard-lock while S4 task runs |
| S5–S10 pages | Browse → controller `set_session` / `prepare_session` | **No** | Soft-nav: page cannot change its own session mid-task; MainWindow soft-guards different-session push |
| S10 | `prepare_session` → `close_session(release_lock=True)` | Via MainWindow only if remembered | Draft lock released on session rebind |

Per-page science selection (preserve):

| Controller | Selection fields |
| --- | --- |
| PoseInference | `selected_run_id` |
| PoseFinalization | `selected_s3_run_id`, `selected_s4_run_id` |
| BlobFeatures | `selected_s5_run_id` |
| CombinedFeatures | `selected_s4_run_id`, `selected_s5_run_id`, `selected_s6_run_id` |
| BehavioralInference | `selected_s6_run_id`, `selected_s7_run_id` |
| BehavioralArbitration | `selected_s7_run_id`, `selected_s8_run_id` |
| ResultsPackaging | `selected_s8_run_id`, `selected_s9_run_id` |
| BehavioralReview | `selected_s9_run_id`, `selected_review_run_id` (+ discovery/open generations) |

---

## 2. Workspace data model (recommendation)

### 2.1 Hierarchy

```text
Project
  └── Session
        ├── [no subsession]  → analysis target = Session
        └── Subsession?      → analysis target = Subsession (where supported)
              └── Subsystem Runs (S1 binding + S2–S10 run refs)
```

**Rules**

1. A **Session without subsessions** remains a valid analysis target (covers all current lab data).
2. A **Subsession** may be selected as the analysis target where supported (see §3).
3. **Subsystem runs remain the scientific identity unit** — catalog entries *reference* them; they do not replace `run_id` / digests.
4. The catalog is **sidecar metadata**. It must not rewrite science trees, journals, or published manifests.

### 2.2 Stable identifiers

| Entity | Stable id | Path / location | Parent |
| --- | --- | --- | --- |
| **Project** | `project_id` (UUID string, catalog-assigned) | `root_path` (directory; may equal a legacy session path) | none |
| **Session** | `session_id` (UUID) | `session_root` (directory that owns or groups analysis trees) | `project_id` |
| **Subsession** | `subsession_id` (UUID) | `analysis_root` (directory) and/or temporal bounds | `session_id` |
| **Analysis target** | `analysis_target_id` | Discriminated: `session:{session_id}` or `subsession:{subsession_id}` | derived |
| **Subsystem run** | Existing `run_id` + absolute `run_dir` | Under analysis root subsystem dirnames | analysis target + subsystem |

**Compatibility alias (fact → recommendation):** today’s validated project/session directory becomes a **Session** whose default analysis target is itself. For migration, that Session is registered under a Project that is either:

- **Legacy self-project:** `project.root_path == session.session_root` (1:1; zero folder moves), or  
- **Parent project:** operator-approved grouping under a parent directory / catalog-only membership (**Operator decision W1**).

### 2.3 Catalog location and schema sketch

**Recommendation:** durable JSON catalog, not a database (roadmap “project database not justified” remains true).

```text
# Preferred for multi-session projects
{project_root}/.behavior_suite/workspace_catalog_v1.json

# Legacy self-project (session ≡ project)
{session_root}/.behavior_suite/workspace_catalog_v1.json
```

Illustrative schema (names normative only after acceptance):

```text
workspace_catalog_v1:
  schema_version: "workspace_catalog_v1"
  catalog_id: uuid
  created_utc, updated_utc
  behavior_suite_version_written

  project:
    project_id, display_name, root_path
    registration_mode: auto | manual
    notes?

  sessions[]:
    session_id, display_name, session_root
    registration_mode: auto | manual
    legacy_self_project: bool
    has_preprocess_dir: bool
    last_seen_utc, path_status: ok | missing | relocated_candidate
    fingerprint:
      preprocess_present: bool
      prepared_video_sha256?: str   # from prepare_meta / file digest if available
      prepare_meta_mtime?: str
    subsessions[]: [] | SubsessionRecord[]
    discovered_runs_summary?:   # optional cache; rebuildable
      by_subsystem: { s2: [{run_id, run_dir, classification?, completed_at?}], ... }

  SubsessionRecord:
    subsession_id, display_name
    kind: temporal_segment | separate_recording | hybrid
    preprocess_binding: shared_parent | independent | unbound
    analysis_root?: Path          # required when independent or when runs live here
    temporal_bounds?:             # prepared and/or raw frame indices; dense bounds only if sync exists
      domain: prepared | raw_source
      start_idx: int
      end_idx_exclusive: int | null
    source_recording?:
      raw_video_path?, sync_source_path?, notes?
    shared_preprocess_ref?:
      parent_session_id
      prepared_video_sha256
      prepared_sync_sha256?
    path_status, last_seen_utc
    notes?
```

**Idempotent re-scan:** re-running discovery updates `last_seen_utc`, `path_status`, and rebuildable run summaries; it does **not** mint new ids for paths already registered (match on resolved path, then digest fingerprint).

### 2.4 Analysis-target resolution

```text
resolve_analysis_target(context) -> AnalysisTarget
  AnalysisTarget:
    analysis_target_id
    kind: session | subsession
    discovery_root: Path     # root passed to existing summarize_*_project scanners
    preprocess_root: Path | None
    preprocess_binding: shared_parent | independent | session_default | unbound
    temporal_filter: TemporalBounds | None   # view/filter only unless independent pipeline
    session_id, subsession_id?, project_id
```

**Hard rule:** existing S2–S10 discovery modules continue to take a **filesystem root**. D0 adds a resolver in front; it does not fork ten scanners.

---

## 3. Subsession semantics (recommendation)

### 3.1 What a subsession can mean

| Kind | Meaning | Typical preprocess | May host S2–S10 runs? |
| --- | --- | --- | --- |
| **Temporal segment** | Index/time window over an existing recording / prepared video | Often **shared_parent** for inspection; **independent** if re-trimmed S1 is required for new science | Independent tree **or** filter-only view |
| **Separate source recording** | Another raw video (and usually sync) in the same experimental session | **independent** S1 required | Yes, under its `analysis_root` |
| **Hybrid** | Separate recording **and** a declared temporal window | **independent** | Yes |

### 3.2 Shared vs independent synchronization / preprocessing

| Binding | Sync / S1 | Scientific implication |
| --- | --- | --- |
| `session_default` | Session’s singleton `preprocess/` | Current v0.1.1 behavior |
| `shared_parent` | Reuses parent session prepared video + sync digests | Allowed for **read/filter/review views** only unless a future windowed-pipeline contract exists |
| `independent` | Own `preprocess/` under subsession `analysis_root` | Required for any new S2–S9 science that must not inherit full-session S1 |
| `unbound` | No S1 yet | Registration placeholder; discovery shows no eligible S1 handoff |

**Do not assume** all subsessions can reuse a parent’s S1 outputs.

Reasons (fact-backed):

- S1 is a **singleton overwrite** under `preprocess/` today — not versioned `preprocess__*` runs.
- Downstream eligibility pins prepared video / sync digests; a different trim, crop, mask, or sync source is a different scientific lineage.
- Many-to-one dense↔prepared mapping and S9 `sync_index` are authored for one prepared timeline.

### 3.3 Supported operations by binding (recommendation)

| Operation | Session (no subsession) | Subsession `shared_parent` | Subsession `independent` |
| --- | --- | --- | --- |
| Browse / open in GUI | Yes | Yes | Yes |
| Run S1 | Session preprocess | **Forbidden** (would mutate parent) — create independent or new session | Yes, on `analysis_root/preprocess` |
| Run S2–S9 science | Yes | **Deferred** until windowed-pipeline contract (**Operator decision W3**) | Yes under `analysis_root` |
| Open S9/S10 / gallery on full package | Yes | Filter bouts/playhead by temporal bounds (view) | Yes on packages under that root |
| PI report | Yes | Allowed only with explicit filter + manifest note of bounds | Yes |

### 3.4 On-disk layout for independent subsessions (recommendation)

```text
{session_root}/
  preprocess/                          # session-default S1 (existing)
  pose_inference/ ...                  # existing session runs
  .behavior_suite/workspace_catalog_v1.json
  subsessions/
    {subsession_slug}/                 # analysis_root
      preprocess/
      pose_inference/
      ...
      results_package/
      behavioral_review/
```

**Alternative (operator-optional):** register an existing sibling folder as a subsession `analysis_root` without moving it (manual registration) — catalog points at the path.

### 3.5 Sessions without subsession metadata

Migrated legacy sessions:

- `subsessions: []`
- `analysis_target_id = session:{session_id}`
- Discovery root = `session_root` (unchanged)
- No behavior change for S1–S10 scanners

### 3.6 Artifact-level contracts beyond the catalog?

| Need | Verdict |
| --- | --- |
| Register / navigate / filter existing data | **Catalog only** |
| Produce new runs under independent subsession | Discovery root change sufficient for D0; **optional** `workspace` provenance block on **new** `run_meta` / package manifests (**Recommendation:** add thin optional fields in D0-C for new writes only) |
| Rewrite historical run_meta with subsession ids | **Forbidden** |
| Temporal shared-parent science pipeline | **New artifact contract later** — out of D0 unless W3 approved |
| Gallery / PI report provenance | Continue digest-pinned package/review ids; if temporal filter applied, record filter in **report/gallery export** manifest, not by mutating S9 |

**Recommendation:** D0-C does **not** require changing S8 ethogram schemas. Optional new-write provenance:

```text
workspace_ref_v1:   # optional block on NEW run_meta / package_manifest only
  project_id, session_id, subsession_id?
  analysis_target_id
  preprocess_binding
```

---

## 4. Existing-data migration strategy (recommendation)

### 4.1 Goals

- Automatically register existing project/session folders.
- Detect previously completed subsystem runs via **existing** discovery modules.
- Support manual registration for ambiguous / nonstandard paths.
- Detect duplicates; handle relocated/missing paths.
- Idempotent re-scanning.
- Operator review of proposed associations before commit-to-catalog.
- Catalog **references** results; never rewrites them.
- Safe rollback / catalog rebuild.

### 4.2 Automatic registration algorithm

```text
scan_candidate(path):
  if path/preprocess/ is dir → candidate Session (legacy project≡session)
  elif path has children with preprocess/ → candidate Project with child Sessions
  else → ambiguous (manual)

propose_registration(candidate):
  mint or reuse ids (match resolved path in existing catalogs)
  run summarize_*_project for S2–S10 under session_root
  summarize S1 handoff presence/digests
  emit RegistrationProposal for operator review

commit_proposals(selected[]):
  write/update workspace_catalog_v1.json
  never move/rename science dirs in D0-B
```

### 4.3 Duplicate detection

| Signal | Action |
| --- | --- |
| Same resolved `session_root` already in catalog | Update `last_seen`; do not duplicate session |
| Two catalog entries, same path after normalize | Merge proposal (keep older `session_id`) |
| Same `prepared_video_sha256` under different paths | Flag **possible copy/relocation**; operator chooses primary vs alias |
| Same display name, different paths | Allow; show both |

### 4.4 Relocated / missing paths

| `path_status` | Meaning | GUI behavior |
| --- | --- | --- |
| `ok` | Path exists and passes gate | Normal open |
| `missing` | Path gone | Show in catalog as unavailable; do not auto-delete entry |
| `relocated_candidate` | Operator pointed Browse at a folder whose fingerprint matches a missing entry | Propose rebind `root_path` / `session_root` |

**Rebind** updates catalog paths only. Science manifests may still contain absolute historical paths — eligibility already uses digests where required; do not rewrite those files in D0.

### 4.5 Manual registration

Use when:

- `preprocess/` missing but downstream runs exist (browse-only / recovery).
- Nonstandard nesting.
- Grouping multiple existing session folders under one Project without moving them (**catalog membership**, paths stay put).

Gate options (**Operator decision W2**):

- Default open gate remains `validate_project_root` (`preprocess/` required) for **analysis**.
- Catalog may still **list** incomplete sessions with warnings.

### 4.6 Operator review UX (recommendation)

Before writing catalog:

1. Show proposed Project/Session table (path, fingerprint, run counts by subsystem, warnings).
2. Operator confirms / edits display names / parent project.
3. Persist only confirmed rows.
4. Re-scan is always available; proposals that match existing ids are no-ops.

### 4.7 Rollback / rebuild

| Strategy | How |
| --- | --- |
| Soft rollback | Delete or rename `.behavior_suite/workspace_catalog_v1.json`; app falls back to legacy path-only open (v0.1.1 behavior) |
| Rebuild | `rescan → propose → commit` regenerates run summaries; stable ids preserved when paths match |
| Backup | On each commit, write `workspace_catalog_v1.json.bak` / timestamped sibling under `.behavior_suite/` |

**Guarantee:** removing the catalog never deletes science artifacts.

### 4.8 Representation of sessions without subsession metadata

Exactly as §3.5: empty `subsessions`, session-level analysis target, identical discovery roots to v0.1.1.

---

## 5. Shared application context (recommendation)

### 5.1 `WorkspaceContext` (New)

Central GUI + headless-friendly controller (Qt object owning signals; pure model underneath):

```text
WorkspaceContext:
  # Active selection
  project_id: str | None
  session_id: str | None
  subsession_id: str | None          # optional
  analysis_target_id: str | None     # session:* or subsession:*
  discovery_root: Path | None        # resolved filesystem root for scanners
  catalog_path: Path | None

  # Signals
  projectChanged(project_id, root_path)
  sessionChanged(session_id, session_root)
  subsessionChanged(subsession_id | None)
  analysisTargetChanged(AnalysisTarget)
  contextCleared()

  # Persistence (QSettings) — remember-only
  recent_targets: list[{project_id?, session_id, paths, display}]
  last_analysis_target_id?
  # NEVER persist selected science run ids

  # API
  open_session_path(path, *, register_if_missing=True)
  set_project / set_session / set_subsession / clear_subsession
  resolve_analysis_target() -> AnalysisTarget
  begin_navigation_guard() / note for soft-hard locks
```

Replaces ad-hoc `MainWindow._session_root` as the **source of truth**; MainWindow becomes a subscriber/orchestrator.

### 5.2 Context-change integrity rules

| User action | Required behavior |
| --- | --- |
| **Switch project** | Clear session/subsession/analysis target; clear page run selections that are session-scoped; reject if hard-locked; soft-nav pages keep running jobs bound to **old** discovery root until complete (do not retarget mid-job) |
| **Switch session** | Clear subsession; emit `sessionChanged` + `analysisTargetChanged`; S10 `close_session(release_lock=True)` when leaving a session that held a draft; bump discovery generations so in-flight results are rejected |
| **Switch / clear subsession** | Emit `subsessionChanged` + `analysisTargetChanged`; same job guards; do not silently keep S9/S10 pins from prior target if discovery_root or temporal filter changed |
| **Navigate during active job** | Preserve existing hard-lock (S2/S4) and soft-nav (S5–S10); changing analysis target on a page with a running task is **blocked** for that page |
| **Unfinished S10 draft** | Leaving analysis target releases draft lock (existing); draft remains on disk under old root; do not auto-reopen against a different target |
| **Publish after context change** | Publish path must use the draft’s pinned parent package + current editing session identity; if GUI context no longer matches the open draft’s `session_root` / package digests, require re-open or block with explicit error — **never silently reassign provenance** |

### 5.3 Stale asynchronous result rejection

**Reuse** existing generation / request-id patterns (`DiscoveryRequestIdentity`, S10 open identities, `GuiTaskRunner`).

Extend: every async discovery/open callback must include `analysis_target_id` (or discovery_root + generation). Mismatch → drop result.

### 5.4 Persisted selection

| Persist | Do not persist |
| --- | --- |
| Recent project/session paths + ids | `selected_*_run_id` |
| Last analysis target id (optional; **P11** / **W4**) | Ethogram buffers, draft contents |
| Catalog file on disk | Auto-picked “latest reviewed” |

Default recommendation aligned with **P11**: remember-only list; no forced auto-open unless operator approves W4.

### 5.5 Safe navigation summary

```text
hard lock (S2/S4 running): block all group/subsystem nav (existing)
soft nav (S5–S10 running): allow leaving page; block that page’s set_session/target change
context switch with open S10 draft: close_session(release_lock=True) then bind new target
context switch never mutates journals, packages, or run_meta
```

---

## 6. S1–S10 integration impact

### 6.1 Consumption model

| Layer | Change |
| --- | --- |
| Pages | Browse/Open call `WorkspaceContext.open_session_path` / `set_subsession` instead of page-only roots |
| Controllers | `set_session(discovery_root)` still works; MainWindow passes `context.discovery_root` |
| Discovery modules | Unchanged signatures; root comes from resolver |
| Run selection UI | Remains per-page; eligibility unchanged |
| Status bar | Show project / session / subsession / analysis-target display names |

### 6.2 Per-subsystem impact matrix

| Sub | Owns selection today | Consumes shared context how | Keeps local |
| --- | --- | --- | --- |
| S1 | Project create/open | Emits context on create/open; S1 writes session-default preprocess | Wizard step state |
| S2 | Page Browse | Subscribe `analysisTargetChanged` → `set_session` | `selected_run_id`, tool paths |
| S3 | Shell + handoff | Same discovery_root from context | Review session / accepted run |
| S4 | Page / shell Browse | Same | `selected_s3_run_id`, `selected_s4_run_id` |
| S5 | Page Browse | Same + soft-nav guard uses context id | `selected_s5_run_id` |
| S6 | Page Browse | Same | S4/S5/S6 selections |
| S7 | Page Browse | Same | S6/S7 selections + model registry path |
| S8 | Page Browse | Same | S7/S8 selections |
| S9 | Page Browse | Same | S8/S9 + export job selection |
| S10 | Page Browse | Same; draft close on target change | S9/review selection, playhead, edit mode |

### 6.3 Source eligibility

Unchanged scientifically:

- Explicit automated vs reviewed (**B4** lab checklist).
- No silent “latest reviewed” global bind.
- Stale-parent still blocks S10 publish.

### 6.4 Gallery / PI report coupling (confirm)

| Concern | Rule |
| --- | --- |
| GUI navigation | May read `WorkspaceContext` for default discovery root |
| Gallery query / cards | Bound to **opened** package/draft identity + digests (`GallerySource`) |
| PI report manifest | Pins `source_type`, package/review run ids, digests, optional temporal filter — **not** `WorkspaceContext` object state |
| Reopen published report | Uses bundle `manifest.json` alone; no live GUI context required |
| Scientific contracts | Remain S9/S10 artifacts; workspace catalog is not a science authority |

---

## 7. D0 implementation sequence

### D0-A — Core workspace schema and context controller (**done**)

| Item | Content |
| --- | --- |
| **Affected files (landed)** | **New:** `src/workspace/{__init__,errors,models,catalog,resolve,policy,context,qt_context}.py`. **Edit:** `pyproject.toml` (package/coverage/mypy/ruff first-party). **Tests:** `tests/workspace/`. **Docs:** this file. **Not edited (deferred D0-D):** `src/ui/main_window.py`, `src/project/*`, S1–S10 controllers |
| **Catalog ownership** | `{project.root_path}/.behavior_suite/workspace_catalog_v1.json` via `catalog_path_for_root` |
| **Contracts / APIs** | See §13 (D0-A contract summary) |
| **Backward compatibility** | No catalog required; `open_session_path` builds ephemeral legacy self-project; `ProjectService` / science schemas untouched |
| **Tests / acceptance** | `tests/workspace/` covers identity, hierarchy, session-only, optional subsessions, serialize/reopen, atomic write, duplicates, missing paths, context switch, safe restore, unsupported versions, job stale rejection |
| **Risks / deps** | W4 launch auto-restore still operator-gated; Qt bridge unused until D0-D. **Dep:** none |

### D0-B — Existing-session discovery, registration, migration (**done**)

| Item | Content |
| --- | --- |
| **Affected files (landed)** | **New:** `src/workspace/{scan,duplicate,migrate}.py`. **Edit:** `src/workspace/__init__.py`, this document. **Tests:** `tests/workspace/test_migrate.py`. **Not edited (deferred D0-D):** MainWindow, registration GUI page, S1–S10 controllers |
| **Contracts / APIs** | See §14 (D0-B contract summary): `scan_candidate`, `preview_registration` / `commit_registration`, `preview_rebind` / `commit_rebind`, `reopen_registered_session`, `rescan_catalog` |
| **Backward compatibility** | No moves/renames of science dirs; catalog optional; delete catalog = full rollback; ephemeral path open still works without registration |
| **Tests / acceptance** | Unit coverage for legacy self-project, multi-session parent, incomplete, idempotent rescan, duplicates, ambiguous hierarchy, rebind, missing paths, validation. E-session read-only smoke: discovery + temp catalog commit + reopen; prepare_meta / prepared_sync / package_manifest SHA256 unchanged |
| **Operator defaults applied** | **W4(c)** remember-only (no launch auto-activate). **W2(b)** incomplete sessions listable. Auto-migrate default remains legacy self-project when `preprocess/` is local |
| **Risks / deps** | Absolute paths in old manifests after relocation (document; don’t rewrite). **Dep:** D0-A |

### D0-C — Optional subsession support (**deferred — not implemented**)

| Item | Content |
| --- | --- |
| **Status** | **Deferred.** Catalog schema already allows empty `subsessions[]`; no subsession science processing, no `workspace_ref_v1` writers, no independent/shared-parent pipeline in this MVP. |
| **Affected files (when scheduled)** | **Edit:** `src/workspace/models.py`, `resolve.py`, catalog schema; MainWindow/status bar; controllers only as needed for target display. **New:** `subsessions/` path helpers; tests for shared vs independent resolve. **Optional new-write:** thin `workspace_ref_v1` writers in run_meta builders (S2–S10) — only if approved |
| **Contracts / APIs** | `SubsessionRecord`; preprocess_binding rules §3; temporal filter attach to AnalysisTarget; forbid S1 mutate on `shared_parent` |
| **Backward compatibility** | Sessions with `subsessions: []` identical to pre-D0-C; no rewrite of old artifacts |
| **Tests / acceptance** | Independent subsession discovery_root isolation; shared_parent blocks S1 write API; temporal filter recorded on analysis target; legacy session still opens |
| **Risks / deps** | Operators assuming shared S1 implies independent science lineage. **Dep:** D0-A/B. **Operator:** W3 (whether shared-parent may ever run windowed S2–S9 in this phase — default **no**) |
| **Extra artifact contract?** | Catalog sufficient for navigation; optional `workspace_ref_v1` on **new** writes only; no S8 schema change |

### D0-D1 — Workspace GUI shell + navigation infrastructure (**done**)

| Item | Content |
| --- | --- |
| **Affected files (landed)** | **New:** `src/ui/workspace/{__init__,recent_store,shell_controller,selector_bar,registration_dialog,rebind_dialog}.py`. **Edit:** `src/ui/main_window.py`, `src/ui/tasks.py` (`GuiTaskKind` workspace scan kinds), `src/workspace/policy.py` (`ContextChangeKind` export), this document. **Tests:** `tests/ui/test_workspace_gui.py`. **Not edited (deferred D0-D2):** per-page Browse write-back on S2/S5–S10 pages, broad controller `set_session` subscription to `analysisTargetChanged` |
| **Contracts / APIs** | See §15 (D0-D1 contract summary) |
| **Backward compatibility** | Legacy `_session_root` retained as shell mirror of `WorkspaceContext.discovery_root`; per-page `selected_*_run_id` untouched; science schemas untouched |
| **Tests / acceptance** | MainWindow init + remember-only launch; ephemeral browse; registration preview/confirm; background stale rejection; rescan idempotency; rebind UUID retention; missing-path recovery; unsafe switch during jobs; no scientific writes on preview; soft/hard nav preservation |
| **Risks / deps** | Dual-write gap closed in D0-D2. **Dep:** D0-A/B |

### D0-D2 — S1–S10 GUI integration and laboratory acceptance (**done**)

| Item | Content |
| --- | --- |
| **Affected files (landed)** | **New:** `src/ui/workspace/page_binding.py`, `tests/ui/test_workspace_d0_d2_integration.py`. **Edit:** `src/ui/main_window.py` (authoritative propagation), `src/ui/workspace/__init__.py`, S2/S4–S10 pages (Browse/Open write-back), `src/ui/controllers/behavioral_review_controller.py` (publish/target match + `session_root` on publish identity), this document. **Not edited:** science schemas, discovery scanners, D0-C subsession writers |
| **Contracts / APIs** | See §16 (D0-D2 contract summary) |
| **Backward compatibility** | Explicit per-page `selected_*_run_id` preserved on same-root sync; eligibility unchanged; ephemeral path open still works without catalog; legacy `_session_root` remains a mirror only |
| **Tests / acceptance** | `tests/ui/test_workspace_d0_d2_integration.py` + prior `test_workspace_gui.py`; E/F lab smoke read-only (fingerprints unchanged) |
| **Risks / deps** | Soft discovery still blocks shell workspace switch until idle (D0-D1 policy). **Dep:** D0-D1 |

### Suggested coding order

```text
D0-A ──► D0-B ──► D0-D1 (shell + selector) ──► D0-D2 (S1–S10 write-back)  ✓
              └──► D0-C (deferred; session-only lab acceptance first)
                         └──► (gallery A / PI B consume context)
```

**Recommendation:** Workspace MVP closed; proceed to gallery/PI against shared `discovery_root`. D0-C only when multi-recording / segment workflows are scheduled.

---

## 8. Gallery / report architecture revisit

| Question | Answer |
| --- | --- |
| Can gallery consume workspace context? | **Yes** — for default `discovery_root` and display labels only |
| Does gallery scientific contract depend on GUI state? | **No** — `GallerySource` = pinned package/draft/review + digests + prepared video identity |
| Can PI report reopen offline? | **Yes** — bundle `data/manifest.json` + media; no catalog or `WorkspaceContext` required |
| Where do temporal subsession filters go? | In gallery query / `pi_report_manifest_v1.selection`, not in S9 science tables |
| Provenance authority | Package/review manifests and digests remain authoritative |

No change to recommended Increment A/B data contracts beyond optional `analysis_target_id` / temporal filter fields on **export/view** manifests.

---

## 9. P1–P14 decision matrix (revisit)

| ID | Topic | Proposed default | Scientific / usability implication | Blocks D0? | Blocks gallery? | Blocks PI report? | Recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **P1** | Default PI report source | Explicit chooser: automated S9 **or** reviewed S10 | Prevents silent provenance mix | No | No | Chooser UX | Keep |
| **P2** | Exclusive acceptance meaning | “Endorsed for analysis,” not pixel GT | Reporting language | No | Copy only | Copy only | Keep |
| **P3** | Bulk accept policy | Filtered unreviewed exclusive + confirm | Mis-accept risk if too broad | No | Yes (UX) | No | Keep |
| **P4** | Contact/SBS in PI report | Optional section; default exclusive-only | Channel independence | No | Optional mode | Section flag | Keep |
| **P5** | U6 vs new PI exporter | New PI module; S9 exporters stay automated | Avoids reopening frozen U6 | No | No | Module ownership | Keep |
| **P6** | U3 `uncertain` | Keep deferred | Triage vocabulary | No | Nice-to-have | No | Keep deferred |
| **P7** | U2 reciprocal tool | Keep deferred; UI sibling link only | Directional science | No | Link-only | No | Keep deferred |
| **P8** | Gallery preview pad | 0.5 s default; configurable | Preview context only | No | Default | Export pad | Keep |
| **P9** | Overlay export frame domain | Default prepared-fps; dense opt-in | Timing confusion if wrong | No | No | No (overlay C) | Keep |
| **P10** | Plateau HUD policy | Interactive playhead; offline last-in-plateau | Dense/prepared HUD truth | No | No | No (overlay C) | Keep |
| **P11** | Auto-open last session | Remember-only list; no forced auto-open | Surprising context on launch | **Partial** (persistence policy) | No | No | Keep; see **W4** if auto-open desired |
| **P12** | GIF in PI bundle | Explicit opt-in only | Encode cost / authority confusion | No | No | Flag | Keep |
| **P13** | Gallery disk preview cache | Yes, deletable, digest-keyed under session cache | Disk use; not science | No | Cache path | No | Keep; cache under analysis_root |
| **P14** | Accept-all ⇒ publish gate | Soft coverage meter only | Preserves R10 / no forced accept | No | Meter UX | Meter in report optional | Keep |

None of P1–P14 silently decide Project nesting or subsession preprocess binding — those are **W\*** below.

---

## 10. Decisions requiring operator approval

### 10.1 Workspace-specific — **D0-A finalized defaults**

| ID | Decision | Finalized for D0-A | Implementation note |
| --- | --- | --- | --- |
| **W1** | Project ↔ Session on disk | **Logical membership independent of physical nesting.** Catalog links `project_id` → sessions by id/path. D0-A ephemeral open uses legacy self-project (`root_path == session_root`). D0-B auto-migrate still defaults to (a); manual (c) and created parent (b) remain D0-B/D0-D UX. | `ProjectRecord` + `SessionRecord`; no folder moves |
| **W2** | Incomplete sessions (no `preprocess/`) | **(b)** List/register with warning; **block analysis open** when `require_preprocess=True` | `has_preprocess_dir`, `analysis_open_blocked_reason` |
| **W3** | Shared-parent new S2–S9 science | **(a)** Read-only / filter-view only this phase | `science_writes_allowed=False` for `shared_parent` |
| **W4** | Launch restore | **(c) Remember-only** — finalized for D0-B/D0-D. Application launch must not auto-activate a workspace, start processing, or resume an S10 draft. | Recent list only; no startup `restore_selection` |
| **W5** | S1 versioning | **(a)** Keep singleton `preprocess/` | Unchanged; catalog only references |
| **W6** | Optional `workspace_ref_v1` | Contract stub defined; **no writers** to run_meta in D0-A. Emit on new writes in D0-C **(b)**. | `WorkspaceRefV1`, `WorkspaceContext.to_workspace_ref_dict` |
| **W7** | Subsession slug / path | UUID = identity; slug = human path only. **(a)** `subsessions/{slug}/` helper for created; **(b)** manual arbitrary `analysis_root` allowed | `validate_subsession_slug`, `default_subsession_analysis_root` |

#### W4 finalized (D0-B)

Operator approved **W4(c) remember-only recent list** for D0-B:

- Startup / `WorkspaceContext()` does **not** auto-open.
- Application launch must not automatically activate a workspace, start processing, or resume an S10 draft.
- `propose_restore` + `restore_selection` remain available for **explicit** caller use (D0-D may expose recent-list UI only).

No other W1–W7 discrepancies block D0-B.

### 10.2 From parent roadmap (still open)

P1–P14 as in §9 — none auto-approved by this document.

---

## 11. Deliverable map (this document)

| # | Deliverable | Section |
| --- | --- | --- |
| 1 | Workspace data model | §2 |
| 2 | Subsession semantics | §3 |
| 3 | Existing-data migration strategy | §4 |
| 4 | Shared GUI context | §5 |
| 5 | S1–S10 integration impact | §6 |
| 6 | D0 implementation sequence | §7 |
| 7 | P1–P14 decision matrix | §9 |
| 8 | Decisions requiring operator approval | §10 |

---

## 12. MVP closure

**D0 shared workspace MVP is closed.** Increments:

| Increment | Status |
| --- | --- |
| **D0-A** | **Complete** — schema + `WorkspaceContext` |
| **D0-B** | **Complete** — discovery / registration / rebind |
| **D0-D1** | **Complete** — GUI shell + navigation infrastructure |
| **D0-D2** | **Complete** — S1–S10 Browse write-back + target propagation |
| **D0-C** | **Deferred** — optional subsession science processing **not** implemented |

MainWindow owns one shared `WorkspaceContext`. S1–S10 Browse/Open write through `WorkspaceShellController`. Target changes propagate with same-root short-circuits, run-selection invalidation, S10 draft release (idle only), and publish provenance gates. No scientific schema changes.

### Known limitations (accepted for this MVP)

| Limitation | Note |
| --- | --- |
| **No launch auto-restore** | W4(c) remember-only; launch does not activate a workspace, start processing, or resume an S10 draft |
| **Soft discovery navigation lock** | Soft page tasks (including S5–S10 discovery) block shell workspace switches until idle or cancelled |
| **Historical absolute paths after rebind** | Catalog rebind updates catalog paths only; science manifests may still embed pre-move absolute paths (eligibility remains digest-based) |
| **Subsession science processing deferred** | D0-C not implemented; session-level analysis targets only |

**Next step:** S10 gallery (Increment A) and PI report (Increment B) against shared `discovery_root` / pinned package+review identity. Schedule **D0-C** only when independent/shared-parent subsession science is required. Do not claim D0-C is implemented.

---

## 13. D0-A finalized contracts (normative for implementers)

### 13.1 Schema

```text
workspace_catalog_v1  (SCHEMA_VERSION)
  catalog_id: uuid
  created_utc, updated_utc
  behavior_suite_version_written?
  project: ProjectRecord { project_id, display_name, root_path, registration_mode, notes? }
  sessions[]: SessionRecord {
    session_id, display_name, session_root, registration_mode,
    legacy_self_project, has_preprocess_dir, path_status, last_seen_utc?,
    fingerprint, notes?,
    subsessions[]: SubsessionRecord {
      subsession_id, display_name, kind, preprocess_binding, slug?,
      analysis_root?, temporal_bounds?, path_status, last_seen_utc?, notes?
    }
  }
```

- Stable identity = UUID strings; slugs/display names are never identities.
- Empty `subsessions` ⇒ session is the analysis target (`session:{session_id}`).
- Catalog is sidecar-only under `.behavior_suite/`; atomic write + optional `.bak`; malformed → quarantine (`*.corrupt.*`); unsupported future `schema_version` → error, file preserved.

### 13.2 `WorkspaceContext` API (headless)

| Method / property | Role |
| --- | --- |
| `project_id`, `session_id`, `subsession_id`, `analysis_target_id`, `discovery_root`, `generation` | Active selection |
| `bind_catalog` / `load_from_path` / `save_catalog` / `clear` | Catalog lifecycle |
| `set_project` / `set_session` / `set_subsession` / `clear_subsession` | Selection; rejects cross-project session ids |
| `open_session_path` | Legacy path open → ephemeral self-project; optional `persist` |
| `propose_restore` / `restore_selection` / `recent_targets` | W4 remember + explicit gated restore |
| `add_listener` / `snapshot` | Context-change notifications |
| `evaluate_navigation` / `bind_job` / `is_stale_job_result` | Async GUI job safeguards (D0-A policy only) |
| `to_workspace_ref_dict` | W6 stub; does not write science artifacts |

Optional Qt adapter: `workspace.qt_context.create_qt_workspace_bridge` (not wired into MainWindow).

### 13.3 Job / editing safeguards (D0-A only)

| Rule | Behavior |
| --- | --- |
| Hard lock | `ContextChangeRequest.hard_locked` → `WorkspaceNavigationBlocked` |
| Soft page task | `page_task_running` → blocked for that change request |
| Active jobs | `JobBinding` retains original target/generation; never retarget mid-flight |
| S10 draft | Decision sets `release_s10_draft_lock=True` when `has_open_s10_draft`; actual `close_session` remains D0-D |
| Run selection | Remains per-controller; context never stores `selected_*_run_id` |

### 13.4 D0-B prerequisites — satisfied

1. **W4(c)** remember-only finalized (no launch auto-activate).
2. Headless scan/register/rebind/reopen landed (`scan` / `duplicate` / `migrate`); GUI Register/Rescan deferred to D0-D.
3. Commit proposals never move science dirs; catalog under `.behavior_suite/` (or explicit alternate path for smoke tests).
4. Incomplete-session listing (W2) supported in preview/commit; analysis open remains separately gated.
5. Historical run_meta untouched; optional `workspace_ref_v1` still waits for D0-C.

---

## 14. D0-B finalized contracts (normative for implementers)

### 14.1 Discovery APIs (`workspace.scan`)

| API | Role |
| --- | --- |
| `scan_candidate(path, *, discover_runs=True)` | Headless project/session layout scan. Returns `ProjectDiscovery` (`legacy_self_project` \| `parent_project` \| `ambiguous` \| `empty`) |
| `discover_session_artifacts(session_root)` | Read-only S1 handoff + S2–S10 run refs via existing `summarize_*` / S10 discovery |
| `build_session_fingerprint(session_root)` | Rebuildable fingerprint (`preprocess_present`, optional recorded `prepared_video_sha256`, `prepare_meta_mtime`) — never hashes large media during scan |

**Hard rules:** scanning never creates catalogs, never moves/renames science dirs, never launches scientific processing.

### 14.2 Preview / commit separation (`workspace.migrate`)

| API | Writes disk? | Notes |
| --- | --- | --- |
| `preview_registration(path, ...)` | **No** | Builds `RegistrationPreview` with proposals, duplicates, ambiguous associations, missing/relocated notes, and an in-memory `proposed_catalog` |
| `commit_registration(preview, *, operator_approved=True, approved_proposal_ids=..., catalog_path=...)` | **Yes** (catalog only) | Requires explicit `operator_approved=True`. Validates full catalog before save. Optional `catalog_path` override |
| `rescan_catalog(catalog_or_path)` | **No** (preview only) | Idempotent rediscovery; path match ⇒ `action=reuse` with preserved UUIDs |

Proposal actions: `create` | `reuse` | `update` | `skip`. Ambiguous hierarchy blocks **new** creates until associations are resolved; `reuse`/`update` of already-registered rows may still commit.

### 14.3 UUID preservation and idempotency

- Match key for rediscovery: normalized absolute `session_root`.
- Same path already in catalog ⇒ reuse `session_id` / `project_id` / `catalog_id`; refresh `last_seen_utc`, `path_status`, fingerprint; preserve `subsessions`.
- Duplicate path create is rejected (`WorkspaceDuplicateError`).
- Incomplete sessions may register (W2); analysis open still requires `preprocess/` when gated.
- Per-subsystem run identities remain the existing `run_id` / `review_run_id` values discovered under the session root (referenced in preview `run_counts` / `DiscoveredRunRef`; not rewritten into science manifests).

### 14.4 Relocation / rebind

| API | Role |
| --- | --- |
| `preview_rebind(catalog, *, session_id, new_session_root, rebind_project_root=...)` | Validate candidate path; preview affected paths; detect conflicts/duplicates; report fingerprint match |
| `commit_rebind(catalog, preview, *, operator_approved=True, allow_without_fingerprint_match=False, catalog_path=...)` | Catalog-only path update; retains stable UUIDs and subsession associations |

**Rules**

- Digest mismatch between registered fingerprint and candidate ⇒ conflict (no silent reinterpretation).
- Missing digest ⇒ warnings; commit requires `allow_without_fingerprint_match=True` after operator confirmation.
- Never rewrite historical science manifests that may still embed absolute paths.

### 14.5 Safe reopen and backward compatibility

| API | Role |
| --- | --- |
| `reopen_registered_session(catalog_or_path, session_id=..., *, require_preprocess=False, discover_runs=False)` | Resolve analysis root through catalog; optional rediscovery |
| `ephemeral_open_without_catalog(session_root)` | v0.1.1-compatible path open without registration |

- Catalog registration is **not** a new eligibility gate for science.
- Missing paths stay listed with `path_status=missing` and actionable `analysis_open_blocked`.
- Existing S1–S10 scanners continue to take the resolved `discovery_root`.

### 14.6 Laboratory acceptance evidence (D0-B)

| Check | Result |
| --- | --- |
| `tests/workspace/` (incl. D0-A) | Passing |
| E-session smoke (`test_real_e_session_smoke_read_only`) | Registers/reopens `test_cases/E_full_session_integration` via temp catalog; S2/S9/S10 runs discovered; prepare_meta / prepared_sync / package_manifest SHA256 unchanged before/after |
| Ruff / mypy on `src/workspace` | Clean |
| GUI / MainWindow | Not wired (D0-D) |

### 14.7 D0-D GUI integration prerequisites

| # | Prerequisite | D0-D1 | D0-D2 |
| --- | --- | --- | --- |
| 1 | Wire MainWindow Browse/Open → `WorkspaceContext` / `reopen_registered_session` | **Done** (shell selector + S1/`_remember_session` dual-write; `_session_root` mirrors discovery root) | Finish per-page Browse write-back |
| 2 | Register / Rescan operator UI (`preview` → confirm → `commit`) | **Done** | — |
| 3 | Relocate / Rebind dialog | **Done** | — |
| 4 | Remember-only recent list (W4(c)); no launch auto-restore | **Done** | — |
| 5 | Preserve per-page `selected_*_run_id` | **Done** (unchanged) | Must not clear on shared target sync incorrectly |
| 6 | Soft/hard navigation guards; S10 draft close on target change | **Done** (shell-level) | Page-level Browse guards stay; subscribe pages to target changes |

---

## 15. D0-D1 finalized contracts (normative for implementers)

### 15.1 MainWindow ownership

```text
MainWindow
  workspace: WorkspaceContext                 # single shared instance
  workspace_controller: WorkspaceShellController
  workspace_selector: WorkspaceSelectorBar
  _session_root: Path | None                  # legacy mirror of discovery_root
```

- One controller for the whole shell. Pages must **not** construct their own `WorkspaceContext`.
- Optional Qt bridge: `workspace_controller.bridge` (`create_qt_workspace_bridge`).
- Exposed: active project/session/subsession ids, `analysis_target` / `discovery_root`, registration status (`registration_status()`), generation.

### 15.2 GUI state transitions

```text
Launch
  load recent via QSettings → context.recent_targets
  do NOT activate / process / restore S10 draft

Browse / Open without registration
  ensure_switch_allowed → open_session_path(ephemeral)
  remember recent; no catalog write

Register…
  background preview_registration (GuiTaskKind.WORKSPACE_SCAN)
  show RegistrationPreviewDialog
  discard on cancel / navigate-away (bump scan generation)
  commit_registration(operator_approved=True) → catalog only → activate

Rescan
  background rescan_catalog → same preview/confirm path
  path match ⇒ reuse UUIDs

Rebind…
  select missing/registered session → browse replacement
  background preview_rebind → RebindPreviewDialog
  warn on fingerprint/digest mismatch; explicit override required
  commit_rebind → catalog paths only; retain UUIDs

Clear
  cancel background scan; clear context; mirror _session_root=None
```

### 15.3 Job / navigation safety (shell)

| Condition | Workspace switch |
| --- | --- |
| S2 or S4 task running (`hard_locked`) | **Blocked** |
| Any S5–S10 page task running (`page_task_running`) | **Blocked** for shared context change |
| Open S10 draft | Allowed only when not hard/soft blocked; on successful target change MainWindow calls `close_session(release_lock=True)` |
| Background discovery in flight | Cancel / bump generation; stale results ignored |
| Navigate away during registration preview | Discard pending preview; no catalog write |
| Per-page `selected_*_run_id` | Never cleared by D0-D1 shell |

Existing soft/hard page navigation (group buttons, page Browse) preserved unchanged.

### 15.4 Responsiveness and errors

- Directory scans / registration preview / rescan / rebind preview run on `GuiTaskRunner` worker threads.
- Qt main thread must not recursively discover, hash media, or migrate catalogs.
- Operator-facing errors: missing paths, read-only catalog locations, unsupported catalog versions, corrupted/quarantined catalogs, ambiguous layouts, registration conflicts.

### 15.5 Laboratory / test evidence (D0-D1)

| Check | Result |
| --- | --- |
| `tests/workspace/` | Passing (backend unchanged semantically) |
| `tests/ui/test_workspace_gui.py` | Passing (shell/controller coverage listed in D0-D1 row above) |
| Ruff on touched UI/workspace files | Clean |
| Mypy on `src/workspace` | Clean |
| Broad S1–S10 page integration | Completed in D0-D2 (§16) |

### 15.6 D0-D2 handoff — satisfied

All D0-D1 → D0-D2 prerequisites in the prior handoff list are implemented in §16.

---

## 16. D0-D2 finalized contracts (normative for implementers)

### 16.1 Single source of truth

```text
MainWindow.workspace                         # authoritative WorkspaceContext
MainWindow.workspace_controller              # WorkspaceShellController
MainWindow._session_root                     # compatibility mirror of discovery_root only
pages do NOT construct WorkspaceContext
```

- S1 wizard `session_changed` and S3/S4 shell Browse continue to call `_remember_session` → `sync_from_legacy_path`.
- S2/S4–S10 page Browse/Open call `request_shared_session` → `MainWindow._open_session_from_page_browse` → `WorkspaceShellController.open_session_path`.
- Helper module: `ui.workspace.page_binding`.

### 16.2 Shared selection propagation

On `workspace_controller.workspace_changed` (Qt bridge `analysisTargetChanged` / snapshot):

1. Mirror `discovery_root` → `_session_root`.
2. Update `BehavioralReviewController.set_workspace_discovery_root`.
3. If an S10 draft was open and the root changed → `close_session(release_lock=True)` (journal remains on disk; no auto-reopen).
4. Propagate to bound pages via `_sync_bound_pages_to_discovery_root`:
   - **Same root:** `refresh_from_state` only (preserves `selected_*_run_id` / open draft).
   - **Never opened:** leave idle unless it is the current page (`force_bind`).
   - **Different root + busy:** skip with status; do not retarget the active job.
   - **Different root + idle:** `set_session(discovery_root)` (controllers clear lineage-invalid run ids).
5. S3: `open_project` when previously bound or current.

Explicit per-page run pickers remain unchanged. Preferences (tool paths, QSettings) are not cleared by workspace signals.

### 16.3 Browse behavior

| Entry | Writes shared context? |
| --- | --- |
| Workspace selector Browse/Open/Recent/Register/Rebind | Yes (D0-D1) |
| S1 create/open | Yes via `_remember_session` |
| S3/S4 shell Browse | Yes via `_remember_session` |
| S2/S4–S10 page Browse/Open | Yes via `request_shared_session` (D0-D2) |
| Legacy path open without catalog | Yes ephemeral (`open_session_path`) |

Catalog registration is never required to open existing recordings.

### 16.4 S10 draft / publication safety

| Condition | Behavior |
| --- | --- |
| Target change while edit/publish/open mutation active | **Blocked** (`has_blocking_mutation` in nav probe) |
| Target change with idle open draft | Allowed; draft lock released; restore only via explicit reopen |
| `build_publish_preview` / `begin_publish` | `_assert_publish_matches_active_target`: package/draft under `session_root` **and** `session_root == workspace_discovery_root` |
| Publish identity | Includes `session_root`; stale apply rejected on mismatch |
| Auto-publish / auto-accept | **Forbidden** (unchanged) |

### 16.5 Navigation and worker safeguards

- Hard lock (S2/S4) and soft lock (any S5–S10 task **including discovery**) still block shell workspace switches (D0-D1 policy retained).
- Active jobs keep original source identity; context change cannot retarget mid-flight.
- Discovery/open/edit/publish callbacks remain generation / request-id / session_root guarded.
- Same-root workspace signals do **not** re-enter `prepare_session` (avoids duplicate scientific discovery jobs and accidental S10 draft close).

### 16.6 Tests and laboratory evidence (D0-D2)

| Check | Result |
| --- | --- |
| `tests/ui/test_workspace_gui.py` | Passing |
| `tests/ui/test_workspace_d0_d2_integration.py` | Passing (Browse write-back S2/S4–S10; S1→S8/S10; S9→S2/S5; ephemeral/registered/incomplete; run invalidation; stale callback; S10 draft close/reopen; mutation/publish block; publish mismatch; no science writes) |
| Lab smoke E + F (`test_laboratory_e_and_f_shared_context_smoke`) | Passing read-only: select E then F via workspace open; navigate S2/S5–S10; switch back to E; `prepare_meta` / `prepared_sync` / package_manifest SHA256 unchanged |
| Manual workaround | None required for shared-context navigation. Soft discovery must finish (or be cancelled) before a shell target switch — same as D0-D1 soft-nav policy |

### 16.7 Deferred / remaining risks

| Item | Status |
| --- | --- |
| D0-C subsession science processing / `workspace_ref_v1` writers | **Deferred (not implemented)** |
| Scientific schema changes | Not in D0-D2 |
| Automatic launch restore | Rejected (W4(c)) |
| Absolute paths in historical manifests after rebind | Unrewritten by design |
| Soft discovery blocks workspace switch until idle | Retained; operators may perceive brief switch latency after Browse |

### 16.8 MVP closure verification (2026-11-01)

| Check | Result |
| --- | --- |
| One authoritative `WorkspaceContext` | Confirmed (MainWindow-owned; pages do not construct) |
| Session-selection propagation + run invalidation | Confirmed |
| Registered + ephemeral session behavior | Confirmed |
| S10 draft lock release only when idle | Confirmed (mutation/publish block switch; idle switch releases lock) |
| S10 publish/source-target compatibility | Confirmed (`Path.relative_to` containment + workspace root equality + digest pins) |
| Similar-prefix sibling session false-accept | Confirmed blocked (`test_s10_publish_blocked_for_similar_prefix_sibling_session`) |
| Stale worker callback rejection | Confirmed |
| Scientific artifact mutation from workspace nav | Confirmed none |
| `tests/workspace/` + workspace GUI/D0-D2 integration | Passing |
| S10 GUI + edit-worker suites (fixtures under session root for publish) | Passing |
| Ruff (workspace/UI surfaces) + mypy `src/workspace` | Clean |
| D0-C | **Not claimed complete** |

_End of D0 workspace architecture document._
