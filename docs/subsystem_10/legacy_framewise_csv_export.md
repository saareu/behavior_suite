# S10 legacy framewise CSV export

S10 reuses the shared `legacy_framewise_v1` builder documented in
[`../subsystem_09/legacy_framewise_csv_export.md`](../subsystem_09/legacy_framewise_csv_export.md).

## Source rules

| Mode | Behavior labels | Eligible? |
| --- | --- | --- |
| Published reviewed run | `reviewed_behavior_states.parquet` | Yes |
| Working draft | — | **No** (final CSV refused) |
| Automated (S9 GUI) | S8 `behavior_states.parquet` via package pin | S9 path only |

Do **not** derive corrected labels from S9 bout summaries. Pose and original-frame
mapping still come from the parent S9 package sync index + lineage-pinned S4
(or an explicit S4 run when lineage is absent). Automated S8 labels are never
mixed into a reviewed export. Working drafts are excluded. S4 pose is pinned by
digest; upstream scientific artifacts are never rewritten. Rounding to `.0` is
serialization-only for this profile.

## GUI

Subsystem 10 Review page → **Export legacy framewise CSV** requires an explicitly
selected **published** review. Default profile is `legacy_framewise_v1`.

Pose source follows the shared S9 rules: lineage pin by default; when the parent
package lacks S6→S4 lineage, the page shows a minimal completed-S4 picker and
refuses export until an explicit compatible run is selected. Newest-S4
auto-selection is never used. Working drafts remain ineligible.

## Out-of-scope fixture note (separate closure)

`tests/behavioral_review/test_e_session_sprint5.py::test_e_session_controlled_edit_publish_reopen`
currently creates the controlled draft under
`C:\behavior_suite\tmp_s10_sprint5_review_gui\...` (outside the E-session root).
D0 publish containment correctly refuses that draft. Minimum fixture correction
(do **not** relax D0): place `ISOLATED_REVIEW_ROOT` under the intended session
root as an explicitly labeled test subtree, e.g.
`SESSION / "_test_isolated_review_drafts" / "isolated_e_session_controlled_gui"`.
This CSV export work does not change that fixture or D0 gates.
