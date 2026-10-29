# Subsystem 10 — Laboratory Acceptance Checklist

**Status:** Pending operator completion (do **not** mark S10 lab-validated until signed off).  
**Package under test:** `test_cases/E_full_session_integration/results_package/results_package__20261028T181714`  
**Date opened:** 2026-10-29  

Controlled synthetic edits used during validation are **not** genuine biological annotations. Do not modify S8/S9 automated artifacts.

---

## A. Navigation and session

| # | Check | Pass? | Notes |
| --- | --- | --- | --- |
| A1 | Group 4 shows S9 and S10 side by side | ☐ | |
| A2 | Shared session root preserved when switching S9 ↔ S10 and across groups | ☐ | |
| A3 | S2/S4 hard locks still block navigation; S10 tasks do not hard-lock groups | ☐ | |
| A4 | Leaving S10 during discovery/publish leaves soft background status; return restores page | ☐ | |

---

## B. Discovery and source selection

| # | Check | Pass? | Notes |
| --- | --- | --- | --- |
| B1 | Eligible S9 packages listed; ineligible packages cannot open as review parents | ☐ | |
| B2 | Working drafts and published reviewed runs listed with classification / eligibility | ☐ | |
| B3 | Superseded reviewed runs are visible and not silently treated as current | ☐ | |
| B4 | Explicit automated vs reviewed analysis-source selection (no silent auto-pick) | ☐ | |

---

## C. Inspection and synchronization

| # | Check | Pass? | Notes |
| --- | --- | --- | --- |
| C1 | Open E-session package: ~36 528 dense samples, ~609 bouts | ☐ | |
| C2 | Dual clocks: analysis sample/time vs prepared frame/time | ☐ | |
| C3 | Play / pause / scrub / ±1 follow dense playhead | ☐ | |
| C4 | Prepared-frame plateaus keep pixmap; ethogram/evidence still update | ☐ | |
| C5 | Both mouse directions selectable; exclusive lanes mutually exclusive per direction | ☐ | |
| C6 | Contact and SBS lanes independent of exclusive ethogram | ☐ | |
| C7 | Bout filters, prev/next, jump start/end | ☐ | |
| C8 | S8 evidence / provenance / reconstruction visible; invalid ≠ confident zero | ☐ | |
| C9 | Optional S7 evidence loads without freezing UI | ☐ | |
| C10 | Opening package does **not** create drafts, accept bouts, or publish | ☐ | |
| C11 | GUI remains responsive during scrub/play (no long GUI freezes) | ☐ | |

---

## D. Editing mode and corrections (controlled draft only)

Use an isolated review root / new draft. Label notes as controlled validation.

| # | Check | Pass? | Notes |
| --- | --- | --- | --- |
| D1 | Editing blocked until reviewer identity + Enter editing mode + Create/Resume draft | ☐ | |
| D2 | Mode banner clearly shows EDITING ACTIVE when writable | ☐ | |
| D3 | Exclusive label correction via backend (not Qt-local science) | ☐ | |
| D4 | Contact and/or SBS correction | ☐ | |
| D5 | Bout trim and/or split; prior accept/reject on affected intervals voided | ☐ | |
| D6 | Accept / reject / unreviewed; reject does not clear labels | ☐ | |
| D7 | Reviewer note attached | ☐ | |
| D8 | Undo / redo | ☐ | |
| D9 | Restart recovery: reopen draft after app restart / session close | ☐ | |
| D10 | No reciprocal tool; no `uncertain` status offered | ☐ | |
| D11 | Create draft / corrections / undo-redo do **not** freeze the GUI for multi-second stalls (edit worker) | ☐ | |
| D12 | Competing edit/undo/publish clicks while a mutation runs are blocked with a clear working state | ☐ | |
| D13 | Navigating away during an edit does not discard a committed journal event; reopening the draft reflects it | ☐ | |

---

## E. Publication and reopening

| # | Check | Pass? | Notes |
| --- | --- | --- | --- |
| E1 | Publish preview shows parent, reviewer, corrected intervals, status counts, blockers | ☐ | |
| E2 | Publish blocked when parent stale / lock conflict / invalid draft (exercise one blocker) | ☐ | |
| E3 | Successful Publish Reviewed Results creates immutable `behavioral_review__*` | ☐ | |
| E4 | Published run reopens for inspection without mutation | ☐ | |
| E5 | Editing a published result requires an explicit new working draft | ☐ | |
| E6 | S8/S9 package fingerprints unchanged after edit+publish | ☐ | |

---

## F. Sign-off

| Role | Name | Date | Result |
| --- | --- | --- | --- |
| Operator | | | ☐ Pass / ☐ Fail |
| Reviewer (optional) | | | |

**Outstanding items before lab-validated status:** complete all critical rows (C, D, E) and record workstation performance notes below.

### Performance notes (operator)

| Metric | Observation |
| --- | --- |
| Open package latency | |
| Scrub / play smoothness | |
| Evidence window latency | |
| Create-draft GUI begin vs worker complete | |
| Exclusive / contact-SBS correction GUI begin vs worker | |
| Undo / redo GUI begin vs worker | |
| Publish duration | |

Prior accepted audit: seek/scrub/decode/window/open were already responsive; remaining ~1.8 s stalls were draft create / corrections / undo-redo on the GUI thread (now moved to `BEHAVIORAL_REVIEW_EDIT`).

---

_End of S10 laboratory acceptance checklist._
