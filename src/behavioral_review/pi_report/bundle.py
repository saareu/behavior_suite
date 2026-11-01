"""Static HTML/JS/CSS PI report bundle writer (offline, no CDN)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from behavioral_review.gallery import GalleryIndex
from behavioral_review.pi_report.contracts import (
    APP_CSS_FILENAME,
    APP_JS_FILENAME,
    ASSETS_DIRNAME,
    INDEX_HTML_FILENAME,
    PI_REPORT_DATA_SCHEMA_VERSION,
    README_FILENAME,
    REPORT_DATA_FILENAME,
    REPORT_EMBED_JS_FILENAME,
    REPORT_SCOPE_COMPLETE,
    REPORT_SCOPE_SELECTED,
)
from behavioral_review.pi_report.media import MediaMaterializationResult
from results_packaging.export import write_json

# Portable relative asset paths only — never absolute laboratory paths.
_INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Behavior Suite — PI Behavioral Report</title>
  <link rel="stylesheet" href="assets/app.css"/>
</head>
<body>
  <header class="top">
    <div>
      <h1>PI Behavioral Report</h1>
      <p id="source-badge" class="badge">Loading…</p>
    </div>
    <p id="summary-line" class="muted"></p>
  </header>
  <p id="scope-banner" class="scope-banner hidden" role="status"></p>

  <section class="filters" aria-label="Report filters">
    <label>Focal
      <select id="filter-focal"></select>
    </label>
    <label>Label
      <select id="filter-label"></select>
    </label>
    <label>Status
      <select id="filter-status"></select>
    </label>
    <label class="search">Search bout id
      <input id="filter-search" type="search" placeholder="bout id…"/>
    </label>
  </section>

  <main id="groups"></main>

  <footer class="provenance">
    <h2>Provenance</h2>
    <pre id="provenance"></pre>
    <p class="muted">Offline report — no network or Python runtime required. Contact and side-by-side channels are excluded from this exclusive-bout view.</p>
  </footer>

  <script src="assets/report_embed.js"></script>
  <script src="assets/app.js"></script>
</body>
</html>
"""

_APP_CSS = """:root {
  --bg: #f7f4ef;
  --ink: #1c1a17;
  --muted: #5c574f;
  --card: #fffdf9;
  --line: #d9d2c6;
  --accent: #0f5c4c;
  --accepted: #1b6b3a;
  --rejected: #8a2f2f;
  --unreviewed: #6b5b2e;
  --voided: #5a5a5a;
  --auto: #245b8a;
  --reviewed: #6b3a7a;
  font-family: "Segoe UI", "Helvetica Neue", sans-serif;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  color: var(--ink);
  background:
    radial-gradient(circle at top left, #efe7d8 0, transparent 40%),
    linear-gradient(180deg, #fbf8f2, var(--bg));
  min-height: 100vh;
}
.top, .filters, .provenance, main {
  max-width: 1200px;
  margin: 0 auto;
  padding: 1rem 1.25rem;
}
.top { display: flex; justify-content: space-between; gap: 1rem; flex-wrap: wrap; }
h1 { margin: 0 0 0.35rem; font-size: 1.6rem; }
.muted { color: var(--muted); }
.badge {
  display: inline-block;
  margin: 0;
  padding: 0.2rem 0.55rem;
  border-radius: 999px;
  font-size: 0.85rem;
  font-weight: 600;
  color: white;
}
.badge.automated { background: var(--auto); }
.badge.reviewed { background: var(--reviewed); }
.scope-banner {
  max-width: 1200px;
  margin: 0 auto 0.5rem;
  padding: 0.65rem 1rem;
  border: 1px solid #b45309;
  border-radius: 8px;
  background: #fff7ed;
  color: #9a3412;
  font-weight: 600;
}
.filters {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem 1rem;
  border-top: 1px solid var(--line);
  border-bottom: 1px solid var(--line);
  background: rgba(255,255,255,0.55);
}
.filters label { display: flex; flex-direction: column; gap: 0.25rem; font-size: 0.85rem; }
.filters select, .filters input {
  min-width: 10rem;
  padding: 0.35rem 0.5rem;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: white;
}
.group { margin: 1.25rem 0 1.75rem; }
.group-head {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  align-items: baseline;
  margin-bottom: 0.75rem;
}
.group-head h2 { margin: 0; font-size: 1.2rem; }
.counts { font-size: 0.9rem; color: var(--muted); }
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 0.85rem;
}
.card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 10px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}
.card video {
  width: 100%;
  aspect-ratio: 4 / 3;
  object-fit: cover;
  background: #111;
  display: block;
}
.meta { padding: 0.65rem 0.75rem 0.8rem; font-size: 0.82rem; line-height: 1.35; }
.meta .id { font-family: Consolas, "Courier New", monospace; word-break: break-all; }
.status {
  display: inline-block;
  margin-top: 0.35rem;
  padding: 0.1rem 0.45rem;
  border-radius: 999px;
  color: white;
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
}
.status.accepted { background: var(--accepted); }
.status.rejected { background: var(--rejected); }
.status.unreviewed { background: var(--unreviewed); }
.status.voided { background: var(--voided); }
.corrected { color: var(--reviewed); font-weight: 600; }
.hidden { display: none !important; }
.provenance pre {
  white-space: pre-wrap;
  word-break: break-word;
  background: #fff;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 0.75rem;
  font-size: 0.78rem;
}
.empty {
  padding: 2rem;
  text-align: center;
  color: var(--muted);
  border: 1px dashed var(--line);
  border-radius: 8px;
}
"""

_APP_JS = r"""(function () {
  "use strict";

  var data = window.PI_REPORT_DATA;
  if (!data || !Array.isArray(data.bouts)) {
    document.body.innerHTML = "<p style='padding:2rem'>Report data missing. Open this folder's index.html from a complete PI report export.</p>";
    return;
  }

  var sourceBadge = document.getElementById("source-badge");
  var summaryLine = document.getElementById("summary-line");
  var scopeBanner = document.getElementById("scope-banner");
  var provenance = document.getElementById("provenance");
  var groupsEl = document.getElementById("groups");
  var focalSel = document.getElementById("filter-focal");
  var labelSel = document.getElementById("filter-label");
  var statusSel = document.getElementById("filter-status");
  var searchInput = document.getElementById("filter-search");

  var sourceType = (data.source_type || "automated").toLowerCase();
  sourceBadge.textContent = sourceType === "reviewed" ? "Reviewed S10 source" : "Automated S9 source";
  sourceBadge.className = "badge " + (sourceType === "reviewed" ? "reviewed" : "automated");

  var summary = data.summary || {};
  var reportScope = data.report_scope || (summary.report_scope || "all_exclusive_bouts");
  var selectedIds = data.selected_bout_ids || summary.selected_bout_ids || [];
  if (reportScope === "selected_bouts" || data.subset_of_source) {
    scopeBanner.className = "scope-banner";
    scopeBanner.textContent =
      "SUBSET REPORT — explicitly selected exclusive bouts only (" +
      (selectedIds.length || summary.n_bouts || data.bouts.length) +
      "). This is not a complete exclusive-bout export.";
  }

  summaryLine.textContent =
    "Scope: " + (reportScope === "selected_bouts" ? "selected bouts" : "all exclusive bouts") +
    " · Exclusive bouts: " + (summary.n_bouts || data.bouts.length) +
    " · Groups: " + (summary.n_groups || (data.groups || []).length) +
    " · Duration sum: " + Number(summary.total_duration_sec || 0).toFixed(3) + " s" +
    " · Media ok: " + (summary.n_media_succeeded != null ? summary.n_media_succeeded : "?");

  provenance.textContent = JSON.stringify(data.provenance || {}, null, 2);

  function fillSelect(sel, values, allLabel) {
    sel.innerHTML = "";
    var all = document.createElement("option");
    all.value = "";
    all.textContent = allLabel;
    sel.appendChild(all);
    values.forEach(function (v) {
      var opt = document.createElement("option");
      opt.value = String(v);
      opt.textContent = String(v);
      sel.appendChild(opt);
    });
  }

  var filters = data.filters || {};
  fillSelect(focalSel, filters.available_focal_ids || [], "All focals");
  fillSelect(labelSel, filters.available_labels || [], "All labels");
  fillSelect(statusSel, filters.available_statuses || [], "All statuses");

  function matches(bout) {
    var focal = focalSel.value;
    var label = labelSel.value;
    var status = statusSel.value;
    var q = (searchInput.value || "").trim().toLowerCase();
    if (focal !== "" && String(bout.focal_id) !== focal) return false;
    if (label !== "" && String(bout.effective_label) !== label) return false;
    if (status !== "" && String(bout.status) !== status) return false;
    if (q && String(bout.bout_id).toLowerCase().indexOf(q) < 0 &&
        String(bout.card_id || "").toLowerCase().indexOf(q) < 0) return false;
    return true;
  }

  function render() {
    groupsEl.innerHTML = "";
    var groups = data.groups || [];
    var any = false;
    groups.forEach(function (group) {
      var cards = (group.card_ids || []).map(function (id) {
        return data.bouts.find(function (b) { return b.card_id === id; });
      }).filter(function (b) { return b && matches(b); });
      if (!cards.length) return;
      any = true;
      var section = document.createElement("section");
      section.className = "group";
      var head = document.createElement("div");
      head.className = "group-head";
      head.innerHTML = "<h2>" + escapeHtml(group.label) + "</h2>" +
        "<div class='counts'>" + cards.length + " bout(s)</div>";
      var grid = document.createElement("div");
      grid.className = "grid";
      cards.forEach(function (bout) {
        grid.appendChild(renderCard(bout));
      });
      section.appendChild(head);
      section.appendChild(grid);
      groupsEl.appendChild(section);
    });
    if (!any) {
      var empty = document.createElement("div");
      empty.className = "empty";
      empty.textContent = "No exclusive bouts match the current filters.";
      groupsEl.appendChild(empty);
    }
  }

  function renderCard(bout) {
    var card = document.createElement("article");
    card.className = "card";
    var video = document.createElement("video");
    video.muted = true;
    video.loop = true;
    video.playsInline = true;
    video.controls = true;
    video.preload = "metadata";
    if (bout.media_relpath) {
      video.src = String(bout.media_relpath).replace(/\\\\/g, "/");
      video.addEventListener("mouseenter", function () { video.play().catch(function () {}); });
      video.addEventListener("mouseleave", function () { video.pause(); });
    }
    var meta = document.createElement("div");
    meta.className = "meta";
    var corrected = bout.corrected ? "<div class='corrected'>corrected label</div>" : "";
    meta.innerHTML =
      "<div class='id'>" + escapeHtml(bout.bout_id) + "</div>" +
      "<div>" + escapeHtml(String(bout.focal_id)) + " → " + escapeHtml(String(bout.partner_id)) + "</div>" +
      "<div>" + escapeHtml(bout.effective_label) +
      " · " + Number(bout.bout_duration_sec || 0).toFixed(3) + " s" +
      " (" + escapeHtml(String(bout.bout_duration_frames)) + " dense)</div>" +
      "<div>dense [" + escapeHtml(String(bout.start_sample_idx)) + ", " +
      escapeHtml(String(bout.end_sample_idx)) + "]</div>" +
      corrected +
      "<span class='status " + escapeHtml(String(bout.status || "unreviewed")) + "'>" +
      escapeHtml(String(bout.status || "unreviewed")) + "</span>";
    card.appendChild(video);
    card.appendChild(meta);
    return card;
  }

  function escapeHtml(text) {
    return String(text)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/\"/g, "&quot;");
  }

  focalSel.addEventListener("change", render);
  labelSel.addEventListener("change", render);
  statusSel.addEventListener("change", render);
  searchInput.addEventListener("input", render);
  render();
})();
"""

_README = """Behavior Suite — PI Behavioral Report
=====================================

Open index.html in a desktop browser (double-click). No Python runtime,
local web server, or network access is required.

Contents
--------
- index.html, assets/  — offline report UI (no CDN)
- report_data.json     — machine-readable bout + provenance payload
- media/               — local H.264 MP4 previews (relative paths only)
- report_manifest.json — digests, settings, source identity
- qc.json              — export QC summary

Notes
-----
- Default browsing view is exclusive ethogram bouts only (contact / SBS omitted).
- Rejected and unreviewed bouts remain visible; filters never delete them.
- Automated S9 and published S10 sources are never mixed in one report.
- This report is immutable after finalize.
"""


def build_report_data_payload(
    *,
    source_type: str,
    gallery: GalleryIndex,
    media: MediaMaterializationResult,
    provenance: Mapping[str, Any],
    settings: Mapping[str, Any],
    generated_at_utc: str,
    report_id: str,
    behavior_suite_version: str | None,
    report_scope: str = REPORT_SCOPE_COMPLETE,
    selected_bout_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Assemble portable report_data.json (relative media paths only)."""

    scope = (
        REPORT_SCOPE_SELECTED
        if report_scope == REPORT_SCOPE_SELECTED
        else REPORT_SCOPE_COMPLETE
    )
    selected = list(selected_bout_ids) if selected_bout_ids is not None else None
    media_by_bout = {c.bout_id: c for c in media.clips}
    bouts: list[dict[str, Any]] = []
    total_duration = 0.0
    statuses: set[str] = set()
    for card in gallery.cards:
        total_duration += float(card.bout_duration_sec)
        statuses.add(card.status)
        clip = media_by_bout.get(card.bout_id)
        row = card.to_dict()
        # Strip any absolute path fields that gallery source refs may carry.
        row.pop("package_dir", None)
        row.pop("review_run_dir", None)
        row["media_relpath"] = (
            clip.media_relpath if clip is not None else None
        )
        row["encode_status"] = (
            clip.encode_status if clip is not None else None
        )
        # Ensure relative separators for browsers on Windows writers.
        if isinstance(row.get("media_relpath"), str):
            row["media_relpath"] = row["media_relpath"].replace("\\", "/")
        bouts.append(row)

    filters = {
        "available_labels": list(gallery.available_labels),
        "available_focal_ids": list(gallery.available_focal_ids),
        "available_statuses": sorted(statuses),
        "default_channel": "exclusive",
        "contact_sbs_included": False,
    }
    summary = {
        "n_bouts": gallery.n_cards,
        "n_groups": gallery.n_groups,
        "total_duration_sec": total_duration,
        "status_counts": dict(gallery.status_counts),
        "n_media_succeeded": media.n_succeeded,
        "n_media_failed": media.n_failed,
        "report_scope": scope,
        "selected_bout_ids": selected,
        "subset_of_source": scope == REPORT_SCOPE_SELECTED,
    }
    # Provenance must not embed absolute laboratory paths.
    clean_prov = _strip_absolute_paths(dict(provenance))
    clean_settings = _strip_absolute_paths(dict(settings))
    return {
        "schema_version": PI_REPORT_DATA_SCHEMA_VERSION,
        "report_id": report_id,
        "generated_at_utc": generated_at_utc,
        "behavior_suite_version": behavior_suite_version,
        "source_type": source_type,
        "report_scope": scope,
        "selected_bout_ids": selected,
        "subset_of_source": scope == REPORT_SCOPE_SELECTED,
        "summary": summary,
        "filters": filters,
        "groups": [g.to_dict() for g in gallery.groups],
        "bouts": bouts,
        "media_mapping": {
            k: str(v).replace("\\", "/") for k, v in media.media_mapping.items()
        },
        "settings": clean_settings,
        "provenance": clean_prov,
    }


def write_report_bundle_files(
    *,
    report_root: Path,
    report_data: Mapping[str, Any],
) -> dict[str, Path]:
    """Write HTML shell, local assets, README, and report_data.json."""

    root = Path(report_root).expanduser().resolve()
    assets = root / ASSETS_DIRNAME
    assets.mkdir(parents=True, exist_ok=True)

    index_path = root / INDEX_HTML_FILENAME
    css_path = assets / APP_CSS_FILENAME
    js_path = assets / APP_JS_FILENAME
    embed_path = assets / REPORT_EMBED_JS_FILENAME
    data_path = root / REPORT_DATA_FILENAME
    readme_path = root / README_FILENAME

    index_path.write_text(_INDEX_HTML, encoding="utf-8")
    css_path.write_text(_APP_CSS, encoding="utf-8")
    js_path.write_text(_APP_JS, encoding="utf-8")
    embed_path.write_text(
        "window.PI_REPORT_DATA = "
        + json.dumps(report_data, ensure_ascii=False, indent=2)
        + ";\n",
        encoding="utf-8",
    )
    write_json(data_path, dict(report_data))
    readme_path.write_text(_README, encoding="utf-8")

    _assert_no_absolute_media_refs(report_data)
    _assert_no_absolute_media_refs_in_text(index_path.read_text(encoding="utf-8"))
    _assert_no_absolute_media_refs_in_text(embed_path.read_text(encoding="utf-8"))

    return {
        "index_html": index_path,
        "app_css": css_path,
        "app_js": js_path,
        "report_embed_js": embed_path,
        "report_data": data_path,
        "readme": readme_path,
    }


def _strip_absolute_paths(payload: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in payload.items():
        key_l = str(key).lower()
        if (
            any(token in key_l for token in ("_dir", "path", "root"))
            and isinstance(value, str)
            and _looks_absolute(value)
        ):
            # Keep basename-only identity crumbs; drop absolute laboratory paths.
            cleaned[key] = Path(value).name
            cleaned[f"{key}_basename"] = Path(value).name
            continue
        if isinstance(value, dict):
            cleaned[key] = _strip_absolute_paths(value)
        elif isinstance(value, list):
            cleaned[key] = [
                _strip_absolute_paths(v) if isinstance(v, dict) else v for v in value
            ]
        else:
            cleaned[key] = value
    return cleaned


def _looks_absolute(text: str) -> bool:
    value = str(text)
    if value.startswith("\\\\") or (len(value) >= 3 and value[1:3] == ":\\"):
        return True
    return value.startswith("/") and not value.startswith("./") and not value.startswith("../")


def _assert_no_absolute_media_refs(report_data: Mapping[str, Any]) -> None:
    for bout in report_data.get("bouts", []):
        if not isinstance(bout, Mapping):
            continue
        rel = bout.get("media_relpath")
        if rel is None:
            continue
        text = str(rel).replace("\\", "/")
        if text.startswith("/") or "://" in text or (len(text) >= 2 and text[1] == ":"):
            raise ValueError(f"Absolute media path leaked into report_data: {rel!r}")
        if not text.startswith("media/"):
            raise ValueError(f"Media path must be relative under media/: {rel!r}")


def _assert_no_absolute_media_refs_in_text(text: str) -> None:
    lowered = text.lower()
    # Cheap guard against accidental absolute prepared-video embedding.
    if (
        "prepared_video." in lowered
        and (":\\" in text or text.count("/home/") > 0)
        and (
            "file:///" in lowered
            or ":\\users\\" in lowered
            or ":\\behavior" in lowered
        )
    ):
        raise ValueError("Absolute laboratory path leaked into report HTML/JS assets.")


__all__ = [
    "build_report_data_payload",
    "write_report_bundle_files",
]
