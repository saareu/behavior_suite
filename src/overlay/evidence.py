"""Assemble overlay evidence from timing + stage adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from behavioral_review.sync import SyncMap
from overlay.adapters.base import OverlayStageAdapter
from overlay.context import OverlayEvidenceBundle, OverlayObservation
from overlay.contracts import LayerId, OverlayContractError, PlateauPolicy, parse_layer_id
from overlay.timing import PreparedFrameTiming, resolve_prepared_frame_timing


def collect_evidence(
    sync: SyncMap,
    prepared_frame_idx: int,
    adapters: Mapping[LayerId, OverlayStageAdapter] | Sequence[OverlayStageAdapter],
    *,
    selected_layers: Sequence[str | LayerId],
    plateau_policy: PlateauPolicy = PlateauPolicy.LAST_IN_PLATEAU,
) -> OverlayEvidenceBundle:
    """Resolve timing and observe selected layers (read-only)."""

    selected = tuple(parse_layer_id(x) for x in selected_layers)
    if not selected:
        raise OverlayContractError("No layers selected for evidence collection.")

    adapter_map: dict[LayerId, OverlayStageAdapter]
    if isinstance(adapters, Mapping):
        adapter_map = dict(adapters)
    else:
        adapter_map = {a.layer_id: a for a in adapters}

    timing = resolve_prepared_frame_timing(
        sync,
        prepared_frame_idx,
        plateau_policy=plateau_policy,
    )
    observations: dict[LayerId, OverlayObservation] = {}
    for layer in selected:
        adapter = adapter_map.get(layer)
        if adapter is None:
            observations[layer] = OverlayObservation(
                layer_id=layer,
                available=False,
                reason=f"No adapter registered for layer {layer.value}.",
            )
            continue
        observations[layer] = adapter.observe(
            timing,
            prepared_frame_idx=int(prepared_frame_idx),
        )
    return OverlayEvidenceBundle(
        prepared_frame_idx=int(prepared_frame_idx),
        timing=timing,
        observations=observations,
    )


def debug_counter_observation(timing: PreparedFrameTiming) -> OverlayObservation:
    """Always-available debug observation carrying timing indices."""

    return OverlayObservation(
        layer_id=LayerId.DEBUG_FRAME_COUNTER,
        available=True,
        reason=None,
        payload={
            "prepared_frame_idx": timing.prepared_frame_idx,
            "dense_sample_idx": timing.dense_sample_idx,
            "plateau_policy": timing.plateau_policy.value,
        },
    )
