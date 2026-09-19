"""
Helper functions for finalizing tracks DataFrame before saving.
Handles column cleanup, flag computation, schema validation, and sorting.
"""
import pandas as pd
import numpy as np
from typing import Optional, Set


def _test_finalize_tracks_df():
    """
    Self-test: verify schema, flags, sorting, and dtypes.
    """
    print("[TEST] finalize_tracks_df_for_save()...")
    
    # Build tiny test dataframe
    df = pd.DataFrame({
        'frame': [0, 0, 1, 1],
        'track': [0, 0, 1, 1],
        'node': ['nose', 'tail', 'nose', 'tail'],
        'x': [1.0, 2.0, np.nan, 4.0],
        'y': [1.0, 2.0, np.nan, 4.0],
        'score': [0.9, 0.8, 0.7, 0.6],  # Will be dropped
        'point_score': [0.95, 0.94, 0.93, 0.92],
        'instance_score': [0.5, 0.5, 0.6, 0.6],
        'x_raw': [1.0, 2.0, 3.0, 4.0],  # Row 2: raw finite but final NaN -> blanked
        'y_raw': [1.0, 2.0, 3.0, 4.0],
        'point_score_raw': [0.95, 0.94, 0.93, 0.92],
        'instance_score_raw': [0.5, 0.5, 0.6, 0.6],
    })
    
    swapped_frames = {1}  # Frame 1 was swapped
    blanked_nodes = {(0, 0, 'tail')}  # Track 0, node 'tail' in frame 0 was explicitly blanked
    
    result = finalize_tracks_df_for_save(df, swapped_frames=swapped_frames, blanked_nodes=blanked_nodes)
    
    # Assertions
    # 1. Column order
    expected_cols = [
        'frame', 'track', 'node',
        'x', 'y', 'point_score', 'instance_score',
        'x_raw', 'y_raw', 'point_score_raw', 'instance_score_raw',
        'was_track_swapped', 'was_node_blanked'
    ]
    assert list(result.columns) == expected_cols, f"Column order mismatch: {list(result.columns)} != {expected_cols}"
    
    # 2. 'score' dropped
    assert 'score' not in result.columns, "Column 'score' should be dropped"
    
    # 3. Dtypes
    assert result['frame'].dtype == np.int64, f"frame dtype should be int64, got {result['frame'].dtype}"
    assert result['x'].dtype == np.float32, f"x dtype should be float32, got {result['x'].dtype}"
    assert result['y'].dtype == np.float32, f"y dtype should be float32, got {result['y'].dtype}"
    assert result['point_score'].dtype == np.float32, f"point_score dtype should be float32, got {result['point_score'].dtype}"
    assert result['instance_score'].dtype == np.float32, f"instance_score dtype should be float32, got {result['instance_score'].dtype}"
    assert result['x_raw'].dtype == np.float32, f"x_raw dtype should be float32, got {result['x_raw'].dtype}"
    assert result['y_raw'].dtype == np.float32, f"y_raw dtype should be float32, got {result['y_raw'].dtype}"
    assert result['was_track_swapped'].dtype == np.uint8, f"was_track_swapped dtype should be uint8, got {result['was_track_swapped'].dtype}"
    assert result['was_node_blanked'].dtype == np.uint8, f"was_node_blanked dtype should be uint8, got {result['was_node_blanked'].dtype}"
    
    # 4. Sorting (should be sorted by frame, track, node)
    expected_order = result.sort_values(['frame', 'track', 'node'], kind='mergesort')
    pd.testing.assert_frame_equal(result, expected_order, check_dtype=True)
    
    # 5. Flag values
    # Row 0 (frame=0, track=0, node='nose'): not blanked (not in blanked_nodes), not swapped (frame 0 not in swapped_frames)
    assert result.iloc[0]['was_node_blanked'] == 0, "Row 0 should not be blanked"
    assert result.iloc[0]['was_track_swapped'] == 0, "Row 0 should not be swapped (frame 0 not in swapped_frames)"
    
    # Row 1 (frame=0, track=0, node='tail'): blanked (explicitly in blanked_nodes), not swapped
    assert result.iloc[1]['was_node_blanked'] == 1, "Row 1 should be blanked (explicitly in blanked_nodes)"
    assert result.iloc[1]['was_track_swapped'] == 0, "Row 1 should not be swapped (frame 0 not in swapped_frames)"
    
    # Row 2 (frame=1, track=1, node='nose'): blanked (x_raw=3.0 finite, x=NaN), swapped (frame 1 in swapped_frames)
    assert result.iloc[2]['was_node_blanked'] == 1, "Row 2 should be blanked (x,y are NaN but x_raw,y_raw are finite)"
    assert result.iloc[2]['was_track_swapped'] == 1, "Row 2 should be swapped (frame 1 in swapped_frames)"
    
    print("  ✓ Column order correct")
    print("  ✓ Dtypes correct")
    print("  ✓ Sorting correct")
    print("  ✓ Flags computed correctly (explicit + implicit blanking, frame-level swaps)")
    print("[TEST] PASSED\n")


def finalize_tracks_df_for_save(
    df: pd.DataFrame,
    swapped_frames: Optional[Set[int]] = None,
    blanked_nodes: Optional[Set] = None
) -> pd.DataFrame:
    """
    Finalize tracks DataFrame before saving to parquet.
    
    Operations:
    1. Drop 'score' column (redundant with point_score)
    2. Ensure instance_score and instance_score_raw exist (fill with NaN if missing)
    3. Compute was_node_blanked flag (explicit tracking + implicit detection: raw finite but final NaN)
    4. Compute was_track_swapped flag (1 iff frame was involved in track swap)
    5. Ensure correct dtypes (frame: int64, coords/scores: float32, flags: uint8)
    6. Sort by (frame, track, node) stably
    7. Return with exact column order
    
    Args:
        df: Input DataFrame (typically output of process_video_minimal + any constraints)
        swapped_frames: Set of frame indices where tracks were swapped (can be None)
        blanked_nodes: Set of (frame, track, node_name) tuples explicitly blanked during corrections (can be None)
    
    Returns:
        DataFrame ready for parquet save with guaranteed schema and ordering
    """
    df = df.copy()
    
    if swapped_frames is None:
        swapped_frames = set()
    if blanked_nodes is None:
        blanked_nodes = set()
    
    # ======================================================================
    # 1. Drop 'score' column (redundant with point_score)
    # ======================================================================
    if 'score' in df.columns:
        df = df.drop(columns=['score'])
    
    # ======================================================================
    # 2. Ensure instance_score and instance_score_raw exist
    # ======================================================================
    if 'instance_score' not in df.columns:
        df['instance_score'] = np.nan
    if 'instance_score_raw' not in df.columns:
        df['instance_score_raw'] = np.nan
    
    # ======================================================================
    # 3. Compute was_node_blanked flag
    # ======================================================================
    # Combine explicit tracking (from corrections) and implicit detection (raw finite but final NaN)
    was_blanked_explicit = df.apply(
        lambda row: 1 if (row['frame'], row['track'], row['node']) in blanked_nodes else 0,
        axis=1
    ).astype(np.uint8)
    
    # Also detect blanking from coordinate mismatches
    x_raw_finite = np.isfinite(df['x_raw'].values)
    y_raw_finite = np.isfinite(df['y_raw'].values)
    x_final_nan = np.isnan(df['x'].values)
    y_final_nan = np.isnan(df['y'].values)
    was_blanked_implicit = (x_raw_finite & y_raw_finite & (x_final_nan | y_final_nan)).astype(np.uint8)
    
    # Use OR logic: if either explicit or implicit blanking detected, mark as blanked
    df['was_node_blanked'] = np.maximum(was_blanked_explicit, was_blanked_implicit)
    
    # ======================================================================
    # 4. Compute was_track_swapped flag
    # ======================================================================
    if len(swapped_frames) == 0:
        df['was_track_swapped'] = np.uint8(0)
    else:
        df['was_track_swapped'] = df['frame'].isin(swapped_frames).astype(np.uint8)
    
    # ======================================================================
    # 5. Ensure correct dtypes
    # ======================================================================
    dtype_map = {
        'frame': np.int64,
        'x': np.float32,
        'y': np.float32,
        'point_score': np.float32,
        'instance_score': np.float32,
        'x_raw': np.float32,
        'y_raw': np.float32,
        'point_score_raw': np.float32,
        'instance_score_raw': np.float32,
        'was_track_swapped': np.uint8,
        'was_node_blanked': np.uint8,
    }
    
    for col, dtype in dtype_map.items():
        if col in df.columns:
            if df[col].dtype != dtype:
                df[col] = df[col].astype(dtype)
    
    # ======================================================================
    # 6. Sort by (frame, track, node) stably
    # ======================================================================
    df = df.sort_values(['frame', 'track', 'node'], kind='mergesort').reset_index(drop=True)
    
    # ======================================================================
    # 7. Select and reorder columns
    # ======================================================================
    final_cols = [
        'frame', 'track', 'node',
        'x', 'y', 'point_score', 'instance_score',
        'x_raw', 'y_raw', 'point_score_raw', 'instance_score_raw',
        'was_track_swapped', 'was_node_blanked'
    ]
    
    # Verify all required columns exist
    missing = [col for col in final_cols if col not in df.columns]
    if missing:
        raise ValueError(f"[finalize_tracks_df_for_save] Missing required columns: {missing}")
    
    df = df[final_cols]
    
    return df


if __name__ == '__main__':
    _test_finalize_tracks_df()
