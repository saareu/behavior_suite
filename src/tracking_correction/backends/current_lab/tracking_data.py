"""
Efficient data container for tracking information.

Wraps NumPy tensors with convenient access methods.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

class TrackingData:
    """
    Container for tracking data in tensor format.
    
    Stores x, y, score as dense NumPy arrays with shape [frames, tracks, nodes]
    for efficient vectorized operations.
    
    Extra columns (instance_score, *_raw, etc.) are preserved via df_orig and
    merged back in to_dataframe() to ensure no confidence data is lost.
    """
    
    def __init__(self, df: pd.DataFrame, nodes: Optional[List[str]] = None):
        """
        Initialize from DataFrame with columns: frame, track, node, x, y, score, and optional extra columns.
        
        Args:
            df: Input DataFrame
            nodes: Optional ordered list of node names (auto-detected if None)
        """
        self.df_orig = df.copy()
        
        # Identify extra columns (non-spatial, non-tracking) to preserve
        standard_cols = {'frame', 'track', 'node', 'x', 'y', 'score'}
        self.extra_cols = [col for col in df.columns if col not in standard_cols]
        
        # Parse track names: convert 'track_1' -> '1', etc.
        df = df.copy()
        if 'track' in df.columns:
            df['track'] = df['track'].apply(lambda x: 
                x.split('_', 1)[1] if isinstance(x, str) and x.startswith('track_') else x
            )
            df['track'] = df['track'].astype(int)
        
        # Detect or use provided node list
        if nodes is None:
            nodes = list(sorted(df["node"].unique()))
        
        # Ensure stable body order: prioritize common nodes
        preferred = ["nose", "neck", "spine_base", "tail_base", "headstage"]
        rest = [n for n in nodes if n not in preferred]
        self.nodes = [n for n in preferred if n in nodes] + rest
        self.node_index: Dict[str, int] = {n: i for i, n in enumerate(self.nodes)}
        
        # Detect number of tracks from data (don't hardcode)
        self.num_tracks = int(df["track"].max()) + 1
        self.num_frames = int(df["frame"].max()) + 1
        self.num_nodes = len(self.nodes)
        
        # Initialize tensors: shape [F, T, K]
        self.x = np.full((self.num_frames, self.num_tracks, self.num_nodes), np.nan, dtype=float)
        self.y = np.full((self.num_frames, self.num_tracks, self.num_nodes), np.nan, dtype=float)
        self.score = np.full((self.num_frames, self.num_tracks, self.num_nodes), np.nan, dtype=float)
        
        # Track which (frame, track, node) tuples were blanked during corrections
        self.blanked_nodes: set = set()  # Set of (frame, track, node_name) tuples
        
        # Track which frames had track swaps during corrections
        self.swapped_frames: set = set()  # Set of frame indices
        
        self._fill_from_dataframe(df)
    
    def _fill_from_dataframe(self, df: pd.DataFrame):
        """Populate tensors from DataFrame efficiently."""
        frames = df["frame"].to_numpy(dtype=int)
        tracks = df["track"].to_numpy(dtype=int)
        node_names = df["node"].to_numpy()
        xs = df["x"].to_numpy()
        ys = df["y"].to_numpy()
        scores = df.get("score", pd.Series(np.nan, index=df.index)).to_numpy()
        
        # Vectorized fill
        for f, t, n, xv, yv, sv in zip(frames, tracks, node_names, xs, ys, scores):
            k = self.node_index.get(n)
            if k is None:
                continue
            if 0 <= f < self.num_frames and 0 <= t < self.num_tracks:
                self.x[f, t, k] = xv
                self.y[f, t, k] = yv
                self.score[f, t, k] = sv
    
    def get_node_index(self, node_name: str) -> Optional[int]:
        """
        Get tensor index for a node name.

        Args:
            node_name: Exact name of the node to find

        Returns:
            Integer index if found, None otherwise

        Note:
            This method only performs exact matches. No alias resolution is done
            to avoid ambiguity with future node types (e.g., spine_base vs spine_end).
        """
        return self.node_index.get(node_name)

    def get_node_name(self, idx: int) -> Optional[str]:
        """Get node name for a given index (or return the string unchanged).

        Accepts an integer index and returns the corresponding node name if
        valid. If a string is passed, it's returned unchanged to simplify
        callers that may pass either form.
        """
        if isinstance(idx, str):
            return idx
        try:
            if 0 <= int(idx) < len(self.nodes):
                return self.nodes[int(idx)]
        except Exception:
            pass
        return None
    
    def get_position(self, frame: int, track: int, node: str) -> Tuple[float, float]:
        """Get (x, y) position for a specific node."""
        k = self.get_node_index(node)
        if k is None:
            return (np.nan, np.nan)
        return (self.x[frame, track, k], self.y[frame, track, k])
    
    def get_centroid(self, frame: int, track: int, min_score: float = 0.0) -> Tuple[float, float]:
        """
        Compute centroid of visible nodes for a track at a frame.
        
        Args:
            frame: Frame index
            track: Track index (0 or 1)
            min_score: Minimum score threshold for visibility
            
        Returns:
            (cx, cy) centroid, or (nan, nan) if no visible nodes
        """
        xs = self.x[frame, track, :]
        ys = self.y[frame, track, :]
        scores = self.score[frame, track, :]
        
        mask = (~np.isnan(xs)) & (~np.isnan(ys)) & (scores >= min_score)
        if not mask.any():
            return (np.nan, np.nan)
        
        return (float(np.mean(xs[mask])), float(np.mean(ys[mask])))
    
    def count_visible_nodes(self, frame: int, track: int, min_score: float = 0.0) -> int:
        """Count visible nodes for a track at a frame."""
        scores = self.score[frame, track, :]
        xs = self.x[frame, track, :]
        ys = self.y[frame, track, :]
        mask = (~np.isnan(xs)) & (~np.isnan(ys)) & (scores >= min_score)
        return int(mask.sum())
    
    def compute_separation(self, frame: int, min_score: float = 0.0) -> float:
        """
        Compute distance between track centroids at a frame.
        
        Returns:
            Distance in pixels, or nan if either centroid is invalid
        """
        c0 = self.get_centroid(frame, 0, min_score)
        c1 = self.get_centroid(frame, 1, min_score)
        
        if np.isnan(c0[0]) or np.isnan(c1[0]):
            return np.nan
        
        dx = c0[0] - c1[0]
        dy = c0[1] - c1[1]
        return float(np.hypot(dx, dy))
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert tensors back to DataFrame format.
        
        Preserves extra columns (instance_score, *_raw, etc.) from df_orig
        by merging on (frame, track, node) to ensure no confidence data is lost.
        
        Returns:
            DataFrame with columns: frame, track, node, x, y, score, + any extra columns
        """
        records = []
        for f in range(self.num_frames):
            for t in range(self.num_tracks):
                for k, node_name in enumerate(self.nodes):
                    xv = self.x[f, t, k]
                    yv = self.y[f, t, k]
                    sv = self.score[f, t, k]
                    
                    # Only include rows with valid positions
                    if not np.isnan(xv) and not np.isnan(yv):
                        records.append({
                            'frame': f,
                            'track': t,
                            'node': node_name,
                            'x': xv,
                            'y': yv,
                            'score': sv
                        })
        
        df_out = pd.DataFrame.from_records(records)
        
        # --- NEW: Merge in extra columns from df_orig ---
        if self.extra_cols and len(df_out) > 0:
            # Prepare df_orig for merge: ensure track is int
            df_orig_merge = self.df_orig.copy()
            if 'track' in df_orig_merge.columns:
                df_orig_merge['track'] = df_orig_merge['track'].apply(lambda x: 
                    x.split('_', 1)[1] if isinstance(x, str) and x.startswith('track_') else x
                )
                try:
                    df_orig_merge['track'] = df_orig_merge['track'].astype(int)
                except (ValueError, TypeError):
                    pass  # Keep original type if conversion fails
            
            # Keep only (frame, track, node) + extra columns
            merge_cols = ['frame', 'track', 'node'] + self.extra_cols
            df_orig_merge = df_orig_merge[[col for col in merge_cols if col in df_orig_merge.columns]]
            
            # Merge: left join to preserve all rows from df_out
            df_out = df_out.merge(
                df_orig_merge,
                on=['frame', 'track', 'node'],
                how='left'
            )
        
        return df_out
    
    def copy(self) -> 'TrackingData':
        """Create a deep copy of this data container."""
        new_data = TrackingData.__new__(TrackingData)
        new_data.nodes = self.nodes.copy()
        new_data.node_index = self.node_index.copy()
        new_data.num_frames = self.num_frames
        new_data.num_tracks = self.num_tracks
        new_data.num_nodes = self.num_nodes
        new_data.x = self.x.copy()
        new_data.y = self.y.copy()
        new_data.score = self.score.copy()
        new_data.df_orig = self.df_orig.copy()
        return new_data
    
    def blank_frame(self, frame: int, track: int):
        """Set all nodes for a track at a frame to NaN."""
        self.x[frame, track, :] = np.nan
        self.y[frame, track, :] = np.nan
        # Keep scores for logging purposes
    
    def blank_node(self, frame: int, track: int, node: str):
        """Set a specific node to NaN and track the blanking."""
        k = self.get_node_index(node)
        if k is not None:
            self.x[frame, track, k] = np.nan
            self.y[frame, track, k] = np.nan
            self.score[frame, track, k] = 0.0  # Also blank the score
            # Track this blanking operation
            self.blanked_nodes.add((frame, track, node))
    
    def swap_tracks(self, frame: int):
        """Swap track 0 and track 1 for all nodes at a frame and track the swap."""
        self.x[frame, [0, 1], :] = self.x[frame, [1, 0], :]
        self.y[frame, [0, 1], :] = self.y[frame, [1, 0], :]
        self.score[frame, [0, 1], :] = self.score[frame, [1, 0], :]
        # Track this swap operation
        self.swapped_frames.add(frame)
    
    def swap_node(self, frame: int, node: str):
        """Swap a specific node between tracks at a frame and track the swap."""
        k = self.get_node_index(node)
        if k is not None:
            self.x[frame, [0, 1], k] = self.x[frame, [1, 0], k]
            self.y[frame, [0, 1], k] = self.y[frame, [1, 0], k]
            self.score[frame, [0, 1], k] = self.score[frame, [1, 0], k]
            # Track this swap operation
            self.swapped_frames.add(frame)
