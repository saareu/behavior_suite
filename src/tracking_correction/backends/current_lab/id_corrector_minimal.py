"""
Minimal Intervention ID Corrector - Rewritten

PHILOSOPHY:
- Sequential frame-by-frame processing starting after frame_window
- Stage 1: Jump handling with trajectory prediction and proximity checks
- Stage 2: Headstage removal from incorrect track
- Stage 3: Skeleton stretch validation
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional progress display
    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        return iterable

from .tracking_data import TrackingData
from .correction_params import CorrectionParams

class HeadstageRemovalHandler:
    """Stage 2: Headstage removal from incorrect track."""
    
    def __init__(self, params: CorrectionParams, data: TrackingData):
        self.params = params
        self.data = data
        self.corrections: List[Dict] = []
        self.hs_idx = data.get_node_index(self.params.headstage_node)
        self.neck_idx = data.get_node_index('neck')
        self.max_dist = self.params.headstage_neck_max_distance
    
    def _distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two positions."""
        if pos1 is None or pos2 is None:
            return np.nan
        return float(np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2))
    
    def handle_frame(self, frame: int) -> bool:
        """Handle headstage removal for one frame. Returns True if any changes were made."""
        
        hs0_pos = (self.data.x[frame, 0, self.hs_idx], self.data.y[frame, 0, self.hs_idx])
        hs1_pos = (self.data.x[frame, 1, self.hs_idx], self.data.y[frame, 1, self.hs_idx])
        hs0_prev = (self.data.x[frame - 1, 0, self.hs_idx], self.data.y[frame - 1, 0, self.hs_idx]) if frame > 0 else (np.nan, np.nan)
        hs0_valid = not np.isnan(hs0_pos[0])
        hs1_valid = not np.isnan(hs1_pos[0])
        
        if hs0_valid and not hs1_valid:
            return False
        if hs1_valid and not hs0_valid:
            # reassign headstage node to track 0 (GT)
            self.data.x[frame, 0, self.hs_idx] = self.data.x[frame, 1, self.hs_idx]
            self.data.y[frame, 0, self.hs_idx] = self.data.y[frame, 1, self.hs_idx]
            self.data.score[frame, 0, self.hs_idx] = self.data.score[frame, 1, self.hs_idx]
            self.data.blank_node(frame, 1, self.params.headstage_node)
            # add correction count
            self.corrections.append({'frame': frame, 'track': 1, 'type': 'headstage_reassigned', 'action': f'Headstage reassigned from track 1 to track 0'})
            return True
        if hs0_valid and hs1_valid:
            # check if headstage 1 is closer to headstage 0 prev, if so, swap nodes before blanking 1
            if self._distance(hs1_pos, hs0_prev) < self._distance(hs0_pos, hs0_prev):
                self.data.swap_node(frame, self.params.headstage_node)
            self.data.blank_node(frame, 1, self.params.headstage_node)
            # add correction count
            self.corrections.append({'frame': frame, 'track': 1, 'type': 'headstage_duplicate_removed', 'action': f'Headstage duplicate removed from track 1'})
            return True
        # If neither track has headstage but original input contained it, assign to track 0
        if (not hs0_valid) and (not hs1_valid) and self.hs_idx is not None:
            df_orig = self.data.df_orig
            if 'frame' in df_orig.columns and 'node' in df_orig.columns:
                mask = (df_orig['frame'] == frame) & (df_orig['node'] == self.params.headstage_node)
                if mask.any():
                    assigned = False
                    for _, row in df_orig.loc[mask].iterrows():
                        orig_x = row.get('x', np.nan)
                        orig_y = row.get('y', np.nan)
                        orig_score = row.get('score', np.nan)
                        if not (pd.isna(orig_x) or pd.isna(orig_y)):
                            # Assign into track 0
                            self.data.x[frame, 0, self.hs_idx] = float(orig_x)
                            self.data.y[frame, 0, self.hs_idx] = float(orig_y)
                            try:
                                self.data.score[frame, 0, self.hs_idx] = float(orig_score) if not pd.isna(orig_score) else 0.0
                            except Exception:
                                self.data.score[frame, 0, self.hs_idx] = 0.0

                            # If track 1 has the same coords, blank it
                            other = 1
                            if not (np.isnan(self.data.x[frame, other, self.hs_idx]) or np.isnan(self.data.y[frame, other, self.hs_idx])):
                                if np.isclose(self.data.x[frame, other, self.hs_idx], orig_x) and np.isclose(self.data.y[frame, other, self.hs_idx], orig_y):
                                    self.data.blank_node(frame, other, self.params.headstage_node)

                            self.corrections.append({'frame': frame, 'track': 0, 'type': 'headstage_assigned', 'action': f'Assigned headstage to track 0 from original data'})
                            assigned = True
                            break
                    if assigned:
                        return True
        return False

class JumpHandler:
    """Stage 2: Jump handling with trajectory prediction."""
    
    def __init__(self, params: CorrectionParams, data: TrackingData, frame_window: int):
        self.params = params
        self.data = data
        self.frame_window = frame_window
        self.corrections: List[Dict] = []
        
        # Cache thresholds to avoid repeated attribute access
        self.Jt = params.max_centroid_velocity
        self.proximity_threshold = params.proximity_threshold
        self.overlap_dist = params.overlap_dist  # tolerance for node overlap detection
        self.proximate_jump_thresh = params.proximate_jump_thresh
        self.orientation_flip_threshold = params.orientation_flip_threshold
        self.medium_proximity_threshold = params.medium_proximity_threshold
        self.orientation_cosine_threshold = params.orientation_cosine_threshold
        self.minimum_separation_distance = params.minimum_separation_distance
        # Get headstage index to exclude from position calculations
        self.headstage_idx = data.get_node_index(params.headstage_node)
        
        # Performance caches
        self._position_cache = {}  # Cache expensive position computations
        self._stable_nodes_cache = {}  # Cache stable node computations
        self._track_validity_cache = {}  # Cache track validity checks
        
        

    def _get_node_position(self, frame: int, track: int, node_idx: int) -> Optional[Tuple[float, float]]:
        # Direct numpy indexing without intermediate variables
        x_val = self.data.x[frame, track, node_idx]
        if np.isnan(x_val):
            return None
        y_val = self.data.y[frame, track, node_idx]
        if np.isnan(y_val):
            return None
        return (float(x_val), float(y_val))

    def _distance(self, pos1: Optional[Tuple[float, float]], pos2: Optional[Tuple[float, float]]) -> Optional[float]:
        if pos1 is None or pos2 is None:
            return None
        # Direct computation without creating intermediate arrays
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return float(dx * dx + dy * dy) ** 0.5
    
    def _get_visible_nodes(self, frame: int, track: int) -> List[int]:
        """Return list of node indices that are visible at the given frame for the specified track."""
        # Vectorized approach: check all nodes at once
        x_vals = self.data.x[frame, track, :]
        y_vals = self.data.y[frame, track, :] 
        valid_mask = ~(np.isnan(x_vals) | np.isnan(y_vals))
        return list(np.where(valid_mask)[0])

    def _get_stable_nodes(self, frame: int, track: int) -> List[int]:
        """Return list of node indices that have been stable (visible) over the last frame_window frames.
        Falls back to currently visible nodes if no stable nodes are found. Excludes headstage node."""
        # Check cache first
        cache_key = (frame, track)
        if cache_key in self._stable_nodes_cache:
            return self._stable_nodes_cache[cache_key]
            
        start_frame = max(0, frame - self.frame_window + 1)
        end_frame = frame + 1
        
        # Vectorized visibility check over the window
        x_window = self.data.x[start_frame:end_frame, track, :]
        y_window = self.data.y[start_frame:end_frame, track, :]
        valid_mask = ~(np.isnan(x_window) | np.isnan(y_window))
        
        # Count valid frames for each node
        visible_counts = np.sum(valid_mask, axis=0)
        min_required = max(1, self.frame_window // 3)  # Reduced threshold: 1/3 instead of 1/2
        
        # Must be visible in current frame AND meet stability requirement
        current_visible = ~(np.isnan(self.data.x[frame, track, :]) | np.isnan(self.data.y[frame, track, :]))
        stable_mask = (visible_counts >= min_required) & current_visible
        
        # Exclude headstage node from stable nodes
        if self.headstage_idx is not None:
            stable_mask[self.headstage_idx] = False
        
        result = list(np.where(stable_mask)[0])
        
        # Fallback: if no stable nodes found, use currently visible nodes (only if any exist)
        if not result:
            currently_visible_nodes = list(np.where(current_visible)[0])
            # Remove headstage from fallback as well
            if self.headstage_idx is not None and self.headstage_idx in currently_visible_nodes:
                currently_visible_nodes.remove(self.headstage_idx)
            # Only use fallback if there are actually visible nodes
            if currently_visible_nodes:
                result = currently_visible_nodes
        
        # Cache result (even if empty - this is important!)
        self._stable_nodes_cache[cache_key] = result
        return result

    def _get_highly_stable_nodes(self, frame: int, track: int) -> List[int]:
        """Return list of node indices that have been highly stable for orientation calculations.
        More stringent than _get_stable_nodes, used specifically for orientation. Excludes headstage node."""
        start_frame = max(0, frame - self.frame_window + 1)
        end_frame = frame + 1
        
        # Vectorized visibility check over the window
        x_window = self.data.x[start_frame:end_frame, track, :]
        y_window = self.data.y[start_frame:end_frame, track, :]
        valid_mask = ~(np.isnan(x_window) | np.isnan(y_window))
        
        # Count valid frames for each node - more stringent for orientation
        visible_counts = np.sum(valid_mask, axis=0)
        min_required = max(2, self.frame_window // 2)  # At least half the window, minimum 2
        
        # Must be visible in current frame AND meet high stability requirement
        current_visible = ~(np.isnan(self.data.x[frame, track, :]) | np.isnan(self.data.y[frame, track, :]))
        stable_mask = (visible_counts >= min_required) & current_visible
        
        # Exclude headstage node from highly stable nodes
        if self.headstage_idx is not None:
            stable_mask[self.headstage_idx] = False
        
        return list(np.where(stable_mask)[0])
    def _get_track_orientation(self, track: int, frame: int) -> Optional[float]:
        """Calculate the orientation of a track at a given frame based on stable nodes."""
        # Use highly stable nodes for orientation (more stringent)
        stable_nodes = self._get_highly_stable_nodes(frame, track)
        if len(stable_nodes) < 2:
            # Fallback to regular stable nodes if highly stable ones are insufficient
            stable_nodes = self._get_stable_nodes(frame, track)
            if len(stable_nodes) < 2:  # Early exit for insufficient nodes
                return None
            
        # Extract positions directly as numpy array (avoid list comprehension)
        x_vals = self.data.x[frame, track, stable_nodes]
        y_vals = self.data.y[frame, track, stable_nodes]
        valid_mask = ~(np.isnan(x_vals) | np.isnan(y_vals))
        
        if np.sum(valid_mask) < 2:
            return None
            
        # Stack valid coordinates directly
        positions = np.column_stack([x_vals[valid_mask], y_vals[valid_mask]])
        
        # Fast covariance computation
        mean_pos = np.mean(positions, axis=0)
        centered = positions - mean_pos
        
        # Use SVD instead of eigendecomposition (faster for small matrices)
        if centered.shape[0] < 3:
            # For very small point sets, use simple line fitting
            if centered.shape[0] == 2:
                direction = centered[1] - centered[0]
                return float(np.arctan2(direction[1], direction[0]))
        
        # SVD approach (faster than eigh for small matrices)
        U, s, Vt = np.linalg.svd(centered, full_matrices=False)
        principal_dir = Vt[0]  # First principal component
        
        # Orientation angle
        orientation = np.arctan2(principal_dir[1], principal_dir[0])
        return float(orientation)

    def _predict_orientation(self, track: int, frame: int) -> Optional[float]:
        """Predict orientation based on previous frames."""
        if frame <= self.frame_window:
            return None
        orientations = []
        for f in range(frame - self.frame_window, frame):
            orient = self._get_track_orientation(track, f)
            if orient is not None:
                orientations.append(orient)
        if not orientations:
            return None
        # Average angles properly using complex numbers
        angles = np.array(orientations)
        mean_angle = np.angle(np.mean(np.exp(1j * angles)))
        return float(mean_angle)

    def _get_track_position(self, track: int, frame: int) -> Optional[Tuple[float, float]]:
        if frame < 0:
            return None
            
        # Check cache first
        cache_key = (track, frame)
        if cache_key in self._position_cache:
            return self._position_cache[cache_key]
        
        stable_nodes = self._get_stable_nodes(frame, track)
        if not stable_nodes:
            self._position_cache[cache_key] = None
            return None
        
        # Vectorized position extraction and averaging
        x_vals = self.data.x[frame, track, stable_nodes]
        y_vals = self.data.y[frame, track, stable_nodes]
        valid_mask = ~(np.isnan(x_vals) | np.isnan(y_vals))
        
        # Double-check: ensure we have at least one valid node position
        if not np.any(valid_mask):
            result = None
        else:
            # Additional safety: make sure the values are actually finite numbers
            x_valid = x_vals[valid_mask]
            y_valid = y_vals[valid_mask]
            if len(x_valid) > 0 and np.all(np.isfinite(x_valid)) and np.all(np.isfinite(y_valid)):
                result = (float(np.mean(x_valid)), float(np.mean(y_valid)))
            else:
                result = None
            
        # Cache result
        self._position_cache[cache_key] = result
        return result

    def _predict_trajectory(self, track: int, frame: int) -> Optional[Tuple[float, float]]:
        if frame <= self.frame_window:
            return None
        
        # Cache positions for the window to avoid recomputation
        positions = []
        for f in range(frame - self.frame_window, frame):
            pos = self._get_track_position(track, f)
            if pos is not None:
                positions.append(pos)
        
        if len(positions) < 2:
            return None
        
        # Vectorized velocity computation
        pos_array = np.array(positions)
        velocities = np.diff(pos_array, axis=0)
        avg_velocity = np.mean(velocities, axis=0)
        
        # Predict next position using average velocity
        last_pos = positions[-1]
        return (last_pos[0] + avg_velocity[0], last_pos[1] + avg_velocity[1])

    def _calculate_jump(self, track: int, frame: int) -> Optional[float]:
        curr_pos = self._get_track_position(track, frame)
        pred_pos = self._predict_trajectory(track, frame)
        prev_pos = self._get_track_position(track, frame - 1)
        if curr_pos is None:
            return None
        if pred_pos is not None:
            return self._distance(curr_pos, pred_pos)
        if prev_pos is not None:
            return self._distance(curr_pos, prev_pos)
        return 0.0

    def _both_tracks_present(self, frame: int) -> bool:
        # Use cache to avoid repeated NaN checking
        cache_key = frame
        if cache_key in self._track_validity_cache:
            return self._track_validity_cache[cache_key]
        
        # Batch check both tracks at once
        both_tracks_data = self.data.x[frame, :, :]  # Shape: [2, num_nodes]
        track_valid = ~np.all(np.isnan(both_tracks_data), axis=1)  # [track0_valid, track1_valid]
        result = track_valid[0] and track_valid[1]
        
        self._track_validity_cache[cache_key] = result
        return result

    def _both_tracks_null(self, frame: int) -> bool:
        return not self._both_tracks_present(frame)

    def _one_track_jumped(self, frame: int) -> bool:
        # Simplified logic with early returns
        if self.jt0 is None:
            return self.jt1 is not None and self.jt1 > self.Jt
        if self.jt1 is None:
            return self.jt0 > self.Jt
        return (self.jt0 > self.Jt) or (self.jt1 > self.Jt)
        
    def _both_track_jumped(self, frame: int) -> bool:
        return (self.jt0 is not None and self.jt0 > self.Jt and 
                self.jt1 is not None and self.jt1 > self.Jt)
                
    def _track_0_jumped(self, _frame: int) -> bool:
        return self.jt0 is not None and self.jt0 > self.Jt
    
    def _track_1_jumped(self, _frame: int) -> bool:
        return self.jt1 is not None and self.jt1 > self.Jt

    def _one_track_reappeared(self, frame: int) -> bool:
        return (self.jt0 == 0) or (self.jt1 == 0)

    def _one_inter_track_overlap(self, _frame: int) -> bool:
        return (self._track_1_overlap_with_track_0_pred(_frame)) or (self._track_0_overlap_with_track_1_pred(_frame))
    
    def _intra_track_overlap_with_prev(self, frame: int, track: int) -> bool:
        """Check if a track's current nodes overlap with at least two nodes from its previous frame."""
        if frame == 0:
            return False
        
        # Vectorized overlap computation - get all coordinates at once
        curr_coords = self.data.x[frame, track, :] + 1j * self.data.y[frame, track, :]
        prev_coords = self.data.x[frame - 1, track, :] + 1j * self.data.y[frame - 1, track, :]
        
        # Valid positions in both frames (use complex number NaN check)
        valid_both = ~(np.isnan(curr_coords) | np.isnan(prev_coords))
        
        if np.sum(valid_both) < 2:
            return False
        
        # Compute distances using complex number magnitude (faster than hypot)
        distances = np.abs(curr_coords[valid_both] - prev_coords[valid_both])
        
        # Count overlaps (distance < threshold)
        overlap_count = np.sum(distances < self.overlap_dist)
        return overlap_count >= 2
    
    def _compute_frame_state(self, frame: int) -> None:
        """Compute and cache per-frame state used by jump/orientation handlers."""
        self.orient0_curr = self._get_track_orientation(0, frame)
        self.orient1_curr = self._get_track_orientation(1, frame)
        self.orient0_pred = self._predict_orientation(0, frame)
        self.orient1_pred = self._predict_orientation(1, frame)
        self.orient0_prev = self._get_track_orientation(0, frame - 1)
        self.orient1_prev = self._get_track_orientation(1, frame - 1)
        self.Xt0_pred = self._predict_trajectory(0, frame)
        self.Xt1_pred = self._predict_trajectory(1, frame)
        self.Xt0_prev = self._get_track_position(0, frame - 1)
        self.Xt1_prev = self._get_track_position(1, frame - 1)
        self.Xt0_curr = self._get_track_position(0, frame)
        self.Xt1_curr = self._get_track_position(1, frame)
        self.visible_seperation = self.get_cross_visible_seperation(frame)
        
            
        self.jt0 = self._calculate_jump(0, frame)
        self.jt1 = self._calculate_jump(1, frame)
        self.d01 = self._distance(self.Xt0_curr, self.Xt1_pred)
        self.d10 = self._distance(self.Xt1_curr, self.Xt0_pred)

    def calculate_cosine(self, pos1: Tuple[float, float], pos2: Tuple[float, float], pos3: Tuple[float, float]) -> Optional[float]:
        """Calculate cosine of angle formed at pos2 by pos1 and pos3."""
        if pos1 is None or pos2 is None or pos3 is None:
            return None
        vec1 = np.array([pos2[0] - pos1[0], pos2[1] - pos1[1]])
        vec2 = np.array([pos3[0] - pos2[0], pos3[1] - pos2[1]])
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return None
        cos_angle = np.dot(vec1, vec2) / (norm1 * norm2)
        return float(cos_angle)

    def _recompute_current_frame_state(self, frame: int) -> None:
        """Recompute frame state variables after a swap to ensure consistency.
        
        This method recalculates state variables for the current frame using 
        the updated (post-swap) data, ensuring that any subsequent processing
        in this frame uses the correct positions and distances.
        """
        # Recalculate current positions with post-swap data
        self.Xt0_curr = self._get_track_position(0, frame)
        self.Xt1_curr = self._get_track_position(1, frame)
        
        # Recalculate previous positions with post-swap data
        # This is crucial for the next frame's calculations
        if frame > 0:
            self.Xt0_prev = self._get_track_position(0, frame - 1)
            self.Xt1_prev = self._get_track_position(1, frame - 1)
            self.orient0_prev = self._get_track_orientation(0, frame - 1)
            self.orient1_prev = self._get_track_orientation(1, frame - 1)
        
        # Recalculate trajectory predictions since they depend on previous data
        self.Xt0_pred = self._predict_trajectory(0, frame)
        self.Xt1_pred = self._predict_trajectory(1, frame)
        
        # Recalculate inter-track distances
        self.d01 = self._distance(self.Xt0_curr, self.Xt1_pred)
        self.d10 = self._distance(self.Xt1_curr, self.Xt0_pred)

    def _handle_distance_jumps(self, frame: int) -> bool:
        """Distance-based jump handler. Returns True if any correction occurred."""
        tail0_curr = self.data.get_position(frame, 0, 'tail_base')
        tail1_curr = self.data.get_position(frame, 1, 'tail_base')
        spine0_curr = self.data.get_position(frame, 0, 'spine_base')
        spine1_curr = self.data.get_position(frame, 1, 'spine_base')
        neck0_curr = self.data.get_position(frame, 0, 'neck')
        neck1_curr = self.data.get_position(frame, 1, 'neck')
        tail0_prev = self.data.get_position(frame - 1, 0, 'tail_base')
        tail1_prev = self.data.get_position(frame - 1, 1, 'tail_base')
        spine0_prev = self.data.get_position(frame - 1, 0, 'spine_base')
        spine1_prev = self.data.get_position(frame - 1, 1, 'spine_base')
        neck0_prev = self.data.get_position(frame - 1, 0, 'neck')
        neck1_prev = self.data.get_position(frame - 1, 1, 'neck')
        headstage0_curr = self.data.get_position(frame, 0, 'headstage')
        headstage0_prev = self.data.get_position(frame - 1, 0, 'headstage')
        counths = 0
        nose1_curr = self.data.get_position(frame, 1, 'nose')
        nose0_prev = self.data.get_position(frame - 1, 0, 'nose')
        nose0_curr = self.data.get_position(frame, 0, 'nose')
        nose1_prev = self.data.get_position(frame - 1, 1, 'nose')
        count0_nose = 0
        if self._distance(nose1_curr, nose0_prev) < 8:
            count0_nose += 1
        count1_nose = 0
        if self._distance(nose0_curr, nose1_prev) < 8:
            count1_nose += 1
        if self._distance(headstage0_curr, headstage0_prev) < 8:
            counths += 1
        count0 = 0
        if self._distance(tail0_curr, tail1_prev) < 8:
            count0 += 1 
        if self._distance(spine0_curr, spine1_prev) < 9:
            count0 += 1
        if self._distance(neck0_curr, neck1_prev) < 9:
            count0 += 1
        count1 = 0
        if self._distance(tail1_curr, tail0_prev) < 8:
            count1 += 1
        if self._distance(spine1_curr, spine0_prev) < 9:
            count1 += 1
        if self._distance(neck1_curr, neck0_prev) < 9:
            count1 += 1
        count00 = 0
        if self._distance(tail0_curr, tail0_prev) < self.overlap_dist:
            count00 += 1
        if self._distance(spine0_curr, spine0_prev) < 8:
            count00 += 1
        if self._distance(neck0_curr, neck0_prev) < 8:
            count00 += 1
        if self._distance(headstage0_curr, headstage0_prev) < self.overlap_dist:
            count00 += 1
        if self._distance(nose0_curr, nose0_prev) < self.overlap_dist:
            count00 += 1
        count11 = 0
        if self._distance(tail1_curr, tail1_prev) < self.overlap_dist:
            count11 += 1
        if self._distance(spine1_curr, spine1_prev) < 8:
            count11 += 1
        if self._distance(neck1_curr, neck1_prev) < 8:
            count11 += 1
        if self._distance(nose1_curr, nose1_prev) < self.overlap_dist:
            count11 += 1
        count2 = 0
        if self._distance(tail0_curr, tail1_prev) < self.proximate_jump_thresh:
            count2 += 1 
        if self._distance(spine0_curr, spine1_prev) < self.proximate_jump_thresh:
            count2 += 1
        if self._distance(neck0_curr, neck1_prev) < self.proximate_jump_thresh:
            count2 += 1
        count3 = 0
        if self._distance(tail1_curr, tail0_prev) < self.proximate_jump_thresh:
            count3 += 1
        if self._distance(spine1_curr, spine0_prev) < self.proximate_jump_thresh:
            count3 += 1
        if self._distance(neck1_curr, neck0_prev) < self.proximate_jump_thresh:
            count3 += 1
        if  self.data.count_visible_nodes(frame-1,1) != 0 and ((self.data.count_visible_nodes(frame-1,0) != 0 and count00 / self.data.count_visible_nodes(frame-1,0) == 1.0) or (self.data.count_visible_nodes(frame,0) != 0 and count00 / self.data.count_visible_nodes(frame,0) == 1.0)) and count11 / self.data.count_visible_nodes(frame-1,1) >= 0.5:
            if counths == 1 and count00 == 1:
                pass
            else:
                return False

                
        if self.data.count_visible_nodes(frame-1,1) != 0 and self.data.count_visible_nodes(frame-1,0) != 0 and count11 / self.data.count_visible_nodes(frame-1,1) == 1.0 and count00 / self.data.count_visible_nodes(frame-1,0) >= 0.5:
            return False
        if ((count0 >= 3 or count1 >= 3) or (count0 >=2 and count1 >=2) or (count2 >=2 or count3 >=2)) and (self.d10 is not None and self.d10 < 2* self.proximity_threshold and self.d01 is not None and self.d01 < 2* self.proximity_threshold):
            self._swap_ids(frame)
            return True
        if self.data.count_visible_nodes(frame,0) == 0 and self.data.count_visible_nodes(frame,1) > 0 and (count11 + count1 + count0_nose) /self.data.count_visible_nodes(frame,1) == 1.0 and count11 > 0:
            changed = False
            if self._distance(spine1_curr, spine0_prev) is not None and self._distance(spine1_curr, spine0_prev) < 3:
                self.data.swap_node(frame, 'spine_base')
                changed = True
            if self._distance(tail1_curr, tail0_prev) is not None and self._distance(tail1_curr, tail0_prev) < 3:
                self.data.swap_node(frame, 'tail_base')
                changed = True
            if self._distance(neck1_curr, neck0_prev) is not None and self._distance(neck1_curr, neck0_prev) < 3:
                self.data.swap_node(frame, 'neck')
                changed = True
            if self._distance(nose1_curr, nose0_prev) is not None and self._distance(nose1_curr, nose0_prev) < 3:
                self.data.swap_node(frame, 'nose')
                changed = True
            if changed:
                return True
        if self.data.count_visible_nodes(frame,0) > 1 and count00 / self.data.count_visible_nodes(frame,0) == 1.0:
            if count11 == 0:
                if self.data.count_visible_nodes(frame,1) > 0 and count1 / self.data.count_visible_nodes(frame,1) > 0.6:
                    if self.jt1 is not None and self.jt1 > 3 * self.Jt:
                        self._extract_and_blank(frame, 1)
                        return True
                    if self._distance(spine1_curr, spine1_prev) is not None and self._distance(spine1_curr, spine1_prev) > 60:
                        if self._distance(tail1_curr, tail1_prev) is not None and self._distance(tail1_curr, tail1_prev) > 60:
                            if  self._distance(spine1_curr, spine0_prev) is not None and self._distance(spine1_curr, spine0_prev) < 3:
                                if self._distance(tail1_curr, tail0_prev) is not None and self._distance(tail1_curr, tail0_prev) < 3:
                                    self._extract_and_blank(frame, 1)
                                    return True
                    if self.jt1 is not None and self.jt1 == 0:
                        if count1 == 2 and count0_nose == 0:
                           if self.get_cross_visible_seperation(frame) is not None and self._distance(neck1_curr, neck0_prev) is not None and self.get_cross_visible_seperation(frame) - self._distance(neck1_curr, neck0_prev) == 0:
                               if abs(self.calculate_cosine(tail1_curr,spine1_curr, neck0_curr)-self.calculate_cosine(tail0_prev,spine0_prev, neck0_prev)) is not None and abs(self.calculate_cosine(tail1_curr,spine1_curr, neck0_curr)-self.calculate_cosine(tail0_prev,spine0_prev, neck0_prev)) < 0.01:
                                   self._extract_and_blank(frame, 1)
                                   return True
                           if self.get_cross_visible_seperation(frame) is not None and self._distance(neck1_curr, neck0_prev) is not None and self.get_cross_visible_seperation(frame) - self._distance(neck1_curr, neck0_prev) <= 3:
                               if abs(self.calculate_cosine(tail1_curr,spine1_curr, neck0_curr)-self.calculate_cosine(tail0_prev,spine0_prev, neck0_prev)) is not None and abs(self.calculate_cosine(tail1_curr,spine1_curr, neck0_curr)-self.calculate_cosine(tail0_prev,spine0_prev, neck0_prev)) < 0.1:
                                   if abs(self.calculate_cosine(tail1_curr,spine1_curr, headstage0_curr)-self.calculate_cosine(tail0_prev,spine0_prev, headstage0_prev)) is not None and abs(self.calculate_cosine(tail1_curr,spine1_curr, headstage0_curr)-self.calculate_cosine(tail0_prev,spine0_prev, headstage0_prev)) < 0.1:
                                       self._extract_and_blank(frame, 1)
                                       return True
                           if self.get_cross_visible_seperation(frame) is None:
                               if abs(self.calculate_cosine(tail1_curr,spine1_curr, neck0_curr)-self.calculate_cosine(tail0_prev,spine0_prev, neck0_prev))  is not None and abs(self.calculate_cosine(tail1_curr,spine1_curr, neck0_curr)-self.calculate_cosine(tail0_prev,spine0_prev, neck0_prev)) < 0.1:
                                   if abs(self.calculate_cosine(tail1_curr,spine1_curr, headstage0_curr)-self.calculate_cosine(tail0_prev,spine0_prev, headstage0_prev)) and abs(self.calculate_cosine(tail1_curr,spine1_curr, headstage0_curr)-self.calculate_cosine(tail0_prev,spine0_prev, headstage0_prev)) < 0.1:
                                       self._extract_and_blank(frame, 1)
                                       return True
                        if count1 == 1 and self.data.count_visible_nodes(frame,1) == 1:
                           if self._distance(neck1_curr, neck0_prev) <= 8:
                               if abs(self.calculate_cosine(tail0_curr,spine0_curr, neck1_curr)-self.calculate_cosine(tail0_prev,spine0_prev, neck0_prev)) is not None and abs(self.calculate_cosine(tail0_curr,spine0_curr, neck1_curr)-self.calculate_cosine(tail0_prev,spine0_prev, neck0_prev)) < 0.01:
                                   self._extract_and_blank(frame, 1)
                                   return True
                if self._distance(spine1_curr, spine0_prev) is not None and self._distance(spine1_curr, spine0_prev) == 0:
                    if self.get_cross_visible_seperation(frame) - self._distance(neck1_curr, neck0_prev) is not None and self.get_cross_visible_seperation(frame) - self._distance(neck1_curr, neck0_prev) == 0:
                        if self._distance(neck1_curr, neck1_prev) is not None and self._distance(neck1_curr, neck1_prev) > 30:
                            if self.jt1 is not None and self.jt1 > 60:
                                if abs(self._distance(headstage0_curr, tail0_prev) - self._distance(headstage0_curr, tail1_curr)) is not None and abs(self._distance(headstage0_curr, tail0_prev) - self._distance(headstage0_curr, tail1_curr)) < 9:
                                    self._extract_and_blank(frame, 1)
                                    return True
        if not np.isnan(headstage0_curr[0]) and np.isnan(headstage0_prev[0]) is not None and count00 >= 2:
            if count11 == 0 and self.data.count_visible_nodes(frame,1) > 0:
                if self.data.count_visible_nodes(frame,1) == 1:
                    if self._distance(self.data.get_position(frame+1,1,'neck'), neck1_curr) is not None and self._distance(self.data.get_position(frame+1,1,'neck'), neck1_curr) == 0:
                        if self._distance(self.data.get_position(frame+1,1,'spine_base'), spine0_curr) is not None and self._distance(self.data.get_position(frame+1,1,'spine_base'), spine0_curr) == 0:
                            if self._distance(self.data.get_position(frame+1,1,'tail_base'), tail0_curr) is not None and self._distance(self.data.get_position(frame+1,1,'tail_base'), tail0_curr) == 0:
                                self.data.swap_node(frame, 'neck')
                                return True




        if self.data.count_visible_nodes(frame,1) > 0 and (count1 + count0_nose) / self.data.count_visible_nodes(frame,1) == 1.0:
            if self.data.count_visible_nodes(frame,0) > 1 and count00 / self.data.count_visible_nodes(frame,0) >= 0.75:
                if self._distance(neck0_curr, neck0_prev) is not None and self._distance(neck0_curr, neck0_prev) > 26:
                    if self._distance(neck1_curr, neck0_prev) < 3:
                        self.data.swap_node(frame, 'neck')
                        if count11 == 0:
                            self._extract_and_blank(frame, 1)
                            return True
            if self.data.count_visible_nodes(frame,0) > 1 and count00 / self.data.count_visible_nodes(frame,0) >= 0.6:
                if self.jt1 is not None and self.jt1 > 3 * self.Jt:
                    if self._distance(neck1_curr, neck0_prev) is not None and self._distance(neck1_curr, neck0_prev) <= 4:
                        self.data.swap_node(frame, 'neck')
                        if count11 == 0:
                            self._extract_and_blank(frame, 1)
                            return True
                    if self._distance(neck1_curr, neck0_prev) is not None and self._distance(neck1_curr, neck0_prev) <= 8 and self._distance(neck0_curr, neck0_prev) is not None and self._distance(neck0_curr, neck0_prev) > self._distance(neck1_curr, neck0_prev):
                        self.data.swap_node(frame, 'neck')
                        if count11 == 0:
                            self._extract_and_blank(frame, 1)
                            return True

            if self._distance(spine0_curr, spine0_prev) is not None and self._distance(spine0_curr, spine0_prev) < 8:
                if self._distance(neck0_curr, neck0_prev) is not None and self._distance(neck0_curr, neck0_prev) > 30:
                    if self._distance(neck1_curr, neck0_prev) < 3:
                        if abs(self.calculate_cosine(spine0_curr,neck1_curr, headstage0_curr)-self.calculate_cosine(spine0_prev,neck0_prev, headstage0_prev)) is not None and abs(self.calculate_cosine(spine0_curr,neck1_curr, headstage0_curr)-self.calculate_cosine(spine0_prev,neck0_prev, headstage0_prev)) < 0.1:
                            self.data.swap_node(frame, 'neck')
                            if count11 == 0:
                                self._extract_and_blank(frame, 1)
                                return True
                    if self._distance(neck1_curr, neck0_prev) < 5:
                        if abs(self.calculate_cosine(spine0_curr,neck1_curr, headstage0_curr)-self.calculate_cosine(spine0_prev,neck0_prev, headstage0_prev)) is not None and abs(self.calculate_cosine(spine0_curr,neck1_curr, headstage0_curr)-self.calculate_cosine(spine0_prev,neck0_prev, headstage0_prev)) < 0.05:
                            if self.jt1 is not None and self.jt1 > 3 * self.Jt:
                                self.data.swap_node(frame, 'neck')
                                if count11 == 0:
                                    self._extract_and_blank(frame, 1)
                                    return True
            if self.data.count_visible_nodes(frame,0) == 2 and count00 == 1:
                if self._distance(neck0_curr, neck0_prev) is not None and self._distance(neck0_curr, neck0_prev) > 12:
                    if self.calculate_cosine(spine1_curr, neck1_curr, neck0_curr) is not None and self.calculate_cosine(spine1_curr, neck1_curr, neck0_curr) > 0.88:
                        if self._distance(neck1_curr, neck0_curr) is not None and self._distance(neck1_curr, neck0_curr) < 16:
                            if abs(self._distance(headstage0_curr, neck0_curr) - self._distance(headstage0_curr, neck1_curr)) is not None and abs(self._distance(headstage0_curr, neck0_curr) - self._distance(headstage0_curr, neck1_curr)) < 2:
                                if self._distance(neck1_curr, neck0_prev) < 3:
                                    self.data.swap_node(frame, 'neck')
                                    if count11 == 0:
                                        self._extract_and_blank(frame, 1)
                                        return True



        if self.data.count_visible_nodes(frame,1) > 0 and count11 / self.data.count_visible_nodes(frame,1) == 1.0:
            if count00 == 0:
                if self.data.count_visible_nodes(frame,0) > 1 and (count0 + count1_nose) / self.data.count_visible_nodes(frame,0) > 0.6:
                    if self.jt0 is not None and self.jt0 > 80:
                        self._extract_and_blank(frame, 0)
                        return True
        
        if self.get_cross_visible_seperation(frame) is not None and self.get_cross_visible_seperation(frame) <= 18:
            if self._distance(spine0_curr, spine1_curr) < 22:
                if self.jt1 is not None and self.jt1 > 2 * self.Jt:
                    if not np.isnan(neck0_curr[0]) and not np.isnan(neck1_curr[0]):
                        if self._distance(neck0_curr, neck1_curr) < 13:
                            if not np.isnan(tail0_curr[0]) and np.isnan(tail1_curr[0]):
                                if not np.isnan(nose1_curr[0]) and np.isnan(nose0_curr[0]):
                                    if count00 > 2 and count11 == 0:
                                        if self.data.score[frame,1,0] > 0.8:
                                            if self.calculate_cosine(spine0_curr,neck0_curr, nose1_curr) > 0.88:
                                                self.data.swap_node(frame, 'nose')
                                        self.data.blank_frame(frame, 1)
                                        self.corrections.append({'frame': frame, 'track': 1, 'type': 'jump_overlap_blank', 'action': f'Blanked track 1 due to jump with overlap'})
                                        return True
            if count00 > 2 and (count1 + count0_nose) > 1:
                if self.jt1 is not None and self.jt1 > 4 * self.Jt:
                    if self.jt0 is not None and self.jt0 < self.Jt:
                        if count0 + count1_nose == 0:
                            if counths == 1:
                                if self._distance(self.Xt1_curr, headstage0_curr) is not None and self._distance(self.Xt1_curr, headstage0_curr) < 30:
                                    if count11 == 0:
                                        if self.d01 is not None and self.d01 > 4 * self.Jt:
                                            if self._distance(self.Xt0_curr, self.Xt0_prev) is not None and self._distance(self.Xt0_curr, self.Xt0_prev) < 30:
                                                self._extract_and_blank(frame, 1)
                                                return True
                                
                if self.jt0 is not None and self.jt0 > 2 * self.Jt:
                    if not np.isnan(neck0_curr[0]) and not np.isnan(neck1_curr[0]):
                        if self._distance(neck0_curr, neck1_curr) < 13:
                            if np.isnan(tail1_curr[0]) and not np.isnan(tail0_curr[0]):
                                if not np.isnan(nose0_curr[0]) and np.isnan(nose1_curr[0]):
                                    if count00 <= 1 and count11 > 2:
                                        if self.data.score[frame,0,0] > 0.8:
                                            if self.calculate_cosine(spine1_curr,neck1_curr, nose0_curr) > 0.88:
                                                self.data.swap_node(frame, 'nose')
                                        self.data.blank_frame(frame, 0)
                                        self.corrections.append({'frame': frame, 'track': 0, 'type': 'jump_overlap_blank', 'action': f'Blanked track 0 due to jump with overlap'})
                                        return True
            if count11 > 1 and (count0 + count1_nose) > 1:
                if count00 == 0:
                    if self.jt1 is not None and self.jt1 < 31:
                        if self.jt0 is not None and self.jt0 == 0:
                            if self._distance(spine1_curr, spine1_prev) is not None and self._distance(spine1_curr, spine1_prev) <= 2 and self._distance(tail1_curr, tail1_prev) is not None and self._distance(tail1_curr, tail1_prev) == 0:
                                if self._distance(neck0_curr, neck1_prev) is not None and self._distance(neck0_curr, neck1_prev) <= 2:
                                    if self._distance(nose0_curr, nose1_prev) is not None and self._distance(nose0_curr, nose1_prev) <= 2:
                                        if np.isnan(nose1_curr[0]) and np.isnan(tail0_curr[0]) and np.isnan(spine0_curr[0]):
                                            # could add also tail_spine_neck
                                            if abs(self.calculate_cosine(tail1_prev, spine1_prev,nose1_prev) - self.calculate_cosine(tail1_curr, spine1_curr,nose0_curr)) is not None and abs(self.calculate_cosine(tail1_prev, spine1_prev,nose1_prev) - self.calculate_cosine(tail1_curr, spine1_curr,nose0_curr)) < 0.01:
                                                if not np.isnan(neck1_curr[0]):
                                                    self.data.swap_node(frame, 'neck')
                                                self._extract_and_blank(frame, 0)
                                                return True
                        

                                    



        if count1 == 3 and count11 == 0 and count00 < 2:
            if self.d10 is not None and self.d10 < 16 and self._distance(self.Xt1_curr,self.Xt0_prev) < 16:
                self._swap_ids(frame)
                return True
            if count0_nose == 1:
                if self.d10 is not None and self.jt1 is not None and abs(self.d10 - self.jt1) < 18:
                    if counths == 1:
                        if self.calculate_cosine(spine1_curr, neck1_curr, headstage0_curr) is not None and self.calculate_cosine(spine1_curr, neck1_curr, headstage0_curr) >  0.69:
                            self._swap_ids(frame)
                            return True
                if self.jt1 is not None and self.jt1 > 3 * self.Jt:
                    self._swap_ids(frame)
                    return True
            if count0_nose == 0:
                if self._distance(self.Xt1_curr,self.Xt1_pred) is not None and self._distance(self.Xt1_curr,self.Xt1_pred) <= 2:
                    self._swap_ids(frame)
                    return True
                if np.isnan(nose0_curr[0]) and not np.isnan(nose0_prev[0]):
                    self._swap_ids(frame)
                    return True
            if counths == 1:
                if self.jt1 is not None and self.jt1 > self.Jt:
                    if abs(self._distance(spine1_curr, headstage0_curr) - self._distance(spine0_prev, headstage0_curr)) < 6:
                        if self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) is not None and self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) > 0.69:
                            self._swap_ids(frame)
                            return True
                if self.jt1 is not None and self.jt1 == 0:
                    if self.Xt0_pred is not None and self._distance(self.Xt0_pred,spine1_curr) < 32:
                        if self._distance(spine1_curr,self.Xt0_prev) < 32:
                           if self.calculate_cosine(spine1_curr, neck1_curr, headstage0_curr) > 0.6:
                             self._swap_ids(frame)
                             return True
                    if abs(self.calculate_cosine(tail0_prev, spine0_prev,headstage0_prev) - self.calculate_cosine(tail1_curr, spine1_curr,headstage0_curr)) is not None and abs(self.calculate_cosine(tail0_prev, spine0_prev,headstage0_prev) - self.calculate_cosine(tail1_curr, spine1_curr,headstage0_curr)) < 0.01:
                        self._swap_ids(frame)
                        return True
                if self._distance(spine1_curr, self.Xt0_prev) is not None and self._distance(spine1_curr, self.Xt0_prev) < 9:
                    if self.calculate_cosine(spine1_curr, neck1_curr, headstage0_curr) is not None and self.calculate_cosine(spine1_curr, neck1_curr, headstage0_curr) > 0.88:
                        self._swap_ids(frame)
                        return True
            if not np.isnan(nose1_curr[0]) and np.isnan(nose0_prev[0]):
                if self._distance(self.Xt1_curr,self.Xt0_prev) < 40:
                    if self.Xt0_pred is not None and self._distance(self.Xt0_pred,spine1_curr) < 17:
                        if self.calculate_cosine(tail1_curr, spine1_curr, nose1_curr) > 0.45:
                            self._swap_ids(frame)
                            return True
                        if  self._distance(spine1_curr, spine0_prev) < 4:
                            self._swap_ids(frame)
                            return True
                    if self.d10 is not None and self.d10 < 35:
                        if self.calculate_cosine(spine1_curr, neck1_curr, headstage0_curr) is not None and self.calculate_cosine(spine1_curr, neck1_curr, headstage0_curr) > 0.45:
                            self._swap_ids(frame)
                            return True
            if np.isnan(nose1_curr[0]) and not np.isnan(nose0_prev[0]):
                if self.Xt0_pred is not None and self._distance(spine1_curr, self.Xt0_pred) < 30:
                    if self._distance(spine0_prev, self.Xt0_pred) < 30:
                        if self.calculate_cosine(tail1_curr, spine1_curr, nose0_prev) > 0.69:
                            self._swap_ids(frame)
                            return True
        if count0 == 3 and count00 < 2 and count11 == 0:
            if self.d01 is not None and self.d01 < 16 and self._distance(self.Xt0_curr,self.Xt1_prev) < 16:
                self._swap_ids(frame)
                return True
            if count1_nose == 1:
                if self.d01 is not None and self.jt0 is not None and abs(self.d01 - self.jt0) < 18:
                    if counths == 1:
                        if self.calculate_cosine(spine0_curr, neck0_curr, headstage0_curr) is not None and self.calculate_cosine(spine0_curr, neck0_curr, headstage0_curr) < - 0.45:
                            self._swap_ids(frame)
                            return True
            if not np.isnan(nose0_curr[0]) and np.isnan(nose1_prev[0]):
                if self._distance(self.Xt0_curr,self.Xt1_prev) < 30:
                    if self._distance(self.Xt1_pred,spine0_curr) < 16:
                        if self.calculate_cosine(tail0_curr, spine0_curr, nose0_curr) > 0.69:
                            self._swap_ids(frame)
                            return True
            if np.isnan(nose0_curr[0]) and not np.isnan(nose1_prev[0]):
                if self.Xt1_pred is not None and self._distance(spine1_curr, self.Xt1_pred) < 30:
                    if self._distance(spine1_prev,self.Xt1_pred) < 30:
                        if self.calculate_cosine(tail0_curr, spine0_curr, nose1_prev) > 0.69:
                            self._swap_ids(frame)
                            return True
        if count1 == 2 and count11 < 2 and count00 < 2:
            if self._distance(neck1_curr, neck0_prev) is not None and self._distance(neck1_curr, neck0_prev) < 20:
                if self._distance(spine1_curr, self.Xt0_prev) is not None and self._distance(spine1_curr, self.Xt0_prev) < 11:
                    if self.calculate_cosine(spine1_curr, neck1_curr, headstage0_curr) is not None and self.calculate_cosine(spine1_curr, neck1_curr, headstage0_curr) > 0.88:
                        self._swap_ids(frame)
                        return True
            if self._distance(self.Xt1_curr, self.Xt0_prev) < 6:
                if self.d10 is not None and self.d10 < 10:
                    if self.calculate_cosine(spine1_curr, neck1_curr, headstage0_curr) is not None and self.calculate_cosine(spine1_curr, neck1_curr, headstage0_curr) > 0.88:
                        self._swap_ids(frame)
                        return True
            if not np.isnan(neck1_curr[0]) and np.isnan(neck0_prev[0]):
                if self.jt1 is not None and self.jt1 > 4 * self.Jt:
                    if self.data.score[frame,1,1] < 0.35:
                        if self.Xt0_pred is not None and self._distance(self.Xt0_pred,spine1_curr) < 40:
                            self.data.blank_node(frame,1,1)
                            self._swap_ids(frame)
                            return True
                if self.d10 is not None and self.d10 < 30:
                    if self._distance(self.Xt1_curr, self.Xt0_prev) < 30:
                        if self.calculate_cosine(tail1_curr, spine1_curr, neck1_curr) > 0.69:
                            self._swap_ids(frame)
                            return True
                if not np.isnan(nose1_curr[0]) and np.isnan(nose0_prev[0]):
                    if self._distance(self.Xt0_pred,spine1_curr) < 35 and self._distance(spine0_prev,self.Xt0_pred) < 35:
                        if self.calculate_cosine(tail1_curr, spine1_curr, neck1_curr) > 0.69:
                            if self.calculate_cosine(tail1_curr, spine1_curr, nose1_curr) > 0.69:
                                self._swap_ids(frame)
                                return True
            if np.isnan(neck1_curr[0]) and not np.isnan(neck0_prev[0]):
                if np.isnan(nose1_curr[0]) and not np.isnan(nose0_prev[0]):
                    if self._distance(self.Xt0_pred,spine1_curr) < 35 and self._distance(spine0_prev,self.Xt0_pred) < 35:
                        if self.calculate_cosine(tail1_curr, spine1_curr, neck0_prev) > 0.69:
                            self._swap_ids(frame)
                            return True
            if count0_nose == 1:
                if self.d10 is not None and self.d10 < 16 and self._distance(self.Xt1_curr,self.Xt0_prev) < 16:
                    if self._distance(spine1_curr, spine0_prev) < 6:
                        self._swap_ids(frame)
                        return True
            if not np.isnan(nose1_curr[0]) and np.isnan(nose0_prev[0]):
                if not np.isnan(spine1_curr[0]) and np.isnan(spine0_prev[0]):
                    if self.d10 is not None and self.d10 < 30:
                        if self.Xt0_prev is not None and self._distance(self.Xt1_curr, self.Xt0_prev) < 30:
                            if self.calculate_cosine(tail1_curr, neck1_curr,headstage0_curr) is not None and self.calculate_cosine(tail1_curr, neck1_curr,headstage0_curr) > 0.69:
                                self._swap_ids(frame)
                                return True
                if np.isnan(tail1_curr[0]) and np.isnan(tail0_prev[0]):
                    if self.jt1 is not None and self.jt1 == 0 and self.jt0 is None and self.d01 is None:
                        if self._distance(spine1_curr,self.Xt0_prev) < 30:
                            self._swap_ids(frame)
                            return True
                        if self._distance(self.Xt1_curr, self.Xt0_prev) < 35 and self._distance(spine1_curr,self.Xt0_prev) < 35:
                            self._swap_ids(frame)
                            return True
                if not np.isnan(tail1_curr[0]) and np.isnan(tail0_prev[0]):
                    if self.jt1 is not None and self.jt1 == 0 and self.jt0 is None and self.d01 is None:
                        if self._distance(spine1_curr,self.Xt0_prev) < 30 and self.Xt0_pred is not None and self._distance(self.Xt0_pred,spine1_curr) < 33:
                            self._swap_ids(frame)
                            return True
                if self._distance(spine1_curr, spine0_prev) is not None and self._distance(spine1_curr, spine0_prev) < 12:
                    if self.jt1 is not None and self.jt1 == 0:
                        if self.d10 is not None and self.d10 < 20:
                            if self._distance(self.Xt1_curr, self.Xt0_prev) < 20:
                                self._swap_ids(frame)
                                return True
            if self._distance(spine1_curr, spine0_prev) is not None and self._distance(spine1_curr, spine0_prev) == 0:
                if abs(self.calculate_cosine(tail0_prev, spine0_prev,headstage0_prev) - self.calculate_cosine(tail1_curr, spine1_curr,headstage0_curr)) is not None and abs(self.calculate_cosine(tail0_prev, spine0_prev,headstage0_prev) - self.calculate_cosine(tail1_curr, spine1_curr,headstage0_curr)) < 0.01:
                    self._swap_ids(frame)
                    return True
            if self.d10 is not None and self.d10 < 12:
                if self._distance(spine1_curr, spine0_prev) is not None and self._distance(spine1_curr, spine0_prev) < 1:
                    if counths == 1:
                        if abs(self.calculate_cosine(tail0_prev, spine0_prev,headstage0_prev) - self.calculate_cosine(tail1_curr, spine1_curr,headstage0_curr)) is not None and abs(self.calculate_cosine(tail0_prev, spine0_prev,headstage0_prev) - self.calculate_cosine(tail1_curr, spine1_curr,headstage0_curr)) < 0.01:
                            if self._distance(neck1_curr, neck0_prev) is not None and self._distance(neck1_curr, neck0_prev) > 30:
                                if self._distance(neck0_curr, neck0_prev) is not None and self._distance(neck0_curr, neck0_prev) < 13:
                                    if self.data.count_visible_nodes(frame,0) is not None and self.data.count_visible_nodes(frame,0) == 2:
                                        if self.jt1 is not None and self.jt1 == 0:
                                            self._extract_and_blank(frame, 1)
                                            return True



        if count0 == 2 and count00 < 2 and count11 == 0:
            if not np.isnan(neck0_curr[0]) and np.isnan(neck1_prev[0]):
                if self.jt0 is not None and self.jt0 > 4 * self.Jt:
                    if self.data.score[frame,0,1] < 0.35:
                        if self._distance(self.Xt1_pred,spine0_curr) < 40:
                            self.data.blank_node(frame,0,1)
                            self._swap_ids(frame)
                            return True
                if not np.isnan(nose0_curr[0]) and np.isnan(nose1_prev[0]):
                    if self.Xt1_pred is not None and self._distance(self.Xt1_pred,spine0_curr) < 35 and self._distance(spine1_prev,self.Xt1_pred) < 35:
                        if self.calculate_cosine(tail0_curr, spine0_curr, neck0_curr) > 0.69:
                            if self.calculate_cosine(tail0_curr, spine0_curr, nose0_curr) > 0.69:
                                self._swap_ids(frame)
                                return True
            if np.isnan(neck0_curr[0]) and not np.isnan(neck1_prev[0]):
                if np.isnan(nose0_curr[0]) and not np.isnan(nose1_prev[0]):
                    if self.Xt1_pred is not None and self._distance(self.Xt1_pred,spine0_curr) < 35 and self._distance(spine1_prev,self.Xt1_pred) < 35:
                        if self.calculate_cosine(tail0_curr, spine0_curr, neck1_prev) > 0.69:
                            self._swap_ids(frame)
                            return True
            if not np.isnan(tail0_curr[0]) and np.isnan(tail1_prev[0]):
                if self._distance(nose0_curr, nose1_prev) is not None and self._distance(nose0_curr, nose1_prev) < 9:
                    if self.jt1 is not None and self.jt1 > 3 * self.Jt:
                        if self.d01 is not None and self.d01 < 23:
                            self._swap_ids(frame)
                            return True

            if count1_nose == 1:
                if self.d01 is not None and self.d01 < 16 and self._distance(self.Xt0_curr,self.Xt1_prev) < 16:
                    if self._distance(spine0_curr, spine1_prev) < 6:
                        self._swap_ids(frame)
                        return True
                if self._distance(spine0_curr, spine1_prev) < 6:
                    if count00 == 0:
                        self._swap_ids(frame)
                        return True

        if count1 == 2:
            if (count00 == 3 and counths == 0) or (count00 >= 4 and counths ==1):
                if count11 == 0:
                    if count0 == 0:
                        if not np.isnan(headstage0_curr[0]):
                            if not np.isnan(spine1_curr[0]) and not np.isnan(spine0_curr[0]):
                                if self._distance(headstage0_curr, spine0_curr) > 120 and self._distance(headstage0_curr, spine1_curr) < 100:
                                    if self.calculate_cosine(spine1_curr, neck1_curr, headstage0_curr) is not None and self.calculate_cosine(spine1_curr, neck1_curr, headstage0_curr) > 0.88:
                                        if self.jt1 is not None and self.jt1 > 90:
                                            self._swap_ids(frame)
                                            return True
            if count00 == 2:
                if counths == 1:
                    if count11 == 0:
                        if self.Xt0_pred is not None and self.Xt1_pred is not None and self._distance(spine0_curr, self.Xt0_pred) > 27 and self._distance(spine1_curr, self.Xt0_pred) < 19:
                            if self.calculate_cosine(spine1_curr, neck1_curr,headstage0_curr) is not None and self.calculate_cosine(spine1_curr, neck1_curr,headstage0_curr) > 0.88:
                                if self._distance(headstage0_curr, spine0_curr) > 120 and self._distance(headstage0_curr, spine1_curr) < 100:
                                    if self.jt1 is not None and self.jt1 > 30:
                                        self._swap_ids(frame)
                                        return True
            if count11 == 1:
                if count00 < 2:
                    if  not np.isnan(neck1_curr[0]) and not np.isnan(neck0_prev[0]) and not np.isnan(neck1_prev[0]):
                        if self._distance(neck1_curr, neck1_prev) < 9 and self._distance(neck1_curr, neck0_prev) < 23:
                            if self._distance(self.Xt0_prev,spine1_curr) < 15:
                                if self.calculate_cosine(tail1_curr, spine1_curr,neck1_curr) > 0.69:
                                    if self.data.count_visible_nodes(frame,0) == 0:
                                        self._swap_ids(frame)
                                        return True
                    if not np.isnan(neck1_curr[0]) and not np.isnan(neck0_curr[0]) and not np.isnan(neck1_prev[0]) and np.isnan(neck0_prev[0]):
                        if self._distance(neck1_curr, neck1_prev) < 9 and self._distance(neck1_curr, neck0_curr) > 60:
                             if self._distance(spine1_curr, self.Xt0_prev) < 30:
                                 if self.jt0 is not None and self.jt0 > 100:
                                     if self.calculate_cosine(tail1_curr, spine1_curr,headstage0_curr) is not None and self.calculate_cosine(tail1_curr, spine1_curr,headstage0_curr) > 0.69:
                                        self._swap_ids(frame)
                                        return True
                        

        if self._distance(spine0_curr, spine1_prev) < 6:
            if not np.isnan(nose0_curr[0]) and np.isnan(nose1_prev[0]):
                if self.jt0 is not None and self.jt0 > 50:
                    if self.d01 is not None and abs(self.d01 - self.jt0) < 6:
                        if self.data.score[frame,0,0] < 0.35:
                            self.data.blank_node(frame,0,0)
                            self._swap_ids(frame)
                            return True
                    if self.jt0 > 4 * self.Jt:
                        if self.Xt1_pred is not None and self._distance(spine0_curr, self.Xt1_pred) < 30 and self._distance(spine1_prev, self.Xt1_pred) < 30:
                            self._swap_ids(frame)
                            return True
                    if self.d01 is not None and abs(self.d01 - self.jt0) < 23:
                        if self.calculate_cosine(tail0_curr, spine0_curr, headstage0_prev) is not None and self.calculate_cosine(tail0_curr, spine0_curr, headstage0_prev) < -0.45:
                            self._swap_ids(frame)
                            return True
            if self._distance(spine0_curr, spine1_prev) < 1:
                if self.jt0 is not None and self.jt0 > 50:
                    if self.d01 is not None and abs(self.d01 - self.jt0) < 7:
                        if self._distance(self.Xt1_pred,spine0_curr) < 35:
                            self._swap_ids(frame)
                            return True

        if self._distance(spine1_curr, spine0_prev) < 7:
            if not np.isnan(nose1_curr[0]) and np.isnan(nose0_prev[0]):
                if self.jt1 is not None and self.jt1 > 50:
                    if self.d10 is not None and abs(self.d10 - self.jt1) < 6:
                        if self.data.score[frame,1,0] < 0.35:
                            self.data.blank_node(frame,1,0)
                            self._swap_ids(frame)
                            return True
                        if not np.isnan(nose1_prev[0]) and self._distance(nose1_curr, nose1_prev) < 6:
                            self._swap_ids(frame)
                            self.data.swap_node(frame,'nose')
                            return True
                    if self.jt1 > 4 * self.Jt:
                        if self.Xt0_pred is not None and self._distance(spine1_curr, self.Xt0_pred) < 30 and self._distance(spine0_prev, self.Xt0_pred) < 30:
                            self._swap_ids(frame)
                            return True 
            if not np.isnan(neck1_curr[0]) and np.isnan(neck0_prev[0]):
                if self.jt1 is not None and self.jt1 > 50:
                    if self.d10 is not None and self.d10 < 30:
                        if self._distance(self.Xt1_curr, self.Xt0_prev) < 30:
                            if self._distance(spine1_curr, headstage0_curr) < 110:
                                self._swap_ids(frame)
                                return True
            if self._distance(spine1_curr, spine0_prev) < 1:
                if self.jt1 is not None and self.jt1 > 50:
                    if self.d10 is not None and abs(self.d10 - self.jt1) < 7:
                        if self._distance(self.Xt0_pred,spine1_curr) < 35:
                            self._swap_ids(frame)
                            return True
                if self.jt1 is not None and self.jt1 == 0:
                    if self._distance(self.Xt1_curr, self.Xt0_prev) is not None and self._distance(self.Xt1_curr, self.Xt0_prev) < 7:
                        if self._distance(spine1_curr, self.Xt0_pred) is not None and self._distance(spine1_curr, self.Xt0_pred) < 35:
                            self._swap_ids(frame)
                            return True
            if not np.isnan(neck1_curr[0]) and not np.isnan(neck0_prev[0]):
                if  self._distance(neck1_curr, neck0_prev) > 50:
                    if self.Xt0_pred is not None and self._distance(self.Xt0_pred,spine1_curr) < 12:
                        if self.Xt1_pred is not None and self._distance(spine1_curr, self.Xt1_pred) > 60:
                            if self.calculate_cosine(spine1_curr, neck0_prev, headstage0_curr) is not None and self.calculate_cosine(spine1_curr, neck0_prev, headstage0_curr) > 0.69:
                                self._swap_ids(frame)
                                return True
            if np.isnan(tail1_curr[0]) and np.isnan(tail0_prev[0]):
                if self._distance(neck1_curr, neck0_prev) is not None and self._distance(neck1_curr, neck0_prev) > 50:
                    if count0_nose == 1:
                        if self.d10 is not None and self.d10 < 30:
                            if self._distance(self.Xt1_curr,self.Xt0_prev) < 30:
                                if self.jt1 is not None and self.jt1 > 4 * self.Jt:
                                    if self.calculate_cosine(spine1_curr, neck0_prev,nose1_curr) > 0.69:
                                        if self.data.score[frame,1,1] < 0.6:
                                            self.data.blank_node(frame,1,1)
                                        self._swap_ids(frame)
                                        return True
                if not np.isnan(neck1_curr[0]) and np.isnan(neck0_prev[0]):
                    if np.isnan(nose1_curr[0]) and not np.isnan(nose0_prev[0]):
                        if self.d10 is not None and self.d10 < 40:
                            if self.jt1 is not None and self.jt1 > 50:
                                self._swap_ids(frame)
                                return True
                                    

            if count0_nose == 1:
                if self.d10 is not None and self.d10 < 16 and self._distance(self.Xt1_curr,self.Xt0_prev) < 16 and count00 < 2:
                    self._swap_ids(frame)
                    return True
                

        if self._distance(tail0_curr, tail1_prev) < 6:
            if self._distance(spine0_curr, spine1_prev) < 12:
                if np.isnan(neck0_curr[0]) and not np.isnan(neck1_prev[0]):
                    if not np.isnan(nose0_curr[0]) and np.isnan(nose1_prev[0]):
                        if self.calculate_cosine(tail0_curr, spine0_curr, nose0_curr) > 0.69:
                            if self._distance(self.Xt0_curr,self.Xt1_prev) < 35:
                                # can add headstage cosine
                                self._swap_ids(frame)
                                return True
                            if self.jt0 is not None and self.jt0 > 50:
                                if self.d01 is not None and abs(self.d01 - self.jt0) < 6:
                                    self._swap_ids(frame)
                                    return True
                if not np.isnan(neck0_curr[0]) and np.isnan(neck1_prev[0]):
                    if np.isnan(nose0_curr[0]) and not np.isnan(nose1_prev[0]):
                        if self.calculate_cosine(tail0_curr, spine0_curr, neck0_curr) > 0.69:
                            if self._distance(self.Xt0_curr,self.Xt1_prev) < 35:
                                # can add headstage cosine
                                self._swap_ids(frame)
                                return True
            if self._distance(spine0_curr, spine1_prev) < 18:
                if self.d01 is not None and self.Xt1_prev is not None and self.d01 < 30 and self._distance(self.Xt0_curr,self.Xt1_prev) < 25:
                    if self.jt1 is not None and self.jt1 > self.Jt:
                        if self.calculate_cosine(tail0_curr, spine0_curr, neck0_curr) is not None and self.calculate_cosine(tail0_curr, spine0_curr, neck0_curr) > 0.69:
                            if self.calculate_cosine(tail0_curr, spine0_curr, headstage0_curr) is not None and self.calculate_cosine(tail0_curr, spine0_curr, headstage0_curr) > 0.69:
                                if self.calculate_cosine(spine1_curr, neck1_curr, headstage0_curr) is not None and self.calculate_cosine(spine1_curr, neck1_curr, headstage0_curr) < - 0.3:  
                                    self._swap_ids(frame)
                                    return True
                if self.jt0 is not None and self.jt0 > 4 * self.Jt:
                    if self.Xt1_prev is not None and self._distance(spine0_curr, self.Xt1_prev) < 18:
                        if self.data.count_visible_nodes(frame,1) == 0 and self.data.count_visible_nodes(frame-1,0) == 0:
                            self._swap_ids(frame)
                            return True

        
        if self._distance(tail1_curr, tail0_prev) < 6:
            if self._distance(spine1_curr, spine0_prev) < 12:
                if np.isnan(neck1_curr[0]) and not np.isnan(neck0_prev[0]):
                    if not np.isnan(nose1_curr[0]) and np.isnan(nose0_prev[0]):
                        if self.calculate_cosine(tail1_curr, spine1_curr, nose1_curr) > 0.69:
                            if self._distance(self.Xt1_curr,self.Xt0_prev) < 35:
                                # can add headstage cosine
                                self._swap_ids(frame)
                                return True
                            if self.jt1 is not None and self.jt1 > 50:
                                if abs(self.d10 - self.jt1) < 6:
                                    self._swap_ids(frame)
                                    return True
                if np.isnan(neck1_curr[0]) and np.isnan(neck0_prev[0]):
                    if not np.isnan(nose1_curr[0]) and np.isnan(nose0_prev[0]):
                        if self.jt1 is not None and self.jt1 > 3 * self.Jt:
                            if self.calculate_cosine(tail1_curr, spine0_prev, headstage0_curr) is not None and self.calculate_cosine(tail1_curr, spine0_prev, headstage0_curr) > 0.69:
                                if self._distance(spine1_curr, spine0_prev) < 10:
                                    self._swap_ids(frame)
                                    return True
                if not np.isnan(neck1_curr[0]) and np.isnan(neck0_prev[0]):
                    if np.isnan(nose1_curr[0]) and not np.isnan(nose0_prev[0]):
                        if self.calculate_cosine(tail1_curr, spine1_curr, neck1_curr) > 0.69:
                            if self._distance(self.Xt1_curr,self.Xt0_prev) < 35:
                                # can add headstage cosine
                                self._swap_ids(frame)
                                return True
            if self._distance(spine1_curr, spine0_prev) < 18:
                if self.d10 is not None and self.Xt0_prev is not None and self.d10 < 30 and self._distance(self.Xt1_curr,self.Xt0_prev) < 25:
                    if self.jt0 is not None and self.jt0 > self.Jt:
                        if self.calculate_cosine(tail1_curr, spine1_curr, neck1_curr) is not None and self.calculate_cosine(tail1_curr, spine1_curr, neck1_curr) > 0.69:
                            if self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) is not None and self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) > 0.69:
                                if self.calculate_cosine(spine0_curr, neck0_curr, headstage0_curr) is not None and self.calculate_cosine(spine0_curr, neck0_curr, headstage0_curr) < - 0.3:  
                                    self._swap_ids(frame)
                                    return True
        if self._distance(neck1_curr, neck0_prev) < 6:
            if np.isnan(tail1_curr[0]) and not np.isnan(tail0_prev[0]):
                if self._distance(spine1_curr, spine0_prev) < 13:
                    if self.Xt1_pred is not None and self._distance(spine1_curr, self.Xt1_pred) > 50:
                        if self._distance(self.Xt0_prev,spine1_curr) < 18:
                            if self.calculate_cosine(spine1_curr, neck1_curr,headstage0_curr) is not None and self.calculate_cosine(spine1_curr, neck1_curr,headstage0_curr) > 0.6:
                                self._swap_ids(frame)
                                return True
            if not np.isnan(tail1_curr[0]) and np.isnan(tail0_prev[0]):
                if self.jt1 is not None and self.jt1 > self.Jt:
                    if self.d10 is not None and abs(self.d10 - self.jt1) < 21:
                        if count00 < 2 and count11 == 0:
                            self._swap_ids(frame)
                            return True
            if np.isnan(tail1_curr[0]) and np.isnan(tail0_prev[0]):
                if self._distance(spine1_curr, spine0_prev) < 10:
                    if not np.isnan(nose1_curr[0]) and np.isnan(nose0_prev[0]):
                        if self.jt1 is not None and self.jt1 == 0 and self.jt0 is None and self.d01 is None and self._distance(self.Xt1_curr, self.Xt0_prev) < 38:
                            if self.calculate_cosine(spine1_curr, neck1_curr, self.data.get_position(frame-2,0,'headstage')) is not None and self.calculate_cosine(spine1_curr, neck1_curr, self.data.get_position(frame-2,0,'headstage')) > 0.88:
                                self._swap_ids(frame)
                                return True
                if np.isnan(spine1_curr[0]) and np.isnan(spine0_prev[0]):
                    if count0_nose == 1:
                        if self._distance(self.Xt1_curr, self.Xt0_prev) is not None and self._distance(self.Xt1_curr, self.Xt0_prev) < 6:
                            if count00 < 2 and count11 == 0:
                                self._swap_ids(frame)
                                return True
                    
        if self._distance(neck0_curr, neck1_prev) < 6:
            if np.isnan(tail0_curr[0]) and np.isnan(tail1_prev[0]):
                if np.isnan(spine0_curr[0]) and np.isnan(spine1_prev[0]):
                    if self._distance(nose0_curr, nose1_prev) < 6:
                        if self.jt1 is not None and self.jt1 > 3 * self.Jt:
                            if count11 == 0 and count00 < 2:
                                self._swap_ids(frame)
                                return True
        if self.jt1 is not None and self.jt1 > 4 * self.Jt:
            if count00 == 0 and count11 == 0:
                if count0 + count1_nose > 1:
                   if self.d01 is not None and self.d01 < 6:
                    self._swap_ids(frame)
                    return True
                if self.jt0 is not None and self.jt0 > 3 * self.Jt:
                    if self._distance(self.Xt0_curr, self.Xt1_prev) is not None and self._distance(self.Xt0_curr, self.Xt1_prev) < 20:
                        self._swap_ids(frame)
                        return True




        if self.jt1 is not None and self.jt0 is not None and self.jt1 > 3 * self.Jt and self.jt0 > 3 * self.Jt:
            if self._distance(spine0_curr, self.Xt1_prev) is not None and self._distance(spine0_curr, self.Xt1_prev) < 23:
                if self.calculate_cosine(tail1_curr, spine1_curr,headstage0_prev) is not None and self.calculate_cosine(tail1_curr, spine1_curr,headstage0_prev) > 0.69:
                    if self.calculate_cosine(spine0_curr, neck0_curr,headstage0_prev) is not None and self.calculate_cosine(spine0_curr, neck0_curr,headstage0_prev) < -0.45:
                        if (self._distance(spine0_curr, headstage0_prev) - self._distance(spine1_curr, headstage0_prev)) > 20:
                            self._swap_ids(frame)
                            return True
        if self.jt1 is not None and self.jt1 > self.Jt:
            if self.d10 is not None and self.d10 < 7 and self.Xt0_prev is not None and self._distance(self.Xt1_curr, self.Xt0_prev) < 7:
                if count1 + count0_nose >= 1 and count11 == 0 and count00 <= 1:
                    if self.Xt0_pred is not None and self._distance(self.Xt0_pred,spine1_curr) > 50:
                        if self.calculate_cosine(spine1_curr, neck1_curr, headstage0_curr) is not None and self.calculate_cosine(spine1_curr, neck1_curr, headstage0_curr) > 0.69:
                            self._swap_ids(frame)
                            return True
            if counths == 1 and self.data.count_visible_nodes(frame,0) == 1 and self.data.count_visible_nodes(frame-1,0) == 1 and self.data.count_visible_nodes(frame,1) > 0 and self.data.count_visible_nodes(frame-1,1) > 0:
                if self.Xt1_prev is not None and self._distance(spine1_curr, self.Xt1_prev) > 50:
                    if np.isnan(tail1_curr[0]) and np.isnan(tail1_prev[0]):
                        if self.calculate_cosine(spine1_prev, neck1_prev,headstage0_prev) is not None and self.calculate_cosine(spine1_prev, neck1_prev,headstage0_prev) < - 0.45:
                            if self.calculate_cosine(spine1_curr, neck1_curr,headstage0_curr) is not None and self.calculate_cosine(spine1_curr, neck1_curr,headstage0_curr) > 0.69:
                                if self._distance(spine1_curr, spine1_prev) > 28:
                                    if self.calculate_cosine(spine1_prev, neck1_prev,nose1_prev) is not None and self.calculate_cosine(spine1_prev, neck1_prev,nose1_prev) > 0.69:
                                        if self.calculate_cosine(spine1_curr, neck1_curr,nose1_prev) is not None and self.calculate_cosine(spine1_curr, neck1_curr,nose1_prev) < - 0.45:
                                            if self._distance(nose1_curr, nose1_prev) < 6:
                                                self._swap_ids(frame)
                                                # self.data.swap_node(frame,'nose')
                                                return True

        if self.jt0 is not None and self.jt0 > self.Jt:
            if self.d01 is not None and self.d01 < 7 and self.Xt1_prev is not None and self._distance(self.Xt0_curr, self.Xt1_prev) < 7:
                if count0 + count1_nose >= 1 and count11 == 0 and count00 <= 1:
                    if self.Xt1_pred is not None and self._distance(self.Xt1_pred,spine0_curr) > 50:
                        if self.calculate_cosine(spine0_curr, neck0_curr, headstage0_curr) is not None and self.calculate_cosine(spine0_curr, neck0_curr, headstage0_curr) < -0.45:
                            self._swap_ids(frame)
                            return True
            if count00 < 2 and count11 == 0:
                if self.Xt0_pred is not None and self.Xt1_pred is not None and self.jt0 > self.d01 + 5:
                    if self.calculate_cosine(spine0_curr, neck0_curr, headstage0_prev) is not None and self.calculate_cosine(spine0_curr, neck0_curr, headstage0_prev) < -0.69:
                        if self.calculate_cosine(spine0_prev, neck0_prev, headstage0_prev) is not None and self.calculate_cosine(spine0_prev, neck0_prev, headstage0_prev) > 0.69:
                            if self.jt1 is None:
                                self._swap_ids(frame)
                                return True

        if counths <= 1 and (count0 + count1 + count1_nose + count0_nose + count00 + count11) < 2:
            if self.data.count_visible_nodes(frame,0) <= 1 and self.data.count_visible_nodes(frame-1,0) <= 1:      
                if self._distance(spine1_curr, spine1_prev) is not None and self._distance(spine1_curr, spine1_prev) > 50:
                    if self._distance(spine1_curr, self.Xt1_pred) is not None and self._distance(spine1_curr, self.Xt1_pred) > 2 * self.Jt:    
                        if self.jt1 is not None and self.jt1 > self.Jt:
                            if self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) is not None and self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) > 0.69:
                                if self._distance(spine1_curr, headstage0_curr) < 110:
                                    self._swap_ids(frame)
                                    return True
                            if self.calculate_cosine(tail1_curr, spine1_curr, headstage0_prev) is not None and self.calculate_cosine(tail1_curr, spine1_curr, headstage0_prev) > 0.65:
                                if self._distance(spine1_curr, headstage0_prev) < 110:
                                    self._swap_ids(frame)
                                    return True
                            if self.calculate_cosine(spine1_curr, neck1_curr, headstage0_curr) is not None and self.calculate_cosine(spine1_curr, neck1_curr, headstage0_curr) > 0.69:
                                if self._distance(spine1_curr, headstage0_curr) < 110:
                                    self._swap_ids(frame)
                                    return True
                            if self.Xt0_pred is not None and self._distance(spine1_curr, self.Xt0_pred) < 16:
                                self._swap_ids(frame)
                                return True
                    if self._distance(spine1_curr, self.Xt1_prev) is not None and self._distance(spine1_curr, self.Xt1_prev) > 50:
                        if self.jt1 is not None and self.jt1 > self.Jt:
                            if self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) is not None and self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) > 0.65:
                                if self._distance(spine1_curr, headstage0_curr) < 110:
                                    self._swap_ids(frame)
                                    return True
                if self.data.count_visible_nodes(frame-1,0) == 0 and self.data.count_visible_nodes(frame-1,1) == 0:
                    if self.jt1 is not None and self.jt1 > 3 * self.Jt:
                        if self.d10 is not None and abs(self.d10 - self.jt1) < 23:
                            if self.Xt1_pred is not None and self._distance(spine1_curr, self.Xt1_pred) > 100:
                                if self.calculate_cosine(neck1_curr, headstage0_curr, nose1_curr) is not None and self.calculate_cosine(neck1_curr, headstage0_curr, nose1_curr) > 0.65:
                                    self._swap_ids(frame)
                                    return True




            if self.data.count_visible_nodes(frame,1) == 0 and self.data.count_visible_nodes(frame-1,1) == 0:
                if self._distance(spine0_curr, spine0_prev) is not None and self._distance(spine0_curr, spine0_prev) > 50:
                    if self._distance(spine0_curr, self.Xt0_pred) is not None and self._distance(spine0_curr, self.Xt0_pred) > 50:
                        if self.jt0 is not None and self.jt0 > self.Jt:
                            if self.calculate_cosine(tail0_curr, spine0_curr, headstage0_curr) is not None and self.calculate_cosine(tail0_curr, spine0_curr, headstage0_curr) < 0.69:
                                if self.Xt1_pred is not None and self._distance(spine0_curr, self.Xt1_pred) < 30:
                                    self._swap_ids(frame)
                                    return True
                            if self.calculate_cosine(spine0_curr, neck0_curr, headstage0_prev) is not None and self.calculate_cosine(spine0_curr, neck0_curr, headstage0_prev) < 0.69:
                                if self.data.get_position(frame-2,1,'spine_base') is not None and self._distance(spine0_curr, self.data.get_position(frame-2,1,'spine_base')) < 12:
                                    self._swap_ids(frame)
                                    return True
                                if self.calculate_cosine(spine0_curr, neck0_curr, headstage0_prev) < -0.45:
                                    self._swap_ids(frame)
                                    return True
                            if self.calculate_cosine(tail0_prev, spine0_prev, headstage0_prev) is not None and self.calculate_cosine(tail0_prev, spine0_prev, headstage0_prev) > 0.69:
                                if self.calculate_cosine(tail0_prev, spine0_curr, headstage0_prev) < - 0.45:
                                    self._swap_ids(frame)
                                    return True
                            if np.isnan(headstage0_curr[0]) and np.isnan(headstage0_prev[0]):
                                if self.d01 is not None and abs(self.d01 - self.jt0) < 6:
                                    if self.Xt1_pred is not None and self._distance(spine0_curr, self.Xt1_pred) < 18:
                                        self._swap_ids(frame)
                                        return True
                if self.data.count_visible_nodes(frame-1,0) == 0 and self.data.count_visible_nodes(frame,0) > 1:
                    if self.data.get_position(frame-2,0,'spine_base') is not None and not np.isnan(spine0_curr[0]) and self._distance(spine0_curr, self.data.get_position(frame-2,0,'spine_base')) > 60:
                        if self.jt0 is not None and self.jt0 > 60:
                            if self.d01 is not None and self.d01 < 16:
                                self._swap_ids(frame)
                                return True

                            


                        
        if count00 >= 2:
            if (count1 + count0_nose) == 4:
                if self.d10 is not None and self.d10 < 6:
                    if (count0 + count1_nose) < 1:
                        if self.data.count_visible_nodes(frame,0) > count00:
                            if self.jt1 is not None and self.jt1 == 0:
                                if self._distance(spine1_curr, spine0_prev) < 1:
                                    self._swap_ids(frame)
                                    return True


        if self.jt0 is None and self.d01 is None:
            if self.data.count_visible_nodes(frame-1,0) == 0 and self.data.count_visible_nodes(frame-1,1) == 0:
                if self.data.count_visible_nodes(frame,0) == 1 and not np.isnan(headstage0_curr[0]) and self.data.count_visible_nodes(frame,1) > 0:
                    if self.d10 is not None and self.d10 < self.proximity_threshold:
                        if self.jt1 is not None and self.jt1 > self.Jt:
                            if self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) is not None and self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) > 0.69:
                                if self._distance(spine1_curr, headstage0_curr) < 110:
                                    self._swap_ids(frame)
                                    return True
            if (self.data.count_visible_nodes(frame,0) == 0 and self.data.count_visible_nodes(frame-1,0) > 0) or (self.data.count_visible_nodes(frame,0) == 1 and self.data.count_visible_nodes(frame-1,0) > 0 and not np.isnan(headstage0_curr[0])):
                if self.data.count_visible_nodes(frame,1) > 0 and self.data.count_visible_nodes(frame-1,1) == 0:
                    if (count1 + count0_nose) / self.data.count_visible_nodes(frame,1) == 1.0 and self.d10 is not None and self.d10 < 31 and min(self._distance(self.Xt1_curr,self.Xt0_prev),self.d10) < 30:
                        self._swap_ids(frame)
                        return True
                    if (count1 + count0_nose) / self.data.count_visible_nodes(frame,1) >= 0.75 and self.d10 is not None and self.d10 < 27 and min(self._distance(self.Xt1_curr,self.Xt0_prev),self.d10) < 26:
                        self._swap_ids(frame)
                        return True
                    if (count1 + count0_nose) / self.data.count_visible_nodes(frame,1) >= 0.6 and self.d10 is not None and self.d10 < 6 and min(self._distance(self.Xt1_curr,self.Xt0_prev),self.d10) < 6:
                        self._swap_ids(frame)
                        return True
                    if ((count1 + count0_nose + counths) / self.data.count_visible_nodes(frame-1,0) == 1.0 or (count1 + count0_nose) / self.data.count_visible_nodes(frame -1,0) >= 0.75)  and (np.isnan(headstage0_curr[0]) and not np.isnan(headstage0_prev[0])):
                        if self._distance(self.Xt1_curr,self.Xt0_prev) < self.proximity_threshold:
                            if self._distance(spine1_curr, spine0_prev) < 6 and self._distance(tail1_curr, tail0_prev) < 6:
                                if self.calculate_cosine(tail1_curr, spine1_curr, neck1_curr) is not None and self.calculate_cosine(tail1_curr, spine1_curr, neck1_curr) > 0.69:
                                    self._swap_ids(frame)
                                    return True
                            if self.d10 is not None and self.d10 < 23 and min(self._distance(self.Xt1_curr,self.Xt0_prev),self.d10) < 20:
                                if self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) is not None and self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) > 0.69:
                                    self._swap_ids(frame)
                                    return True
                        if (count1 + count0_nose) / self.data.count_visible_nodes(frame,1) == 1.0:
                            if min(self._distance(self.Xt1_curr,self.Xt0_prev), self.d10) < 40:
                                if self._distance(spine1_curr, spine0_prev) < 6 and self._distance(tail1_curr, tail0_prev) < 6:
                                    if self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) is not None and self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) > 0.69:
                                        self._swap_ids(frame)
                                        return True
                    if (count3 + count0_nose) / self.data.count_visible_nodes(frame,1) == 1.0 and self._distance(self.Xt1_curr,self.Xt0_prev) < 40:
                        if self._distance(spine1_curr, spine0_prev) < 1 and self._distance(tail1_curr, tail0_prev) < 1:
                            if self.calculate_cosine(tail1_curr, spine1_curr, neck0_prev) is not None and self.calculate_cosine(tail1_curr, spine1_curr, neck0_prev) > 0.69:
                                if self._distance(spine1_curr, neck0_prev) < 30:
                                    self._swap_ids(frame)
                                    return True
                    if (count1 + count0_nose) / self.data.count_visible_nodes(frame,1) >= 0.5 and self._distance(spine1_curr, spine0_prev) < 4 and self._distance(tail1_curr, tail0_prev) < 4:
                        if self.Xt1_pred is None:
                            if self.data.score[frame,1,1] < 0.32:
                                if count1 >= 2:
                                    self.data.blank_node(frame,1,1)
                                    self._swap_ids(frame)
                                    return True
                    if (count1 + count0_nose) / self.data.count_visible_nodes(frame,1) == 1.0 and (count1 + count0_nose + counths) / self.data.count_visible_nodes(frame-1,0) == 1.0:
                        if self.jt1 is not None and self.jt1 > 2 * self.Jt:
                            if self.d10 is not None and abs(self.jt1 - self.d10) < 30:
                                if self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) is not None and self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) > 0.69:
                                    self._swap_ids(frame)
                                    return True
                    if self._distance(tail1_curr, tail0_prev) < 9 and self._distance(spine1_curr, spine0_prev) < 18:
                        if np.isnan(neck0_prev[0]) and not np.isnan(neck1_curr[0]):
                            if self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) > 0.69:
                                if abs(self._distance(spine1_curr, headstage0_curr) - self._distance(spine0_prev, headstage0_curr)) < 12:
                                    if self.Xt1_pred is not None and self._distance(self.Xt1_curr, self.Xt1_pred) > self.Jt:
                                        self._swap_ids(frame)
                                        return True
                        if not np.isnan(neck0_prev[0]) and np.isnan(neck1_curr[0]):
                            if self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) > 0.69:
                                if abs(self._distance(spine1_curr, headstage0_curr) - self._distance(spine0_prev, headstage0_curr)) < 12:
                                    if self.Xt1_pred is not None and self._distance(self.Xt1_curr, self.Xt1_pred) > self.Jt:
                                        self._swap_ids(frame)
                                        return True
                        if np.isnan(nose0_prev[0]) and not np.isnan(nose1_curr[0]):
                            if self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) > 0.69:
                                if abs(self._distance(spine1_curr, headstage0_curr) - self._distance(spine0_prev, headstage0_curr)) < 12:
                                    if self.Xt1_pred is not None and self._distance(self.Xt1_curr, self.Xt1_pred) > self.Jt:
                                        if self.data.score[frame,1,0] < 0.22:
                                            self.data.blank_node(frame,1,0)
                                        self._swap_ids(frame)
                                        return True
                        if not np.isnan(nose0_prev[0]) and np.isnan(nose1_curr[0]):
                            if self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) > 0.69:
                                if abs(self._distance(spine1_curr, headstage0_curr) - self._distance(spine0_prev, headstage0_curr)) < 12:
                                    if self.Xt1_pred is not None and self._distance(self.Xt1_curr, self.Xt1_pred) > self.Jt:
                                        if self.data.score[frame,1,0] < 0.22:
                                            self.data.blank_node(frame,1,0)
                                        self._swap_ids(frame)
                                        return True
                        if np.isnan(headstage0_prev[0]) and not np.isnan(headstage0_curr[0]):
                            if abs(self._distance(spine1_curr, headstage0_curr) - self._distance(spine0_prev, headstage0_curr)) < 6:
                               if self.calculate_cosine(tail1_curr, spine0_prev, headstage0_curr) is not None and self.calculate_cosine(tail1_curr, spine0_prev, headstage0_curr) > 0.69:
                                    if self._distance(neck1_curr, neck0_prev) is not None and self._distance(neck1_curr, neck0_prev) < 30:
                                        if self.calculate_cosine(tail1_curr, neck0_prev, headstage0_curr) > 0.69:
                                            if self.calculate_cosine(tail1_curr, spine1_curr, neck1_curr) > 0.69:
                                                if abs(self.d10 - self.jt1) < 10:
                                                    if self.Xt1_pred is not None and self._distance(self.Xt1_curr,self.Xt1_pred) > self.Jt:
                                                        self._swap_ids(frame)
                                                        return True
                    if np.isnan(headstage0_curr[0]) and np.isnan(headstage0_prev[0]):
                        if not np.isnan(neck1_curr[0]) and np.isnan(neck0_prev[0]):
                            if self.calculate_cosine(tail1_curr, spine1_curr, neck1_curr) is not None and self.calculate_cosine(tail1_curr, spine1_curr, neck1_curr) > 0.69:
                                if self._distance(tail1_curr, tail0_prev) < 6 and self._distance(spine1_curr, spine0_prev) < 12:
                                    if abs(self.d10 - self.jt1) < 10:
                                        if self.Xt1_pred is not None and self._distance(self.Xt1_curr,self.Xt1_pred) > self.Jt:
                                            self._swap_ids(frame)
                                            return True
                    
                    if self.Xt1_pred is not None and self._distance(spine1_curr, self.Xt1_pred) > 50:
                        if self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) is not None and self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) > 0.69:
                            if np.isnan(tail0_prev[0]):
                                if self.d10 < 23:
                                    if self._distance(spine1_curr, headstage0_curr) < 110:
                                        if self.calculate_cosine(tail1_curr, spine1_curr, neck1_curr) is not None and self.calculate_cosine(tail1_curr, spine1_curr, neck1_curr) > 0.69:
                                            if np.isnan(neck0_prev[0]):
                                                self._swap_ids(frame)
                                                return True



                        


                    if abs(self._distance(spine1_curr, headstage0_curr) -self._distance(spine0_prev, headstage0_curr)) < 6:
                        if  self.d10 is not None and self.jt1 is not None and abs(self.d10 - self.jt1) < 15:
                            if self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) is not None and not self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) > 0.69:
                                if self._distance(tail1_curr, tail0_prev) < 1:
                                    if self.data.score[frame,1,1] < 0.35:
                                        self.data.blank_node(frame,1,1)
                                    if self._distance(self.Xt1_curr,self.Xt1_pred) > 27:
                                        self._swap_ids(frame)
                                        return True
                        if self._distance(spine1_curr, spine0_prev) < 6:
                            if not np.isnan(tail1_curr[0]) and np.isnan(tail0_prev[0]):
                                if self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) is not None and self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) > 0.69:
                                    if not np.isnan(nose1_curr[0]) and np.isnan(nose0_prev[0]):
                                        if self.calculate_cosine(tail1_curr, spine1_curr, nose1_curr) is not None and self.calculate_cosine(tail1_curr, spine1_curr, nose1_curr) < - 0.45:
                                            if self.data.score[frame,1,0] < 0.44:
                                                self.data.blank_node(frame,1,0)
                                            if np.isnan(neck1_curr[0]) and not np.isnan(neck0_prev[0]):
                                                self._swap_ids(frame)
                                                return True
                            if np.isnan(tail1_curr[0]) and not np.isnan(tail0_prev[0]):
                                if self.calculate_cosine(spine1_curr, neck1_curr, headstage0_curr) is not None and self.calculate_cosine(spine1_curr, neck1_curr, headstage0_curr) > 0.69:
                                    if np.isnan(nose0_prev[0]) and not np.isnan(nose1_curr[0]):
                                        if self.calculate_cosine(spine1_curr, neck1_curr, nose1_curr) is not None and self.calculate_cosine(spine1_curr, neck1_curr, nose1_curr) < - 0.45:
                                            if self.data.score[frame,1,0] < 0.44:
                                                self.data.blank_node(frame,1,0)
                                            if np.isnan(neck0_prev[0]) and not np.isnan(neck1_curr[0]):
                                                self._swap_ids(frame)
                                                return True
        
        if self.jt1 is None and self.d10 is None and self.data.count_visible_nodes(frame,1) == 0 and self.data.count_visible_nodes(frame-1,1) > 0:
            if self.data.count_visible_nodes(frame,0) > 0 and (self.data.count_visible_nodes(frame-1,0) == 0 or (self.data.count_visible_nodes(frame-1,0) == 1 and not np.isnan(headstage0_prev[0]))):
                if (count0 + count1_nose) / self.data.count_visible_nodes(frame,0) == 1.0 and max(self._distance(self.Xt0_curr,self.Xt1_prev),self.d01) < 31 and min(self._distance(self.Xt0_curr,self.Xt1_prev),self.d01) < 31:
                    self._swap_ids(frame)
                    return True
                if self._distance(spine0_curr, spine1_prev) < 10:
                    if (count0 + count1_nose) > 0:
                        if not np.isnan(neck0_curr[0]) and np.isnan(neck1_prev[0]):
                            if count1_nose == 1:
                                if self.calculate_cosine(spine0_curr, headstage0_curr, nose0_curr) is not None and self.calculate_cosine(spine0_curr, headstage0_curr, nose0_curr) < - 0.45:
                                    if self.Xt0_pred is not None and self._distance(self.Xt0_curr,self.Xt0_pred) > self.Jt:
                                        self._swap_ids(frame)
                                        return True
                                if self.calculate_cosine(spine0_curr, headstage0_prev, nose0_curr) is not None and self.calculate_cosine(spine0_curr, headstage0_prev, nose0_curr) < - 0.45:
                                    if self.Xt0_pred is not None and self._distance(self.Xt0_curr,self.Xt0_pred) > self.Jt:
                                        self._swap_ids(frame)
                                        return True
                        if self.data.count_visible_nodes(frame,0) == 1 and np.isnan(headstage0_curr[0]):
                            if self._distance(self.Xt0_curr,self.Xt1_prev) < 30:
                                self._swap_ids(frame)
                                return True

            if (count0 + count1_nose) / self.data.count_visible_nodes(frame,0) == 1.0 and self.d01 is not None and max(self._distance(self.Xt0_curr,self.Xt1_prev),self.d01) < 31 and min(self._distance(self.Xt0_curr,self.Xt1_prev),self.d01) < 31:
                if count0 >= 2:
                    self._swap_ids(frame)
                    return True
            if (count0 + count1_nose) / self.data.count_visible_nodes(frame - 1,1) == 1.0:
                if np.isnan(tail1_prev[0]) and not np.isnan(tail0_curr[0]):
                    if self.d01 is not None and self.d01 < 40:
                        if self._distance(spine0_curr, spine0_prev) > 50:
                            if self.calculate_cosine(spine0_curr, neck0_curr, headstage0_curr) < -0.45:
                                self._swap_ids(frame)
                                return True
        if self.track_1_reappeared_following_framed_window():
            if self.data.compute_separation(frame) < 16:
                if count00 > 0:
                    if (count1 + count0_nose) > 0:
                        if self.jt0 < 30:
                            if (count1 + count0_nose) / self.data.count_visible_nodes(frame,1) >= 0.6:
                                if  self._distance(nose1_curr, nose0_prev) < 6 and self._distance(tail1_curr, tail0_prev) < 6:
                                    if self._distance(headstage0_curr, spine0_curr) - self._distance(headstage0_curr, spine1_curr) < self.Jt and self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) is not None and self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) > 0.69:
                                        self._swap_ids(frame)
                                        return True
                        if self.jt0 >= self.Jt:
                            if (count1 + count0_nose) / self.data.count_visible_nodes(frame,1) == 1.0:
                                if self.d10 is not None and self.d10 < self.proximity_threshold:
                                    if self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) is not None and self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) > 0.69:
                                        self._swap_ids(frame)
                                        return True
            if self.jt0 is None and self.d01 is None:
                if self.data.count_visible_nodes(frame-1,0) > 0 and (count1 + count0_nose + counths) / self.data.count_visible_nodes(frame-1,0) == 1.0:
                    if  self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) is not None and self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) > 0.69:
                        if self._distance(tail1_curr, tail0_prev) < 8 and self._distance(spine1_curr, spine0_prev) < 8:
                            if abs(self._distance(spine1_curr, headstage0_curr) -self._distance(spine0_prev, headstage0_curr)) < 6:
                                if self.data.score[frame,1,1] < 0.36:
                                    self.data.blank_node(frame,1,1)
                                self._swap_ids(frame)
                                return True
                if self.Xt0_pred is not None and self._distance(self.Xt0_pred,spine1_curr) < 35:
                    if (count1 + count0_nose) > 0:
                        if self.calculate_cosine(tail1_curr, spine1_curr,headstage0_curr) is not None and self.calculate_cosine(tail1_curr, spine1_curr,headstage0_curr) > 0.85:
                            if self.calculate_cosine(spine1_curr, neck1_curr,headstage0_curr) is not None and self.calculate_cosine(spine1_curr, neck1_curr,headstage0_curr) > 0.85:
                                if self.d10 is not None and self.d10 < 40:
                                    if  self._distance(spine1_curr, spine0_prev) is not None and  self._distance(spine1_curr, spine0_prev) < 35:
                                        self._swap_ids(frame)
                                        return True





        
        if self._both_tracks_present(frame):
            if self._both_track_jumped(frame):
                if not self._both_intra_track_overlap_with_prev(frame):
                    if self.data.compute_separation(frame) < 10 and self.get_cross_visible_seperation(frame) < 19:
                        if (count1 + count0_nose) == 0 and (count0 + count1_nose) > 1 and self.data.count_visible_nodes(frame-1,1) > 0 and (count0 + count1_nose) / self.data.count_visible_nodes(frame-1,1) == 1.0:
                            self._extract_and_blank(frame,0)
                            return True
                    if self._both_inter_track_proximity_with_pred():
                        self._swap_ids(frame)
                        return True
                    elif self._both_inter_track_proximity_with_prev():
                        self._swap_ids(frame)
                        return True
                    if self.data.count_visible_nodes(frame-1,1) == 0 and self.jt0 + self.jt1 > 2 * self.Jt and self.data.count_visible_nodes(frame-1,0) > 0 and self.data.count_visible_nodes(frame,0) > 0 and self.data.count_visible_nodes(frame,1) > 0:
                        if (count1 + count0_nose) / self.data.count_visible_nodes(frame-1,0) == 1.0 and self.data.count_visible_nodes(frame,1) < self.minimum_separation_distance:
                            if self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) is not None and self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) > 0.69:
                                self._swap_ids(frame)
                                return True
                    if self.Xt1_pred is not None and ((self.d01 is not None and self.d01 < self.proximity_threshold and count3 >=1) or (self.d10 is not None and self.d10 < self.proximity_threshold and count2 >=1)) and self._distance(self.Xt1_curr,self.Xt1_pred)> 2*self.Jt and self._distance(self.Xt0_curr,self.Xt0_pred) is not None and self._distance(self.Xt0_curr,self.Xt0_pred)> 2*self.Jt:
                        self._swap_ids(frame)
                        return True
                    if count00 < 2 and self.Xt0_pred is not None and self.Xt1_pred is not None and count0 >=2 and count1 >= 2 and self._distance(self.Xt1_curr,self.Xt1_pred) > 2*self.Jt and self._distance(self.Xt0_curr,self.Xt0_pred) > 2*self.Jt:
                        self._swap_ids(frame)
                        return True
                    if self.data.count_visible_nodes(frame-1,1) == 0 and counths == 1 and self.data.count_visible_nodes(frame-1,0) > 1:
                        if (count1 + count0_nose) / (self.data.count_visible_nodes(frame-1,0) - 1) == 1.0 and self.d10 < self.minimum_separation_distance and abs(self.d01 - self.jt0) is not None and abs(self.d01 - self.jt0) < self.proximity_threshold:
                            self._swap_ids(frame)
                            return True
                        if count1 >= 1 and self._distance(self.Xt0_curr,self.Xt1_pred) < self.overlap_dist and self.d10 < (self.proximity_threshold + 3.5) and self.jt0 + self.jt1 > 5 * self.Jt:
                            self._swap_ids(frame)
                            return True
                    if abs(self.jt1 - self.jt0) < self.overlap_dist and (count2 + count3 >= 2):
                        self._swap_ids(frame)
                        return True
                    if self.d01 is not None and self.d10 is not None and self.d10 > self.Jt and self.d01 > self.Jt and (count3 + count2 >= 3):
                        self._swap_ids(frame)
                        return True
                    if self.jt0 + self.jt1 > (4 * self.Jt - 10) and ((count0 + count1 + count0_nose + count1_nose) >= 2) and count00 < 2 and count11 == 0:
                        self._swap_ids(frame)
                        return True
                    if self.jt0 + self.jt1 > (3 * self.Jt) and (count0 + count1 + count0_nose + count1_nose>= 3) and count3 + count2 >=1 and self.d01 is not None and self.d01 <= self.proximity_threshold:
                        self._swap_ids(frame)
                        return True
                    if self.d01 is not None and self.jt0 + self.jt1 > (3 * self.Jt) and self.data.count_visible_nodes(frame-1,0) > 0 and (count1 + count0_nose) / self.data.count_visible_nodes(frame-1,0)  > 0.6 and (count0 + count1_nose) >= 1 and self.d10 < 1 and abs(self.d01 - self.jt0) < self.proximity_threshold:
                        self._swap_ids(frame)
                        return True
                    if  count0 + count1 >=3 and ((count1 >= 2 and self.d01 is not None and abs(self.jt1 - self.d01) < self.overlap_dist) or (count0 >= 2 and self.d10 is not None and abs(self.jt0 - self.d10) < self.overlap_dist)):
                        self._swap_ids(frame)
                        return True
                    if count0 + count1 >= 2 and self.d10 is not None and abs(self.d10 - self.jt1) < self.medium_proximity_threshold and abs(self.jt0 - self.d01) < self.medium_proximity_threshold:
                        self._swap_ids(frame)
                        return True
                    if count0 + count1 > 0 and self.jt1 + self.jt0 > 4 * self.Jt and self.d01 is not None and self.d10 is not None and self.d01 < self.proximity_threshold and abs(self.d10 - self.jt1) < self.proximity_threshold:
                        self._swap_ids(frame)
                        return True
                    if self.d01 is not None and self.d10 is not None and count0 + count1 >= 3 and abs(self.d10 - self.jt1) < self.proximity_threshold and abs(self.jt0 - self.d01) < self.proximity_threshold:
                        self._swap_ids(frame)
                        return True
                    if self.jt0 >= 4 * self.Jt and self.jt1 >= 4 * self.Jt and count0 + count1 >= 3:
                        self._swap_ids(frame)
                        return True
                    if self.d01 is not None and self.d10 is not None and abs(self.d10 - self.jt0) < self.overlap_dist and abs(self.d01 - self.jt1) < self.overlap_dist:
                        self._swap_ids(frame)
                        return True
                    if count00 < 2 and self.jt0 + self.jt1 > 6 * self.Jt and (count0 + count1 + count1_nose + count0_nose) >= 3:
                        self._swap_ids(frame)
                        return True
                    if count00 < 2 and self.Xt0_prev is not None and self.Xt1_prev is not None and self._distance(self.Xt0_curr,self.Xt0_prev) > self.proximity_threshold and self._distance(self.Xt1_curr,self.Xt1_prev) > self.proximity_threshold and count0 + count1 >= 3:
                        self._swap_ids(frame)
                        return True
                    if self.jt0 + self.jt1 >= 4 * self.Jt and self.d01 is not None and self.d01 < self.overlap_dist and count0 / self.data.count_visible_nodes(frame,0) > 0.6:
                        self._swap_ids(frame)
                        return True
                    if self.d10 is not None and self.d01 is not None and abs(self.d10 - self.jt1) -1 < self.medium_proximity_threshold and abs(self.jt0 - self.d01) < self.medium_proximity_threshold and count1 / self.data.count_visible_nodes(frame,1) >= 0.5:
                        self._swap_ids(frame)
                        return True
                    if (self.jt0 + self.jt1 >= 5 * self.Jt and self.Xt0_prev is not None and self._distance(self.Xt0_curr,self.Xt0_prev) > 2 * self.proximity_threshold and 
                        self.d10 is not None and self.d10 < self.overlap_dist and count3 / self.data.count_visible_nodes(frame,1) >= 0.5):
                            self._swap_ids(frame)
                            return True
                    if count3 >=2 and self.Xt1_prev is None:
                        self._remove_overlapping_points(frame,1)
                        return True
                    if count2 >=2 and self.Xt0_prev is None:
                        self._remove_overlapping_points(frame,0)
                        return True
                    if self.jt0 > 2 * self.Jt and self.jt1 > 2 * self.Jt and self.d10 is not None and self.d10 < self.proximity_threshold and self.Xt1_prev is not None and self._distance(self.Xt0_curr,self.Xt1_prev) > self.proximity_threshold:
                        self._swap_ids(frame)
                        return True
                    if count00 < 2 and self.jt0 + self.jt1 > 6 * self.Jt and (count0 + count1) >= 1:
                        self._swap_ids(frame)
                        return True
                    if self.d10 is not None and self._distance(self.Xt0_curr,self.Xt1_prev) is not None and self._distance(self.Xt0_curr,self.Xt1_prev) < self.medium_proximity_threshold and abs(self.d10 - self.jt1) < (self.medium_proximity_threshold + 2) and (count0 + count1 + count0_nose + count1_nose) >= 3:
                        self._swap_ids(frame)
                        return True
                        
                    print (f"Frame {frame}: Both tracks jumped and are not overlapping with previous frame. But are far from pred/prev positions. No swap performed.")
                if not self._intra_track_overlap_with_prev(frame, 0):
                    print (f"Frame {frame}: Both tracks jumped but one track (track 0) does not overlap with previous frame.")
                    if self.Xt0_pred is not None:
                        # track 0 jumped but does not overlap with previous frame.
                        if self._track_0_overlap_with_track_1_pred(frame) or self._track_0_overlap_with_track_1_prev(frame):
                            # likely identity confusion; prefer swapping if strongly proximate
                            if self.d10 is not None and self.d10 < self.proximity_threshold:
                                self._swap_ids(frame)
                                return True
                            else:
                                self._remove_overlapping_points(frame, 0)
                                return True
                    else:
                        # track 0 reappeared?
                        # If reappeared on top of other track's prediction, swap
                        if self._track_0_overlap_with_track_1_pred(frame):
                            self._swap_ids(frame)
                            return True
                if not self._intra_track_overlap_with_prev(frame, 1):
                    if self.Xt1_pred is not None:
                        # track 1 jumped but does not overlap with previous frame.
                        if self._track_1_overlap_with_track_0_pred(frame) or self._track_1_overlap_with_track_0_prev(frame):
                            if self.d01 is not None and self.d01 < self.proximity_threshold:
                                self._swap_ids(frame)
                                return True
                            else:
                                self._remove_overlapping_points(frame, 1)
                                return True
                    else:
                        # track 1 reappeared?
                        if self._track_1_overlap_with_track_0_pred(frame):
                            self._swap_ids(frame)
                            return True
                # both tracks overlap with previous frame and both tracks jumped
            if self._one_track_jumped(frame):
                # both tracks present, one track jumped
                if self._track_0_jumped(frame):
                    # only track 0 jumped
                    # check if track 0 overlaps with track 1's predicted position or current position
                    if self.jt0 > 4 * self.Jt and self.d10 is not None and self.d10 >  4 * self.Jt:
                        if count11 == 4 and count0 > 0:
                            if self._distance(tail1_curr, tail0_prev) > 4 * self.Jt:
                                if self.calculate_cosine(tail0_prev, spine0_curr, headstage0_curr) is not None and self.calculate_cosine(tail0_prev, spine0_curr, headstage0_curr) < - 0.45:
                                    self.data.blank_frame(frame, 0)
                                    return True

                    if self.get_cross_visible_seperation(frame) is not None and self._distance(self.Xt1_curr,self.Xt0_prev) is not None and self._distance(self.Xt1_curr,self.Xt0_prev) <= 3.5 and count00 <= 2 and count11 == 0 and count3 >= 2 and self.get_cross_visible_seperation(frame) <= 10:
                        self._swap_and_blank(frame,1)
                        return True
                    if self.data.count_visible_nodes(frame-1,1) > 0 and self.data.count_visible_nodes(frame,1) > 0 and count11 / self.data.count_visible_nodes(frame-1,1) == 1.0 and count11 / self.data.count_visible_nodes(frame,1) == 1.0:
                        if self.get_cross_visible_seperation(frame) is not None and self.get_cross_visible_seperation(frame) <= 9 and self.data.compute_separation(frame) < self.minimum_separation_distance:
                            if (count0 + count1_nose) / self.data.count_visible_nodes(frame-1,1) >= 0.6:
                                self._extract_and_blank(frame,0)
                                return True
                    if self.get_cross_visible_seperation(frame) is not None and self.get_cross_visible_seperation(frame) <= 4 and self.data.compute_separation(frame) < 17:
                        if self.data.count_visible_nodes(frame-1,1) > 0 and count11 / self.data.count_visible_nodes(frame-1,1) >= 0.6 and (count0 + count1_nose) / self.data.count_visible_nodes(frame-1,1) >= 0.6:
                            if self.jt1 < 9 and self.jt0 > 2 * self.Jt:
                                self._extract_and_blank(frame,0)
                                return True
                    if self._track_0_overlap_with_track_1_pred(frame) or self._track_0_overlap_with_track_1_prev(frame):
                        if not self._intra_track_overlap_with_prev(frame, 0):
                            # likely stole identity or collided; clean by removing overlaps or swap when very close to the other prediction
                            if self.d10 is not None and self.d10 < self.proximity_threshold:
                                self._swap_ids(frame)
                                return True
                            elif self.data.count_visible_nodes(frame - 1,1) != 0 and self.Xt0_prev is not None and count0 / self.data.count_visible_nodes(frame - 1,1) == 1 and self._distance(self.Xt0_curr,self.Xt0_prev) > self.proximity_threshold:
                                self._swap_ids(frame)
                                return True
                            elif self.data.count_visible_nodes(frame,0) > 1 and (count0 + count1_nose) / self.data.count_visible_nodes(frame,0) > 0.6 and self.data.count_visible_nodes(frame,1) > 1 and (count1 + count0_nose) / self.data.count_visible_nodes(frame,1) > 0.6:
                                if self.data.compute_separation(frame) < self.overlap_dist:
                                    if self._distance(headstage0_curr, self.Xt0_curr) - self._distance(headstage0_curr, self.Xt1_curr) > 18:
                                        self._swap_and_blank(frame,1)
                            self._remove_overlapping_points(frame, 0)
                            return True
                        else:
                            # Consistent with own prev; keep identity but prune overlaps
                            self._remove_overlapping_points(frame, 0)
                            return True
                    if self.d01 is not None and self.d01 > 2 * self.proximity_threshold and self.Xt1_prev is None and self.d10 < self.overlap_dist:
                            self._swap_ids(frame)
                            return True
                    if self.jt1 is not None and self.jt1 + self.jt0 > 80 and self.d01 is not None and self.d10 is not None and abs(self.d10 - self.jt1) < self.overlap_dist and abs(self.d01 - self.jt0) < self.overlap_dist and abs(self.d01 - self.jt0) < self.proximity_threshold and (count0 + count1) > 0:
                        self._swap_ids(frame)
                        return True
                    if self.jt1 == 0 and self.d10 is not None and self.d10 < self.proximity_threshold and self.d01 is None and (count1 >= 3 or count3 >= 2):
                        self._swap_ids(frame)
                        return True
                    if self.jt1 == 0 and self.jt0 > 2 * self.Jt and (count3 + count0_nose) / self.data.count_visible_nodes(frame,1) >= 0.5 and abs(self.jt0 - self.data.compute_separation(frame)) < self.proximity_threshold:
                        self._swap_ids(frame)
                        return True
                    if self.jt0 > 3 * self.Jt and self.jt1 is not None and self.d01 is not None and self.d10 is not None and abs(self.d10 - self.jt1) < self.medium_proximity_threshold and abs(self.d01 - self.jt0) < self.overlap_dist and count1 > 1:
                        self._swap_ids(frame)
                        return True
                    if self.d10 is not None and self.d10 <= self.overlap_dist and count3 >= 0:
                        self._swap_ids(frame)
                        return True
                    if self.jt1 is not None and self.jt1 + 2 > self. Jt and self.Xt1_prev is not None and self.Xt0_prev is not None and self._distance(self.Xt1_curr,self.Xt1_prev) > self.proximity_threshold and self._distance(self.Xt0_curr,self.Xt0_prev) > self.proximity_threshold and count0 +count1 >=3:
                        self._swap_ids(frame)
                        return True
                    if self.jt1 is not None and self.d10 is not None and self.d10 is not None and self.d01 is not None and abs(self.d10 - self.jt0) < self.overlap_dist and abs(self.d01 - self.jt1) < self.overlap_dist and count0 + count1 >= 3:
                        self._swap_ids(frame)
                        return True
                    if self.Xt0_prev is not None and self.jt0 > 2 * self.Jt and self._distance(self.Xt0_curr,self.Xt0_prev) > self.proximity_threshold and count1 / self.data.count_visible_nodes(frame,1) >= 0.75: # and there are additional points in  1 prev relative to 0 current ?
                        self._swap_ids(frame)
                        return True
                    if count00 + count11 < 2 and self.data.count_visible_nodes(frame-1,0) != 0 and count1 > 0 and ((count1 + count0_nose) / self.data.count_visible_nodes(frame-1,0)) == 1:
                        self._swap_ids(frame)
                        return True 
                    if self.data.count_visible_nodes(frame-1,0) != 0 and self.data.count_visible_nodes(frame-1,1) != 0 and ((count1 + count0_nose) / self.data.count_visible_nodes(frame-1,0)) >= 0.5 and ((count0 + count1_nose) / self.data.count_visible_nodes(frame-1,1)) >= 0.5:
                        self._swap_ids(frame)
                        return True
                    if self.d01 is not None and ((count1 + count0_nose) / self.data.count_visible_nodes(frame,1)) >= 0.75 and abs(self.jt0 - self.d01) < self.proximity_threshold:
                        self._swap_ids(frame)
                        return True
                    if count3 / self.data.count_visible_nodes(frame,1) >= 0.5 and self.d01 is not None and abs(self.jt0 - self.d01) < self.overlap_dist and self.d10 is not None and self.d10 < self.proximity_threshold:
                        self._swap_ids(frame)
                        return True
                    if (self.jt1 == 0 and self.d10 is not None and self.d10 < self.proximity_threshold and count1 >= 1 and self.calculate_cosine(spine1_curr, neck1_curr, headstage0_curr) is not None and self.calculate_cosine(spine1_curr, neck1_curr, headstage0_curr) >= 0.88 and
                          self._distance(headstage0_curr, self.Xt0_curr) - self._distance(headstage0_curr, self.Xt1_curr) > self.medium_proximity_threshold and self.calculate_cosine(spine0_curr, neck0_curr, headstage0_curr) is not None and
                          self.calculate_cosine(spine0_curr, neck0_curr, headstage0_curr) < - 0.3):
                        self._swap_ids(frame)
                        return True
                if self._track_1_jumped(frame):
                    # only track 1 jumped
                    # check if track 1 overlaps with track 0's predicted position or current position
                    if self._track_1_overlap_with_track_0_pred(frame) or self._track_1_overlap_with_track_0_prev(frame):
                        if not self._intra_track_overlap_with_prev(frame, 1):
                            if self.data.count_visible_nodes(frame - 1,0) > 0 and count00 / self.data.count_visible_nodes(frame - 1,0) >= 0.75 and self.data.compute_separation(frame) < self.minimum_separation_distance and self._distance(self.Xt1_curr,self.Xt0_prev) <= self.overlap_dist and count11 == 0 and count3 >= 2:
                                self._extract_and_blank(frame,1)
                                return True
                            if self.data.count_visible_nodes(frame,1) > 0 and self.data.count_visible_nodes(frame-1,0) > 0 and (count1 + count0_nose) / self.data.count_visible_nodes(frame,1) and (count1 + count0_nose + counths) / self.data.count_visible_nodes(frame-1,0) == 1.0:
                                if self.d10 is not None and max(self._distance(self.Xt1_curr,self.Xt0_prev),self.d10) < 16 and min(self._distance(self.Xt1_curr,self.Xt0_prev),self.d10) < 12:
                                    self._swap_ids(frame)
                                    return True
                            if self.d01 is not None and self.d01 < self.proximity_threshold:
                                self._swap_ids(frame)
                                return True
                            if count00 < 3 and count1 >= 2 and self.jt1 >= 3 * self.Jt:  # and d01 > 3 Jt
                                self._swap_ids(frame)
                                return True
                            if self.d01 is not None and abs(self.jt1 - self.d01) < self.proximity_threshold and ((count3 >= 1) or (self.d10 is not None and self.jt0 is not None and abs(self.jt0 - self.d10) < self.proximity_threshold)):
                                self._swap_ids(frame)
                                return True
                            if self.jt0 is None and self.d10 is not None and self.d10 < self.overlap_dist and self.jt1 > 2 * self.Jt:
                                if self.data.count_visible_nodes(frame,0) == 1 and headstage0_prev is not None:
                                    if self.Xt1_prev is not None and self._distance(headstage0_curr,headstage0_prev) < self.overlap_dist and self._distance(self.Xt1_curr,self.Xt1_prev) > self.proximity_threshold:
                                        count1 += 1
                                        if count1 / (self.data.count_visible_nodes(frame,1) + 1) >= 0.6:
                                            self._swap_ids(frame)
                                            return True
                                    if self.Xt1_prev is None and count1 > 0 and ((count1 + count0_nose) / self.data.count_visible_nodes(frame-1,0)) >= 0.6:
                                        self._swap_ids(frame)
                                        return True
                            if count1 / self.data.count_visible_nodes(frame,1) == 1.0 and self.data.count_visible_nodes(frame-1,1) > 0 and count0 / self.data.count_visible_nodes(frame-1,1) == 1.0:
                                self._swap_ids(frame)
                                return True

                            self._remove_overlapping_points(frame, 1)
                            return True
                        else:
                            if ((count3) / self.data.count_visible_nodes(frame-1,0)) == 1:
                                self._swap_ids(frame)
                                return True
                            self._remove_overlapping_points(frame, 1)
                            return True
                    if self.data.count_visible_nodes(frame,0) > 1 and count00 / self.data.count_visible_nodes(frame,0) == 1.0:
                        if self._distance(tail1_curr, tail0_prev) is not None and self._distance(tail1_curr, tail0_prev) < 6:
                            if self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) is not None and self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) > 0.69:
                                if self.jt1 > 2 * self.Jt:
                                    if abs(self._distance(self.Xt1_curr,self.Xt0_curr) - self.d10) < 6 and count11 == 0:
                                        self.data.blank_frame(frame,1)
                                        # TODO:add to the correction log
                                        return True
                    if self.jt0 is None and self.d01 is None:
                        if self.data.count_visible_nodes(frame,0) == 1 and not np.isnan(headstage0_curr[0]):
                            if self.Xt0_prev is not None and self.d10 is not None and max(self._distance(self.Xt1_curr,self.Xt0_prev),self.d10) < 16 and min(self._distance(self.Xt1_curr,self.Xt0_prev),self.d10) < 13 and (count1 + count0_nose) / (self.data.count_visible_nodes(frame,1)) >= 0.5:
                                if self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) is not None and self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) > 0.69:
                                    if abs(self._distance(spine1_curr, headstage0_curr) -self._distance(spine0_prev, headstage0_curr)) < 16:
                                        self._swap_ids(frame)
                                        return True
                            if self.Xt0_prev is not None and self.d10 is not None and ((count1 + count0_nose) / self.data.count_visible_nodes(frame,1)) >= 0.75 and max(self._distance(self.Xt1_curr,self.Xt0_prev),self.d10) < 25 and min(self._distance(self.Xt1_curr,self.Xt0_prev),self.d10) < 24:
                                if self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) is not None and self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) > 0.69:
                                    if abs(self._distance(spine1_curr, headstage0_curr) -self._distance(spine0_prev, headstage0_curr)) < 16:
                                        self._swap_ids(frame)
                                        return True
                            if self.Xt0_prev is not None and self.d10 is not None and ((count1 + count0_nose) / self.data.count_visible_nodes(frame,1)) == 1.0 and max(self._distance(self.Xt1_curr,self.Xt0_prev),self.d10) < 31 and min(self._distance(self.Xt1_curr,self.Xt0_prev),self.d10) < 30:
                                if self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) is not None and self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) > 0.69:
                                    if abs(self._distance(spine1_curr, headstage0_curr) -self._distance(spine0_prev, headstage0_curr)) < 16:
                                        self._swap_ids(frame)
                                        return True
                                if self.d10 is not None and max(self._distance(self.Xt1_curr,self.Xt0_prev),self.d10) < 15 and min(self._distance(self.Xt1_curr,self.Xt0_prev),self.d10) < 15:
                                    if abs(self._distance(spine1_curr, headstage0_curr) -self._distance(spine0_prev, headstage0_curr)) < 12:
                                        self._swap_ids(frame)
                                        return True
                            if self.Xt0_prev is not None and (count1 + count0_nose) / self.data.count_visible_nodes(frame,1) == 1.0 and (count1 + count0_nose + counths) / self.data.count_visible_nodes(frame-1,0) == 1.0 and self.d10 is not None and min(self._distance(self.Xt1_curr,self.Xt0_prev),self.d10) < 37:
                                if self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) is not None and self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) > 0.69:
                                    if abs(self._distance(spine1_curr, headstage0_curr) -self._distance(spine0_prev, headstage0_curr)) < 16:
                                        self._swap_ids(frame)
                                        return True
                            if self.data.count_visible_nodes(frame-1,0) > 0 and (count1 + count0_nose + counths) / self.data.count_visible_nodes(frame-1,0) >= 0.75:
                                if abs(self._distance(spine1_curr, headstage0_curr) - self._distance(spine0_prev, headstage0_curr)) < 6:
                                    if self.calculate_cosine(spine1_curr, neck1_curr, headstage0_curr) is not None and self.calculate_cosine(spine1_curr, neck1_curr, headstage0_curr) > 0.69:
                                        if self.calculate_cosine(spine1_curr, neck1_curr, nose1_curr) is not None and self.calculate_cosine(spine1_curr, neck1_curr, nose1_curr) < - 0.45:
                                            if self._distance(nose1_curr, headstage0_curr) > 100:
                                                if self.data.score[frame,1,0] < 0.63:
                                                    self.data.blank_node(frame,1,0)
                                                    self._swap_ids(frame)
                                                    return True
                            if count1 >= 3 and (counths == 1 or (not np.isnan(headstage0_curr[0]) and np.isnan(headstage0_prev[0]))):
                                if self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) is not None and self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) > 0.69:
                                    if self.calculate_cosine(spine1_curr, neck1_curr, headstage0_curr) is not None and self.calculate_cosine(spine1_curr, neck1_curr, headstage0_curr) > 0.69:
                                        if abs(self._distance(spine1_curr, headstage0_curr) -self._distance(spine0_prev, headstage0_curr)) < 6:
                                                if np.isnan(nose1_curr[0]) and not np.isnan(nose0_prev[0]):
                                                    self._swap_ids(frame)
                                                    return True
                                                if not np.isnan(nose1_curr[0]) and np.isnan(nose0_prev[0]):
                                                    self._swap_ids(frame)
                                                    return True
                            if count1 >= 2 and (counths == 1 or (not np.isnan(headstage0_curr[0]) and np.isnan(headstage0_prev[0]))):
                                if self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) is not None and self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) > 0.69:
                                    if self.calculate_cosine(spine1_curr, neck1_curr, headstage0_curr) is not None and self.calculate_cosine(spine1_curr, neck1_curr, headstage0_curr) > 0.69:
                                        if abs(self._distance(spine1_curr, headstage0_curr) -self._distance(spine0_prev, headstage0_curr)) < 12:
                                                if self._distance(neck1_curr, neck0_prev) is not None and self._distance(neck1_curr, neck0_prev) < 10:
                                                    if np.isnan(nose1_curr[0]) and not np.isnan(nose0_prev[0]):
                                                        self._swap_ids(frame)
                                                        return True
                                                    if not np.isnan(nose1_curr[0]) and np.isnan(nose0_prev[0]):
                                                        self._swap_ids(frame)
                                                        return True
                                                if np.isnan(neck0_prev[0]):
                                                    self._swap_ids(frame)
                                                    return True
                                    if self.calculate_cosine(spine1_curr, neck1_curr, headstage0_curr) is not None and self.calculate_cosine(spine1_curr, neck1_curr, headstage0_curr) < - 0.45:
                                        if self.data.score[frame,1,1] < 0.65:
                                            if abs(self._distance(spine1_curr, headstage0_curr) - self._distance(spine0_prev, headstage0_curr)) < 6:
                                                if self._distance(self.Xt1_curr,self.Xt1_prev) > 2 * self.Jt:
                                                    if self._distance(self.Xt1_curr,self.Xt0_prev) < 23:
                                                        self._swap_ids(frame)
                                                        return True
                                    if np.isnan(neck1_curr[0]) and np.isnan(neck0_prev[0]):
                                        if abs(self._distance(spine1_curr, headstage0_curr) - self._distance(spine0_prev, headstage0_curr)) < 6:
                                            if self.Xt1_pred is not None and self._distance(self.Xt1_curr,self.Xt1_pred) > 2 * self.Jt:
                                                if self._distance(self.Xt1_curr,self.Xt0_prev) < 32:
                                                    self._swap_ids(frame)
                                                    return True
                    if self.jt0 is not None and self.d01 is not None and self.d10 is not None:
                        if abs(self._distance(spine1_curr, headstage0_curr) - self._distance(spine0_prev, headstage0_curr)) < 12:
                            if abs(self.d10 - self.jt1) < 15 and abs(self.d01 - self.jt0) < 15:
                                if self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) is not None and self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) > 0.69:
                                    if self._distance(spine1_curr, headstage0_curr) < 100:
                                        if self._distance(spine0_curr, spine1_prev) < 6:
                                            if self._distance(self.Xt1_curr,self.Xt1_prev) > self.Jt:
                                                self._swap_ids(frame)
                                                return True
                        if self.data.count_visible_nodes(frame-1,0) == 0:
                            if self.data.count_visible_nodes(frame - 1,1) > 0 and (count0 + count1_nose) / self.data.count_visible_nodes(frame - 1,1) == 1.0:
                                if abs(self.d10 - self.jt1) < 12:
                                    if self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) is not None and self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) > 0.69:
                                        if self._distance(spine1_curr, headstage0_curr) < 110:
                                            self._swap_ids(frame)
                                            return True
                    if self.d10 is not None and self.d10 < 25 and self._distance(self.Xt1_curr,self.Xt1_pred) is not None and self._distance(self.Xt1_curr,self.Xt1_pred) > self.Jt and abs(self.d10 - self.jt1) < 10:
                        if self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) is not None and self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) > 0.69 and self._distance(spine1_curr, headstage0_curr) < 90:
                            self._swap_ids(frame)
                            return True
                        if self.calculate_cosine(spine1_curr, neck1_curr, headstage0_curr) is not None and self.calculate_cosine(spine1_curr, neck1_curr, headstage0_curr) > 0.69:
                            if abs(self._distance(spine1_curr, headstage0_curr) -self._distance(spine0_prev, headstage0_curr)) < 12:
                                if self.d01 is not None and self.d01 < 18:
                                    self._swap_ids(frame)
                                    return True
                    if self.d10 is not None and self.d01 is not None and count00 > 1 and self.d10 < 7 and self.d01 > 100 and self.get_cross_visible_seperation(frame) < 23 and self._distance(self.Xt0_curr,self.Xt1_curr) < 10 and self._distance(self.Xt1_curr,self.Xt0_prev) < 8 and self._distance(self.Xt0_curr,self.Xt1_prev) > 100:
                        if self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) - self.calculate_cosine(spine0_curr, neck0_curr, headstage0_curr) > 0.1:
                            self._swap_and_blank(frame,1)
                            return True

                    if self.jt0 is None and self.d01 is None and self.d10 is not None and self.d10 < 13 and self.data.count_visible_nodes(frame,1) > 0 and (count1 + count0_nose) / self.data.count_visible_nodes(frame,1) == 1.0:
                        self._swap_ids(frame)
                        return True
                    if self.d01 is not None and self.d10 is not None and self.d10 > 2 * self.proximity_threshold and self.Xt0_prev is None and self.d01 < self.overlap_dist:
                            self._swap_ids(frame)
                            return True
                    if self.d10 is not None and self.d01 is not None and self.jt0 is not None and abs(self.d10 - self.jt1) < self.minimum_separation_distance and abs(self.d01 - self.jt0) < self.minimum_separation_distance and (count1 + count0) > 0:
                        if self._distance(headstage0_curr, self.Xt0_curr) - self._distance(headstage0_curr, self.Xt1_curr) > 18 and self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) is not None and self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) > 0.90:
                            self._swap_ids(frame)
                            return True
                        if abs(self.d10 - self.jt1) < 9 and abs(self.d01 - self.jt0) < 9 and count0 + count1 > 2:
                            self._swap_ids(frame)
                            return True
                    if self.d10 is not None and self.d01 is not None and abs(self.d10 - self.d01) < self.overlap_dist and (count2 + count3 >= 2):
                        self._swap_ids(frame)
                        return True
                    if (count0 >= 2 and ((count2 + count3) >= 1)) or count0 >=3:
                        self._swap_ids(frame)
                        return True
                    if (count1 + count0_nose) / self.data.count_visible_nodes(frame,1) == 1 and self.jt1 > 3 * self.Jt and self.jt0 is None and count1 > 0:
                        self._swap_ids(frame)
                        return True
                    if count3 >= 2 and count2 >= 1 and self.d01 > 2*self.Jt and self.d10 is not None and self.d10 >= self.Jt:
                        self._swap_ids(frame)
                        return True
                    if count1 > 0 and ((count1 + count0_nose) / self.data.count_visible_nodes(frame-1,0)) == 1 and count00 < 2 and count11 == 0:
                        self._swap_ids(frame)
                        return True
                    if self.data.count_visible_nodes(frame-1,0) != 0 and count1 / self.data.count_visible_nodes(frame-1,0) >= 0.75 and self.data.count_visible_nodes(frame - 1,1) != 0 and count0 / self.data.count_visible_nodes(frame - 1,1) >= 0.75:
                        self._swap_ids(frame)
                        return True
                    if self.jt0 ==0 and self.jt1 > 5 * self.Jt and self.d10 is None and self.d01 is not None and self.d01 < 2*self.Jt and count2 >= 1:
                        self._swap_ids(frame)
                        return True
                    if self.d10 is not None and self.d10 <= self.proximity_threshold and count0 + count1 >= 3:
                        self._swap_ids(frame)
                        return True
                    if self.jt0 is None and self.jt1 > 4 * self.Jt and ((count1 / self.data.count_visible_nodes(frame,1) >= 0.75) or (count3) / (self.data.count_visible_nodes(frame,1)) >= 0.5):
                        self._swap_ids(frame)
                        return True
                    if self.jt0 is None and self.jt1 > 3 * self.Jt and ((count1 / self.data.count_visible_nodes(frame,1) == 1) or (count3) / (self.data.count_visible_nodes(frame,1)) >= 0.75):
                        self._swap_ids(frame)
                        return True
                    #elif self.d01 is not None and self.d10 is not None and self.jt0 is not None and abs(self.d10 - self.jt0) < self.proximity_threshold and abs(self.d01 - self.jt1) < self.overlap_dist:
                        # self._swap_ids(frame)
                        # return True
                    if self.jt1 > 2 * self.Jt and self.jt0 is None and count3 / self.data.count_visible_nodes(frame,1) >= 0.6:
                        self._swap_ids(frame)
                        return True
                    if self.jt0 is None and self.d10 is not None and self.d10 < self.proximity_threshold and self.jt1 + 2 > 2 * self.Jt and count1 >= 2:
                        self._swap_ids(frame)
                        return True
                    if self.jt0 is None and self.d10 is not None and self.d10 < self.medium_proximity_threshold and self.jt1 > 3 * self.Jt and count1 / self.data.count_visible_nodes(frame,1) >= 0.5:
                        self._swap_ids(frame)
                        return True
                    if self.jt0 is not None and self.d01 is not None and self.jt0 >= self.Jt -1 and abs(self.jt1- self.d01) < self.proximity_threshold and abs(self.jt0- self.d10) < self.proximity_threshold and count1+count0 >= 1:
                        self._swap_ids(frame)
                        return True
                    if self.d10 is not None and abs(self.d10 - self.jt1) <= self.proximity_threshold and count3 / self.data.count_visible_nodes(frame,1) >= 0.6 and count00 < 2:
                        self._swap_ids(frame)
                        return True
                    if self.d10 is not None and abs(self.d10 - self.jt1) <= self.overlap_dist and count1 / self.data.count_visible_nodes(frame,1) >= 0.5 and count00 < 2:
                        self._swap_ids(frame)
                        return True
                    if self.d10 is not None and self.d01 is not None and self.jt0 is not None and abs(self.d10 - self.jt1) < 9 and abs(self.d01 - self.jt0) < 9 and count0 + count1 > 2:
                        self._swap_ids(frame)
                        return True
                    if self.jt0 is None and self.jt1 > 2 * self.Jt and count1 / self.data.count_visible_nodes(frame,1) >= 0.6:
                        self._swap_ids(frame)
                        return True
                    if self.jt0 is None and self.jt1 > 3 * self.Jt and ((count1 + count0_nose) / (self.data.count_visible_nodes(frame,1) + 1)) >= 0.6 and self._distance(self.data.get_centroid(frame-1,0),self.data.get_centroid(frame,1)) < self.proximity_threshold:
                        self._swap_ids(frame)
                        return True
                    if self.jt0 is None and self.jt1 > 3 * self.Jt and self.d10 is not None and self.d10 < self.proximity_threshold and abs(self.calculate_cosine(spine1_curr, neck1_curr, headstage0_curr)) > 0.69:
                        self._swap_ids(frame)
                        return True
                    if self.Xt1_prev is not None and self.data.count_visible_nodes(frame,1) is not None and self.jt0 is None and self._distance(self.Xt1_curr,self.Xt1_prev) <= self.proximity_threshold and count1 / self.data.count_visible_nodes(frame,1) >= 0.6:
                        self._swap_ids(frame)
                        return True
                    if self.d10 is not None and self.jt0 is None and self.Xt1_prev is not None and self._distance(self.Xt1_curr,self.Xt1_prev) > self.proximity_threshold and abs(self.d10 - self.jt1) <= self.medium_proximity_threshold and count1 >= 2:
                        self._swap_ids(frame)
                        return True
                    if self.jt0 is None and self.Xt1_prev is None and self.d10 is not None and self.jt1 is not None and abs(self.d10 - self.jt1) <= self.medium_proximity_threshold and count1 / self.data.count_visible_nodes(frame,1) >= 0.5:
                        self._swap_ids(frame)
                        return True
                    if self.jt0 is None and self.d10 is not None and abs(self.d10 - self.jt1) <= self.proximity_threshold and count1 / self.data.count_visible_nodes(frame,1) >= 0.6:
                        self._swap_ids(frame)
                        return True
                    if self.data.count_visible_nodes(frame-1,0) > 0 and self.jt0 is None and self.d10 is not None and abs(self.d10 - self.jt1) <self.proximate_jump_thresh and count1 / self.data.count_visible_nodes(frame-1,0) >= 0.5:
                        self._swap_ids(frame)
                        return True
                    if self.data.count_visible_nodes(frame-1,0) > 0 and self.jt0 is None and self.jt1 > 3 * self.Jt and self.Xt1_prev is not None and self._distance(self.Xt1_curr,self.Xt1_prev) > 3 * self.proximity_threshold and ((count1 + count0_nose) / self.data.count_visible_nodes(frame-1,0)) >= 0.6:
                        self._swap_ids(frame)
                        return True
                    if self.jt0 is None and self.jt1 > 3 * self.Jt and self.d10 is not None and abs(self.d10 - self.jt1) <= 1.2 * self.proximity_threshold and count1 >= 1 and count11 == 0 and headstage0_curr is not None and self._distance(headstage0_curr, self.Xt1_curr) <= 1.1 * self.proximity_threshold:
                        self._swap_ids(frame)
                        return True
                    if self.data.compute_separation(frame) is not None and self.Xt0_prev is not None and self.jt0 is None and self.jt1 > 3 * self.Jt and headstage0_curr is not None and headstage0_prev is not None and abs(self._distance(headstage0_curr, self.Xt0_prev) - self.data.compute_separation(frame)) < self.overlap_dist and count1 / (self.data.count_visible_nodes(frame,1)) >= 0.5:
                        self._swap_ids(frame)
                        return True
                    if self.jt0 is None and self.jt1 > 2 * self.Jt and self.d10 is not None and self.d10 < (self.Jt + 1) and self.data.count_visible_nodes(frame-1,0) > 0 and (count1 + count0_nose) / self.data.count_visible_nodes(frame-1,0) >= 0.66:
                        self._swap_ids(frame)
                        return True
                    # check spine-neck-headstage allignment and distance and or number of overlapping point divided by those visible in earlier frame
                    if (self.calculate_cosine(spine1_curr, neck1_curr, headstage0_curr) is not None and abs(self.calculate_cosine(spine1_curr, neck1_curr, headstage0_curr)) > 0.69 and
                        self.d10 is not None and self._distance(neck1_curr, headstage0_curr) - self._distance(neck1_curr, spine1_curr) < 1.1 * (self.proximity_threshold) and abs(self.d10 - self.jt1) <= self.proximity_threshold and
                        count1 / self.data.count_visible_nodes(frame,0) >= 0.5 and
                        self._distance(self.data.get_centroid(frame-1,0),self.data.get_centroid(frame,1)) < self.proximity_threshold):
                        self._swap_ids(frame)
                        return True
                    if self.data.count_visible_nodes(frame,0) == 1 and headstage0_curr is not None:
                        if self._distance(headstage0_curr, headstage0_prev) < self.overlap_dist:
                            count1 += 1
                            if (self.d10 is not None and abs(self.d10 - self.jt1) <= self.proximity_threshold and count1 / (self.data.count_visible_nodes(frame,1) + 1) >= 0.5 and
                                self.jt1 >= 2 * self.Jt and self._distance(self.data.get_centroid(frame-1,0),self.data.get_centroid(frame,0)) > self.proximity_threshold):
                                self._swap_ids(frame)
                                return True
                        if self._distance(headstage0_curr, headstage0_prev) < self.medium_proximity_threshold:
                            count1 += 1
                            if (self.d10 is not None and abs(self.d10 - self.jt1) <= self.medium_proximity_threshold and count1 / (self.data.count_visible_nodes(frame,1) + 1) >= 0.5 and
                                self.jt1 >= 2 * self.Jt and self._distance(self.data.get_centroid(frame-1,0),self.data.get_centroid(frame,0)) > self.proximity_threshold):
                                self._swap_ids(frame)
                                return True
                    if self.jt1 > 3 * self.Jt and count3 >= 1 and self.data.count_visible_nodes(frame-1,0) != 0 and count00 / self.data.count_visible_nodes(frame-1,0) == 1.0 and (self.data.compute_separation(frame) <= self.minimum_separation_distance or self.get_cross_visible_seperation(frame) <= self.medium_proximity_threshold):
                        self._extract_and_blank(frame, 1)
                        return True
                    if self.jt1 > 3 * self.Jt and count3 >= 1 and self.data.count_visible_nodes(frame,0) != 0 and count00 / self.data.count_visible_nodes(frame,0) == 1.0 and self.data.compute_separation(frame) <= self.minimum_separation_distance:
                        self._extract_and_blank(frame, 1)
                        return True
                    if self.jt0 is not None and self.jt0 < 2.2 and self.jt1 > 3 * self.Jt and self.data.count_visible_nodes(frame-1,0) != 0 and count00 / self.data.count_visible_nodes(frame-1,0) == 1.0 and count1 >= 2:
                        if self.get_cross_visible_seperation(frame) < self.minimum_separation_distance and self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) is not None and self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) >= 0.88:
                            self._extract_and_blank(frame,1)
                            return True
                        if self.get_cross_visible_seperation(frame) < 26 and self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) is not None and self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) >= 0.88:
                            if self.data.compute_separation(frame) < 31 and self._distance(self.Xt1_curr,self.Xt0_prev) is not None and self._distance(self.Xt1_curr,self.Xt0_prev) < 31 and self.jt1 > 4 * self.Jt:
                                self._extract_and_blank(frame,1)
                                return True
                        if self.get_cross_visible_seperation(frame) < 28 and self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) is not None and self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) >= 0.98:
                            if self.data.compute_separation(frame) < 33 and self._distance(self.Xt1_curr,self.Xt0_prev) is not None and self._distance(self.Xt1_curr,self.Xt0_prev) < 10:
                                self._extract_and_blank(frame,1)
                                return True
                    if self.d01 is not None and self.jt0 is not None and self.d10 is not None and self.jt1 is not None and abs(self.d01 - self.jt0) - 1 <= self.proximity_threshold and abs(self.d10 - self.jt1) <= self.proximity_threshold and (self.jt0 + self.jt1) > 2 * self.Jt and (count0 + count1) >= 2:
                        self._swap_ids(frame)
                        return True
                    
            else:
                # both tracks present, none jumped

                if self.jt0 is None and self.jt1 is not None and (self.jt1 == 0 or (self.data.count_visible_nodes(frame-1,0) == 1 and headstage0_prev is not None)) and self.data.count_visible_nodes(frame,0) == 1 and self._distance(headstage0_curr, self.data.get_centroid(frame,1)) > 4 * self.Jt:
                    if self.data.count_visible_nodes(frame-1,1) == 0 and (count0_nose + count1) / self.data.count_visible_nodes(frame,1) == 1.0 and self.d10 is not None and self.d10 < self.overlap_dist:
                        # jt1 == 0 and number of points in track 1 and track 0 are the same in track 1 previous. can also add one count3 and or counths
                        # also abs(self._distance(headstage0_curr, self.Xt0_prev) - self._distance(headstage0_curr, self.Xt1_curr)) < 2.0
                        self._swap_ids(frame)
                        return True
                    if self.data.count_visible_nodes(frame-1,1) == 0 and (count0_nose + count3) / self.data.count_visible_nodes(frame,1) == 1.0 and self.d10 is not None and self.d10 < self.proximity_threshold:
                        self._swap_ids(frame)
                        return True
                    if (self.data.count_visible_nodes(frame-1,1) == 0 and ((count0_nose + count1) / self.data.count_visible_nodes(frame,1) >= 0.66 or (count0_nose + count1) / self.data.count_visible_nodes(frame-1,0) >= 0.66) and count3 >= 1 and self._distance(self.Xt1_curr,self.Xt0_prev) is not None and self._distance(self.Xt1_curr,self.Xt0_prev) < (1 + self.proximity_threshold) and
                        self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) is not None and self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) > 0.69) and self._distance(spine1_curr, headstage0_curr) < 4 * self.Jt and self._distance(spine1_curr, spine0_prev) < self.overlap_dist:
                        self._swap_ids(frame)
                        return True
                    if self._distance(spine1_curr, headstage0_curr) is not None and self._distance(spine1_curr, headstage0_curr) < 4 * self.Jt and self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) is not None and self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) > 0.69:
                        if self._distance(tail1_curr, tail0_prev) is not None and self._distance(tail1_curr, tail0_prev) < self.overlap_dist and self._distance(spine1_curr, spine0_prev) is not None and self._distance(spine1_curr, spine0_prev) < self.overlap_dist and counths == 1:
                            # could also blank neck and or nose, cosine can be greater than 0.98
                            self._swap_ids(frame)
                            return True
                    
                    return False
                if (self.track_1_reappeared_following_framed_window() and self.track_0_didnt_jumped() and (count00 >= 3 or (count00 / self.data.count_visible_nodes(frame,0) == 1))): 
                    if ((count1 + count0_nose) >= 3 or count3 >= 2) and ((self.data.compute_separation(frame) < self.minimum_separation_distance and (self.track_1_overlapped_with_track_0_prev())) or self.get_cross_visible_seperation(frame) <= self.medium_proximity_threshold):
                        self._extract_and_blank(frame,1)
                        return True
                    if count3 / self.data.count_visible_nodes(frame - 1,0) >= 0.5 and self.get_cross_visible_seperation(frame) <= self.overlap_dist:
                        self._extract_and_blank(frame,1)
                        return True
                    if (count1 + count0_nose) / self.data.count_visible_nodes(frame,1) >= 0.6 and (self.data.compute_separation(frame) < self.minimum_separation_distance or self._distance(self.Xt1_curr,self.Xt0_curr) < 13) and count11 == 0:
                        self._extract_and_blank(frame,1)
                        return True
                    if self.get_cross_visible_seperation(frame) is not None and self.get_cross_visible_seperation(frame) <= self.overlap_dist and count3 >= 1 and self._distance(self.Xt1_curr,self.Xt0_prev) < self.minimum_separation_distance:
                        self._extract_and_blank(frame,1)
                        return True
                    if self.get_cross_visible_seperation(frame) is not None and self.get_cross_visible_seperation(frame) <= self.minimum_separation_distance and self.data.compute_separation(frame) < self.proximity_threshold and self._distance(self.Xt1_curr,self.Xt0_curr) < 1.1 * self.medium_proximity_threshold and count1 >= 1:
                        if self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) is not None and self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) > 0.88:
                            self._extract_and_blank(frame,1)
                            return True
                if self.track_1_reappeared_following_framed_window() and self.track_0_didnt_jumped() and self.data.count_visible_nodes(frame-1,0) > 1 and count00 / self.data.count_visible_nodes(frame-1,0) == 1.0:
                    if (count1 + count0_nose) / self.data.count_visible_nodes(frame,1) >= 0.66 and self.d10 is not None and self.d10 < self.medium_proximity_threshold:
                        if self._distance(tail1_curr, tail0_prev) is not None and self._distance(tail1_curr, tail0_prev) < 4 and self._distance(neck1_curr, neck0_prev) is not None and self._distance(neck1_curr, neck0_prev) < 8:
                            if self._distance(spine1_curr, spine0_prev) is not None and self._distance(spine1_curr, spine0_prev) < 31:
                                if self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) is not None and self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) > 0.88:
                                    self._extract_and_blank(frame,1)
                                    return True
                if self.jt0 is not None and abs(self.jt0 - self.data.compute_separation(frame)) < 3.5 and (self.get_cross_visible_seperation(frame) <= self.medium_proximity_threshold or (self.Xt0_prev is not None and self._distance(self.Xt1_curr,self.Xt0_prev) <= 4.5)):
                    if count00 >= 2 and count11 == 0 and (count0_nose + count1) / self.data.count_visible_nodes(frame-1,0) >= 0.5:
                        self._extract_and_blank(frame,1)
                        return True
                    if self.data.count_visible_nodes(frame-1,0) > 0 and count00 < 2 and count11 == 0 and (count0_nose + count1) / self.data.count_visible_nodes(frame-1,0) == 1.0 and self._distance(self.Xt1_curr,self.Xt0_prev) <= 1.0:
                        self._swap_and_blank(frame,1)
                        return True
                    if count00 < 2 and self.data.count_visible_nodes(frame-1,0) > 0 and (count0_nose + count1) / self.data.count_visible_nodes(frame-1,0) >= 0.75  and self.d10 is not None and self.d10 < self.overlap_dist and self.get_cross_visible_seperation(frame) <= self.minimum_separation_distance:
                        self._swap_and_blank(frame,1)
                        return True
                if (self.track_1_reappeared_following_framed_window() and self.track_0_didnt_jumped()) and (count1 + count0_nose) / self.data.count_visible_nodes(frame,1) == 1.0:
                    if self.data.compute_separation(frame) < self.minimum_separation_distance and count00 >= 2 and self._distance(self.Xt1_curr,self.Xt0_prev) <= 3.0:
                        self._swap_and_blank(frame,1)
                        return True
                    if self._distance(self.Xt1_curr,self.Xt0_curr) < 13 and count00 >= 2 and self.get_cross_visible_seperation(frame) <= self.minimum_separation_distance:
                        self._swap_and_blank(frame,1)
                        return True
                    if self.data.count_visible_nodes(frame,1) > 0 and (count3 + count0_nose) / self.data.count_visible_nodes(frame,1) == 1.0 and count00 >= 2 and self.get_cross_visible_seperation(frame) is not None and self.get_cross_visible_seperation(frame) < 26 and self.data.compute_separation(frame) < 26:
                        self._swap_and_blank(frame,1)
                        return True
                    
                if (self.track_1_reappeared_following_framed_window() and self.track_0_didnt_jumped()) and (count1 + count0_nose) / self.data.count_visible_nodes(frame,1) >= 0.75:
                    if self.get_cross_visible_seperation(frame) is not None and self.get_cross_visible_seperation(frame) <= 16 and count00 >= 1 and self.data.compute_separation(frame) < self.minimum_separation_distance and abs(self.jt0 - self.data.compute_separation(frame)) < 3.5:
                        self._swap_and_blank(frame,1)
                        return True
                if self.get_cross_visible_seperation(frame) is not None and self.get_cross_visible_seperation(frame) < self.minimum_separation_distance and self._distance(self.Xt1_curr,self.Xt0_curr) < 15 and self.data.compute_separation(frame) < 23.1:
                    if self.jt1 > 23 and count1 / self.data.count_visible_nodes(frame,1) >= 0.6 and self.d10 is not None and self.d10 < 7 and count0 > 0 and self.jt0 < 9 and abs(self._distance(self.Xt1_curr,self.Xt0_curr) - self._distance(self.Xt0_curr,self.Xt1_prev)) < 1 and self.calculate_cosine(tail1_curr, spine1_curr, headstage0_curr) > 0.95:
                        self._swap_and_blank(frame,1)
                        return True
                
                if self.data.count_visible_nodes(frame,0) > 1 and count2 / (self.data.count_visible_nodes(frame,0) - 1) >= 0.6 and abs(self.d01 - self.jt1) < self.overlap_dist:
                    self._swap_ids(frame)
                    return True
                if self.jt0 is not None and self.jt1 is not None and abs(self.jt1 - self.jt0) < self.overlap_dist and (count2 + count3 >= 2):
                    self._swap_ids(frame)
                    return True
                if self.d10 is not None and self.d01 is not None and abs(self.d10 - self.d01) < self.overlap_dist and (count2 + count3 >= 2):
                    self._swap_ids(frame)
                    return True
                if self.jt0 is None and self.data.count_visible_nodes(frame-1,0)!= 0 and count1 > 0 and ((count1 + count0_nose) / self.data.count_visible_nodes(frame-1,0)) == 1: # can be replaced with count3 as well
                    self._swap_ids(frame)
                    return True
                if self.jt0 is None and self.jt1 == 0 and ((self.d10 is not None and self.d10 < self.proximity_threshold and (count1 + count0_nose) >=2) or count3 >= 3 or (count1 + count0_nose) / self.data.count_visible_nodes(frame,1) >= 0.6):
                    self._swap_ids(frame)
                    return True
                if self.jt0 is None and self.jt1 is not None and self.jt1 == 0 and count1 >=2 and self.data.count_visible_nodes(frame,1) - self.data.count_visible_nodes(frame-1,0) >=1 and self.data.count_visible_nodes(frame-1,1) == 0:
                    self._swap_ids(frame)
                    return True
                if self.jt0 is None and self.jt1 is not None and self.d10 is not None and abs(self.d10 - self.jt1) > 3 * self. Jt and self.data.count_visible_nodes(frame,0) <= 1 and count3 / self.data.count_visible_nodes(frame,1) >= 0.6:
                    self._swap_ids(frame)
                    return True
                if self.jt1 is not None and self.jt1 == 0 and count3 / self.data.count_visible_nodes(frame,1) >= 0.75 and count00 < 2:
                    self._swap_ids(frame)
                    return True
                elif self.jt0 is not None and self.jt1 is not None and self.d10 is not None and self.d01 is not None and abs(self.d10 - self.jt1) < self.minimum_separation_distance and abs(self.d01 - self.jt0) < self.minimum_separation_distance and self.data.count_visible_nodes(frame,0) > 0 and self.data.count_visible_nodes(frame-1,0) > 0 and count0 / self.data.count_visible_nodes(frame,0) == 1.0 and count1 / self.data.count_visible_nodes(frame-1,0) == 1.0:
                    self._swap_ids(frame)
                    return True
                elif self.jt0 is None and self.jt1 is not None and self.data.count_visible_nodes(frame,0) == 1 and headstage0_curr is not None and self.d10 is not None and abs(self.d10 - self.jt1) < 1.1 * self.proximity_threshold and ((count0_nose + count1) / self.data.count_visible_nodes(frame,1)) >= 0.66:
                    self._swap_ids(frame)
                    return True
                elif self.jt0 is None and self.jt1 is not None and self.jt1 == 0  and self.d10 is not None and abs(self.d10 - self.jt1) < self.minimum_separation_distance and ((count0_nose + count1) / self.data.count_visible_nodes(frame,1)) >= 0.5 and self._distance(self.Xt1_curr,self.Xt0_prev) is not None and self._distance(self.Xt1_curr,self.Xt0_prev) < self.minimum_separation_distance:
                    self._swap_ids(frame)
                    return True

                
        else:
            # one track is present
            if self.Xt0_curr:
                # only track 0 present
                if self._track_0_overlap_with_track_1_pred(frame) or self._track_0_overlap_with_track_1_prev(frame):
                    if self._track_0_jumped(frame):
                        # track 0 jumped and overlaps with track 1's predicted or previous position
                        self._swap_ids(frame)
                        return True
                    elif self.jt0 == 0:
                        # track 0 reappeared and overlaps with track 1's predicted or previous position
                        # reappearance onto other track's path – prefer swap
                        self._swap_ids(frame)
                        return True
                    # track 0 is present but overlaps with track 1's predicted or previous position
                elif self._intra_track_overlap_with_prev(frame, 0) or self._intra_track_overlap_with_pred(frame, 0):
                    # track 0 overlaps with its own previous frame
                    # likely fine – no action
                    if (self.jt1 is None and self.jt0 > self.Jt and self.d01 is not None and self.d01 < self.proximity_threshold and self.Xt1_prev is not None and
                         self._distance(self.Xt0_curr,self.Xt1_prev) < self.proximity_threshold and count0 / self.data.count_visible_nodes(frame - 1,1) >= 0.5 and
                         self.Xt0_prev is not None and self._distance(self.Xt0_curr,self.Xt0_prev) > self.proximity_threshold):
                        self._swap_ids(frame)
                        return True
                    pass
                elif self._track_0_jumped(frame):
                    # track 0 jumped and does not overlap with track 1's and 0 predicted or previous position
                    dist_0_prev = self._distance(self.Xt0_curr, self.Xt0_prev)
                    if (count0 + count1_nose) >=2 and ((dist_0_prev is not None and dist_0_prev > self.Jt and self.Xt0_prev is not None) or self.Xt0_prev is None):
                        self._swap_ids(frame)
                        return True
                    elif count2 >= 2 and self.jt0 > 5*self.Jt:
                        self._swap_ids(frame)
                        return True
                    elif self.jt0 > 4 * self.Jt and self.d01 is not None and self.d01 < self.proximity_threshold:
                        self._swap_ids(frame)
                        return True
                    if self.data.count_visible_nodes(frame-1,1) != 0 and ((count0 + count1_nose) / self.data.count_visible_nodes(frame-1,1)) == 1 and abs(self.jt0 - self.d01) < 1.1 * self.proximity_threshold:
                        self._swap_ids(frame)
                        return True
                    if self.jt1 is None and self.d01 is not None and abs(self.d01 - self.jt0) < self.proximity_threshold:
                        if counths == 1 and self.data.count_visible_nodes(frame,0) == 2:
                            if self._distance(spine0_curr, spine1_prev) is not None and self._distance(spine0_curr, spine0_prev) is not None and self._distance(spine0_curr, spine1_prev) < self.minimum_separation_distance and self._distance(spine0_curr, spine0_prev) > self.proximity_threshold:
                                self._swap_ids(frame)
                                return True
                elif self.jt0 == 0 or self.Xt0_prev is None:
                    # track 0 reappeared and does not overlap with track 1's and 0 predicted or previous position
                    if count0 >=2:
                        self._swap_ids(frame)
                        return True
                    elif count2 >= 1 and self.jt1 is None and self.d01 >= self.Jt:
                        self._swap_ids(frame)
                        return True
                    elif count0 >= 1 and self.d01 is not None and self.d01 <= (self.proximity_threshold):
                        self._swap_ids(frame)
                        return True
                    elif self.data.count_visible_nodes(frame,0) <= 2 and self.data.count_visible_nodes(frame,1) == 0 and count0 >= 1:
                        self._swap_ids(frame)
                        return True
                    pass
            elif self.Xt1_curr:
                # only track 1 present
                if self._track_1_overlap_with_track_0_pred(frame) or self._track_1_overlap_with_track_0_prev(frame):
                    if self._track_1_jumped(frame):
                        # track 1 jumped and overlaps with track 0's predicted or previous position
                        self._swap_ids(frame)
                        return True
                    elif self.jt1 == 0:
                        # track 1 reappeared and overlaps with track 0's predicted or previous position
                        self._swap_ids(frame)
                        return True
                    # track 1 is present but overlaps with track 0's predicted or previous position
                elif self._intra_track_overlap_with_prev(frame, 1) or self._intra_track_overlap_with_pred(frame, 1):
                    # track 1 overlaps with its own previous frame
                    pass
                elif self._track_1_jumped(frame):
                    # track 1 jumped and does not overlap with track 0's and 1 predicted or previous position
                    dist_1_prev = self._distance(self.Xt1_curr, self.Xt1_prev)
                    if (count3 >= 1 or count1 >=2) and ((dist_1_prev is not None and dist_1_prev > self.Jt) or self.Xt1_prev is None):
                        self._swap_ids(frame)
                        return True
                    if self.data.count_visible_nodes(frame,1) > 0 and count1 / self.data.count_visible_nodes(frame,1) >= 0.5 and count11 == 0 and abs(self.d10 - self.jt1) < 18 and self._distance(self.Xt1_curr,self.Xt1_prev) > 2 * self.Jt and self._distance(spine1_curr, spine0_prev) < 6:
                        self._swap_ids(frame)
                        return True
                    if self.data.count_visible_nodes(frame,1) > 0 and count1 >0 and count11 == 0 and self.d10 < 24 and self._distance(self.Xt1_curr,self.Xt0_prev) < 20 and self._distance(spine1_curr, spine0_prev) < 6 and self._distance(self.Xt1_curr,self.Xt1_pred) > 2 * self.Jt:
                        self._swap_ids(frame)
                        return True
                elif self.jt0 is None and self.jt1 == 0:
                    if self.d10 is not None and self.d10 < self.proximity_threshold and self._distance(spine1_curr, spine0_prev) < 4 and self._distance(self.Xt1_curr,self.Xt0_prev) < 16 and count00 == 0:
                        self._swap_ids(frame)
                        return True

                    pass
                elif self.jt1 == 0 or self.Xt1_prev is None:
                    # track 1 reappeared and does not overlap with track 0's and 1 predicted or previous position
                    if count1 >=2:
                        self._swap_ids(frame)
                        return True
                    if self.data.count_visible_nodes(frame-1,0) > 0 and (count3 + count0_nose) / self.data.count_visible_nodes(frame-1,0) >= 0.5 and self.d10 is not None and self.d10 < self.medium_proximity_threshold:
                        self._swap_ids(frame)
                        return True
                    if (count3 + count0_nose) / self.data.count_visible_nodes(frame,1) >= 0.5:
                        self._swap_ids(frame)
                        return True
                elif self.jt0 is None and self.d10 is not None and abs(self.d10 - self.jt1) < self.proximity_threshold and count3 / self.data.count_visible_nodes(frame,1) == 1.0:
                    self._swap_ids(frame)
                    return True
                
                    
        return False

    
    def track_1_reappeared_following_framed_window (self) -> bool:
        """ Check if track 1 reappeared following the framed window of absence."""
        if self.jt1 is None or self.jt1 != 0:
            return False
        return self.jt1 == 0
    



    
    def _swap_and_blank (self, frame: int, track: int):
        """Swap IDs and blank the specified `track` at `frame`.

        Ensures caches are cleared and current-frame state recomputed
        after the swap/blank so downstream logic uses consistent data.
        """
        # Perform swap
        self.data.swap_tracks(frame)

        # Blank the specified track (not the opposite)
        self.data.blank_frame(frame, track)

        # Clear caches impacted by swap/blank
        self._position_cache.clear()
        self._stable_nodes_cache.clear()
        self._track_validity_cache.clear()

        # Recompute state for the current frame using updated data
        self._recompute_current_frame_state(frame)

        # Log correction
        self.corrections.append({
            'frame': frame,
            'type': 'swap_and_blank',
            'action': 'Swapped tracks and blanked track {}'.format(track)
        })



    def track_0_didnt_jumped (self) -> bool:
        """ Check if track 0 didn't jumped in the current frame."""
        if self.jt0 is None:
            return False
        return self.jt0 < self.minimum_separation_distance + 1


    def track_1_overlapped_with_track_0_prev (self) -> bool:
        return (self.Xt0_prev is not None and self._distance(self.Xt1_curr,self.Xt0_prev) <= self.overlap_dist) or self._distance(self.Xt1_curr,self.Xt0_pred) <= self.overlap_dist
    
    
    def _handle_orientation_jumps(self, frame: int) -> bool:
        """Orientation-based jump handler.
        Uses abrupt orientation changes combined with proximity to other track's trajectory to decide swaps or pruning.
        Returns True if any correction occurred."""
        # Orientation flip booleans (None means insufficient data)
        t0_flip = self.orientation_flip_check(frame, 0)
        t1_flip = self.orientation_flip_check(frame, 1)

        # Normalize None to False for decision purposes
        t0_flip = bool(t0_flip) if t0_flip is not None else False
        t1_flip = bool(t1_flip) if t1_flip is not None else False

        if not (t0_flip or t1_flip):
            return False

        # If both flipped, only act when there's strong mutual proximity evidence
        if t0_flip and t1_flip:
            if self._both_inter_track_proximity_with_pred() or self._both_inter_track_proximity_with_prev():
                self._swap_ids(frame)
                return True
            return False

        # Helper lambdas to keep conditions readable
        def _prox_to_other_pred_curr_is(track_flipped: int) -> bool:
            return (
                (track_flipped == 0 and self._track_0_overlap_with_track_1_pred(frame)) or
                (track_flipped == 1 and self._track_1_overlap_with_track_0_pred(frame))
            )

        def _prox_to_other_prev_curr_is(track_flipped: int) -> bool:
            return (
                (track_flipped == 0 and self._track_0_overlap_with_track_1_prev(frame)) or
                (track_flipped == 1 and self._track_1_overlap_with_track_0_prev(frame))
            )

        def _consistent_with_own_prev_or_pred(track_flipped: int) -> bool:
            return (
                self._intra_track_overlap_with_prev(frame, track_flipped) or
                self._intra_track_overlap_with_pred(frame, track_flipped)
            )

        # Single-track flip case
        flipped_track = 0 if t0_flip else 1
        other_track = 1 - flipped_track

        # If flipped track is still consistent with its own trajectory, don't overcorrect
        if _consistent_with_own_prev_or_pred(flipped_track):
            return False

        # If flipped track is proximate to the other track's predicted or previous location, treat as likely identity confusion
        prox_other = _prox_to_other_pred_curr_is(flipped_track) or _prox_to_other_prev_curr_is(flipped_track)
        if prox_other:
            # Prefer swap when strongly proximate; otherwise prune overlapping nodes on the flipped track
            if flipped_track == 0:
                if self.d10 is not None and self.d10 < self.proximity_threshold:
                    self._swap_ids(frame)
                    return True
                self._remove_overlapping_points(frame, 0)
                return True
            else:
                if self.d01 is not None and self.d01 < self.proximity_threshold:
                    self._swap_ids(frame)
                    return True
                self._remove_overlapping_points(frame, 1)
                return True

        # Flip detected but no strong proximity evidence; leave as-is
        return False

    def _handle_single_node_jumps(self, frame: int) -> bool:
        """Check for individual nodes that jumped more than 2*Jt and eliminate them.
        
        This handles cases where a single node from either track makes a large jump
        (> 2*Jt) compared to its previous position, which is likely an outlier.
        Returns True if any nodes were removed.
        """
        changed = False
        
        # Check each track
        for track in [0, 1]:
            # Check each node for large jumps
            for node_idx in range(self.data.num_nodes):
                curr_pos = self._get_node_position(frame, track, node_idx)
                prev_pos = self._get_node_position(frame - 1, track, node_idx)
                
                if curr_pos is not None and prev_pos is not None:
                    jump_distance = self._distance(curr_pos, prev_pos)
                    # FIXME: check if score is been improved and consider increasing the threshold
                    # If this node jumped more than 2*Jt, populate with the previous position, if not possible blank it
                    if jump_distance is not None and jump_distance > 2 * self.Jt:
                        node_name = self.data.nodes[node_idx]
                        # Check if the node was already reassigned in the previous frame to prevent consecutive assignments
                        prev_jump_large = False
                        if frame > 1:
                            prev_prev_pos = self._get_node_position(frame - 2, track, node_idx)
                            if prev_prev_pos is not None and prev_pos == prev_prev_pos:
                                prev_jump_large = True
                        if self._is_node_closer_to_median(frame, track, node_idx):
                            continue
                        if not prev_jump_large and prev_pos is not None:
                            self._set_node_position(frame, track, node_idx, prev_pos)
                            self.corrections.append({
                            'frame': frame,
                            'track': track,
                            'type': 'single_node_jump',
                            'action': f'Node "{node_name}" reassigned previous position from track {track} due to large jump ({jump_distance:.1f}px > {2 * self.Jt:.1f}px)'
                        })
                        else:
                            self.data.blank_node(frame, track, node_name)
                            self.corrections.append({
                            'frame': frame,
                            'track': track,
                            'type': 'single_node_jump',
                            'action': f'Node "{node_name}" removed from track {track} due to large jump ({jump_distance:.1f}px > {2 * self.Jt:.1f}px)'
                        })
                        

                        changed = True
        
        return changed
    def _set_node_position(self, frame: int, track: int, node_idx: int, pos: Tuple[float, float]) -> None:
        """Set the position of a specific node for a given frame and track."""
        self.data.x[frame, track, node_idx] = pos[0]
        self.data.y[frame, track, node_idx] = pos[1]
    

    def cross_visible_nodes(self, frame: int) -> List[str]:
        """Get the list of node NAMES that are visible in both tracks for a given frame.

        Internal helpers like `_get_visible_nodes` return node *indices*; this method
        normalizes those to node names so callers consistently receive names.
        """
        idx0 = self._get_visible_nodes(frame, 0)
        idx1 = self._get_visible_nodes(frame, 1)
        # Map indices to names and compute intersection
        names0 = {self.data.nodes[i] for i in idx0}
        names1 = {self.data.nodes[i] for i in idx1}
        cross_nodes = list(names0 & names1)
        return cross_nodes
    

    def get_median_position(self, frame: int, track: int, node_list: List[str]) -> Optional[Tuple[float, float]]:
        """Get the median position of a list of nodes.

        Accepts a list of node NAMES (preferred). Will also tolerate a list
        containing integer indices. Uses nan-aware median so missing points
        don't produce NaN results when possible.
        """
        if not node_list:
            return None

        node_indices: List[int] = []
        for node in node_list:
            if isinstance(node, int):
                # assume already an index
                if 0 <= node < self.data.num_nodes:
                    node_indices.append(node)
                continue
            # try to resolve name to index
            try:
                idx = self.data.get_node_index(node)
            except Exception:
                # skip unknown names
                continue
            node_indices.append(idx)

        if not node_indices:
            return None

        x_vals = self.data.x[frame, track, node_indices]
        y_vals = self.data.y[frame, track, node_indices]

        # Use nanmedian to ignore missing values; if all are nan this returns nan
        median_x = np.nanmedian(x_vals)
        median_y = np.nanmedian(y_vals)
        if np.isnan(median_x) or np.isnan(median_y):
            return None
        return (float(median_x), float(median_y))
    

    def get_cross_median_position(self, frame: int, track: int) -> Optional[Tuple[float, float]]:
        """Get the median position of cross-visible nodes for a given frame and track."""
        cross_nodes = self.cross_visible_nodes(frame)
        return self.get_median_position(frame, track, cross_nodes)
    
    def get_cross_visible_seperation(self, frame: int) -> Optional[float]:
        """Get the separation distance between the median positions of cross-visible nodes for both tracks."""
        median_pos_track0 = self.get_cross_median_position(frame, 0)
        median_pos_track1 = self.get_cross_median_position(frame, 1)
        if median_pos_track0 is None or median_pos_track1 is None:
            return None
        return self._distance(median_pos_track0, median_pos_track1)


    def _get_track_median_position(self, frame: int, track: int) -> Optional[Tuple[float, float]]:
        """Get the median position of all visible nodes for a given frame and track."""
        visible_nodes = self._get_visible_nodes(frame, track)
        if not visible_nodes:
            return None
        x_vals = self.data.x[frame, track, visible_nodes]
        y_vals = self.data.y[frame, track, visible_nodes]
        median_x = np.median(x_vals)
        median_y = np.median(y_vals)
        return (float(median_x), float(median_y))
    
    def _is_node_closer_to_median(self, frame: int, track: int, node_idx: int) -> bool:
        """Check if the current position is closer to the track median relative to the previous position."""
        curr_pos = self._get_node_position(frame, track, node_idx)
        prev_pos = self._get_node_position(frame - 1, track, node_idx)
        if curr_pos is None or prev_pos is None:
            return False
        curr_median = self._get_track_median_position(frame, track)
        prev_median = self._get_track_median_position(frame - 1, track)
        if curr_median is None or prev_median is None:
            return False
        dist_curr = self._distance(curr_pos, curr_median)
        dist_prev = self._distance(prev_pos, prev_median)
        if dist_curr is None or dist_prev is None:
            return False
        return dist_curr < dist_prev
    
    def _one_intra_track_overlap_with_prev(self, frame: int) -> bool:
        return self._intra_track_overlap_with_prev(frame, 0) or self._intra_track_overlap_with_prev(frame, 1)

    def _both_intra_track_overlap_with_prev(self, frame: int) -> bool:
        return self._intra_track_overlap_with_prev(frame, 0) and self._intra_track_overlap_with_prev(frame, 1)
    def _both_inter_track_proximity_with_pred(self) -> bool:
        return (self.d01 is not None and self.d01 < self.proximity_threshold and 
                self.d10 is not None and self.d10 < self.proximity_threshold)
                
    def _both_inter_track_proximity_with_prev(self) -> bool:
        return (self._distance(self.Xt0_curr, self.Xt1_prev) < self.proximity_threshold and 
                self._distance(self.Xt1_curr, self.Xt0_prev) < self.proximity_threshold
                if (self.Xt0_curr is not None and self.Xt1_prev is not None and 
                    self.Xt1_curr is not None and self.Xt0_prev is not None) else False)
                
    def _track_0_proximate_with_track_1_prev(self) -> bool:
        return (self._distance(self.Xt0_curr, self.Xt1_prev) < self.proximity_threshold 
                if (self.Xt0_curr is not None and self.Xt1_prev is not None) else False)
                
    def _track_1_proximate_with_track_0_prev(self) -> bool:
        return (self._distance(self.Xt1_curr, self.Xt0_prev) < self.proximity_threshold 
                if (self.Xt1_curr is not None and self.Xt0_prev is not None) else False)
                
    def _both_inter_track_overlap_with_prev(self, frame: int) -> bool:
        return (self._distance(self.Xt0_curr, self.Xt1_prev) < self.overlap_dist and 
                self._distance(self.Xt1_curr, self.Xt0_prev) < self.overlap_dist
                if (self.Xt0_curr is not None and self.Xt1_prev is not None and 
                    self.Xt1_curr is not None and self.Xt0_prev is not None) else False)
                    
    def _track_0_overlap_with_track_1_prev(self, frame: int) -> bool:
        return (self._distance(self.Xt0_curr, self.Xt1_prev) < self.overlap_dist 
                if (self.Xt0_curr is not None and self.Xt1_prev is not None) else False)
                
    def _track_1_overlap_with_track_0_prev(self, frame: int) -> bool:
        return (self._distance(self.Xt1_curr, self.Xt0_prev) < self.overlap_dist 
                if (self.Xt1_curr is not None and self.Xt0_prev is not None) else False)

    def _track_1_overlap_with_track_0_pred(self, _frame: int) -> bool:
        return self.d10 is not None and self.d10 < self.overlap_dist
    
    def _track_0_overlap_with_track_1_pred(self, _frame: int) -> bool:
        return self.d01 is not None and self.d01 < self.overlap_dist

    def _intra_track_overlap_with_pred(self, frame: int, track: int) -> bool:
        """Check if current position of a track is close to its own predicted position."""
        if track == 0:
            return self._distance(self.Xt0_curr, self.Xt0_pred) < self.overlap_dist if (self.Xt0_curr is not None and self.Xt0_pred is not None) else False
        else:
            return self._distance(self.Xt1_curr, self.Xt1_pred) < self.overlap_dist if (self.Xt1_curr is not None and self.Xt1_pred is not None) else False

    def _both_tracks_null(self, frame: int) -> bool:
        t0_valid = not np.all(np.isnan(self.data.x[frame, 0, :]))
        t1_valid = not np.all(np.isnan(self.data.x[frame, 1, :]))
        return not (t0_valid or t1_valid)

    def handle_frame(self, frame: int) -> bool:
        """Handle jump detection and correction for one frame."""
        # Clear old cache entries to prevent memory bloat (keep last 10 frames)
        if frame > 10:
            old_frame = frame - 10
            keys_to_remove = [k for k in self._position_cache.keys() if k[1] < old_frame]
            for k in keys_to_remove:
                del self._position_cache[k]
            keys_to_remove = [k for k in self._stable_nodes_cache.keys() if k[0] < old_frame]
            for k in keys_to_remove:
                del self._stable_nodes_cache[k]
            keys_to_remove = [k for k in self._track_validity_cache.keys() if k < old_frame]
            for k in keys_to_remove:
                del self._track_validity_cache[k]
        
        # Compute all per-frame state first
        self._compute_frame_state(frame)

        # Early out if nothing to do
        if self._both_tracks_null(frame):
            return False

        changed = False
        # 1) Distance/proximity-based jump handling
        if self._handle_distance_jumps(frame):
            changed = True
        # 2) Orientation-based jump handling
        if self._handle_orientation_jumps(frame):
            changed = True
        # 3) single node jump cleanup
        if self._handle_single_node_jumps(frame):
            changed = True


        return changed


    def _extract_and_blank(self, frame: int, target_track: int) -> None:
        """Extract nodes from the target that are not found in the other track, assign them to the other track, and blank the target track."""
        other_track = 1 - target_track
        for node_idx in range(self.data.num_nodes):
            pos_target_curr = self._get_node_position(frame, target_track, node_idx)
            pos_other_curr = self._get_node_position(frame, other_track, node_idx)
            if pos_target_curr is not None and pos_other_curr is None:
                try:
                    node_name = self.data.nodes[node_idx]
                    if node_name == 'headstage':
                        continue
                    self._swap_nodes(frame, node_name, other_track)
                except Exception:
                    pass
        self.data.blank_frame(frame, target_track)
        self.corrections.append({
            'frame': frame,
            'type': 'extract_and_blank',
            'action': f'Extracted unique nodes from track {target_track} to track {other_track} and blanked track {target_track} at frame {frame}'
        })




    def _remove_overlapping_points(self, frame: int, track: int) -> None:
        """Remove overlapping points from the specified track at the given frame."""
        # check all nodes for overlap with the other track's same nodes
        other_track = 1 - track
        for node_idx in range(self.data.num_nodes):
            pos_track_curr = self._get_node_position(frame, track, node_idx)
            pos_other_curr = self._get_node_position(frame, other_track, node_idx)
            pos_track_prev = self._get_node_position(frame - 1, track, node_idx)
            pos_other_prev = self._get_node_position(frame - 1, other_track, node_idx)
            if pos_other_curr is not None and pos_track_curr is not None:
                if self._distance(pos_track_curr, pos_other_curr) < 18:
                    self.data.blank_node(frame, track, self.data.nodes[node_idx])
                    continue
                if self.data.nodes[node_idx] == 'spine_base' and self._distance(pos_track_curr, pos_other_curr) < self.Jt + 1:
                    self.data.blank_node(frame, track, self.data.nodes[node_idx])
                    continue
            if pos_track_curr is not None and pos_other_prev is not None:
                if self._distance(pos_track_curr, pos_other_prev) < self.overlap_dist:
                    # move this node from current track to the other track for current frame
                    try:
                        node_name = self.data.nodes[node_idx]
                        target_track = other_track
                        self._swap_nodes(frame, node_name, target_track)
                    except Exception:
                        # If swap_node not available or fails, ignore and continue
                        pass
            if pos_other_curr is not None and pos_track_prev is not None:
                if self._distance(pos_other_curr, pos_track_prev) < self.overlap_dist:
                    try:
                        node_name = self.data.nodes[node_idx]
                        target_track = track
                        self._swap_nodes(frame, node_name, target_track)
                    except Exception:
                        pass
                

    def _swap_ids(self, frame: int):
        self.data.swap_tracks(frame)
        
        # Clear ALL position and stable node caches since the swap affects
        # how we interpret the data from this frame onwards
        self._position_cache.clear()
        self._stable_nodes_cache.clear()
        
        # Recompute current frame state with the updated (post-swap) data
        # This ensures that any subsequent decisions in this frame use correct data
        self._recompute_current_frame_state(frame)
            
        self.corrections.append({
            'frame': frame,
            'type': 'identity_swap',
            'action': 'Both tracks jumped and swapped - corrected by swapping IDs'
        })

    def _swap_nodes(self, frame: int, node_name: str, target_track: int):
        """Move a single node to the specified target track at the given frame.

        This explicitly moves x,y,score for `node_name` into `target_track` and
        blanks the source. It clears caches and recomputes frame state so
        subsequent logic uses consistent data. A correction entry is appended.
        """
        # Determine node index
        try:
            node_idx = self.data.nodes.index(node_name)
        except ValueError:
            raise

        # Choose source based on which track currently has a valid position
        x0 = self.data.x[frame, 0, node_idx]
        y0 = self.data.y[frame, 0, node_idx]
        x1 = self.data.x[frame, 1, node_idx]
        y1 = self.data.y[frame, 1, node_idx]

        if not (np.isnan(x0) or np.isnan(y0)) and (np.isnan(x1) or np.isnan(y1)):
            src = 0
        elif not (np.isnan(x1) or np.isnan(y1)) and (np.isnan(x0) or np.isnan(y0)):
            src = 1
        else:
            # Ambiguous: pick the opposite of the requested target as source
            src = 1 - target_track

        other = target_track

        # Move values from src -> other
        self.data.x[frame, other, node_idx] = self.data.x[frame, src, node_idx]
        self.data.y[frame, other, node_idx] = self.data.y[frame, src, node_idx]
        self.data.score[frame, other, node_idx] = self.data.score[frame, src, node_idx]

        # Blank the source position
        self.data.blank_node(frame, src, node_name)

        # Clear caches affected by the change
        self._position_cache.clear()
        self._stable_nodes_cache.clear()

        # Recompute current frame state with the updated (post-swap) data
        self._recompute_current_frame_state(frame)

        # Log correction
        self.corrections.append({
            'frame': frame,
            'type': 'node_swap',
            'action': f'Node "{node_name}" moved to track {other} at frame {frame} (source was {src})'
        })
    
    def _remove_wrong_track(self, frame: int, Xt0_pred: Optional[Tuple[float, float]], Xt1_pred: Optional[Tuple[float, float]]):
        """Remove the track that is further from the track's prediction.
        
        When one track jumped and landed near where the other was predicted to be,
        that's the "wrong" track (likely stole the other's identity).
        """
        Xt0_curr = self._get_track_position(0, frame)
        Xt1_curr = self._get_track_position(1, frame)
        
        # How close is each track to the OTHER track's predicted position?
        d01 = self._distance(Xt0_curr, Xt1_pred)  # track 0 → track 1's prediction
        d10 = self._distance(Xt1_curr, Xt0_pred)  # track 1 → track 0's prediction
        
        # Remove the track that's closer to the other's prediction (the impostor)
        if d01 < d10:
            self.data.blank_frame(frame, 1)
            self.corrections.append({'frame': frame, 'track': 1, 'type': 'track_removed', 'action': f'Track 0 removed - landed {d01:.1f}px from track 1 prediction (likely stole identity)'})
        else:
            self.data.blank_frame(frame, 0)
            self.corrections.append({'frame': frame, 'track': 0, 'type': 'track_removed', 'action': f'Track 1 removed - landed {d10:.1f}px from track 0 prediction (likely stole identity)'})
    
    def _reassign_track(self, frame: int, source_track: int, target_track: int):
        self._swap_ids(frame)
        self.data.blank_frame(frame, target_track)
        self.corrections.append({'frame': frame, 'type': 'track_reassignment', 'action': f'Track {source_track} reassigned with Track {target_track} data'})

    def orientation_flip_check(self, frame: int, track: int) -> bool | None:
        orientation_current = self._get_track_orientation(track, frame)
        orientation_previous = self._get_track_orientation(track, frame - 1)
        if orientation_current is None or orientation_previous is None:
            return None
        angle_diff = abs(np.tan(orientation_current) - np.tan(orientation_previous))
        return angle_diff > self.orientation_flip_threshold  # more than 90 degrees   












class SkeletonStretchHandler:
    """Stage 3: Skeleton stretch and orientation validation."""
    def __init__(self, params: CorrectionParams, data: TrackingData):
        self.params = params
        self.data = data
        self.corrections: List[Dict] = []

        self.nose_idx = data.get_node_index('nose')
        self.neck_idx = data.get_node_index('neck')
        self.spine_idx = data.get_node_index('spine_base')
        self.tail_idx = data.get_node_index('tail_base')
        self.hs_idx = data.get_node_index(self.params.headstage_node)
        self.Jt = self.params.max_centroid_velocity
        self.spine_overlap_dist = self.params.spine_overlap_dist
        self.overlap_jump_dist = self.params.overlap_jump_dist
        self.overlap_dist = self.params.overlap_dist
        self.proximate_jump_thresh = self.params.proximate_jump_thresh
        # Orientation handler is now called externally after stretch validation

        # TODO: pass this through correction params
        # Handle plausibility_radius if provided
        if self.params.plausibility_radius is not None:
            self.max_nose_neck = self.params.plausibility_radius.get('nose_neck')
            self.max_neck_spine = self.params.plausibility_radius.get('neck_spine')
            self.max_spine_tail = self.params.plausibility_radius.get('spine_tail')
            self.max_hs_nose = self.params.plausibility_radius.get('headstage_nose')
            self.max_headstage_neck = self.params.plausibility_radius.get('headstage_neck')
            self.max_neck_tail = self.params.plausibility_radius.get('neck_tail')
        else:
            self.max_nose_neck = 100
            self.max_neck_spine = 100
            self.max_spine_tail = 100
            self.max_hs_nose = 65

    def _get_node_position(self, frame: int, track: int, node_idx: int) -> Optional[Tuple[float, float]]:
        """Get the position of a specific node for a given frame and track."""
        x = self.data.x[frame, track, node_idx]
        y = self.data.y[frame, track, node_idx]
        if np.isnan(x) or np.isnan(y):
            return None
        return (x, y)

    def _distance(self, pos1: Optional[Tuple[float, float]], pos2: Optional[Tuple[float, float]]) -> Optional[float]:
        """Calculate Euclidean distance between two positions."""
        if pos1 is None or pos2 is None:
            return None
        return np.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])

    def _set_node_position(self, frame: int, track: int, node_idx: int, pos: Tuple[float, float]) -> None:
        """Set the position of a specific node for a given frame and track."""
        self.data.x[frame, track, node_idx] = pos[0]
        self.data.y[frame, track, node_idx] = pos[1]

    def _check_strech(self, pos1: Optional[Tuple[float, float]], pos2: Optional[Tuple[float, float]], max_dist: float, frame: int, track: int) -> Optional[bool]:
        """Check distance between two nodes and return True if it exceeds max_dist."""
        if pos1 is None or pos2 is None:
            return None
        dist = self._distance(pos1, pos2)
        if dist is None:
            return None
        return dist > max_dist

    def _get_pos_array(self, frame: int, track: int) -> List[Optional[Tuple[float, float]]]:
        """Get positions of all nodes for a given frame and track as a list of tuples."""
        positions = []
        for node_idx in range(self.data.num_nodes):
            positions.append(self._get_node_position(frame, track, node_idx))
        return positions

    def handle_frame(self, frame: int) -> bool:
        """Handle skeleton stretch validation for one frame. Returns True if any changes were made."""
        changed = False
        for track in [0, 1]:
            if self._check_intra_node_overlap(frame, track):
                changed = True
            if self._check_track_skeleton(frame, track):
                changed = True
        if self._check_node_overlap(frame):
            changed = True
        return changed
    
    def _check_intra_node_overlap(self, frame: int, track: int) -> bool:
        """Check for overlaps within a track's skeleton. Returns True if any changes were made."""
        changed = False
        num_nodes = self.data.num_nodes
        outlier_nodes = [False] * num_nodes
        overlap_pairs = {}  # Use dict with (i,j) tuple keys
        
        for i in range(num_nodes):
            if outlier_nodes[i]:
                continue
            pos_i = self._get_node_position(frame, track, i)
            if pos_i is None:
                continue
            pos_i_prev = self._get_node_position(frame - 1, track, i) if frame > 0 else None
            dist_i_prev = self._distance(pos_i, pos_i_prev) if pos_i_prev is not None else None
            
            for j in range(i + 1, num_nodes):
                if outlier_nodes[j]:
                    continue
                pos_j = self._get_node_position(frame, track, j)
                if pos_j is None:
                    continue
                pos_j_prev = self._get_node_position(frame - 1, track, j) if frame > 0 else None
                dist_j_prev = self._distance(pos_j, pos_j_prev) if pos_j_prev is not None else None
                dist_i_j = self._distance(pos_i, pos_j)
                if dist_i_j is not None and dist_i_j < self.overlap_jump_dist:
                    if dist_i_prev is not None and dist_j_prev is not None:
                        if dist_i_prev > self.Jt and dist_j_prev > self.Jt:
                            overlap_pairs[(i, j)] = True
                            continue  # both jumped
                        if dist_i_prev < self.Jt and dist_j_prev > self.Jt:
                            # i and j overlapped; j jumped
                            outlier_nodes[j] = True
                            continue
                        if dist_j_prev < self.Jt and dist_i_prev > self.Jt:
                            # i and j overlapped; i jumped
                            outlier_nodes[i] = True
                            break
                    if dist_i_j < self.overlap_dist:
                        # Jump not detected; overlap pair
                        overlap_pairs[(i, j)] = True
                        continue
        # loop through overlap pairs
        for (i, j), is_overlap in overlap_pairs.items():
            if is_overlap:
                if outlier_nodes[i] or outlier_nodes[j]:
                    overlap_pairs[(i, j)] = False
                    continue
        # find the nodes that overlap with more than one other node
        overlap_count = [0] * num_nodes
        for (i, j), is_overlap in overlap_pairs.items():
            if is_overlap:
                overlap_count[i] += 1
                overlap_count[j] += 1
        for i in range(num_nodes):
            if outlier_nodes[i]:
                continue
            if overlap_count[i] > 1:
                outlier_nodes[i] = True
                overlap_count[i] = 0
        for (i, j), is_overlap in overlap_pairs.items():
            if is_overlap:
                if outlier_nodes[i] or outlier_nodes[j]:
                    overlap_pairs[(i, j)] = False
                    continue
        # marking overlap pairs: mark one node based on previous frame proximity, if available.
        # If no previous frame data, mark the one with lower score
        for (i, j), is_overlap in overlap_pairs.items():
            if is_overlap:
                pos_i_prev = self._get_node_position(frame - 1, track, i) if frame > 0 else None
                pos_j_prev = self._get_node_position(frame - 1, track, j) if frame > 0 else None
                pos_i_curr = self._get_node_position(frame, track, i)
                pos_j_curr = self._get_node_position(frame, track, j)
                dist_i_prev = self._distance(pos_i_curr, pos_i_prev) if pos_i_prev is not None and pos_i_curr is not None else None
                dist_j_prev = self._distance(pos_j_curr, pos_j_prev) if pos_j_prev is not None and pos_j_curr is not None else None
                if dist_i_prev is not None and dist_j_prev is not None:
                    if dist_i_prev > self.overlap_dist and dist_j_prev < self.overlap_dist:
                        outlier_nodes[i] = True
                        overlap_pairs[(i, j)] = False
                    elif dist_j_prev > self.overlap_dist and dist_i_prev < self.overlap_dist:
                        outlier_nodes[j] = True
                        overlap_pairs[(i, j)] = False
                elif dist_i_prev is not None:
                    if dist_i_prev < self.overlap_dist:
                        outlier_nodes[j] = True
                        overlap_pairs[(i, j)] = False
                elif dist_j_prev is not None:
                    if dist_j_prev < self.overlap_dist:
                        outlier_nodes[i] = True
                        overlap_pairs[(i, j)] = False
                else:

                    score_i = self.data.score[frame, track, i]
                    score_j = self.data.score[frame, track, j]
                    if score_i < score_j:
                        outlier_nodes[i] = True
                        overlap_pairs[(i, j)] = False
                    else:
                        outlier_nodes[j] = True
                        overlap_pairs[(i, j)] = False
        # assign to outlier nodes previous frame position if available, else blank
        for i in range(num_nodes):
            if outlier_nodes[i]:
                pos_i_prev = self._get_node_position(frame - 1, track, i) if frame > 0 else None
                if pos_i_prev is not None:
                    # Check if the previous position was already a corrected value to prevent chaining corrections
                    assign = True
                    if frame > 1:
                        pos_i_prev_prev = self._get_node_position(frame - 2, track, i)
                        if pos_i_prev_prev is not None and pos_i_prev == pos_i_prev_prev:
                            assign = False
                    if assign:
                        self._set_node_position(frame, track, i, pos_i_prev)
                        self.corrections.append({'frame': frame, 'track': track, 'type': 'intra_node_overlap', 'action': f'Node {self.data.nodes[i]} position set to previous frame due to overlap (track {track})'})
                    else:
                        self.data.blank_node(frame, track, self.data.nodes[i])
                        self.corrections.append({'frame': frame, 'track': track, 'type': 'intra_node_overlap', 'action': f'Node {self.data.nodes[i]} removed due to overlap (previous position was corrected, avoiding chaining)'})
                else:
                    self.data.blank_node(frame, track, self.data.nodes[i])
                    self.corrections.append({'frame': frame, 'track': track, 'type': 'intra_node_overlap', 'action': f'Node {self.data.nodes[i]} removed due to overlap with no previous frame data (track {track})'})
                changed = True
        return changed

    def _check_node_overlap(self, frame: int) -> bool:
        """Check and remove overlapping nodes. Returns True if any changes were made."""
        changed = False
        num_nodes = self.data.num_nodes

        def restore_or_blank(target_track: int, node_index: int, prev_x: float, prev_y: float, node_name: str) -> None:
            """Restore node to prior frame if available; otherwise blank it."""
            if not (np.isnan(prev_x) or np.isnan(prev_y)):
                self._set_node_position(frame, target_track, node_index, (prev_x, prev_y))
            else:
                self.data.blank_node(frame, target_track, node_name)

        for node_idx in range(num_nodes):
            x0 = self.data.x[frame, 0, node_idx]
            y0 = self.data.y[frame, 0, node_idx]
            x1 = self.data.x[frame, 1, node_idx]
            y1 = self.data.y[frame, 1, node_idx]
            x0_prev = self.data.x[frame - 1, 0, node_idx]
            y0_prev = self.data.y[frame - 1, 0, node_idx]
            x1_prev = self.data.x[frame - 1, 1, node_idx]
            y1_prev = self.data.y[frame - 1, 1, node_idx]
            valid0_prev = not (np.isnan(x0_prev) or np.isnan(y0_prev))
            valid1_prev = not (np.isnan(x1_prev) or np.isnan(y1_prev))
            d0_prev = np.hypot(x0 - x0_prev, y0 - y0_prev) if valid0_prev else np.inf
            d1_prev = np.hypot(x1 - x1_prev, y1 - y1_prev) if valid1_prev else np.inf
            if not (np.isnan(x0) or np.isnan(y0) or np.isnan(x1) or np.isnan(y1)):
                if abs(x0 - x1) < self.params.overlap_dist and abs(y0 - y1) < self.params.overlap_dist:
                    removed = False
                    # First, if possible, check proximity to previous frame for each track
                    if frame > 0:




                        within0 = d0_prev <= self.params.overlap_dist
                        within1 = d1_prev <= self.params.overlap_dist

                        # If only one is within the threshold relative to its own previous frame, keep that one
                        if within0 and not within1:
                            restore_or_blank(1, node_idx, x1_prev, y1_prev, self.data.get_node_name(node_idx))
                            self.corrections.append({'frame': frame, 'track': 1, 'type': 'node_overlap', 'action': f'Node {self.data.get_node_name(node_idx)} removed from track 1 due to overlap (track 0 consistent with previous frame)'})
                            removed = True
                            changed = True
                        elif within1 and not within0:
                            restore_or_blank(0, node_idx, x0_prev, y0_prev, self.data.get_node_name(node_idx))
                            self.corrections.append({'frame': frame, 'track': 0, 'type': 'node_overlap', 'action': f'Node {self.data.get_node_name(node_idx)} removed from track 0 due to overlap (track 1 consistent with previous frame)'})
                            removed = True
                            changed = True
                        elif d0_prev is not np.inf and d0_prev > self.Jt:
                            restore_or_blank(0, node_idx, x0_prev, y0_prev, self.data.get_node_name(node_idx))
                            self.corrections.append({'frame': frame, 'track': 0, 'type': 'node_overlap', 'action': f'Node {self.data.get_node_name(node_idx)} removed from track 0 due to jump (track 0 jumped)'})
                            removed = True
                            changed = True
                        elif d1_prev is not np.inf and d1_prev > self.Jt:
                            restore_or_blank(1, node_idx, x1_prev, y1_prev, self.data.get_node_name(node_idx))
                            self.corrections.append({'frame': frame, 'track': 1, 'type': 'node_overlap', 'action': f'Node {self.data.get_node_name(node_idx)} removed from track 1 due to jump (track 1 jumped)'})
                            removed = True
                            changed = True
                    if not removed:
                        # check prev prev is frame -1 is not available, check frame -2
                        if frame > 1:
                            x0_prev2 = self.data.x[frame - 2, 0, node_idx]
                            y0_prev2 = self.data.y[frame - 2, 0, node_idx]
                            x1_prev2 = self.data.x[frame - 2, 1, node_idx]
                            y1_prev2 = self.data.y[frame - 2, 1, node_idx]
                            valid0_prev2 = not (np.isnan(x0_prev2) or np.isnan(y0_prev2))
                            valid1_prev2 = not (np.isnan(x1_prev2) or np.isnan(y1_prev2))
                            d0_prev2 = np.hypot(x0 - x0_prev2, y0 - y0_prev2) if valid0_prev2 else np.inf
                            d1_prev2 = np.hypot(x1 - x1_prev2, y1 - y1_prev2) if valid1_prev2 else np.inf
                            within0_2 = d0_prev2 <= self.params.overlap_dist
                            within1_2 = d1_prev2 <= self.params.overlap_dist
                            jump0_2 = d0_prev2 > self.Jt
                            jump1_2 = d1_prev2 > self.Jt
                            if within0_2 and not within1_2:
                                restore_or_blank(1, node_idx, x1_prev, y1_prev, self.data.get_node_name(node_idx))
                                self.corrections.append({'frame': frame, 'track': 1, 'type': 'node_overlap', 'action': f'Node {self.data.get_node_name(node_idx)} removed from track 1 due to overlap (track 0 consistent with frame-2)'})
                                removed = True
                                changed = True
                            elif within1_2 and not within0_2:
                                restore_or_blank(0, node_idx, x0_prev, y0_prev, self.data.get_node_name(node_idx))
                                self.corrections.append({'frame': frame, 'track': 0, 'type': 'node_overlap', 'action': f'Node {self.data.get_node_name(node_idx)} removed from track 0 due to overlap (track 1 consistent with frame-2)'})
                                removed = True
                                changed = True
                            elif d0_prev2 is not np.inf and d0_prev2 > self.Jt:
                                restore_or_blank(0, node_idx, x0_prev, y0_prev, self.data.get_node_name(node_idx))
                                self.corrections.append({'frame': frame, 'track': 0, 'type': 'node_overlap', 'action': f'Node {self.data.get_node_name(node_idx)} removed from track 0 due to jump (track 0 jumped relative to frame-2)'})
                                removed = True
                                changed = True
                            elif d1_prev2 is not np.inf and d1_prev2 > self.Jt:
                                restore_or_blank(1, node_idx, x1_prev, y1_prev, self.data.get_node_name(node_idx))
                                self.corrections.append({'frame': frame, 'track': 1, 'type': 'node_overlap', 'action': f'Node {self.data.get_node_name(node_idx)} removed from track 1 due to jump (track 1 jumped relative to frame-2)'})
                                removed = True
                                changed = True
                    
                    if not removed:
                        # Fallback to median-based decision
                        valid0 = ~(np.isnan(self.data.x[:, 0, node_idx]) | np.isnan(self.data.y[:, 0, node_idx]))
                        valid1 = ~(np.isnan(self.data.x[:, 1, node_idx]) | np.isnan(self.data.y[:, 1, node_idx]))
                        median_x0 = np.median(self.data.x[valid0, 0, node_idx]) if np.any(valid0) else x0
                        median_y0 = np.median(self.data.y[valid0, 0, node_idx]) if np.any(valid0) else y0
                        median_x1 = np.median(self.data.x[valid1, 1, node_idx]) if np.any(valid1) else x1
                        median_y1 = np.median(self.data.y[valid1, 1, node_idx]) if np.any(valid1) else y1

                        dist0 = np.hypot(x0 - median_x0, y0 - median_y0)
                        dist1 = np.hypot(x1 - median_x1, y1 - median_y1)

                        if dist0 > dist1:
                            restore_or_blank(0, node_idx, x0_prev, y0_prev, self.data.get_node_name(node_idx))
                            self.corrections.append({'frame': frame, 'track': 0, 'type': 'node_overlap', 'action': f'Node {self.data.get_node_name(node_idx)} removed from track 0 due to overlap (furthest from median)'})
                            changed = True
                        else:
                            restore_or_blank(1, node_idx, x1_prev, y1_prev, self.data.get_node_name(node_idx))
                            self.corrections.append({'frame': frame, 'track': 1, 'type': 'node_overlap', 'action': f'Node {self.data.get_node_name(node_idx)} removed from track 1 due to overlap (furthest from median)'})
                            changed = True
                else:
                    
                    dist = np.hypot(x0 - x1, y0 - y1)
                    if frame > 0:
                        x0_prev = self.data.x[frame - 1, 0, node_idx]
                        y0_prev = self.data.y[frame - 1, 0, node_idx]
                        x1_prev = self.data.x[frame - 1, 1, node_idx]
                        y1_prev = self.data.y[frame - 1, 1, node_idx]
                        valid0_prev = not (np.isnan(x0_prev) or np.isnan(y0_prev))
                        valid1_prev = not (np.isnan(x1_prev) or np.isnan(y1_prev))
                        d0_prev = np.hypot(x0 - x0_prev, y0 - y0_prev) if valid0_prev else np.inf
                        d1_prev = np.hypot(x1 - x1_prev, y1 - y1_prev) if valid1_prev else np.inf
                        jump0 = d0_prev > self.Jt
                        jump1 = d1_prev > self.Jt
                        if (dist < self.spine_overlap_dist and self.data.nodes[node_idx] == 'spine_base') :
                            if jump0 and not jump1:
                                restore_or_blank(0, node_idx, x0_prev, y0_prev, 'spine_base')
                                self.corrections.append({'frame': frame, 'track': 0, 'type': 'spine_overlap', 'action': f'Spine base removed from track 0 due to overlap (track 0 spine jumped)'})
                                changed = True
                            elif jump1 and not jump0:
                                restore_or_blank(1, node_idx, x1_prev, y1_prev, 'spine_base')
                                self.corrections.append({'frame': frame, 'track': 1, 'type': 'spine_overlap', 'action': f'Spine base removed from track 1 due to overlap (track 1 spine jumped)'})
                                changed = True
                            elif (self.data.count_visible_nodes(frame,1) > 2 and self.data.count_visible_nodes(frame,0) <= 2):
                                restore_or_blank(0, node_idx, x0_prev, y0_prev, 'spine_base')
                                self.corrections.append({'frame': frame, 'track': 0, 'type': 'spine_overlap', 'action': f'Spine base removed from track 0 due to overlap (track 0 spine jumped)'})
                                changed = True
                            elif (self.data.count_visible_nodes(frame,0) > 2 and self.data.count_visible_nodes(frame,1) <= 2):
                                restore_or_blank(1, node_idx, x1_prev, y1_prev, 'spine_base')
                                self.corrections.append({'frame': frame, 'track': 1, 'type': 'spine_overlap', 'action': f'Spine base removed from track 1 due to overlap (track 1 spine jumped)'})
                                changed = True
                        if (dist < self.params.overlap_jump_dist and self.data.nodes[node_idx] != 'spine_base'):
                            if jump0 and not jump1:
                                restore_or_blank(0, node_idx, x0_prev, y0_prev, self.data.get_node_name(node_idx))
                                self.corrections.append({'frame': frame, 'track': 0, 'type': 'node_overlap', 'action': f'Node {self.data.get_node_name(node_idx)} removed from track 0 due to overlap (track 0 node jumped)'})
                                changed = True
                            elif jump1 and not jump0:
                                restore_or_blank(1, node_idx, x1_prev, y1_prev, self.data.get_node_name(node_idx))
                                self.corrections.append({'frame': frame, 'track': 1, 'type': 'node_overlap', 'action': f'Node {self.data.get_node_name(node_idx)} removed from track 1 due to overlap (track 1 node jumped)'})
                                changed = True
                    
        return changed

    def _check_track_skeleton(self, frame: int, track: int) -> bool:
        """Check track skeleton for stretch issues. Returns True if any changes were made."""
        changed = False
        outlier_nodes = [False] * self.data.num_nodes
        outlier_suspects = [False] * self.data.num_nodes
        
        # Get position arrays for current and previous frames
        pos_array_curr = self._get_pos_array(frame, track)
        pos_array_prev = self._get_pos_array(frame - 1, track) if frame > 0 else None
        
        # Calculate distances from current to previous frame for each node
        dist_curr_prev_array = [None] * self.data.num_nodes
        if pos_array_prev is not None:
            for i in range(self.data.num_nodes):
                if pos_array_curr[i] is not None and pos_array_prev[i] is not None:
                    dist_curr_prev_array[i] = self._distance(pos_array_curr[i], pos_array_prev[i])
        
        # Check stretch distances between node pairs
        hs_nose = self._check_strech(pos_array_curr[self.hs_idx], pos_array_curr[self.nose_idx], self.max_hs_nose, frame, track)
        # Check neck-headstage distance
        neck_headstage = self._check_strech(pos_array_curr[self.neck_idx], pos_array_curr[self.hs_idx], self.max_headstage_neck, frame, track)
        # Check neck-nose distance
        neck_nose = self._check_strech(pos_array_curr[self.neck_idx], pos_array_curr[self.nose_idx], self.max_nose_neck, frame, track)
        # Check spine-neck distance
        spine_neck = self._check_strech(pos_array_curr[self.spine_idx], pos_array_curr[self.neck_idx], self.max_neck_spine, frame, track)
        # Check tail-spine distance
        tail_spine = self._check_strech(pos_array_curr[self.tail_idx], pos_array_curr[self.spine_idx], self.max_spine_tail, frame, track)
        
        # Analyze stretch violations and determine outliers
        if self.hs_idx is not None and pos_array_curr[self.hs_idx] is not None:
            if hs_nose is not None and hs_nose:
                # Headstage-nose stretch exceeded
                # Check which is the outlier: headstage or nose
                if dist_curr_prev_array[self.hs_idx] is not None and dist_curr_prev_array[self.nose_idx] is not None:
                    if dist_curr_prev_array[self.hs_idx] < self.overlap_dist and dist_curr_prev_array[self.nose_idx] > self.overlap_dist:
                        # headstage is consistent with previous frame; nose jumped
                        outlier_nodes[self.nose_idx] = True
                    elif dist_curr_prev_array[self.nose_idx] < self.overlap_dist and dist_curr_prev_array[self.hs_idx] > self.overlap_dist:
                        # nose is consistent with previous frame; headstage jumped
                        outlier_nodes[self.hs_idx] = True
                    else:
                        outlier_suspects[self.hs_idx] = True
                        outlier_suspects[self.nose_idx] = True
                elif dist_curr_prev_array[self.hs_idx] is not None:
                    if dist_curr_prev_array[self.hs_idx] > self.overlap_dist:
                        outlier_nodes[self.hs_idx] = True
                    else:
                        outlier_suspects[self.nose_idx] = True
                elif dist_curr_prev_array[self.nose_idx] is not None:
                    if dist_curr_prev_array[self.nose_idx] > self.overlap_dist:
                        outlier_nodes[self.nose_idx] = True
                    else:
                        outlier_suspects[self.hs_idx] = True
                else:
                    outlier_suspects[self.hs_idx] = True
                    outlier_suspects[self.nose_idx] = True
            if neck_headstage is not None and neck_headstage:
                # Neck-headstage stretch exceeded
                # Check which is the outlier: neck or headstage
                if dist_curr_prev_array[self.neck_idx] is not None and dist_curr_prev_array[self.hs_idx] is not None:
                    if dist_curr_prev_array[self.neck_idx] < self.overlap_dist and dist_curr_prev_array[self.hs_idx] > self.overlap_dist:
                        # neck is consistent with previous frame; headstage jumped
                        outlier_nodes[self.hs_idx] = True
                    elif dist_curr_prev_array[self.hs_idx] < self.overlap_dist and dist_curr_prev_array[self.neck_idx] > self.overlap_dist:
                        # headstage is consistent with previous frame; neck jumped
                        outlier_nodes[self.neck_idx] = True
                    else:
                        outlier_suspects[self.neck_idx] = True
                        outlier_suspects[self.hs_idx] = True
                elif dist_curr_prev_array[self.neck_idx] is not None:
                    if dist_curr_prev_array[self.neck_idx] > self.overlap_dist:
                        outlier_nodes[self.neck_idx] = True
                    else:
                        outlier_suspects[self.hs_idx] = True
                elif dist_curr_prev_array[self.hs_idx] is not None:
                    if dist_curr_prev_array[self.hs_idx] > self.overlap_dist:
                        outlier_nodes[self.hs_idx] = True
                    else:
                        outlier_suspects[self.neck_idx] = True
                else:
                    outlier_suspects[self.neck_idx] = True
                    outlier_suspects[self.hs_idx] = True
        if neck_nose is not None and neck_nose:
            # Neck-nose stretch exceeded
            # Check which is the outlier: neck or nose
            if dist_curr_prev_array[self.neck_idx] is not None and dist_curr_prev_array[self.nose_idx] is not None:
                if dist_curr_prev_array[self.neck_idx] < self.overlap_dist and dist_curr_prev_array[self.nose_idx] > self.overlap_dist:
                    # neck is consistent with previous frame; nose jumped
                    outlier_nodes[self.nose_idx] = True
                elif dist_curr_prev_array[self.nose_idx] < self.overlap_dist and dist_curr_prev_array[self.neck_idx] > self.overlap_dist:
                    # nose is consistent with previous frame; neck jumped
                    outlier_nodes[self.neck_idx] = True
                else:
                    outlier_suspects[self.neck_idx] = True
                    outlier_suspects[self.nose_idx] = True
            elif dist_curr_prev_array[self.neck_idx] is not None:
                if dist_curr_prev_array[self.neck_idx] > self.overlap_dist:
                    outlier_nodes[self.neck_idx] = True
                else:
                    outlier_suspects[self.nose_idx] = True
            elif dist_curr_prev_array[self.nose_idx] is not None:
                if dist_curr_prev_array[self.nose_idx] > self.overlap_dist:
                    outlier_nodes[self.nose_idx] = True
                else:
                    outlier_suspects[self.neck_idx] = True
            else:
                outlier_suspects[self.neck_idx] = True
                outlier_suspects[self.nose_idx] = True
        if spine_neck is not None and spine_neck:
            # Spine-neck stretch exceeded
            # Check which is the outlier: spine or neck
            if dist_curr_prev_array[self.spine_idx] is not None and dist_curr_prev_array[self.neck_idx] is not None:
                if dist_curr_prev_array[self.spine_idx] < self.overlap_dist and dist_curr_prev_array[self.neck_idx] > self.overlap_dist:
                    # spine is consistent with previous frame; neck jumped
                    outlier_nodes[self.neck_idx] = True
                elif dist_curr_prev_array[self.neck_idx] < self.overlap_dist and dist_curr_prev_array[self.spine_idx] > self.overlap_dist:
                    # neck is consistent with previous frame; spine jumped
                    outlier_nodes[self.spine_idx] = True
                else:
                    outlier_suspects[self.spine_idx] = True
                    outlier_suspects[self.neck_idx] = True
            elif dist_curr_prev_array[self.spine_idx] is not None:
                if dist_curr_prev_array[self.spine_idx] > self.overlap_dist:
                    outlier_nodes[self.spine_idx] = True
                else:
                    outlier_suspects[self.neck_idx] = True
            elif dist_curr_prev_array[self.neck_idx] is not None:
                if dist_curr_prev_array[self.neck_idx] > self.overlap_dist:
                    outlier_nodes[self.neck_idx] = True
                else:
                    outlier_suspects[self.spine_idx] = True
            else:
                outlier_suspects[self.spine_idx] = True
                outlier_suspects[self.neck_idx] = True
        if tail_spine is not None and tail_spine:
            # Tail-spine stretch exceeded
            # Check which is the outlier: tail or spine
            if dist_curr_prev_array[self.tail_idx] is not None and dist_curr_prev_array[self.spine_idx] is not None:
                if dist_curr_prev_array[self.tail_idx] < self.overlap_dist and dist_curr_prev_array[self.spine_idx] > self.overlap_dist:
                    # tail is consistent with previous frame; spine jumped
                    outlier_nodes[self.spine_idx] = True
                elif dist_curr_prev_array[self.spine_idx] < self.overlap_dist and dist_curr_prev_array[self.tail_idx] > self.overlap_dist:
                    # spine is consistent with previous frame; tail jumped
                    outlier_nodes[self.tail_idx] = True
                else:
                    outlier_suspects[self.tail_idx] = True
                    outlier_suspects[self.spine_idx] = True
            elif dist_curr_prev_array[self.tail_idx] is not None:
                if dist_curr_prev_array[self.tail_idx] > self.overlap_dist:
                    outlier_nodes[self.tail_idx] = True
                else:
                    outlier_suspects[self.spine_idx] = True
            elif dist_curr_prev_array[self.spine_idx] is not None:
                if dist_curr_prev_array[self.spine_idx] > self.overlap_dist:
                    outlier_nodes[self.spine_idx] = True
                else:
                    outlier_suspects[self.tail_idx] = True
            else:
                outlier_suspects[self.tail_idx] = True
                outlier_suspects[self.spine_idx] = True
        # Handle suspects:
        # Check if suspects overlaps with previous frame position; if so, clear suspect
        outlier_suspects = self._filter_outlier_suspects_by_overlap(outlier_suspects, frame, track)
        # Check for each edge, if only one suspect in the edge, mark it as outlier
        edges = [(self.hs_idx, self.nose_idx), (self.neck_idx, self.hs_idx), (self.neck_idx, self.nose_idx), (self.spine_idx, self.neck_idx), (self.tail_idx, self.spine_idx)]
        edges_bol = [hs_nose, neck_headstage, neck_nose, spine_neck, tail_spine]
        suspect_count = [0] * self.data.num_nodes
        z = 0
        # if even number of suspects:
        if sum(outlier_suspects) == 1 and sum(outlier_nodes) == 0:
            suspect_count[outlier_suspects.index(True)] = 1
        for (i, j) in edges:
            if outlier_suspects[i] and not outlier_suspects[j] and edges_bol[z]:
                suspect_count[i] += 1
            elif outlier_suspects[j] and not outlier_suspects[i] and edges_bol[z]:
                suspect_count[j] += 1
            elif outlier_suspects[i] and outlier_suspects[j] and edges_bol[z]:
                suspect_count[i] += 1
                suspect_count[j] += 1
            elif outlier_suspects[i] and outlier_suspects[j] and not edges_bol[z]:
                suspect_count[i] += 1
                suspect_count[j] += 1
            z += 1
        # mark nodes with more than one suspect count as outliers
        for i in range(self.data.num_nodes):
            if suspect_count[i] > 1:
                outlier_nodes[i] = True
                outlier_suspects[i] = False
        
        
        # Check for remaining suspects, mark the one with lower score, per edge
        for (i, j) in edges:
            if outlier_suspects[i] and outlier_suspects[j]:
                score_i = self.data.score[frame, track, i]
                score_j = self.data.score[frame, track, j]
                if score_i < score_j:
                    outlier_nodes[j] = True
                else:
                    outlier_nodes[i] = True
                outlier_suspects[i] = False
                outlier_suspects[j] = False
        # Assign to outlier nodes previous frame position if available, else blank
        for i in range(self.data.num_nodes):
            if outlier_nodes[i]:
                if pos_array_prev is not None and pos_array_prev[i] is not None:
                    # Check if the previous position was already a corrected value to prevent chaining corrections
                    assign = True
                    if frame > 1:
                        pos_array_prev_prev = self._get_pos_array(frame - 2, track)
                        if pos_array_prev_prev is not None and pos_array_prev_prev[i] is not None and pos_array_prev[i] == pos_array_prev_prev[i]:
                            assign = False
                    if assign:
                        self._set_node_position(frame, track, i, pos_array_prev[i])
                        self.corrections.append({'frame': frame, 'track': track, 'type': 'skeleton_stretch', 'action': f'Node {self.data.nodes[i]} position set to previous frame due to stretch (track {track})'})
                    else:
                        self.data.blank_node(frame, track, self.data.nodes[i])
                        self.corrections.append({'frame': frame, 'track': track, 'type': 'skeleton_stretch', 'action': f'Node {self.data.nodes[i]} removed due to stretch (previous position was corrected, avoiding chaining)'})
                else:
                    self.data.blank_node(frame, track, self.data.nodes[i])
                    self.corrections.append({'frame': frame, 'track': track, 'type': 'skeleton_stretch', 'action': f'Node {self.data.nodes[i]} removed due to stretch with no previous frame data (track {track})'})
                changed = True
        
        return changed

    def _filter_outlier_suspects_by_overlap(self, outlier_suspects: List[bool], frame: int, track: int) -> List[bool]:
        """Return a modified outlier_suspects list where suspects overlapping with previous frame position are cleared to False."""
        for i in range(len(outlier_suspects)):
            if not outlier_suspects[i]:
                continue
            curr_pos = self._get_node_position(frame, track, i)
            prev_pos = self._get_node_position(frame - 1, track, i) if frame > 0 else None
            if curr_pos is not None and prev_pos is not None:
                dist = self._distance(curr_pos, prev_pos)
                if dist is not None and dist < self.overlap_dist:
                    outlier_suspects[i] = False
        return outlier_suspects

class SkeletonOrientationHandler:
    """Handles skeleton orientation checks and node removal based on orientation cosines."""
    def __init__(self, params: CorrectionParams, data: TrackingData):
        self.params = params
        self.data = data
        self.corrections: List[Dict] = []
        self.nose_idx = data.get_node_index('nose')
        self.neck_idx = data.get_node_index('neck')
        self.spine_idx = data.get_node_index('spine_base')
        self.tail_idx = data.get_node_index('tail_base')
        self.hs_idx = data.get_node_index('headstage')
        self.max_hs_nose = self.params.plausibility_radius.get('headstage_nose')
        self.cosine_threshold = -0.6  # cosine of angle threshold for orientation checks
    
    def get_node_position(self, frame: int, track: int, node_idx: int) -> Optional[Tuple[float, float]]:
        """Get the (x, y) position of a node for a given frame and track."""
        x = self.data.x[frame, track, node_idx]
        y = self.data.y[frame, track, node_idx]
        if np.isnan(x) or np.isnan(y):
            return None
        return (x, y)
    def get_angle_cosine(self, vec1: np.ndarray, vec2: np.ndarray) -> Optional[float]:
        """Calculate the cosine of the angle between two vectors."""
        if vec1 is None or vec2 is None:
            return None
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return None
        return np.dot(vec1, vec2) / (norm1 * norm2)
    def get_vector(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> np.ndarray:
        """Get the vector from pos1 to pos2."""
        if pos1 is None or pos2 is None:
            return None
        return np.array(pos2) - np.array(pos1)
    def _check_overlap_with_prev(self, suspects: Dict[str, bool], frame: int, track: int) -> None:
        """Modify suspects to False if current position overlaps with previous frame position."""
        for name, is_suspect in suspects.items():
            if not is_suspect:
                continue
            idx = {'tail': self.tail_idx, 'spine': self.spine_idx, 'neck': self.neck_idx, 'nose': self.nose_idx}[name]
            curr_pos = self.get_node_position(frame, track, idx)
            prev_pos = self.get_node_position(frame - 1, track, idx) if frame > 0 else None
            if curr_pos is not None and prev_pos is not None:
                dist = np.linalg.norm(np.array(curr_pos) - np.array(prev_pos))
                if dist < self.params.overlap_dist:  # assuming self.params.overlap_dist is available, else use a fixed value
                    suspects[name] = False
    def handle_frame(self, frame: int) -> bool:
        """Handle skeleton orientation validation for one frame. Returns True if any changes were made."""
        changed = False
        for track in [0, 1]:
            nose_pos = self.get_node_position(frame, track, self.nose_idx)
            neck_pos = self.get_node_position(frame, track, self.neck_idx)
            spine_pos = self.get_node_position(frame, track, self.spine_idx)
            tail_pos = self.get_node_position(frame, track, self.tail_idx)
            headstage_pose = self.get_node_position(frame, track, self.hs_idx)
            tail_spine_vec = self.get_vector(tail_pos, spine_pos)
            spine_neck_vec = self.get_vector(spine_pos, neck_pos)
            neck_nose_vec = self.get_vector(neck_pos, nose_pos)
            neck_hs_vec = self.get_vector(neck_pos, headstage_pose)
            hs_nose_vec = self.get_vector(headstage_pose, nose_pos)
            spine_nose_vec = self.get_vector(spine_pos, nose_pos)
            tail_neck_vec = self.get_vector(tail_pos, neck_pos)
            cos_tail_spine_neck = self.get_angle_cosine(tail_spine_vec, spine_neck_vec)
            cos_spine_neck_nose = self.get_angle_cosine(spine_neck_vec, neck_nose_vec)
            cos_neck_hs_nose = self.get_angle_cosine(neck_hs_vec, hs_nose_vec)
            cos_tail_spine_nose = self.get_angle_cosine(tail_spine_vec, spine_nose_vec)
            cos_tail_neck_nose = self.get_angle_cosine(tail_neck_vec, neck_nose_vec)
            cos_spine_neck_hs = self.get_angle_cosine(spine_neck_vec, neck_hs_vec)
            # ensure orientation consistency
            suspects = {
                'tail': False,
                'spine': False,
                'neck': False,
                'nose': False,
            }
            if cos_tail_spine_neck is None and cos_spine_neck_nose is not None and cos_spine_neck_nose < self.cosine_threshold:
                suspects['neck'] = True
                suspects['nose'] = True
                if self.data.score[frame,1,2] < 0.85:
                    suspects['spine'] = True
            if cos_spine_neck_nose is None and cos_tail_spine_neck is not None and cos_tail_spine_neck < self.cosine_threshold:
                if cos_spine_neck_hs is not None and cos_spine_neck_hs > self.cosine_threshold:
                    suspects['tail'] = True
                else:
                    suspects['tail'] = True
                    if self.data.score[frame,1,2] < 0.85:
                        suspects['spine'] = True
                    suspects['neck'] = True
            # check if the current pos overlaps with previous frame pos to avoid false positives

            
            if cos_tail_spine_neck is not None and cos_spine_neck_nose is not None:
                # Determine which nodes are orientation suspects based on cosine patterns

                if cos_tail_spine_neck < self.cosine_threshold and not (cos_spine_neck_nose < self.cosine_threshold):
                    suspects['tail'] = True
                if cos_tail_spine_neck < self.cosine_threshold and not (cos_tail_neck_nose < self.cosine_threshold if cos_tail_neck_nose is not None else False):
                    suspects['spine'] = True
                if cos_tail_spine_neck < self.cosine_threshold and not (cos_tail_spine_nose < self.cosine_threshold if cos_tail_spine_nose is not None else False):
                    suspects['neck'] = True
                if cos_tail_spine_nose is not None and cos_tail_spine_nose < self.cosine_threshold and not (cos_tail_spine_neck < self.cosine_threshold):
                    suspects['nose'] = True
                if cos_tail_neck_nose is not None and cos_tail_neck_nose < self.cosine_threshold and not (cos_spine_neck_nose < self.cosine_threshold):
                    suspects['tail'] = True

            # Check if suspects overlap with previous frame to avoid false positives
            self._check_overlap_with_prev(suspects, frame, track)

            # For each suspect node, assign previous-frame value if available
            # and not already corrected in previous frame; otherwise blank.
            node_idx_map = {
                'tail': self.tail_idx,
                'spine': self.spine_idx,
                'neck': self.neck_idx,
                'nose': self.nose_idx,
            }
            for name, is_suspect in suspects.items():
                if not is_suspect:
                    continue

                idx = node_idx_map[name]
                curr_pos = self.get_node_position(frame, track, idx)
                if curr_pos is None:
                    continue

                prev_pos = self.get_node_position(frame - 1, track, idx) if frame > 0 else None

                # Check if previous position was already a corrected value,
                # similar to stretch handler logic: if frame-1 equals frame-2,
                # consider it corrected and avoid chaining.
                prev_corrected = False
                if frame > 1 and prev_pos is not None:
                    prev_prev_pos = self.get_node_position(frame - 2, track, idx)
                    if prev_prev_pos is not None and prev_prev_pos == prev_pos:
                        prev_corrected = True

                if prev_pos is not None and not prev_corrected:
                    # Assign previous-frame position
                    self.data.x[frame, track, idx] = prev_pos[0]
                    self.data.y[frame, track, idx] = prev_pos[1]
                    self.corrections.append({
                        'frame': frame,
                        'track': track,
                        'type': 'orientation',
                        'action': f'Node {self.data.nodes[idx]} position set to previous frame due to orientation inconsistency (track {track})',
                    })
                else:
                    # No reliable previous value; blank node
                    self.data.blank_node(frame, track, self.data.nodes[idx])
                    self.corrections.append({
                        'frame': frame,
                        'track': track,
                        'type': 'orientation',
                        'action': f'Node {self.data.nodes[idx]} removed due to orientation inconsistency with no reliable previous frame data (track {track})',
                    })

                changed = True

        return changed




def process_video_minimal(df: pd.DataFrame, config: Dict, show_progress: bool = False) -> Tuple[pd.DataFrame, List[Dict], set, set]:
    """
    Process video with minimal intervention corrector - sequential frame-by-frame.
    
    COLUMN PRESERVATION: This function preserves all input columns except:
    - 'track', 'x', 'y' (which may be modified by corrections)
    All confidence columns (point_score, instance_score, etc.) are preserved unchanged.
    
    Args:
        df: Input DataFrame with columns [frame, track, node, x, y, score, point_score, instance_score, ...]
        config: Configuration dict
        show_progress: Whether to show progress bar
    
    Returns:
        Tuple of (corrected_df, corrections_list, blanked_nodes_set, swapped_frames_set)
        where:
        - corrected_df: Corrected tracking DataFrame
        - corrections_list: List of correction dicts
        - blanked_nodes_set: Set of (frame, track, node_name) tuples that were blanked
        - swapped_frames_set: Set of frame indices where track swaps occurred
    """
    print("  Using MINIMAL INTERVENTION corrector (Sequential)")
    
    # --- NEW: Guard confidence columns ---
    confidence_cols = [col for col in ['point_score', 'instance_score', 'point_score_raw', 'instance_score_raw'] 
                       if col in df.columns]
    input_confidence_data = {col: df[col].copy() for col in confidence_cols}
    
    video_fps = config.get('fps')
    params = CorrectionParams.from_config(config, video_fps=video_fps)
    data = TrackingData(df)
    
    frame_window = config.get('frame_window', 5)
    print(f"  Frame window for trajectory prediction: {frame_window}")
    
    all_corrections = []
    
    headstage_handler = HeadstageRemovalHandler(params, data)
    jump_handler = JumpHandler(params, data, frame_window)
    stretch_handler = SkeletonStretchHandler(params, data)
    orientation_handler = SkeletonOrientationHandler(params, data)
    
    print(f"  Processing frames {frame_window + 1} to {data.num_frames - 1}...")
    
    pbar = tqdm(range(0, data.num_frames), desc="Processing frames", disable=not show_progress)
    
    for frame in pbar:
        # Track if any changes were made to invalidate caches
        frame_changed = False
        
        # Stage 1: Headstage removal
        if headstage_handler.handle_frame(frame):
            frame_changed = True
        if frame_changed:
            jump_handler._position_cache.clear()
            jump_handler._stable_nodes_cache.clear()  

        # Stage 2: Jump handling (only after frame_window)
        if frame > frame_window:
            if jump_handler.handle_frame(frame):
                frame_changed = True
        # If changes were made, clear jump handler caches so subsequent 
        # processing uses updated data
        if frame_changed:
            jump_handler._position_cache.clear()
            jump_handler._stable_nodes_cache.clear()                

        # stage 2.5: reassign headstage
                    # Stage 1: Headstage removal
        if headstage_handler.handle_frame(frame):
            frame_changed = True


            
        # Stage 3: Skeleton stretch validation
        if stretch_handler.handle_frame(frame):
            frame_changed = True
            
        # If changes were made, clear caches again
        if frame_changed:
            jump_handler._position_cache.clear()
            jump_handler._stable_nodes_cache.clear()
            
        # Stage 4: Skeleton orientation validation
        if orientation_handler.handle_frame(frame):
            frame_changed = True
        # If changes were made, clear jump handler caches so subsequent 
        # processing uses updated data
        if frame_changed:
            jump_handler._position_cache.clear()
            jump_handler._stable_nodes_cache.clear()

    pbar.close()
    

    
    all_corrections.extend(jump_handler.corrections)
    all_corrections.extend(headstage_handler.corrections)
    all_corrections.extend(stretch_handler.corrections)
    all_corrections.extend(orientation_handler.corrections)

    # Ensure corrections are returned in chronological order by frame.
    # Stable sort: primary key = frame, secondary = track (if present), tertiary = type.
    try:
        all_corrections.sort(key=lambda c: (int(c.get('frame', -1)), int(c.get('track', -1)) if 'track' in c else -1, str(c.get('type', ''))))
    except Exception:
        # If any correction entries are missing expected fields or have non-int frames,
        # fall back to a safer sort by string representation to avoid raising.
        all_corrections.sort(key=lambda c: str(c))
    
    
    print(f"    Stage 1: Jump corrections: {len(jump_handler.corrections)}")
    print(f"    Stage 2: Headstage removals: {len(headstage_handler.corrections)}")
    print(f"    Stage 3: Skeleton stretch corrections: {len(stretch_handler.corrections)}")
    print(f"    Stage 4: Orientation corrections: {len(orientation_handler.corrections)}")
    print(f"    Total corrections: {len(all_corrections)}")
    
    # --- NEW: Convert back to DataFrame (to_dataframe() already merges confidence columns via df_orig) ---
    df_corrected = data.to_dataframe()
    
    # Verify confidence columns were preserved by the merge in to_dataframe()
    for col in confidence_cols:
        if col not in df_corrected.columns:
            raise ValueError(
                f"[id_corrector_minimal] CRITICAL: {col} was not restored by to_dataframe() merge. "
                f"This indicates TrackingData.to_dataframe() is not properly merging extra_cols from df_orig."
            )
    
    # Ensure columns are in original order (with new cols at end)
    col_order = list(df.columns)
    for col in df_corrected.columns:
        if col not in col_order:
            col_order.append(col)
    df_corrected = df_corrected[col_order]
    
    # Extract tracking metadata from data object
    blanked_nodes = data.blanked_nodes.copy()
    swapped_frames = data.swapped_frames.copy()
    
    return df_corrected, all_corrections, blanked_nodes, swapped_frames
