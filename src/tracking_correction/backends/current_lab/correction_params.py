"""Container for correction parameters constructed from a config dict."""
from typing import Dict, Any


class CorrectionParams:
    """Container for correction parameters constructed from a config dict.

    Usage:
        params = CorrectionParams.from_config(cfg_dict)
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        cfg = config or {}
        ## Minimal corrector settings
        # Jump detection
        self.max_centroid_velocity = cfg.get('jump_threshold')
        self.proximity_threshold = cfg.get('proximity_threshold')
        self.overlap_dist = cfg.get('overlap_dist')
        self.frame_window = cfg.get('frame_window')
        self.proximate_jump_thresh = cfg.get('proximate_jump_thresh')
        self.orientation_flip_threshold = cfg.get('orientation_flip_threshold')
        self.spine_overlap_dist = cfg.get('spine_overlap_dist')
        self.overlap_jump_dist = cfg.get('overlap_jump_dist')
        self.medium_proximity_threshold = cfg.get('medium_proximity_threshold')
        self.orientation_cosine_threshold = cfg.get('orientation_cosine_threshold')
        self.minimum_separation_distance = cfg.get('minimum_separation_distance')
        # Skeleton plausibility
        self.plausibility_radius = cfg.get('plausibility_radius')

        # Headstage ground truth
        self.headstage_node = cfg.get('headstage_node')
        # =====================================================================
        ## Smoothing settings
        self.max_gap_sec = cfg.get('max_gap_sec')
        self.min_node_score = cfg.get('min_node_score')
        self.teleport_thresh_bl = cfg.get('teleport_thresh_bl')
        self.smooth_win_sec = cfg.get('smooth_win_sec')
        self.smooth_poly = cfg.get('smooth_poly')

        # Headstage constraints during smoothing (optional)
        self.enforce_headstage = cfg.get('enforce_headstage')
        self.implanted_track = cfg.get('implanted_track')
        self.neck_node = cfg.get('neck_node')
        self.headstage_neck_max_distance = cfg.get('headstage_neck_max_distance')

    @classmethod
    def from_config(cls, config: Dict[str, Any], video_fps: float = None) -> 'CorrectionParams':
        """
        Create CorrectionParams with FPS-based adjustments.
        
        Parameters tuned for 120 FPS are automatically scaled for other frame rates.
        Only velocity/movement-related parameters are scaled; anatomical constraints remain fixed.
        
        Args:
            config: Configuration dictionary  
            video_fps: Video frame rate (required for proper parameter scaling)
            
        Returns:
            CorrectionParams instance with adjusted parameters
            
        Raises:
            ValueError: If video_fps is not provided
        """
        if video_fps is None:
            raise ValueError("video_fps must be provided for proper parameter scaling")
            
        # Create a copy of config to modify
        adjusted_config = config.copy()
        
        # Scale factor from 120 FPS baseline
        fps_scale = video_fps / 120.0
        
        # Velocity/movement parameters that need FPS scaling (pixels/frame -> pixels/frame)
        velocity_params = [
            'jump_threshold',           # centroid movement per frame
            'proximity_threshold',      # movement-based proximity detection  
            'proximate_jump_thresh'     # jump detection threshold
        ]
        
        # Frame-based parameters that need FPS scaling (frames -> frames)
        # For frame_window: at higher FPS, need more frames for same time window
        frame_params = [
            'frame_window'              # trajectory prediction window
        ]
        
        # Apply scaling to velocity parameters
        for param in velocity_params:
            if param in adjusted_config:
                original_val = adjusted_config[param]
                adjusted_config[param] = original_val * fps_scale
                
        # Apply inverse scaling to frame parameters (more frames needed at higher FPS)
        for param in frame_params:
            if param in adjusted_config:
                original_val = adjusted_config[param]
                # Keep frame window roughly constant in time duration
                adjusted_config[param] = max(1, int(original_val * fps_scale))
        
        # Parameters that should NOT be scaled (anatomical/spatial constraints):
        # - overlap_dist: spatial overlap tolerance
        # - plausibility_radius: anatomical skeleton constraints  
        # - headstage_neck_max_distance: anatomical constraint
        # - orientation_flip_threshold: angular threshold
        
        if video_fps != 120:
            scaled_params = {p: f"{config.get(p, 'N/A')} -> {adjusted_config.get(p, 'N/A')}" 
                           for p in velocity_params + frame_params if p in config}
            print(f"⚠️ CorrectionParams: Scaled parameters from 120 FPS to {video_fps} FPS (factor: {fps_scale:.3f})")
            for param, change in scaled_params.items():
                print(f"   {param}: {change}")
        
        return cls(adjusted_config)
