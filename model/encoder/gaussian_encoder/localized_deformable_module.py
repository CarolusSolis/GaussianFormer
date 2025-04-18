import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional

from mmengine.registry import MODELS
from mmengine.model import BaseModule
from mmengine import build_from_cfg
from mmengine.model import xavier_init, constant_init

from .deformable_module import DeformableFeatureAggregation
from ...utils.safe_ops import safe_sigmoid
from ...utils.utils import get_rotation_matrix
from .utils import linear_relu_ln

from .ops import DeformableAggregationFunction as DAF


@MODELS.register_module()
class LocalizedDeformableFeatureAggregation(DeformableFeatureAggregation):
    """Localized version of Deformable Feature Aggregation that restricts attention to
    a spatial window around each projected Gaussian.
    """
    def __init__(
        self,
        embed_dims: int = 256,
        num_groups: int = 8,
        num_levels: int = 4,
        num_cams: int = 6,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        kps_generator: dict = None,
        use_deformable_func=False,
        use_camera_embed=False,
        residual_mode="add",
        # Parameters specific to localized attention
        window_size: int = 15,  # Window size in pixels
        max_points_per_gaussian: int = 8,  # Max number of feature points to attend to per Gaussian
        use_depth_for_window: bool = True,  # Scale window size based on depth
        min_window_ratio: float = 0.5,  # Minimum window size ratio for close objects
        max_window_ratio: float = 2.0,  # Maximum window size ratio for far objects
    ):
        # Call parent class constructor with required parameters
        super().__init__(
            embed_dims=embed_dims,
            num_groups=num_groups,
            num_levels=num_levels,
            num_cams=num_cams,
            proj_drop=proj_drop,
            attn_drop=attn_drop,
            kps_generator=kps_generator,
            use_deformable_func=use_deformable_func,
            use_camera_embed=use_camera_embed,
            residual_mode=residual_mode,
        )
        
        # Add additional parameters for localized attention
        self.window_size = window_size
        self.max_points_per_gaussian = max_points_per_gaussian
        self.use_depth_for_window = use_depth_for_window
        self.min_window_ratio = min_window_ratio
        self.max_window_ratio = max_window_ratio
        
        # Create a position encoding for sampling points
        self.sampling_offsets = nn.Linear(embed_dims, num_groups * max_points_per_gaussian * 2)
        
        # Initialize with small values to start with a small region
        self.init_weight()
        
    def init_weight(self):
        super().init_weight()
        if hasattr(self, 'sampling_offsets'):
            nn.init.zeros_(self.sampling_offsets.weight)
            nn.init.zeros_(self.sampling_offsets.bias)
            self.sampling_offsets.bias.data = torch.randn(self.sampling_offsets.bias.shape) * 0.01
    
    def _get_adapted_window_size(self, depth, image_wh):
        """Scale window size based on depth - closer objects get smaller windows"""
        if not self.use_depth_for_window:
            return self.window_size
            
        # Normalize depths (assuming min visibility ~1.0, max ~100)
        normalized_depth = torch.clamp((depth - 1.0) / 99.0, 0.0, 1.0)
        
        # Scale ratio from min to max based on depth
        scale_ratio = self.min_window_ratio + normalized_depth * (self.max_window_ratio - self.min_window_ratio)
        
        # Scale window size
        scaled_window = self.window_size * scale_ratio
        
        # Scale by image size to get appropriate pixel values
        # Assuming image_wh is in format [bs, num_cams, 2]
        image_size = image_wh.mean(dim=1, keepdim=True)  # Average width/height
        scaled_window = scaled_window * image_size.mean(dim=-1, keepdim=True) / 1000.0  # Normalize by typical image size
        
        return scaled_window

    def _sample_localized_points(self, points_2d, instance_feature, depth, image_wh=None):
        """
        Sample a fixed number of points around each projected Gaussian center
        
        Args:
            points_2d: Projected 2D points [bs, num_anchor, num_cams, 2]
            instance_feature: Features [bs, num_anchor, embed_dims]
            depth: Depth of projected points [bs, num_anchor, num_cams]
            image_wh: Image dimensions [bs, num_cams, 2]
            
        Returns:
            sampled_points: [bs, num_anchor, num_cams, max_points, 2]
            sampling_mask: [bs, num_anchor, num_cams, max_points]
        """
        bs, num_anchor = instance_feature.shape[:2]
        
        # Generate offsets for sampling points
        offsets = self.sampling_offsets(instance_feature)
        offsets = offsets.view(bs, num_anchor, self.num_groups, self.max_points_per_gaussian, 2)
        
        # Get window size adjusted for depth
        window_size = self._get_adapted_window_size(depth, image_wh)
        window_size = window_size.unsqueeze(-1)  # [bs, num_anchor, num_cams, 1]
        
        # Scale offsets by window size - tanh ensures values are between -1 and 1
        # then scale by window size
        offsets = torch.tanh(offsets) * window_size.view(bs, num_anchor, 1, 1, 1)
        
        # Expand points_2d to match offsets dimensions
        expanded_points = points_2d.unsqueeze(2).expand(
            -1, -1, self.num_groups, -1, -1)  # [bs, num_anchor, num_groups, num_cams, 2]
        
        # Add offsets to base points
        sampled_points = expanded_points.unsqueeze(3) + offsets.unsqueeze(3)  # [bs, num_anchor, num_groups, num_cams, max_points, 2]
        
        # Ensure sampled points are within image boundaries (0-1 normalized)
        valid_x = (sampled_points[..., 0] >= 0) & (sampled_points[..., 0] <= 1)
        valid_y = (sampled_points[..., 1] >= 0) & (sampled_points[..., 1] <= 1)
        sampling_mask = valid_x & valid_y  # [bs, num_anchor, num_groups, num_cams, max_points]
        
        return sampled_points, sampling_mask
        
    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        feature_maps: List[torch.Tensor],
        metas: dict,
        **kwargs: dict,
    ):
        # For debugging - control debug output with this flag
        debug = False
        
        bs, num_anchor = instance_feature.shape[:2]
        key_points = self.kps_generator(anchor, instance_feature)
        temp_key_points_list = feature_queue = meta_queue = temp_anchor_embeds = []
        
        if debug:
            print(f"Using DAF: {self.use_deformable_func}")
            print(f"Instance feature shape: {instance_feature.shape}")
            print(f"Key points shape: {key_points.shape}")
            print(f"Number of feature maps: {len(feature_maps)}")
            for i, fm in enumerate(feature_maps):
                print(f"Feature map {i} shape: {fm.shape}")
        
        # We now know from debug info:
        # - Each Gaussian has 13 key points
        # - There are 6 cameras
        # - There are 4 feature map levels
        # - Each feature map has shape [bs, num_cams, embed_dims, h, w]
        
        # Initialize features as None - will be set in the loop
        features = None
        
        # Make sure DAF is available before using it
        if self.use_deformable_func and DAF is not None:
            feature_maps = DAF.feature_maps_format(feature_maps)

        for (
            temp_feature_maps,
            temp_metas,
            temp_key_points,
            temp_anchor_embed,
        ) in zip(
            feature_queue[::-1] + [feature_maps],
            meta_queue[::-1] + [metas],
            temp_key_points_list[::-1] + [key_points],
            temp_anchor_embeds[::-1] + [anchor_embed],
        ):
            # Get attention weights (original weights calculation)
            if debug:
                print(f"\n----- STARTING WEIGHTS CALCULATION -----")
                
            weights, weight_mask = self._get_weights(
                instance_feature, temp_anchor_embed, metas
            )
            
            if debug:
                print(f"Initial weights shape: {weights.shape}")
                print(f"Initial weight_mask shape: {weight_mask.shape}")
            
            if self.use_deformable_func:
                if debug:
                    print(f"\n----- PROJECTING 3D POINTS TO 2D -----")
                    print(f"key_points shape: {temp_key_points.shape}")
                    print(f"projection_mat shape: {temp_metas['projection_mat'].shape}")
                    if temp_metas.get("image_wh") is not None:
                        print(f"image_wh shape: {temp_metas.get('image_wh').shape}")
                
                # Project 3D points to 2D image space
                points_2d, mask = self.project_points(
                    temp_key_points,
                    temp_metas["projection_mat"],
                    temp_metas.get("image_wh"),
                )
                
                if debug:
                    print(f"After projection, points_2d shape: {points_2d.shape}")
                    print(f"After projection, mask shape: {mask.shape}")
                
                # Get depth for window size calculation
                depth = torch.matmul(
                    temp_metas["projection_mat"][:, :, None, None], 
                    torch.cat([temp_key_points, torch.ones_like(temp_key_points[..., :1])], dim=-1)[..., None]
                ).squeeze(-1)[..., 2]  # [bs, num_cams, num_anchor, num_pts]
                depth = depth.permute(0, 2, 1, 3).mean(dim=-1)  # Average over points [bs, num_anchor, num_cams]
                
                # Convert to format expected by DAF
                if debug:
                    print(f"\n----- FORMAT CONVERSION FOR DAF -----")
                    print(f"Points_2d original shape before permute: {points_2d.shape}")
                    print(f"Mask original shape before permute: {mask.shape}")
                
                points_2d_orig = points_2d.permute(0, 2, 3, 1, 4).reshape(
                    bs, num_anchor * self.num_pts, self.num_cams, 2)
                mask = mask.permute(0, 2, 3, 1)
                
                if debug:
                    print(f"After permute, points_2d_orig shape: {points_2d_orig.shape}")
                    print(f"After permute, mask shape: {mask.shape}")
                    print(f"self.num_pts = {self.num_pts}")
                
                # --- LOCALIZATION MODIFICATION STARTS HERE ---
                
                # Generate localized sampling offsets based on window size
                # window_size needs to be scaled by image dimensions
                img_dims = temp_metas.get("image_wh").float()  # [bs, num_cams, 2]
                
                if debug:
                    print(f"Image dimensions shape: {img_dims.shape}")
                    print(f"Depth shape: {depth.shape}")
                
                # Calculate base window size - normalize by average image dimension
                base_window_size = self.window_size / torch.mean(img_dims, dim=-1, keepdim=True)  # [bs, num_cams, 1]
                
                # Reshape base_window_size to allow broadcasting with window_scale
                # [bs, num_cams, 1] -> [bs, 1, num_cams, 1] for broadcasting with [bs, num_anchor, num_cams, 1]
                base_window_size = base_window_size.permute(0, 1, 2).unsqueeze(1)  # [bs, 1, num_cams, 1]
                
                # Scale window size based on depth if enabled
                if self.use_depth_for_window:
                    # Normalize depth to range [0,1] for scaling window size
                    # depth shape: [bs, num_anchor, num_cams]
                    norm_depth = torch.clamp((depth - 1.0) / 99.0, 0.0, 1.0)
                    # Close objects (small depth) get small windows, far objects get larger windows
                    window_scale = self.min_window_ratio + norm_depth * (self.max_window_ratio - self.min_window_ratio)
                    window_scale = window_scale.unsqueeze(-1)  # [bs, num_anchor, num_cams, 1]
                else:
                    window_scale = torch.ones_like(depth).unsqueeze(-1)  # [bs, num_anchor, num_cams, 1]
                
                # Generate sample points around each Gaussian center
                # Basic approach: generate a grid of points around each projected Gaussian
                # For each Gaussian center, we'll generate points in a grid pattern
                
                # We need a perfect square number of points for our grid
                # Current debug shows we're creating a 2x2 grid = 4 points
                grid_size = 2  # This means 2x2=4 points total
                num_pts_per_gaussian = grid_size * grid_size  # Actually used points
                
                # Update instance variable to match what we're actually using
                self.max_points_per_gaussian = num_pts_per_gaussian
                
                # Convert points_2d from shape [bs, num_anchor*num_pts, num_cams, 2] to [bs, num_anchor, num_pts, num_cams, 2]
                points_2d_reshaped = points_2d_orig.reshape(bs, num_anchor, self.num_pts, self.num_cams, 2)
                
                # Take the mean location of all key points for each Gaussian to get the center
                gauss_centers = points_2d_reshaped.mean(dim=2)  # [bs, num_anchor, num_cams, 2]
                
                # Get a grid of offsets centered at zero
                offsets_y = torch.linspace(-1.0, 1.0, grid_size, device=gauss_centers.device)
                offsets_x = torch.linspace(-1.0, 1.0, grid_size, device=gauss_centers.device)
                grid_y, grid_x = torch.meshgrid(offsets_y, offsets_x, indexing='ij')
                grid_offsets = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # [grid_size^2, 2]
                
                if debug:
                    print(f"Grid offsets shape before expansion: {grid_offsets.shape}")
                
                # Scale offsets by window size and depth-adaptive factor
                # First let's reshape grid_offsets to be compatible with adaptive_window
                # [num_pts_per_gaussian, 2] → [1, 1, 1, num_pts_per_gaussian, 2]
                grid_offsets = grid_offsets.reshape(1, 1, 1, -1, 2)
                
                # Now base_window_size [bs, 1, num_cams, 1] and window_scale [bs, num_anchor, num_cams, 1]
                # They will broadcast correctly along dimension 1 (num_anchor)
                adaptive_window = base_window_size * window_scale  # [bs, num_anchor, num_cams, 1]
                
                if debug:
                    print(f"Adaptive window shape: {adaptive_window.shape}")
                    print(f"Grid offsets shape: {grid_offsets.shape}")
                
                # Reshape adaptive_window to be compatible with grid_offsets for broadcasting
                # [bs, num_anchor, num_cams, 1] → [bs, num_anchor, num_cams, 1, 1]
                adaptive_window = adaptive_window.unsqueeze(-1)  # Add extra dim for broadcasting with 2D offset coordinates
                
                # Scale the grid offsets by the adaptive window size
                # [bs, num_anchor, num_cams, 1, 1] * [1, 1, 1, num_pts_per_gaussian, 2]
                scaled_offsets = adaptive_window * grid_offsets  # Result: [bs, num_anchor, num_cams, num_pts_per_gaussian, 2]
                
                # Add offsets to Gaussian centers to get localized sampling points
                # [bs, num_anchor, num_cams, 1, 2] + [bs, num_anchor, num_cams, num_pts_per_gaussian, 2]
                localized_points = gauss_centers.unsqueeze(3) + scaled_offsets
                
                # Ensure points are within [0,1] range (valid image coordinates)
                localized_points = torch.clamp(localized_points, 0.0, 1.0)
                
                if debug:
                    print(f"Localized points shape: {localized_points.shape}")
                    print(f"num_anchor: {num_anchor}, num_pts_per_gaussian: {num_pts_per_gaussian}")
                    print(f"Total elements in tensor: {localized_points.numel()}")
                    print(f"Expected elements after reshape: {bs * num_anchor * num_pts_per_gaussian * self.num_cams * 2}")
                
                # We need to reorganize dimensions to match DAF's expected format
                # First, permute to get dimensions in the right order
                # From [bs, num_anchor, num_cams, num_pts_per_gaussian, 2] to [bs, num_anchor, num_pts_per_gaussian, num_cams, 2]
                localized_points = localized_points.permute(0, 1, 3, 2, 4)
                
                if debug:
                    print(f"After permute shape: {localized_points.shape}")
                
                # Then reshape to [bs, num_anchor*num_pts_per_gaussian, num_cams, 2]
                # We need to use the view operation which is safer than reshape for this case
                localized_points = localized_points.contiguous().view(bs, num_anchor * num_pts_per_gaussian, self.num_cams, 2)
                
                # Get a mask for valid localized points (all localized points are valid since we clamped them)
                localized_mask = torch.ones_like(localized_points[..., 0], dtype=torch.bool)
                
                if debug:
                    print(f"\n----- LOCALIZED MASK INFO -----")
                    print(f"Localized mask shape: {localized_mask.shape}")
                
                # Format mask dimensions to match weight_mask for proper filtering
                if debug:
                    print(f"\n----- DEBUGGING MASK DIMENSIONS -----")
                    print(f"Original mask shape: {mask.shape}")
                    print(f"Weight mask shape: {weight_mask.shape}")
                
                # We need to permute the mask to match the dimension ordering of weight_mask
                # Original mask: [bs, num_anchor, num_pts, num_cams]
                # Need to reorder to: [bs, num_anchor, num_cams, num_pts] first
                mask = mask.permute(0, 1, 3, 2)
                
                if debug:
                    print(f"Mask after permute: {mask.shape}")
                
                mask_expanded = mask.unsqueeze(-1).unsqueeze(-1)
                
                if debug:
                    print(f"Mask expanded after unsqueeze: {mask_expanded.shape}")
                    print(f"num_pts: {self.num_pts}")
                    print(f"num_cams: {self.num_cams}")
                    print(f"num_levels: {self.num_levels}")
                    print(f"num_groups: {self.num_groups}")
                    # Print detailed dimension info
                    for i, dim in enumerate(mask_expanded.shape):
                        print(f"Mask expanded dim {i}: {dim}")
                    for i, dim in enumerate(weight_mask.shape):
                        print(f"Weight mask dim {i}: {dim}")
                    # Print dimensions that need to match
                    print(f"Mask expanded dim [2]: {mask_expanded.shape[2]} vs Weight mask dim [2]: {weight_mask.shape[2]}")
                    print(f"Mask expanded dim [3]: {mask_expanded.shape[3]} vs Weight mask dim [3]: {weight_mask.shape[3]}")
                    
                    # Extra detailed analysis of where dimensions don't match
                    print(f"\n----- TENSOR SHAPE ANALYSIS -----")
                    print(f"Weight mask shape breakdown: {weight_mask.shape}")
                    print(f"- Dimension 0: {weight_mask.shape[0]}  (batch size)")
                    print(f"- Dimension 1: {weight_mask.shape[1]}  (num_anchor)")
                    print(f"- Dimension 2: {weight_mask.shape[2]}  (related to num_pts? = {self.num_pts})")
                    print(f"- Dimension 3: {weight_mask.shape[3]}  (related to num_cams? = {self.num_cams})")
                    print(f"- Dimension 4: {weight_mask.shape[4]}  (related to num_levels? = {self.num_levels})")
                    print(f"- Dimension 5: {weight_mask.shape[5]}  (related to num_groups? = {self.num_groups})")
                    
                    print(f"\nMask expanded shape breakdown: {mask_expanded.shape}")
                    print(f"- Dimension 0: {mask_expanded.shape[0]}  (batch size)")
                    print(f"- Dimension 1: {mask_expanded.shape[1]}  (num_anchor)")
                    print(f"- Dimension 2: {mask_expanded.shape[2]}  (num_pts != weight dimension 2)")
                    print(f"- Dimension 3: {mask_expanded.shape[3]}  (num_cams vs {weight_mask.shape[3]})")
                    print(f"- Dimension 4: {mask_expanded.shape[4]}  (unsqueezed dimension)")
                    print(f"- Dimension 5: {mask_expanded.shape[5] if len(mask_expanded.shape) > 5 else 'N/A'}  (unsqueezed dimension)")
                    
                    # Analyze dimension ordering mismatch
                    print(f"\n----- DIMENSION ORDERING ANALYSIS -----")
                    print(f"Weight mask expected pattern:  [bs, num_anchor, pts?, cams?, levels?, groups?]")
                    print(f"Mask expanded current pattern: [bs, num_anchor, pts?, cams?, 1, 1]")
                
                # Expand mask_expanded to match weight_mask dimensions
                # Note: This is where the error occurs
                # Need to permute dimensions to match weight_mask
                # Current mask_expanded: [bs, num_anchor, num_cams, num_pts, 1, 1]
                # Weight mask:          [bs, num_anchor, num_cams, num_pts_per_gaussian, num_pts, num_groups]
                # Need to add dimension for num_pts_per_gaussian between num_cams and num_pts
                
                if debug:
                    print(f"Before expansion attempt - mask_expanded: {mask_expanded.shape}, weight_mask: {weight_mask.shape}")
                
                try:
                    # First reshape to add the missing dimension for num_pts_per_gaussian
                    # We need to reshape to [bs, num_anchor, num_cams, 1, num_pts, 1]
                    mask_expanded = mask_expanded.reshape(mask_expanded.shape[0], mask_expanded.shape[1], 
                                                         mask_expanded.shape[2], 1, mask_expanded.shape[3], 1)
                    
                    if debug:
                        print(f"After reshape: {mask_expanded.shape}")
                    
                    # Now we can safely expand to match weight_mask dimensions
                    mask_expanded = mask_expanded.expand_as(weight_mask)
                    
                    if debug:
                        print(f"Successfully expanded! New shape: {mask_expanded.shape}")
                except RuntimeError as e:
                    if debug:
                        print(f"\n!!! RUNTIME ERROR CAUGHT !!!")
                        print(f"Error details: {e}")
                        print(f"Error when trying to expand: {mask_expanded.shape} to match {weight_mask.shape}")
                        # Continue raising the error
                    raise
                
                # --- MODIFY WEIGHTS FOR LOCALIZED ATTENTION ---
                
                if debug:
                    print(f"\n----- WEIGHTS PROCESSING -----")
                    print(f"Initial weights shape: {weights.shape}")
                    print(f"Mask expanded shape: {mask_expanded.shape}")
                
                # Apply mask to invalid regions and normalize
                try:
                    # 1. First make a copy of weights to preserve original for masking
                    weights_orig = weights.clone()
                    
                    # 2. Apply invalid projection mask (set to -inf)
                    weights_orig[~mask_expanded] = -torch.inf
                    
                    # 3. Calculate regions that are completely invalid (all misses)
                    all_miss = mask_expanded.sum(dim=[2, 3, 4], keepdim=True) == 0
                    
                    if debug:
                        print(f"All miss shape: {all_miss.shape}")
                    
                    # 4. Set completely invalid regions to 0
                    weights_orig[all_miss.expand_as(weights_orig)] = 0.0
                    
                    # 5. Permute weights for softmax over correct dimensions
                    # From [bs, num_anchor, num_cams, num_pts_per_gaussian, num_pts, num_groups]
                    # To [bs, num_anchor, num_pts, num_cams, num_levels, num_groups]
                    weights = weights_orig.permute(0, 1, 4, 2, 3, 5).contiguous()
                    
                    if debug:
                        print(f"After permute: {weights.shape}")
                    
                    # 6. Normalize with softmax and reshape
                    weights = weights.flatten(2, 4).softmax(dim=-2).reshape(
                        bs, num_anchor, self.num_pts, self.num_cams, self.num_levels, self.num_groups
                    )
                    
                    if debug:
                        print(f"After normalize: {weights.shape}")
                    
                    # 7. Skip the complex masking entirely
                    # Just use a direct approach without all the tensor manipulation
                    if debug:
                        print(f"Using simplified weights approach")
                        print(f"Weights shape: {weights.shape}")
                        print(f"All miss shape: {all_miss.shape}")
                    
                    # Let's be 100% clear with our tensor operations
                    # 1. Create a mask for each anchor: 1.0 if ANY valid sample, 0.0 if ALL miss
                    # First sum across all dimensions except batch and anchor to get count of valid samples
                    valid_count = mask_expanded.sum(dim=[2, 3, 4, 5])
                    
                    if debug:
                        print(f"Valid count shape: {valid_count.shape}")
                    
                    # 2. Create binary mask: 1.0 if any valid, 0.0 if all miss
                    has_valid = (valid_count > 0).float()
                    
                    if debug:
                        print(f"Has valid shape: {has_valid.shape}")
                    
                    # 3. Reshape for broadcasting with the weights tensor
                    final_mask = has_valid.view(bs, num_anchor, 1, 1, 1, 1)
                    
                    if debug:
                        print(f"Final mask shape for broadcast: {final_mask.shape}")
                    
                    # 4. Apply the mask by simple multiplication
                    weights = weights * final_mask
                    
                except Exception as e:
                    if debug:
                        print(f"ERROR in weight processing: {e}")
                        import traceback
                        traceback.print_exc()
                    raise
                
                # Apply DAF with localized points
                if DAF is not None:
                    if debug:
                        print(f"\n----- DAF APPLICATION -----")
                        print(f"Feature maps: {[fm.shape for fm in temp_feature_maps]}")
                        print(f"Localized points shape: {localized_points.shape}")
                        print(f"Weights shape before DAF: {weights.shape}")
                        print(f"Expected output dims: {bs}, {num_anchor}, {self.num_pts}, {self.embed_dims}")
                    
                    if debug:
                        debug_temp = DAF.apply(
                            *temp_feature_maps, localized_points, weights
                        )
                        print(f"DAF apply shape: {debug_temp.shape}")

                    # Apply DAF and get the output
                    daf_output = DAF.apply(*temp_feature_maps, localized_points, weights)
                    
                    # Guard against potential None result
                    if daf_output is None:
                        if debug:
                            print("WARNING: DAF.apply returned None")
                        # Create a fallback tensor with zeros
                        daf_output = torch.zeros(bs, num_anchor * num_pts_per_gaussian, self.embed_dims, 
                                              device=weights.device, dtype=weights.dtype)
                    
                    if debug:
                        print(f"DAF output shape: {daf_output.shape}")
                        print(f"num_pts_per_gaussian: {num_pts_per_gaussian}")
                    
                    # Reshape based on the actual output size and our localized points structure
                    # DAF output shape is [bs, num_anchor*num_pts_per_gaussian, embed_dims]
                    # We need to reshape to [bs, num_anchor, num_pts, embed_dims]
                    
                    # First reshape to separate the anchor and points-per-gaussian dimensions
                    temp_features = daf_output.reshape(bs, num_anchor, num_pts_per_gaussian, self.embed_dims)
                    
                    # Now expand this to match our expected num_pts by repeating the features
                    # This is needed because we sampled fewer points than original but still need to output
                    # the same shape for compatibility with the rest of the model
                    if num_pts_per_gaussian < self.num_pts:
                        if debug:
                            print(f"Expanding sampled features from {num_pts_per_gaussian} to {self.num_pts} points")
                        
                        # Create a placeholder of the right size
                        temp_features_next = torch.zeros(
                            bs, num_anchor, self.num_pts, self.embed_dims, 
                            device=temp_features.device, dtype=temp_features.dtype
                        )
                        
                        # Copy the sampled features to the first num_pts_per_gaussian positions
                        temp_features_next[:, :, :num_pts_per_gaussian, :] = temp_features
                        
                        # For any remaining positions, repeat the last feature
                        if num_pts_per_gaussian > 0 and self.num_pts > num_pts_per_gaussian:
                            temp_features_next[:, :, num_pts_per_gaussian:, :] = temp_features[:, :, -1:, :].repeat(
                                1, 1, self.num_pts - num_pts_per_gaussian, 1)
                    else:
                        # If we have enough points, just use them directly
                        temp_features_next = temp_features
                else:
                    # Fallback if DAF is not available
                    print("WARNING: DAF is None, falling back to feature_sampling")
                    temp_features_next = self.feature_sampling(
                        temp_feature_maps,
                        temp_key_points,
                        temp_metas["projection_mat"],
                        temp_metas.get("image_wh"),
                    )
                    temp_features_next = self.multi_view_level_fusion(
                        temp_features_next, weights
                    )
            else:
                # Fallback to original implementation for non-DAF mode
                temp_features_next = self.feature_sampling(
                    temp_feature_maps,
                    temp_key_points,
                    temp_metas["projection_mat"],
                    temp_metas.get("image_wh"),
                )
                temp_features_next = self.multi_view_level_fusion(
                    temp_features_next, weights
                )

            features = temp_features_next

        # Fuse multi-point features
        if features is not None:
            features = features.sum(dim=2)
        else:
            # This should not happen in normal circumstances, but handle it gracefully
            print("WARNING: features is None - using instance_feature directly")
            features = instance_feature
        output = self.proj_drop(self.output_proj(features))
        
        # Apply residual connection
        if self.residual_mode == "add":
            output = output + instance_feature
        elif self.residual_mode == "cat":
            output = torch.cat([output, instance_feature], dim=-1)
        
        return output
