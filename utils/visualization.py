# utils/visualization.py

import numpy as np
from matplotlib.cm import get_cmap
import matplotlib
matplotlib.use("Agg") # Important for server-side matplotlib without GUI
import torch

def state_to_hex_colors(state_grid, colormap_name="viridis"):
    """Converts the NCA state grid (floats 0-1) to hex color strings using vectorized operations."""
    colormap_func = get_cmap(colormap_name)
    
    # Ensure state_grid is a numpy array for colormap_func
    if torch.is_tensor(state_grid):
        state_grid_np = state_grid.cpu().detach().numpy()
    else:
        state_grid_np = state_grid

    # Normalize values to [0, 1] and apply colormap
    normalized_grid = np.clip(state_grid_np, 0., 1.)
    rgba_colors = colormap_func(normalized_grid) # This returns (H, W, 4) array of floats

    # Convert RGBA floats (0-1) to byte integers (0-255)
    byte_colors = (rgba_colors[:, :, :3] * 255).astype(np.uint8) # Take only RGB channels

    # Format to hex strings. This part still involves iteration, but it's on pre-processed data.
    # A more advanced approach might involve a custom C/Cython extension or WebGL for frontend rendering.
    # For now, this is a significant improvement over per-pixel colormap application.
    hex_colors = []
    for r in range(byte_colors.shape[0]):
        row_colors = []
        for c in range(byte_colors.shape[1]):
            r_byte, g_byte, b_byte = byte_colors[r, c]
            row_colors.append(f"#{r_byte:02x}{g_byte:02x}{b_byte:02x}")
        hex_colors.append(row_colors)
    return hex_colors