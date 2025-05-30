# app.py
from flask import Flask, render_template, jsonify, request
import numpy as np
from matplotlib.cm import get_cmap
import matplotlib
matplotlib.use("Agg") # Important for server-side matplotlib without GUI

from nca_core import NeuralCellularAutomaton # Import from your nca_core.py

app = Flask(__name__)

# --- Global NCA instance and settings ---
NCA_GRID_SIZE = 50
nca = None # Will be initialized by a call
current_colormap_name = "viridis"
colormap_func = get_cmap(current_colormap_name)

# --- Presets (seed, layers, activation, weight_scale, bias) ---
PRESETS = {
    "Flicker":  (42,   [9,8,1],    "relu",    1.0,  0.0),
    "Ripples":  (123,  [9,16,1],   "tanh",    2.0,  0.0),
    "Bubbles":  (999,  [9,8,8,1],  "sigmoid", 1.5,  0.5),
    "Patchy":   (555,  [9,8,1],    "relu",    0.5, -0.3),
    "Custom":   (None, [9,8,1],    "relu",    1.0,  0.0) # For user settings
}
AVAILABLE_ACTIVATIONS = ["relu", "sigmoid", "tanh"]
AVAILABLE_COLORMAPS = ["viridis", "plasma", "magma", "cividis", "inferno", "Greys", "Blues", "GnBu", "coolwarm"]

# New: Preset and Parametric Weight Patterns
AVAILABLE_PRESET_PATTERNS = [
    "Identity Pass-through",
    "Horizontal Edge Detector",
    "Vertical Edge Detector",
    "Blur",
    "Concentric"
]

AVAILABLE_PARAMETRIC_PATTERNS = [
    "Gaussian",
    "Laplacian",
    "Directional"
]

# Metadata for parametric patterns (name, type, default, min, max, step)
PARAMETRIC_PATTERNS_META = {
    "Gaussian": [
        {"name": "sigma", "type": "float", "default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}
    ],
    "Laplacian": [
        {"name": "strength", "type": "float", "default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}
    ],
    "Directional": [
        {"name": "angle", "type": "float", "default": 0.0, "min": 0.0, "max": 360.0, "step": 1.0},
        {"name": "magnitude", "type": "float", "default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}
    ]
}

# Constraints for hidden layers
MAX_HIDDEN_LAYERS = 3
MIN_NODE_SIZE = 1
MAX_NODE_SIZE = 64

def state_to_hex_colors(state_grid):
    """Converts the NCA state grid (floats 0-1) to hex color strings."""
    global colormap_func
    hex_colors = []
    for r in range(state_grid.shape[0]):
        row_colors = []
        for c in range(state_grid.shape[1]):
            val = max(0., min(state_grid[r, c], 1.))
            rgba = colormap_func(val)
            r_byte = int(rgba[0] * 255)
            g_byte = int(rgba[1] * 255)
            b_byte = int(rgba[2] * 255)
            row_colors.append(f"#{r_byte:02x}{g_byte:02x}{b_byte:02x}")
        hex_colors.append(row_colors)
    return hex_colors

def initialize_nca(params):
    global nca, NCA_GRID_SIZE
    NCA_GRID_SIZE = int(params.get("grid_size", 50))
    layer_sizes_str = params.get("layer_sizes", "9,8,1")
    try:
        layer_sizes = [int(x.strip()) for x in layer_sizes_str.split(',')]
        if not layer_sizes or layer_sizes[0] != 9 or layer_sizes[-1] != 1: # Basic validation
            raise ValueError("Invalid layer sizes.")
    except:
        layer_sizes = [9,8,1] # Default fallback

    activation = params.get("activation", "relu")
    weight_scale = float(params.get("weight_scale", 1.0))
    bias = float(params.get("bias", 0.0))
    seed_str = params.get("seed", "None")
    seed = None if seed_str == "None" or not seed_str else int(seed_str)

    nca = NeuralCellularAutomaton(
        grid_size=NCA_GRID_SIZE,
        layer_sizes=layer_sizes,
        activation=activation,
        weight_scale=weight_scale,
        bias=bias,
        random_seed=seed
    )
    nca.paused = True # Start paused

# Initialize with a default preset on startup
# This ensures 'nca' is not None when the first API call might arrive
initial_preset_name = "Flicker"
initial_seed, initial_layers, initial_act, initial_w, initial_b = PRESETS[initial_preset_name]
initial_params_for_setup = {
    "grid_size": NCA_GRID_SIZE,
    "layer_sizes": ",".join(map(str, initial_layers)),
    "activation": initial_act,
    "weight_scale": initial_w,
    "bias": initial_b,
    "seed": initial_seed
}
initialize_nca(initial_params_for_setup)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/config', methods=['GET'])
def get_config():
    global nca
    if nca is None: # Should not happen due to startup init
        return jsonify({"error": "NCA not initialized"}), 500

    return jsonify({
        "presets": PRESETS,
        "available_activations": AVAILABLE_ACTIVATIONS,
        "available_colormaps": AVAILABLE_COLORMAPS,
        "available_preset_patterns": AVAILABLE_PRESET_PATTERNS, # New
        "available_parametric_patterns": AVAILABLE_PARAMETRIC_PATTERNS, # New
        "parametric_patterns_meta": PARAMETRIC_PATTERNS_META, # New
        "default_params": nca.get_current_params(),
        "current_colormap": current_colormap_name,
        "initial_grid_colors": state_to_hex_colors(nca.state),
        "mlp_params_for_viz": nca.mlp.get_params_for_viz(),
        "is_paused": nca.paused
    })

@app.route('/api/reinit', methods=['POST'])
def reinit_nca_route():
    global nca
    params = request.json
    initialize_nca(params)
    return jsonify({
        "message": "NCA re-initialized.",
        "grid_colors": state_to_hex_colors(nca.state),
        "mlp_params_for_viz": nca.mlp.get_params_for_viz(),
        "current_params": nca.get_current_params(),
        "is_paused": nca.paused
    })


@app.route('/api/step', methods=['POST'])
def step_nca():
    global nca
    if nca is None: return jsonify({"error": "NCA not initialized"}), 500
    nca.step()
    return jsonify({
        "grid_colors": state_to_hex_colors(nca.state)
    })

@app.route('/api/step_back', methods=['POST'])
def step_back_nca():
    global nca
    if nca is None: return jsonify({"error": "NCA not initialized"}), 500
    nca.step_back()
    return jsonify({
        "grid_colors": state_to_hex_colors(nca.state)
    })

@app.route('/api/toggle_pause', methods=['POST'])
def toggle_pause():
    global nca
    if nca is None: return jsonify({"error": "NCA not initialized"}), 500
    nca.paused = not nca.paused
    return jsonify({"is_paused": nca.paused, "message": "Toggled pause."})

@app.route('/api/reset_grid', methods=['POST'])
def reset_grid():
    global nca
    if nca is None: return jsonify({"error": "NCA not initialized"}), 500
    data = request.json
    seed = data.get("seed")
    if seed is not None:
        try:
            seed = int(seed)
        except ValueError:
            seed = None # Or handle error appropriately
            
    nca.reset_grid(random_seed=seed)
    nca.paused = True # Pause after reset
    return jsonify({
        "grid_colors": state_to_hex_colors(nca.state),
        "is_paused": nca.paused,
        "message": "Grid reset."
    })

@app.route('/api/randomize_weights', methods=['POST'])
def randomize_weights_route():
    global nca
    if nca is None: return jsonify({"error": "NCA not initialized"}), 500

    data = request.json
    try:
        weight_scale = float(data.get("weight_scale", nca.mlp.weight_scale))
        bias = float(data.get("bias", nca.mlp.bias_value))
        random_seed = data.get("random_seed")
        if random_seed is not None:
            try:
                random_seed = int(random_seed)
            except ValueError:
                random_seed = None # Fallback if not a valid int

        # Re-initialize MLP weights, keeping grid state.
        # Use current layer sizes and activation, only randomize weights.
        nca.randomize_weights(
            layer_sizes=nca.mlp.layer_sizes, # Keep current architecture
            activation=nca.mlp.activation_name, # Keep current activation
            weight_scale=weight_scale,
            bias=bias,
            random_seed=random_seed
        )

        return jsonify({
            "message": "NCA weights randomized.",
            "mlp_params_for_viz": nca.mlp.get_params_for_viz(),
            "current_params": nca.get_current_params(),
            "is_paused": nca.paused
        })

    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid parameters: {e}"}), 400
    except KeyError as e:
        return jsonify({"error": f"Missing parameter: {e}"}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500

@app.route('/api/set_preset_weights', methods=['POST'])
def set_preset_weights_route():
    global nca
    if nca is None: return jsonify({"error": "NCA not initialized"}), 500

    data = request.json
    preset_name = data.get("preset_name")
    if preset_name not in AVAILABLE_PRESET_PATTERNS:
        return jsonify({"error": "Invalid preset pattern name"}), 400

    try:
        nca.mlp.set_weights_from_preset(preset_name)
        return jsonify({
            "message": f"Preset '{preset_name}' weights applied.",
            "mlp_params_for_viz": nca.mlp.get_params_for_viz(),
            "current_params": nca.get_current_params(),
            "is_paused": nca.paused
        })
    except Exception as e:
        return jsonify({"error": f"Failed to apply preset weights: {e}"}), 500

@app.route('/api/set_parametric_weights', methods=['POST'])
def set_parametric_weights_route():
    global nca
    if nca is None: return jsonify({"error": "NCA not initialized"}), 500

    data = request.json
    pattern_type = data.get("pattern_type")
    parameters = data.get("parameters", {})

    if pattern_type not in AVAILABLE_PARAMETRIC_PATTERNS:
        return jsonify({"error": "Invalid parametric pattern type"}), 400

    try:
        # This method will need to be implemented in nca_core.py
        nca.mlp.generate_parametric_weights(pattern_type, parameters)
        return jsonify({
            "message": f"Parametric '{pattern_type}' weights generated and applied.",
            "mlp_params_for_viz": nca.mlp.get_params_for_viz(),
            "current_params": nca.get_current_params(),
            "is_paused": nca.paused
        })
    except Exception as e:
        return jsonify({"error": f"Failed to generate parametric weights: {e}"}), 500

@app.route('/api/set_direct_weights', methods=['POST'])
def set_direct_weights_route():
    global nca
    if nca is None: return jsonify({"error": "NCA not initialized"}), 500

    data = request.json
    weights_matrix = data.get("weights_matrix")

    if not isinstance(weights_matrix, list) or len(weights_matrix) != 3 or \
       not all(isinstance(row, list) and len(row) == 3 for row in weights_matrix):
        return jsonify({"error": "Invalid weights_matrix format. Must be a 3x3 array."}), 400

    try:
        # This method will need to be implemented in nca_core.py
        nca.mlp.set_first_layer_weights(np.array(weights_matrix))
        return jsonify({
            "message": "Direct weights applied.",
            "mlp_params_for_viz": nca.mlp.get_params_for_viz(),
            "current_params": nca.get_current_params(),
            "is_paused": nca.paused
        })
    except Exception as e:
        return jsonify({"error": f"Failed to apply direct weights: {e}"}), 500

@app.route('/api/upload_weights', methods=['POST'])
def upload_weights_route():
    global nca
    if nca is None: return jsonify({"error": "NCA not initialized"}), 500

    data = request.json
    file_content = data.get("file_content")
    filename = data.get("filename")

    if not file_content:
        return jsonify({"error": "No file content provided."}), 400

    try:
        # This method will need to be implemented in nca_core.py
        nca.mlp.load_weights_from_file(file_content, filename)
        return jsonify({
            "message": f"Weights from '{filename}' uploaded successfully.",
            "mlp_params_for_viz": nca.mlp.get_params_for_viz(),
            "current_params": nca.get_current_params(),
            "is_paused": nca.paused
        })
    except Exception as e:
        return jsonify({"error": f"Failed to upload weights: {e}"}), 500

@app.route('/api/download_weights', methods=['GET'])
def download_weights_route():
    global nca
    if nca is None: return jsonify({"error": "NCA not initialized"}), 500

    try:
        # This method will need to be implemented in nca_core.py
        file_content, filename = nca.mlp.get_weights_for_export()
        return jsonify({
            "file_content": file_content,
            "filename": filename
        })
    except Exception as e:
        return jsonify({"error": f"Failed to prepare weights for download: {e}"}), 500


@app.route('/api/set_colormap', methods=['POST'])
def set_colormap_route():
    global current_colormap_name, colormap_func, nca
    if nca is None: return jsonify({"error": "NCA not initialized"}), 500

    data = request.json
    cmap_name = data.get("colormap_name")
    if cmap_name not in AVAILABLE_COLORMAPS:
        return jsonify({"error": "Invalid colormap name"}), 400
    
    current_colormap_name = cmap_name
    colormap_func = get_cmap(current_colormap_name)
    
    return jsonify({
        "message": f"Colormap set to {cmap_name}.",
        "grid_colors": state_to_hex_colors(nca.state) # Re-send grid with new colors
    })

@app.route('/api/apply_settings', methods=['POST'])
def apply_settings():
    global nca, current_colormap_name, colormap_func
    if nca is None: return jsonify({"error": "NCA not initialized"}), 500

    data = request.json
    
    # 1. Handle colormap
    colormap_name = data.get("colormap_name")
    if colormap_name:
        if colormap_name not in AVAILABLE_COLORMAPS:
            return jsonify({"error": f"Invalid colormap name: {colormap_name}"}), 400
        current_colormap_name = colormap_name
        colormap_func = get_cmap(current_colormap_name)

    # 2. Handle preset or custom MLP settings
    # 2. Handle preset or custom MLP settings
    # The 'apply_settings' route is now primarily for architecture and colormap.
    # Weight initialization is handled by dedicated routes.
    try:
        layer_sizes_str = data.get("layer_sizes")
        activation = data.get("activation")
        
        # Validate and parse layer_sizes
        if layer_sizes_str is None:
            return jsonify({"error": "Missing 'layer_sizes' for custom settings."}), 400
        layer_sizes = [int(x.strip()) for x in layer_sizes_str.split(',')]
        if not layer_sizes or layer_sizes[0] != 9 or layer_sizes[-1] != 1:
            raise ValueError("Invalid layer sizes format or dimensions (must be 9, ..., 1).")
        
        # Additional backend validation for hidden layers
        if len(layer_sizes) - 2 > MAX_HIDDEN_LAYERS:
            raise ValueError(f"Too many hidden layers. Maximum allowed: {MAX_HIDDEN_LAYERS}.")
        for i in range(1, len(layer_sizes) - 1):
            if not (MIN_NODE_SIZE <= layer_sizes[i] <= MAX_NODE_SIZE):
                raise ValueError(f"Hidden layer size {layer_sizes[i]} is out of bounds ({MIN_NODE_SIZE}-{MAX_NODE_SIZE}).")

        # Validate activation
        if activation not in AVAILABLE_ACTIVATIONS:
            return jsonify({"error": f"Invalid activation: {activation}"}), 400

        # Update MLP architecture and activation. Weights are NOT randomized here.
        nca.mlp.set_architecture(layer_sizes, activation)
        message = "Settings applied: Custom MLP architecture and activation."

    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid custom settings parameters: {e}"}), 400
    except KeyError as e:
        return jsonify({"error": f"Missing parameter for custom settings: {e}"}), 400

    # Prepare response
    response_data = {
        "message": message,
        "grid_colors": state_to_hex_colors(nca.state),
        "mlp_params_for_viz": nca.mlp.get_params_for_viz(),
        "current_params": nca.get_current_params(),
        "is_paused": nca.paused
    }
    return jsonify(response_data)


@app.route('/api/randomize_grid', methods=['POST'])
def randomize_grid_route():
    global nca
    if nca is None: return jsonify({"error": "NCA not initialized"}), 500

    print(f"DEBUG: /api/randomize_grid received request.data: {request.data}")
    print(f"DEBUG: /api/randomize_grid received request.json: {request.json}")
    
    try:
        data = request.json
    except Exception as e:
        print(f"ERROR: /api/randomize_grid JSON parsing failed: {e}")
        return jsonify({"error": f"Invalid JSON in request body: {e}"}), 400

    seed = data.get("seed")
    
    # Validate and normalize seed
    if seed is not None:
        try:
            seed = int(seed)
            # Ensure seed is within numpy's valid range [0, 2**32 - 1]
            if not (0 <= seed < 2**32): # 2**32 is the exclusive upper bound for numpy.random.seed
                print(f"WARNING: Seed {seed} is out of numpy's valid range. Normalizing.")
                seed = seed % (2**32) # Use modulo to bring it into range
        except ValueError:
            print(f"ERROR: Invalid 'seed' parameter received: {seed}. Using None.")
            seed = None # Fallback to None if conversion fails
    
    nca.reset_grid(random_seed=seed) # This method already handles None for seed
    # nca.paused = True # Removed: Do not force pause after reset

    return jsonify({
        "message": "NCA grid randomized.",
        "grid_colors": state_to_hex_colors(nca.state),
        "is_paused": nca.paused
    })

@app.route('/api/cell_details', methods=['GET'])
def get_cell_details():
    global nca
    if nca is None: return jsonify({"error": "NCA not initialized"}), 500
    try:
        r = int(request.args.get('r'))
        c = int(request.args.get('c'))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid row/column parameters"}), 400

    if not (0 <= r < nca.grid_size and 0 <= c < nca.grid_size):
        return jsonify({"error": "Row/column out of bounds"}), 400

    neighborhood = nca.get_neighborhood(r, c)
    layer_activations = nca.mlp.get_activations(neighborhood) # Get all layer activations

    # Prepare neighborhood for display (3x3)
    neighborhood_grid = neighborhood.reshape(3,3).tolist()

    return jsonify({
        "selected_cell": {"r": r, "c": c},
        "neighborhood": neighborhood_grid,
        "layer_activations": layer_activations # This now includes input layer
        # MLP params (weights, layer_sizes) are fetched separately or on init
    })


if __name__ == '__main__':
    app.run(debug=True)