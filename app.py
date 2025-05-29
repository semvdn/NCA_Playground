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
    if not nca.paused:
        nca.step()
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

@app.route('/api/randomize_all', methods=['POST'])
def randomize_all():
    global nca
    if nca is None: return jsonify({"error": "NCA not initialized"}), 500
    params = request.json
    try:
        layer_sizes = [int(x.strip()) for x in params["layer_sizes"].split(',')]
        if not layer_sizes or layer_sizes[0] != 9 or layer_sizes[-1] != 1:
             raise ValueError("Invalid layer sizes.")
    except Exception as e:
        # print(f"Error parsing layer_sizes: {e}, using default.")
        layer_sizes = [9,8,1] # Default fallback

    activation = params["activation"]
    weight_scale = float(params["weight_scale"])
    bias = float(params["bias"])
    
    # Generate a new random seed for the randomization process for both MLP and grid
    new_random_seed = np.random.randint(0, 1000000)

    nca.randomize_weights(layer_sizes, activation, weight_scale, bias, random_seed=new_random_seed)
    nca.reset_grid(random_seed=new_random_seed) # Use the same new seed for grid for consistency
    nca.paused = True # Pause after randomization
    
    return jsonify({
        "message": "NCA weights and grid randomized.",
        "grid_colors": state_to_hex_colors(nca.state),
        "mlp_params_for_viz": nca.mlp.get_params_for_viz(),
        "current_params": nca.get_current_params(),
        "is_paused": nca.paused
    })

@app.route('/api/apply_preset', methods=['POST'])
def apply_preset():
    global nca, PRESETS
    if nca is None: return jsonify({"error": "NCA not initialized"}), 500
    
    preset_name = request.json.get("preset_name")
    if preset_name not in PRESETS:
        return jsonify({"error": "Invalid preset name"}), 400

    seed, layers, act, w_scale, b_val = PRESETS[preset_name]
    
    init_params = {
        "grid_size": NCA_GRID_SIZE, # Keep current grid size
        "layer_sizes": ",".join(map(str, layers)),
        "activation": act,
        "weight_scale": w_scale,
        "bias": b_val,
        "seed": seed
    }
    initialize_nca(init_params) # This also sets paused = True

    return jsonify({
        "message": f"Preset '{preset_name}' applied.",
        "grid_colors": state_to_hex_colors(nca.state),
        "mlp_params_for_viz": nca.mlp.get_params_for_viz(),
        "current_params": nca.get_current_params(),
        "is_paused": nca.paused
    })

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
    preset_name = data.get("preset_name")
    if preset_name:
        if preset_name not in PRESETS:
            return jsonify({"error": f"Invalid preset name: {preset_name}"}), 400
        
        seed, layers, act, w_scale, b_val = PRESETS[preset_name]
        
        # Reinitialize NCA with preset parameters. This will also reset the grid.
        # Note: initialize_nca also sets nca.paused = True
        init_params = {
            "grid_size": nca.grid_size, # Keep current grid size
            "layer_sizes": ",".join(map(str, layers)),
            "activation": act,
            "weight_scale": w_scale,
            "bias": b_val,
            "seed": seed
        }
        initialize_nca(init_params)
        message = f"Settings applied: Preset '{preset_name}'."
    else:
        # Apply custom MLP settings
        try:
            layer_sizes_str = data.get("layers")
            activation = data.get("activation")
            weight_scale = data.get("weight_scale")
            bias = data.get("bias")

            # Validate and parse layer_sizes
            if layer_sizes_str is None:
                return jsonify({"error": "Missing 'layers' for custom settings."}), 400
            layer_sizes = [int(x.strip()) for x in layer_sizes_str.split(',')]
            if not layer_sizes or layer_sizes[0] != 9 or layer_sizes[-1] != 1:
                raise ValueError("Invalid layer sizes format or dimensions (must be 9, ..., 1).")

            # Validate activation
            if activation not in AVAILABLE_ACTIVATIONS:
                return jsonify({"error": f"Invalid activation: {activation}"}), 400

            # Validate weight_scale and bias
            weight_scale = float(weight_scale)
            bias = float(bias)

            # Update MLP weights. This keeps the current grid state.
            nca.randomize_weights(layer_sizes, activation, weight_scale, bias)
            nca.paused = True # Pause after changing weights
            message = "Settings applied: Custom MLP parameters."

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

@app.route('/api/randomize_weights', methods=['POST'])
def randomize_weights_route():
    global nca
    if nca is None: return jsonify({"error": "NCA not initialized"}), 500

    data = request.json
    try:
        layer_sizes_str = data.get("layers")
        activation = data.get("activation")
        weight_scale = data.get("weight_scale")
        bias = data.get("bias")

        # Validate and parse layer_sizes
        if layer_sizes_str is None:
            return jsonify({"error": "Missing 'layers'."}), 400
        layer_sizes = [int(x.strip()) for x in layer_sizes_str.split(',')]
        if not layer_sizes or layer_sizes[0] != 9 or layer_sizes[-1] != 1:
            raise ValueError("Invalid layer sizes format or dimensions (must be 9, ..., 1).")

        # Validate activation
        if activation not in AVAILABLE_ACTIVATIONS:
            return jsonify({"error": f"Invalid activation: {activation}"}), 400

        # Validate weight_scale and bias
        weight_scale = float(weight_scale)
        bias = float(bias)

        # Re-initialize MLP weights, keeping grid state.
        # A new random seed will be generated by randomize_weights if None is passed.
        nca.randomize_weights(layer_sizes, activation, weight_scale, bias)
        nca.paused = True # Pause after randomization

        return jsonify({
            "message": "NCA weights randomized.",
            "grid_colors": state_to_hex_colors(nca.state), # Grid state is preserved
            "mlp_params_for_viz": nca.mlp.get_params_for_viz(),
            "current_params": nca.get_current_params(),
            "is_paused": nca.paused
        })

    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid parameters: {e}"}), 400
    except KeyError as e:
        return jsonify({"error": f"Missing parameter: {e}"}), 400

@app.route('/api/randomize_grid', methods=['POST'])
def randomize_grid_route():
    global nca
    if nca is None: return jsonify({"error": "NCA not initialized"}), 500

    data = request.json
    seed = data.get("seed")
    
    # Validate seed
    if seed is not None:
        try:
            seed = int(seed)
        except ValueError:
            return jsonify({"error": "Invalid 'seed' parameter. Must be an integer or null."}), 400
    
    nca.reset_grid(random_seed=seed) # This method already handles None for seed
    nca.paused = True # Pause after reset

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