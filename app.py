# app.py
from flask import Flask, render_template, jsonify, request
import numpy as np
from matplotlib.cm import get_cmap
import matplotlib
matplotlib.use("Agg") # Important for server-side matplotlib without GUI
import torch
from nca_core import NeuralCellularAutomaton, DEVICE # Import DEVICE from nca_core.py

app = Flask(__name__)

# --- Global NCA instance and settings ---
NCA_GRID_SIZE = 50
nca = None # Will be initialized by a call
current_colormap_name = "viridis"
colormap_func = get_cmap(current_colormap_name)

# --- Presets (seed, layers, activation, weight_scale, bias) ---
PRESETS = {
    "Linear": (None, [9, 1], "relu", 1.0, 0.0),
    "Shallow ReLU": (None, [9, 16, 1], "relu", 1.0, 0.0),
    "Deep Tanh": (None, [9, 32, 16, 1], "tanh", 1.0, 0.0),
    "Wide Sigmoid": (None, [9, 32, 1], "sigmoid", 1.0, 0.0),
    "Custom": (None, [9, 8, 1], "relu", 1.0, 0.0)
}
AVAILABLE_ACTIVATIONS = ["relu", "sigmoid", "tanh"]
AVAILABLE_COLORMAPS = ["viridis", "plasma", "magma", "cividis", "inferno", "Greys", "Blues", "GnBu", "coolwarm"]

# Constraints for hidden layers for backend validation
MAX_HIDDEN_LAYERS_COUNT = 3 # Max number of hidden layers
MIN_NODE_COUNT_PER_LAYER = 1
MAX_NODE_COUNT_PER_LAYER = 32 # Reduced for sanity, was 64

# Constraints for architecture randomization
MIN_RANDOM_LAYERS = 1
MAX_RANDOM_LAYERS = 4
MIN_RANDOM_NODES = 2
MAX_RANDOM_NODES = 10


def state_to_hex_colors(state_grid):
    """Converts the NCA state grid (floats 0-1) to hex color strings using vectorized operations."""
    global colormap_func
    
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

def validate_layer_params(layer_sizes_list):
    if not layer_sizes_list or layer_sizes_list[0] != 9 or layer_sizes_list[-1] != 1:
        raise ValueError("Layer structure must start with 9 (input) and end with 1 (output).")
    
    hidden_layers = layer_sizes_list[1:-1]
    if len(hidden_layers) > MAX_HIDDEN_LAYERS_COUNT:
        raise ValueError(f"Maximum number of hidden layers is {MAX_HIDDEN_LAYERS_COUNT}.")
    for size in hidden_layers:
        if not (MIN_NODE_COUNT_PER_LAYER <= size <= MAX_NODE_COUNT_PER_LAYER):
            raise ValueError(f"Hidden layer node count must be between {MIN_NODE_COUNT_PER_LAYER} and {MAX_NODE_COUNT_PER_LAYER}. Found: {size}")
    return True


def initialize_nca(params):
    global nca, NCA_GRID_SIZE
    NCA_GRID_SIZE = int(params.get("grid_size", 50))
    layer_sizes_str = params.get("layer_sizes", "9,8,1")
    try:
        layer_sizes = [int(x.strip()) for x in layer_sizes_str.split(',')]
        validate_layer_params(layer_sizes) # Use the new validation
    except ValueError as e:
        print(f"Layer size validation error during init: {e}. Falling back to default.")
        layer_sizes = [9,8,1] # Default fallback

    activation = params.get("activation", "relu")
    if activation not in AVAILABLE_ACTIVATIONS:
        activation = "relu"

    weight_scale = float(params.get("weight_scale", 1.0))
    bias = float(params.get("bias", 0.0))
    seed_str = params.get("seed", "None") # Handles if seed is passed as None object
    seed = None if str(seed_str).lower() == "none" or not seed_str else int(seed_str)


    nca = NeuralCellularAutomaton(
        grid_size=NCA_GRID_SIZE,
        layer_sizes=layer_sizes,
        activation=activation,
        weight_scale=weight_scale,
        bias=bias,
        random_seed=seed
    )
    # nca.paused = True # NCA starts paused by default in its __init__

# Initialize with a default preset on startup
initial_preset_name = "Linear"
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
    global nca, MAX_HIDDEN_LAYERS_COUNT, MIN_NODE_COUNT_PER_LAYER, MAX_NODE_COUNT_PER_LAYER
    if nca is None: 
        return jsonify({"error": "NCA not initialized"}), 500

    return jsonify({
        "presets": PRESETS,
        "available_activations": AVAILABLE_ACTIVATIONS,
        "available_colormaps": AVAILABLE_COLORMAPS,
        "default_params": nca.get_current_params(),
        "current_colormap": current_colormap_name,
        "initial_grid_colors": state_to_hex_colors(nca.state),
        "mlp_params_for_viz": nca.mlp.get_params_for_viz(),
        "is_paused": nca.paused,
        "constraints": {
            "max_hidden_layers": MAX_HIDDEN_LAYERS_COUNT,
            "min_node_size": MIN_NODE_COUNT_PER_LAYER,
            "max_node_size": MAX_NODE_COUNT_PER_LAYER
        }
    })

@app.route('/api/step', methods=['POST'])
def step_nca():
    global nca
    if nca is None: return jsonify({"error": "NCA not initialized"}), 500
    nca.step() # Step method in NCA now checks for pause state.
    return jsonify({
        "grid_colors": state_to_hex_colors(nca.state)
    })

@app.route('/api/step_back', methods=['POST'])
def step_back_nca():
    global nca
    if nca is None: return jsonify({"error": "NCA not initialized"}), 500
    nca.step_back()
    return jsonify({
        "grid_colors": state_to_hex_colors(nca.state),
        "is_paused": nca.paused # Step back might change pause state
    })

@app.route('/api/toggle_pause', methods=['POST'])
def toggle_pause():
    global nca
    if nca is None: return jsonify({"error": "NCA not initialized"}), 500
    nca.paused = not nca.paused
    return jsonify({"is_paused": nca.paused, "message": "Toggled pause."})


@app.route('/api/apply_settings', methods=['POST'])
def apply_settings():
    global nca, current_colormap_name, colormap_func
    if nca is None: return jsonify({"error": "NCA not initialized"}), 500

    data = request.json
    
    preset_name = data.get("preset_name")
    current_nca_params = nca.get_current_params()
    
    try:
        if preset_name and preset_name != "Custom":
            seed, layers_list, act, w_scale, b_val = PRESETS[preset_name]
            validate_layer_params(layers_list) # Validate preset layers
            init_params = {
                "grid_size": nca.grid_size,
                "layer_sizes": ",".join(map(str, layers_list)),
                "activation": act,
                "weight_scale": w_scale,
                "bias": b_val,
                "seed": seed
            }
            initialize_nca(init_params) # This resets grid and sets paused to True
            message = f"Settings applied: Preset '{preset_name}' loaded."
        else: # Custom settings or "Custom" preset selected
            layer_sizes_str = data.get("layer_sizes", ",".join(map(str, current_nca_params["layer_sizes"])))
            layers_list = [int(x.strip()) for x in layer_sizes_str.split(',')]
            validate_layer_params(layers_list) # Validate custom layers

            activation = data.get("activation", current_nca_params["activation"])
            if activation not in AVAILABLE_ACTIVATIONS:
                raise ValueError(f"Invalid activation: {activation}")
            
            weight_scale = float(data.get("weight_scale", current_nca_params["weight_scale"]))
            bias = float(data.get("bias", current_nca_params["bias"]))

            # Rebuild MLP with new parameters, keeping current grid state if layers are same
            # If layer structure changes, it's effectively a new network, so randomize_weights is appropriate.
            # If only activation/scale/bias change for the *same* structure, we might want a different method,
            # but randomize_weights with a new seed will also work.
            # For simplicity, we'll use randomize_weights. A new seed will be generated.
            nca.randomize_weights(layers_list, activation, weight_scale, bias)
            message = "Settings applied: Custom MLP parameters."
            if preset_name == "Custom" and data.get("layer_sizes") == ",".join(map(str,PRESETS["Custom"][1])):
                message = "Settings applied: 'Custom' preset parameters re-applied (weights randomized)."

    except ValueError as e:
        return jsonify({"error": f"Invalid parameters: {e}"}), 400
    except KeyError as e:
        return jsonify({"error": f"Missing parameter for custom settings: {e}"}), 400

    return jsonify({
        "message": message,
        "grid_colors": state_to_hex_colors(nca.state),
        "mlp_params_for_viz": nca.mlp.get_params_for_viz(),
        "current_params": nca.get_current_params(),
        "is_paused": nca.paused
    })

@app.route('/api/set_colormap', methods=['POST'])
def set_colormap_route():
    global nca, current_colormap_name, colormap_func
    if nca is None: return jsonify({"error": "NCA not initialized"}), 500

    data = request.json
    colormap_name = data.get("colormap_name")

    if not colormap_name:
        return jsonify({"error": "Colormap name is required"}), 400
    
    if colormap_name not in AVAILABLE_COLORMAPS:
        return jsonify({"error": f"Invalid colormap name: {colormap_name}"}), 400
    
    current_colormap_name = colormap_name
    colormap_func = get_cmap(current_colormap_name)

    return jsonify({
        "message": f"Colormap set to {current_colormap_name}.",
        "grid_colors": state_to_hex_colors(nca.state)
    })

@app.route('/api/randomize_weights', methods=['POST'])
def randomize_weights_route():
    global nca
    if nca is None: return jsonify({"error": "NCA not initialized"}), 500
    
    # Store current state and pause status
    current_state = nca.state.cpu().clone()
    was_paused = nca.paused
    
    data = request.json
    try:
        current_mlp_params = nca.mlp
        layer_sizes_str = data.get("layer_sizes")
        if layer_sizes_str:
            layer_sizes = [int(x.strip()) for x in layer_sizes_str.split(',')]
            validate_layer_params(layer_sizes)
        else:
            layer_sizes = current_mlp_params.layer_sizes

        activation = data.get("activation", current_mlp_params.activation_name)
        if activation not in AVAILABLE_ACTIVATIONS:
            raise ValueError(f"Invalid activation: {activation}")

        weight_scale = float(data.get("weight_scale", current_mlp_params.weight_scale))
        bias = float(data.get("bias", current_mlp_params.bias_value))

        # Reinitialize NCA with new weights but preserve the grid state
        # This will also reset history, which is desired as the network changed
        nca = NeuralCellularAutomaton(
            grid_size=nca.grid_size,
            layer_sizes=layer_sizes,
            activation=activation,
            weight_scale=weight_scale,
            bias=bias,
            random_seed=None, # Generate new random weights
            initial_state=current_state # Reinitialize with the last frame
        )
        nca.paused = was_paused # Restore original pause state

        return jsonify({
            "message": "NCA weights randomized.",
            "grid_colors": state_to_hex_colors(nca.state), # Send updated grid colors
            "mlp_params_for_viz": nca.mlp.get_params_for_viz(),
            "current_params": nca.get_current_params(),
            "is_paused": nca.paused
        })
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid parameters for randomizing weights: {e}"}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500

@app.route('/api/randomize_grid', methods=['POST'])
def randomize_grid_route():
    global nca
    if nca is None: return jsonify({"error": "NCA not initialized"}), 500
    data = request.json
    seed = data.get("seed")
    if seed is not None:
        try:
            seed = int(seed)
            if not (0 <= seed < 2**32): seed = seed % (2**32)
        except ValueError: seed = None
    
    nca.reset_grid(random_seed=seed)
    return jsonify({
        "message": "NCA grid randomized.",
        "grid_colors": state_to_hex_colors(nca.state),
        "is_paused": nca.paused
    })

@app.route('/api/randomize_architecture', methods=['POST'])
def randomize_architecture_route():
    global nca
    if nca is None: return jsonify({"error": "NCA not initialized"}), 500

    data = request.json
    was_running = data.get("was_running", False) # Get the running state from frontend

    try:
        num_hidden_layers = np.random.randint(MIN_RANDOM_LAYERS, MAX_RANDOM_LAYERS + 1)
        new_layer_sizes = [9] # Input layer
        for _ in range(num_hidden_layers):
            new_layer_sizes.append(np.random.randint(MIN_RANDOM_NODES, MAX_RANDOM_NODES + 1))
        new_layer_sizes.append(1) # Output layer

        # Select a random activation function
        new_activation = np.random.choice(AVAILABLE_ACTIVATIONS)

        # Randomize weight scale and bias within reasonable bounds
        new_weight_scale = round(np.random.uniform(0.5, 2.5), 1)
        new_bias = round(np.random.uniform(-0.5, 0.5), 1)

        # Reinitialize NCA with new architecture and new random weights
        init_params = {
            "grid_size": nca.grid_size,
            "layer_sizes": ",".join(map(str, new_layer_sizes)),
            "activation": new_activation,
            "weight_scale": new_weight_scale,
            "bias": new_bias,
            "seed": None # Generate new random weights
        }
        initialize_nca(init_params) # This resets grid and sets paused to True

        # If it was running before, set it to not paused
        if was_running:
            nca.paused = False

        return jsonify({
            "message": "NCA architecture randomized and reinitialized.",
            "grid_colors": state_to_hex_colors(nca.state),
            "mlp_params_for_viz": nca.mlp.get_params_for_viz(),
            "current_params": nca.get_current_params(),
            "is_paused": nca.paused
        })
    except Exception as e:
        return jsonify({"error": f"Failed to randomize architecture: {e}"}), 500


@app.route('/api/restart', methods=['POST'])
def restart_nca():
    global nca, current_colormap_name, colormap_func
    if nca is None: return jsonify({"error": "NCA not initialized"}), 500

    # Get current parameters and seed to reinitialize with
    current_params = nca.get_current_params()
    current_seed = current_params["initial_seed"] # Store the last used seed

    # Prepare parameters for reinitialization
    init_params = {
        "grid_size": current_params["grid_size"],
        "layer_sizes": ",".join(map(str, current_params["layer_sizes"])),
        "activation": current_params["activation"],
        "weight_scale": current_params["weight_scale"],
        "bias": current_params["bias"],
        "seed": current_seed # Use the last seed
    }

    initialize_nca(init_params) # This reinitializes NCA
    nca.paused = False # Ensure it starts running after restart

    return jsonify({
        "message": "NCA reinitialized and restarted from last seed.",
        "initial_grid_colors": state_to_hex_colors(nca.state),
        "mlp_params_for_viz": nca.mlp.get_params_for_viz(),
        "current_params": nca.get_current_params(),
        "is_paused": nca.paused # Should be False now
    })

@app.route('/api/neuron_weights', methods=['GET', 'POST'])
def neuron_weights_route():
    global nca
    if nca is None: return jsonify({"error": "NCA not initialized"}), 500

    if request.method == 'GET':
        try:
            layer_idx = int(request.args.get('layer_idx')) # 0-indexed for MLP's W array
            neuron_idx = int(request.args.get('neuron_idx'))
            
            weights = nca.mlp.get_incoming_weights_for_neuron(layer_idx, neuron_idx)
            return jsonify({"weights": weights})
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    
    if request.method == 'POST':
        try:
            data = request.json
            layer_idx = int(data['layer_idx']) # 0-indexed for MLP's W array
            new_weights_pattern = data['weights_pattern'] # This will be a list of floats

            # Handle "All Neurons" case
            if data['neuron_idx'] == 'all':
                num_neurons_in_layer = nca.mlp.layers[layer_idx].weight.shape[0] # out_dim
                for i in range(num_neurons_in_layer):
                    # Ensure the pattern matches the expected input size for this layer
                    expected_input_size = nca.mlp.layers[layer_idx].weight.shape[1] # in_dim
                    if len(new_weights_pattern) != expected_input_size:
                        return jsonify({"error": f"Weight pattern length mismatch for layer {layer_idx+1}. Expected {expected_input_size}, got {len(new_weights_pattern)}."}),400
                    nca.mlp.set_incoming_weights_for_neuron(layer_idx, i, new_weights_pattern)
                message = f"Weight pattern applied to all neurons in Layer {layer_idx + 1}."
            else: # Single neuron case
                neuron_idx = int(data['neuron_idx'])
                nca.mlp.set_incoming_weights_for_neuron(layer_idx, neuron_idx, new_weights_pattern)
                message = f"Weights updated for Layer {layer_idx + 1}, Neuron {neuron_idx + 1}."
            
            nca.history.clear() # Clear history as network changed significantly
            return jsonify({
                "message": message,
                "mlp_params_for_viz": nca.mlp.get_params_for_viz(), # Send updated weights for viz
                "current_params": nca.get_current_params(), # For consistency
                "is_paused": nca.paused
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 400


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
    layer_activations = nca.mlp.get_activations(neighborhood) 
    neighborhood_grid = neighborhood.reshape(3,3).tolist()

    return jsonify({
        "selected_cell": {"r": r, "c": c},
        "neighborhood": neighborhood_grid,
        "layer_activations": layer_activations
    })


if __name__ == '__main__':
    app.run(debug=True)