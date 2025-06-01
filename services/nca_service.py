# services/nca_service.py

import numpy as np
import torch
from nca_core import NeuralCellularAutomaton, DEVICE
from config import PRESETS, AVAILABLE_ACTIVATIONS, AVAILABLE_COLORMAPS, MIN_RANDOM_LAYERS, MAX_RANDOM_LAYERS, MIN_RANDOM_NODES, MAX_RANDOM_NODES, MAX_HIDDEN_LAYERS_COUNT, MIN_NODE_COUNT_PER_LAYER, MAX_NODE_COUNT_PER_LAYER
from utils.validation import validate_layer_params
from utils.visualization import state_to_hex_colors

class NCAService:
    def __init__(self):
        self.nca = None
        self.current_colormap_name = "viridis"
        self.NCA_GRID_SIZE = 50 # Default grid size

        # Initialize with a default preset on startup
        initial_preset_name = "Linear"
        initial_seed, initial_layers, initial_act, initial_w, initial_b = PRESETS[initial_preset_name]
        initial_params_for_setup = {
            "grid_size": self.NCA_GRID_SIZE,
            "layer_sizes": ",".join(map(str, initial_layers)),
            "activation": initial_act,
            "weight_scale": initial_w,
            "bias": initial_b,
            "seed": initial_seed
        }
        self.initialize_nca(initial_params_for_setup)

    def initialize_nca(self, params):
        self.NCA_GRID_SIZE = int(params.get("grid_size", 50))
        layer_sizes_str = params.get("layer_sizes", "9,8,1")
        try:
            layer_sizes = [int(x.strip()) for x in layer_sizes_str.split(',')]
            validate_layer_params(layer_sizes)
        except ValueError as e:
            print(f"Layer size validation error during init: {e}. Falling back to default.")
            layer_sizes = [9,8,1]

        activation = params.get("activation", "relu")
        if activation not in AVAILABLE_ACTIVATIONS:
            activation = "relu"

        weight_scale = float(params.get("weight_scale", 1.0))
        bias = float(params.get("bias", 0.0))
        seed_str = params.get("seed", "None")
        seed = None if str(seed_str).lower() == "none" or not seed_str else int(seed_str)

        self.nca = NeuralCellularAutomaton(
            grid_size=self.NCA_GRID_SIZE,
            layer_sizes=layer_sizes,
            activation=activation,
            weight_scale=weight_scale,
            bias=bias,
            random_seed=seed
        )

    def get_nca_config(self):
        if self.nca is None:
            raise Exception("NCA not initialized")

        return {
            "presets": PRESETS,
            "available_activations": AVAILABLE_ACTIVATIONS,
            "available_colormaps": AVAILABLE_COLORMAPS,
            "default_params": self.nca.get_current_params(),
            "current_colormap": self.current_colormap_name,
            "initial_grid_colors": state_to_hex_colors(self.nca.state, self.current_colormap_name),
            "mlp_params_for_viz": self.nca.mlp.get_params_for_viz(),
            "is_paused": self.nca.paused,
            "constraints": {
                "max_hidden_layers": MAX_HIDDEN_LAYERS_COUNT,
                "min_node_size": MIN_NODE_COUNT_PER_LAYER,
                "max_node_size": MAX_NODE_COUNT_PER_LAYER
            }
        }

    def step_nca_simulation(self):
        if self.nca is None:
            raise Exception("NCA not initialized")
        self.nca.step()
        return {
            "grid_colors": state_to_hex_colors(self.nca.state, self.current_colormap_name)
        }

    def step_back_nca_simulation(self):
        if self.nca is None:
            raise Exception("NCA not initialized")
        self.nca.step_back()
        return {
            "grid_colors": state_to_hex_colors(self.nca.state, self.current_colormap_name),
            "is_paused": self.nca.paused
        }

    def toggle_nca_pause(self):
        if self.nca is None:
            raise Exception("NCA not initialized")
        self.nca.paused = not self.nca.paused
        return {"is_paused": self.nca.paused, "message": "Toggled pause."}

    def apply_nca_settings(self, data):
        if self.nca is None:
            raise Exception("NCA not initialized")

        preset_name = data.get("preset_name")
        current_nca_params = self.nca.get_current_params()
        
        if preset_name and preset_name != "Custom":
            seed, layers_list, act, w_scale, b_val = PRESETS[preset_name]
            validate_layer_params(layers_list)
            init_params = {
                "grid_size": self.nca.grid_size,
                "layer_sizes": ",".join(map(str, layers_list)),
                "activation": act,
                "weight_scale": w_scale,
                "bias": b_val,
                "seed": seed
            }
            self.initialize_nca(init_params)
            message = f"Settings applied: Preset '{preset_name}' loaded."
        else:
            layer_sizes_str = data.get("layer_sizes", ",".join(map(str, current_nca_params["layer_sizes"])))
            layers_list = [int(x.strip()) for x in layer_sizes_str.split(',')]
            validate_layer_params(layers_list)

            activation = data.get("activation", current_nca_params["activation"])
            if activation not in AVAILABLE_ACTIVATIONS:
                raise ValueError(f"Invalid activation: {activation}")
            
            weight_scale = float(data.get("weight_scale", current_nca_params["weight_scale"]))
            bias = float(data.get("bias", current_nca_params["bias"]))

            self.nca.randomize_weights(layers_list, activation, weight_scale, bias)
            message = "Settings applied: Custom MLP parameters."
            if preset_name == "Custom" and data.get("layer_sizes") == ",".join(map(str,PRESETS["Custom"][1])):
                message = "Settings applied: 'Custom' preset parameters re-applied (weights randomized)."

        return {
            "message": message,
            "grid_colors": state_to_hex_colors(self.nca.state, self.current_colormap_name),
            "mlp_params_for_viz": self.nca.mlp.get_params_for_viz(),
            "current_params": self.nca.get_current_params(),
            "is_paused": self.nca.paused
        }

    def set_nca_colormap(self, colormap_name):
        if self.nca is None:
            raise Exception("NCA not initialized")

        if not colormap_name:
            raise ValueError("Colormap name is required")
        
        if colormap_name not in AVAILABLE_COLORMAPS:
            raise ValueError(f"Invalid colormap name: {colormap_name}")
        
        self.current_colormap_name = colormap_name
        return {
            "message": f"Colormap set to {self.current_colormap_name}.",
            "grid_colors": state_to_hex_colors(self.nca.state, self.current_colormap_name)
        }

    def randomize_nca_weights(self, data):
        if self.nca is None:
            raise Exception("NCA not initialized")
        
        current_state = self.nca.state.cpu().clone()
        was_paused = self.nca.paused
        
        current_mlp_params = self.nca.mlp
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

        self.nca = NeuralCellularAutomaton(
            grid_size=self.nca.grid_size,
            layer_sizes=layer_sizes,
            activation=activation,
            weight_scale=weight_scale,
            bias=bias,
            random_seed=None,
            initial_state=current_state
        )
        self.nca.paused = was_paused

        return {
            "message": "NCA weights randomized.",
            "grid_colors": state_to_hex_colors(self.nca.state, self.current_colormap_name),
            "mlp_params_for_viz": self.nca.mlp.get_params_for_viz(),
            "current_params": self.nca.get_current_params(),
            "is_paused": self.nca.paused
        }

    def randomize_nca_grid(self, data):
        if self.nca is None:
            raise Exception("NCA not initialized")
        seed = data.get("seed")
        if seed is not None:
            try:
                seed = int(seed)
                if not (0 <= seed < 2**32): seed = seed % (2**32)
            except ValueError: seed = None
        
        self.nca.reset_grid(random_seed=seed)
        return {
            "message": "NCA grid randomized.",
            "grid_colors": state_to_hex_colors(self.nca.state, self.current_colormap_name),
            "is_paused": self.nca.paused
        }

    def randomize_nca_architecture(self, data):
        if self.nca is None:
            raise Exception("NCA not initialized")

        was_running = data.get("was_running", False)

        num_hidden_layers = np.random.randint(MIN_RANDOM_LAYERS, MAX_RANDOM_LAYERS + 1)
        new_layer_sizes = [9]
        for _ in range(num_hidden_layers):
            new_layer_sizes.append(np.random.randint(MIN_RANDOM_NODES, MAX_RANDOM_NODES + 1))
        new_layer_sizes.append(1)

        new_activation = np.random.choice(AVAILABLE_ACTIVATIONS)

        new_weight_scale = round(np.random.uniform(0.5, 2.5), 1)
        new_bias = round(np.random.uniform(-0.5, 0.5), 1)

        init_params = {
            "grid_size": self.nca.grid_size,
            "layer_sizes": ",".join(map(str, new_layer_sizes)),
            "activation": new_activation,
            "weight_scale": new_weight_scale,
            "bias": new_bias,
            "seed": None
        }
        self.initialize_nca(init_params)

        if was_running:
            self.nca.paused = False

        return {
            "message": "NCA architecture randomized and reinitialized.",
            "grid_colors": state_to_hex_colors(self.nca.state, self.current_colormap_name),
            "mlp_params_for_viz": self.nca.mlp.get_params_for_viz(),
            "current_params": self.nca.get_current_params(),
            "is_paused": self.nca.paused
        }

    def restart_nca_simulation(self):
        if self.nca is None:
            raise Exception("NCA not initialized")

        current_params = self.nca.get_current_params()
        current_mlp_params = self.nca.mlp.get_params_for_viz()
        current_seed = current_params["initial_seed"]

        self.nca = NeuralCellularAutomaton(
            grid_size=current_params["grid_size"],
            layer_sizes=current_params["layer_sizes"],
            activation=current_params["activation"],
            weight_scale=current_params["weight_scale"],
            bias=current_params["bias"],
            random_seed=current_seed
        )
        for i, layer_weights in enumerate(current_mlp_params["weights"]):
            transposed_weights = torch.tensor(layer_weights, dtype=torch.float32).T.to(DEVICE)
            with torch.no_grad():
                self.nca.mlp.layers[i].weight.copy_(transposed_weights)

        self.nca.paused = False
        self.nca.history.clear()

        return {
            "message": "NCA reinitialized and restarted from last seed with current weights.",
            "initial_grid_colors": state_to_hex_colors(self.nca.state, self.current_colormap_name),
            "mlp_params_for_viz": self.nca.mlp.get_params_for_viz(),
            "current_params": self.nca.get_current_params(),
            "is_paused": self.nca.paused
        }

    def get_neuron_weights(self, layer_idx, neuron_idx):
        if self.nca is None:
            raise Exception("NCA not initialized")
        weights = self.nca.mlp.get_incoming_weights_for_neuron(layer_idx, neuron_idx)
        return {"weights": weights}

    def set_neuron_weights(self, data):
        if self.nca is None:
            raise Exception("NCA not initialized")
        
        layer_idx = int(data['layer_idx'])
        new_weights_pattern = data['weights_pattern']

        if data['neuron_idx'] == 'all':
            num_neurons_in_layer = self.nca.mlp.layers[layer_idx].weight.shape[0]
            for i in range(num_neurons_in_layer):
                expected_input_size = self.nca.mlp.layers[layer_idx].weight.shape[1]
                if len(new_weights_pattern) != expected_input_size:
                    raise ValueError(f"Weight pattern length mismatch for layer {layer_idx+1}. Expected {expected_input_size}, got {len(new_weights_pattern)}.")
                self.nca.mlp.set_incoming_weights_for_neuron(layer_idx, i, new_weights_pattern)
            message = f"Weight pattern applied to all neurons in Layer {layer_idx + 1}."
        else:
            neuron_idx = int(data['neuron_idx'])
            self.nca.mlp.set_incoming_weights_for_neuron(layer_idx, neuron_idx, new_weights_pattern)
            message = f"Weights updated for Layer {layer_idx + 1}, Neuron {neuron_idx + 1}."
        
        self.nca.history.clear()
        return {
            "message": message,
            "mlp_params_for_viz": self.nca.mlp.get_params_for_viz(),
            "current_params": self.nca.get_current_params(),
            "is_paused": self.nca.paused
        }

    def get_cell_details(self, r, c):
        if self.nca is None:
            raise Exception("NCA not initialized")
        
        if not (0 <= r < self.nca.grid_size and 0 <= c < self.nca.grid_size):
            raise ValueError("Row/column out of bounds")

        neighborhood = self.nca.get_neighborhood(r, c)
        layer_activations = self.nca.mlp.get_activations(neighborhood) 
        neighborhood_grid = neighborhood.reshape(3,3).tolist()

        return {
            "selected_cell": {"r": r, "c": c},
            "neighborhood": neighborhood_grid,
            "layer_activations": layer_activations
        }

    def set_nca_grid_state(self, grid_state_data, was_running):
        if self.nca is None:
            raise Exception("NCA not initialized")
        
        try:
            # Convert list of lists to a PyTorch tensor
            new_grid_state = torch.tensor(grid_state_data, dtype=torch.float32).to(DEVICE)
            # Ensure the grid state has the correct shape (grid_size, grid_size)
            if new_grid_state.shape[0] != self.nca.grid_size or new_grid_state.shape[1] != self.nca.grid_size:
                raise ValueError(f"Provided grid state dimensions ({new_grid_state.shape[0]}x{new_grid_state.shape[1]}) do not match current NCA grid size ({self.nca.grid_size}x{self.nca.grid_size}).")
            
            self.nca.state = new_grid_state.unsqueeze(0).unsqueeze(0) # Add batch and channel dimensions
            self.nca.history.clear() # Clear history as grid state has been manually set
            self.nca.paused = not was_running # Set pause state based on whether it was running
            
            return {
                "message": "Grid state updated successfully.",
                "grid_colors": state_to_hex_colors(self.nca.state, self.current_colormap_name),
                "is_paused": self.nca.paused
            }
        except Exception as e:
            raise ValueError(f"Error setting grid state: {e}")