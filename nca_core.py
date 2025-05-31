# nca_core.py
import numpy as np
from collections import deque

# ------------------------------------------------------------------------------------
# PART 1: Flexible Multi-Layer Neural Network Definition
# ------------------------------------------------------------------------------------
def get_activation_func(name):
    """Return a Python function f(x) for the chosen activation."""
    name = name.lower()
    if name == "relu":
        return lambda x: np.maximum(x, 0.0)
    elif name == "sigmoid":
        return lambda x: 1.0 / (1.0 + np.exp(-x))
    elif name == "tanh":
        return lambda x: np.tanh(x)
    else:
        # Default to ReLU if invalid
        return lambda x: np.maximum(x, 0.0)

class FlexibleMLP:
    def __init__(self, layer_sizes, activation="relu", weight_scale=1.0,
                 bias=0.0, random_seed=None):
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1
        self.activation_name = activation
        self.activation = get_activation_func(activation)
        self.weight_scale = weight_scale
        self.bias_value = bias
        if random_seed is not None:
            np.random.seed(random_seed)

        self.W = []
        self.b = []
        for i in range(self.n_layers):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i+1]
            W_i = weight_scale * np.random.randn(in_dim, out_dim)
            b_i = bias * np.random.randn(out_dim) # Can also be self.bias_value * np.ones(out_dim) for constant bias
            self.W.append(W_i)
            self.b.append(b_i)

    def forward(self, x):
        h = x
        for i in range(self.n_layers):
            z = np.dot(h, self.W[i]) + self.b[i]
            if i < (self.n_layers - 1):
                h = self.activation(z)
            else:
                if self.layer_sizes[-1] == 1: # Output layer for NCA state
                    z = 1.0 / (1.0 + np.exp(-z))
                h = z
        return h

    def set_params(self, layer_sizes, activation, weight_scale, bias, random_seed=None):
        self.__init__(layer_sizes, activation, weight_scale, bias, random_seed)

    def get_params_for_viz(self):
        return {
            "layer_sizes": self.layer_sizes,
            "weights": [w.tolist() for w in self.W]
        }

    def get_activations(self, x):
        layer_acts = [x.tolist()]
        h = x
        for i in range(self.n_layers):
            z = np.dot(h, self.W[i]) + self.b[i]
            if i < (self.n_layers - 1):
                h = self.activation(z)
            else:
                if self.layer_sizes[-1] == 1:
                    z = 1.0 / (1.0 + np.exp(-z))
                h = z
            layer_acts.append(h.tolist() if isinstance(h, np.ndarray) else [h])
        return layer_acts

    def get_incoming_weights_for_neuron(self, layer_idx, neuron_idx):
        """
        Get the incoming weights for a specific neuron.
        layer_idx: Index of the layer the neuron belongs to (0 for first hidden layer, etc., corresponding to W[layer_idx]).
        neuron_idx: Index of the neuron within that layer.
        """
        if not (0 <= layer_idx < self.n_layers):
            raise ValueError(f"Invalid layer index: {layer_idx}")
        if not (0 <= neuron_idx < self.W[layer_idx].shape[1]): # W[i] is (prev_layer_size, current_layer_size)
            raise ValueError(f"Invalid neuron index: {neuron_idx} for layer {layer_idx+1} with size {self.W[layer_idx].shape[1]}")
        
        # Weights coming into neuron `neuron_idx` in layer `layer_idx+1` (target layer)
        # are in W[layer_idx][:, neuron_idx]
        return self.W[layer_idx][:, neuron_idx].tolist()


    def set_incoming_weights_for_neuron(self, layer_idx, neuron_idx, new_weights):
        """
        Set the incoming weights for a specific neuron.
        layer_idx: Index of the layer the neuron belongs to (0 for first hidden layer, etc.).
        neuron_idx: Index of the neuron within that layer.
        new_weights: A list or 1D NumPy array of new weights.
        """
        if not (0 <= layer_idx < self.n_layers):
            raise ValueError("Invalid layer index for setting weights.")
        
        target_weights_shape = self.W[layer_idx][:, neuron_idx].shape
        num_expected_weights = self.W[layer_idx].shape[0] # Number of neurons in the previous layer

        if len(new_weights) != num_expected_weights:
            raise ValueError(f"Incorrect number of weights provided. Expected {num_expected_weights}, got {len(new_weights)}.")

        if not (0 <= neuron_idx < self.W[layer_idx].shape[1]):
            raise ValueError("Invalid neuron index for setting weights.")

        self.W[layer_idx][:, neuron_idx] = np.array(new_weights)


# ------------------------------------------------------------------------------------
# PART 2: Neural Cellular Automaton Class
# ------------------------------------------------------------------------------------
class NeuralCellularAutomaton:
    def __init__(self, grid_size=50, layer_sizes=[9,8,1], activation="relu",
                 weight_scale=1.0, bias=0.0, random_seed=None):
        self.grid_size = grid_size
        self.initial_seed = random_seed

        if random_seed is not None:
            np.random.seed(random_seed)
        self.state = np.random.rand(grid_size, grid_size)
        self.history = deque(maxlen=20)

        self.mlp = FlexibleMLP(layer_sizes=layer_sizes,
                               activation=activation,
                               weight_scale=weight_scale,
                               bias=bias,
                               random_seed=random_seed)
        self.paused = True

    def get_neighborhood(self, r, c):
        neighbors = []
        for dr in [-1,0,1]:
            for dc in [-1,0,1]:
                rr = (r + dr) % self.grid_size
                cc = (c + dc) % self.grid_size
                neighbors.append(self.state[rr, cc])
        return np.array(neighbors)

    def step(self):
        if not self.paused: # Only step if not paused
            self.history.append(np.copy(self.state))

            # Vectorized neighborhood extraction
            neighborhood_channels = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    # np.roll shifts the array. To get the value of (r+dr, c+dc) at (r,c),
                    # we need to roll the grid by (-dr, -dc)
                    shifted_grid = np.roll(self.state, shift=(-dr, -dc), axis=(0, 1))
                    neighborhood_channels.append(shifted_grid)

            # Stack the 9 grids along a new axis to get a (grid_size, grid_size, 9) array
            neighborhood_tensor = np.stack(neighborhood_channels, axis=-1)

            # Reshape to (grid_size * grid_size, 9) for batch processing by MLP
            # Each row is a 9-element neighborhood for a cell
            batched_neighborhoods = neighborhood_tensor.reshape(-1, 9)

            # Perform a single forward pass for all cells
            # The MLP's forward method is already designed to handle batch inputs
            # where x is (batch_size, input_dim)
            new_state_flat = self.mlp.forward(batched_neighborhoods)

            # Reshape the output back to the original grid shape
            # The output from MLP is (grid_size * grid_size, 1)
            new_state = new_state_flat.reshape(self.grid_size, self.grid_size)

            self.state = new_state

    def step_back(self):
        if self.history:
            self.state = self.history.pop()
            # self.paused = True # Let user decide if they want to resume or stay paused
        else:
            print("History is empty. Cannot step back further.")

    def reset_grid(self, random_seed=None):
        current_seed = random_seed if random_seed is not None else np.random.randint(0, 1000000)
        np.random.seed(current_seed)
        self.state = np.random.rand(self.grid_size, self.grid_size)
        self.history.clear() # Clear history on grid reset

    def randomize_weights(self, layer_sizes, activation, weight_scale, bias, random_seed=None):
        current_seed = random_seed if random_seed is not None else np.random.randint(0, 1000000)
        self.mlp.set_params(layer_sizes, activation, weight_scale, bias, random_seed=current_seed)

    def get_current_params(self):
        return {
            "layer_sizes": self.mlp.layer_sizes,
            "activation": self.mlp.activation_name,
            "weight_scale": self.mlp.weight_scale,
            "bias": self.mlp.bias_value,
            "grid_size": self.grid_size,
            "initial_seed": self.initial_seed
        }