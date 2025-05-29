# nca_core.py
import numpy as np

# For the matplotlib colormaps (used by the app.py, but good to keep awareness)
# import matplotlib
# from matplotlib.cm import get_cmap
# matplotlib.use("Agg") # Ensure no GUI backend is attempted if get_cmap is used here

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
    """
    A feedforward neural network with a variable number of layers.
    Example: layer_sizes = [9, 16, 8, 1]
      - input dimension = 9
      - hidden layers = 16, then 8
      - output dimension = 1
    """
    def __init__(self, layer_sizes, activation="relu", weight_scale=1.0,
                 bias=0.0, random_seed=None):
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1
        self.activation_name = activation
        self.activation = get_activation_func(activation)
        self.weight_scale = weight_scale
        self.bias_value = bias
        if random_seed is not None:
            # print(f"MLP seed: {random_seed}")
            np.random.seed(random_seed)

        # Initialize weights & biases
        self.W = []
        self.b = []
        for i in range(self.n_layers):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i+1]
            W_i = weight_scale * np.random.randn(in_dim, out_dim)
            # Option: random biases scaled by 'bias'
            b_i = bias * np.random.randn(out_dim)
            self.W.append(W_i)
            self.b.append(b_i)

    def forward(self, x):
        """
        Forward pass of the MLP.
        x: shape (input_dim, ) for a single example
        returns: shape (output_dim, )
        """
        h = x
        for i in range(self.n_layers):
            z = np.dot(h, self.W[i]) + self.b[i]
            if i < (self.n_layers - 1):
                # Hidden layer(s): apply chosen activation
                h = self.activation(z)
            else:
                # Final layer: if output is dimension 1, clamp with sigmoid to keep CA states in [0..1]
                if self.layer_sizes[-1] == 1:
                    z = 1.0 / (1.0 + np.exp(-z))
                h = z
        return h  # shape (out_dim,)

    def set_params(self, layer_sizes, activation, weight_scale, bias, random_seed=None):
        """Rebuild the network with new parameters."""
        self.__init__(layer_sizes, activation, weight_scale, bias, random_seed)

    def get_params_for_viz(self):
        """Returns parameters needed for visualization (weights, layer_sizes)."""
        return {
            "layer_sizes": self.layer_sizes,
            "weights": [w.tolist() for w in self.W] # Convert numpy arrays to lists for JSON
        }

    def get_activations(self, x):
        """
        Performs a forward pass and returns all layer activations.
        x: shape (input_dim, )
        returns: list of activations for each layer, including input
        """
        layer_acts = [x.tolist()] # Store input layer activations as list
        h = x
        for i in range(self.n_layers):
            z = np.dot(h, self.W[i]) + self.b[i]
            if i < (self.n_layers - 1): # Hidden layer
                h = self.activation(z)
            else: # Output layer
                if self.layer_sizes[-1] == 1: # Special sigmoid for CA state
                    z = 1.0 / (1.0 + np.exp(-z))
                h = z
            layer_acts.append(h.tolist() if isinstance(h, np.ndarray) else [h]) # Store as list
        return layer_acts


# ------------------------------------------------------------------------------------
# PART 2: Neural Cellular Automaton Class
# ------------------------------------------------------------------------------------
class NeuralCellularAutomaton:
    def __init__(self, grid_size=50, layer_sizes=[9,8,1], activation="relu",
                 weight_scale=1.0, bias=0.0, random_seed=None):
        self.grid_size = grid_size
        self.initial_seed = random_seed # Store the seed for re-initialization if needed

        # Random initial state in [0..1]
        if random_seed is not None:
            # print(f"NCA Grid seed: {random_seed}")
            np.random.seed(random_seed)
        self.state = np.random.rand(grid_size, grid_size)

        # Create the MLP
        self.mlp = FlexibleMLP(layer_sizes=layer_sizes,
                               activation=activation,
                               weight_scale=weight_scale,
                               bias=bias,
                               random_seed=random_seed)
        self.paused = True  # Start paused

    def get_neighborhood(self, r, c):
        """
        Extract 3x3 neighborhood around (r, c), wrapping around edges (toroidal).
        Returns a flattened array of shape (9,).
        """
        neighbors = []
        for dr in [-1,0,1]:
            for dc in [-1,0,1]:
                rr = (r + dr) % self.grid_size
                cc = (c + dc) % self.grid_size
                neighbors.append(self.state[rr, cc])
        return np.array(neighbors)

    def step(self):
        """Compute one iteration of the CA using the MLP rule."""
        if self.paused:
            return
        new_state = np.zeros_like(self.state)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                neigh = self.get_neighborhood(r, c)
                out = self.mlp.forward(neigh)
                new_state[r, c] = out[0] if isinstance(out, np.ndarray) and out.ndim > 0 else out
        self.state = new_state

    def reset_grid(self, random_seed=None):
        """Re-randomize the grid state in [0..1]."""
        current_seed = random_seed if random_seed is not None else np.random.randint(0, 1000000)
        # print(f"NCA Reset Grid seed: {current_seed}")
        np.random.seed(current_seed)
        self.state = np.random.rand(self.grid_size, self.grid_size)


    def randomize_weights(self, layer_sizes, activation, weight_scale, bias, random_seed=None):
        """Rebuild the MLP with new random weights."""
        current_seed = random_seed if random_seed is not None else np.random.randint(0, 1000000)
        # print(f"NCA Randomize Weights seed: {current_seed}")
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