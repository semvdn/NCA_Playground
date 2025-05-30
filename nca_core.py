# nca_core.py
import numpy as np
from collections import deque

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

    def set_architecture(self, layer_sizes, activation):
        """Update the network's architecture (layer sizes and activation) without re-randomizing weights."""
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1
        self.activation_name = activation
        self.activation = get_activation_func(activation)
        
        # Re-initialize weights and biases with current scale/bias but new architecture
        # This will re-randomize weights, but it's necessary when architecture changes.
        # The alternative is to try to adapt existing weights, which is more complex
        # and not requested by the plan.
        self.W = []
        self.b = []
        for i in range(self.n_layers):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i+1]
            W_i = self.weight_scale * np.random.randn(in_dim, out_dim)
            b_i = self.bias_value * np.random.randn(out_dim)
            self.W.append(W_i)
            self.b.append(b_i)

    def set_params(self, layer_sizes, activation, weight_scale, bias, random_seed=None):
        """Rebuild the network with new parameters (full re-initialization)."""
        self.__init__(layer_sizes, activation, weight_scale, bias, random_seed)

    def set_weights_from_preset(self, preset_name):
        """
        Sets the first layer weights (9 inputs to first hidden layer) based on a predefined preset pattern.
        The rest of the network's weights remain as they are.
        """
        # Ensure the first layer has 9 inputs
        if self.layer_sizes[0] != 9:
            raise ValueError("MLP input layer must be 9 for preset patterns.")
        
        # Define preset 3x3 kernels. These will be reshaped to (9, out_dim)
        # Assuming output dimension of first layer is 1 for simplicity for now,
        # or that the kernel is applied to each output node of the first layer.
        # For now, let's assume the first hidden layer has at least 1 node,
        # and we apply the kernel to its first node.
        
        # Kernels are 3x3, flattened to 9 for input to MLP
        # Identity: center pixel passes through
        identity_kernel = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ])

        # Horizontal Edge Detector: highlights horizontal differences
        horizontal_edge_kernel = np.array([
            [1, 1, 1],
            [0, 0, 0],
            [-1, -1, -1]
        ])

        # Vertical Edge Detector: highlights vertical differences
        vertical_edge_kernel = np.array([
            [1, 0, -1],
            [1, 0, -1],
            [1, 0, -1]
        ])

        # Blur: averages neighborhood values
        blur_kernel = np.array([
            [1/9, 1/9, 1/9],
            [1/9, 1/9, 1/9],
            [1/9, 1/9, 1/9]
        ])

        # Concentric: center-surround activation
        concentric_kernel = np.array([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ])

        kernels = {
            "Identity Pass-through": identity_kernel,
            "Horizontal Edge Detector": horizontal_edge_kernel,
            "Vertical Edge Detector": vertical_edge_kernel,
            "Blur": blur_kernel,
            "Concentric": concentric_kernel
        }

        if preset_name not in kernels:
            raise ValueError(f"Unknown preset pattern: {preset_name}")

        kernel = kernels[preset_name]
        
        # The first weight matrix self.W[0] connects 9 inputs to layer_sizes[1] outputs.
        # We need to apply the 3x3 kernel to each output node of the first hidden layer.
        # So, if layer_sizes[1] is N, the kernel will be replicated N times.
        
        # Reshape kernel to (9,)
        flat_kernel = kernel.flatten()

        # Create the new first layer weight matrix
        # It should have shape (input_dim, output_dim_first_hidden_layer) i.e., (9, self.layer_sizes[1])
        new_W0 = np.zeros((self.layer_sizes[0], self.layer_sizes[1]))
        
        for out_node_idx in range(self.layer_sizes[1]):
            # Apply the flattened kernel to each output node of the first hidden layer
            new_W0[:, out_node_idx] = flat_kernel
        
        self.W[0] = new_W0
        # Biases for the first layer can remain as they are or be reset to zero.
        # For now, let's keep them as they are.

    def generate_parametric_weights(self, pattern_type, parameters):
        """
        Generates first layer weights based on parametric patterns.
        """
        if self.layer_sizes[0] != 9:
            raise ValueError("MLP input layer must be 9 for parametric patterns.")

        kernel = np.zeros((3, 3))

        if pattern_type == "Gaussian":
            sigma = parameters.get("sigma", 1.0)
            # Create a 3x3 Gaussian kernel
            for i in range(3):
                for j in range(3):
                    x, y = i - 1, j - 1 # Center at (0,0)
                    kernel[i, j] = (1 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
            kernel /= np.sum(kernel) # Normalize
        elif pattern_type == "Laplacian":
            strength = parameters.get("strength", 1.0)
            # Standard 3x3 Laplacian kernel
            kernel = strength * np.array([
                [ 0,  1,  0],
                [ 1, -4,  1],
                [ 0,  1,  0]
            ])
        elif pattern_type == "Directional":
            angle_deg = parameters.get("angle", 0.0)
            magnitude = parameters.get("magnitude", 1.0)
            angle_rad = np.deg2rad(angle_deg)

            # Simple directional filter (e.g., Sobel-like)
            # This is a simplified example. More complex directional filters exist.
            # For a 3x3 kernel, we can approximate a directional gradient.
            # Let's create a gradient along the specified angle.
            # Example: a horizontal filter (angle 0 or 180) would be like [-1, 0, 1]
            # A vertical filter (angle 90 or 270) would be like [[-1],[0],[1]]
            
            # Create a coordinate grid for the kernel
            x_coords = np.array([-1, 0, 1])
            y_coords = np.array([-1, 0, 1])
            
            for i in range(3):
                for j in range(3):
                    x, y = j - 1, -(i - 1) # Adjust for image coordinates (y-axis inverted)
                    # Project (x,y) onto the direction vector (cos(angle), sin(angle))
                    # The value is proportional to the dot product
                    kernel[i, j] = magnitude * (x * np.cos(angle_rad) + y * np.sin(angle_rad))
            
            # Normalize to prevent extreme values, but preserve relative differences
            if np.max(np.abs(kernel)) > 0:
                kernel /= np.max(np.abs(kernel))
        else:
            raise ValueError(f"Unknown parametric pattern type: {pattern_type}")

        flat_kernel = kernel.flatten()
        new_W0 = np.zeros((self.layer_sizes[0], self.layer_sizes[1]))
        for out_node_idx in range(self.layer_sizes[1]):
            new_W0[:, out_node_idx] = flat_kernel
        self.W[0] = new_W0

    def set_first_layer_weights(self, weights_matrix):
        """
        Directly sets the weights of the first layer.
        weights_matrix: A 2D numpy array of shape (9, output_dim_first_hidden_layer).
        """
        if not isinstance(weights_matrix, np.ndarray) or weights_matrix.shape[0] != 9:
            raise ValueError("weights_matrix must be a numpy array with 9 rows (for 3x3 input).")
        if weights_matrix.shape[1] != self.layer_sizes[1]:
            raise ValueError(f"weights_matrix must have {self.layer_sizes[1]} columns (for first hidden layer output nodes).")
        
        self.W[0] = weights_matrix

    def load_weights_from_file(self, file_content, filename):
        """
        Loads all weights and biases from a JSON or CSV file.
        Assumes JSON for now, will extend for CSV if needed.
        """
        import json
        import os

        file_extension = os.path.splitext(filename)[1].lower()

        if file_extension == '.json':
            data = json.loads(file_content)
            if "weights" not in data or "biases" not in data or "layer_sizes" not in data:
                raise ValueError("JSON file must contain 'weights', 'biases', and 'layer_sizes'.")
            
            new_W = [np.array(w) for w in data["weights"]]
            new_b = [np.array(b) for b in data["biases"]]
            new_layer_sizes = data["layer_sizes"]

            # Basic validation: check if dimensions match
            if len(new_W) != len(new_b) or len(new_W) != len(new_layer_sizes) - 1:
                raise ValueError("Mismatched dimensions in loaded weights/biases/layer_sizes.")
            
            # Update MLP's internal state
            self.layer_sizes = new_layer_sizes
            self.n_layers = len(new_layer_sizes) - 1
            self.W = new_W
            self.b = new_b
            # Note: activation, weight_scale, bias_value are not loaded from file
            # They retain their current values or would need to be part of the file format.
            # For simplicity, we only load W, b, layer_sizes.
            print(f"Weights and biases loaded from {filename}")
        elif file_extension == '.csv':
            # Implement CSV parsing if necessary. This would be more complex
            # as CSV typically represents a single matrix, not multiple layers.
            raise NotImplementedError("CSV weight import is not yet implemented. Please use JSON.")
        else:
            raise ValueError("Unsupported file type. Only .json is supported for now.")

    def get_weights_for_export(self):
        """
        Returns current weights and biases formatted for export (e.g., JSON).
        """
        import json
        export_data = {
            "layer_sizes": self.layer_sizes,
            "weights": [w.tolist() for w in self.W],
            "biases": [b.tolist() for b in self.b]
        }
        return json.dumps(export_data, indent=4), "nca_weights.json"

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
        self.history = deque(maxlen=20) # Store last 20 states for step_back

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
        # Save current state to history before stepping forward
        self.history.append(np.copy(self.state))

        new_state = np.zeros_like(self.state)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                neigh = self.get_neighborhood(r, c)
                out = self.mlp.forward(neigh)
                new_state[r, c] = out[0] if isinstance(out, np.ndarray) and out.ndim > 0 else out
        self.state = new_state

    def step_back(self):
        """Revert the simulation to a previous state from history."""
        if self.history:
            self.state = self.history.pop()
            self.paused = True # Pause after stepping back
        else:
            print("History is empty. Cannot step back further.")

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