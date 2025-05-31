# nca_core.py
import numpy as np
import torch
from collections import deque

# Determine if CUDA is available and set the device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ------------------------------------------------------------------------------------
# PART 1: Flexible Multi-Layer Neural Network Definition
# ------------------------------------------------------------------------------------
class FlexibleMLP(torch.nn.Module):
    def __init__(self, layer_sizes, activation="relu", weight_scale=1.0,
                 bias=0.0, random_seed=None):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1
        self.activation_name = activation
        
        if activation.lower() == "relu":
            self.activation = torch.relu
        elif activation.lower() == "sigmoid":
            self.activation = torch.sigmoid
        elif activation.lower() == "tanh":
            self.activation = torch.tanh
        else:
            self.activation = torch.relu # Default to ReLU

        self.weight_scale = weight_scale
        self.bias_value = bias
        
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed) # Keep for numpy operations outside MLP if any

        self.layers = torch.nn.ModuleList()
        for i in range(self.n_layers):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i+1]
            linear_layer = torch.nn.Linear(in_dim, out_dim, bias=True)
            
            # Initialize weights and biases
            torch.nn.init.normal_(linear_layer.weight, mean=0.0, std=weight_scale)
            if bias != 0.0:
                torch.nn.init.constant_(linear_layer.bias, bias)
            else:
                torch.nn.init.zeros_(linear_layer.bias)
            
            self.layers.append(linear_layer)
        
        self.to(DEVICE) # Move model to the specified device

    def forward(self, x):
        # Ensure input is a torch tensor and on the correct device
        if isinstance(x, np.ndarray):
            h = torch.from_numpy(x).float().to(DEVICE)
        else:
            h = x.float().to(DEVICE)

        for i, layer in enumerate(self.layers):
            z = layer(h)
            if i < (self.n_layers - 1):
                h = self.activation(z)
            else:
                if self.layer_sizes[-1] == 1: # Output layer for NCA state
                    z = torch.sigmoid(z) # Ensure output is between 0 and 1
                h = z
        return h.cpu().detach().numpy() # Return numpy array on CPU

    def set_params(self, layer_sizes, activation, weight_scale, bias, random_seed=None):
        # Reinitialize the MLP with new parameters
        self.__init__(layer_sizes, activation, weight_scale, bias, random_seed)

    def get_params_for_viz(self):
        weights = []
        for layer in self.layers:
            weights.append(layer.weight.T.cpu().detach().numpy().tolist()) # Transpose to match original W[i] shape (in_dim, out_dim)
        return {
            "layer_sizes": self.layer_sizes,
            "weights": weights
        }

    def get_activations(self, x):
        # Ensure input is a torch tensor and on the correct device
        if isinstance(x, np.ndarray):
            h = torch.from_numpy(x).float().to(DEVICE)
        else:
            h = x.float().to(DEVICE)

        layer_acts = [h.cpu().detach().numpy().tolist()]
        for i, layer in enumerate(self.layers):
            z = layer(h)
            if i < (self.n_layers - 1):
                h = self.activation(z)
            else:
                if self.layer_sizes[-1] == 1:
                    z = torch.sigmoid(z)
                h = z
            layer_acts.append(h.cpu().detach().numpy().tolist())
        return layer_acts

    def get_incoming_weights_for_neuron(self, layer_idx, neuron_idx):
        """
        Get the incoming weights for a specific neuron.
        layer_idx: Index of the layer the neuron belongs to (0 for first hidden layer, etc., corresponding to self.layers[layer_idx]).
        neuron_idx: Index of the neuron within that layer.
        """
        if not (0 <= layer_idx < self.n_layers):
            raise ValueError(f"Invalid layer index: {layer_idx}")
        
        # Weights are stored as (out_dim, in_dim) in PyTorch Linear layer.weight
        # We need to get the column corresponding to neuron_idx after transpose, so it's row neuron_idx before transpose.
        # Or simply, layer.weight[neuron_idx, :]
        layer_weights = self.layers[layer_idx].weight.cpu().detach().numpy()
        
        if not (0 <= neuron_idx < layer_weights.shape[0]): # layer_weights.shape[0] is out_dim
            raise ValueError(f"Invalid neuron index: {neuron_idx} for layer {layer_idx+1} with size {layer_weights.shape[0]}")
        
        return layer_weights[neuron_idx, :].tolist()


    def set_incoming_weights_for_neuron(self, layer_idx, neuron_idx, new_weights):
        """
        Set the incoming weights for a specific neuron.
        layer_idx: Index of the layer the neuron belongs to (0 for first hidden layer, etc.).
        neuron_idx: Index of the neuron within that layer.
        new_weights: A list or 1D NumPy array of new weights.
        """
        if not (0 <= layer_idx < self.n_layers):
            raise ValueError("Invalid layer index for setting weights.")
        
        # PyTorch weights are (out_dim, in_dim)
        layer_weight_tensor = self.layers[layer_idx].weight
        num_expected_weights = layer_weight_tensor.shape[1] # Number of neurons in the previous layer (in_dim)

        if len(new_weights) != num_expected_weights:
            raise ValueError(f"Incorrect number of weights provided. Expected {num_expected_weights}, got {len(new_weights)}.")

        if not (0 <= neuron_idx < layer_weight_tensor.shape[0]): # layer_weight_tensor.shape[0] is out_dim
            raise ValueError("Invalid neuron index for setting weights.")

        with torch.no_grad(): # Ensure no gradient computation during direct weight modification
            self.layers[layer_idx].weight[neuron_idx, :] = torch.tensor(new_weights, dtype=torch.float32).to(DEVICE)


# ------------------------------------------------------------------------------------
# PART 2: Neural Cellular Automaton Class
# ------------------------------------------------------------------------------------
class NeuralCellularAutomaton:
    def __init__(self, grid_size=50, layer_sizes=[9,8,1], activation="relu",
                 weight_scale=1.0, bias=0.0, random_seed=None, initial_state=None):
        self.grid_size = grid_size
        self.initial_seed = random_seed

        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed) # Keep for numpy operations outside torch if any
        
        if initial_state is not None:
            self.state = initial_state.to(DEVICE) # Use provided initial state
        else:
            self.state = torch.rand(grid_size, grid_size, device=DEVICE) # Initialize random state on device
        
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
                neighbors.append(self.state[rr, cc].item()) # Convert tensor to scalar for list
        return np.array(neighbors) # Return numpy array as this is for single cell detail, not batch processing

    def step(self):
        if not self.paused: # Only step if not paused
            self.history.append(self.state.cpu().clone()) # Store a copy on CPU

            # Vectorized neighborhood extraction using torch.roll
            neighborhood_channels = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    # torch.roll shifts the tensor. To get the value of (r+dr, c+dc) at (r,c),
                    # we need to roll the grid by (-dr, -dc)
                    shifted_grid = torch.roll(self.state, shifts=(-dr, -dc), dims=(0, 1))
                    neighborhood_channels.append(shifted_grid)

            # Stack the 9 grids along a new axis to get a (grid_size, grid_size, 9) tensor
            neighborhood_tensor = torch.stack(neighborhood_channels, dim=-1)

            # Reshape to (grid_size * grid_size, 9) for batch processing by MLP
            # Each row is a 9-element neighborhood for a cell
            batched_neighborhoods = neighborhood_tensor.reshape(-1, 9)

            # Perform a single forward pass for all cells
            # The MLP's forward method is already designed to handle batch inputs
            # where x is (batch_size, input_dim)
            new_state_flat = self.mlp.forward(batched_neighborhoods) # This returns numpy array on CPU

            # Reshape the output back to the original grid shape and move to device
            # The output from MLP is (grid_size * grid_size, 1)
            new_state = torch.from_numpy(new_state_flat).float().reshape(self.grid_size, self.grid_size).to(DEVICE)

            self.state = new_state

    def step_back(self):
        if self.history:
            self.state = self.history.pop().to(DEVICE) # Move popped state back to device
            # self.paused = True # Let user decide if they want to resume or stay paused
        else:
            print("History is empty. Cannot step back further.")

    def reset_grid(self, random_seed=None):
        current_seed = random_seed if random_seed is not None else torch.randint(0, 1000000, (1,)).item()
        torch.manual_seed(current_seed)
        np.random.seed(current_seed) # Keep for numpy operations outside torch if any
        self.state = torch.rand(self.grid_size, self.grid_size, device=DEVICE) # Reset state on device
        self.history.clear() # Clear history on grid reset

    def randomize_weights(self, layer_sizes, activation, weight_scale, bias, random_seed=None):
        current_seed = random_seed if random_seed is not None else torch.randint(0, 1000000, (1,)).item()
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