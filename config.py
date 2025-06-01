# config.py

# Presets (seed, layers, activation, weight_scale, bias)
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