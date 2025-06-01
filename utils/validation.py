# utils/validation.py

from config import MAX_HIDDEN_LAYERS_COUNT, MIN_NODE_COUNT_PER_LAYER, MAX_NODE_COUNT_PER_LAYER

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