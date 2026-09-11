// Browser-native NCA engine.
//
// This module mirrors the response shapes of the former Flask/PyTorch service so
// the existing UI can run unchanged on a static host such as GitHub Pages.

const PRESETS = {
    'Linear': [null, [9, 1], 'relu', 1.0, 0.0],
    'Shallow ReLU': [null, [9, 16, 1], 'relu', 1.0, 0.0],
    'Deep Tanh': [null, [9, 32, 16, 1], 'tanh', 1.0, 0.0],
    'Wide Sigmoid': [null, [9, 32, 1], 'sigmoid', 1.0, 0.0],
    'Custom': [null, [9, 8, 1], 'relu', 1.0, 0.0]
};

const AVAILABLE_ACTIVATIONS = ['relu', 'sigmoid', 'tanh'];
const AVAILABLE_COLORMAPS = [
    'viridis', 'plasma', 'magma', 'cividis', 'inferno',
    'Greys', 'Blues', 'GnBu', 'coolwarm'
];

const CONSTRAINTS = {
    max_hidden_layers: 3,
    min_node_size: 1,
    max_node_size: 32
};

const MIN_RANDOM_LAYERS = 1;
const MAX_RANDOM_LAYERS = CONSTRAINTS.max_hidden_layers;
const MIN_RANDOM_NODES = 2;
const MAX_RANDOM_NODES = 10;
const HISTORY_LIMIT = 20;

const COLORMAPS = {
    viridis: ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725'],
    plasma: ['#0d0887', '#7e03a8', '#cc4778', '#f89540', '#f0f921'],
    magma: ['#000004', '#3b0f70', '#8c2981', '#de4968', '#fe9f6d', '#fcfdbf'],
    cividis: ['#00204c', '#424086', '#6c5d7c', '#958f78', '#c8c46c', '#ffffe0'],
    inferno: ['#000004', '#420a68', '#932667', '#dd513a', '#fca50a', '#fcffa4'],
    Greys: ['#ffffff', '#d9d9d9', '#969696', '#525252', '#000000'],
    Blues: ['#f7fbff', '#c6dbef', '#6baed6', '#2171b5', '#08306b'],
    GnBu: ['#f7fcf0', '#ccebc5', '#7bccc4', '#2b8cbe', '#084081'],
    coolwarm: ['#3b4cc0', '#8db0fe', '#dddddd', '#f49a7b', '#b40426']
};

function randomSeed() {
    if (globalThis.crypto?.getRandomValues) {
        const value = new Uint32Array(1);
        globalThis.crypto.getRandomValues(value);
        return value[0] >>> 0;
    }
    return Math.floor(Math.random() * 0x100000000) >>> 0;
}

// Small, deterministic PRNG for reproducible grids/weights when a seed is set.
function mulberry32(seed) {
    let a = seed >>> 0;
    return function rng() {
        a = (a + 0x6D2B79F5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

function createGaussianSampler(rng) {
    let spare = null;
    return function gaussian() {
        if (spare !== null) {
            const value = spare;
            spare = null;
            return value;
        }
        let u = 0;
        let v = 0;
        while (u <= Number.EPSILON) u = rng();
        while (v <= Number.EPSILON) v = rng();
        const radius = Math.sqrt(-2 * Math.log(u));
        const theta = 2 * Math.PI * v;
        spare = radius * Math.sin(theta);
        return radius * Math.cos(theta);
    };
}

function sigmoid(x) {
    if (x >= 0) {
        const z = Math.exp(-x);
        return 1 / (1 + z);
    }
    const z = Math.exp(x);
    return z / (1 + z);
}

function activate(name, x) {
    switch (name) {
        case 'sigmoid': return sigmoid(x);
        case 'tanh': return Math.tanh(x);
        case 'relu':
        default: return x > 0 ? x : 0;
    }
}

function clamp01(value) {
    if (!Number.isFinite(value)) return 0;
    return Math.min(1, Math.max(0, value));
}

function parseHex(hex) {
    return [
        parseInt(hex.slice(1, 3), 16),
        parseInt(hex.slice(3, 5), 16),
        parseInt(hex.slice(5, 7), 16)
    ];
}

function toHexByte(value) {
    return Math.round(Math.min(255, Math.max(0, value))).toString(16).padStart(2, '0');
}

function interpolateColor(stops, value) {
    const t = clamp01(value) * (stops.length - 1);
    const lo = Math.floor(t);
    const hi = Math.min(stops.length - 1, lo + 1);
    const f = t - lo;
    const a = parseHex(stops[lo]);
    const b = parseHex(stops[hi]);
    const r = a[0] + (b[0] - a[0]) * f;
    const g = a[1] + (b[1] - a[1]) * f;
    const bl = a[2] + (b[2] - a[2]) * f;
    return `#${toHexByte(r)}${toHexByte(g)}${toHexByte(bl)}`;
}

function validateLayerSizes(layerSizes) {
    if (!Array.isArray(layerSizes) || layerSizes.length < 2) {
        throw new Error('Layer sizes must contain at least input and output layers.');
    }
    if (layerSizes[0] !== 9 || layerSizes[layerSizes.length - 1] !== 1) {
        throw new Error('NCA layers must start with 9 inputs and end with 1 output.');
    }
    const hidden = layerSizes.slice(1, -1);
    if (hidden.length > CONSTRAINTS.max_hidden_layers) {
        throw new Error(`A maximum of ${CONSTRAINTS.max_hidden_layers} hidden layers is supported.`);
    }
    for (const size of hidden) {
        if (!Number.isInteger(size) || size < CONSTRAINTS.min_node_size || size > CONSTRAINTS.max_node_size) {
            throw new Error(`Hidden layer sizes must be integers from ${CONSTRAINTS.min_node_size} to ${CONSTRAINTS.max_node_size}.`);
        }
    }
}

function parseLayerSizes(value, fallback) {
    if (Array.isArray(value)) {
        const parsed = value.map(Number);
        validateLayerSizes(parsed);
        return parsed;
    }
    if (typeof value === 'string' && value.trim()) {
        const parsed = value.split(',').map(part => Number.parseInt(part.trim(), 10));
        validateLayerSizes(parsed);
        return parsed;
    }
    return [...fallback];
}

function cloneWeights(weights) {
    return weights.map(layer => new Float64Array(layer));
}

class FlexibleMLP {
    constructor(layerSizes, activationName = 'relu', weightScale = 1.0, bias = 0.0, seed = null) {
        validateLayerSizes(layerSizes);
        this.layerSizes = [...layerSizes];
        this.activationName = AVAILABLE_ACTIVATIONS.includes(activationName) ? activationName : 'relu';
        this.weightScale = Number(weightScale);
        this.biasValue = Number(bias);
        this.weights = [];
        this.biases = [];

        const rng = mulberry32(seed === null ? randomSeed() : (Number(seed) >>> 0));
        const gaussian = createGaussianSampler(rng);

        for (let layerIdx = 0; layerIdx < this.layerSizes.length - 1; layerIdx++) {
            const inSize = this.layerSizes[layerIdx];
            const outSize = this.layerSizes[layerIdx + 1];
            const layerWeights = new Float64Array(inSize * outSize);
            for (let i = 0; i < layerWeights.length; i++) {
                layerWeights[i] = gaussian() * this.weightScale;
            }
            const layerBiases = new Float64Array(outSize);
            layerBiases.fill(this.biasValue);
            this.weights.push(layerWeights);
            this.biases.push(layerBiases);
        }
    }

    createWorkspace() {
        return this.layerSizes.slice(1).map(size => new Float64Array(size));
    }

    forwardScalar(input, workspace) {
        let current = input;
        const lastLayerIdx = this.weights.length - 1;

        for (let layerIdx = 0; layerIdx < this.weights.length; layerIdx++) {
            const inSize = this.layerSizes[layerIdx];
            const outSize = this.layerSizes[layerIdx + 1];
            const weights = this.weights[layerIdx];
            const biases = this.biases[layerIdx];
            const output = workspace[layerIdx];

            for (let outIdx = 0; outIdx < outSize; outIdx++) {
                let sum = biases[outIdx];
                for (let inIdx = 0; inIdx < inSize; inIdx++) {
                    sum += current[inIdx] * weights[inIdx * outSize + outIdx];
                }
                output[outIdx] = layerIdx === lastLayerIdx
                    ? sigmoid(sum)
                    : activate(this.activationName, sum);
            }
            current = output;
        }

        return current[0];
    }

    getActivations(input) {
        let current = Float64Array.from(input);
        const activations = [Array.from(current)];
        const lastLayerIdx = this.weights.length - 1;

        for (let layerIdx = 0; layerIdx < this.weights.length; layerIdx++) {
            const inSize = this.layerSizes[layerIdx];
            const outSize = this.layerSizes[layerIdx + 1];
            const weights = this.weights[layerIdx];
            const biases = this.biases[layerIdx];
            const output = new Float64Array(outSize);

            for (let outIdx = 0; outIdx < outSize; outIdx++) {
                let sum = biases[outIdx];
                for (let inIdx = 0; inIdx < inSize; inIdx++) {
                    sum += current[inIdx] * weights[inIdx * outSize + outIdx];
                }
                output[outIdx] = layerIdx === lastLayerIdx
                    ? sigmoid(sum)
                    : activate(this.activationName, sum);
            }
            current = output;
            activations.push(Array.from(current));
        }

        return activations;
    }

    getParamsForViz() {
        const weightsForViz = this.weights.map((weights, layerIdx) => {
            const inSize = this.layerSizes[layerIdx];
            const outSize = this.layerSizes[layerIdx + 1];
            const matrix = new Array(inSize);
            for (let inIdx = 0; inIdx < inSize; inIdx++) {
                const row = new Array(outSize);
                for (let outIdx = 0; outIdx < outSize; outIdx++) {
                    row[outIdx] = weights[inIdx * outSize + outIdx];
                }
                matrix[inIdx] = row;
            }
            return matrix;
        });
        return {
            layer_sizes: [...this.layerSizes],
            weights: weightsForViz
        };
    }


}

class NeuralCellularAutomaton {
    constructor({
        gridSize = 50,
        layerSizes = [9, 8, 1],
        activation = 'relu',
        weightScale = 1.0,
        bias = 0.0,
        seed = null,
        initialState = null
    } = {}) {
        this.gridSize = Number.parseInt(gridSize, 10);
        this.initialSeed = seed === null || seed === undefined || seed === '' ? null : (Number(seed) >>> 0);
        this.paused = true;
        this.history = [];

        this.state = initialState
            ? Float32Array.from(initialState)
            : this.createRandomState(this.initialSeed);

        this.mlp = new FlexibleMLP(
            layerSizes,
            activation,
            Number(weightScale),
            Number(bias),
            this.initialSeed
        );
    }

    createRandomState(seed = null) {
        const rng = mulberry32(seed === null ? randomSeed() : (Number(seed) >>> 0));
        const state = new Float32Array(this.gridSize * this.gridSize);
        for (let i = 0; i < state.length; i++) state[i] = rng();
        return state;
    }

    getNeighborhood(r, c) {
        const neighbors = new Float64Array(9);
        let idx = 0;
        for (let dr = -1; dr <= 1; dr++) {
            const rr = (r + dr + this.gridSize) % this.gridSize;
            for (let dc = -1; dc <= 1; dc++) {
                const cc = (c + dc + this.gridSize) % this.gridSize;
                neighbors[idx++] = this.state[rr * this.gridSize + cc];
            }
        }
        return neighbors;
    }

    step() {
        this.history.push(this.state.slice());
        if (this.history.length > HISTORY_LIMIT) this.history.shift();

        const next = new Float32Array(this.state.length);
        const neighborhood = new Float64Array(9);
        const workspace = this.mlp.createWorkspace();
        const size = this.gridSize;

        for (let r = 0; r < size; r++) {
            const r0 = (r - 1 + size) % size;
            const r1 = r;
            const r2 = (r + 1) % size;
            for (let c = 0; c < size; c++) {
                const c0 = (c - 1 + size) % size;
                const c1 = c;
                const c2 = (c + 1) % size;

                neighborhood[0] = this.state[r0 * size + c0];
                neighborhood[1] = this.state[r0 * size + c1];
                neighborhood[2] = this.state[r0 * size + c2];
                neighborhood[3] = this.state[r1 * size + c0];
                neighborhood[4] = this.state[r1 * size + c1];
                neighborhood[5] = this.state[r1 * size + c2];
                neighborhood[6] = this.state[r2 * size + c0];
                neighborhood[7] = this.state[r2 * size + c1];
                neighborhood[8] = this.state[r2 * size + c2];

                next[r * size + c] = this.mlp.forwardScalar(neighborhood, workspace);
            }
        }
        this.state = next;
    }

    stepBack() {
        if (this.history.length) this.state = this.history.pop();
    }

    resetGrid(seed = null) {
        const normalizedSeed = seed === null || seed === undefined || seed === ''
            ? randomSeed()
            : (Number(seed) >>> 0);
        this.state = this.createRandomState(normalizedSeed);
        this.history = [];
    }

    getCurrentParams() {
        return {
            layer_sizes: [...this.mlp.layerSizes],
            activation: this.mlp.activationName,
            weight_scale: this.mlp.weightScale,
            bias: this.mlp.biasValue,
            grid_size: this.gridSize,
            initial_seed: this.initialSeed
        };
    }
}

function stateToHexColors(state, gridSize, colormapName) {
    const stops = COLORMAPS[colormapName] || COLORMAPS.viridis;
    const rows = new Array(gridSize);
    for (let r = 0; r < gridSize; r++) {
        const row = new Array(gridSize);
        for (let c = 0; c < gridSize; c++) {
            row[c] = interpolateColor(stops, state[r * gridSize + c]);
        }
        rows[r] = row;
    }
    return rows;
}

export class BrowserNCAService {
    constructor() {
        this.currentColormapName = 'viridis';
        this.nca = new NeuralCellularAutomaton({
            gridSize: 50,
            layerSizes: PRESETS.Linear[1],
            activation: PRESETS.Linear[2],
            weightScale: PRESETS.Linear[3],
            bias: PRESETS.Linear[4],
            seed: PRESETS.Linear[0]
        });
    }

    colors() {
        return stateToHexColors(this.nca.state, this.nca.gridSize, this.currentColormapName);
    }

    getConfig() {
        return {
            presets: PRESETS,
            available_activations: AVAILABLE_ACTIVATIONS,
            available_colormaps: AVAILABLE_COLORMAPS,
            default_params: this.nca.getCurrentParams(),
            current_colormap: this.currentColormapName,
            initial_grid_colors: this.colors(),
            mlp_params_for_viz: this.nca.mlp.getParamsForViz(),
            is_paused: this.nca.paused,
            constraints: CONSTRAINTS
        };
    }

    step() {
        // An explicit UI step advances once even when the continuous simulation is paused.
        this.nca.step();
        return { grid_colors: this.colors() };
    }

    stepBack() {
        this.nca.stepBack();
        return {
            grid_colors: this.colors(),
            is_paused: this.nca.paused
        };
    }

    togglePause() {
        this.nca.paused = !this.nca.paused;
        return { is_paused: this.nca.paused, message: 'Toggled pause.' };
    }

    applySettings(data = {}) {
        const presetName = data.preset_name;
        const current = this.nca.getCurrentParams();
        const wasPaused = this.nca.paused;
        let message;

        if (presetName && presetName !== 'Custom' && PRESETS[presetName]) {
            const [seed, layerSizes, activation, weightScale, bias] = PRESETS[presetName];
            this.nca = new NeuralCellularAutomaton({
                gridSize: current.grid_size,
                layerSizes,
                activation,
                weightScale,
                bias,
                seed
            });
            this.nca.paused = wasPaused;
            message = `Settings applied: Preset '${presetName}' loaded.`;
        } else {
            const layerSizes = parseLayerSizes(data.layer_sizes, current.layer_sizes);
            const activation = data.activation ?? current.activation;
            if (!AVAILABLE_ACTIVATIONS.includes(activation)) throw new Error(`Invalid activation: ${activation}`);
            const weightScale = Number(data.weight_scale ?? current.weight_scale);
            const bias = Number(data.bias ?? current.bias);
            if (!Number.isFinite(weightScale) || !Number.isFinite(bias)) throw new Error('Weight scale and bias must be finite numbers.');

            const state = this.nca.state.slice();
            const wasPaused = this.nca.paused;
            this.nca = new NeuralCellularAutomaton({
                gridSize: current.grid_size,
                layerSizes,
                activation,
                weightScale,
                bias,
                seed: null,
                initialState: state
            });
            this.nca.paused = wasPaused;
            message = 'Settings applied: Custom MLP parameters.';
        }

        return {
            message,
            grid_colors: this.colors(),
            mlp_params_for_viz: this.nca.mlp.getParamsForViz(),
            current_params: this.nca.getCurrentParams(),
            is_paused: this.nca.paused
        };
    }

    setColormap(colormapName) {
        if (!AVAILABLE_COLORMAPS.includes(colormapName)) throw new Error(`Invalid colormap name: ${colormapName}`);
        this.currentColormapName = colormapName;
        return {
            message: `Colormap set to ${colormapName}.`,
            grid_colors: this.colors()
        };
    }

    randomizeWeights(data = {}) {
        const current = this.nca.getCurrentParams();
        const layerSizes = parseLayerSizes(data.layer_sizes, current.layer_sizes);
        const activation = data.activation ?? current.activation;
        if (!AVAILABLE_ACTIVATIONS.includes(activation)) throw new Error(`Invalid activation: ${activation}`);
        const weightScale = Number(data.weight_scale ?? current.weight_scale);
        const bias = Number(data.bias ?? current.bias);
        const state = this.nca.state.slice();
        const wasPaused = this.nca.paused;

        this.nca = new NeuralCellularAutomaton({
            gridSize: current.grid_size,
            layerSizes,
            activation,
            weightScale,
            bias,
            seed: null,
            initialState: state
        });
        this.nca.paused = wasPaused;

        return {
            message: 'NCA weights randomized.',
            grid_colors: this.colors(),
            mlp_params_for_viz: this.nca.mlp.getParamsForViz(),
            current_params: this.nca.getCurrentParams(),
            is_paused: this.nca.paused
        };
    }

    randomizeGrid(data = {}) {
        const seed = data.seed === null || data.seed === undefined ? null : Number(data.seed);
        this.nca.resetGrid(Number.isFinite(seed) ? seed : null);
        return {
            message: 'NCA grid randomized.',
            grid_colors: this.colors(),
            is_paused: this.nca.paused
        };
    }

    randomizeArchitecture(data = {}) {
        const hiddenCount = MIN_RANDOM_LAYERS + Math.floor(Math.random() * (MAX_RANDOM_LAYERS - MIN_RANDOM_LAYERS + 1));
        const layerSizes = [9];
        for (let i = 0; i < hiddenCount; i++) {
            layerSizes.push(MIN_RANDOM_NODES + Math.floor(Math.random() * (MAX_RANDOM_NODES - MIN_RANDOM_NODES + 1)));
        }
        layerSizes.push(1);

        const activation = AVAILABLE_ACTIVATIONS[Math.floor(Math.random() * AVAILABLE_ACTIVATIONS.length)];
        const weightScale = Math.round((0.5 + Math.random() * 2.0) * 10) / 10;
        const bias = Math.round((-0.5 + Math.random()) * 10) / 10;
        const gridSize = this.nca.gridSize;

        this.nca = new NeuralCellularAutomaton({
            gridSize,
            layerSizes,
            activation,
            weightScale,
            bias,
            seed: null
        });
        if (data.was_running) this.nca.paused = false;

        return {
            message: 'NCA architecture randomized and reinitialized.',
            grid_colors: this.colors(),
            mlp_params_for_viz: this.nca.mlp.getParamsForViz(),
            current_params: this.nca.getCurrentParams(),
            is_paused: this.nca.paused
        };
    }

    restart() {
        const current = this.nca.getCurrentParams();
        const savedWeights = cloneWeights(this.nca.mlp.weights);
        this.nca = new NeuralCellularAutomaton({
            gridSize: current.grid_size,
            layerSizes: current.layer_sizes,
            activation: current.activation,
            weightScale: current.weight_scale,
            bias: current.bias,
            seed: current.initial_seed
        });
        this.nca.mlp.weights = savedWeights;
        this.nca.paused = false;
        this.nca.history = [];

        return {
            message: 'NCA reinitialized and restarted from last seed with current weights.',
            initial_grid_colors: this.colors(),
            mlp_params_for_viz: this.nca.mlp.getParamsForViz(),
            current_params: this.nca.getCurrentParams(),
            is_paused: this.nca.paused
        };
    }

    getCellDetails(r, c) {
        const row = Number(r);
        const col = Number(c);
        if (!Number.isInteger(row) || !Number.isInteger(col) || row < 0 || col < 0 || row >= this.nca.gridSize || col >= this.nca.gridSize) {
            throw new Error('Row/column out of bounds.');
        }
        const neighborhood = this.nca.getNeighborhood(row, col);
        return {
            selected_cell: { r: row, c: col },
            neighborhood: [
                Array.from(neighborhood.slice(0, 3)),
                Array.from(neighborhood.slice(3, 6)),
                Array.from(neighborhood.slice(6, 9))
            ],
            layer_activations: this.nca.mlp.getActivations(neighborhood)
        };
    }

    setGridState(data = {}) {
        const grid = data.grid_state;
        const size = this.nca.gridSize;
        if (!Array.isArray(grid) || grid.length !== size || grid.some(row => !Array.isArray(row) || row.length !== size)) {
            throw new Error(`Provided grid state must be ${size}x${size}.`);
        }
        const next = new Float32Array(size * size);
        for (let r = 0; r < size; r++) {
            for (let c = 0; c < size; c++) {
                const value = Number(grid[r][c]);
                if (!Number.isFinite(value)) throw new Error('Grid values must be finite numbers.');
                next[r * size + c] = value;
            }
        }
        this.nca.state = next;
        this.nca.history = [];
        this.nca.paused = !Boolean(data.was_running);
        return {
            message: 'Grid state updated successfully.',
            grid_colors: this.colors(),
            is_paused: this.nca.paused
        };
    }
}

export const browserNcaService = new BrowserNCAService();
