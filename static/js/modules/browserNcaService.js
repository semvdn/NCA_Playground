// Browser-native NCA engine.
//
// The UI still uses its original API-shaped adapter, but all state and compute live
// locally in this module so the playground can run as a static GitHub Pages site.

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
const CONSTRAINTS = { max_hidden_layers: 3, min_node_size: 1, max_node_size: 32 };
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

function normalizeSeed(seed) {
    if (seed === null || seed === undefined || seed === '') return null;
    const numeric = Number(seed);
    return Number.isFinite(numeric) ? (numeric >>> 0) : null;
}

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
    if (name === 'sigmoid') return sigmoid(x);
    if (name === 'tanh') return Math.tanh(x);
    return x > 0 ? x : 0;
}

function clamp01(value) {
    if (!Number.isFinite(value)) return 0;
    return Math.min(1, Math.max(0, value));
}

function parseHex(hex) {
    return [
        Number.parseInt(hex.slice(1, 3), 16),
        Number.parseInt(hex.slice(3, 5), 16),
        Number.parseInt(hex.slice(5, 7), 16)
    ];
}

function toHexByte(value) {
    return Math.round(Math.min(255, Math.max(0, value))).toString(16).padStart(2, '0');
}

function interpolateColor(stops, value) {
    const position = clamp01(value) * (stops.length - 1);
    const lo = Math.floor(position);
    const hi = Math.min(stops.length - 1, lo + 1);
    const fraction = position - lo;
    const a = parseHex(stops[lo]);
    const b = parseHex(stops[hi]);
    const red = a[0] + (b[0] - a[0]) * fraction;
    const green = a[1] + (b[1] - a[1]) * fraction;
    const blue = a[2] + (b[2] - a[2]) * fraction;
    return `#${toHexByte(red)}${toHexByte(green)}${toHexByte(blue)}`;
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

class FlexibleMLP {
    constructor(layerSizes, activationName = 'relu', weightScale = 1, bias = 0, seed = null) {
        validateLayerSizes(layerSizes);
        this.layerSizes = [...layerSizes];
        this.activationName = AVAILABLE_ACTIVATIONS.includes(activationName) ? activationName : 'relu';
        this.weightScale = Number(weightScale);
        this.biasValue = Number(bias);
        this.weights = [];
        this.biases = [];

        const rng = mulberry32(normalizeSeed(seed) ?? randomSeed());
        const gaussian = createGaussianSampler(rng);
        for (let layerIdx = 0; layerIdx < this.layerSizes.length - 1; layerIdx++) {
            const inSize = this.layerSizes[layerIdx];
            const outSize = this.layerSizes[layerIdx + 1];
            const weights = new Float64Array(inSize * outSize);
            for (let i = 0; i < weights.length; i++) weights[i] = gaussian() * this.weightScale;
            const biases = new Float64Array(outSize);
            biases.fill(this.biasValue);
            this.weights.push(weights);
            this.biases.push(biases);
        }
    }

    createWorkspace() {
        return this.layerSizes.slice(1).map(size => new Float64Array(size));
    }

    forwardScalar(input, workspace) {
        let current = input;
        const lastLayer = this.weights.length - 1;
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
                output[outIdx] = layerIdx === lastLayer ? sigmoid(sum) : activate(this.activationName, sum);
            }
            current = output;
        }
        return current[0];
    }

    getActivations(input) {
        let current = Float64Array.from(input);
        const activations = [Array.from(current)];
        const workspace = this.createWorkspace();
        const lastLayer = this.weights.length - 1;
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
                output[outIdx] = layerIdx === lastLayer ? sigmoid(sum) : activate(this.activationName, sum);
            }
            current = output;
            activations.push(Array.from(current));
        }
        return activations;
    }

    getParamsForViz() {
        const weights = this.weights.map((flat, layerIdx) => {
            const inSize = this.layerSizes[layerIdx];
            const outSize = this.layerSizes[layerIdx + 1];
            return Array.from({ length: inSize }, (_, inIdx) =>
                Array.from({ length: outSize }, (_, outIdx) => flat[inIdx * outSize + outIdx])
            );
        });
        return { layer_sizes: [...this.layerSizes], weights };
    }
}

class NeuralCellularAutomaton {
    constructor({ gridSize = 50, layerSizes = [9, 8, 1], activation = 'relu', weightScale = 1, bias = 0, seed = null } = {}) {
        this.gridSize = Number.parseInt(gridSize, 10);
        this.initialSeed = normalizeSeed(seed) ?? randomSeed();
        this.paused = true;
        this.history = [];
        this.state = this.createRandomState(this.initialSeed);
        this.initialState = this.state.slice();
        this.mlp = new FlexibleMLP(layerSizes, activation, Number(weightScale), Number(bias), this.initialSeed);
    }

    createRandomState(seed) {
        const rng = mulberry32(normalizeSeed(seed) ?? randomSeed());
        const state = new Float32Array(this.gridSize * this.gridSize);
        for (let i = 0; i < state.length; i++) state[i] = rng();
        return state;
    }

    getNeighborhood(r, c) {
        const neighbors = new Float64Array(9);
        let index = 0;
        for (let dr = -1; dr <= 1; dr++) {
            const rr = (r + dr + this.gridSize) % this.gridSize;
            for (let dc = -1; dc <= 1; dc++) {
                const cc = (c + dc + this.gridSize) % this.gridSize;
                neighbors[index++] = this.state[rr * this.gridSize + cc];
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
        this.initialSeed = normalizeSeed(seed) ?? randomSeed();
        this.state = this.createRandomState(this.initialSeed);
        this.initialState = this.state.slice();
        this.history = [];
    }

    setInitialState(state, seed = null) {
        if (state.length !== this.gridSize * this.gridSize) throw new Error('Initial state has the wrong size.');
        this.state = Float32Array.from(state);
        this.initialState = this.state.slice();
        this.initialSeed = normalizeSeed(seed);
        this.history = [];
    }

    restartGrid() {
        this.state = this.initialState.slice();
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
    return Array.from({ length: gridSize }, (_, row) =>
        Array.from({ length: gridSize }, (_, col) => interpolateColor(stops, state[row * gridSize + col]))
    );
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

    withGrid(payload = {}) {
        return { ...payload, grid_colors: this.colors(), grid_values: this.nca.state };
    }

    getConfig() {
        return {
            presets: PRESETS,
            available_activations: AVAILABLE_ACTIVATIONS,
            available_colormaps: AVAILABLE_COLORMAPS,
            default_params: this.nca.getCurrentParams(),
            current_colormap: this.currentColormapName,
            initial_grid_colors: this.colors(),
            initial_grid_values: this.nca.state,
            mlp_params_for_viz: this.nca.mlp.getParamsForViz(),
            is_paused: this.nca.paused,
            constraints: CONSTRAINTS
        };
    }

    step() {
        this.nca.step();
        return this.withGrid();
    }

    stepBack() {
        this.nca.stepBack();
        return this.withGrid({ is_paused: this.nca.paused });
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
            this.nca.mlp = new FlexibleMLP(layerSizes, activation, weightScale, bias, null);
            this.nca.history = [];
            message = 'Settings applied: Custom MLP parameters.';
        }

        return this.withGrid({
            message,
            mlp_params_for_viz: this.nca.mlp.getParamsForViz(),
            current_params: this.nca.getCurrentParams(),
            is_paused: this.nca.paused
        });
    }

    setColormap(colormapName) {
        if (!AVAILABLE_COLORMAPS.includes(colormapName)) throw new Error(`Invalid colormap name: ${colormapName}`);
        this.currentColormapName = colormapName;
        return this.withGrid({ message: `Colormap set to ${colormapName}.` });
    }

    randomizeWeights(data = {}) {
        const current = this.nca.getCurrentParams();
        const layerSizes = parseLayerSizes(data.layer_sizes, current.layer_sizes);
        const activation = data.activation ?? current.activation;
        if (!AVAILABLE_ACTIVATIONS.includes(activation)) throw new Error(`Invalid activation: ${activation}`);
        const weightScale = Number(data.weight_scale ?? current.weight_scale);
        const bias = Number(data.bias ?? current.bias);
        if (!Number.isFinite(weightScale) || !Number.isFinite(bias)) throw new Error('Weight scale and bias must be finite numbers.');
        this.nca.mlp = new FlexibleMLP(layerSizes, activation, weightScale, bias, null);
        this.nca.history = [];
        return this.withGrid({
            message: 'NCA weights randomized.',
            mlp_params_for_viz: this.nca.mlp.getParamsForViz(),
            current_params: this.nca.getCurrentParams(),
            is_paused: this.nca.paused
        });
    }

    randomizeGrid(data = {}) {
        this.nca.resetGrid(data.seed);
        return this.withGrid({
            message: 'NCA grid randomized.',
            is_paused: this.nca.paused
        });
    }

    randomizeArchitecture(data = {}) {
        const hiddenCount = MIN_RANDOM_LAYERS + Math.floor(Math.random() * (MAX_RANDOM_LAYERS - MIN_RANDOM_LAYERS + 1));
        const layerSizes = [9];
        for (let i = 0; i < hiddenCount; i++) {
            layerSizes.push(MIN_RANDOM_NODES + Math.floor(Math.random() * (MAX_RANDOM_NODES - MIN_RANDOM_NODES + 1)));
        }
        layerSizes.push(1);

        const activation = AVAILABLE_ACTIVATIONS[Math.floor(Math.random() * AVAILABLE_ACTIVATIONS.length)];
        const weightScale = Math.round((0.5 + Math.random() * 2) * 10) / 10;
        const bias = Math.round((-0.5 + Math.random()) * 10) / 10;
        this.nca = new NeuralCellularAutomaton({
            gridSize: this.nca.gridSize,
            layerSizes,
            activation,
            weightScale,
            bias,
            seed: null
        });
        this.nca.paused = !Boolean(data.was_running);

        return this.withGrid({
            message: 'NCA architecture randomized and reinitialized.',
            mlp_params_for_viz: this.nca.mlp.getParamsForViz(),
            current_params: this.nca.getCurrentParams(),
            is_paused: this.nca.paused
        });
    }

    restart() {
        this.nca.restartGrid();
        this.nca.paused = false;
        return {
            message: 'NCA restarted from its initial grid with the current network.',
            initial_grid_colors: this.colors(),
            initial_grid_values: this.nca.state,
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
            cell_value: this.nca.state[row * this.nca.gridSize + col],
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
        this.nca.setInitialState(next, null);
        this.nca.paused = !Boolean(data.was_running);
        return this.withGrid({
            message: 'Grid state updated successfully.',
            is_paused: this.nca.paused
        });
    }
}

export const browserNcaService = new BrowserNCAService();
