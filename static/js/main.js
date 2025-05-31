// static/js/main.js
document.addEventListener('DOMContentLoaded', () => {
    const ncaCanvas = document.getElementById('ncaCanvas');
    const ncaCtx = ncaCanvas.getContext('2d');
    const networkCanvas = document.getElementById('networkCanvas');
    const networkCtx = networkCanvas.getContext('2d');

    const toggleRunButton = document.getElementById('toggleRunButton');
    const stepButton = document.getElementById('stepButton');
    const stepBackButton = document.getElementById('stepBackButton');
    const presetSelector = document.getElementById('presetSelector');
    const colormapSelector = document.getElementById('colormapSelector');
    
    const layerBuilderContainer = document.getElementById('layerBuilderContainer');
    const addHiddenLayerButton = document.getElementById('addHiddenLayerButton');
    const removeHiddenLayerButton = document.getElementById('removeHiddenLayerButton');
    
    const activationSelector = document.getElementById('activationSelector');
    const weightScaleSlider = document.getElementById('weightScaleSlider');
    const weightScaleValue = document.getElementById('weightScaleValue');
    const biasSlider = document.getElementById('biasSlider');
    const biasValue = document.getElementById('biasValue');
    const applySettingsButton = document.getElementById('applySettingsButton');
    
    const randomizeWeightsButton = document.getElementById('randomizeWeightsButton');
    const randomizeGridButton = document.getElementById('randomizeGridButton');
    const speedSlider = document.getElementById('speedSlider');
    const speedValue = document.getElementById('speedValue');

    // Manual Weight Editor Elements
    const manualWeightLayerSelector = document.getElementById('manualWeightLayerSelector');
    const manualWeightNeuronSelector = document.getElementById('manualWeightNeuronSelector');
    const manualWeightPresetSelector = document.getElementById('manualWeightPresetSelector');
    const manualWeightInputContainer = document.getElementById('manualWeightInputContainer');
    const manualWeightInputContainerTitle = document.getElementById('manualWeightInputContainerTitle');
    const applyManualWeightsButton = document.getElementById('applyManualWeightsButton');
    const manualWeightInfoText = document.getElementById('manualWeightInfoText');


    const cellInfoLabel = document.getElementById('cellInfoLabel');
    const neighborhoodDisplay = document.getElementById('neighborhoodDisplay');
    const activationDisplay = document.getElementById('activationDisplay');
    const networkLegend = document.getElementById('networkLegend');
    const clearSelectionButton = document.getElementById('clearSelectionButton');
    const hoverCellInfo = document.createElement('div');
    hoverCellInfo.style.cssText = `
        position: absolute; background: rgba(0, 0, 0, 0.7); color: white;
        padding: 5px; border-radius: 3px; pointer-events: none; display: none; z-index: 100;`;
    document.body.appendChild(hoverCellInfo);
    
    const CELL_SIZE = 10; // Adjusted for better visibility on smaller screens
    let gridSize = 50; 
    let isRunning = false;
    let animationIntervalId = null;
    let currentSpeed = 200;
    let currentGridColors = null; 
    let hiddenLayerSizes = []; // Stores sizes of hidden layers from UI e.g. [8,16]

    // Constraints from backend
    let MAX_HIDDEN_LAYERS_COUNT_FROM_BACKEND = 3;
    let MIN_NODE_COUNT_PER_LAYER_FROM_BACKEND = 1;
    let MAX_NODE_COUNT_PER_LAYER_FROM_BACKEND = 32;


    let mlpParamsForViz = null; 
    let selectedCell = null; 
    let currentLayerActivations = null;
    let selectedNeuronForEditing = { layer: null, neuron: null }; // For manual weight editor


    async function fetchApi(endpoint, method = 'GET', body = null) {
        // Simple loading indicator (optional)
        // document.body.style.cursor = 'wait'; 
        const options = { method };
        if (body) {
            options.headers = { 'Content-Type': 'application/json' };
            options.body = JSON.stringify(body);
        }
        try {
            const response = await fetch(endpoint, options);
            // document.body.style.cursor = 'default';
            if (!response.ok) {
                const errorData = await response.json();
                console.error(`API Error (${response.status}) for ${endpoint}:`, errorData.error || response.statusText);
                alert(`Error: ${errorData.error || response.statusText}`);
                return null;
            }
            return await response.json();
        } catch (error) {
            // document.body.style.cursor = 'default';
            console.error('Network or API call failed:', error);
            alert(`Network error: ${error.message}`);
            return null;
        }
    }

    function drawNcaGrid(gridColors) {
        if (!gridColors || gridColors.length === 0) return;
        currentGridColors = gridColors; // Keep a reference
        gridSize = gridColors.length; 
        ncaCanvas.width = gridSize * CELL_SIZE;
        ncaCanvas.height = gridSize * CELL_SIZE;

        for (let r = 0; r < gridSize; r++) {
            for (let c = 0; c < gridSize; c++) {
                ncaCtx.fillStyle = gridColors[r][c];
                ncaCtx.fillRect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE);
            }
        }
        if (selectedCell) {
            highlightNeighborhood(selectedCell.r, selectedCell.c);
        }
    }
    
    function updateUiControls(params, fromPreset = false) {
        if (params.layer_sizes) {
            hiddenLayerSizes = params.layer_sizes.slice(1, -1); // [9, HL1, HL2, 1] -> [HL1, HL2]
            renderLayerBuilder();
        }
        if (params.activation) activationSelector.value = params.activation;
        if (params.weight_scale !== undefined) {
            weightScaleSlider.value = params.weight_scale;
            weightScaleValue.textContent = parseFloat(params.weight_scale).toFixed(1);
        }
        if (params.bias !== undefined) {
            biasSlider.value = params.bias;
            biasValue.textContent = parseFloat(params.bias).toFixed(1);
        }
        
        if (fromPreset) { // If called after selecting a preset (not "Custom")
             applySettingsButton.disabled = true; // Presets are applied immediately (or considered "applied")
        } else {
            // For other changes, let the individual event listeners handle the button state.
        }
        // Reset manual weight editor if layer structure might have changed
        populateManualWeightLayerSelector();
        resetManualWeightEditorUI();
    }

    async function loadInitialConfig() {
        const config = await fetchApi('/api/config');
        if (!config) return;

        // Store constraints
        MAX_HIDDEN_LAYERS_COUNT_FROM_BACKEND = config.constraints.max_hidden_layers;
        MIN_NODE_COUNT_PER_LAYER_FROM_BACKEND = config.constraints.min_node_size;
        MAX_NODE_COUNT_PER_LAYER_FROM_BACKEND = config.constraints.max_node_size;


        gridSize = config.default_params.grid_size;
        mlpParamsForViz = config.mlp_params_for_viz;
        
        Object.keys(config.presets).forEach(name => {
            presetSelector.add(new Option(name, name));
        });
        config.available_activations.forEach(name => {
            activationSelector.add(new Option(name, name));
        });
        config.available_colormaps.forEach(name => {
            colormapSelector.add(new Option(name, name));
        });
        colormapSelector.value = config.current_colormap;

        const initialPresetName = "Flicker"; // Start with Flicker
        presetSelector.value = initialPresetName;
        const initialPresetData = config.presets[initialPresetName];
        if (initialPresetData) {
            const [_seed, layers, act, w_scale, b_val] = initialPresetData;
            updateUiControls({ layer_sizes: layers, activation: act, weight_scale: w_scale, bias: b_val }, true);
        } else { 
             updateUiControls(config.default_params, true);
        }

        drawNcaGrid(config.initial_grid_colors);
        buildNetworkViz(); 
        updateNetworkLegend();

        if (config.is_paused) {
            toggleRunButton.textContent = 'Start';
            toggleRunButton.classList.remove('running');
            isRunning = false;
        } else {
            toggleRunButton.textContent = 'Stop';
            toggleRunButton.classList.add('running');
            isRunning = true;
            startAnimationLoop();
        }
        applySettingsButton.disabled = true; 
        applyManualWeightsButton.disabled = true;
        populateManualWeightLayerSelector(); // Populate after mlpParamsForViz is set
    }

    async function handleStep(isBack = false) {
        const endpoint = isBack ? '/api/step_back' : '/api/step';
        const data = await fetchApi(endpoint, 'POST');
        if (data) {
            drawNcaGrid(data.grid_colors);
            if (selectedCell) updateCellDetails(selectedCell.r, selectedCell.c);
            if (isBack && data.is_paused !== undefined && data.is_paused && isRunning) {
                // If stepping back paused the simulation on server, update client
                isRunning = false;
                toggleRunButton.textContent = 'Start';
                toggleRunButton.classList.remove('running');
                if (animationIntervalId) clearInterval(animationIntervalId);
            }
        }
    }

    function startAnimationLoop() {
        if (animationIntervalId) clearInterval(animationIntervalId);
        animationIntervalId = setInterval(async () => {
            if (isRunning) {
                await handleStep(false);
            }
        }, currentSpeed);
    }

    toggleRunButton.addEventListener('click', async () => {
        const data = await fetchApi('/api/toggle_pause', 'POST');
        if (data) {
            isRunning = !data.is_paused;
            if (isRunning) {
                toggleRunButton.textContent = 'Stop';
                toggleRunButton.classList.add('running');
                startAnimationLoop();
            } else {
                toggleRunButton.textContent = 'Start';
                toggleRunButton.classList.remove('running');
                if (animationIntervalId) clearInterval(animationIntervalId);
            }
        }
    });

    stepButton.addEventListener('click', () => handleStep(false));
    stepBackButton.addEventListener('click', () => handleStep(true));

    // --- General Settings Change Listeners ---
    function onGeneralParamChange() {
        applySettingsButton.disabled = false;
        presetSelector.value = "Custom"; // Any manual change makes it custom
    }
    colormapSelector.addEventListener('change', onGeneralParamChange);
    activationSelector.addEventListener('change', onGeneralParamChange);
    weightScaleSlider.addEventListener('input', onGeneralParamChange);
    biasSlider.addEventListener('input', onGeneralParamChange);
    // Layer builder changes also call onGeneralParamChange indirectly or directly.

    presetSelector.addEventListener('change', async () => {
        const selectedPresetName = presetSelector.value;
        if (selectedPresetName !== "Custom") {
            const config = await fetchApi('/api/config'); // Fetch fresh presets info
            if (config && config.presets[selectedPresetName]) {
                const [_seed, layers, act, w_scale, b_val] = config.presets[selectedPresetName];
                updateUiControls({layer_sizes: layers, activation: act, weight_scale: w_scale, bias: b_val}, true);
                applySettingsButton.disabled = false; // Allow applying the chosen preset
            }
        } else {
             applySettingsButton.disabled = false; // If user explicitly selects "Custom"
        }
    });


    applySettingsButton.addEventListener('click', async () => {
        const finalLayerSizes = [9, ...hiddenLayerSizes, 1];
        // Client-side validation mirroring backend
        try {
            if (finalLayerSizes[0] !== 9 || finalLayerSizes[finalLayerSizes.length -1] !== 1) throw new Error("Layers must start with 9 and end with 1.");
            const hl = finalLayerSizes.slice(1, -1);
            if (hl.length > MAX_HIDDEN_LAYERS_COUNT_FROM_BACKEND) throw new Error(`Max ${MAX_HIDDEN_LAYERS_COUNT_FROM_BACKEND} hidden layers.`);
            for(const size of hl) {
                if (size < MIN_NODE_COUNT_PER_LAYER_FROM_BACKEND || size > MAX_NODE_COUNT_PER_LAYER_FROM_BACKEND) {
                    throw new Error(`Hidden layer size out of range (${MIN_NODE_COUNT_PER_LAYER_FROM_BACKEND}-${MAX_NODE_COUNT_PER_LAYER_FROM_BACKEND}).`);
                }
            }
        } catch (e) {
            alert(`Input Error: ${e.message}`);
            return;
        }

        const params = {
            preset_name: presetSelector.value, // Send "Custom" if it is, or actual preset name
            colormap_name: colormapSelector.value,
            layer_sizes: finalLayerSizes.join(','),
            activation: activationSelector.value,
            weight_scale: parseFloat(weightScaleSlider.value),
            bias: parseFloat(biasSlider.value)
        };
        const data = await fetchApi('/api/apply_settings', 'POST', params);
        if (data) {
            drawNcaGrid(data.grid_colors);
            mlpParamsForViz = data.mlp_params_for_viz; 
            updateUiControls(data.current_params, data.message.includes("Preset")); // Pass true if preset was applied
            buildNetworkViz(); 
            updateNetworkLegend();
            if (selectedCell) updateCellDetails(selectedCell.r, selectedCell.c); 
            applySettingsButton.disabled = true; 
            if (data.is_paused && isRunning) { // If server paused it
                isRunning = false;
                toggleRunButton.textContent = 'Start';
                toggleRunButton.classList.remove('running');
                if (animationIntervalId) clearInterval(animationIntervalId);
            }
             populateManualWeightLayerSelector(); // Crucial: update after potential layer changes
             resetManualWeightEditorUI();
        }
    });

    randomizeWeightsButton.addEventListener('click', async () => {
        const currentLayers = mlpParamsForViz ? mlpParamsForViz.layer_sizes.join(',') : [9, ...hiddenLayerSizes, 1].join(',');
        const params = {
            layer_sizes: currentLayers, // Use current structure
            activation: activationSelector.value,
            weight_scale: parseFloat(weightScaleSlider.value),
            bias: parseFloat(biasSlider.value)
        };
        const data = await fetchApi('/api/randomize_weights', 'POST', params);
        if (data) {
            mlpParamsForViz = data.mlp_params_for_viz; 
            updateUiControls(data.current_params); // Update if backend changed something
            buildNetworkViz(); 
            updateNetworkLegend();
            if (selectedCell) updateCellDetails(selectedCell.r, selectedCell.c);
            presetSelector.value = "Custom"; // Randomizing weights implies custom setup
            applySettingsButton.disabled = true; // Settings are now "applied" by randomization
            resetManualWeightEditorUI(); // Weights changed, so refresh editor
        }
    });

    randomizeGridButton.addEventListener('click', async () => {
        const data = await fetchApi('/api/randomize_grid', 'POST', { seed: Date.now() });
        if (data) {
            drawNcaGrid(data.grid_colors);
            if (selectedCell) updateCellDetails(selectedCell.r, selectedCell.c);
            if (data.is_paused && isRunning) {
                 isRunning = false;
                toggleRunButton.textContent = 'Start';
                toggleRunButton.classList.remove('running');
                if (animationIntervalId) clearInterval(animationIntervalId);
            }
        }
    });

    weightScaleSlider.addEventListener('input', (e) => weightScaleValue.textContent = parseFloat(e.target.value).toFixed(1));
    biasSlider.addEventListener('input', (e) => biasValue.textContent = parseFloat(e.target.value).toFixed(1));
    speedSlider.addEventListener('input', (e) => {
        currentSpeed = parseInt(e.target.value);
        speedValue.textContent = currentSpeed;
        if (isRunning) startAnimationLoop();
    });
    currentSpeed = parseInt(speedSlider.value); 

    // --- Layer Builder ---
    function renderLayerBuilder() {
        layerBuilderContainer.innerHTML = ''; 
        hiddenLayerSizes.forEach((size, index) => {
            const layerDiv = document.createElement('div');
            layerDiv.classList.add('layer-input-group');
            const inputId = `hiddenLayer${index}`;
            layerDiv.innerHTML = `
                <label for="${inputId}">Hidden Layer ${index + 1} Size:</label>
                <input type="number" id="${inputId}" value="${size}"
                       min="${MIN_NODE_COUNT_PER_LAYER_FROM_BACKEND}" max="${MAX_NODE_COUNT_PER_LAYER_FROM_BACKEND}" class="hidden-layer-input">`;
            layerBuilderContainer.appendChild(layerDiv);
            layerDiv.querySelector(`#${inputId}`).addEventListener('input', (e) => {
                let value = parseInt(e.target.value);
                if (isNaN(value)) value = MIN_NODE_COUNT_PER_LAYER_FROM_BACKEND;
                value = Math.max(MIN_NODE_COUNT_PER_LAYER_FROM_BACKEND, Math.min(value, MAX_NODE_COUNT_PER_LAYER_FROM_BACKEND));
                e.target.value = value; 
                hiddenLayerSizes[index] = value;
                onGeneralParamChange();
            });
        });
        addHiddenLayerButton.disabled = hiddenLayerSizes.length >= MAX_HIDDEN_LAYERS_COUNT_FROM_BACKEND;
        removeHiddenLayerButton.disabled = hiddenLayerSizes.length === 0;
    }

    addHiddenLayerButton.addEventListener('click', () => {
        if (hiddenLayerSizes.length < MAX_HIDDEN_LAYERS_COUNT_FROM_BACKEND) {
            hiddenLayerSizes.push(8); 
            renderLayerBuilder();
            onGeneralParamChange();
        }
    });
    removeHiddenLayerButton.addEventListener('click', () => {
        if (hiddenLayerSizes.length > 0) {
            hiddenLayerSizes.pop();
            renderLayerBuilder();
            onGeneralParamChange();
        }
    });

    // --- Manual Neuron Weight Editor Logic ---
    function populateManualWeightLayerSelector() {
        manualWeightLayerSelector.innerHTML = '<option value="">- Select Layer -</option>';
        manualWeightNeuronSelector.innerHTML = '<option value="">- Select Neuron -</option>'; // Reset neuron selector
        manualWeightInputContainer.innerHTML = ''; // Clear weight inputs
        manualWeightInputContainerTitle.textContent = '';
        applyManualWeightsButton.disabled = true;

        if (!mlpParamsForViz || !mlpParamsForViz.layer_sizes || mlpParamsForViz.layer_sizes.length < 2) return;

        // Layers are 1-indexed for user, but 0-indexed for mlp.W (W[0] is weights TO first hidden layer)
        // So, if mlp.layer_sizes = [9, 8, 1], mlp.W has 2 elements.
        // W[0] are weights to layer 1 (size 8). W[1] are weights to layer 2 (size 1).
        // User sees "Layer 1" (meaning the first hidden layer) and "Layer 2" (output layer).
        const numMLPLayers = mlpParamsForViz.layer_sizes.length; // Total layers including input/output
        for (let i = 1; i < numMLPLayers; i++) { // Iterate from first hidden layer to output layer
            const option = document.createElement('option');
            option.value = i; // Store 1-indexed actual layer number
            option.textContent = `Layer ${i} (Size: ${mlpParamsForViz.layer_sizes[i]})`;
            if (i === numMLPLayers - 1) option.textContent += " - Output";
            else option.textContent += " - Hidden";
            manualWeightLayerSelector.add(option);
        }
    }

    manualWeightLayerSelector.addEventListener('change', () => {
        populateManualWeightNeuronSelector();
        manualWeightInputContainer.innerHTML = '';
        manualWeightInputContainerTitle.textContent = '';
        applyManualWeightsButton.disabled = true;
        selectedNeuronForEditing.layer = manualWeightLayerSelector.value ? parseInt(manualWeightLayerSelector.value) : null;
        selectedNeuronForEditing.neuron = null;
        buildNetworkViz(); // Redraw to clear old neuron highlight
    });

    function populateManualWeightNeuronSelector() {
        manualWeightNeuronSelector.innerHTML = '<option value="">- Select Neuron -</option>';
        const selectedLayerDisplayIdx = parseInt(manualWeightLayerSelector.value); // 1-indexed from UI

        if (isNaN(selectedLayerDisplayIdx) || !mlpParamsForViz) return;
        
        // Example: layer_sizes = [9, 8, 16, 1].
        // UI Layer 1 (Hidden) -> mlpParamsForViz.layer_sizes[1] = 8 neurons
        // UI Layer 2 (Hidden) -> mlpParamsForViz.layer_sizes[2] = 16 neurons
        // UI Layer 3 (Output) -> mlpParamsForViz.layer_sizes[3] = 1 neuron
        const numNeuronsInSelectedLayer = mlpParamsForViz.layer_sizes[selectedLayerDisplayIdx];

        if (numNeuronsInSelectedLayer > 0) {
            manualWeightNeuronSelector.add(new Option("All Neurons in Layer", "all"));
            for (let i = 0; i < numNeuronsInSelectedLayer; i++) {
                manualWeightNeuronSelector.add(new Option(`Neuron ${i + 1}`, i)); // 0-indexed value
            }
        }
    }
    
    manualWeightNeuronSelector.addEventListener('change', async () => {
        const layerDisplayIdx = parseInt(manualWeightLayerSelector.value); // 1-indexed layer from UI
        const neuronVal = manualWeightNeuronSelector.value; // Can be "all" or 0-indexed neuron

        selectedNeuronForEditing.layer = layerDisplayIdx;
        selectedNeuronForEditing.neuron = (neuronVal === "all" || neuronVal === "") ? neuronVal : parseInt(neuronVal);
        
        buildNetworkViz(); // Redraw to highlight selected neuron or clear highlight

        if (layerDisplayIdx && neuronVal !== "") {
            if (neuronVal === "all") {
                manualWeightInputContainer.innerHTML = '<p>Apply a preset pattern to all neurons in this layer.</p>';
                manualWeightInputContainerTitle.textContent = `Editing Incoming Weights for: All Neurons in Layer ${layerDisplayIdx}`;
                applyManualWeightsButton.disabled = manualWeightPresetSelector.value === ""; // Enable if a preset is chosen
            } else {
                const neuronIdx = parseInt(neuronVal); // 0-indexed
                // layer_idx for API is 0-indexed from perspective of W matrix.
                // W[0] are weights to layer_sizes[1]. So, weights TO UI Layer L are in W[L-1].
                const apiLayerIdx = layerDisplayIdx - 1; 
                const data = await fetchApi(`/api/neuron_weights?layer_idx=${apiLayerIdx}&neuron_idx=${neuronIdx}`);
                if (data && data.weights) {
                    renderManualWeightInputs(data.weights, layerDisplayIdx, neuronIdx);
                }
                applyManualWeightsButton.disabled = false; // Enable for single neuron editing
            }
        } else {
            manualWeightInputContainer.innerHTML = '';
            manualWeightInputContainerTitle.textContent = '';
            applyManualWeightsButton.disabled = true;
        }
    });
    
    manualWeightPresetSelector.addEventListener('change', () => {
        const layerDisplayIdx = parseInt(manualWeightLayerSelector.value);
        const neuronVal = manualWeightNeuronSelector.value;
        const preset = manualWeightPresetSelector.value;

        if (!layerDisplayIdx || neuronVal === "" || preset === "") {
             if (neuronVal === "all" && preset === "") applyManualWeightsButton.disabled = true;
            return;
        }
        
        // Determine the size of the previous layer (number of incoming weights)
        // Weights to UI Layer L (layer_sizes[L]) come from UI Layer L-1 (layer_sizes[L-1])
        const prevLayerSize = mlpParamsForViz.layer_sizes[layerDisplayIdx - 1];
        let pattern = generateWeightPattern(preset, prevLayerSize);

        if (neuronVal === "all") {
            // For "all neurons", we don't populate inputs, just enable apply button
            applyManualWeightsButton.disabled = false;
            manualWeightInputContainer.innerHTML = `<p>Pattern '${preset}' selected. Click "Apply Manual Weights" to affect all neurons in Layer ${layerDisplayIdx}.</p>`;

        } else if (pattern) { // Single neuron and valid pattern
            renderManualWeightInputs(pattern, layerDisplayIdx, parseInt(neuronVal));
            applyManualWeightsButton.disabled = false;
        }
    });

    function renderManualWeightInputs(weights, layerDisplayIdx, neuronDisplayIdx) {
        manualWeightInputContainer.innerHTML = ''; // Clear previous
        manualWeightInputContainerTitle.textContent = `Editing Incoming Weights for: Layer ${layerDisplayIdx}, Neuron ${neuronDisplayIdx + 1}`;
        
        const prevLayerSize = mlpParamsForViz.layer_sizes[layerDisplayIdx - 1];
        manualWeightInfoText.textContent = `Editing ${weights.length} incoming weights from Layer ${layerDisplayIdx -1 } (Size: ${prevLayerSize}) to Layer ${layerDisplayIdx}, Neuron ${neuronDisplayIdx + 1}.`;


        weights.forEach((weight, index) => {
            const group = document.createElement('div');
            group.classList.add('weight-input-group');
            const label = document.createElement('label');
            label.textContent = `W_${index + 1}`; // From Neuron 'index+1' in prev layer
            label.title = `Weight from Neuron ${index+1} of previous layer`;
            
            const input = document.createElement('input');
            input.type = 'number';
            input.step = '0.01';
            input.value = parseFloat(weight).toFixed(3);
            input.classList.add('manual-weight-value');
            input.dataset.index = index;

            input.addEventListener('change', () => { applyManualWeightsButton.disabled = false; });

            group.appendChild(label);
            group.appendChild(input);
            manualWeightInputContainer.appendChild(group);
        });
    }

    function generateWeightPattern(patternName, numWeights) {
        let pattern = new Array(numWeights).fill(0.0); // Default to zeros
        if (patternName === "zeros") {
            // Already filled with zeros
        } else if (patternName === "ones") {
            pattern = new Array(numWeights).fill(1.0);
        } else if (patternName === "random_small") {
            for (let i = 0; i < numWeights; i++) {
                pattern[i] = parseFloat((Math.random() * 0.2 - 0.1).toFixed(3)); // Small random values ~[-0.1, 0.1]
            }
        } else if (numWeights === 9) { // 3x3 specific patterns
            // Assuming weights map to a 3x3 grid (row by row)
            //  0 1 2
            //  3 4 5
            //  6 7 8
            if (patternName === "center_on") pattern = [0,0,0, 0,1,0, 0,0,0];
            else if (patternName === "vertical_lines") pattern = [0,1,0, 0,1,0, 0,1,0];
            else if (patternName === "horizontal_lines") pattern = [0,0,0, 1,1,1, 0,0,0];
            else if (patternName === "diag_tl_br") pattern = [1,0,0, 0,1,0, 0,0,1];
            else if (patternName === "diag_tr_bl") pattern = [0,0,1, 0,1,0, 1,0,0];
        } else {
            // For other numWeights, presets might not map well visually, just use random/zeros/ones
            console.warn(`Preset ${patternName} is not specifically designed for ${numWeights} inputs. Using default or simpler pattern.`);
            if (patternName.includes("3x3")) return null; // Don't apply 3x3 specific to non-9 inputs
        }
        return pattern;
    }
    
    applyManualWeightsButton.addEventListener('click', async () => {
        const layerDisplayIdx = parseInt(manualWeightLayerSelector.value); // 1-indexed
        const neuronVal = manualWeightNeuronSelector.value; // "all" or 0-indexed string
        const presetName = manualWeightPresetSelector.value;

        if (isNaN(layerDisplayIdx) || neuronVal === "") {
            alert("Please select a layer and a neuron (or 'All Neurons').");
            return;
        }

        let weightsToApply;
        if (neuronVal === "all") {
            if (presetName === "") {
                alert("Please select a preset pattern to apply to all neurons.");
                return;
            }
            const prevLayerSize = mlpParamsForViz.layer_sizes[layerDisplayIdx - 1];
            weightsToApply = generateWeightPattern(presetName, prevLayerSize);
            if (!weightsToApply) {
                alert(`Could not generate pattern '${presetName}' for an input size of ${prevLayerSize}.`);
                return;
            }
        } else { // Single neuron
            weightsToApply = Array.from(manualWeightInputContainer.querySelectorAll('.manual-weight-value'))
                                 .map(input => parseFloat(input.value));
            if (weightsToApply.some(isNaN)) {
                alert("Invalid number in manual weight inputs.");
                return;
            }
        }
        
        // API layer_idx is 0-indexed from W matrix perspective (W[0] are weights TO layer_sizes[1])
        const apiLayerIdx = layerDisplayIdx - 1; 

        const payload = {
            layer_idx: apiLayerIdx,
            neuron_idx: neuronVal, // Send "all" or the 0-indexed neuron string
            weights_pattern: weightsToApply // Backend will handle applying this to one or all
        };

        const data = await fetchApi('/api/neuron_weights', 'POST', payload);
        if (data) {
            mlpParamsForViz = data.mlp_params_for_viz; // Update with new weights
            buildNetworkViz(); // Redraw network
            updateNetworkLegend();
            if (selectedCell) updateCellDetails(selectedCell.r, selectedCell.c); // Refresh cell details
            alert(data.message || "Weights applied.");
            applyManualWeightsButton.disabled = true;
            manualWeightPresetSelector.value = ""; // Reset preset selector
            
            // If a single neuron was edited and successful, re-fetch and re-render its weights
            if (neuronVal !== "all") {
                 const updatedWeightsData = await fetchApi(`/api/neuron_weights?layer_idx=${apiLayerIdx}&neuron_idx=${parseInt(neuronVal)}`);
                 if (updatedWeightsData && updatedWeightsData.weights) {
                    renderManualWeightInputs(updatedWeightsData.weights, layerDisplayIdx, parseInt(neuronVal));
                 }
            } else {
                // For "all", clear the specific message for preset application
                manualWeightInputContainer.innerHTML = '<p>Preset applied to all neurons. Select a single neuron to see/edit its new weights.</p>';
            }
        }
    });

    function resetManualWeightEditorUI() {
        manualWeightLayerSelector.value = "";
        manualWeightNeuronSelector.innerHTML = '<option value="">- Select Neuron -</option>';
        manualWeightPresetSelector.value = "";
        manualWeightInputContainer.innerHTML = "";
        manualWeightInputContainerTitle.textContent = "";
        applyManualWeightsButton.disabled = true;
        selectedNeuronForEditing = { layer: null, neuron: null };
        buildNetworkViz(); // Redraw to remove any neuron highlight
    }


    // --- Network Visualization ---
    let netNodePositions = []; 
    
    function weightToColor(w) { 
        const maxVal = 3.0;
        const wClamped = Math.max(-maxVal, Math.min(w, maxVal));
        const norm = (wClamped + maxVal) / (2 * maxVal); 
        const r = Math.floor((1.0 - norm) * 255);
        const g = Math.floor(norm * 255);
        return `rgb(${r},${g},0)`;
    }
    function weightToThickness(w) {
        const maxWidth = 3.0;
        return 1 + Math.min(Math.abs(w), 3.0) / 3.0 * (maxWidth - 1);
    }
    function activationToColor(a) {
        const aClamped = Math.max(0.0, Math.min(a, 1.5)); 
        const b = Math.floor((aClamped / 1.5) * 255);
        const r_g = 255 - b; 
        return `rgb(${r_g},${r_g},255)`;
    }

    function buildNetworkViz() {
        if (!mlpParamsForViz || !mlpParamsForViz.layer_sizes) {
            networkCtx.clearRect(0, 0, networkCanvas.width, networkCanvas.height);
            return;
        }

        const layerSizes = mlpParamsForViz.layer_sizes;
        const weights = mlpParamsForViz.weights; 
        
        networkCtx.clearRect(0, 0, networkCanvas.width, networkCanvas.height);
        netNodePositions = [];

        const width = networkCanvas.width;
        const height = networkCanvas.height;
        const layerCount = layerSizes.length;
        const xSpacing = width / (layerCount + 1);
        
        // Calculate dynamic node radius based on number of nodes in largest layer & canvas height
        const maxNodesInAnyLayer = Math.max(...layerSizes, 1); // Avoid division by zero if layerSizes is empty
        const baseRadius = 8;
        const nodeRadius = Math.max(3, Math.min(baseRadius, height / (maxNodesInAnyLayer + 2) / 2.2));


        for (let i = 0; i < layerCount; i++) {
            const numNodesInLayer = layerSizes[i];
            const currentX = (i + 1) * xSpacing;
            const layerNodePos = [];
            const ySpacing = height / (numNodesInLayer + 1);
            for (let n = 0; n < numNodesInLayer; n++) {
                const currentY = (n + 1) * ySpacing;
                layerNodePos.push({ x: currentX, y: currentY });
            }
            netNodePositions.push(layerNodePos);
        }

        if (weights) {
            for (let i = 0; i < weights.length; i++) { 
                const W_i = weights[i]; 
                const prevLayerNodes = netNodePositions[i];
                const nextLayerNodes = netNodePositions[i+1];
                for (let inIdx = 0; inIdx < W_i.length; inIdx++) {
                    for (let outIdx = 0; outIdx < W_i[inIdx].length; outIdx++) {
                        const wVal = W_i[inIdx][outIdx];
                        networkCtx.beginPath();
                        networkCtx.moveTo(prevLayerNodes[inIdx].x, prevLayerNodes[inIdx].y);
                        networkCtx.lineTo(nextLayerNodes[outIdx].x, nextLayerNodes[outIdx].y);
                        networkCtx.strokeStyle = weightToColor(wVal);
                        networkCtx.lineWidth = weightToThickness(wVal);
                        networkCtx.stroke();
                    }
                }
            }
        }
        
        for (let i = 0; i < netNodePositions.length; i++) { // Iterate through layers (0 to N)
            for (let j = 0; j < netNodePositions[i].length; j++) { // Iterate through neurons in layer i
                const pos = netNodePositions[i][j];
                networkCtx.beginPath();
                networkCtx.arc(pos.x, pos.y, nodeRadius, 0, 2 * Math.PI);
                networkCtx.fillStyle = "white"; 
                networkCtx.fill();
                networkCtx.strokeStyle = "black";
                networkCtx.lineWidth = 1;

                // Highlight selected neuron for editing
                // selectedNeuronForEditing.layer is 1-indexed (UI layer number)
                // selectedNeuronForEditing.neuron is 0-indexed (neuron index in that UI layer) or "all"
                if (selectedNeuronForEditing.layer === i && // current layer `i` (0-indexed) matches selected UI layer (1-indexed)
                    (selectedNeuronForEditing.neuron === "all" || selectedNeuronForEditing.neuron === j)) {
                    networkCtx.strokeStyle = "magenta"; // Highlight color
                    networkCtx.lineWidth = 2.5;
                }
                networkCtx.stroke();
            }
        }
        if (selectedCell && currentLayerActivations) {
            updateNetworkNodeColors(currentLayerActivations, nodeRadius);
        }
        // updateNetworkLegend(); // Called separately now or after build
    }
    
    function updateNetworkNodeColors(layerActivations, nodeRadius) {
         if (!netNodePositions || netNodePositions.length === 0 || !layerActivations || !mlpParamsForViz) return;
         nodeRadius = nodeRadius || Math.max(3, Math.min(8, networkCanvas.height / (Math.max(...mlpParamsForViz.layer_sizes,1) + 2) / 2.2));


        for (let layerIdx = 0; layerIdx < layerActivations.length; layerIdx++) {
            if (layerIdx < netNodePositions.length) { 
                const activationsInLayer = layerActivations[layerIdx];
                const nodesInLayer = netNodePositions[layerIdx];
                for (let nodeIdx = 0; nodeIdx < activationsInLayer.length; nodeIdx++) {
                    if (nodeIdx < nodesInLayer.length) { 
                        const val = activationsInLayer[nodeIdx];
                        const color = activationToColor(val);
                        const pos = nodesInLayer[nodeIdx];
                        
                        networkCtx.beginPath();
                        networkCtx.arc(pos.x, pos.y, nodeRadius, 0, 2 * Math.PI);
                        networkCtx.fillStyle = color;
                        networkCtx.fill();
                        networkCtx.strokeStyle = "black"; 
                        networkCtx.lineWidth = 1;
                         // Re-apply highlight if this is the selected neuron for editing
                        if (selectedNeuronForEditing.layer === layerIdx &&
                            selectedNeuronForEditing.neuron !== "all" &&
                            selectedNeuronForEditing.neuron === nodeIdx) {
                            networkCtx.strokeStyle = "magenta";
                            networkCtx.lineWidth = 2.5;
                        }
                        networkCtx.stroke();
                    }
                }
            }
        }
    }

    function updateNetworkLegend() {
        if (!networkLegend) return;
        networkLegend.innerHTML = `
            <div><span class="color-box" style="background-color: rgb(255,0,0);"></span> Neg. Weight</div>
            <div><span class="color-box" style="background-color: rgb(0,255,0);"></span> Pos. Weight</div>
            <div><span class="color-box" style="background-color: rgb(255,255,255); border: 1px solid #ccc;"></span> Low Activation</div>
            <div><span class="color-box" style="background-color: rgb(0,0,255);"></span> High Activation</div>
            <div><span class="color-box" style="border: 2.5px solid magenta; background-color: white;"></span> Sel. Neuron (Edit)</div>
        `;
    }
    
    async function updateCellDetails(r, c) {
        const data = await fetchApi(`/api/cell_details?r=${r}&c=${c}`);
        if (data) {
            selectedCell = data.selected_cell;
            let cellValueDisplay = "N/A";
            if(currentGridColors && currentGridColors[r] && currentGridColors[r][c]){
                 // Simple approximation from hex for display
                 const hex = currentGridColors[r][c].substring(1);
                 const R = parseInt(hex.substring(0,2), 16) / 255;
                 const G = parseInt(hex.substring(2,4), 16) / 255;
                 const B = parseInt(hex.substring(4,6), 16) / 255;
                 cellValueDisplay = ((R+G+B)/3).toFixed(3); // Average, crude
            }
            cellInfoLabel.innerHTML = `Selected Cell: (Row=${r}, Col=${c})<br>Approx. Value: ${cellValueDisplay}`;
            
            let neighText = "Neighborhood (3x3 Input - Row Major):\n"; 
            data.neighborhood.forEach(row => {
                neighText += row.map(val => val.toFixed(3)).join("  ") + "\n";
            });
            neighborhoodDisplay.textContent = neighText;

            currentLayerActivations = data.layer_activations;
            let actText = "Layer Activations (Input, Hidden(s), Output):\n";
            data.layer_activations.forEach((layerAct, i) => {
                let actSample = layerAct.slice(0, 8).map(val => typeof val === 'number' ? val.toFixed(3) : val).join(", "); // Show more values
                if(layerAct.length > 8) actSample += ", ...";
                actText += `L${i} (Size ${layerAct.length}): [${actSample}]\n`;
            });
            activationDisplay.textContent = actText;

            updateNetworkNodeColors(data.layer_activations);
            clearSelectionButton.style.display = 'inline-block'; 
        } else {
            clearCellDetailsDisplay();
        }
    }

    function clearCellDetailsDisplay() {
        cellInfoLabel.textContent = "Click on a CA cell to see details.";
        neighborhoodDisplay.textContent = "";
        activationDisplay.textContent = "";
        selectedCell = null;
        currentLayerActivations = null;
        clearSelectionButton.style.display = 'none'; 
        if(currentGridColors) drawNcaGrid(currentGridColors); // Redraw grid to remove highlight
        if (mlpParamsForViz) buildNetworkViz(); 
    }

    ncaCanvas.addEventListener('click', (event) => {
        const rect = ncaCanvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        const c = Math.floor(x / CELL_SIZE);
        const r = Math.floor(y / CELL_SIZE);

        if (r >= 0 && r < gridSize && c >= 0 && c < gridSize) {
            if (selectedCell && selectedCell.r === r && selectedCell.c === c) {
                // Clicked same cell, deselect it
                clearCellDetailsDisplay();
            } else {
                updateCellDetails(r, c);
                if(currentGridColors) drawNcaGrid(currentGridColors); 
                highlightNeighborhood(r, c); 
            }
        }
    });

    ncaCanvas.addEventListener('mousemove', (event) => {
        if(!currentGridColors) return;
        const rect = ncaCanvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        const c = Math.floor(x / CELL_SIZE);
        const r = Math.floor(y / CELL_SIZE);

        if (r >= 0 && r < gridSize && c >= 0 && c < gridSize) {
            const hex = currentGridColors[r][c].substring(1);
            const R = parseInt(hex.substring(0,2), 16);
            const G = parseInt(hex.substring(2,4), 16);
            const B = parseInt(hex.substring(4,6), 16);
            const approxVal = ((R/255 + G/255 + B/255)/3).toFixed(3);
            hoverCellInfo.textContent = `(${r},${c}): ${approxVal}`;
            hoverCellInfo.style.left = `${event.pageX + 10}px`;
            hoverCellInfo.style.top = `${event.pageY + 10}px`;
            hoverCellInfo.style.display = 'block';
        } else {
            hoverCellInfo.style.display = 'none';
        }
    });
    
    ncaCanvas.addEventListener('mouseout', () => {
        hoverCellInfo.style.display = 'none';
    });

    clearSelectionButton.addEventListener('click', clearCellDetailsDisplay);

    function highlightNeighborhood(r, c) {
        ncaCtx.strokeStyle = 'yellow'; // Highlight color
        ncaCtx.lineWidth = 1.5; // Thinner highlight
        const offset = CELL_SIZE * 0.05; // Small offset to draw inside cell boundary
        const size = CELL_SIZE * 0.9;   // Slightly smaller rectangle
        for (let dr = -1; dr <= 1; dr++) {
            for (let dc = -1; dc <= 1; dc++) {
                const rr = (r + dr + gridSize) % gridSize;
                const cc = (c + dc + gridSize) % gridSize;
                ncaCtx.strokeRect(cc * CELL_SIZE + offset, rr * CELL_SIZE + offset, size, size);
            }
        }
    }
    
    let resizeTimeout;
    new ResizeObserver(() => {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(() => {
            const container = document.querySelector('.network-viz-container');
            if (container) {
                networkCanvas.width = container.clientWidth;
                networkCanvas.height = container.clientHeight;
                buildNetworkViz(); 
                if (selectedCell && currentLayerActivations) {
                    updateNetworkNodeColors(currentLayerActivations);
                }
            }
        }, 100); 
    }).observe(document.querySelector('.network-viz-container'));

    loadInitialConfig().then(() => {
        const netContainer = document.querySelector('.network-viz-container');
        if (netContainer) { 
            networkCanvas.width = netContainer.clientWidth;
            networkCanvas.height = netContainer.clientHeight;
            buildNetworkViz(); 
        }
        updateNetworkLegend(); 
    });

    var coll = document.getElementsByClassName("collapsible");
    for (let i = 0; i < coll.length; i++) {
      coll[i].addEventListener("click", function() {
        this.classList.toggle("active");
        var content = this.nextElementSibling;
        if (content.style.display === "block") {
          content.style.display = "none";
        } else {
          content.style.display = "block";
        }
      });
      // Make some sections open by default if they have 'active' class in HTML
      if (coll[i].classList.contains('active')) {
          coll[i].nextElementSibling.style.display = "block";
      }
    }
});