// static/js/main.js
document.addEventListener('DOMContentLoaded', () => {
    const ncaCanvas = document.getElementById('ncaCanvas');
    const ncaCtx = ncaCanvas.getContext('2d');
    const networkCanvas = document.getElementById('networkCanvas');
    const networkCtx = networkCanvas.getContext('2d');

    const toggleRunButton = document.getElementById('toggleRunButton');
    const stepButton = document.getElementById('stepButton');
    const stepBackButton = document.getElementById('stepBackButton'); // New
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
    const applySettingsButton = document.getElementById('applySettingsButton'); // New
    const randomizeWeightsButton = document.getElementById('randomizeWeightsButton');
    const randomizeGridButton = document.getElementById('randomizeGridButton');
    const speedSlider = document.getElementById('speedSlider');
    const speedValue = document.getElementById('speedValue');

    // New Weight Initialization UI Elements
    const weightSettingModeSelector = document.getElementById('weightSettingModeSelector');
    const randomWeightSettings = document.getElementById('randomWeightSettings');
    const presetWeightSettings = document.getElementById('presetWeightSettings');
    const parametricWeightSettings = document.getElementById('parametricWeightSettings');
    const directWeightSettings = document.getElementById('directWeightSettings');
    const fileWeightSettings = document.getElementById('fileWeightSettings');

    const randomSeedInput = document.getElementById('randomSeedInput');
    const presetPatternSelect = document.getElementById('presetPatternSelect');
    const applyPresetWeightsBtn = document.getElementById('applyPresetWeightsBtn');
    const parametricPatternSelect = document.getElementById('parametricPatternSelect');
    const parametricInputs = document.getElementById('parametricInputs');
    const generateParametricWeightsBtn = document.getElementById('generateParametricWeightsBtn');
    const directWeightGrid = directWeightSettings.querySelector('.weight-grid');
    const applyDirectWeightsBtn = document.getElementById('applyDirectWeightsBtn');
    const resetDirectWeightsBtn = document.getElementById('resetDirectWeightsBtn');
    const weightFileInput = document.getElementById('weightFileInput');
    const uploadWeightsBtn = document.getElementById('uploadWeightsBtn');
    const downloadWeightsBtn = document.getElementById('downloadWeightsBtn');

    const cellInfoLabel = document.getElementById('cellInfoLabel');
    const neighborhoodDisplay = document.getElementById('neighborhoodDisplay');
    const activationDisplay = document.getElementById('activationDisplay');
    const networkLegend = document.getElementById('networkLegend');
    const clearSelectionButton = document.getElementById('clearSelectionButton'); // New
    const hoverCellInfo = document.createElement('div');
    hoverCellInfo.style.cssText = `
        position: absolute;
        background: rgba(0, 0, 0, 0.7);
        color: white;
        padding: 5px;
        border-radius: 3px;
        pointer-events: none;
        display: none;
        z-index: 100;
    `;
    document.body.appendChild(hoverCellInfo);
    
    const CELL_SIZE = 15; // Pixel size for each CA cell
    let gridSize = 50; // Default, will be updated from backend
    let isRunning = false;
    let animationIntervalId = null;
    let currentSpeed = 200;
    let currentGridColors = null; // Store grid colors for hover
    let hiddenLayerSizes = []; // Stores sizes of hidden layers, e.g., [8, 16]
    const MAX_HIDDEN_LAYERS = 3;
    const MIN_NODE_SIZE = 1;
    const MAX_NODE_SIZE = 64;

    let mlpParamsForViz = null; // { layer_sizes: [], weights: [[]] }
    let selectedCell = null; // { r: null, c: null }
    let currentLayerActivations = null;


    async function fetchApi(endpoint, method = 'GET', body = null) {
        const options = { method };
        if (body) {
            options.headers = { 'Content-Type': 'application/json' };
            options.body = JSON.stringify(body);
        }
        try {
            const response = await fetch(endpoint, options);
            if (!response.ok) {
                const errorData = await response.json();
                console.error(`API Error (${response.status}):`, errorData.error || response.statusText);
                alert(`Error: ${errorData.error || response.statusText}`);
                return null;
            }
            return await response.json();
        } catch (error) {
            console.error('Network or API call failed:', error);
            alert(`Network error: ${error.message}`);
            return null;
        }
    }

    function drawNcaGrid(gridColors) {
        if (!gridColors) return;
        gridSize = gridColors.length; // Assuming square grid
        ncaCanvas.width = gridSize * CELL_SIZE;
        ncaCanvas.height = gridSize * CELL_SIZE;

        for (let r = 0; r < gridSize; r++) {
            for (let c = 0; c < gridSize; c++) {
                ncaCtx.fillStyle = gridColors[r][c];
                ncaCtx.fillRect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE);
            }
        }
        // Re-apply highlight if a cell is selected
        if (selectedCell) {
            highlightNeighborhood(selectedCell.r, selectedCell.c);
        }
    }
    
    function updateUiControls(params) {
        if (params.layer_sizes) {
            // Assuming params.layer_sizes is like [9, 8, 16, 1]
            // Extract hidden layers (excluding input and output)
            hiddenLayerSizes = params.layer_sizes.slice(1, -1);
            renderLayerBuilder(); // Re-render the dynamic layer inputs
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
         if (params.initial_seed !== undefined && params.initial_seed !== null) {
            randomSeedInput.value = params.initial_seed;
         } else {
            randomSeedInput.value = ''; // Clear if no seed
         }
     }

    async function loadInitialConfig() {
        const config = await fetchApi('/api/config');
        if (!config) return;

        gridSize = config.default_params.grid_size;
        mlpParamsForViz = config.mlp_params_for_viz;
        
        // Populate selectors
        Object.keys(config.presets).forEach(name => {
            const option = new Option(name, name);
            presetSelector.add(option);
        });
        config.available_activations.forEach(name => {
            const option = new Option(name, name);
            activationSelector.add(option);
        });
        config.available_colormaps.forEach(name => {
            const option = new Option(name, name);
            colormapSelector.add(option);
        });
        colormapSelector.value = config.current_colormap;

        // Populate new weight initialization selectors
        if (config.available_preset_patterns) {
            config.available_preset_patterns.forEach(name => {
                const option = new Option(name, name);
                presetPatternSelect.add(option);
            });
        }
        if (config.available_parametric_patterns) {
            config.available_parametric_patterns.forEach(name => {
                const option = new Option(name, name);
                parametricPatternSelect.add(option);
            });
        }

        // Set initial control values from the first preset or default
        const initialPresetName = presetSelector.value || "Flicker";
        const initialPresetData = config.presets[initialPresetName];
         if (initialPresetData) {
            const [seed, layers, act, w_scale, b_val] = initialPresetData;
            updateUiControls({
                layer_sizes: layers,
                activation: act,
                weight_scale: w_scale,
                bias: b_val,
                initial_seed: seed
            });
            presetSelector.value = initialPresetName; // Set preset selector to the loaded preset
        } else { // Fallback if no preset selected or preset data is missing
             updateUiControls(config.default_params);
        }


        currentGridColors = config.initial_grid_colors; // Store initial grid colors
        drawNcaGrid(currentGridColors);
        buildNetworkViz(); // Initial network draw
        updateNetworkLegend(); // Initial legend draw

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
        applySettingsButton.disabled = true; // Initialize apply settings button as disabled
    }

    async function handleStep() {
        const data = await fetchApi('/api/step', 'POST');
        if (data) {
            currentGridColors = data.grid_colors; // Update stored grid colors
            drawNcaGrid(currentGridColors);
            if (selectedCell) updateCellDetails(selectedCell.r, selectedCell.c); // Refresh if cell selected
        }
    }

    function startAnimationLoop() {
        if (animationIntervalId) clearInterval(animationIntervalId);
        animationIntervalId = setInterval(async () => {
            if (isRunning) {
                await handleStep();
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

    stepButton.addEventListener('click', async () => {
        // The step action should proceed regardless of the paused state.
        // Removed logic that pauses the simulation before sending a step request.
        await handleStep();
    });

    stepBackButton.addEventListener('click', async () => {
        const data = await fetchApi('/api/step_back', 'POST');
        if (data) {
            currentGridColors = data.grid_colors; // Update stored grid colors
            drawNcaGrid(currentGridColors);
            if (selectedCell) updateCellDetails(selectedCell.r, selectedCell.c); // Refresh if cell selected
        }
    });

    // Enable Apply Settings button when NCA parameters change
    const ncaParameterControls = [
        colormapSelector, // presetSelector is handled separately for 'Custom' logic
        activationSelector
    ];
    // Add event listener for changes within the layer builder container
    layerBuilderContainer.addEventListener('input', () => {
        applySettingsButton.disabled = false;
    });

    // Controls that enable applySettingsButton (for NCA Parameters section)
    [colormapSelector, activationSelector].forEach(control => {
        control.addEventListener('change', () => {
            applySettingsButton.disabled = false;
        });
    });

    // Handle preset selection separately to update UI and potentially mark as custom
    presetSelector.addEventListener('change', async () => {
        applySettingsButton.disabled = false; // Enable apply button
        const selectedPresetName = presetSelector.value;
        if (selectedPresetName !== "Custom") {
            // Fetch config to get preset details and update UI controls
            const config = await fetchApi('/api/config');
            if (config && config.presets[selectedPresetName]) {
                const [seed, layers, act, w_scale, b_val] = config.presets[selectedPresetName];
                updateUiControls({
                    layer_sizes: layers,
                    activation: act,
                    weight_scale: w_scale,
                    bias: b_val,
                    initial_seed: seed
                });
                // Do NOT update network visualization here. It updates on Apply Settings.
            }
        }
    });

    // Controls that enable applySettingsButton (for NCA Parameters section)
    // Note: weightScaleSlider and biasSlider are now part of randomWeightSettings
    // and will not enable applySettingsButton directly.
    // Their changes will be applied via randomizeWeightsButton or applySettingsButton
    // if the mode is 'random' and 'Apply Settings' is clicked.

    applySettingsButton.addEventListener('click', async () => {
        const params = {
            preset_name: presetSelector.value,
            colormap_name: colormapSelector.value,
            // Construct layer_sizes from hiddenLayerSizes array
            layer_sizes: [9, ...hiddenLayerSizes, 1].join(','),
            activation: activationSelector.value,
            weight_scale: parseFloat(weightScaleSlider.value),
            bias: parseFloat(biasSlider.value)
        };
        const data = await fetchApi('/api/apply_settings', 'POST', params);
        if (data) {
            currentGridColors = data.grid_colors;
            drawNcaGrid(currentGridColors);
            updateUiControls(data.current_params);
            mlpParamsForViz = data.mlp_params_for_viz; // Update mlpParamsForViz from backend response
            buildNetworkViz(); // Redraw network visualization with new active architecture
            updateNetworkLegend();
            // selectedCell = null; // Removed: Do not clear selected cell
            // clearCellDetailsDisplay(); // Removed: Do not clear details display
            if (selectedCell) updateCellDetails(selectedCell.r, selectedCell.c); // Refresh if cell selected
            applySettingsButton.disabled = true; // Disable after successful application
            if (data.is_paused) {
                isRunning = false;
                toggleRunButton.textContent = 'Start';
                toggleRunButton.classList.remove('running');
                if (animationIntervalId) clearInterval(animationIntervalId);
            }
        }
    });

    randomizeWeightsButton.addEventListener('click', async () => {
        const params = {
            weight_scale: parseFloat(weightScaleSlider.value),
            bias: parseFloat(biasSlider.value),
            random_seed: randomSeedInput.value ? parseInt(randomSeedInput.value) : null
        };
        const data = await fetchApi('/api/randomize_weights', 'POST', params);
        if (data) {
            mlpParamsForViz = data.mlp_params_for_viz; // Update mlpParamsForViz from backend response
            buildNetworkViz(); // Redraw network visualization with new active architecture
            updateNetworkLegend();
            if (selectedCell) updateCellDetails(selectedCell.r, selectedCell.c); // Refresh if cell selected
            presetSelector.value = "Custom"; // Randomizing weights makes it a custom setup
        }
    });

    randomizeGridButton.addEventListener('click', async () => {
        const data = await fetchApi('/api/randomize_grid', 'POST', { seed: Date.now() });
        if (data) {
            currentGridColors = data.grid_colors;
            drawNcaGrid(currentGridColors);
            // selectedCell = null; // Removed: Do not clear selected cell
            // clearCellDetailsDisplay(); // Removed: Do not clear details display
            if (selectedCell) updateCellDetails(selectedCell.r, selectedCell.c); // Refresh if cell selected
            if (data.is_paused) {
                isRunning = false;
                toggleRunButton.textContent = 'Start';
                toggleRunButton.classList.remove('running');
                if (animationIntervalId) clearInterval(animationIntervalId);
            }
        }
    });


    // UI Updates for sliders (now only for random mode)
    weightScaleSlider.addEventListener('input', (e) => weightScaleValue.textContent = parseFloat(e.target.value).toFixed(1));
    biasSlider.addEventListener('input', (e) => biasValue.textContent = parseFloat(e.target.value).toFixed(1));
    speedSlider.addEventListener('input', (e) => {
        currentSpeed = parseInt(e.target.value);
        speedValue.textContent = currentSpeed;
        if (isRunning) { // Restart animation with new speed
            startAnimationLoop();
        }
    });
    currentSpeed = parseInt(speedSlider.value); // Initialize speed

    // --- Weight Setting Mode Selection Logic ---
    const allSettingsContainers = [
        randomWeightSettings,
        presetWeightSettings,
        parametricWeightSettings,
        directWeightSettings,
        fileWeightSettings
    ];

    function showSelectedModeSettings(selectedMode) {
        allSettingsContainers.forEach(container => {
            container.style.display = 'none'; // Hide all first
        });

        switch (selectedMode) {
            case 'random':
                randomWeightSettings.style.display = 'block';
                break;
            case 'preset':
                presetWeightSettings.style.display = 'block';
                break;
            case 'parametric':
                parametricWeightSettings.style.display = 'block';
                break;
            case 'direct':
                directWeightSettings.style.display = 'block';
                break;
            case 'file':
                fileWeightSettings.style.display = 'block';
                break;
            default:
                // Default to random if something goes wrong
                randomWeightSettings.style.display = 'block';
        }
    }

    // Initial display based on checked radio button
    const initialMode = weightSettingModeSelector.querySelector('input[name="weightMode"]:checked').value;
    showSelectedModeSettings(initialMode);

    // Event listener for mode changes
    weightSettingModeSelector.addEventListener('change', (event) => {
        showSelectedModeSettings(event.target.value);
    });

    // --- Preset Weight Patterns ---
    applyPresetWeightsBtn.addEventListener('click', async () => {
        const presetName = presetPatternSelect.value;
        const data = await fetchApi('/api/set_preset_weights', 'POST', { preset_name: presetName });
        if (data) {
            mlpParamsForViz = data.mlp_params_for_viz;
            buildNetworkViz();
            updateNetworkLegend();
            if (selectedCell) updateCellDetails(selectedCell.r, selectedCell.c);
            presetSelector.value = "Custom"; // Applying a preset makes it a custom setup
        }
    });

    // --- Parametric Weight Generation ---
    parametricPatternSelect.addEventListener('change', async () => {
        // Clear previous parametric inputs
        parametricInputs.innerHTML = '';
        const selectedPattern = parametricPatternSelect.value;
        // Fetch parameters for the selected pattern from backend
        const config = await fetchApi('/api/config'); // Re-fetch config to get pattern details
        if (config && config.parametric_patterns_meta && config.parametric_patterns_meta[selectedPattern]) {
            const paramsMeta = config.parametric_patterns_meta[selectedPattern];
            paramsMeta.forEach(param => {
                const label = document.createElement('label');
                label.textContent = `${param.name}:`;
                label.setAttribute('for', `param-${param.name}`);
                parametricInputs.appendChild(label);

                const input = document.createElement('input');
                input.type = param.type === 'float' ? 'number' : 'range';
                input.id = `param-${param.name}`;
                input.name = param.name;
                input.value = param.default;
                if (param.min !== undefined) input.min = param.min;
                if (param.max !== undefined) input.max = param.max;
                if (param.step !== undefined) input.step = param.step;
                if (param.type === 'float') input.setAttribute('step', 'any'); // Allow decimal input

                parametricInputs.appendChild(input);
                
                if (param.type === 'range') {
                    const span = document.createElement('span');
                    span.id = `param-${param.name}-value`;
                    span.textContent = param.default;
                    input.addEventListener('input', (e) => span.textContent = parseFloat(e.target.value).toFixed(param.step ? (param.step.toString().split('.')[1] || '').length : 0));
                    parametricInputs.appendChild(span);
                }
                parametricInputs.appendChild(document.createElement('br'));
            });
        }
    });

    generateParametricWeightsBtn.addEventListener('click', async () => {
        const patternType = parametricPatternSelect.value;
        const parameters = {};
        parametricInputs.querySelectorAll('input').forEach(input => {
            parameters[input.name] = parseFloat(input.value);
        });

        const data = await fetchApi('/api/set_parametric_weights', 'POST', { pattern_type: patternType, parameters: parameters });
        if (data) {
            mlpParamsForViz = data.mlp_params_for_viz;
            buildNetworkViz();
            updateNetworkLegend();
            if (selectedCell) updateCellDetails(selectedCell.r, selectedCell.c);
            presetSelector.value = "Custom";
        }
    });

    // Trigger initial load of parametric inputs if a pattern is pre-selected
    if (parametricPatternSelect.value) {
        parametricPatternSelect.dispatchEvent(new Event('change'));
    }

    // --- Direct Weight Editor ---
    function createDirectWeightGrid() {
        directWeightGrid.innerHTML = ''; // Clear existing
        for (let r = 0; r < 3; r++) {
            const rowDiv = document.createElement('div');
            rowDiv.classList.add('weight-grid-row');
            for (let c = 0; c < 3; c++) {
                const input = document.createElement('input');
                input.type = 'number';
                input.classList.add('weight-input');
                input.setAttribute('data-row', r);
                input.setAttribute('data-col', c);
                input.value = '0.0'; // Default to zero
                input.step = 'any'; // Allow decimal input
                rowDiv.appendChild(input);
            }
            directWeightGrid.appendChild(rowDiv);
        }
    }

    function getDirectWeightMatrix() {
        const matrix = [];
        for (let r = 0; r < 3; r++) {
            const row = [];
            for (let c = 0; c < 3; c++) {
                const input = directWeightGrid.querySelector(`input[data-row="${r}"][data-col="${c}"]`);
                row.push(parseFloat(input.value));
            }
            matrix.push(row);
        }
        return matrix;
    }

    function setDirectWeightMatrix(matrix) {
        if (!matrix || matrix.length !== 3 || matrix[0].length !== 3) return;
        for (let r = 0; r < 3; r++) {
            for (let c = 0; c < 3; c++) {
                const input = directWeightGrid.querySelector(`input[data-row="${r}"][data-col="${c}"]`);
                if (input) input.value = matrix[r][c].toFixed(3); // Display with 3 decimal places
            }
        }
    }

    applyDirectWeightsBtn.addEventListener('click', async () => {
        const weightsMatrix = getDirectWeightMatrix();
        const data = await fetchApi('/api/set_direct_weights', 'POST', { weights_matrix: weightsMatrix });
        if (data) {
            mlpParamsForViz = data.mlp_params_for_viz;
            buildNetworkViz();
            updateNetworkLegend();
            if (selectedCell) updateCellDetails(selectedCell.r, selectedCell.c);
            presetSelector.value = "Custom";
        }
    });

    resetDirectWeightsBtn.addEventListener('click', () => {
        setDirectWeightMatrix([[0,0,0],[0,0,0],[0,0,0]]);
    });

    // Initial creation of the direct weight grid
    createDirectWeightGrid();

    // --- Import/Export Weights ---
    uploadWeightsBtn.addEventListener('click', async () => {
        const file = weightFileInput.files[0];
        if (!file) {
            alert('Please select a file to upload.');
            return;
        }

        const reader = new FileReader();
        reader.onload = async (e) => {
            const fileContent = e.target.result;
            const data = await fetchApi('/api/upload_weights', 'POST', { file_content: fileContent, filename: file.name });
            if (data) {
                mlpParamsForViz = data.mlp_params_for_viz;
                buildNetworkViz();
                updateNetworkLegend();
                if (selectedCell) updateCellDetails(selectedCell.r, selectedCell.c);
                presetSelector.value = "Custom";
                alert('Weights uploaded successfully!');
            }
        };
        reader.readAsText(file);
    });

    downloadWeightsBtn.addEventListener('click', async () => {
        const data = await fetchApi('/api/download_weights', 'GET');
        if (data && data.file_content && data.filename) {
            const blob = new Blob([data.file_content], { type: 'application/json' }); // Assuming JSON for now
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = data.filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        } else {
            alert('Failed to download weights.');
        }
    });


    // --- Network Visualization ---
    let netNodePositions = []; // List of lists of {x,y}
    
    function weightToColor(w) {
        const maxVal = 3.0; // Sync with Python version
        const wClamped = Math.max(-maxVal, Math.min(w, maxVal));
        const norm = (wClamped + maxVal) / (2 * maxVal); // 0..1
        const r = Math.floor((1.0 - norm) * 255);
        const g = Math.floor(norm * 255);
        const b = 0;
        return `rgb(${r},${g},${b})`;
    }

    function weightToThickness(w) {
        const maxWidth = 3.0; // Sync with Python version
        return 1 + Math.min(Math.abs(w), 3.0) / 3.0 * (maxWidth - 1);
    }

    function activationToColor(a) {
        // Convert activation [0..something] to white→blue gradient
        const aClamped = Math.max(0.0, Math.min(a, 1.5)); // Clamp to [0..1.5]
        const b = Math.floor((aClamped / 1.5) * 255);
        const r = 255 - b;
        const g = 255 - b; // Creates a white to cyan/light blue gradient
        return `rgb(${r},${g},255)`;
    }

    function buildNetworkViz() {
        if (!mlpParamsForViz || !mlpParamsForViz.layer_sizes) {
            // console.warn("MLP params not available for visualization.");
            networkCtx.clearRect(0, 0, networkCanvas.width, networkCanvas.height);
            return;
        }

        const layerSizes = mlpParamsForViz.layer_sizes;
        const weights = mlpParamsForViz.weights; // This is W[i][in_idx][out_idx]
        
        networkCtx.clearRect(0, 0, networkCanvas.width, networkCanvas.height);
        netNodePositions = [];

        const width = networkCanvas.width;
        const height = networkCanvas.height;
        const layerCount = layerSizes.length;
        const xSpacing = width / (layerCount + 1);
        const nodeRadius = 8;

        // Calculate node positions
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

        // Draw lines (weights)
        if (weights) {
            for (let i = 0; i < weights.length; i++) { // Iterate through weight matrices (between layers)
                const W_i = weights[i]; // W_i is from layer i to layer i+1
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
        
        // Draw nodes (over lines)
        // Default fill color: white. Will be updated by updateNetworkNodeColors.
        for (let i = 0; i < netNodePositions.length; i++) {
            for (let j = 0; j < netNodePositions[i].length; j++) {
                const pos = netNodePositions[i][j];
                networkCtx.beginPath();
                networkCtx.arc(pos.x, pos.y, nodeRadius, 0, 2 * Math.PI);
                networkCtx.fillStyle = "white"; // Default
                networkCtx.fill();
                networkCtx.strokeStyle = "black";
                networkCtx.lineWidth = 1;
                networkCtx.stroke();
            }
        }
        // If there's a selected cell and activations, color them now
        if (selectedCell && currentLayerActivations) {
            updateNetworkNodeColors(currentLayerActivations);
        }
        updateNetworkLegend(); // Ensure legend is updated after network viz
    }
    
    function updateNetworkNodeColors(layerActivations) {
        if (!netNodePositions || netNodePositions.length === 0 || !layerActivations) return;
        const nodeRadius = 8;

        for (let layerIdx = 0; layerIdx < layerActivations.length; layerIdx++) {
            if (layerIdx < netNodePositions.length) { // Ensure we have positions for this layer
                const activationsInLayer = layerActivations[layerIdx];
                const nodesInLayer = netNodePositions[layerIdx];
                for (let nodeIdx = 0; nodeIdx < activationsInLayer.length; nodeIdx++) {
                    if (nodeIdx < nodesInLayer.length) { // Ensure we have this node
                        const val = activationsInLayer[nodeIdx];
                        const color = activationToColor(val);
                        const pos = nodesInLayer[nodeIdx];
                        
                        networkCtx.beginPath();
                        networkCtx.arc(pos.x, pos.y, nodeRadius, 0, 2 * Math.PI);
                        networkCtx.fillStyle = color;
                        networkCtx.fill();
                        networkCtx.strokeStyle = "black"; // Re-stroke border
                        networkCtx.lineWidth = 1;
                        networkCtx.stroke();
                    }
                }
            }
        }
    }

    function updateNetworkLegend() {
        if (!networkLegend) return;
        networkLegend.innerHTML = `
            <div><span class="color-box" style="background-color: rgb(255,0,0);"></span> Negative Weight</div>
            <div><span class="color-box" style="background-color: rgb(0,255,0);"></span> Positive Weight</div>
            <div><span class="color-box" style="background-color: rgb(255,255,255);"></span> Low Activation</div>
            <div><span class="color-box" style="background-color: rgb(0,0,255);"></span> High Activation</div>
        `;
    }
    
    async function updateCellDetails(r, c) {
        const data = await fetchApi(`/api/cell_details?r=${r}&c=${c}`);
        if (data) {
            selectedCell = data.selected_cell;
            cellInfoLabel.innerHTML = `Selected Cell: (Row=${r}, Col=${c}) <br> Current Value: ${currentGridColors[r][c]}`; // Display current value
            
            let neighText = "Neighborhood (3x3) - Input to MLP:\n"; // More descriptive
            data.neighborhood.forEach(row => {
                neighText += row.map(val => val.toFixed(3)).join("  ") + "\n";
            });
            neighborhoodDisplay.textContent = neighText;

            let actText = "Layer Activations (showing first few values if long):\n";
            currentLayerActivations = data.layer_activations; // Store for re-draws
            data.layer_activations.forEach((layerAct, i) => {
                let actSample = layerAct.slice(0, 5).map(val => typeof val === 'number' ? val.toFixed(3) : val).join(", ");
                if(layerAct.length > 5) actSample += ", ...";
                actText += `Layer ${i}: [${actSample}]\n`;
            });
            actText = "Layer Activations (Input, Hidden, Output):\n"; // More descriptive
            currentLayerActivations = data.layer_activations; // Store for re-draws
            data.layer_activations.forEach((layerAct, i) => {
                let actSample = layerAct.slice(0, 5).map(val => typeof val === 'number' ? val.toFixed(3) : val).join(", ");
                if(layerAct.length > 5) actSample += ", ...";
                actText += `Layer ${i}: [${actSample}]\n`;
            });
            activationDisplay.textContent = actText;

            // Redraw network nodes with new activation colors
            // The base network structure (lines, default nodes) should already be drawn by buildNetworkViz
            updateNetworkNodeColors(data.layer_activations);
            clearSelectionButton.style.display = 'inline-block'; // Show clear button
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
        clearSelectionButton.style.display = 'none'; // Hide clear button
        drawNcaGrid(currentGridColors); // Redraw grid to remove highlight
        if (mlpParamsForViz) buildNetworkViz(); // This will redraw with default node colors
    }

    // --- Dynamic Layer Builder Functions ---
    function renderLayerBuilder() {
        layerBuilderContainer.innerHTML = ''; // Clear existing inputs

        // Add input for each hidden layer
        hiddenLayerSizes.forEach((size, index) => {
            const layerDiv = document.createElement('div');
            layerDiv.classList.add('layer-input-group');
            layerDiv.innerHTML = `
                <label for="hiddenLayer${index}">Hidden Layer ${index + 1} Size:</label>
                <input type="number" id="hiddenLayer${index}" value="${size}"
                       min="${MIN_NODE_SIZE}" max="${MAX_NODE_SIZE}" class="hidden-layer-input">
            `;
            layerBuilderContainer.appendChild(layerDiv);

            // Add event listener for changes to this specific input
            layerDiv.querySelector(`#hiddenLayer${index}`).addEventListener('input', (e) => {
                let value = parseInt(e.target.value);
                if (isNaN(value) || value < MIN_NODE_SIZE) {
                    value = MIN_NODE_SIZE;
                } else if (value > MAX_NODE_SIZE) {
                    value = MAX_NODE_SIZE;
                }
                e.target.value = value; // Update input field with clamped value
                hiddenLayerSizes[index] = value;
                applySettingsButton.disabled = false;
                presetSelector.value = "Custom"; // Set preset to Custom on manual change
            });
        });

        // Update button states based on current number of hidden layers
        addHiddenLayerButton.disabled = hiddenLayerSizes.length >= MAX_HIDDEN_LAYERS;
        removeHiddenLayerButton.disabled = hiddenLayerSizes.length === 0;

        // No immediate network visualization update here. It will update on Apply Settings.
    }


    // Event listeners for Add/Remove Hidden Layer buttons
    addHiddenLayerButton.addEventListener('click', () => {
        if (hiddenLayerSizes.length < MAX_HIDDEN_LAYERS) {
            hiddenLayerSizes.push(8); // Default new hidden layer size
            renderLayerBuilder();
            applySettingsButton.disabled = false;
            presetSelector.value = "Custom"; // Set preset to Custom on adding layer
        }
    });

    removeHiddenLayerButton.addEventListener('click', () => {
        if (hiddenLayerSizes.length > 0) {
            hiddenLayerSizes.pop();
            renderLayerBuilder();
            applySettingsButton.disabled = false;
            presetSelector.value = "Custom"; // Set preset to Custom on removing layer
        }
    });

    ncaCanvas.addEventListener('click', (event) => {
        const rect = ncaCanvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        const c = Math.floor(x / CELL_SIZE);
        const r = Math.floor(y / CELL_SIZE);

        if (r >= 0 && r < gridSize && c >= 0 && c < gridSize) {
            updateCellDetails(r, c);
            drawNcaGrid(currentGridColors); // Redraw grid to clear previous highlight
            highlightNeighborhood(r, c); // Highlight the new neighborhood
        }
    });

    // Cell hover functionality
    ncaCanvas.addEventListener('mousemove', (event) => {
        const rect = ncaCanvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        const c = Math.floor(x / CELL_SIZE);
        const r = Math.floor(y / CELL_SIZE);

        if (r >= 0 && r < gridSize && c >= 0 && c < gridSize && currentGridColors) {
            const cellValue = parseFloat(currentGridColors[r][c].substring(1, 7).match(/.{2}/g).map(hex => parseInt(hex, 16) / 255).reduce((a, b) => a + b) / 3).toFixed(3); // Approximate value from hex color
            hoverCellInfo.textContent = `Cell (${r}, ${c}): ${cellValue}`;
            hoverCellInfo.style.left = `${event.clientX + 10}px`;
            hoverCellInfo.style.top = `${event.clientY + 10}px`;
            hoverCellInfo.style.display = 'block';
        } else {
            hoverCellInfo.style.display = 'none';
        }
    });
    
    ncaCanvas.addEventListener('mouseout', () => {
        hoverCellInfo.style.display = 'none';
    });

    clearSelectionButton.addEventListener('click', () => {
        clearCellDetailsDisplay();
    });

    function highlightNeighborhood(r, c) {
        ncaCtx.strokeStyle = 'yellow';
        ncaCtx.lineWidth = 2;
        for (let dr = -1; dr <= 1; dr++) {
            for (let dc = -1; dc <= 1; dc++) {
                const rr = (r + dr + gridSize) % gridSize;
                const cc = (c + dc + gridSize) % gridSize;
                ncaCtx.strokeRect(cc * CELL_SIZE, rr * CELL_SIZE, CELL_SIZE, CELL_SIZE);
            }
        }
    }
    
    // Responsive Network Canvas (redraw on resize)
    // Debounce resize event
    let resizeTimeout;
    new ResizeObserver(() => {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(() => {
            const container = document.querySelector('.network-viz-container');
            if (container) {
                 // Set canvas drawinglogical  size to its display CSS size
                networkCanvas.width = container.clientWidth;
                networkCanvas.height = container.clientHeight;
                buildNetworkViz(); // Redraw the network based on new dimensions
                // If a cell is selected, re-apply its activation colors
                if (selectedCell && currentLayerActivations) {
                    updateNetworkNodeColors(currentLayerActivations);
                }
            }
        }, 100); // Adjust debounce delay as needed
    }).observe(document.querySelector('.network-viz-container'));


    // Initialize
    loadInitialConfig().then(() => {
        // Set initial canvas size for network viz after config is loaded and container is sized
        const netContainer = document.querySelector('.network-viz-container');
        networkCanvas.width = netContainer.clientWidth;
        networkCanvas.height = netContainer.clientHeight;
        buildNetworkViz(); // Draw network based on initial mlpParamsForViz
        updateNetworkLegend(); // Ensure legend is drawn on init
    });
// Collapsible sections logic
    var coll = document.getElementsByClassName("collapsible");
    var i;
 
    for (i = 0; i < coll.length; i++) {
      coll[i].addEventListener("click", function() {
        this.classList.toggle("active");
        var content = this.nextElementSibling;
        if (content.style.display === "block") {
          content.style.display = "none";
        } else {
          content.style.display = "block";
        }
      });
    }
});