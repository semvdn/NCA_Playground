// static/js/modules/eventHandlers.js

import {
    toggleRunButton, stepButton, stepBackButton, randomizeGridButton,
    randomizeArchitectureButton, restartButton, randomizeWeightsButton,
    captureScreenshotButton, toggleRecordingButton,
    activationSelector, weightScaleSlider, biasSlider, colormapSelector, presetSelector,
    speedSlider, speedValue, clearSelectionButton
} from './domElements.js';
import { state, setIsRunning, setMlpParamsForViz, setHiddenLayerSizes, setMaxHiddenLayersCount, setMinNodeCountPerLayer, setMaxNodeCountPerLayer, setCurrentSpeed } from './state.js';
import { fetchApi } from './api.js';
import { drawNcaGrid } from './ncaCanvasRenderer.js';
import { updateUiControls, updateNetworkLegend, updateCellDetails, applyGeneralSettings, clearCellDetailsDisplay } from './uiManager.js';
import { buildNetworkViz } from './networkVisualizer.js';
import { startRecording, stopRecording } from './recordingManager.js';
import { resetManualWeightEditorUI } from './manualWeightEditor.js';
import { renderLayerBuilder } from './layerBuilder.js'; // Needed for initial render

let animationIntervalId = null;

async function handleStep(isBack = false) {
    const endpoint = isBack ? '/api/step_back' : '/api/step';
    const data = await fetchApi(endpoint, 'POST');
    if (data) {
        drawNcaGrid(data.grid_colors);
        if (state.selectedCell) updateCellDetails(state.selectedCell.r, state.selectedCell.c);
        if (isBack && data.is_paused !== undefined && data.is_paused && state.isRunning) {
            setIsRunning(false);
            toggleRunButton.textContent = 'Start';
            toggleRunButton.classList.remove('running');
            if (animationIntervalId) clearInterval(animationIntervalId);
        }
    }
}

function startAnimationLoop() {
    if (animationIntervalId) clearInterval(animationIntervalId);
    animationIntervalId = setInterval(async () => {
        if (state.isRunning) {
            await handleStep(false);
        }
    }, state.currentSpeed);
    state.animationIntervalId = animationIntervalId; // Store in state
}

export async function loadInitialConfig() {
    const config = await fetchApi('/api/config');
    if (!config) return;

    setMaxHiddenLayersCount(config.constraints.max_hidden_layers);
    setMinNodeCountPerLayer(config.constraints.min_node_size);
    setMaxNodeCountPerLayer(config.constraints.max_node_size);

    state.gridSize = config.default_params.grid_size;
    setMlpParamsForViz(config.mlp_params_for_viz);

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

    const initialPresetName = "Flicker";
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
    renderLayerBuilder(); // Initial render of layer builder

    if (config.is_paused) {
        toggleRunButton.textContent = 'Start';
        toggleRunButton.classList.remove('running');
        setIsRunning(false);
    } else {
        toggleRunButton.textContent = 'Stop';
        toggleRunButton.classList.add('running');
        setIsRunning(true);
        startAnimationLoop();
    }
    applyManualWeightsButton.disabled = true;
    // populateManualWeightLayerSelector(); // Called by updateUiControls
}


export function setupGlobalEventListeners() {
    captureScreenshotButton.addEventListener('click', () => {
        const dataURL = ncaCanvas.toDataURL('image/png');
        const a = document.createElement('a');
        a.href = dataURL;
        const timestamp = new Date().toISOString().replace(/[:.-]/g, '');
        a.download = `canvas_screenshot_${timestamp}.png`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    });

    toggleRunButton.addEventListener('click', async () => {
        const data = await fetchApi('/api/toggle_pause', 'POST');
        if (data) {
            setIsRunning(!data.is_paused);
            if (state.isRunning) {
                toggleRunButton.textContent = 'Stop';
                toggleRunButton.classList.add('running');
                startAnimationLoop();
            } else {
                toggleRunButton.textContent = 'Start';
                toggleRunButton.classList.remove('running');
                if (state.animationIntervalId) clearInterval(state.animationIntervalId);
            }
        }
    });

    stepButton.addEventListener('click', () => handleStep(false));
    stepBackButton.addEventListener('click', () => handleStep(true));

    randomizeGridButton.addEventListener('click', async () => {
        const data = await fetchApi('/api/randomize_grid', 'POST', { seed: Date.now() });
        if (data) {
            drawNcaGrid(data.grid_colors);
            if (state.selectedCell) updateCellDetails(state.selectedCell.r, state.selectedCell.c);
            if (data.is_paused && state.isRunning) {
                setIsRunning(false);
                toggleRunButton.textContent = 'Start';
                toggleRunButton.classList.remove('running');
                if (state.animationIntervalId) clearInterval(state.animationIntervalId);
            }
        }
    });

    randomizeArchitectureButton.addEventListener('click', async () => {
        const wasRunning = state.isRunning;
        const data = await fetchApi('/api/randomize_architecture', 'POST', { was_running: wasRunning });
        if (data) {
            drawNcaGrid(data.grid_colors);
            setMlpParamsForViz(data.mlp_params_for_viz);
            updateUiControls(data.current_params, true);
            buildNetworkViz();
            updateNetworkLegend();
            if (state.selectedCell) updateCellDetails(state.selectedCell.r, state.selectedCell.c);
            if (data.is_paused) {
                setIsRunning(false);
                toggleRunButton.textContent = 'Start';
                toggleRunButton.classList.remove('running');
                if (state.animationIntervalId) clearInterval(state.animationIntervalId);
            } else {
                setIsRunning(true);
                toggleRunButton.textContent = 'Stop';
                toggleRunButton.classList.add('running');
                startAnimationLoop();
            }
            applyManualWeightsButton.disabled = true;
            // populateManualWeightLayerSelector(); // Called by updateUiControls
            resetManualWeightEditorUI();
        }
    });

    restartButton.addEventListener('click', async () => {
        const data = await fetchApi('/api/restart', 'POST');
        if (data) {
            drawNcaGrid(data.initial_grid_colors);
            setMlpParamsForViz(data.mlp_params_for_viz);
            updateUiControls(data.current_params, true);
            buildNetworkViz();
            updateNetworkLegend();
            if (state.selectedCell) updateCellDetails(state.selectedCell.r, state.selectedCell.c);
            if (data.is_paused) {
                setIsRunning(false);
                toggleRunButton.textContent = 'Start';
                toggleRunButton.classList.remove('running');
                if (state.animationIntervalId) clearInterval(state.animationIntervalId);
            } else {
                setIsRunning(true);
                toggleRunButton.textContent = 'Stop';
                toggleRunButton.classList.add('running');
                startAnimationLoop();
            }
            // applySettingsButton.disabled = true; // Removed
            applyManualWeightsButton.disabled = true;
            // populateManualWeightLayerSelector(); // Called by updateUiControls
            resetManualWeightEditorUI();
        }
    });

    activationSelector.addEventListener('change', applyGeneralSettings);
    weightScaleSlider.addEventListener('input', (e) => {
        state.weightScaleValue.textContent = parseFloat(e.target.value).toFixed(1); // Direct DOM update
        applyGeneralSettings();
    });
    biasSlider.addEventListener('input', (e) => {
        state.biasValue.textContent = parseFloat(e.target.value).toFixed(1); // Direct DOM update
        applyGeneralSettings();
    });

    colormapSelector.addEventListener('change', async (e) => {
        const newColormap = e.target.value;
        const data = await fetchApi('/api/set_colormap', 'POST', { colormap_name: newColormap });
        if (data) {
            drawNcaGrid(data.grid_colors);
            if (state.selectedCell) updateCellDetails(state.selectedCell.r, state.selectedCell.c);
        }
    });

    presetSelector.addEventListener('change', async () => {
        const selectedPresetName = presetSelector.value;
        if (selectedPresetName !== "Custom") {
            const config = await fetchApi('/api/config');
            if (config && config.presets[selectedPresetName]) {
                const [_seed, layers, act, w_scale, b_val] = config.presets[selectedPresetName];
                updateUiControls({ layer_sizes: layers, activation: act, weight_scale: w_scale, bias: b_val }, true);
                applyGeneralSettings();
            }
        } else {
            applyGeneralSettings();
        }
    });

    speedSlider.addEventListener('input', (e) => {
        setCurrentSpeed(parseInt(e.target.value));
        speedValue.textContent = state.currentSpeed;
        if (state.isRunning) startAnimationLoop();
    });
    setCurrentSpeed(parseInt(speedSlider.value));

    clearSelectionButton.addEventListener('click', clearCellDetailsDisplay);
}