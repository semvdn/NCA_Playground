// static/js/modules/eventHandlers.js

import {
    ncaCanvas, toggleRunButton, stepButton, stepBackButton, randomizeGridButton,
    randomizeArchitectureButton, restartButton, randomizeWeightsButton,
    captureScreenshotButton, activationSelector, weightScaleSlider, weightScaleValue,
    biasSlider, biasValue, colormapSelector, presetSelector, speedSlider, speedValue,
    clearSelectionButton, presetGridPatternSelector, applyPresetGridPatternButton
} from './domElements.js';
import {
    state, setIsRunning, setAnimationIntervalId, setMlpParamsForViz,
    setMaxHiddenLayersCount, setMinNodeCountPerLayer, setMaxNodeCountPerLayer,
    setCurrentFPS
} from './state.js';
import { browserNcaService } from './browserNcaService.js';
import { drawNcaGrid } from './ncaCanvasRenderer.js';
import { updateUiControls, updateNetworkLegend, updateCellDetails, applyGeneralSettings, clearCellDetailsDisplay } from './uiManager.js';
import { buildNetworkViz } from './networkVisualizer.js';
import { renderLayerBuilder } from './layerBuilder.js';
import { gridPresets } from './gridPresets.js';

let animationIntervalId = null;

function callService(description, action) {
    try {
        return action();
    } catch (error) {
        console.error(`NCA service error while ${description}:`, error);
        alert(`Error: ${error.message}`);
        return null;
    }
}

function stopAnimationLoop() {
    if (animationIntervalId) clearInterval(animationIntervalId);
    animationIntervalId = null;
    setAnimationIntervalId(null);
}

function updateRunButton() {
    toggleRunButton.textContent = state.isRunning ? 'Stop' : 'Start';
    toggleRunButton.classList.toggle('running', state.isRunning);
}

function setRunning(running) {
    setIsRunning(Boolean(running));
    updateRunButton();
    if (state.isRunning) startAnimationLoop();
    else stopAnimationLoop();
}

function syncRunState(isPaused) {
    if (typeof isPaused === 'boolean') setRunning(!isPaused);
}

function handleStep(isBack = false) {
    const data = callService(
        isBack ? 'stepping backward' : 'stepping the simulation',
        () => isBack ? browserNcaService.stepBack() : browserNcaService.step()
    );
    if (!data) return;

    drawNcaGrid(data.grid_colors, data.grid_values);
    if (state.selectedCell) updateCellDetails(state.selectedCell.r, state.selectedCell.c);
    if (isBack) syncRunState(data.is_paused);
}

function startAnimationLoop() {
    stopAnimationLoop();
    animationIntervalId = setInterval(() => {
        if (state.isRunning) handleStep(false);
    }, 1000 / state.currentFPS);
    setAnimationIntervalId(animationIntervalId);
}

function useCustomConfiguration() {
    presetSelector.value = 'Custom';
}

export function loadInitialConfig() {
    const config = callService('loading the initial configuration', () => browserNcaService.getConfig());
    if (!config) return;

    setMaxHiddenLayersCount(config.constraints.max_hidden_layers);
    setMinNodeCountPerLayer(config.constraints.min_node_size);
    setMaxNodeCountPerLayer(config.constraints.max_node_size);
    state.gridSize = config.default_params.grid_size;
    setMlpParamsForViz(config.mlp_params_for_viz);

    presetSelector.innerHTML = '';
    Object.keys(config.presets).forEach(name => presetSelector.add(new Option(name, name)));
    activationSelector.innerHTML = '';
    config.available_activations.forEach(name => activationSelector.add(new Option(name, name)));
    colormapSelector.innerHTML = '';
    config.available_colormaps.forEach(name => colormapSelector.add(new Option(name, name)));
    colormapSelector.value = config.current_colormap;

    const initialPresetName = config.presets.Linear ? 'Linear' : 'Custom';
    presetSelector.value = initialPresetName;
    const preset = config.presets[initialPresetName];
    if (preset) {
        const [, layers, activation, weightScale, bias] = preset;
        updateUiControls({ layer_sizes: layers, activation, weight_scale: weightScale, bias });
    } else {
        updateUiControls(config.default_params);
    }

    drawNcaGrid(config.initial_grid_colors, config.initial_grid_values);
    buildNetworkViz();
    updateNetworkLegend();
    renderLayerBuilder();
    setRunning(!config.is_paused);
}

export function setupGlobalEventListeners() {
    for (const key in gridPresets) {
        const option = document.createElement('option');
        option.value = key;
        option.textContent = gridPresets[key].name;
        presetGridPatternSelector.appendChild(option);
    }

    captureScreenshotButton.addEventListener('click', () => {
        const anchor = document.createElement('a');
        anchor.href = ncaCanvas.toDataURL('image/png');
        anchor.download = `canvas_screenshot_${new Date().toISOString().replace(/[:.-]/g, '')}.png`;
        document.body.appendChild(anchor);
        anchor.click();
        anchor.remove();
    });

    toggleRunButton.addEventListener('click', () => {
        const data = callService('toggling simulation playback', () => browserNcaService.togglePause());
        if (data) syncRunState(data.is_paused);
    });

    stepButton.addEventListener('click', () => handleStep(false));
    stepBackButton.addEventListener('click', () => handleStep(true));

    randomizeGridButton.addEventListener('click', () => {
        const data = callService('randomizing the grid', () => browserNcaService.randomizeGrid({ seed: Date.now() }));
        if (!data) return;
        drawNcaGrid(data.grid_colors, data.grid_values);
        if (state.selectedCell) updateCellDetails(state.selectedCell.r, state.selectedCell.c);
        syncRunState(data.is_paused);
    });

    randomizeArchitectureButton.addEventListener('click', () => {
        const data = callService(
            'randomizing the architecture',
            () => browserNcaService.randomizeArchitecture({ was_running: state.isRunning })
        );
        if (!data) return;
        drawNcaGrid(data.grid_colors, data.grid_values);
        setMlpParamsForViz(data.mlp_params_for_viz);
        updateUiControls(data.current_params);
        presetSelector.value = 'Custom';
        renderLayerBuilder();
        buildNetworkViz();
        updateNetworkLegend();
        if (state.selectedCell) updateCellDetails(state.selectedCell.r, state.selectedCell.c);
        syncRunState(data.is_paused);
    });

    restartButton.addEventListener('click', () => {
        const data = callService('restarting the simulation', () => browserNcaService.restart());
        if (!data) return;
        drawNcaGrid(data.initial_grid_colors, data.initial_grid_values);
        setMlpParamsForViz(data.mlp_params_for_viz);
        updateUiControls(data.current_params);
        renderLayerBuilder();
        buildNetworkViz();
        updateNetworkLegend();
        if (state.selectedCell) updateCellDetails(state.selectedCell.r, state.selectedCell.c);
        syncRunState(data.is_paused);
    });

    randomizeWeightsButton.addEventListener('click', () => {
        const data = callService('randomizing network weights', () => browserNcaService.randomizeWeights());
        if (!data) return;
        drawNcaGrid(data.grid_colors, data.grid_values);
        setMlpParamsForViz(data.mlp_params_for_viz);
        updateUiControls(data.current_params);
        buildNetworkViz();
        updateNetworkLegend();
        if (state.selectedCell) updateCellDetails(state.selectedCell.r, state.selectedCell.c);
        syncRunState(data.is_paused);
    });

    activationSelector.addEventListener('change', () => {
        useCustomConfiguration();
        applyGeneralSettings();
    });
    weightScaleSlider.addEventListener('input', event => {
        weightScaleValue.textContent = Number.parseFloat(event.target.value).toFixed(1);
        useCustomConfiguration();
        applyGeneralSettings();
    });
    biasSlider.addEventListener('input', event => {
        biasValue.textContent = Number.parseFloat(event.target.value).toFixed(1);
        useCustomConfiguration();
        applyGeneralSettings();
    });

    colormapSelector.addEventListener('change', event => {
        const data = callService('changing the colormap', () => browserNcaService.setColormap(event.target.value));
        if (!data) return;
        drawNcaGrid(data.grid_colors, data.grid_values);
        if (state.selectedCell) updateCellDetails(state.selectedCell.r, state.selectedCell.c);
    });

    presetSelector.addEventListener('change', () => {
        const selected = presetSelector.value;
        if (selected === 'Custom') return;
        const config = callService('loading preset configuration', () => browserNcaService.getConfig());
        if (!config?.presets[selected]) return;
        const [, layers, activation, weightScale, bias] = config.presets[selected];
        updateUiControls({ layer_sizes: layers, activation, weight_scale: weightScale, bias });
        renderLayerBuilder();
        applyGeneralSettings();
    });

    speedSlider.addEventListener('input', event => {
        setCurrentFPS(Number.parseInt(event.target.value, 10));
        speedValue.textContent = state.currentFPS;
        if (state.isRunning) startAnimationLoop();
    });
    setCurrentFPS(Number.parseInt(speedSlider.value, 10));

    clearSelectionButton.addEventListener('click', clearCellDetailsDisplay);

    applyPresetGridPatternButton.addEventListener('click', () => {
        const selected = presetGridPatternSelector.value;
        if (!selected) {
            alert('Please select a grid pattern to apply.');
            return;
        }
        const pattern = gridPresets[selected];
        if (!pattern) return;

        const data = callService(
            'applying the grid preset',
            () => browserNcaService.setGridState({
                grid_state: pattern.pattern(state.gridSize, state.gridSize),
                was_running: state.isRunning
            })
        );
        if (!data) return;
        drawNcaGrid(data.grid_colors, data.grid_values);
        if (state.selectedCell) updateCellDetails(state.selectedCell.r, state.selectedCell.c);
        syncRunState(data.is_paused);
    });
}
