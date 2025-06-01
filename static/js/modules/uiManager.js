// static/js/modules/uiManager.js

import {
    activationSelector, weightScaleSlider, weightScaleValue, biasSlider, biasValue,
    cellInfoLabel, neighborhoodDisplay, activationDisplay, networkLegend, clearSelectionButton,
    presetSelector
} from './domElements.js';
import { state, setHiddenLayerSizes, setMlpParamsForViz, setSelectedCell, setCurrentLayerActivations } from './state.js';
import { drawNcaGrid } from './ncaCanvasRenderer.js';
import { buildNetworkViz, updateNetworkNodeColors } from './networkVisualizer.js';
import { populateManualWeightLayerSelector, resetManualWeightEditorUI } from './manualWeightEditor.js'; // Will be created later

export function updateUiControls(params, fromPreset = false) {
    if (params.layer_sizes) {
        setHiddenLayerSizes(params.layer_sizes.slice(1, -1)); // [9, HL1, HL2, 1] -> [HL1, HL2]
        // renderLayerBuilder(); // This will be called from layerBuilder.js
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

    populateManualWeightLayerSelector();
    resetManualWeightEditorUI();
}

export function updateNetworkLegend() {
    if (!networkLegend) return;
    networkLegend.innerHTML = `
        <div><span class="color-box" style="background-color: rgb(255,0,0);"></span> Neg. Weight</div>
        <div><span class="color-box" style="background-color: rgb(0,255,0);"></span> Pos. Weight</div>
        <div><span class="color-box" style="background-color: rgb(255,255,255); border: 1px solid #ccc;"></span> Low Activation</div>
        <div><span class="color-box" style="background-color: rgb(0,0,255);"></span> High Activation</div>
        <div><span class="color-box" style="border: 2.5px solid magenta; background-color: white;"></span> Sel. Neuron (Edit)</div>
    `;
}

export async function updateCellDetails(r, c) {
    const { fetchApi } = await import('./api.js'); // Dynamic import to avoid circular dependency
    const data = await fetchApi(`/api/cell_details?r=${r}&c=${c}`);
    if (data) {
        setSelectedCell(data.selected_cell);
        let cellValueDisplay = "N/A";
        if (state.currentGridColors && state.currentGridColors[r] && state.currentGridColors[r][c]) {
            const hex = state.currentGridColors[r][c].substring(1);
            const R = parseInt(hex.substring(0, 2), 16) / 255;
            const G = parseInt(hex.substring(2, 4), 16) / 255;
            const B = parseInt(hex.substring(4, 6), 16) / 255;
            cellValueDisplay = ((R + G + B) / 3).toFixed(3);
        }
        cellInfoLabel.innerHTML = `Selected Cell: (Row=${r}, Col=${c})<br>Approx. Value: ${cellValueDisplay}`;

        let neighText = "Neighborhood (3x3 Input - Row Major):\n";
        data.neighborhood.forEach(row => {
            neighText += row.map(val => val.toFixed(3)).join("  ") + "\n";
        });
        neighborhoodDisplay.textContent = neighText;

        setCurrentLayerActivations(data.layer_activations);
        let actText = "Layer Activations (Input, Hidden(s), Output):\n";
        state.currentLayerActivations.forEach((layerAct, i) => {
            let actSample = layerAct.slice(0, 8).map(val => typeof val === 'number' ? val.toFixed(3) : val).join(", ");
            if (layerAct.length > 8) actSample += ", ...";
            actText += `L${i} (Size ${layerAct.length}): [${actSample}]\n`;
        });
        activationDisplay.textContent = actText;

        updateNetworkNodeColors(state.currentLayerActivations);
        clearSelectionButton.style.display = 'inline-block';
    } else {
        clearCellDetailsDisplay();
    }
}

export function clearCellDetailsDisplay() {
    cellInfoLabel.textContent = "Click on a CA cell to see details.";
    neighborhoodDisplay.textContent = "";
    activationDisplay.textContent = "";
    setSelectedCell(null);
    setCurrentLayerActivations(null);
    clearSelectionButton.style.display = 'none';
    if (state.currentGridColors) drawNcaGrid(state.currentGridColors);
    if (state.mlpParamsForViz) buildNetworkViz();
}

export function setupCollapsibleSections() {
    var coll = document.getElementsByClassName("collapsible");
    for (let i = 0; i < coll.length; i++) {
        coll[i].addEventListener("click", function () {
            this.classList.toggle("active");
            var content = this.nextElementSibling;
            if (content.style.display === "block") {
                content.style.display = "none";
            } else {
                content.style.display = "block";
            }
        });
        if (coll[i].classList.contains('active')) {
            coll[i].nextElementSibling.style.display = "block";
        }
    }
}

export async function applyGeneralSettings() {
    const { fetchApi } = await import('./api.js'); // Dynamic import
    const { renderLayerBuilder } = await import('./layerBuilder.js'); // Dynamic import

    const finalLayerSizes = [9, ...state.hiddenLayerSizes, 1];
    try {
        if (finalLayerSizes[0] !== 9 || finalLayerSizes[finalLayerSizes.length - 1] !== 1) throw new Error("Layers must start with 9 and end with 1.");
        const hl = finalLayerSizes.slice(1, -1);
        if (hl.length > state.MAX_HIDDEN_LAYERS_COUNT_FROM_BACKEND) throw new Error(`Max ${state.MAX_HIDDEN_LAYERS_COUNT_FROM_BACKEND} hidden layers.`);
        for (const size of hl) {
            if (size < state.MIN_NODE_COUNT_PER_LAYER_FROM_BACKEND || size > state.MAX_NODE_COUNT_PER_LAYER_FROM_BACKEND) {
                throw new Error(`Hidden layer size out of range (${state.MIN_NODE_COUNT_PER_LAYER_FROM_BACKEND}-${state.MAX_NODE_COUNT_PER_LAYER_FROM_BACKEND}).`);
            }
        }
    } catch (e) {
        alert(`Input Error: ${e.message}`);
        return;
    }

    const params = {
        preset_name: presetSelector.value,
        layer_sizes: finalLayerSizes.join(','),
        activation: activationSelector.value,
        weight_scale: parseFloat(weightScaleSlider.value),
        bias: parseFloat(biasSlider.value)
    };
    const data = await fetchApi('/api/apply_settings', 'POST', params);
    if (data) {
        drawNcaGrid(data.grid_colors);
        setMlpParamsForViz(data.mlp_params_for_viz);
        updateUiControls(data.current_params, data.message.includes("Preset"));
        buildNetworkViz();
        updateNetworkLegend();
        if (state.selectedCell) updateCellDetails(state.selectedCell.r, state.selectedCell.c);
        if (data.is_paused && state.isRunning) {
            state.isRunning = false; // Directly update state, event handler will update button
            // No need to clear interval here, event handler will do it
        }
        populateManualWeightLayerSelector();
        resetManualWeightEditorUI();
    }
}