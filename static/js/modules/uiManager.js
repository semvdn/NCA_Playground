// static/js/modules/uiManager.js

import {
    activationSelector, weightScaleSlider, weightScaleValue, biasSlider, biasValue,
    cellInfoLabel, neighborhoodDisplay, activationDisplay, networkLegend, clearSelectionButton,
    presetSelector
} from './domElements.js';
import { state, setHiddenLayerSizes, setMlpParamsForViz, setSelectedCell, setCurrentLayerActivations } from './state.js';
import { drawNcaGrid } from './ncaCanvasRenderer.js';
import { buildNetworkViz, updateNetworkNodeColors } from './networkVisualizer.js';

export function updateUiControls(params) {
    if (params.layer_sizes) setHiddenLayerSizes(params.layer_sizes.slice(1, -1));
    if (params.activation) activationSelector.value = params.activation;
    if (params.weight_scale !== undefined) {
        weightScaleSlider.value = params.weight_scale;
        weightScaleValue.textContent = Number(params.weight_scale).toFixed(1);
    }
    if (params.bias !== undefined) {
        biasSlider.value = params.bias;
        biasValue.textContent = Number(params.bias).toFixed(1);
    }
}

export function updateNetworkLegend() {
    if (!networkLegend) return;
    networkLegend.innerHTML = `
        <div><span class="color-box" style="background-color: rgb(255,0,0);"></span> Neg. Weight</div>
        <div><span class="color-box" style="background-color: rgb(0,255,0);"></span> Pos. Weight</div>
        <div><span class="color-box" style="background-color: rgb(255,255,255); border: 1px solid #ccc;"></span> Low Activation</div>
        <div><span class="color-box" style="background-color: rgb(0,0,255);"></span> High Activation</div>
    `;
}

export async function updateCellDetails(r, c) {
    const { fetchApi } = await import('./api.js');
    const data = await fetchApi(`/api/cell_details?r=${r}&c=${c}`);
    if (!data) {
        clearCellDetailsDisplay();
        return;
    }

    setSelectedCell(data.selected_cell);
    const cellValueDisplay = Number.isFinite(data.cell_value) ? data.cell_value.toFixed(6) : 'N/A';
    cellInfoLabel.innerHTML = `Selected Cell: (Row=${r}, Col=${c})<br>Value: ${cellValueDisplay}`;

    let neighborhoodText = 'Neighborhood (3x3 Input - Row Major):\n';
    data.neighborhood.forEach(row => {
        neighborhoodText += row.map(value => value.toFixed(3)).join('  ') + '\n';
    });
    neighborhoodDisplay.textContent = neighborhoodText;

    setCurrentLayerActivations(data.layer_activations);
    let activationText = 'Layer Activations (Input, Hidden(s), Output):\n';
    state.currentLayerActivations.forEach((layer, index) => {
        let sample = layer.slice(0, 8).map(value => typeof value === 'number' ? value.toFixed(3) : value).join(', ');
        if (layer.length > 8) sample += ', ...';
        activationText += `L${index} (Size ${layer.length}): [${sample}]\n`;
    });
    activationDisplay.textContent = activationText;

    updateNetworkNodeColors(state.currentLayerActivations);
    clearSelectionButton.style.display = 'inline-block';
}

export function clearCellDetailsDisplay() {
    cellInfoLabel.textContent = 'Click on a CA cell to see details.';
    neighborhoodDisplay.textContent = '';
    activationDisplay.textContent = '';
    setSelectedCell(null);
    setCurrentLayerActivations(null);
    clearSelectionButton.style.display = 'none';
    if (state.currentGridColors) drawNcaGrid(state.currentGridColors, state.currentGridValues);
    if (state.mlpParamsForViz) buildNetworkViz();
}

export function setupCollapsibleSections() {
    for (const button of document.getElementsByClassName('collapsible')) {
        button.addEventListener('click', function () {
            this.classList.toggle('active');
            const content = this.nextElementSibling;
            content.style.display = content.style.display === 'block' ? 'none' : 'block';
        });
        if (button.classList.contains('active')) button.nextElementSibling.style.display = 'block';
    }
}

export async function applyGeneralSettings() {
    const { fetchApi } = await import('./api.js');
    const finalLayerSizes = [9, ...state.hiddenLayerSizes, 1];

    try {
        const hidden = finalLayerSizes.slice(1, -1);
        if (hidden.length > state.maxHiddenLayersCount) throw new Error(`Max ${state.maxHiddenLayersCount} hidden layers.`);
        for (const size of hidden) {
            if (size < state.minNodeCountPerLayer || size > state.maxNodeCountPerLayer) {
                throw new Error(`Hidden layer size out of range (${state.minNodeCountPerLayer}-${state.maxNodeCountPerLayer}).`);
            }
        }
    } catch (error) {
        alert(`Input Error: ${error.message}`);
        return null;
    }

    const data = await fetchApi('/api/apply_settings', 'POST', {
        preset_name: presetSelector.value,
        layer_sizes: finalLayerSizes.join(','),
        activation: activationSelector.value,
        weight_scale: Number.parseFloat(weightScaleSlider.value),
        bias: Number.parseFloat(biasSlider.value)
    });

    if (data) {
        drawNcaGrid(data.grid_colors, data.grid_values);
        setMlpParamsForViz(data.mlp_params_for_viz);
        updateUiControls(data.current_params);
        buildNetworkViz();
        updateNetworkLegend();
        if (state.selectedCell) updateCellDetails(state.selectedCell.r, state.selectedCell.c);
    }
    return data;
}
