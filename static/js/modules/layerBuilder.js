// static/js/modules/layerBuilder.js

import { layerBuilderContainer, addHiddenLayerButton, removeHiddenLayerButton, presetSelector } from './domElements.js';
import { state, setHiddenLayerSizes } from './state.js';
import { applyGeneralSettings } from './uiManager.js';

export function renderLayerBuilder() {
    layerBuilderContainer.innerHTML = '';
    state.hiddenLayerSizes.forEach((size, index) => {
        const layerDiv = document.createElement('div');
        layerDiv.classList.add('layer-input-group');
        const inputId = `hiddenLayer${index}`;
        layerDiv.innerHTML = `
            <label for="${inputId}">Hidden Layer ${index + 1} Size:</label>
            <input type="number" id="${inputId}" value="${size}"
                   min="${state.MIN_NODE_COUNT_PER_LAYER_FROM_BACKEND}" max="${state.MAX_NODE_COUNT_PER_LAYER_FROM_BACKEND}" class="hidden-layer-input">`;
        layerBuilderContainer.appendChild(layerDiv);
        layerDiv.querySelector(`#${inputId}`).addEventListener('input', (e) => {
            let value = parseInt(e.target.value);
            if (isNaN(value)) value = state.MIN_NODE_COUNT_PER_LAYER_FROM_BACKEND;
            value = Math.max(state.MIN_NODE_COUNT_PER_LAYER_FROM_BACKEND, Math.min(value, state.MAX_NODE_COUNT_PER_LAYER_FROM_BACKEND));
            e.target.value = value;
            state.hiddenLayerSizes[index] = value;
            applyGeneralSettings();
        });
    });
    addHiddenLayerButton.disabled = state.hiddenLayerSizes.length >= state.MAX_HIDDEN_LAYERS_COUNT_FROM_BACKEND;
    removeHiddenLayerButton.disabled = state.hiddenLayerSizes.length === 0;
}

export function setupLayerBuilderEvents() {
    addHiddenLayerButton.addEventListener('click', () => {
        if (state.hiddenLayerSizes.length < state.MAX_HIDDEN_LAYERS_COUNT_FROM_BACKEND) {
            state.hiddenLayerSizes.push(8);
            renderLayerBuilder();
            presetSelector.value = "Custom";
            applyGeneralSettings();
        }
    });
    removeHiddenLayerButton.addEventListener('click', () => {
        if (state.hiddenLayerSizes.length > 0) {
            state.hiddenLayerSizes.pop();
            renderLayerBuilder();
            applyGeneralSettings();
        }
    });
}