// static/js/modules/layerBuilder.js

import { layerBuilderContainer, addHiddenLayerButton, removeHiddenLayerButton, presetSelector } from './domElements.js';
import { state } from './state.js';
import { applyGeneralSettings } from './uiManager.js';

function useCustomConfiguration() {
    presetSelector.value = 'Custom';
}

export function renderLayerBuilder() {
    layerBuilderContainer.innerHTML = '';
    state.hiddenLayerSizes.forEach((size, index) => {
        const layerDiv = document.createElement('div');
        layerDiv.classList.add('layer-input-group');
        const inputId = `hiddenLayer${index}`;
        layerDiv.innerHTML = `
            <label for="${inputId}">Hidden Layer ${index + 1} Size:</label>
            <input type="number" id="${inputId}" value="${size}"
                   min="${state.minNodeCountPerLayer}" max="${state.maxNodeCountPerLayer}" class="hidden-layer-input">`;
        layerBuilderContainer.appendChild(layerDiv);

        layerDiv.querySelector(`#${inputId}`).addEventListener('change', (event) => {
            let value = Number.parseInt(event.target.value, 10);
            if (Number.isNaN(value)) value = state.minNodeCountPerLayer;
            value = Math.max(state.minNodeCountPerLayer, Math.min(value, state.maxNodeCountPerLayer));
            event.target.value = value;
            state.hiddenLayerSizes[index] = value;
            useCustomConfiguration();
            applyGeneralSettings();
        });
    });

    addHiddenLayerButton.disabled = state.hiddenLayerSizes.length >= state.maxHiddenLayersCount;
    removeHiddenLayerButton.disabled = state.hiddenLayerSizes.length === 0;
}

export function setupLayerBuilderEvents() {
    addHiddenLayerButton.addEventListener('click', () => {
        if (state.hiddenLayerSizes.length < state.maxHiddenLayersCount) {
            state.hiddenLayerSizes.push(8);
            useCustomConfiguration();
            renderLayerBuilder();
            applyGeneralSettings();
        }
    });

    removeHiddenLayerButton.addEventListener('click', () => {
        if (state.hiddenLayerSizes.length > 0) {
            state.hiddenLayerSizes.pop();
            useCustomConfiguration();
            renderLayerBuilder();
            applyGeneralSettings();
        }
    });
}
