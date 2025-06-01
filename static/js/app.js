// static/js/app.js

import { setupNcaCanvasEvents } from './modules/ncaCanvasRenderer.js';
import { setupNetworkVizResizeObserver, buildNetworkViz } from './modules/networkVisualizer.js';
import { setupCollapsibleSections, updateNetworkLegend } from './modules/uiManager.js';
import { setupLayerBuilderEvents, renderLayerBuilder } from './modules/layerBuilder.js';
import { setupRecordingEvents } from './modules/recordingManager.js';
import { setupManualWeightEditorEvents, populateManualWeightLayerSelector } from './modules/manualWeightEditor.js';
import { loadInitialConfig, setupGlobalEventListeners } from './modules/eventHandlers.js';
import { networkCanvas } from './modules/domElements.js';

document.addEventListener('DOMContentLoaded', () => {
    // Setup all event listeners and initial configurations
    setupNcaCanvasEvents();
    setupRecordingEvents();
    setupLayerBuilderEvents();
    setupManualWeightEditorEvents();
    setupGlobalEventListeners();
    setupCollapsibleSections();

    // Load initial configuration and render UI
    loadInitialConfig().then(() => {
        const netContainer = document.querySelector('.network-viz-container');
        if (netContainer) {
            networkCanvas.width = netContainer.clientWidth;
            networkCanvas.height = netContainer.clientHeight;
            buildNetworkViz();
        }
        updateNetworkLegend();
        renderLayerBuilder(); // Ensure layer builder is rendered after initial config
        populateManualWeightLayerSelector(); // Ensure manual weight editor is populated
    });

    // Setup resize observer for network visualization
    setupNetworkVizResizeObserver();
});