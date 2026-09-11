// static/js/app.js

import { setupNcaCanvasEvents } from './modules/ncaCanvasRenderer.js';
import { setupNetworkVizResizeObserver, buildNetworkViz } from './modules/networkVisualizer.js';
import { setupCollapsibleSections, updateNetworkLegend } from './modules/uiManager.js';
import { setupLayerBuilderEvents, renderLayerBuilder } from './modules/layerBuilder.js';
import { setupRecordingEvents } from './modules/recordingManager.js';
import { loadInitialConfig, setupGlobalEventListeners } from './modules/eventHandlers.js';
import { networkCanvas } from './modules/domElements.js';

document.addEventListener('DOMContentLoaded', () => {
    setupNcaCanvasEvents();
    setupRecordingEvents();
    setupLayerBuilderEvents();
    setupGlobalEventListeners();
    setupCollapsibleSections();

    loadInitialConfig();
    const netContainer = document.querySelector('.network-viz-container');
    if (netContainer) {
        networkCanvas.width = netContainer.clientWidth;
        networkCanvas.height = netContainer.clientHeight;
        buildNetworkViz();
    }
    updateNetworkLegend();
    renderLayerBuilder();

    setupNetworkVizResizeObserver();
});
