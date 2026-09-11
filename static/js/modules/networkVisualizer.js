// static/js/modules/networkVisualizer.js

import { networkCanvas, networkCtx } from './domElements.js';
import { state } from './state.js';

let netNodePositions = [];

function weightToColor(weight) {
    const maxVal = 3.0;
    const clamped = Math.max(-maxVal, Math.min(weight, maxVal));
    const norm = (clamped + maxVal) / (2 * maxVal);
    const red = Math.floor((1 - norm) * 255);
    const green = Math.floor(norm * 255);
    return `rgb(${red},${green},0)`;
}

function weightToThickness(weight) {
    return 1 + (Math.min(Math.abs(weight), 3) / 3) * 2;
}

function activationToColor(activation) {
    const clamped = Math.max(0, Math.min(activation, 1.5));
    const blue = Math.floor((clamped / 1.5) * 255);
    const redGreen = 255 - blue;
    return `rgb(${redGreen},${redGreen},255)`;
}

function nodeRadius() {
    const sizes = state.mlpParamsForViz?.layer_sizes || [1];
    return Math.max(3, Math.min(8, networkCanvas.height / (Math.max(...sizes, 1) + 2) / 2.2));
}

export function buildNetworkViz() {
    if (!state.mlpParamsForViz?.layer_sizes) {
        networkCtx.clearRect(0, 0, networkCanvas.width, networkCanvas.height);
        return;
    }

    const layerSizes = state.mlpParamsForViz.layer_sizes;
    const weights = state.mlpParamsForViz.weights;
    networkCtx.clearRect(0, 0, networkCanvas.width, networkCanvas.height);
    netNodePositions = [];

    const xSpacing = networkCanvas.width / (layerSizes.length + 1);
    const radius = nodeRadius();

    for (let layerIdx = 0; layerIdx < layerSizes.length; layerIdx++) {
        const count = layerSizes[layerIdx];
        const x = (layerIdx + 1) * xSpacing;
        const ySpacing = networkCanvas.height / (count + 1);
        const positions = [];
        for (let nodeIdx = 0; nodeIdx < count; nodeIdx++) {
            positions.push({ x, y: (nodeIdx + 1) * ySpacing });
        }
        netNodePositions.push(positions);
    }

    if (weights) {
        for (let layerIdx = 0; layerIdx < weights.length; layerIdx++) {
            const matrix = weights[layerIdx];
            for (let inIdx = 0; inIdx < matrix.length; inIdx++) {
                for (let outIdx = 0; outIdx < matrix[inIdx].length; outIdx++) {
                    const weight = matrix[inIdx][outIdx];
                    networkCtx.beginPath();
                    networkCtx.moveTo(netNodePositions[layerIdx][inIdx].x, netNodePositions[layerIdx][inIdx].y);
                    networkCtx.lineTo(netNodePositions[layerIdx + 1][outIdx].x, netNodePositions[layerIdx + 1][outIdx].y);
                    networkCtx.strokeStyle = weightToColor(weight);
                    networkCtx.lineWidth = weightToThickness(weight);
                    networkCtx.stroke();
                }
            }
        }
    }

    for (const layer of netNodePositions) {
        for (const position of layer) {
            networkCtx.beginPath();
            networkCtx.arc(position.x, position.y, radius, 0, 2 * Math.PI);
            networkCtx.fillStyle = 'white';
            networkCtx.fill();
            networkCtx.strokeStyle = 'black';
            networkCtx.lineWidth = 1;
            networkCtx.stroke();
        }
    }

    if (state.selectedCell && state.currentLayerActivations) {
        updateNetworkNodeColors(state.currentLayerActivations, radius);
    }
}

export function updateNetworkNodeColors(layerActivations, radius = nodeRadius()) {
    if (!netNodePositions.length || !layerActivations || !state.mlpParamsForViz) return;

    for (let layerIdx = 0; layerIdx < layerActivations.length && layerIdx < netNodePositions.length; layerIdx++) {
        const activations = layerActivations[layerIdx];
        const nodes = netNodePositions[layerIdx];
        for (let nodeIdx = 0; nodeIdx < activations.length && nodeIdx < nodes.length; nodeIdx++) {
            const position = nodes[nodeIdx];
            networkCtx.beginPath();
            networkCtx.arc(position.x, position.y, radius, 0, 2 * Math.PI);
            networkCtx.fillStyle = activationToColor(activations[nodeIdx]);
            networkCtx.fill();
            networkCtx.strokeStyle = 'black';
            networkCtx.lineWidth = 1;
            networkCtx.stroke();
        }
    }
}

export function setupNetworkVizResizeObserver() {
    const container = document.querySelector('.network-viz-container');
    if (!container || !globalThis.ResizeObserver) return;

    let resizeTimeout;
    new ResizeObserver(() => {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(() => {
            networkCanvas.width = container.clientWidth;
            networkCanvas.height = container.clientHeight;
            buildNetworkViz();
        }, 100);
    }).observe(container);
}
