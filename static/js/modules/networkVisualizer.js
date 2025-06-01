// static/js/modules/networkVisualizer.js

import { networkCanvas, networkCtx } from './domElements.js';
import { state } from './state.js';

let netNodePositions = [];

function weightToColor(w) {
    const maxVal = 3.0;
    const wClamped = Math.max(-maxVal, Math.min(w, maxVal));
    const norm = (wClamped + maxVal) / (2 * maxVal);
    const r = Math.floor((1.0 - norm) * 255);
    const g = Math.floor(norm * 255);
    return `rgb(${r},${g},0)`;
}

function weightToThickness(w) {
    const maxWidth = 3.0;
    return 1 + Math.min(Math.abs(w), 3.0) / 3.0 * (maxWidth - 1);
}

function activationToColor(a) {
    const aClamped = Math.max(0.0, Math.min(a, 1.5));
    const b = Math.floor((aClamped / 1.5) * 255);
    const r_g = 255 - b;
    return `rgb(${r_g},${r_g},255)`;
}

export function buildNetworkViz() {
    if (!state.mlpParamsForViz || !state.mlpParamsForViz.layer_sizes) {
        networkCtx.clearRect(0, 0, networkCanvas.width, networkCanvas.height);
        return;
    }

    const layerSizes = state.mlpParamsForViz.layer_sizes;
    const weights = state.mlpParamsForViz.weights;

    networkCtx.clearRect(0, 0, networkCanvas.width, networkCanvas.height);
    netNodePositions = [];

    const width = networkCanvas.width;
    const height = networkCanvas.height;
    const layerCount = layerSizes.length;
    const xSpacing = width / (layerCount + 1);

    const maxNodesInAnyLayer = Math.max(...layerSizes, 1);
    const baseRadius = 8;
    const nodeRadius = Math.max(3, Math.min(baseRadius, height / (maxNodesInAnyLayer + 2) / 2.2));


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

    if (weights) {
        for (let i = 0; i < weights.length; i++) {
            const W_i = weights[i];
            const prevLayerNodes = netNodePositions[i];
            const nextLayerNodes = netNodePositions[i + 1];
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

    for (let i = 0; i < netNodePositions.length; i++) {
        for (let j = 0; j < netNodePositions[i].length; j++) {
            const pos = netNodePositions[i][j];
            networkCtx.beginPath();
            networkCtx.arc(pos.x, pos.y, nodeRadius, 0, 2 * Math.PI);
            networkCtx.fillStyle = "white";
            networkCtx.fill();
            networkCtx.strokeStyle = "black";
            networkCtx.lineWidth = 1;

            if (state.selectedNeuronForEditing.layer === i &&
                (state.selectedNeuronForEditing.neuron === "all" || state.selectedNeuronForEditing.neuron === j)) {
                networkCtx.strokeStyle = "magenta";
                networkCtx.lineWidth = 2.5;
            }
            networkCtx.stroke();
        }
    }
    if (state.selectedCell && state.currentLayerActivations) {
        updateNetworkNodeColors(state.currentLayerActivations, nodeRadius);
    }
}

export function updateNetworkNodeColors(layerActivations, nodeRadius) {
    if (!netNodePositions || netNodePositions.length === 0 || !layerActivations || !state.mlpParamsForViz) return;
    nodeRadius = nodeRadius || Math.max(3, Math.min(8, networkCanvas.height / (Math.max(...state.mlpParamsForViz.layer_sizes, 1) + 2) / 2.2));


    for (let layerIdx = 0; layerIdx < layerActivations.length; layerIdx++) {
        if (layerIdx < netNodePositions.length) {
            const activationsInLayer = layerActivations[layerIdx];
            const nodesInLayer = netNodePositions[layerIdx];
            for (let nodeIdx = 0; nodeIdx < activationsInLayer.length; nodeIdx++) {
                if (nodeIdx < nodesInLayer.length) {
                    const val = activationsInLayer[nodeIdx];
                    const color = activationToColor(val);
                    const pos = nodesInLayer[nodeIdx];

                    networkCtx.beginPath();
                    networkCtx.arc(pos.x, pos.y, nodeRadius, 0, 2 * Math.PI);
                    networkCtx.fillStyle = color;
                    networkCtx.fill();
                    networkCtx.strokeStyle = "black";
                    networkCtx.lineWidth = 1;
                    if (state.selectedNeuronForEditing.layer === layerIdx &&
                        (state.selectedNeuronForEditing.neuron === "all" || state.selectedNeuronForEditing.neuron === nodeIdx)) {
                        networkCtx.strokeStyle = "magenta";
                        networkCtx.lineWidth = 2.5;
                    }
                    networkCtx.stroke();
                }
            }
        }
    }
}

export function setupNetworkVizResizeObserver() {
    let resizeTimeout;
    new ResizeObserver(() => {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(() => {
            const container = document.querySelector('.network-viz-container');
            if (container) {
                networkCanvas.width = container.clientWidth;
                networkCanvas.height = container.clientHeight;
                buildNetworkViz();
                if (state.selectedCell && state.currentLayerActivations) {
                    updateNetworkNodeColors(state.currentLayerActivations);
                }
            }
        }, 100);
    }).observe(document.querySelector('.network-viz-container'));
}