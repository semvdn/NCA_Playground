// static/js/modules/manualWeightEditor.js

import {
    manualWeightLayerSelector, manualWeightNeuronSelector, manualWeightPresetSelector,
    manualWeightInputContainer, manualWeightInputContainerTitle, applyManualWeightsButton,
    manualWeightInfoText
} from './domElements.js';
import { state, setMlpParamsForViz, setSelectedNeuronForEditing } from './state.js';
import { fetchApi } from './api.js';
import { buildNetworkViz, updateNetworkNodeColors } from './networkVisualizer.js';
import { updateNetworkLegend, updateCellDetails } from './uiManager.js';

export function populateManualWeightLayerSelector() {
    manualWeightLayerSelector.innerHTML = '<option value="">- Select Layer -</option>';
    manualWeightNeuronSelector.innerHTML = '<option value="">- Select Neuron -</option>';
    manualWeightInputContainer.innerHTML = '';
    manualWeightInputContainerTitle.textContent = '';
    applyManualWeightsButton.disabled = true;

    if (!state.mlpParamsForViz || !state.mlpParamsForViz.layer_sizes || state.mlpParamsForViz.layer_sizes.length < 2) return;

    const numMLPLayers = state.mlpParamsForViz.layer_sizes.length;
    for (let i = 1; i < numMLPLayers; i++) {
        const option = document.createElement('option');
        option.value = i;
        option.textContent = `Layer ${i} (Size: ${state.mlpParamsForViz.layer_sizes[i]})`;
        if (i === numMLPLayers - 1) option.textContent += " - Output";
        else option.textContent += " - Hidden";
        manualWeightLayerSelector.add(option);
    }
}

export function populateManualWeightNeuronSelector() {
    manualWeightNeuronSelector.innerHTML = '<option value="">- Select Neuron -</option>';
    manualWeightPresetSelector.innerHTML = '<option value="">- Select Preset -</option>';
    const selectedLayerDisplayIdx = parseInt(manualWeightLayerSelector.value);

    if (isNaN(selectedLayerDisplayIdx) || !state.mlpParamsForViz) return;

    const numNeuronsInSelectedLayer = state.mlpParamsForViz.layer_sizes[selectedLayerDisplayIdx];

    if (numNeuronsInSelectedLayer > 0) {
        manualWeightNeuronSelector.add(new Option("All Neurons in Layer", "all"));
        for (let i = 0; i < numNeuronsInSelectedLayer; i++) {
            manualWeightNeuronSelector.add(new Option(`Neuron ${i + 1}`, i));
        }
    }

    manualWeightPresetSelector.add(new Option("Zeros", "zeros"));
    manualWeightPresetSelector.add(new Option("Ones", "ones"));
    manualWeightPresetSelector.add(new Option("Random Small", "random_small"));

    const prevLayerSize = state.mlpParamsForViz.layer_sizes[selectedLayerDisplayIdx - 1];
    if (prevLayerSize === 9) {
        manualWeightPresetSelector.add(new Option("Identity (3x3)", "identity"));
        manualWeightPresetSelector.add(new Option("Edge Detect (3x3)", "edge_detect"));
        manualWeightPresetSelector.add(new Option("Sharpen (3x3)", "sharpen"));
        manualWeightPresetSelector.add(new Option("Gaussian Blur (3x3)", "gaussian_blur"));
    }

    manualWeightPresetSelector.add(new Option("Random Layer", "random_layer"));
    manualWeightPresetSelector.add(new Option("Zeros Layer", "zeros_layer"));
    manualWeightPresetSelector.add(new Option("Ones Layer", "ones_layer"));
}

function renderManualWeightInputs(weights, layerDisplayIdx, neuronDisplayIdx) {
    manualWeightInputContainer.innerHTML = '';
    manualWeightInputContainerTitle.textContent = `Editing Incoming Weights for: Layer ${layerDisplayIdx}, Neuron ${neuronDisplayIdx + 1}`;

    const prevLayerSize = state.mlpParamsForViz.layer_sizes[layerDisplayIdx - 1];
    manualWeightInfoText.textContent = `Editing ${weights.length} incoming weights from Layer ${layerDisplayIdx - 1} (Size: ${prevLayerSize}) to Layer ${layerDisplayIdx}, Neuron ${neuronDisplayIdx + 1}.`;


    weights.forEach((weight, index) => {
        const group = document.createElement('div');
        group.classList.add('weight-input-group');
        const label = document.createElement('label');
        label.textContent = `W_${index + 1}`;
        label.title = `Weight from Neuron ${index + 1} of previous layer`;

        const input = document.createElement('input');
        input.type = 'number';
        input.step = '0.01';
        input.value = parseFloat(weight).toFixed(3);
        input.classList.add('manual-weight-value');
        input.dataset.index = index;

        input.addEventListener('change', () => { applyManualWeightsButton.disabled = false; });

        group.appendChild(label);
        group.appendChild(input);
        manualWeightInputContainer.appendChild(group);
    });
}

function generateWeightPattern(patternName, numWeights) {
    let pattern = new Array(numWeights).fill(0.0);
    if (patternName === "zeros") {
        // Already filled with zeros
    } else if (patternName === "ones") {
        pattern = new Array(numWeights).fill(1.0);
    } else if (patternName === "random_small") {
        for (let i = 0; i < numWeights; i++) {
            pattern[i] = parseFloat((Math.random() * 0.2 - 0.1).toFixed(3));
        }
    } else if (patternName === "identity") {
        if (numWeights === 9) pattern = [0, 0, 0, 0, 1, 0, 0, 0, 0];
        else console.warn(`Preset 'identity' is best for 9 inputs. Using random_small.`);
    } else if (patternName === "edge_detect") {
        if (numWeights === 9) pattern = [-1, -1, -1, 0, 0, 0, 1, 1, 1];
        else console.warn(`Preset 'edge_detect' is best for 9 inputs. Using random_small.`);
    } else if (patternName === "sharpen") {
        if (numWeights === 9) pattern = [0, -1, 0, -1, 5, -1, 0, -1, 0];
        else console.warn(`Preset 'sharpen' is best for 9 inputs. Using random_small.`);
    } else if (patternName === "gaussian_blur") {
        if (numWeights === 9) pattern = [0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625];
    } else {
        console.warn(`Preset ${patternName} not found or not applicable for ${numWeights} inputs. Using random_small.`);
        for (let i = 0; i < numWeights; i++) {
            pattern[i] = parseFloat((Math.random() * 0.2 - 0.1).toFixed(3));
        }
    }
    return pattern;
}

export function resetManualWeightEditorUI() {
    manualWeightLayerSelector.value = "";
    manualWeightNeuronSelector.innerHTML = '<option value="">- Select Neuron -</option>';
    manualWeightPresetSelector.value = "";
    manualWeightInputContainer.innerHTML = "";
    manualWeightInputContainerTitle.textContent = "";
    applyManualWeightsButton.disabled = true;
    setSelectedNeuronForEditing(null, null);
    buildNetworkViz();
}

export function setupManualWeightEditorEvents() {
    manualWeightLayerSelector.addEventListener('change', () => {
        populateManualWeightNeuronSelector();
        manualWeightInputContainer.innerHTML = '';
        manualWeightInputContainerTitle.textContent = '';
        applyManualWeightsButton.disabled = true;
        setSelectedNeuronForEditing(manualWeightLayerSelector.value ? parseInt(manualWeightLayerSelector.value) : null, null);
        buildNetworkViz();
    });

    manualWeightNeuronSelector.addEventListener('change', async () => {
        const layerDisplayIdx = parseInt(manualWeightLayerSelector.value);
        const neuronVal = manualWeightNeuronSelector.value;

        setSelectedNeuronForEditing(layerDisplayIdx, (neuronVal === "all" || neuronVal === "") ? neuronVal : parseInt(neuronVal));

        buildNetworkViz();

        if (layerDisplayIdx && neuronVal !== "") {
            if (neuronVal === "all") {
                manualWeightInputContainer.innerHTML = '<p>Apply a preset pattern to all neurons in this layer.</p>';
                manualWeightInputContainerTitle.textContent = `Editing Incoming Weights for: All Neurons in Layer ${layerDisplayIdx}`;
                applyManualWeightsButton.disabled = manualWeightPresetSelector.value === "";
            } else {
                const neuronIdx = parseInt(neuronVal);
                const apiLayerIdx = layerDisplayIdx - 1;
                const data = await fetchApi(`/api/neuron_weights?layer_idx=${apiLayerIdx}&neuron_idx=${neuronIdx}`);
                if (data && data.weights) {
                    renderManualWeightInputs(data.weights, layerDisplayIdx, neuronIdx);
                }
                applyManualWeightsButton.disabled = false;
            }
        } else {
            manualWeightInputContainer.innerHTML = '';
            manualWeightInputContainerTitle.textContent = '';
            applyManualWeightsButton.disabled = true;
        }
    });

    manualWeightPresetSelector.addEventListener('change', () => {
        const layerDisplayIdx = parseInt(manualWeightLayerSelector.value);
        const neuronVal = manualWeightNeuronSelector.value;
        const preset = manualWeightPresetSelector.value;

        if (!layerDisplayIdx || neuronVal === "" || preset === "") {
            if (neuronVal === "all" && preset === "") applyManualWeightsButton.disabled = true;
            return;
        }

        const prevLayerSize = state.mlpParamsForViz.layer_sizes[layerDisplayIdx - 1];
        let pattern = generateWeightPattern(preset, prevLayerSize);

        if (neuronVal === "all") {
            applyManualWeightsButton.disabled = false;
            manualWeightInputContainer.innerHTML = `<p>Pattern '${preset}' selected. Click "Apply Manual Weights" to affect all neurons in Layer ${layerDisplayIdx}.</p>`;

        } else if (pattern) {
            renderManualWeightInputs(pattern, layerDisplayIdx, parseInt(neuronVal));
            applyManualWeightsButton.disabled = false;
        }
    });

    applyManualWeightsButton.addEventListener('click', async () => {
        const layerDisplayIdx = parseInt(manualWeightLayerSelector.value);
        const neuronVal = manualWeightNeuronSelector.value;
        const presetName = manualWeightPresetSelector.value;

        if (isNaN(layerDisplayIdx) || neuronVal === "") {
            alert("Please select a layer and a neuron (or 'All Neurons').");
            return;
        }

        let weightsToApply;
        if (neuronVal === "all") {
            if (presetName === "") {
                alert("Please select a preset pattern to apply to all neurons.");
                return;
            }
            const prevLayerSize = state.mlpParamsForViz.layer_sizes[layerDisplayIdx - 1];
            weightsToApply = generateWeightPattern(presetName, prevLayerSize);
            if (!weightsToApply) {
                alert(`Could not generate pattern '${presetName}' for an input size of ${prevLayerSize}.`);
                return;
            }
        } else {
            weightsToApply = Array.from(manualWeightInputContainer.querySelectorAll('.manual-weight-value'))
                .map(input => parseFloat(input.value));
            if (weightsToApply.some(isNaN)) {
                alert("Invalid number in manual weight inputs.");
                return;
            }
        }

        const apiLayerIdx = layerDisplayIdx - 1;

        const payload = {
            layer_idx: apiLayerIdx,
            neuron_idx: neuronVal,
            weights_pattern: weightsToApply
        };

        const data = await fetchApi('/api/neuron_weights', 'POST', payload);
        if (data) {
            setMlpParamsForViz(data.mlp_params_for_viz);
            buildNetworkViz();
            updateNetworkLegend();
            if (state.selectedCell) updateCellDetails(state.selectedCell.r, state.selectedCell.c);
            alert(data.message || "Weights applied.");
            applyManualWeightsButton.disabled = true;
            manualWeightPresetSelector.value = "";

            if (neuronVal !== "all") {
                const updatedWeightsData = await fetchApi(`/api/neuron_weights?layer_idx=${apiLayerIdx}&neuron_idx=${parseInt(neuronVal)}`);
                if (updatedWeightsData && updatedWeightsData.weights) {
                    renderManualWeightInputs(updatedWeightsData.weights, layerDisplayIdx, parseInt(neuronVal));
                }
            } else {
                manualWeightInputContainer.innerHTML = '<p>Preset applied to all neurons. Select a single neuron to see/edit its new weights.</p>';
            }
        }
    });
}