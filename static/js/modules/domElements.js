// static/js/modules/domElements.js

export const ncaCanvas = document.getElementById('ncaCanvas');
export const ncaCtx = ncaCanvas.getContext('2d');
export const networkCanvas = document.getElementById('networkCanvas');
export const networkCtx = networkCanvas.getContext('2d');
export const leftPanel = document.getElementById('leftPanel');

export const toggleRunButton = document.getElementById('toggleRunButton');
export const stepButton = document.getElementById('stepButton');
export const stepBackButton = document.getElementById('stepBackButton');
export const presetSelector = document.getElementById('presetSelector');
export const colormapSelector = document.getElementById('colormapSelector');

export const captureScreenshotButton = document.getElementById('captureScreenshotButton');
export const toggleRecordingButton = document.getElementById('toggleRecordingButton');
export const recordingTimerDisplay = document.getElementById('recordingTimer');

export const layerBuilderContainer = document.getElementById('layerBuilderContainer');
export const addHiddenLayerButton = document.getElementById('addHiddenLayerButton');
export const removeHiddenLayerButton = document.getElementById('removeHiddenLayerButton');

export const activationSelector = document.getElementById('activationSelector');
export const weightScaleSlider = document.getElementById('weightScaleSlider');
export const weightScaleValue = document.getElementById('weightScaleValue');
export const biasSlider = document.getElementById('biasSlider');
export const biasValue = document.getElementById('biasValue');

export const randomizeWeightsButton = document.getElementById('randomizeWeightsButton');
export const randomizeGridButton = document.getElementById('randomizeGridButton');
export const randomizeArchitectureButton = document.getElementById('randomizeArchitectureButton');
export const speedSlider = document.getElementById('speedSlider');
export const speedValue = document.getElementById('speedValue');
export const restartButton = document.getElementById('restartButton');

// Manual Weight Editor Elements
export const manualWeightLayerSelector = document.getElementById('manualWeightLayerSelector');
export const manualWeightNeuronSelector = document.getElementById('manualWeightNeuronSelector');
export const manualWeightPresetSelector = document.getElementById('manualWeightPresetSelector');
export const manualWeightInputContainer = document.getElementById('manualWeightInputContainer');
export const manualWeightInputContainerTitle = document.getElementById('manualWeightInputContainerTitle');
export const applyManualWeightsButton = document.getElementById('applyManualWeightsButton');
export const manualWeightInfoText = document.getElementById('manualWeightInfoText');

export const cellInfoLabel = document.getElementById('cellInfoLabel');
export const neighborhoodDisplay = document.getElementById('neighborhoodDisplay');
export const activationDisplay = document.getElementById('activationDisplay');
export const networkLegend = document.getElementById('networkLegend');
export const clearSelectionButton = document.getElementById('clearSelectionButton');
export const hoverCellInfo = document.createElement('div');