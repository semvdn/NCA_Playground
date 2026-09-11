// static/js/modules/state.js

export const state = {
    CELL_SIZE: 10,
    gridSize: 50,
    isRunning: false,
    animationIntervalId: null,
    currentFPS: 10,
    mediaRecorder: null,
    recordedChunks: [],
    isRecording: false,
    recordingStartTime: null,
    recordingTimerInterval: null,
    currentGridColors: null,
    currentGridValues: null,
    hiddenLayerSizes: [],
    maxHiddenLayersCount: 3,
    minNodeCountPerLayer: 1,
    maxNodeCountPerLayer: 32,
    mlpParamsForViz: null,
    selectedCell: null,
    currentLayerActivations: null
};

export function setCellSize(size) { state.CELL_SIZE = size; }
export function setGridSize(size) { state.gridSize = size; }
export function setIsRunning(running) { state.isRunning = running; }
export function setAnimationIntervalId(id) { state.animationIntervalId = id; }
export function setCurrentFPS(fps) { state.currentFPS = fps; }
export function setMediaRecorder(recorder) { state.mediaRecorder = recorder; }
export function setRecordedChunks(chunks) { state.recordedChunks = chunks; }
export function setIsRecording(recording) { state.isRecording = recording; }
export function setRecordingStartTime(time) { state.recordingStartTime = time; }
export function setRecordingTimerInterval(interval) { state.recordingTimerInterval = interval; }
export function setCurrentGridColors(colors) { state.currentGridColors = colors; }
export function setCurrentGridValues(values) { state.currentGridValues = values; }
export function setHiddenLayerSizes(sizes) { state.hiddenLayerSizes = sizes; }
export function setMaxHiddenLayersCount(count) { state.maxHiddenLayersCount = count; }
export function setMinNodeCountPerLayer(count) { state.minNodeCountPerLayer = count; }
export function setMaxNodeCountPerLayer(count) { state.maxNodeCountPerLayer = count; }
export function setMlpParamsForViz(params) { state.mlpParamsForViz = params; }
export function setSelectedCell(cell) { state.selectedCell = cell; }
export function setCurrentLayerActivations(activations) { state.currentLayerActivations = activations; }
