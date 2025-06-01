// static/js/modules/ncaCanvasRenderer.js

import { ncaCanvas, ncaCtx, leftPanel, hoverCellInfo } from './domElements.js';
import { state, setSelectedCell, setCurrentGridColors } from './state.js';
import { updateCellDetails, clearCellDetailsDisplay } from './uiManager.js'; // Will be created later, but needed for dependency

export function drawNcaGrid(gridColors) {
    if (!gridColors || gridColors.length === 0) return;
    setCurrentGridColors(gridColors);
    state.gridSize = gridColors.length;
    ncaCanvas.width = state.gridSize * state.CELL_SIZE;
    ncaCanvas.height = state.gridSize * state.CELL_SIZE;

    for (let r = 0; r < state.gridSize; r++) {
        for (let c = 0; c < state.gridSize; c++) {
            ncaCtx.fillStyle = gridColors[r][c];
            ncaCtx.fillRect(c * state.CELL_SIZE, r * state.CELL_SIZE, state.CELL_SIZE, state.CELL_SIZE);
        }
    }

    if (state.selectedCell) {
        highlightNeighborhood(state.selectedCell.r, state.selectedCell.c);
    }
    leftPanel.style.width = `${ncaCanvas.width}px`;
}

export function highlightNeighborhood(r, c) {
    ncaCtx.strokeStyle = 'yellow';
    ncaCtx.lineWidth = 1.5;
    const offset = state.CELL_SIZE * 0.05;
    const size = state.CELL_SIZE * 0.9;
    for (let dr = -1; dr <= 1; dr++) {
        for (let dc = -1; dc <= 1; dc++) {
            const rr = (r + dr + state.gridSize) % state.gridSize;
            const cc = (c + dc + state.gridSize) % state.gridSize;
            ncaCtx.strokeRect(cc * state.CELL_SIZE + offset, rr * state.CELL_SIZE + offset, size, size);
        }
    }
}

export function setupNcaCanvasEvents() {
    document.body.appendChild(hoverCellInfo);
    hoverCellInfo.style.cssText = `
        position: absolute; background: rgba(0, 0, 0, 0.7); color: white;
        padding: 5px; border-radius: 3px; pointer-events: none; display: none; z-index: 100;`;

    ncaCanvas.addEventListener('click', (event) => {
        const rect = ncaCanvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        const c = Math.floor(x / state.CELL_SIZE);
        const r = Math.floor(y / state.CELL_SIZE);

        if (r >= 0 && r < state.gridSize && c >= 0 && c < state.gridSize) {
            if (state.selectedCell && state.selectedCell.r === r && state.selectedCell.c === c) {
                clearCellDetailsDisplay();
            } else {
                updateCellDetails(r, c);
                if (state.currentGridColors) drawNcaGrid(state.currentGridColors);
                highlightNeighborhood(r, c);
            }
        }
    });

    ncaCanvas.addEventListener('mousemove', (event) => {
        if (!state.currentGridColors) return;
        const rect = ncaCanvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        const c = Math.floor(x / state.CELL_SIZE);
        const r = Math.floor(y / state.CELL_SIZE);

        if (r >= 0 && r < state.gridSize && c >= 0 && c < state.gridSize) {
            const hex = state.currentGridColors[r][c].substring(1);
            const R = parseInt(hex.substring(0, 2), 16);
            const G = parseInt(hex.substring(2, 4), 16);
            const B = parseInt(hex.substring(4, 6), 16);
            const approxVal = ((R / 255 + G / 255 + B / 255) / 3).toFixed(3);
            hoverCellInfo.textContent = `(${r},${c}): ${approxVal}`;
            hoverCellInfo.style.left = `${event.pageX + 10}px`;
            hoverCellInfo.style.top = `${event.pageY + 10}px`;
            hoverCellInfo.style.display = 'block';
        } else {
            hoverCellInfo.style.display = 'none';
        }
    });

    ncaCanvas.addEventListener('mouseout', () => {
        hoverCellInfo.style.display = 'none';
    });
}