/**
 * @fileoverview Defines a collection of preset grid initialization patterns for the NCA simulation.
 * Each pattern is a function that generates a grid based on the specified dimensions.
 */

/**
 * Collection of predefined grid initialization patterns.
 * Each pattern function takes `gridWidth` and `gridHeight` as arguments
 * and returns a 2D array representing the initial grid state.
 *
 * Patterns should ideally be robust to different grid dimensions or provide
 * clear feedback/default behavior if incompatible.
 */
export const gridPresets = {
    "empty": {
        name: "Empty Grid",
        description: "Initializes an empty grid (all cells off).",
        pattern: (gridWidth, gridHeight) => {
            return Array(gridHeight).fill(0).map(() => Array(gridWidth).fill(0));
        }
    },
    "center_dot": {
        name: "Center Dot",
        description: "A single active cell in the center of the grid.",
        pattern: (gridWidth, gridHeight) => {
            const grid = Array(gridHeight).fill(0).map(() => Array(gridWidth).fill(0));
            const centerX = Math.floor(gridWidth / 2);
            const centerY = Math.floor(gridHeight / 2);
            grid[centerY][centerX] = 1;
            return grid;
        }
    },
    "cross": {
        name: "Cross Pattern",
        description: "A cross shape in the center of the grid.",
        pattern: (gridWidth, gridHeight) => {
            const grid = Array(gridHeight).fill(0).map(() => Array(gridWidth).fill(0));
            const centerX = Math.floor(gridWidth / 2);
            const centerY = Math.floor(gridHeight / 2);

            for (let i = -2; i <= 2; i++) {
                if (centerY + i >= 0 && centerY + i < gridHeight) {
                    grid[centerY + i][centerX] = 1;
                }
                if (centerX + i >= 0 && centerX + i < gridWidth) {
                    grid[centerY][centerX + i] = 1;
                }
            }
            return grid;
        }
    },
    "checkerboard": {
        name: "Checkerboard",
        description: "A classic checkerboard pattern.",
        pattern: (gridWidth, gridHeight) => {
            const grid = Array(gridHeight).fill(0).map(() => Array(gridWidth).fill(0));
            for (let y = 0; y < gridHeight; y++) {
                for (let x = 0; x < gridWidth; x++) {
                    grid[y][x] = (x + y) % 2;
                }
            }
            return grid;
        }
    },
    "random_sparse": {
        name: "Random Sparse",
        description: "A grid with a low density of randomly active cells.",
        pattern: (gridWidth, gridHeight) => {
            const grid = Array(gridHeight).fill(0).map(() => Array(gridWidth).fill(0));
            const density = 0.1; // 10% active cells
            for (let y = 0; y < gridHeight; y++) {
                for (let x = 0; x < gridWidth; x++) {
                    grid[y][x] = Math.random() < density ? 1 : 0;
                }
            }
            return grid;
        }
    },
    "glider": {
        name: "Conway's Glider",
        description: "A classic glider pattern from Conway's Game of Life. Requires at least 3x3.",
        pattern: (gridWidth, gridHeight) => {
            const grid = Array(gridHeight).fill(0).map(() => Array(gridWidth).fill(0));
            if (gridWidth < 3 || gridHeight < 3) {
                console.warn("Glider pattern requires at least a 3x3 grid. Returning empty grid.");
                return grid;
            }
            // Glider pattern (top-left corner)
            grid[0][1] = 1;
            grid[1][2] = 1;
            grid[2][0] = 1;
            grid[2][1] = 1;
            grid[2][2] = 1;
            return grid;
        }
    }
};