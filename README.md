# Neural Cellular Automata (NCA) Web Playground

This project provides a web-based interactive playground for exploring Neural Cellular Automata (NCA). Users can visualize NCA grids, experiment with different network architectures, activation functions, weights, and observe the emergent behaviors. The application includes a real-time visualization of the NCA grid and an "under the hood" look at the neural network processing for individual cells.

## Features

*   **Interactive NCA Simulation:** Start, stop, and step through the NCA simulation.
*   **Customizable NCA Parameters:**
    *   Modify neural network layer sizes.
    *   Choose activation functions (ReLU, Sigmoid, Tanh).
    *   Adjust weight scaling and bias values for the MLP.
    *   Control simulation speed.
*   **Presets:** Quickly load predefined NCA configurations to see interesting patterns (e.g., Flicker, Ripples, Bubbles, Patchy).
*   **Colormap Selection:** Choose from a variety of Matplotlib colormaps to render the CA grid.
*   **Grid Interaction:**
    *   Click on any cell in the CA grid to inspect its details.
    *   View the 3x3 neighborhood of the selected cell.
    *   See the step-by-step activation values through the neural network layers for that cell.
*   **Neural Network Visualization:**
    *   A dynamic diagram of the Multi-Layer Perceptron (MLP) used by the NCA.
    *   Weights are visualized by line color (positive/negative) and thickness (magnitude).
    *   Nodes are colored based on their activation values when a cell is selected.
    *   The network diagram is resizable and adapts to the panel size.
*   **Randomization:**
    *   Randomize MLP weights and biases with current settings.
    *   Reset the NCA grid to a new random state.
*   **Web-Based UI:** Built with Flask for the backend and HTML/CSS/JavaScript for a responsive frontend.

## Project Structure

```
nca-web-playground/
├── nca_core.py           # Core NCA and MLP class definitions
├── app.py                # Flask backend application (API endpoints, NCA instance management)
├── static/               # Static assets for the web UI
│   ├── css/
│   │   └── style.css     # Stylesheet for the frontend
│   └── js/
│       └── main.js       # JavaScript for frontend logic, drawing, API calls
└── templates/
    └── index.html        # Main HTML page for the UI
└── README.md             # This file
```

## Prerequisites

*   Python 3.7+
*   pip (Python package installer)

## Installation

1.  **Clone the repository (or create the files as provided):**
    ```bash
    git clone <your-repository-url> # If you've put it on Git
    cd nca-web-playground
    ```
    If you don't have a Git repository, simply create the directory structure and files as listed above.

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Python dependencies:**
    ```bash
    pip install Flask numpy matplotlib
    ```

## Running the Application

1.  Navigate to the root directory of the project (`nca-web-playground/`).
2.  Run the Flask application:
    ```bash
    python app.py
    ```
3.  Open your web browser and go to: `http://127.0.0.1:5000/`

The application will start, and you should see the Neural Cellular Automata playground interface.

## Usage

The web interface is divided into two main panels:

### Left Panel (CA Display & Controls)

*   **CA Canvas:** Displays the live Nellular Cellular Automaton grid. Click on cells here to inspect them.
*   **Controls:**
    *   **Start/Stop:** Toggles the continuous simulation of the NCA.
    *   **Step:** Advances the NCA by a single step (pauses if running).
    *   **Preset:** Select from a list of predefined NCA configurations. This will update layers, activation, weights, bias, and the initial grid seed.
    *   **Colormap:** Choose a Matplotlib colormap for rendering the CA grid values.
    *   **Layers:** Define the MLP architecture (e.g., `9,16,8,1`). The input layer must be 9 (for the 3x3 neighborhood) and the output layer must be 1 (for the new cell state).
    *   **Activation:** Select the activation function for hidden layers (ReLU, Sigmoid, Tanh). The output layer uses a Sigmoid to keep states between 0 and 1.
    *   **Weight Scale:** Adjust the initial range of random weights.
    *   **Bias:** Adjust the initial range of random biases.
    *   **Randomize Weights & Grid:** Applies the current layer, activation, weight, and bias settings to create a new random MLP, and also re-randomizes the grid with a new seed.
    *   **Reset Grid (New Seed):** Re-randomizes only the CA grid with a new random seed, keeping the current MLP weights.
    *   **Speed (ms delay):** Controls the delay between steps when the simulation is running.

### Right Panel (Under the Hood)

*   **Cell Info:** Displays information about the currently selected cell from the CA canvas.
    *   **Selected Cell:** Coordinates of the clicked cell.
    *   **Neighborhood:** The 3x3 grid of values surrounding the selected cell.
    *   **Activation Display:** Shows the intermediate values (pre-activation `z` and post-activation `h`) for each layer of the MLP during the forward pass for the selected cell.
*   **Network Visualization Canvas:**
    *   Displays a diagram of the MLP.
    *   Nodes represent neurons, and lines represent weights.
    *   Line color indicates weight sign (green for positive, red for negative), and thickness indicates magnitude.
    *   When a cell is clicked, the nodes in the diagram are colored according to their activation values during that cell's update calculation (white/light blue for low activation, darker blue for high activation).

## Code Overview

*   **`nca_core.py`**:
    *   `FlexibleMLP`: Defines a multi-layer perceptron with customizable layers, activation functions, and initialization parameters. Handles the forward pass.
    *   `NeuralCellularAutomaton`: Manages the grid of cells, the update rule (using an `FlexibleMLP` instance), neighborhood extraction, and stepping through generations.

*   **`app.py`**:
    *   A Flask web server that exposes API endpoints.
    *   Manages the global `NeuralCellularAutomaton` instance.
    *   Handles requests from the frontend to:
        *   Initialize/re-initialize the NCA with specific parameters or presets.
        *   Step the simulation.
        *   Toggle pause/run state.
        *   Randomize weights or reset the grid.
        *   Change colormaps.
        *   Fetch details for a selected cell (neighborhood, activations).
        *   Provide initial configuration (presets, available options).
    *   Uses Matplotlib's `get_cmap` to convert NCA states (0-1) to hex colors for display.

*   **`static/css/style.css`**:
    *   Provides the visual styling for the HTML elements, arranging the layout and appearance of controls, canvases, and text information.

*   **`static/js/main.js`**:
    *   Handles all client-side interactivity.
    *   Fetches initial configuration and populates UI controls.
    *   Sends requests to the Flask API endpoints based on user actions.
    *   Draws the NCA grid on the HTML canvas using the color data received from the backend.
    *   Draws the neural network visualization (nodes, weights) on a separate canvas.
    *   Updates the "Under the Hood" panel with cell-specific information.
    *   Manages the animation loop for continuous simulation.
    *   Handles resizing of the network visualization canvas.

*   **`templates/index.html`**:
    *   The single HTML page that structures the web interface.
    *   Defines the layout, canvases, buttons, sliders, and selectors.
    *   Links to the CSS and JavaScript files.

## Potential Future Enhancements

*   Allow users to "paint" initial states onto the grid.
*   Save and load custom NCA configurations (MLP parameters + grid state).
*   More sophisticated neighborhood types (e.g., larger, non-uniform).
*   Different update rules or multiple MLPs for different cell types.
*   Performance optimization for very large grids (e.g., using WebGL or WebAssembly for computation).
*   Graphing of cell state statistics over time.
*   Ability to "evolve" NCAs towards certain target patterns.

