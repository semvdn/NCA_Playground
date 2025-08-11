# Neural Cellular Automata Web UI

This project provides an interactive web-based interface for exploring Neural Cellular Automata (NCA). It allows users to visualize the evolution of a cellular automaton driven by a small neural network, experiment with different network architectures, activation functions, and weights, and observe their impact on the grid's behavior.

<img width="1042" height="865" alt="screenshot" src="https://github.com/user-attachments/assets/6464c41b-5233-43d5-95a9-44cab8e0f0a9" />


## What are Neural Cellular Automata?

Neural Cellular Automata are a fascinating class of systems where each cell in a grid updates its state based on the states of its immediate neighbors, using a neural network as the update rule. This project implements a basic NCA where each cell's value (a float between 0 and 1) is determined by a Multi-Layer Perceptron (MLP) that takes the 3x3 neighborhood of the cell as input.

## Features

*   **Interactive Simulation:** Start, stop, step forward, and step backward through the NCA evolution.
*   **Customizable Architecture:**
    *   Select from predefined MLP presets (Linear, Shallow ReLU, Deep Tanh, Wide Sigmoid, Custom).
    *   Dynamically add/remove hidden layers and adjust node counts within constraints.
    *   Choose activation functions (ReLU, Sigmoid, Tanh).
    *   Adjust initial weight scale and bias for the neural network.
*   **Randomization Tools:**
    *   Randomize the grid state.
    *   Randomize network weights based on current architecture.
    *   Randomize the entire network architecture (layers, nodes, activation, weights).
*   **Visualization Options:**
    *   Select from various colormaps to visualize cell states.
    *   Adjust simulation speed.
*   **Inspection Tools:**
    *   Click on any cell to inspect its 3x3 neighborhood input and the activations of each layer in the neural network for that specific cell.
    *   Visualize the neural network architecture with color-coded weights (green for positive, red for negative).
*   **Manual Weight Editor:** Directly modify the incoming weights for individual neurons or apply preset patterns to all neurons in a selected layer.
*   **Capture Tools:**
    *   Capture screenshots of the NCA grid.
    *   Record video of the simulation.

## Visual Showcase

Here are videos demonstrating the Neural Cellular Automata Web UI in action:

#### Canvas Simulations



https://github.com/user-attachments/assets/193f7d69-e515-4e92-ac71-ed1806af617c



https://github.com/user-attachments/assets/9955a7e6-54ed-43b2-b862-1cfa6ab4e89e



https://github.com/user-attachments/assets/9593cd5c-74c5-4153-ba6e-c907327c4107



https://github.com/user-attachments/assets/3921b359-9f6f-4fb2-a14e-8a36d1ff1c52



#### Web UI Walkthrough

https://youtu.be/euN4uQ0BBNc

## Project Structure

*   [`app.py`](app.py): The main Flask application file. It handles web routes, manages the NCA simulation, processes user requests, and serves the frontend.
*   [`config.py`](config.py): Contains configuration settings for the Flask application and NCA parameters.
*   [`nca_core.py`](nca_core.py): Contains the core logic for the Neural Cellular Automaton and the Flexible Multi-Layer Perceptron (MLP). This file defines how the NCA steps and how the neural network processes inputs.
*   [`services/`](services/):
    *   [`nca_service.py`](services/nca_service.py): Provides a service layer for NCA-related operations, abstracting core logic from the Flask app.
*   [`utils/`](utils/):
    *   [`validation.py`](utils/validation.py): Contains utility functions for data validation.
    *   [`visualization.py`](utils/visualization.py): Contains utility functions for visualization-related tasks, such as colormap generation.
*   [`templates/`](templates/):
    *   [`index.html`](templates/index.html): The main HTML template for the web user interface.
*   [`static/`](static/):
    *   [`css/style.css`](static/css/style.css): Contains the CSS styles for the web interface.
    *   [`js/app.js`](static/js/app.js): The main entry point for the frontend JavaScript, responsible for initializing all modules and setting up the application.
    *   [`js/modules/`](static/js/modules/): Contains modularized JavaScript files, each encapsulating specific functionalities:
        *   [`api.js`](static/js/modules/api.js): Handles all interactions with the backend API.
        *   [`state.js`](static/js/modules/state.js): Manages global application state.
        *   [`domElements.js`](static/js/modules/domElements.js): Centralizes DOM element selections.
        *   [`ncaCanvasRenderer.js`](static/js/modules/ncaCanvasRenderer.js): Manages drawing and interactions on the NCA grid canvas.
        *   [`networkVisualizer.js`](static/js/modules/networkVisualizer.js): Handles the rendering and interactions of the neural network visualization.
        *   [`uiManager.js`](static/js/modules/uiManager.js): Manages general UI updates and interactions.
        *   [`layerBuilder.js`](static/js/modules/layerBuilder.js): Manages the dynamic hidden layer builder UI.
        *   [`recordingManager.js`](static/js/modules/recordingManager.js): Handles video recording functionality.
        *   [`manualWeightEditor.js`](static/js/modules/manualWeightEditor.js): Encapsulates the logic for the manual neuron weight editor.
        *   [`eventHandlers.js`](static/js/modules/eventHandlers.js): Consolidates primary event listeners for UI interactions.
*   [`requirements.txt`](requirements.txt): Lists the Python dependencies required to run the application.

## Setup and Running

To set up and run this project locally, follow these steps:

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    *   **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    *   **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install the required Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    The dependencies include:
    *   `Flask`: Web framework for the backend.
    *   `numpy`: For numerical operations, especially array manipulations in NCA and MLP.
    *   `matplotlib`: Used for colormap generation on the backend.

5.  **Run the Flask application:**
    ```bash
    python app.py
    ```
    The application will typically run on `http://127.0.0.1:5000/` (or `localhost:5000`).

6.  **Open in your browser:**
    Navigate to `http://127.0.0.1:5000/` in your web browser to access the Neural Cellular Automata Playground.

## Usage

Once the application is running and you've opened it in your browser, you can:

*   Use the **Simulation Controls** to start/stop the simulation, step through it, or restart.
*   Adjust the **Speed** slider to control the animation rate.
*   Select different **Colormaps** to change the visual representation of the cell states.
*   Utilize the **Randomizer** buttons to generate new grids, randomize network weights, or create entirely new network architectures.
*   In the **NCA Architecture** section, choose **Presets** or configure a **Custom** network by adjusting hidden layer sizes, activation functions, weight scale, and bias. Changes are applied immediately.
*   Explore the **Manual Neuron Weight Editor** to fine-tune individual neuron weights or apply predefined patterns.
*   Click on any cell in the main grid to view its **Neighborhood** input and the **Layer Activations** within the neural network in the "Under the Hood" panel.
*   Observe the **Network Visualization** to see the MLP structure and how weights are distributed.
*   Use the **Capture Tools** to save screenshots or record videos of your simulations.
