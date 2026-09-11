# Neural Cellular Automata Playground

An interactive, browser-native playground for exploring neural cellular automata (NCA). Each cell updates from its wrapped 3x3 neighborhood using a small configurable MLP, making it easy to see how local neural rules produce large-scale spatial dynamics.

**Live app:** https://semvdn.github.io/NCA_Playground/

<img width="1042" height="865" alt="NCA Playground screenshot" src="https://github.com/user-attachments/assets/6464c41b-5233-43d5-95a9-44cab8e0f0a9" />

## Browser-native architecture

The entire simulation runs locally in the browser. There is no Flask server, Python runtime, PyTorch process, database, or remote simulation API.

The NCA state and MLP are implemented in JavaScript using typed arrays. The UI still talks through the small `fetchApi()` abstraction, but that abstraction now dispatches to the in-browser NCA service instead of making HTTP requests. This keeps the UI modules decoupled from the simulation implementation while allowing the project to be hosted as a static site on GitHub Pages.

Because the simulation is local, changing weights, stepping the automaton, inspecting cells, and recording the canvas do not send simulation data to a server.

## What is a neural cellular automaton?

A cellular automaton updates a grid by applying the same local rule at every cell. In this playground, that local rule is neural rather than handwritten: a Multi-Layer Perceptron receives the 9 values in a cell's 3x3 neighborhood and produces the cell's next scalar state.

The grid wraps at the boundaries, so every cell always has nine inputs. Hidden layers can use ReLU, sigmoid, or tanh activations, while the final output is passed through a sigmoid to keep the state in `[0, 1]`.

## Features

- Start, stop, single-step, restart, and step backward through recent NCA states.
- Change the MLP architecture by adding/removing hidden layers and changing their widths.
- Switch activation functions and adjust weight scale and bias.
- Randomize the grid, weights, or complete architecture.
- Apply predefined grid patterns.
- Edit incoming weights for individual neurons or whole layers.
- Click any cell to inspect its 3x3 neighborhood and layer activations.
- Visualize network topology, activation values, and positive/negative weights.
- Switch among several colormaps.
- Capture screenshots and record the simulation canvas to video.
- Run entirely as a static GitHub Pages site with no build step and no runtime dependencies.

## Project structure

```text
.
├── index.html                         # Static app entry point
├── static/
│   ├── css/style.css                  # Interface styling
│   └── js/
│       ├── app.js                     # Frontend bootstrap
│       └── modules/
│           ├── api.js                 # UI-facing service adapter
│           ├── browserNcaService.js   # NCA + MLP engine and app operations
│           ├── state.js               # Shared UI state
│           ├── domElements.js         # DOM references
│           ├── eventHandlers.js       # Main interaction wiring
│           ├── gridPresets.js         # Initial grid patterns
│           ├── layerBuilder.js        # Architecture controls
│           ├── manualWeightEditor.js  # Direct neuron-weight editing
│           ├── ncaCanvasRenderer.js   # Grid rendering and selection
│           ├── networkVisualizer.js   # Network diagram
│           ├── recordingManager.js    # Canvas video capture
│           └── uiManager.js           # UI synchronization
├── .github/workflows/pages.yml        # GitHub Pages deployment
├── WEB_DEPLOYMENT.md                  # Deployment notes
└── assets/                            # Showcase media
```

## Run locally

No install or build step is required. Because the app uses JavaScript modules, serve the repository with any simple static HTTP server rather than opening `index.html` through `file://`.

For example, if Python happens to be installed:

```bash
python -m http.server 8000
```

Then open `http://localhost:8000/`.

Python is only being used here as a convenient static file server; it is not an application dependency. Any equivalent static server works.

## Deploy to GitHub Pages

The included GitHub Actions workflow deploys `index.html` and `static/` on pushes to `main`.

In the repository settings, set **Pages → Build and deployment → Source** to **GitHub Actions**. See [`WEB_DEPLOYMENT.md`](WEB_DEPLOYMENT.md) for details.

## Visual showcase

Some earlier simulations and UI captures:

https://github.com/user-attachments/assets/193f7d69-e515-4e92-ac71-ed1806af617c

https://github.com/user-attachments/assets/9955a7e6-54ed-43b2-b862-1cfa6ab4e89e

https://github.com/user-attachments/assets/9593cd5c-74c5-4153-ba6e-c907327c4107

https://github.com/user-attachments/assets/3921b359-9f6f-4fb2-a14e-8a36d1ff1c52

Web UI walkthrough: https://youtu.be/euN4uQ0BBNc

## License

See [`LICENSE`](LICENSE).
