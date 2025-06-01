# app.py
from flask import Flask, render_template, jsonify, request
from services.nca_service import NCAService
from config import MAX_HIDDEN_LAYERS_COUNT, MIN_NODE_COUNT_PER_LAYER, MAX_NODE_COUNT_PER_LAYER

app = Flask(__name__)

# Initialize NCA Service
nca_service = NCAService()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/config', methods=['GET'])
def get_config():
    try:
        config_data = nca_service.get_nca_config()
        return jsonify(config_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/step', methods=['POST'])
def step_nca():
    try:
        result = nca_service.step_nca_simulation()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/step_back', methods=['POST'])
def step_back_nca():
    try:
        result = nca_service.step_back_nca_simulation()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/toggle_pause', methods=['POST'])
def toggle_pause():
    try:
        result = nca_service.toggle_nca_pause()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/apply_settings', methods=['POST'])
def apply_settings():
    data = request.json
    try:
        result = nca_service.apply_nca_settings(data)
        return jsonify(result)
    except (ValueError, KeyError) as e:
        return jsonify({"error": f"Invalid parameters: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/set_colormap', methods=['POST'])
def set_colormap_route():
    data = request.json
    colormap_name = data.get("colormap_name")
    try:
        result = nca_service.set_nca_colormap(colormap_name)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/randomize_weights', methods=['POST'])
def randomize_weights_route():
    data = request.json
    try:
        result = nca_service.randomize_nca_weights(data)
        return jsonify(result)
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid parameters for randomizing weights: {e}"}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500

@app.route('/api/randomize_grid', methods=['POST'])
def randomize_grid_route():
    data = request.json
    try:
        result = nca_service.randomize_nca_grid(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/randomize_architecture', methods=['POST'])
def randomize_architecture_route():
    data = request.json
    try:
        result = nca_service.randomize_nca_architecture(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Failed to randomize architecture: {e}"}), 500

@app.route('/api/restart', methods=['POST'])
def restart_nca():
    try:
        result = nca_service.restart_nca_simulation()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/neuron_weights', methods=['GET', 'POST'])
def neuron_weights_route():
    if request.method == 'GET':
        try:
            layer_idx = int(request.args.get('layer_idx'))
            neuron_idx = int(request.args.get('neuron_idx'))
            result = nca_service.get_neuron_weights(layer_idx, neuron_idx)
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    
    if request.method == 'POST':
        data = request.json
        try:
            result = nca_service.set_neuron_weights(data)
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 400

@app.route('/api/cell_details', methods=['GET'])
def get_cell_details():
    try:
        r = int(request.args.get('r'))
        c = int(request.args.get('c'))
        result = nca_service.get_cell_details(r, c)
        return jsonify(result)
    except (TypeError, ValueError) as e:
        return jsonify({"error": "Invalid row/column parameters"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/set_grid_state', methods=['POST'])
def set_grid_state_route():
    data = request.json
    grid_state_data = data.get('grid_state')
    was_running = data.get('was_running', False)
    try:
        result = nca_service.set_nca_grid_state(grid_state_data, was_running)
        return jsonify(result)
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid grid state data: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)