// static/js/modules/api.js
//
// The application used to call a Flask/PyTorch JSON API. The service now lives
// in the browser, but this compatibility dispatcher intentionally keeps the old
// fetchApi() contract so the rest of the UI does not need to know or care.

import { browserNcaService } from './browserNcaService.js';

export async function fetchApi(endpoint, method = 'GET', body = null) {
    try {
        const url = new URL(endpoint, window.location.href);
        const path = url.pathname;

        if (path.endsWith('/api/config') && method === 'GET') {
            return browserNcaService.getConfig();
        }
        if (path.endsWith('/api/step') && method === 'POST') {
            return browserNcaService.step();
        }
        if (path.endsWith('/api/step_back') && method === 'POST') {
            return browserNcaService.stepBack();
        }
        if (path.endsWith('/api/toggle_pause') && method === 'POST') {
            return browserNcaService.togglePause();
        }
        if (path.endsWith('/api/apply_settings') && method === 'POST') {
            return browserNcaService.applySettings(body || {});
        }
        if (path.endsWith('/api/set_colormap') && method === 'POST') {
            return browserNcaService.setColormap(body?.colormap_name);
        }
        if (path.endsWith('/api/randomize_weights') && method === 'POST') {
            return browserNcaService.randomizeWeights(body || {});
        }
        if (path.endsWith('/api/randomize_grid') && method === 'POST') {
            return browserNcaService.randomizeGrid(body || {});
        }
        if (path.endsWith('/api/randomize_architecture') && method === 'POST') {
            return browserNcaService.randomizeArchitecture(body || {});
        }
        if (path.endsWith('/api/restart') && method === 'POST') {
            return browserNcaService.restart();
        }
        if (path.endsWith('/api/neuron_weights')) {
            if (method === 'GET') {
                return browserNcaService.getNeuronWeights(
                    url.searchParams.get('layer_idx'),
                    url.searchParams.get('neuron_idx')
                );
            }
            if (method === 'POST') {
                return browserNcaService.setNeuronWeights(body || {});
            }
        }
        if (path.endsWith('/api/cell_details') && method === 'GET') {
            return browserNcaService.getCellDetails(
                url.searchParams.get('r'),
                url.searchParams.get('c')
            );
        }
        if (path.endsWith('/api/set_grid_state') && method === 'POST') {
            return browserNcaService.setGridState(body || {});
        }

        throw new Error(`Unsupported local API route: ${method} ${endpoint}`);
    } catch (error) {
        console.error(`Local NCA API error for ${method} ${endpoint}:`, error);
        alert(`Error: ${error.message}`);
        return null;
    }
}
