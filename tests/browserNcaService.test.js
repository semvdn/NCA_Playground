import test from 'node:test';
import assert from 'node:assert/strict';
import { BrowserNCAService } from '../static/js/modules/browserNcaService.js';

function grid(size = 50, fill = 0) {
    return Array.from({ length: size }, () => Array(size).fill(fill));
}

function values(result, key = 'grid_values') {
    return Array.from(result[key]);
}

test('default configuration exposes a valid 9-to-1 network', () => {
    const service = new BrowserNCAService();
    const config = service.getConfig();
    assert.equal(config.default_params.layer_sizes[0], 9);
    assert.equal(config.default_params.layer_sizes.at(-1), 1);
    assert.equal(config.initial_grid_values.length, 2500);
    assert.equal(config.is_paused, true);
});

test('single-step history restores the exact previous state', () => {
    const service = new BrowserNCAService();
    const before = Array.from(service.getConfig().initial_grid_values);
    service.step();
    const restored = service.stepBack();
    assert.deepEqual(values(restored), before);
});

test('random grids are reproducible by seed and restart from their saved initial state', () => {
    const service = new BrowserNCAService();
    const first = service.randomizeGrid({ seed: 123456 });
    const expected = values(first);
    assert.equal(service.nca.getCurrentParams().initial_seed, 123456);

    service.step();
    const restarted = service.restart();
    assert.deepEqual(values(restarted, 'initial_grid_values'), expected);

    const second = service.randomizeGrid({ seed: 123456 });
    assert.deepEqual(values(second), expected);
});

test('preset grid states also become the restart target', () => {
    const service = new BrowserNCAService();
    const initial = grid();
    initial[4][7] = 0.9;
    initial[25][25] = 1;
    const applied = service.setGridState({ grid_state: initial, was_running: false });
    const expected = values(applied);

    service.step();
    const restarted = service.restart();
    assert.deepEqual(values(restarted, 'initial_grid_values'), expected);
});

test('neighborhood lookup wraps across both grid boundaries', () => {
    const service = new BrowserNCAService();
    const initial = grid();
    initial[49][49] = 0.75;
    service.setGridState({ grid_state: initial, was_running: false });
    const details = service.getCellDetails(0, 0);
    assert.equal(details.neighborhood[0][0], 0.75);
});

test('cell inspection reports the exact scalar state rather than a color estimate', () => {
    const service = new BrowserNCAService();
    const initial = grid();
    initial[7][11] = 0.314159;
    service.setGridState({ grid_state: initial, was_running: false });
    const details = service.getCellDetails(7, 11);
    assert.ok(Math.abs(details.cell_value - 0.314159) < 1e-6);
});

test('custom settings preserve the grid and running state', () => {
    const service = new BrowserNCAService();
    const before = Array.from(service.getConfig().initial_grid_values);
    service.togglePause();

    const result = service.applySettings({
        preset_name: 'Custom',
        layer_sizes: '9,4,1',
        activation: 'tanh',
        weight_scale: 0.7,
        bias: 0.2
    });

    assert.deepEqual(values(result), before);
    assert.deepEqual(result.current_params.layer_sizes, [9, 4, 1]);
    assert.equal(result.current_params.activation, 'tanh');
    assert.equal(result.current_params.weight_scale, 0.7);
    assert.equal(result.current_params.bias, 0.2);
    assert.equal(result.is_paused, false);
});

test('loading a named preset preserves whether the simulation was running', () => {
    const service = new BrowserNCAService();
    service.togglePause();
    const result = service.applySettings({ preset_name: 'Shallow ReLU' });
    assert.deepEqual(result.current_params.layer_sizes, [9, 16, 1]);
    assert.equal(result.current_params.activation, 'relu');
    assert.equal(result.is_paused, false);
});
