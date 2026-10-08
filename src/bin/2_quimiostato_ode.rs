use minifb::{Key, Window, WindowOptions};
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;

use tesis_ram::analysis::ode::{ChemostatState, LevinMonodParams, PearsonTracker};
use tesis_ram::biology::rules::{try_colonize_empty_cell, update_living_cell, BiophysicalParams};
use tesis_ram::core::cell::Cell;
use tesis_ram::core::grid::DoubleBufferGrid;
use tesis_ram::visualizer::osd::{draw_rect, draw_text};

const WIDTH: usize = 1000;
const HEIGHT: usize = 600;
const OSD_HEIGHT: usize = 90;

const REACTOR_CENTER_X: usize = 200;
const REACTOR_CENTER_Y: usize = 345;
const REACTOR_RADIUS: usize = 98; // 30,170 celdas totales; capacidad de equilibrio = 25,130 celdas

fn init_reactor(grid: &mut DoubleBufferGrid<Cell>) {
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            grid.current.set(x, y, Cell::EMPTY);
            grid.next.set(x, y, Cell::EMPTY);
        }
    }
    for dy in -11..=11 {
        for dx in -11..=11 {
            if dx * dx + dy * dy <= 121 {
                let px = (REACTOR_CENTER_X as isize + dx) as usize;
                let py = (REACTOR_CENTER_Y as isize + dy) as usize;
                grid.current.set(px, py, Cell::new(0, 0, 140));
            }
        }
    }
}

fn main() {
    let mut window = Window::new(
        "Tesis RAM - Quimiostato: Validacion Cruzada Isomorfica (CA vs RK4)",
        WIDTH,
        HEIGHT,
        WindowOptions::default(),
    )
    .expect("Fallo al inicializar minifb en WSL");

    window.set_target_fps(60);

    let mut fb = vec![0u32; WIDTH * HEIGHT];
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);

    let ode_params = LevinMonodParams {
        carrying_capacity: 25130.0, // Calibrado al 83.3% de saturación Chamfer
        growth_rate_s: 0.08,
        growth_rate_r: 0.0736,
        dilution_d: 0.015,
        e_max: 0.35,
        mic_s: 0.035,
        mic_r: 0.200,
        hill_h: 2.0,
        mu_sos: 0.0005,
    };

    let bio_params = BiophysicalParams::default();

    let mut ode_state = ChemostatState {
        s: 0.0,
        n_s: 400.0,
        n_r: 0.0,
    };

    let mut grid = DoubleBufferGrid::new(WIDTH, HEIGHT, Cell::EMPTY);
    let conc_grid = tesis_ram::core::grid::Grid::new(WIDTH, HEIGHT, 0.0);
    init_reactor(&mut grid);

    let mut tracker = PearsonTracker::new();
    let mut tick = 0usize;
    let mut antibiotic_conc = 0.0;
    let mut is_paused = false;

    let mut ca_history = Vec::with_capacity(500);
    let mut ode_history = Vec::with_capacity(500);

    while window.is_open() && !window.is_key_down(Key::Escape) {
        if window.is_key_pressed(Key::A, minifb::KeyRepeat::No) {
            antibiotic_conc = if antibiotic_conc > 0.0 { 0.0 } else { 0.08 };
        }

        if window.is_key_pressed(Key::P, minifb::KeyRepeat::No) {
            is_paused = !is_paused;
        }

        if window.is_key_pressed(Key::R, minifb::KeyRepeat::No) {
            antibiotic_conc = 0.0;
            ode_state = ChemostatState { s: 0.0, n_s: 400.0, n_r: 0.0 };
            init_reactor(&mut grid);
            tracker = PearsonTracker::new();
            ca_history.clear();
            ode_history.clear();
            tick = 0;
        }

        let mut living_ca_count = 0.0;
        let r_sq = (REACTOR_RADIUS * REACTOR_RADIUS) as isize;

        if !is_paused {
            ode_state = ode_params.rk4_step(&ode_state, antibiotic_conc, 1.0);
            let p_washout = 1.0 - (-ode_params.dilution_d).exp();

            for y in (REACTOR_CENTER_Y - REACTOR_RADIUS)..(REACTOR_CENTER_Y + REACTOR_RADIUS) {
                for x in (REACTOR_CENTER_X - REACTOR_RADIUS)..(REACTOR_CENTER_X + REACTOR_RADIUS) {
                    let dx = x as isize - REACTOR_CENTER_X as isize;
                    let dy = y as isize - REACTOR_CENTER_Y as isize;
                    if dx * dx + dy * dy > r_sq {
                        continue;
                    }

                    let cell = *grid.current.get(x, y);

                    if cell.is_alive() && rng.random_bool(p_washout) {
                        grid.next.set(x, y, Cell::EMPTY);
                        continue;
                    }

                    let density = grid.current.chamfer_density_5_7(x, y);

                    if cell.is_alive() {
                        let updated = update_living_cell(cell, antibiotic_conc, density, false, &bio_params, 1.0, &mut rng);
                        grid.next.set(x, y, updated);
                    } else {
                        let neighbors = grid.current.get_living_neighbors(x, y);
                        let colonized = try_colonize_empty_cell(
                            x,
                            y,
                            density,
                            &neighbors,
                            &conc_grid,
                            &bio_params,
                            &mut rng,
                        );
                        grid.next.set(x, y, colonized);
                    }
                }
            }
            grid.swap();

            if tick % 2 == 0 {
                let r_i = REACTOR_RADIUS as i32;
                for _ in 0..160 {
                    let dx1 = rng.random_range(-r_i..r_i);
                    let dy1 = rng.random_range(-r_i..r_i);
                    let dx2 = rng.random_range(-r_i..r_i);
                    let dy2 = rng.random_range(-r_i..r_i);
                    if dx1 * dx1 + dy1 * dy1 <= r_i * r_i && dx2 * dx2 + dy2 * dy2 <= r_i * r_i {
                        let x1 = (REACTOR_CENTER_X as i32 + dx1) as usize;
                        let y1 = (REACTOR_CENTER_Y as i32 + dy1) as usize;
                        let x2 = (REACTOR_CENTER_X as i32 + dx2) as usize;
                        let y2 = (REACTOR_CENTER_Y as i32 + dy2) as usize;
                        let c1 = *grid.current.get(x1, y1);
                        let c2 = *grid.current.get(x2, y2);
                        grid.current.set(x1, y1, c2);
                        grid.current.set(x2, y2, c1);
                    }
                }
            }

            tick += 1;
        }

        for y in (REACTOR_CENTER_Y - REACTOR_RADIUS)..(REACTOR_CENTER_Y + REACTOR_RADIUS) {
            for x in (REACTOR_CENTER_X - REACTOR_RADIUS)..(REACTOR_CENTER_X + REACTOR_RADIUS) {
                let dx = x as isize - REACTOR_CENTER_X as isize;
                let dy = y as isize - REACTOR_CENTER_Y as isize;
                if dx * dx + dy * dy <= r_sq && grid.current.get(x, y).is_alive() {
                    living_ca_count += 1.0;
                }
            }
        }

        let total_ode = ode_state.n_s + ode_state.n_r;
        if !is_paused {
            tracker.record(living_ca_count, total_ode);
            ca_history.push(living_ca_count);
            ode_history.push(total_ode);
            if ca_history.len() > 460 {
                ca_history.remove(0);
                ode_history.remove(0);
            }
        }

        let pearson_r = tracker.calculate();

        fb.fill(0x0F172A);

        draw_rect(&mut fb, WIDTH, 0, 0, WIDTH, OSD_HEIGHT, 0x1E293B);
        draw_rect(&mut fb, WIDTH, 0, OSD_HEIGHT - 2, WIDTH, 2, 0x38BDF8);

        draw_text(&mut fb, WIDTH, 20, 14, "VALIDACION CRUZADA: AUTOMATA CELULAR vs EDO (LEVIN-REGOES)", 0x38BDF8);
        let stats_str = format!(
            "TICK: {:05} | CA: {:<5.0} | EDO: {:<5.0} | PEARSON r: {:.4} | ESTADO: {}",
            tick, living_ca_count, total_ode, pearson_r, if is_paused { "PAUSADO" } else { "ACTIVO" }
        );
        draw_text(&mut fb, WIDTH, 20, 32, &stats_str, 0xF1F5F9);

        let ab_str = format!(
            "ANTIBIOTICO: {:.3} ug/mL  {}",
            antibiotic_conc,
            if antibiotic_conc > 0.0 { "[PRESENTE EN MSW]" } else { "[INACTIVO]" }
        );
        draw_text(&mut fb, WIDTH, 20, 50, &ab_str, if antibiotic_conc > 0.0 { 0xF87171 } else { 0x4ADE80 });
        draw_text(&mut fb, WIDTH, 20, 68, "[A] Alternar Dosis  [P] Pausar  [R] Reiniciar  [ESC] Salir", 0x94A3B8);

        for dy in -(REACTOR_RADIUS as isize)..=(REACTOR_RADIUS as isize) {
            for dx in -(REACTOR_RADIUS as isize)..=(REACTOR_RADIUS as isize) {
                if dx * dx + dy * dy <= r_sq {
                    let px = (REACTOR_CENTER_X as isize + dx) as usize;
                    let py = (REACTOR_CENTER_Y as isize + dy) as usize;
                    let cell = grid.current.get(px, py);
                    let color = if cell.is_alive() {
                        match cell.genotype() {
                            0 => 0x22C55E,
                            1 => 0xEAB308,
                            2 => 0xF97316,
                            3 => 0xEF4444,
                            _ => 0xA855F7,
                        }
                    } else {
                        0x1E293B
                    };
                    fb[py * WIDTH + px] = color;
                }
            }
        }
        draw_text(&mut fb, WIDTH, 130, 485, "REACTOR CILINDRICO (CA)", 0x94A3B8);

        let gx = 420;
        let gy = 140;
        let gw = 520;
        let gh = 380;

        draw_rect(&mut fb, WIDTH, gx, gy, gw, gh, 0x020617);
        draw_rect(&mut fb, WIDTH, gx, gy, gw, 1, 0x475569);
        draw_rect(&mut fb, WIDTH, gx, gy + gh, gw, 1, 0x475569);
        draw_rect(&mut fb, WIDTH, gx, gy, 1, gh, 0x475569);
        draw_rect(&mut fb, WIDTH, gx + gw, gy, 1, gh + 1, 0x475569);

        draw_text(&mut fb, WIDTH, gx + 20, gy + 15, "CONVERGENCIA TEMPORAL ISOMORFICA", 0xE2E8F0);
        draw_rect(&mut fb, WIDTH, gx + 310, gy + 18, 15, 4, 0x22C55E);
        draw_text(&mut fb, WIDTH, gx + 330, gy + 15, "AC", 0xCBD5E1);
        draw_rect(&mut fb, WIDTH, gx + 390, gy + 18, 15, 4, 0x38BDF8);
        draw_text(&mut fb, WIDTH, gx + 410, gy + 15, "RK4", 0xCBD5E1);

        let max_scale = 32000.0;
        for i in 1..ca_history.len() {
            let px = gx + 10 + i;
            if px < gx + gw - 10 {
                let py_ca = (gy + gh) - ((ca_history[i] / max_scale) * (gh - 40) as f64).clamp(0.0, (gh - 40) as f64) as usize;
                let py_ode = (gy + gh) - ((ode_history[i] / max_scale) * (gh - 40) as f64).clamp(0.0, (gh - 40) as f64) as usize;
                fb[py_ca * WIDTH + px] = 0x22C55E;
                fb[py_ode * WIDTH + px] = 0x38BDF8;
            }
        }

        window.update_with_buffer(&fb, WIDTH, HEIGHT).unwrap();
    }
}