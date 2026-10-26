use minifb::{Key, Window, WindowOptions};
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;

use tesis_ram::analysis::ode::{LevinMonodParams, MultiClonalState, PearsonTracker};
use tesis_ram::biology::rules::{BiophysicalParams, try_colonize_empty_cell, update_living_cell};
use tesis_ram::core::cell::Cell;
use tesis_ram::core::grid::{DoubleBufferGrid, Grid};
use tesis_ram::visualizer::osd::{draw_rect, draw_text};

const WIDTH: usize = 1000;
const HEIGHT: usize = 600;
const OSD_HEIGHT: usize = 90;

const REACTOR_CENTER_X: usize = 200;
const REACTOR_CENTER_Y: usize = 345;
const REACTOR_RADIUS: usize = 98;

fn init_reactor(
    grid: &mut DoubleBufferGrid<Cell>,
    ode_state: &mut MultiClonalState,
    tracker: &mut PearsonTracker,
    ca_history: &mut Vec<f64>,
    ode_history: &mut Vec<f64>,
) {
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            grid.current.set(x, y, Cell::EMPTY);
            grid.next.set(x, y, Cell::EMPTY);
        }
    }

    let mut seeded = 0usize;
    for dy in -11..=11isize {
        for dx in -11..=11isize {
            if dx * dx + dy * dy <= 121 {
                let px = (REACTOR_CENTER_X as isize + dx) as usize;
                let py = (REACTOR_CENTER_Y as isize + dy) as usize;
                grid.current.set(px, py, Cell::new(0, 0, 140));
                seeded += 1;
            }
        }
    }

    *ode_state = MultiClonalState {
        n: [seeded as f64, 0.0, 0.0, 0.0, 0.0],
    };
    *tracker = PearsonTracker::new();
    ca_history.clear();
    ode_history.clear();
}

/// Retorna la concentración de antibiótico (ug/mL) y el número de escalón según el modo y el tick
fn get_chemostat_dose(tick: usize, is_morbidostat: bool, manual_dose: Option<f64>) -> (f64, usize) {
    if let Some(d) = manual_dose {
        return (d, 1);
    }
    if !is_morbidostat {
        // Modo 1: Quimiostato Clásico (Inyección única en tick 1500)
        if tick >= 1500 {
            (0.08, 1)
        } else {
            (0.0, 0)
        }
    } else {
        // Modo 2: Morbidostat de 4 Escalones
        if tick < 1200 {
            (0.0, 0)
        } else if tick < 2400 {
            (0.08, 1) // Escalón 1: MSW de g0 -> selecciona g1
        } else if tick < 3600 {
            (0.45, 2) // Escalón 2: MSW de g1 -> selecciona g2
        } else if tick < 4800 {
            (3.20, 3) // Escalón 3: MSW de g2 -> selecciona g3
        } else {
            (25.0, 4) // Escalón 4: MSW de g3 -> selecciona g4
        }
    }
}

fn draw_line_y(fb: &mut [u32], x: usize, y0: usize, y1: usize, color: u32) {
    let ymin = y0.min(y1);
    let ymax = y0.max(y1);
    for y in ymin..=ymax {
        if x < WIDTH && y < HEIGHT {
            fb[y * WIDTH + x] = color;
        }
    }
}

fn main() {
    let mut window = Window::new(
        "Tesis RAM - Quimiostato y Morbidostat: CA vs EDO RK4 (Levin-Regoes / Toprak)",
        WIDTH,
        HEIGHT,
        WindowOptions::default(),
    )
    .expect("Fallo al inicializar minifb");

    window.set_target_fps(60);

    let mut fb = vec![0u32; WIDTH * HEIGHT];
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);

    let ode_params = LevinMonodParams::default();
    let bio_params = BiophysicalParams::default();

    let mut grid = DoubleBufferGrid::new(WIDTH, HEIGHT, Cell::EMPTY);
    let mut conc_grid = Grid::new(WIDTH, HEIGHT, 0.0);
    let mut ode_state = MultiClonalState { n: [400.0, 0.0, 0.0, 0.0, 0.0] };
    let mut tracker = PearsonTracker::new();

    let mut ca_history = Vec::with_capacity(500);
    let mut ode_history = Vec::with_capacity(500);

    init_reactor(&mut grid, &mut ode_state, &mut tracker, &mut ca_history, &mut ode_history);

    let mut tick = 0usize;
    let mut is_paused = false;
    let mut is_morbidostat = false; // false = Modo 1 (1 Dosis), true = Modo 2 (4 Dosis Escalonadas)
    let mut manual_dose: Option<f64> = None;
    let mut speed_multiplier = 2usize;

    let r_sq = (REACTOR_RADIUS * REACTOR_RADIUS) as isize;
    let p_washout = 1.0 - (-ode_params.dilution_d).exp();

    while window.is_open() && !window.is_key_down(Key::Escape) {
        if window.is_key_pressed(Key::Space, minifb::KeyRepeat::No) {
            is_paused = !is_paused;
        }
        if window.is_key_pressed(Key::M, minifb::KeyRepeat::No) {
            is_morbidostat = !is_morbidostat;
            manual_dose = None;
            init_reactor(&mut grid, &mut ode_state, &mut tracker, &mut ca_history, &mut ode_history);
            tick = 0;
        }
        if window.is_key_pressed(Key::R, minifb::KeyRepeat::No) {
            manual_dose = None;
            init_reactor(&mut grid, &mut ode_state, &mut tracker, &mut ca_history, &mut ode_history);
            tick = 0;
        }
        if window.is_key_pressed(Key::D, minifb::KeyRepeat::No) {
            // Ciclar manualmente por los 4 niveles de dosis si el usuario presiona [D]
            manual_dose = match manual_dose {
                None => Some(0.08),
                Some(c) if c < 0.20 => Some(0.45),
                Some(c) if c < 1.00 => Some(3.20),
                Some(c) if c < 10.0 => Some(25.0),
                _ => None,
            };
        }
        if window.is_key_pressed(Key::Equal, minifb::KeyRepeat::No) {
            speed_multiplier = (speed_multiplier + 1).min(8);
        }
        if window.is_key_pressed(Key::Minus, minifb::KeyRepeat::No) {
            speed_multiplier = speed_multiplier.saturating_sub(1).max(1);
        }

        let (antibiotic_conc, dose_step) = get_chemostat_dose(tick, is_morbidostat, manual_dose);

        if !is_paused {
            for _ in 0..speed_multiplier {
                let (c_step, _) = get_chemostat_dose(tick, is_morbidostat, manual_dose);
                conc_grid.fill(c_step);

                // 1. Integración continua RK4 (Multi-clonal g0..g4)
                ode_state = ode_params.rk4_step_multiclonal(
                    &ode_state,
                    c_step,
                    &bio_params.mic_by_genotype,
                    &bio_params.mpc_by_genotype,
                    bio_params.fitness_cost_per_mutation,
                    1.0,
                );

                // 2. Actualización estocástica del Autómata Celular en el biorreactor
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
                            let updated = update_living_cell(
                                cell,
                                c_step,
                                density,
                                false,
                                &bio_params,
                                1.0,
                                &mut rng,
                            );
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

                // Agitación magnética turbulenta (mezcla espacial aleatoria en medio líquido)
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

                let mut ca_total = 0.0;
                for y in (REACTOR_CENTER_Y - REACTOR_RADIUS)..(REACTOR_CENTER_Y + REACTOR_RADIUS) {
                    for x in (REACTOR_CENTER_X - REACTOR_RADIUS)..(REACTOR_CENTER_X + REACTOR_RADIUS) {
                        let dx = x as isize - REACTOR_CENTER_X as isize;
                        let dy = y as isize - REACTOR_CENTER_Y as isize;
                        if dx * dx + dy * dy <= r_sq && grid.current.get(x, y).is_alive() {
                            ca_total += 1.0;
                        }
                    }
                }

                let ode_total = ode_state.total();
                tracker.record(ca_total, ode_total);

                if tick % 4 == 0 {
                    ca_history.push(ca_total);
                    ode_history.push(ode_total);
                    if ca_history.len() > 490 {
                        ca_history.remove(0);
                        ode_history.remove(0);
                    }
                }

                tick += 1;
            }
        }

        let mut counts = [0usize; 5];
        for y in (REACTOR_CENTER_Y - REACTOR_RADIUS)..(REACTOR_CENTER_Y + REACTOR_RADIUS) {
            for x in (REACTOR_CENTER_X - REACTOR_RADIUS)..(REACTOR_CENTER_X + REACTOR_RADIUS) {
                let dx = x as isize - REACTOR_CENTER_X as isize;
                let dy = y as isize - REACTOR_CENTER_Y as isize;
                if dx * dx + dy * dy <= r_sq {
                    let c = grid.current.get(x, y);
                    if c.is_alive() {
                        counts[c.genotype() as usize] += 1;
                    }
                }
            }
        }
        let ca_now: usize = counts.iter().sum();
        let ode_now = ode_state.total();
        let pearson_r = tracker.calculate();
        let nrmse_pct = tracker.nrmse(ode_params.carrying_capacity) * 100.0;

        // Visual
        fb.fill(0x0B1320);

        draw_rect(&mut fb, WIDTH, 0, 0, WIDTH, OSD_HEIGHT, 0x1E293B);
        draw_rect(&mut fb, WIDTH, 0, OSD_HEIGHT - 2, WIDTH, 2, 0x38BDF8);

        draw_text(&mut fb, WIDTH, 20, 12, "UNIVERSIDAD MAYOR DE SAN ANDRES - INFORMATICA | TESIS RAM", 0x38BDF8);

        let mode_title = if is_morbidostat {
            "MODO 2: MORBIDOSTAT ESCALONADO (4 DOSIS: 0.08 -> 0.45 -> 3.20 -> 25.0 ug/mL)"
        } else {
            "MODO 1: QUIMIOSTATO CLASICO EDO RK4 vs AUTOMATA CELULAR (1 DOSIS: 0.08 ug/mL)"
        };
        draw_text(&mut fb, WIDTH, 20, 28, mode_title, 0xFBBF24);

        let status = format!(
            "TICK:{:05} | ESCALON:{}/4 ({:.2} ug/mL) | r = {:.4} | NRMSE = {:.2}% | VEL:{}x",
            tick, dose_step, antibiotic_conc, pearson_r, nrmse_pct, speed_multiplier
        );
        draw_text(&mut fb, WIDTH, 20, 46, &status, 0x4ADE80);
        draw_text(
            &mut fb,
            WIDTH,
            20,
            64,
            "[M] Alternar Modo (1 Dosis / 4 Dosis Morbidostat)  [D] Escalar Dosis  [ESPACIO] Pausa  [R] Reset",
            0x94A3B8,
        );

        // Dibujar Biorreactor Circular
        for dy in -(REACTOR_RADIUS as isize + 4)..=(REACTOR_RADIUS as isize + 4) {
            for dx in -(REACTOR_RADIUS as isize + 4)..=(REACTOR_RADIUS as isize + 4) {
                let d2 = dx * dx + dy * dy;
                let px = (REACTOR_CENTER_X as isize + dx) as usize;
                let py = (REACTOR_CENTER_Y as isize + dy) as usize;
                if d2 <= r_sq {
                    let cell = grid.current.get(px, py);
                    if cell.is_alive() {
                        fb[py * WIDTH + px] = match cell.genotype() {
                            0 => 0x22C55E, // Verde: g0
                            1 => 0xEAB308, // Amarillo: g1
                            2 => 0xF97316, // Naranja: g2
                            3 => 0xEF4444, // Rojo: g3
                            _ => 0xA855F7, // Purpura: g4
                        };
                    } else {
                        let drug_tint = (antibiotic_conc.ln_1p() * 14.0).clamp(0.0, 45.0) as u32;
                        fb[py * WIDTH + px] = (0x0F << 16) | ((0x1A + drug_tint / 2) << 8) | (0x2C + drug_tint);
                    }
                } else if d2 <= (REACTOR_RADIUS as isize + 3).pow(2) {
                    fb[py * WIDTH + px] = if antibiotic_conc > 0.0 { 0x38BDF8 } else { 0x475569 };
                }
            }
        }

        draw_text(&mut fb, WIDTH, 85, 195, "BIORREACTOR CONTINUO (E. coli)", 0xE2E8F0);
        let pop_summary = format!("CA: {} | EDO: {:.0}", ca_now, ode_now);
        draw_text(&mut fb, WIDTH, 115, 465, &pop_summary, 0x38BDF8);

        let g_line1 = format!("g0:{} g1:{} g2:{}", counts[0], counts[1], counts[2]);
        let g_line2 = format!("g3:{} g4:{}", counts[3], counts[4]);
        draw_text(&mut fb, WIDTH, 95, 488, &g_line1, 0xCBD5E1);
        draw_text(&mut fb, WIDTH, 125, 506, &g_line2, 0xCBD5E1);

        // Gráfica Derecha: Convergencia CA vs RK4
        let gx = 415;
        let gy = 130;
        let gw = 550;
        let gh = 420;

        draw_rect(&mut fb, WIDTH, gx, gy, gw, gh, 0x050B14);
        draw_rect(&mut fb, WIDTH, gx, gy, gw, 1, 0x334155);
        draw_rect(&mut fb, WIDTH, gx, gy + gh - 1, gw, 1, 0x334155);
        draw_rect(&mut fb, WIDTH, gx, gy, 1, gh, 0x334155);
        draw_rect(&mut fb, WIDTH, gx + gw - 1, gy, 1, gh, 0x334155);

        draw_text(
            &mut fb,
            WIDTH,
            gx + 16,
            gy + 14,
            "CONVERGENCIA ISOMORFICA: AUTOMATA CELULAR (VERDE) vs RK4 (AZUL)",
            0x38BDF8,
        );

        let max_y_pop = 28000.0;
        let plot_top = gy + 45;
        let plot_bot = gy + gh - 25;

        if ca_history.len() > 1 {
            for i in 1..ca_history.len() {
                let x0 = gx + 20 + (i - 1);
                let x1 = gx + 20 + i;
                if x1 < gx + gw - 15 {
                    let cy0 = plot_bot - ((ca_history[i - 1] / max_y_pop).clamp(0.0, 1.0) * (plot_bot - plot_top) as f64) as usize;
                    let cy1 = plot_bot - ((ca_history[i] / max_y_pop).clamp(0.0, 1.0) * (plot_bot - plot_top) as f64) as usize;
                    draw_line_y(&mut fb, x0, cy0, cy1, 0x22C55E);

                    let oy0 = plot_bot - ((ode_history[i - 1] / max_y_pop).clamp(0.0, 1.0) * (plot_bot - plot_top) as f64) as usize;
                    let oy1 = plot_bot - ((ode_history[i] / max_y_pop).clamp(0.0, 1.0) * (plot_bot - plot_top) as f64) as usize;
                    draw_line_y(&mut fb, x0, oy0, oy1, 0x38BDF8);
                }
            }
        }

        window.update_with_buffer(&fb, WIDTH, HEIGHT).unwrap();
    }
}