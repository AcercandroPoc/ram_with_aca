//! src/bin/3_placa_mega.rs
//! Emulador de la Placa MEGA (Baym et al., 2016) con los 3 Modos Experimentales.

use minifb::{Key, Window, WindowOptions};
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;

use tesis_ram::biology::rules::{try_colonize_empty_cell, update_living_cell, BiophysicalParams};
use tesis_ram::core::cell::Cell;
use tesis_ram::core::grid::DoubleBufferGrid;
use tesis_ram::physics::diffusion::step_diffusion_mega_plate;
use tesis_ram::visualizer::osd::{draw_rect, draw_text};

const WIDTH: usize = 1000;
const HEIGHT: usize = 600;
const OSD_HEIGHT: usize = 100;
const SIM_HEIGHT: usize = HEIGHT - OSD_HEIGHT;
const NUM_ZONES: usize = 5;

fn init_simulation(
    conc_grid: &mut DoubleBufferGrid<f64>,
    cell_grid: &mut DoubleBufferGrid<Cell>,
    reservoirs: &[f64; 5],
    zone_w: usize,
    rng: &mut Xoshiro256PlusPlus,
) {
    for y in 0..SIM_HEIGHT {
        for x in 0..WIDTH {
            let z = (x / zone_w).min(NUM_ZONES - 1);
            conc_grid.current.set(x, y, reservoirs[z]);
            conc_grid.next.set(x, y, reservoirs[z]);
            cell_grid.current.set(x, y, Cell::EMPTY);
            cell_grid.next.set(x, y, Cell::EMPTY);
        }
    }

    for y in 20..(SIM_HEIGHT - 20) {
        for x in 10..40 {
            if rng.random_bool(0.6) {
                cell_grid.current.set(x, y, Cell::new(0, 0, 150));
            }
        }
    }
}

fn main() {
    let mut window = Window::new(
        "Tesis RAM - Placa MEGA de Harvard (Baym et al., 2016)",
        WIDTH,
        HEIGHT,
        WindowOptions::default(),
    )
    .expect("Fallo al inicializar minifb en WSL");

    window.set_target_fps(60);

    let mut fb = vec![0u32; WIDTH * HEIGHT];
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(2026);
    let bio_params = BiophysicalParams::default();

    let c_s = bio_params.mic_by_genotype[0]; // 0.035 ug/mL

    let mode_gradual = [0.0, 3.0 * c_s, 30.0 * c_s, 300.0 * c_s, 3000.0 * c_s];
    let mode_intermedio = [0.0, 0.0, 30.0 * c_s, 3000.0 * c_s, 3000.0 * c_s];
    let mode_abrupto = [0.0, 3000.0 * c_s, 3000.0 * c_s, 3000.0 * c_s, 3000.0 * c_s];

    let mut current_mode_idx = 0usize;
    let mut is_paused = false;
    let mut current_reservoirs = mode_gradual;
    let zone_w = WIDTH / NUM_ZONES;

    let mut conc_grid = DoubleBufferGrid::new(WIDTH, SIM_HEIGHT, 0.0);
    let mut cell_grid = DoubleBufferGrid::new(WIDTH, SIM_HEIGHT, Cell::EMPTY);

    init_simulation(&mut conc_grid, &mut cell_grid, &current_reservoirs, zone_w, &mut rng);

    let mut tick = 0usize;
    let mut speed_multiplier = 1usize;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        if window.is_key_pressed(Key::Space, minifb::KeyRepeat::No) {
            current_mode_idx = (current_mode_idx + 1) % 3;
            current_reservoirs = match current_mode_idx {
                0 => mode_gradual,
                1 => mode_intermedio,
                _ => mode_abrupto,
            };
            init_simulation(&mut conc_grid, &mut cell_grid, &current_reservoirs, zone_w, &mut rng);
            tick = 0;
        }

        if window.is_key_pressed(Key::P, minifb::KeyRepeat::No) {
            is_paused = !is_paused;
        }

        if window.is_key_pressed(Key::R, minifb::KeyRepeat::No) {
            init_simulation(&mut conc_grid, &mut cell_grid, &current_reservoirs, zone_w, &mut rng);
            tick = 0;
        }

        if window.is_key_pressed(Key::Equal, minifb::KeyRepeat::No) {
            speed_multiplier = (speed_multiplier + 1).min(5);
        }

        if window.is_key_pressed(Key::Minus, minifb::KeyRepeat::No) {
            speed_multiplier = (speed_multiplier.saturating_sub(1)).max(1);
        }

        let mut max_x = 0usize;
        let mut counts = [0usize; 5];

        if !is_paused {
            for _ in 0..speed_multiplier {
                step_diffusion_mega_plate(&conc_grid.current, &mut conc_grid.next, &current_reservoirs, NUM_ZONES, 0.01);
                conc_grid.swap();

                for y in 0..SIM_HEIGHT {
                    for x in 0..WIDTH {
                        let cell = *cell_grid.current.get(x, y);
                        let local_c = *conc_grid.current.get(x, y);
                        let density = cell_grid.current.chamfer_density_5_7(x, y);

                        if cell.is_alive() {
                            let updated = update_living_cell(cell, local_c, density, true, &bio_params, 1.0, &mut rng);
                            cell_grid.next.set(x, y, updated);
                        } else {
                            let neighbors = cell_grid.current.get_living_neighbors(x, y);
                            let colonized = try_colonize_empty_cell(
                                x,
                                y,
                                density,
                                &neighbors,
                                &conc_grid.current,
                                &bio_params,
                                &mut rng,
                            );
                            cell_grid.next.set(x, y, colonized);
                        }
                    }
                }
                cell_grid.swap();
                tick += 1;
            }
        }

        for y in 0..SIM_HEIGHT {
            for x in 0..WIDTH {
                let cell = *cell_grid.current.get(x, y);
                if cell.is_alive() {
                    counts[cell.genotype() as usize] += 1;
                    if x > max_x {
                        max_x = x;
                    }
                }
            }
        }

        // Renderizado OSD
        draw_rect(&mut fb, WIDTH, 0, 0, WIDTH, OSD_HEIGHT, 0x1A1A24);
        draw_rect(&mut fb, WIDTH, 0, OSD_HEIGHT - 2, WIDTH, 2, 0x3B82F6);

        draw_text(&mut fb, WIDTH, 20, 12, "UNIVERSIDAD MAYOR DE SAN ANDRES - INFORMATICA | TESIS RAM", 0x93C5FD);

        let (mode_text, mode_color) = match current_mode_idx {
            0 => ("MODO 1: GRADUAL BAYM ET AL. (0x -> 3x -> 30x -> 300x -> 3000x)", 0x10B981),
            1 => ("MODO 2: INTERMEDIO AMPLIO (0x -> 30x -> 3000x)", 0xF59E0B),
            _ => ("MODO 3: BARRERA ABRUPTA DIRECTA (0x -> 3000x)", 0xEF4444),
        };
        draw_text(&mut fb, WIDTH, 20, 28, mode_text, mode_color);

        let status_text = format!(
            "TICK: {:05} | VELOCIDAD: {}x | FRENTE X: {} px | ESTADO: {}",
            tick, speed_multiplier, max_x, if is_paused { "PAUSADO" } else { "CORRIENDO" }
        );
        draw_text(&mut fb, WIDTH, 20, 44, &status_text, 0xE2E8F0);

        let colors = [0x2ECC71, 0xF1C40F, 0xE67E22, 0xEF4444, 0x8B5CF6];
        let labels = ["g0", "g1", "g2", "g3", "g4"];
        let mut gx = 20;
        for i in 0..5 {
            draw_rect(&mut fb, WIDTH, gx, 62, 10, 10, colors[i]);
            let g_str = format!("{}: {:<5}", labels[i], counts[i]);
            draw_text(&mut fb, WIDTH, gx + 14, 63, &g_str, 0xCBD5E1);
            gx += 110;
        }

        draw_text(&mut fb, WIDTH, 20, 80, "[ESPACIO] Alternar Paisaje  [P] Pausar  [R] Reset  [+/-] Velocidad", 0x94A3B8);

        // Lienzo de la Placa
        for y in 0..SIM_HEIGHT {
            let fb_y = OSD_HEIGHT + y;
            for x in 0..WIDTH {
                let cell = *cell_grid.current.get(x, y);
                let conc = *conc_grid.current.get(x, y);

                let pixel_color = if cell.is_alive() {
                    colors[cell.genotype() as usize]
                } else {
                    let norm = (conc / (3000.0 * c_s)).clamp(0.0, 1.0);
                    let intensity = (norm * 140.0) as u32;
                    (intensity / 4) << 16 | (intensity / 3) << 8 | (intensity + 15)
                };

                if x % zone_w == 0 {
                    fb[fb_y * WIDTH + x] = 0x475569;
                } else {
                    fb[fb_y * WIDTH + x] = pixel_color;
                }
            }
        }

        for z in 0..NUM_ZONES {
            let label_x = z * zone_w + 10;
            let mult = current_reservoirs[z] / c_s;
            let label = format!("ZONA {}: {:.0}x", z, mult);
            draw_text(&mut fb, WIDTH, label_x, OSD_HEIGHT + 10, &label, 0xFFFFFF);
        }

        window.update_with_buffer(&fb, WIDTH, HEIGHT).unwrap();
    }
}