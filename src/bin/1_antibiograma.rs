//! src/bin/1_antibiograma.rs
//! Simulador in vitro de Kirby-Bauer acoplado a perfiles fenotipicos TOML (CLSI / EUCAST).

use minifb::{Key, MouseButton, MouseMode, Window, WindowOptions};
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;

use tesis_ram::biology::rules::{try_colonize_empty_cell, update_living_cell, BiophysicalParams};
use tesis_ram::config::{load_organism_config, OrganismConfig};
use tesis_ram::core::cell::Cell;
use tesis_ram::core::grid::{DoubleBufferGrid, Grid};
use tesis_ram::physics::diffusion::{step_diffusion_2d_biofilm, DiffusionParams};
use tesis_ram::visualizer::osd::{draw_rect, draw_text};

const WIDTH: usize = 1000;
const HEIGHT: usize = 600;
const OSD_HEIGHT: usize = 90;

const DISH_CENTER_X: usize = 280;
const DISH_CENTER_Y: usize = 345;
const DISH_RADIUS: usize = 205; // 45 mm radio (Placa de 90 mm)
const DISK_RADIUS: usize = 14;  // 6 mm diametro (Papel de celulosa)

const NUM_DISKS: usize = 4;
const MM_PER_PIXEL: f64 = 90.0 / (2.0 * DISH_RADIUS as f64);
const MAX_INCUBATION_TICKS: usize = 2880;

#[derive(Clone, Copy)]
struct DiskPosition {
    x: usize,
    y: usize,
}

fn get_disk_positions() -> [DiskPosition; NUM_DISKS] {
    [
        DiskPosition { x: DISH_CENTER_X, y: DISH_CENTER_Y - 110 }, // 0: Norte
        DiskPosition { x: DISH_CENTER_X + 110, y: DISH_CENTER_Y }, // 1: Este
        DiskPosition { x: DISH_CENTER_X, y: DISH_CENTER_Y + 110 }, // 2: Sur
        DiskPosition { x: DISH_CENTER_X - 110, y: DISH_CENTER_Y }, // 3: Oeste
    ]
}

fn init_petri_dish(
    cell_grid: &mut DoubleBufferGrid<Cell>,
    drug_grids: &mut [DoubleBufferGrid<f64>; NUM_DISKS],
    disk_positions: &[DiskPosition; NUM_DISKS],
    config: &OrganismConfig,
    panel_offset: usize,
    initial_genotype: u8,
    rng: &mut Xoshiro256PlusPlus,
) {
    let r_sq = (DISH_RADIUS * DISH_RADIUS) as isize;
    let disk_r_sq = (DISK_RADIUS * DISK_RADIUS) as isize;

    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            cell_grid.current.set(x, y, Cell::EMPTY);
            cell_grid.next.set(x, y, Cell::EMPTY);

            for d in 0..NUM_DISKS {
                drug_grids[d].current.set(x, y, 0.0);
                drug_grids[d].next.set(x, y, 0.0);
            }

            let dx = x as isize - DISH_CENTER_X as isize;
            let dy = y as isize - DISH_CENTER_Y as isize;
            if dx * dx + dy * dy <= r_sq {
                if rng.random_bool(0.30) {
                    cell_grid.current.set(x, y, Cell::new(initial_genotype, 0, 80));
                }
            }
        }
    }

    // Cargar discos de antibiotico con sus concentraciones
    for (d, pos) in disk_positions.iter().enumerate() {
        let ab_idx = panel_offset + d;
        if ab_idx < config.antibiotics.len() {
            let ab = &config.antibiotics[ab_idx];
            for dy in -(DISK_RADIUS as isize)..=(DISK_RADIUS as isize) {
                for dx in -(DISK_RADIUS as isize)..=(DISK_RADIUS as isize) {
                    if dx * dx + dy * dy <= disk_r_sq {
                        let px = (pos.x as isize + dx) as usize;
                        let py = (pos.y as isize + dy) as usize;
                        drug_grids[d].current.set(px, py, ab.load_ug);
                        drug_grids[d].next.set(px, py, ab.load_ug);
                        cell_grid.current.set(px, py, Cell::EMPTY);
                    }
                }
            }
        }
    }
}

fn measure_halo_diameter(cell_grid: &Grid<Cell>, disk_x: usize, disk_y: usize) -> f64 {
    let num_rays = 16;
    let max_radius_px = 125usize;
    let mut min_clear_radius = max_radius_px;

    for i in 0..num_rays {
        let angle = (i as f64) * (2.0 * std::f64::consts::PI / num_rays as f64);
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        let mut ray_clear_dist = DISK_RADIUS;
        for r in DISK_RADIUS..max_radius_px {
            let px = (disk_x as f64 + r as f64 * cos_a) as usize;
            let py = (disk_y as f64 + r as f64 * sin_a) as usize;

            if px >= WIDTH || py >= HEIGHT {
                break;
            }

            let cell = cell_grid.get(px, py);
            let density = cell_grid.chamfer_density_5_7(px, py);

            // Borde del halo: primera aparicion de cesped denso (densidad >= 20)
            if cell.is_alive() && density >= 20 {
                ray_clear_dist = r;
                break;
            }
            ray_clear_dist = r;
        }

        if ray_clear_dist < min_clear_radius {
            min_clear_radius = ray_clear_dist;
        }
    }

    let diam_mm = 2.0 * (min_clear_radius as f64) * MM_PER_PIXEL;
    diam_mm.max(6.0)
}

fn main() {
    let mut window = Window::new(
        "Tesis RAM - Antibiograma Kirby-Bauer (CLSI M100 / EUCAST v16.0)",
        WIDTH,
        HEIGHT,
        WindowOptions::default(),
    )
    .expect("Fallo al inicializar minifb");

    window.set_target_fps(60);

    let mut fb = vec![0u32; WIDTH * HEIGHT];
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(1024);

    let config_files = [
        "config/ecoli_atcc25922.toml",
        "config/klebsiella_blee.toml",
        "config/pseudomonas_pao1.toml",
    ];
    let mut current_cfg_idx = 0usize;
    let mut initial_genotype = 0u8;
    let mut use_rescue_panel = false; // false = Panel 1ra Linea (0..3), true = Panel Rescate (4..7)
    let mut mouse_was_down = false;

    let mut current_config = load_organism_config(config_files[current_cfg_idx])
        .expect("Error al leer archivo TOML en config/");

    let disk_positions = get_disk_positions();

    let mut cell_grid = DoubleBufferGrid::new(WIDTH, HEIGHT, Cell::EMPTY);
    let mut drug_grids: [DoubleBufferGrid<f64>; NUM_DISKS] = [
        DoubleBufferGrid::new(WIDTH, HEIGHT, 0.0),
        DoubleBufferGrid::new(WIDTH, HEIGHT, 0.0),
        DoubleBufferGrid::new(WIDTH, HEIGHT, 0.0),
        DoubleBufferGrid::new(WIDTH, HEIGHT, 0.0),
    ];

    init_petri_dish(
        &mut cell_grid,
        &mut drug_grids,
        &disk_positions,
        &current_config,
        0,
        initial_genotype,
        &mut rng,
    );

    let mut tick = 0usize;
    let mut is_paused = false;
    let mut speed_multiplier = 1usize;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let mut reload = false;
        if window.is_key_pressed(Key::Key1, minifb::KeyRepeat::No) {
            current_cfg_idx = 0;
            use_rescue_panel = false;
            reload = true;
        }
        if window.is_key_pressed(Key::Key2, minifb::KeyRepeat::No) {
            current_cfg_idx = 1;
            use_rescue_panel = false;
            reload = true;
        }
        if window.is_key_pressed(Key::Key3, minifb::KeyRepeat::No) {
            current_cfg_idx = 2;
            use_rescue_panel = false;
            reload = true;
        }
        if window.is_key_pressed(Key::G, minifb::KeyRepeat::No) {
            initial_genotype = (initial_genotype + 1) % 5;
            reload = true;
        }

        // Detectar clic izquierdo del mouse dentro de la placa de Petri para alternar el Panel de Antibióticos
        let mouse_down = window.get_mouse_down(MouseButton::Left);
        if mouse_down && !mouse_was_down {
            if let Some((mx, my)) = window.get_mouse_pos(MouseMode::Clamp) {
                let dx = mx as isize - DISH_CENTER_X as isize;
                let dy = my as isize - DISH_CENTER_Y as isize;
                if dx * dx + dy * dy <= (DISH_RADIUS * DISH_RADIUS) as isize
                    && current_config.antibiotics.len() >= 8
                {
                    use_rescue_panel = !use_rescue_panel;
                    reload = true;
                }
            }
        }
        mouse_was_down = mouse_down;

        let panel_offset = if use_rescue_panel && current_config.antibiotics.len() >= 8 {
            4usize
        } else {
            0usize
        };

        if reload {
            current_config = load_organism_config(config_files[current_cfg_idx])
                .expect("Error al recargar TOML");
            let offset = if use_rescue_panel && current_config.antibiotics.len() >= 8 { 4 } else { 0 };
            init_petri_dish(
                &mut cell_grid,
                &mut drug_grids,
                &disk_positions,
                &current_config,
                offset,
                initial_genotype,
                &mut rng,
            );
            tick = 0;
        }

        if window.is_key_pressed(Key::Space, minifb::KeyRepeat::No) {
            is_paused = !is_paused;
        }
        if window.is_key_pressed(Key::R, minifb::KeyRepeat::No) {
            init_petri_dish(
                &mut cell_grid,
                &mut drug_grids,
                &disk_positions,
                &current_config,
                panel_offset,
                initial_genotype,
                &mut rng,
            );
            tick = 0;
        }
        if window.is_key_pressed(Key::Equal, minifb::KeyRepeat::No) {
            speed_multiplier = (speed_multiplier + 1).min(5);
        }
        if window.is_key_pressed(Key::Minus, minifb::KeyRepeat::No) {
            speed_multiplier = speed_multiplier.saturating_sub(1).max(1);
        }

        let r_sq = (DISH_RADIUS * DISH_RADIUS) as isize;
        let disk_r_sq = (DISK_RADIUS * DISK_RADIUS) as isize;
        let active_antibiotics = &current_config.antibiotics[panel_offset..(panel_offset + NUM_DISKS).min(current_config.antibiotics.len())];
        let num_ab = active_antibiotics.len();

        let mut bio_params_list = Vec::with_capacity(num_ab);
        for ab in active_antibiotics {
            bio_params_list.push(BiophysicalParams {
                base_division_prob: current_config.kinetics.base_division_prob,
                fitness_cost_per_mutation: current_config.kinetics.fitness_cost_per_mutation,
                maintenance_cost: current_config.kinetics.maintenance_cost,
                division_cost: current_config.kinetics.division_cost,
                min_resource_for_division: current_config.kinetics.min_resource_for_division,
                hill_e_max: ab.hill_e_max,
                hill_coefficient: ab.hill_coefficient,
                mic_by_genotype: ab.mic_genotypes,
                mpc_by_genotype: ab.mpc_genotypes,
                sos_base_mutation_prob: 0.0,
                sos_max_induction_factor: 1.0,
                sos_lethal_fraction: current_config.genetics.sos_lethal_fraction,
            });
        }

        // --- Simulacion Biofisica ---
        if !is_paused && tick < MAX_INCUBATION_TICKS {
            for _ in 0..speed_multiplier {
                if tick >= MAX_INCUBATION_TICKS {
                    break;
                }

                // 1. Difusion radial isotropica independiente para cada sensidisco activo
                for (d, ab) in active_antibiotics.iter().enumerate() {
                    let diff_params = DiffusionParams {
                        diffusion_coeff: ab.diffusion_coeff,
                        clearance_rate: ab.clearance_rate,
                        dx: 1.0,
                        dt: 1.0,
                    };

                    step_diffusion_2d_biofilm(
                        &drug_grids[d].current,
                        &mut drug_grids[d].next,
                        &cell_grid.current,
                        &diff_params,
                        ab.enzymatic_deg_rate,
                    );
                    drug_grids[d].swap();

                    let pos = &disk_positions[d];
                    for dy in -(DISK_RADIUS as isize)..=(DISK_RADIUS as isize) {
                        for dx in -(DISK_RADIUS as isize)..=(DISK_RADIUS as isize) {
                            if dx * dx + dy * dy <= disk_r_sq {
                                let px = (pos.x as isize + dx) as usize;
                                let py = (pos.y as isize + dy) as usize;
                                drug_grids[d].current.set(px, py, ab.load_ug);
                            }
                        }
                    }
                }

                // 2. Proliferacion e inhibicion celular evaluando los 4 discos simultáneamente
                for y in (DISH_CENTER_Y - DISH_RADIUS)..(DISH_CENTER_Y + DISH_RADIUS) {
                    for x in (DISH_CENTER_X - DISH_RADIUS)..(DISH_CENTER_X + DISH_RADIUS) {
                        let dx = x as isize - DISH_CENTER_X as isize;
                        let dy = y as isize - DISH_CENTER_Y as isize;
                        if dx * dx + dy * dy > r_sq {
                            continue;
                        }

                        let mut on_disk = false;
                        let mut min_dist_sq = usize::MAX;
                        let mut closest_d = 0;
                        for (d, pos) in disk_positions.iter().enumerate().take(num_ab) {
                            let ddx = (x as isize - pos.x as isize).unsigned_abs();
                            let ddy = (y as isize - pos.y as isize).unsigned_abs();
                            let dist_sq = ddx * ddx + ddy * ddy;
                            if dist_sq <= disk_r_sq as usize {
                                on_disk = true;
                                break;
                            }
                            if dist_sq < min_dist_sq {
                                min_dist_sq = dist_sq;
                                closest_d = d;
                            }
                        }

                        if on_disk {
                            cell_grid.next.set(x, y, Cell::EMPTY);
                            continue;
                        }

                        let cell = *cell_grid.current.get(x, y);
                        let density = cell_grid.current.chamfer_density_5_7(x, y);

                        if cell.is_alive() {
                            let mut current_c = cell;
                            for d in 0..num_ab {
                                let local_conc = *drug_grids[d].current.get(x, y);
                                current_c = update_living_cell(
                                    current_c,
                                    local_conc,
                                    density,
                                    true,
                                    &bio_params_list[d],
                                    1.0,
                                    &mut rng,
                                );
                                if !current_c.is_alive() {
                                    break;
                                }
                            }
                            cell_grid.next.set(x, y, current_c);
                        } else {
                            let neighbors = cell_grid.current.get_living_neighbors(x, y);
                            let candidate = try_colonize_empty_cell(
                                x,
                                y,
                                density,
                                &neighbors,
                                &drug_grids[closest_d].current,
                                &bio_params_list[closest_d],
                                &mut rng,
                            );

                            if candidate.is_alive() {
                                let g_idx = candidate.genotype() as usize;
                                let mut inhibited_by_other = false;
                                for d in 0..num_ab {
                                    let conc_d = *drug_grids[d].current.get(x, y);
                                    if conc_d >= bio_params_list[d].mic_by_genotype[g_idx] {
                                        inhibited_by_other = true;
                                        break;
                                    }
                                }
                                cell_grid.next.set(
                                    x,
                                    y,
                                    if inhibited_by_other { Cell::EMPTY } else { candidate },
                                );
                            } else {
                                cell_grid.next.set(x, y, Cell::EMPTY);
                            }
                        }
                    }
                }
                cell_grid.swap();
                tick += 1;
            }
        }

        // --- Medicion de Halos de Inhibicion ---
        let mut halo_diams = [0.0; NUM_DISKS];
        for d in 0..num_ab {
            halo_diams[d] = measure_halo_diameter(&cell_grid.current, disk_positions[d].x, disk_positions[d].y);
        }

        // --- Renderizado Visual ---
        fb.fill(0x0F172A);

        // Cabecera OSD
        draw_rect(&mut fb, WIDTH, 0, 0, WIDTH, OSD_HEIGHT, 0x1E293B);
        draw_rect(&mut fb, WIDTH, 0, OSD_HEIGHT - 2, WIDTH, 2, 0x38BDF8);

        draw_text(&mut fb, WIDTH, 20, 12, "UNIVERSIDAD MAYOR DE SAN ANDRES - INFORMATICA | TESIS RAM", 0x38BDF8);
        draw_text(&mut fb, WIDTH, 20, 28, "ANTIBIOGRAMA DE KIRBY-BAUER RADIAL DINAMICO (CLSI M100 / EUCAST)", 0xE2E8F0);

        let panel_name = if use_rescue_panel { "RESCATE (2da)" } else { "PRIMARIO (1ra)" };
        let status_str = format!(
            "TICK:{:04}/{} | {} | INOCULO:g{} | PANEL:{} | VEL:{}x",
            tick,
            MAX_INCUBATION_TICKS,
            current_config.species,
            initial_genotype,
            panel_name,
            speed_multiplier,
        );
        draw_text(&mut fb, WIDTH, 20, 46, &status_str, 0x4ADE80);

        draw_text(
            &mut fb,
            WIDTH,
            20,
            64,
            "[1-3] Cepa  [G] Inoculo g0..g4  [CLICK EN PLACA] Panel Rescate  [ESPACIO] Pausa  [R] Reset",
            0x94A3B8,
        );

        // Placa de Petri
        for dy in -(DISH_RADIUS as isize)..=(DISH_RADIUS as isize) {
            for dx in -(DISH_RADIUS as isize)..=(DISH_RADIUS as isize) {
                let dist_sq = dx * dx + dy * dy;
                if dist_sq <= r_sq {
                    let px = (DISH_CENTER_X as isize + dx) as usize;
                    let py = (DISH_CENTER_Y as isize + dy) as usize;

                    if dist_sq >= (DISH_RADIUS as isize - 3).pow(2) {
                        fb[py * WIDTH + px] = if use_rescue_panel { 0x38BDF8 } else { 0x64748B };
                        continue;
                    }

                    let cell = cell_grid.current.get(px, py);

                    if cell.is_alive() {
                        fb[py * WIDTH + px] = match cell.genotype() {
                            0 => 0x22C55E,
                            1 => 0xEAB308,
                            2 => 0xF97316,
                            3 => 0xEF4444,
                            _ => 0xA855F7,
                        };
                    } else {
                        let mut total_c = 0.0;
                        for d in 0..NUM_DISKS {
                            total_c += *drug_grids[d].current.get(px, py);
                        }
                        let gradient = (total_c * 4.0).clamp(0.0, 35.0) as u32;
                        let r_col = (0x24 + gradient).min(255);
                        let g_col = (0x1F + gradient / 2).min(255);
                        let b_col = 0x14;
                        fb[py * WIDTH + px] = (r_col << 16) | (g_col << 8) | b_col;
                    }
                }
            }
        }

        // Dibujar sensidiscos blancos con sus codigos
        for (d, pos) in disk_positions.iter().enumerate().take(num_ab) {
            let ab = &active_antibiotics[d];
            for dy in -(DISK_RADIUS as isize)..=(DISK_RADIUS as isize) {
                for dx in -(DISK_RADIUS as isize)..=(DISK_RADIUS as isize) {
                    if dx * dx + dy * dy <= disk_r_sq {
                        let px = (pos.x as isize + dx) as usize;
                        let py = (pos.y as isize + dy) as usize;
                        fb[py * WIDTH + px] = 0xF1F5F9;
                    }
                }
            }
            let label = &ab.code;
            let offset_x = (label.len() * 8) / 2;
            draw_text(&mut fb, WIDTH, pos.x - offset_x, pos.y - 4, label, 0x0F172A);
        }

        // Panel de Informe EUCAST v16.0
        let px = 550;
        let py = 120;
        let pw = 425;
        let ph = 445;

        draw_rect(&mut fb, WIDTH, px, py, pw, ph, 0x020617);
        draw_rect(&mut fb, WIDTH, px, py, pw, 1, 0x475569);
        draw_rect(&mut fb, WIDTH, px, py + ph, pw, 1, 0x475569);
        draw_rect(&mut fb, WIDTH, px, py, 1, ph, 0x475569);
        draw_rect(&mut fb, WIDTH, px + pw, py, 1, ph + 1, 0x475569);

        let header_title = if use_rescue_panel {
            "INFORME EUCAST v16.0 [PANEL RESCATE OMS]"
        } else {
            "INFORME DE SUSCEPTIBILIDAD (EUCAST v16.0)"
        };
        draw_text(&mut fb, WIDTH, px + 20, py + 16, header_title, 0x38BDF8);
        draw_rect(&mut fb, WIDTH, px + 20, py + 34, pw - 40, 1, 0x334155);

        let col_y = py + 48;
        draw_text(&mut fb, WIDTH, px + 20, col_y, "FARMACO", 0x94A3B8);
        draw_text(&mut fb, WIDTH, px + 120, col_y, "CARGA", 0x94A3B8);
        draw_text(&mut fb, WIDTH, px + 200, col_y, "HALO (mm)", 0x94A3B8);
        draw_text(&mut fb, WIDTH, px + 300, col_y, "CORTE", 0x94A3B8);
        draw_text(&mut fb, WIDTH, px + 370, col_y, "CAT", 0x94A3B8);

        for (d, ab) in active_antibiotics.iter().enumerate() {
            let row_y = col_y + 26 + d * 38;
            let diam = halo_diams[d];

            let (cat, cat_color, cat_bg) = if diam >= ab.breakpoint_s {
                (" S ", 0x020617, 0x22C55E)
            } else if diam >= ab.breakpoint_r {
                (" I ", 0x020617, 0xF59E0B)
            } else {
                (" R ", 0xFFFFFF, 0xEF4444)
            };

            draw_text(&mut fb, WIDTH, px + 20, row_y, &ab.code, 0xF8FAFC);
            let load_str = format!("{:.0} ug", ab.load_ug);
            draw_text(&mut fb, WIDTH, px + 120, row_y, &load_str, 0xCBD5E1);

            let diam_str = format!("{:.1}", diam);
            draw_text(&mut fb, WIDTH, px + 210, row_y, &diam_str, 0xF8FAFC);

            let bp_str = format!(">={:.0}", ab.breakpoint_s);
            draw_text(&mut fb, WIDTH, px + 300, row_y, &bp_str, 0x94A3B8);

            draw_rect(&mut fb, WIDTH, px + 365, row_y - 2, 32, 14, cat_bg);
            draw_text(&mut fb, WIDTH, px + 369, row_y + 1, cat, cat_color);
        }

        draw_rect(&mut fb, WIDTH, px + 20, py + 225, pw - 40, 1, 0x334155);
        draw_text(&mut fb, WIDTH, px + 20, py + 238, "ANALISIS FARMACODINAMICO:", 0x38BDF8);

        let mut lawn_cells = 0;
        for dy in -(DISH_RADIUS as isize)..=(DISH_RADIUS as isize) {
            for dx in -(DISH_RADIUS as isize)..=(DISH_RADIUS as isize) {
                if dx * dx + dy * dy <= r_sq {
                    let cx = (DISH_CENTER_X as isize + dx) as usize;
                    let cy = (DISH_CENTER_Y as isize + dy) as usize;
                    if cell_grid.current.get(cx, cy).is_alive() {
                        lawn_cells += 1;
                    }
                }
            }
        }
        let total_dish_cells = (std::f64::consts::PI * (DISH_RADIUS as f64).powi(2)) as usize;
        let coverage_pct = (lawn_cells as f64 / total_dish_cells as f64) * 100.0;

        let cov_str = format!("COBERTURA DE CESPED: {:.1}% ({}/{} parches)", coverage_pct, lawn_cells, total_dish_cells);
        draw_text(&mut fb, WIDTH, px + 20, py + 258, &cov_str, 0xCBD5E1);

        draw_text(&mut fb, WIDTH, px + 20, py + 280, "INTERPRETACION CLINICA:", 0xF1F5F9);

        let desc = &current_config.clinical_description;
        if desc.len() > 46 {
            draw_text(&mut fb, WIDTH, px + 20, py + 300, &desc[..46], 0xF59E0B);
            draw_text(&mut fb, WIDTH, px + 20, py + 316, &desc[46..], 0xF59E0B);
        } else {
            draw_text(&mut fb, WIDTH, px + 20, py + 300, desc, 0xF59E0B);
        }

        draw_rect(&mut fb, WIDTH, px + 20, py + 345, pw - 40, 1, 0x334155);
        draw_text(&mut fb, WIDTH, px + 20, py + 358, "PARAMETROS METROLOGICOS:", 0x94A3B8);
        draw_text(&mut fb, WIDTH, px + 20, py + 376, "Diametro placa: 90 mm | Sensidisco: 6 mm", 0x64748B);
        draw_text(&mut fb, WIDTH, px + 20, py + 394, "Agar: Mueller-Hinton con difusion radial Fick", 0x64748B);
        draw_text(&mut fb, WIDTH, px + 20, py + 412, "Criterio CFL: alpha <= 0.24 (Estencil Isotropico 9p)", 0x64748B);

        window.update_with_buffer(&fb, WIDTH, HEIGHT).unwrap();
    }
}