//! src/bin/5_clinica_a_laboratorio.rs
//! Ciclo Traslacional Bedside-to-Bench: Fracaso terapéutico in vivo -> Biopsia -> Re-antibiograma in vitro.

use minifb::{Key, Window, WindowOptions};
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;

use tesis_ram::biology::rules::{try_colonize_empty_cell, update_living_cell, BiophysicalParams};
use tesis_ram::config::{load_organism_config, OrganismConfig};
use tesis_ram::core::cell::Cell;
use tesis_ram::core::grid::{DoubleBufferGrid, Grid};
use tesis_ram::physics::bateman::BatemanRegimen;
use tesis_ram::physics::diffusion::{step_diffusion_2d, DiffusionParams};
use tesis_ram::visualizer::osd::{draw_rect, draw_text};

const WIDTH: usize = 1000;
const HEIGHT: usize = 600;
const OSD_HEIGHT: usize = 90;

// Geometría Fase 1: Tejido vascularizado in vivo
const TISSUE_X: usize = 30;
const TISSUE_Y: usize = 122;
const TISSUE_W: usize = 440;
const TISSUE_H: usize = 448;

// Geometría Fase 2: Placa de Petri in vitro (Kirby-Bauer 90 mm)
const DISH_CENTER_X: usize = 280;
const DISH_CENTER_Y: usize = 345;
const DISH_RADIUS: usize = 205; // Radio 45 mm (1 px ≈ 0.2195 mm)
const DISK_RADIUS: usize = 14;  // Sensidisco 6 mm de diámetro
const NUM_DISKS: usize = 4;
const MM_PER_PIXEL: f64 = 90.0 / (2.0 * DISH_RADIUS as f64);

#[derive(Clone, Copy, PartialEq, Eq)]
enum SimulationPhase {
    ClinicalInVivo,
    LaboratoryInVitro,
}

#[derive(Clone, Copy)]
struct CapillaryPoint {
    x: usize,
    y: usize,
}

#[derive(Clone, Copy)]
struct DiskPosition {
    x: usize,
    y: usize,
}

fn get_capillary_network() -> Vec<CapillaryPoint> {
    let mut capillaries = Vec::with_capacity(256);
    let cols = 8;
    let rows = 8;
    let margin = 18.0;
    let step_x = (TISSUE_W as f64 - 2.0 * margin) / (cols as f64 - 1.0);
    let step_y = (TISSUE_H as f64 - 2.0 * margin) / (rows as f64 - 1.0);

    for r in 0..rows {
        for c in 0..cols {
            let cx = (TISSUE_X as f64 + margin + c as f64 * step_x) as usize;
            let cy = (TISSUE_Y as f64 + margin + r as f64 * step_y) as usize;

            for dy in -1..=1 {
                for dx in -1..=1 {
                    if dx * dx + dy * dy <= 2 {
                        capillaries.push(CapillaryPoint {
                            x: (cx as isize + dx) as usize,
                            y: (cy as isize + dy) as usize,
                        });
                    }
                }
            }
        }
    }
    capillaries
}

fn get_disk_positions() -> [DiskPosition; NUM_DISKS] {
    [
        DiskPosition { x: DISH_CENTER_X, y: DISH_CENTER_Y - 110 }, // 0: Norte
        DiskPosition { x: DISH_CENTER_X + 110, y: DISH_CENTER_Y }, // 1: Este
        DiskPosition { x: DISH_CENTER_X, y: DISH_CENTER_Y + 110 }, // 2: Sur
        DiskPosition { x: DISH_CENTER_X - 110, y: DISH_CENTER_Y }, // 3: Oeste
    ]
}

fn init_infection(
    cell_grid: &mut DoubleBufferGrid<Cell>,
    drug_grid: &mut DoubleBufferGrid<f64>,
    capillaries: &[CapillaryPoint],
    rng: &mut Xoshiro256PlusPlus,
) {
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            cell_grid.current.set(x, y, Cell::EMPTY);
            cell_grid.next.set(x, y, Cell::EMPTY);
            drug_grid.current.set(x, y, 0.0);
            drug_grid.next.set(x, y, 0.0);
        }
    }

    for y in TISSUE_Y..(TISSUE_Y + TISSUE_H) {
        for x in TISSUE_X..(TISSUE_X + TISSUE_W) {
            let is_vessel = capillaries.iter().any(|c| c.x == x && c.y == y);
            if !is_vessel && rng.random_bool(0.40) {
                cell_grid.current.set(x, y, Cell::new(0, 0, 100));
            }
        }
    }
}

/// Inocula la placa de Petri in vitro muestreando la distribución clonal de la biopsia
fn init_petri_dish_from_biopsy(
    cell_grid: &mut DoubleBufferGrid<Cell>,
    petri_drugs: &mut [DoubleBufferGrid<f64>; NUM_DISKS],
    disk_positions: &[DiskPosition; NUM_DISKS],
    config: &OrganismConfig,
    biopsy_distribution: &[f64; 5],
    rng: &mut Xoshiro256PlusPlus,
) {
    let r_sq = (DISH_RADIUS * DISH_RADIUS) as isize;
    let disk_r_sq = (DISK_RADIUS * DISK_RADIUS) as isize;

    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            cell_grid.current.set(x, y, Cell::EMPTY);
            cell_grid.next.set(x, y, Cell::EMPTY);

            for d in 0..NUM_DISKS {
                petri_drugs[d].current.set(x, y, 0.0);
                petri_drugs[d].next.set(x, y, 0.0);
            }

            let dx = x as isize - DISH_CENTER_X as isize;
            let dy = y as isize - DISH_CENTER_Y as isize;
            if dx * dx + dy * dy <= r_sq && rng.random_bool(0.35) {
                // Seleccionar genotipo estocásticamente ponderado según la biopsia
                let roll = rng.random::<f64>();
                let mut cum = 0.0;
                let mut assigned_g = 0u8;
                for (g, &p) in biopsy_distribution.iter().enumerate() {
                    cum += p;
                    if roll <= cum {
                        assigned_g = g as u8;
                        break;
                    }
                }
                cell_grid.current.set(x, y, Cell::new(assigned_g, 0, 80));
            }
        }
    }

    // Inicializar reservorios en los 4 sensidiscos
    for (d, pos) in disk_positions.iter().enumerate().take(config.antibiotics.len().min(NUM_DISKS)) {
        let ab = &config.antibiotics[d];
        for dy in -(DISK_RADIUS as isize)..=(DISK_RADIUS as isize) {
            for dx in -(DISK_RADIUS as isize)..=(DISK_RADIUS as isize) {
                if dx * dx + dy * dy <= disk_r_sq {
                    let px = (pos.x as isize + dx) as usize;
                    let py = (pos.y as isize + dy) as usize;
                    petri_drugs[d].current.set(px, py, ab.load_ug);
                    petri_drugs[d].next.set(px, py, ab.load_ug);
                    cell_grid.current.set(px, py, Cell::EMPTY);
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

fn log_pk_to_y(conc: f64, gy: usize, gh: usize) -> usize {
    let log_min = -2.3f64;
    let log_max = 2.60f64;
    let log_c = conc.max(0.005).log10();
    let norm = ((log_c - log_min) / (log_max - log_min)).clamp(0.0, 1.0);
    (gy + gh) - (norm * (gh - 45) as f64) as usize
}

fn main() {
    let mut window = Window::new(
        "Tesis RAM - Ciclo Traslacional Bedside-to-Bench (Clinica -> Laboratorio)",
        WIDTH,
        HEIGHT,
        WindowOptions::default(),
    )
    .expect("Fallo al inicializar minifb en WSL");

    window.set_target_fps(60);

    let mut fb = vec![0u32; WIDTH * HEIGHT];
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(2026);

    let config_files = [
        "config/ecoli_atcc25922.toml",
        "config/klebsiella_blee.toml",
        "config/pseudomonas_pao1.toml",
    ];
    let mut current_cfg_idx = 0usize;
    let mut config: OrganismConfig = load_organism_config(config_files[current_cfg_idx])
        .expect("Error al leer archivo TOML en config/");

    let capillaries = get_capillary_network();
    let disk_positions = get_disk_positions();

    let regimens_tau = [480usize, 720usize, 1440usize];
    let regimens_labels = ["q8h (cada 8 h)", "q12h (cada 12 h)", "q24h (cada 24 h)"];
    let mut current_reg_idx = 0usize;

    let mut current_ab_idx = 0usize;
    let ab_init = &config.antibiotics[current_ab_idx];
    let mut pk = BatemanRegimen::new(0.038, 0.0048, ab_init.c_max_plasma, regimens_tau[current_reg_idx]);

    // Redes espaciales para Fase 1 (In Vivo)
    let mut tissue_cells = DoubleBufferGrid::new(WIDTH, HEIGHT, Cell::EMPTY);
    let mut tissue_drug = DoubleBufferGrid::new(WIDTH, HEIGHT, 0.0);
    let tissue_diff_params = DiffusionParams {
        diffusion_coeff: 0.50,
        clearance_rate: 0.00018,
        dx: 1.0,
        dt: 1.0,
    };

    // Redes espaciales para Fase 2 (In Vitro)
    let mut petri_cells = DoubleBufferGrid::new(WIDTH, HEIGHT, Cell::EMPTY);
    let mut petri_drugs: [DoubleBufferGrid<f64>; NUM_DISKS] = [
        DoubleBufferGrid::new(WIDTH, HEIGHT, 0.0),
        DoubleBufferGrid::new(WIDTH, HEIGHT, 0.0),
        DoubleBufferGrid::new(WIDTH, HEIGHT, 0.0),
        DoubleBufferGrid::new(WIDTH, HEIGHT, 0.0),
    ];

    init_infection(&mut tissue_cells, &mut tissue_drug, &capillaries, &mut rng);

    let mut phase = SimulationPhase::ClinicalInVivo;
    let mut tick_invivo = 0usize;
    let mut tick_invitro = 0usize;
    let mut is_paused = false;
    let mut speed_multiplier = 1usize;

    let mut pk_history = Vec::with_capacity(450);
    let mut ca_history = Vec::with_capacity(450);
    let mut time_above_mic_count = 0usize;
    let mut total_monitored_ticks = 0usize;

    let mut biopsy_counts = [0usize; 5];
    let mut biopsy_distribution = [1.0, 0.0, 0.0, 0.0, 0.0];
    let mut biopsy_minute = 0usize;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let mut reload_simulation = false;

        if window.is_key_pressed(Key::Key1, minifb::KeyRepeat::No) {
            current_cfg_idx = 0;
            current_ab_idx = 0;
            reload_simulation = true;
        }
        if window.is_key_pressed(Key::Key2, minifb::KeyRepeat::No) {
            current_cfg_idx = 1;
            current_ab_idx = 0;
            reload_simulation = true;
        }
        if window.is_key_pressed(Key::Key3, minifb::KeyRepeat::No) {
            current_cfg_idx = 2;
            current_ab_idx = 0;
            reload_simulation = true;
        }

        if window.is_key_pressed(Key::A, minifb::KeyRepeat::No) && phase == SimulationPhase::ClinicalInVivo {
            current_ab_idx = (current_ab_idx + 1) % config.antibiotics.len();
            reload_simulation = true;
        }

        if window.is_key_pressed(Key::T, minifb::KeyRepeat::No) && phase == SimulationPhase::ClinicalInVivo {
            current_reg_idx = (current_reg_idx + 1) % 3;
            reload_simulation = true;
        }

        if window.is_key_pressed(Key::O, minifb::KeyRepeat::No) && phase == SimulationPhase::ClinicalInVivo {
            pk.toggle_next_dose(tick_invivo);
        }

        // [B] Extraer biopsia clínica y sembrar placa de Petri (Cambio de Fase)
        if window.is_key_pressed(Key::B, minifb::KeyRepeat::No) {
            match phase {
                SimulationPhase::ClinicalInVivo => {
                    // Muestrear células vivas en el tejido
                    let mut counts = [0usize; 5];
                    let mut total = 0usize;
                    for y in TISSUE_Y..(TISSUE_Y + TISSUE_H) {
                        for x in TISSUE_X..(TISSUE_X + TISSUE_W) {
                            let cell = *tissue_cells.current.get(x, y);
                            if cell.is_alive() {
                                counts[cell.genotype() as usize] += 1;
                                total += 1;
                            }
                        }
                    }

                    if total > 0 {
                        biopsy_counts = counts;
                        biopsy_minute = tick_invivo;
                        for g in 0..5 {
                            biopsy_distribution[g] = counts[g] as f64 / total as f64;
                        }

                        // Inocular placa Petri con la muestra del paciente
                        init_petri_dish_from_biopsy(
                            &mut petri_cells,
                            &mut petri_drugs,
                            &disk_positions,
                            &config,
                            &biopsy_distribution,
                            &mut rng,
                        );
                        tick_invitro = 0;
                        phase = SimulationPhase::LaboratoryInVitro;
                    }
                }
                SimulationPhase::LaboratoryInVitro => {
                    // Volver a la cabecera clínica
                    phase = SimulationPhase::ClinicalInVivo;
                }
            }
        }

        if window.is_key_pressed(Key::Space, minifb::KeyRepeat::No) {
            is_paused = !is_paused;
        }

        if window.is_key_pressed(Key::R, minifb::KeyRepeat::No) {
            reload_simulation = true;
        }

        if reload_simulation {
            config = load_organism_config(config_files[current_cfg_idx]).expect("Error al recargar TOML");
            let ab_selected = &config.antibiotics[current_ab_idx.min(config.antibiotics.len() - 1)];
            pk = BatemanRegimen::new(0.038, 0.0048, ab_selected.c_max_plasma, regimens_tau[current_reg_idx]);
            init_infection(&mut tissue_cells, &mut tissue_drug, &capillaries, &mut rng);
            pk_history.clear();
            ca_history.clear();
            time_above_mic_count = 0;
            total_monitored_ticks = 0;
            tick_invivo = 0;
            tick_invitro = 0;
            phase = SimulationPhase::ClinicalInVivo;
        }

        if window.is_key_pressed(Key::Equal, minifb::KeyRepeat::No) {
            speed_multiplier = (speed_multiplier + 1).min(5);
        }
        if window.is_key_pressed(Key::Minus, minifb::KeyRepeat::No) {
            speed_multiplier = speed_multiplier.saturating_sub(1).max(1);
        }

        let ab_active = &config.antibiotics[current_ab_idx.min(config.antibiotics.len() - 1)];
        let mic_s = ab_active.mic_genotypes[0];
        let mpc_s = ab_active.mpc_genotypes[0];

        // --- Ejecución según la Fase Activa ---
        match phase {
            SimulationPhase::ClinicalInVivo => {
                let bio_params = BiophysicalParams {
                    base_division_prob: config.kinetics.base_division_prob,
                    fitness_cost_per_mutation: config.kinetics.fitness_cost_per_mutation,
                    maintenance_cost: config.kinetics.maintenance_cost,
                    division_cost: config.kinetics.division_cost,
                    min_resource_for_division: config.kinetics.min_resource_for_division,
                    hill_e_max: ab_active.hill_e_max,
                    hill_coefficient: ab_active.hill_coefficient,
                    mic_by_genotype: ab_active.mic_genotypes,
                    mpc_by_genotype: ab_active.mpc_genotypes,
                    sos_base_mutation_prob: config.genetics.sos_base_mutation_prob,
                    sos_max_induction_factor: config.genetics.sos_max_induction_factor,
                    sos_lethal_fraction: config.genetics.sos_lethal_fraction,
                };

                if !is_paused {
                    for _ in 0..speed_multiplier {
                        let c_plasma = pk.concentration_at(tick_invivo);
                        if c_plasma >= mic_s {
                            time_above_mic_count += 1;
                        }
                        total_monitored_ticks += 1;

                        step_diffusion_2d(&tissue_drug.current, &mut tissue_drug.next, &tissue_diff_params);
                        tissue_drug.swap();

                        for cap in &capillaries {
                            tissue_drug.current.set(cap.x, cap.y, c_plasma);
                        }

                        for y in TISSUE_Y..(TISSUE_Y + TISSUE_H) {
                            for x in TISSUE_X..(TISSUE_X + TISSUE_W) {
                                let is_vessel = capillaries.iter().any(|c| c.x == x && c.y == y);
                                if is_vessel {
                                    tissue_cells.next.set(x, y, Cell::EMPTY);
                                    continue;
                                }

                                let local_c = *tissue_drug.current.get(x, y);
                                let cell = *tissue_cells.current.get(x, y);
                                let density = tissue_cells.current.chamfer_density_5_7(x, y);

                                if cell.is_alive() {
                                    let mut updated = update_living_cell(cell, local_c, density, false, &bio_params, 1.0, &mut rng);
                                    if updated.is_alive() {
                                        updated.add_resource(1);
                                    }
                                    tissue_cells.next.set(x, y, updated);
                                } else {
                                    let neighbors = tissue_cells.current.get_living_neighbors(x, y);
                                    let colonized = try_colonize_empty_cell(
                                        x,
                                        y,
                                        density,
                                        &neighbors,
                                        &tissue_drug.current,
                                        &bio_params,
                                        &mut rng,
                                    );
                                    tissue_cells.next.set(x, y, colonized);
                                }
                            }
                        }
                        tissue_cells.swap();
                        tick_invivo += 1;
                    }
                }

                let mut total_bacteria = 0usize;
                let mut counts = [0usize; 5];
                for y in TISSUE_Y..(TISSUE_Y + TISSUE_H) {
                    for x in TISSUE_X..(TISSUE_X + TISSUE_W) {
                        let cell = *tissue_cells.current.get(x, y);
                        if cell.is_alive() {
                            counts[cell.genotype() as usize] += 1;
                            total_bacteria += 1;
                        }
                    }
                }

                let c_plasma_now = pk.concentration_at(tick_invivo);
                if !is_paused {
                    pk_history.push(c_plasma_now);
                    ca_history.push(total_bacteria);
                    if pk_history.len() > 420 {
                        pk_history.remove(0);
                        ca_history.remove(0);
                    }
                }

                let pct_time_above_mic = if total_monitored_ticks > 0 {
                    (time_above_mic_count as f64 / total_monitored_ticks as f64) * 100.0
                } else {
                    0.0
                };

                // --- Renderizado Fase 1 (In Vivo) ---
                fb.fill(0x0F172A);

                draw_rect(&mut fb, WIDTH, 0, 0, WIDTH, OSD_HEIGHT, 0x1E293B);
                draw_rect(&mut fb, WIDTH, 0, OSD_HEIGHT - 2, WIDTH, 2, 0x38BDF8);

                draw_text(&mut fb, WIDTH, 20, 10, "UNIVERSIDAD MAYOR DE SAN ANDRES - INFORMATICA | TESIS RAM", 0x38BDF8);
                draw_text(&mut fb, WIDTH, 20, 26, "FASE 1: TRATAMIENTO CLINICO IN VIVO (PRESIONAR [B] PARA BIOPSIA)", 0xFBBF24);

                let current_dose_num = ((tick_invivo / pk.tau_minutes) + 1).min(pk.total_doses);
                let status_str = format!(
                    "MIN: {:04} | {} | {}: {} | DOSIS: {}/{} | ADH: {:.0}%",
                    tick_invivo,
                    config.species,
                    ab_active.code,
                    ab_active.name,
                    current_dose_num,
                    pk.total_doses,
                    pk.adherence_pct(tick_invivo)
                );
                draw_text(&mut fb, WIDTH, 20, 44, &status_str, 0x4ADE80);

                draw_text(
                    &mut fb,
                    WIDTH,
                    20,
                    62,
                    "[B] EXTRAER BIOPSIA  [O] Omitir/Tomar  [A] Farmaco  [T] Pauta  [1..3] Cepa  [R] Reset",
                    0x94A3B8,
                );

                // Calendario de pastillero
                draw_rect(&mut fb, WIDTH, 0, 90, WIDTH, 28, 0x111827);
                draw_rect(&mut fb, WIDTH, 0, 117, WIDTH, 1, 0x334155);
                draw_text(&mut fb, WIDTH, 20, 99, "CALENDARIO:", 0x94A3B8);

                let box_w = 26usize;
                let box_h = 16usize;
                let start_box_x = 120usize;
                let next_idx_opt = pk.next_dose_index(tick_invivo);
                let current_elapsed_idx = tick_invivo / pk.tau_minutes;

                for d in 0..pk.total_doses {
                    let bx = start_box_x + d * (box_w + 5);
                    let by = 96;

                    let is_past = d <= current_elapsed_idx;
                    let is_next = Some(d) == next_idx_opt;
                    let is_taken = pk.doses_taken[d];

                    let (bg_col, text_col) = if is_past {
                        if is_taken { (0x166534, 0xDCFCE7) } else { (0x991B1B, 0xFEE2E2) }
                    } else if is_next {
                        if is_taken { (0x0284C7, 0xF0F9FF) } else { (0xDC2626, 0xFFFFFF) }
                    } else {
                        (0x1F2937, 0x6B7280)
                    };

                    draw_rect(&mut fb, WIDTH, bx, by, box_w, box_h, bg_col);

                    if is_next {
                        let border_col = if is_taken { 0x38BDF8 } else { 0xFCA5A5 };
                        draw_rect(&mut fb, WIDTH, bx, by, box_w, 1, border_col);
                        draw_rect(&mut fb, WIDTH, bx, by + box_h - 1, box_w, 1, border_col);
                        draw_rect(&mut fb, WIDTH, bx, by, 1, box_h, border_col);
                        draw_rect(&mut fb, WIDTH, bx + box_w - 1, by, 1, box_h, border_col);
                    }

                    let num_str = format!("{:>2}", d + 1);
                    draw_text(&mut fb, WIDTH, bx + 5, by + 4, &num_str, text_col);
                }

                if let Some(n_idx) = next_idx_opt {
                    let next_minute = n_idx * pk.tau_minutes;
                    let mins_left = next_minute.saturating_sub(tick_invivo);
                    let next_state_str = if pk.doses_taken[n_idx] {
                        format!("Prox: D{} en {}m [TOMAR]", n_idx + 1, mins_left)
                    } else {
                        format!("Prox: D{} en {}m [OMITIDA!]", n_idx + 1, mins_left)
                    };
                    let state_col = if pk.doses_taken[n_idx] { 0x38BDF8 } else { 0xEF4444 };
                    draw_text(&mut fb, WIDTH, start_box_x + pk.total_doses * 31 + 10, 99, &next_state_str, state_col);
                }

                // Tejido vascular
                draw_rect(&mut fb, WIDTH, TISSUE_X - 2, TISSUE_Y - 2, TISSUE_W + 4, TISSUE_H + 4, 0x334155);

                for y in TISSUE_Y..(TISSUE_Y + TISSUE_H) {
                    for x in TISSUE_X..(TISSUE_X + TISSUE_W) {
                        let is_vessel = capillaries.iter().any(|c| c.x == x && c.y == y);
                        if is_vessel {
                            fb[y * WIDTH + x] = 0xEF4444;
                            continue;
                        }

                        let cell = *tissue_cells.current.get(x, y);
                        if cell.is_alive() {
                            fb[y * WIDTH + x] = match cell.genotype() {
                                0 => 0x22C55E,
                                1 => 0xEAB308,
                                2 => 0xF97316,
                                3 => 0xDC2626,
                                _ => 0xA855F7,
                            };
                        } else {
                            let conc = *tissue_drug.current.get(x, y);
                            let norm = (conc / ab_active.c_max_plasma).clamp(0.0, 1.0);
                            let intensity = (norm * 75.0) as u32;
                            let r_tis = 0x30 + intensity / 2;
                            let g_tis = 0x1A + intensity / 3;
                            let b_tis = 0x25 + intensity;
                            fb[y * WIDTH + x] = (r_tis << 16) | (g_tis << 8) | b_tis;
                        }
                    }
                }
                draw_text(&mut fb, WIDTH, TISSUE_X + 10, TISSUE_Y + TISSUE_H - 18, "LECHO VASCULAR TISULAR (64 CAPILARES)", 0xF1F5F9);

                // Gráfica PK
                let gx = 510;
                let gy = 122;
                let gw = 460;
                let gh = 180;

                draw_rect(&mut fb, WIDTH, gx, gy, gw, gh, 0x020617);
                draw_rect(&mut fb, WIDTH, gx, gy, gw, 1, 0x475569);
                draw_rect(&mut fb, WIDTH, gx, gy + gh, gw, 1, 0x475569);
                draw_rect(&mut fb, WIDTH, gx, gy, 1, gh, 0x475569);
                draw_rect(&mut fb, WIDTH, gx + gw, gy, 1, gh + 1, 0x475569);

                let title_pk = format!("FARMACOCINETICA C(t) [{}]", ab_active.code);
                draw_text(&mut fb, WIDTH, gx + 15, gy + 12, &title_pk, 0x38BDF8);

                let c_act_label = format!("C_act: {:.2} ug/mL", c_plasma_now);
                draw_text(&mut fb, WIDTH, gx + 230, gy + 12, &c_act_label, 0x4ADE80);

                let y_mic = log_pk_to_y(mic_s, gy, gh);
                let y_mpc = log_pk_to_y(mpc_s, gy, gh);

                for sy in y_mpc..=y_mic {
                    for sx in (gx + 1)..(gx + gw) {
                        let orig = fb[sy * WIDTH + sx];
                        let r_m = ((orig >> 16) & 0xFF) + 16;
                        let g_m = ((orig >> 8) & 0xFF) + 12;
                        fb[sy * WIDTH + sx] = (r_m << 16) | (g_m << 8) | (orig & 0xFF);
                    }
                }

                for sx in (gx + 1)..(gx + gw) {
                    fb[y_mic * WIDTH + sx] = 0xF59E0B;
                    fb[y_mpc * WIDTH + sx] = 0xEF4444;
                }

                let mic_label = format!("CMI: {:.3}", mic_s);
                let mpc_label = format!("MPC: {:.3}", mpc_s);
                draw_text(&mut fb, WIDTH, gx + gw - 95, y_mic - 9, &mic_label, 0xF59E0B);
                draw_text(&mut fb, WIDTH, gx + gw - 95, y_mpc - 9, &mpc_label, 0xEF4444);

                for i in 1..pk_history.len() {
                    let px = gx + 15 + i;
                    if px < gx + gw - 10 {
                        let py = log_pk_to_y(pk_history[i], gy, gh);
                        fb[py * WIDTH + px] = 0x38BDF8;
                        fb[(py + 1) * WIDTH + px] = 0x38BDF8;
                    }
                }

                // Carga bacteriana
                let dy_g = 325;
                let dh_g = 150;

                draw_rect(&mut fb, WIDTH, gx, dy_g, gw, dh_g, 0x020617);
                draw_rect(&mut fb, WIDTH, gx, dy_g, gw, 1, 0x475569);
                draw_rect(&mut fb, WIDTH, gx, dy_g + dh_g, gw, 1, 0x475569);
                draw_rect(&mut fb, WIDTH, gx, dy_g, 1, dh_g, 0x475569);
                draw_rect(&mut fb, WIDTH, gx + gw, dy_g, 1, dh_g + 1, 0x475569);

                draw_text(&mut fb, WIDTH, gx + 15, dy_g + 12, "CARGA BACTERIANA INTERSTICIAL", 0xE2E8F0);

                let max_pop = 160000.0;
                for i in 1..ca_history.len() {
                    let px = gx + 15 + i;
                    if px < gx + gw - 10 {
                        let py = (dy_g + dh_g) - ((ca_history[i] as f64 / max_pop) * (dh_g - 35) as f64).clamp(0.0, (dh_g - 35) as f64) as usize;
                        let pop_col = if counts[1] + counts[2] + counts[3] + counts[4] > counts[0] {
                            0xEF4444
                        } else {
                            0x22C55E
                        };
                        fb[py * WIDTH + px] = pop_col;
                        fb[(py + 1) * WIDTH + px] = pop_col;
                    }
                }

                // Diagnóstico y llamado a biopsia
                let info_y = 500;
                draw_rect(&mut fb, WIDTH, gx, info_y, gw, 70, 0x1E293B);
                draw_rect(&mut fb, WIDTH, gx, info_y, gw, 1, 0x475569);

                let pkpd_str = format!("INDICE PK/PD: %T>CMI = {:.1}% (Pauta: {})", pct_time_above_mic, regimens_labels[current_reg_idx]);
                draw_text(&mut fb, WIDTH, gx + 15, info_y + 10, &pkpd_str, if pct_time_above_mic >= 60.0 { 0x4ADE80 } else { 0xF87171 });

                let pop_str = format!(
                    "POBLACION: {} celdas (g0:{} g1:{} g2:{} g3:{} g4:{})",
                    total_bacteria, counts[0], counts[1], counts[2], counts[3], counts[4]
                );
                draw_text(&mut fb, WIDTH, gx + 15, info_y + 28, &pop_str, 0xCBD5E1);

                let biopsy_call = if counts[1] + counts[2] + counts[3] + counts[4] > counts[0] {
                    ("FRACASO RAM DETECTADO: PULSE [B] PARA BIOPSIA", 0xEF4444)
                } else if total_bacteria < 5000 {
                    ("INFECCION CONTROLADA: PULSE [B] PARA CULTIVO", 0x4ADE80)
                } else {
                    ("EN TRATAMIENTO: PULSE [B] PARA MUESTREAR", 0xF59E0B)
                };
                draw_text(&mut fb, WIDTH, gx + 15, info_y + 46, biopsy_call.0, biopsy_call.1);
            }

            SimulationPhase::LaboratoryInVitro => {
                let r_sq = (DISH_RADIUS * DISH_RADIUS) as isize;
                let disk_r_sq = (DISK_RADIUS * DISK_RADIUS) as isize;

                if !is_paused {
                    for _ in 0..speed_multiplier {
                        // 1. Difusión radial independiente para cada sensidisco
                        for (d, ab) in config.antibiotics.iter().enumerate().take(NUM_DISKS) {
                            let diff_params = DiffusionParams {
                                diffusion_coeff: ab.diffusion_coeff,
                                clearance_rate: ab.clearance_rate,
                                dx: 1.0,
                                dt: 1.0,
                            };

                            step_diffusion_2d(&petri_drugs[d].current, &mut petri_drugs[d].next, &diff_params);
                            petri_drugs[d].swap();

                            let pos = &disk_positions[d];
                            for dy in -(DISK_RADIUS as isize)..=(DISK_RADIUS as isize) {
                                for dx in -(DISK_RADIUS as isize)..=(DISK_RADIUS as isize) {
                                    if dx * dx + dy * dy <= disk_r_sq {
                                        let px = (pos.x as isize + dx) as usize;
                                        let py = (pos.y as isize + dy) as usize;
                                        petri_drugs[d].current.set(px, py, ab.load_ug);
                                    }
                                }
                            }
                        }

                        // 2. Dinámica del césped sembrado con la biopsia
                        for y in (DISH_CENTER_Y - DISH_RADIUS)..(DISH_CENTER_Y + DISH_RADIUS) {
                            for x in (DISH_CENTER_X - DISH_RADIUS)..(DISH_CENTER_X + DISH_RADIUS) {
                                let dx = x as isize - DISH_CENTER_X as isize;
                                let dy = y as isize - DISH_CENTER_Y as isize;
                                if dx * dx + dy * dy > r_sq {
                                    continue;
                                }

                                let mut min_dist_sq = usize::MAX;
                                let mut closest_d = 0;
                                for (d, pos) in disk_positions.iter().enumerate() {
                                    let ddx = (x as isize - pos.x as isize).abs() as usize;
                                    let ddy = (y as isize - pos.y as isize).abs() as usize;
                                    let dist_sq = ddx * ddx + ddy * ddy;
                                    if dist_sq < min_dist_sq {
                                        min_dist_sq = dist_sq;
                                        closest_d = d;
                                    }
                                }

                                if min_dist_sq <= disk_r_sq as usize {
                                    petri_cells.next.set(x, y, Cell::EMPTY);
                                    continue;
                                }

                                let ab = &config.antibiotics[closest_d];
                                let local_conc = *petri_drugs[closest_d].current.get(x, y);

                                let bio_params = BiophysicalParams {
                                    base_division_prob: config.kinetics.base_division_prob,
                                    fitness_cost_per_mutation: config.kinetics.fitness_cost_per_mutation,
                                    maintenance_cost: config.kinetics.maintenance_cost,
                                    division_cost: config.kinetics.division_cost,
                                    min_resource_for_division: config.kinetics.min_resource_for_division,
                                    hill_e_max: ab.hill_e_max,
                                    hill_coefficient: ab.hill_coefficient,
                                    mic_by_genotype: ab.mic_genotypes,
                                    mpc_by_genotype: ab.mpc_genotypes,
                                    sos_base_mutation_prob: config.genetics.sos_base_mutation_prob,
                                    sos_max_induction_factor: config.genetics.sos_max_induction_factor,
                                    sos_lethal_fraction: config.genetics.sos_lethal_fraction,
                                };

                                let cell = *petri_cells.current.get(x, y);
                                let density = petri_cells.current.chamfer_density_5_7(x, y);

                                if cell.is_alive() {
                                    let updated = update_living_cell(cell, local_conc, density, true, &bio_params, 1.0, &mut rng);
                                    petri_cells.next.set(x, y, updated);
                                } else {
                                    let neighbors = petri_cells.current.get_living_neighbors(x, y);
                                    let colonized = try_colonize_empty_cell(
                                        x,
                                        y,
                                        density,
                                        &neighbors,
                                        &petri_drugs[closest_d].current,
                                        &bio_params,
                                        &mut rng,
                                    );
                                    petri_cells.next.set(x, y, colonized);
                                }
                            }
                        }
                        petri_cells.swap();
                        tick_invitro += 1;
                    }
                }

                let mut halo_diams = [0.0; NUM_DISKS];
                for d in 0..config.antibiotics.len().min(NUM_DISKS) {
                    halo_diams[d] = measure_halo_diameter(&petri_cells.current, disk_positions[d].x, disk_positions[d].y);
                }

                // --- Renderizado Fase 2 (In Vitro) ---
                fb.fill(0x0F172A);

                draw_rect(&mut fb, WIDTH, 0, 0, WIDTH, OSD_HEIGHT, 0x1E293B);
                draw_rect(&mut fb, WIDTH, 0, OSD_HEIGHT - 2, WIDTH, 2, 0x38BDF8);

                draw_text(&mut fb, WIDTH, 20, 10, "UNIVERSIDAD MAYOR DE SAN ANDRES - INFORMATICA | TESIS RAM", 0x38BDF8);
                draw_text(&mut fb, WIDTH, 20, 26, "FASE 2: RE-ANTIBIOGRAMA DIAGNOSTICO POST-BIOPSIA (KIRBY-BAUER)", 0x10B981);

                let status_str = format!(
                    "INCUBACION: {:04} min | MUESTRA EXTRAIDA AL MINUTO {:04} | CEPA: {}",
                    tick_invitro, biopsy_minute, config.name
                );
                draw_text(&mut fb, WIDTH, 20, 44, &status_str, 0x4ADE80);

                draw_text(
                    &mut fb,
                    WIDTH,
                    20,
                    62,
                    "[B] REGRESAR AL TEJIDO CLINICO  [ESPACIO] Pausa  [R] Reiniciar  [+/-] Velocidad",
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
                                fb[py * WIDTH + px] = 0x64748B;
                                continue;
                            }

                            let cell = petri_cells.current.get(px, py);

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
                                    total_c += *petri_drugs[d].current.get(px, py);
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

                // Discos de celulosa
                for (d, pos) in disk_positions.iter().enumerate().take(config.antibiotics.len().min(NUM_DISKS)) {
                    let ab = &config.antibiotics[d];
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

                // Panel diagnóstico de susceptibilidad comparativa
                let px = 530;
                let py = 120;
                let pw = 445;
                let ph = 445;

                draw_rect(&mut fb, WIDTH, px, py, pw, ph, 0x020617);
                draw_rect(&mut fb, WIDTH, px, py, pw, 1, 0x475569);
                draw_rect(&mut fb, WIDTH, px, py + ph, pw, 1, 0x475569);
                draw_rect(&mut fb, WIDTH, px, py, 1, ph, 0x475569);
                draw_rect(&mut fb, WIDTH, px + pw, py, 1, ph + 1, 0x475569);

                draw_text(&mut fb, WIDTH, px + 15, py + 14, "INFORME COMPARATIVO TRASLACIONAL", 0x38BDF8);
                draw_rect(&mut fb, WIDTH, px + 15, py + 30, pw - 30, 1, 0x334155);

                let bio_str = format!(
                    "BIOPSIA: g0:{:.0}% g1:{:.0}% g2:{:.0}% g3:{:.0}% g4:{:.0}%",
                    biopsy_distribution[0] * 100.0,
                    biopsy_distribution[1] * 100.0,
                    biopsy_distribution[2] * 100.0,
                    biopsy_distribution[3] * 100.0,
                    biopsy_distribution[4] * 100.0,
                );
                draw_text(&mut fb, WIDTH, px + 15, py + 40, &bio_str, 0xFBBF24);

                let col_y = py + 62;
                draw_text(&mut fb, WIDTH, px + 15, col_y, "FARMACO", 0x94A3B8);
                draw_text(&mut fb, WIDTH, px + 110, col_y, "CARGA", 0x94A3B8);
                draw_text(&mut fb, WIDTH, px + 190, col_y, "HALO POST", 0x94A3B8);
                draw_text(&mut fb, WIDTH, px + 290, col_y, "CORTE", 0x94A3B8);
                draw_text(&mut fb, WIDTH, px + 380, col_y, "CAT", 0x94A3B8);

                for (d, ab) in config.antibiotics.iter().enumerate().take(NUM_DISKS) {
                    let row_y = col_y + 24 + d * 36;
                    let diam = halo_diams[d];

                    let (cat, cat_color, cat_bg) = if diam >= ab.breakpoint_s {
                        (" S ", 0x020617, 0x22C55E)
                    } else if diam >= ab.breakpoint_r {
                        (" I ", 0x020617, 0xF59E0B)
                    } else {
                        (" R ", 0xFFFFFF, 0xEF4444)
                    };

                    draw_text(&mut fb, WIDTH, px + 15, row_y, &ab.code, 0xF8FAFC);
                    let load_str = format!("{:.0} ug", ab.load_ug);
                    draw_text(&mut fb, WIDTH, px + 110, row_y, &load_str, 0xCBD5E1);

                    let diam_str = format!("{:.1} mm", diam);
                    draw_text(&mut fb, WIDTH, px + 190, row_y, &diam_str, 0xF8FAFC);

                    let bp_str = format!(">={:.0}", ab.breakpoint_s);
                    draw_text(&mut fb, WIDTH, px + 290, row_y, &bp_str, 0x94A3B8);

                    draw_rect(&mut fb, WIDTH, px + 375, row_y - 2, 32, 14, cat_bg);
                    draw_text(&mut fb, WIDTH, px + 379, row_y + 1, cat, cat_color);
                }

                draw_rect(&mut fb, WIDTH, px + 15, py + 220, pw - 30, 1, 0x334155);
                draw_text(&mut fb, WIDTH, px + 15, py + 234, "EVALUACION DE LA HIPOTESIS 2 (MSW):", 0x38BDF8);

                let dominant_g = biopsy_distribution
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);

                if dominant_g > 0 {
                    draw_text(&mut fb, WIDTH, px + 15, py + 254, "-> Escape Adaptativo Confirmado in vitro.", 0xEF4444);
                    draw_text(&mut fb, WIDTH, px + 15, py + 272, "   Las bacterias seleccionadas por omision", 0xFCA5A5);
                    draw_text(&mut fb, WIDTH, px + 15, py + 290, "   muestran reduccion o perdida de halo.", 0xFCA5A5);
                    let res_note = format!("   Genotipo fijado en paciente: g{} (Resistente)", dominant_g);
                    draw_text(&mut fb, WIDTH, px + 15, py + 308, &res_note, 0xFBBF24);
                } else {
                    draw_text(&mut fb, WIDTH, px + 15, py + 254, "-> Cultivo Sensible (Adherencia Exitosa).", 0x4ADE80);
                    draw_text(&mut fb, WIDTH, px + 15, py + 272, "   El inóculo permanece libre de mutantes;", 0xBBF7D0);
                    draw_text(&mut fb, WIDTH, px + 15, py + 290, "   se conservan los halos de referencia S.", 0xBBF7D0);
                }

                draw_rect(&mut fb, WIDTH, px + 15, py + 340, pw - 30, 1, 0x334155);
                draw_text(&mut fb, WIDTH, px + 15, py + 354, "DATOS DE LA MUESTRA CLINICA:", 0x94A3B8);
                let total_bio_cells: usize = biopsy_counts.iter().sum();
                let b_str = format!("Celulas viables aisladas: {} parches", total_bio_cells);
                draw_text(&mut fb, WIDTH, px + 15, py + 372, &b_str, 0xCBD5E1);
                draw_text(&mut fb, WIDTH, px + 15, py + 390, "Metodo diagnostico: Difusion en disco Kirby-Bauer", 0x64748B);
                draw_text(&mut fb, WIDTH, px + 15, py + 408, "Estandar: Puntos de corte EUCAST v14.0", 0x64748B);
            }
        }

        window.update_with_buffer(&fb, WIDTH, HEIGHT).unwrap();
    }
}