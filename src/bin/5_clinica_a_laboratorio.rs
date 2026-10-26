use minifb::{Key, MouseButton, MouseMode, Window, WindowOptions};
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::fs::{create_dir_all, File};
use std::io::Write;

use tesis_ram::biology::rules::{try_colonize_empty_cell, update_living_cell, BiophysicalParams};
use tesis_ram::config::{load_organism_config, OrganismConfig};
use tesis_ram::core::cell::Cell;
use tesis_ram::core::grid::{DoubleBufferGrid, Grid};
use tesis_ram::physics::bateman::BatemanRegimen;
use tesis_ram::physics::diffusion::{step_diffusion_2d_biofilm, DiffusionParams};
use tesis_ram::visualizer::osd::{draw_rect, draw_text};

const WIDTH: usize = 1000;
const HEIGHT: usize = 600;

// Geometría Fase 1: Tejido vascularizado in vivo
const TISSUE_X: usize = 28;
const TISSUE_Y: usize = 120;
const TISSUE_W: usize = 448;
const TISSUE_H: usize = 448;

// Geometría Fase 2: Placa de Petri in vitro (Kirby-Bauer 90 mm)
const DISH_CENTER_X: usize = 280;
const DISH_CENTER_Y: usize = 345;
const DISH_RADIUS: usize = 205; // Radio 45 mm (1 px ≈ 0.2195 mm)
const DISK_RADIUS: usize = 14;  // Sensidisco 6 mm de diámetro
const NUM_DISKS: usize = 4;
const MM_PER_PIXEL: f64 = 90.0 / (2.0 * DISH_RADIUS as f64);
const MAX_INCUBATION_TICKS: usize = 2200; // Incubación estandarizada de equilibrio (EUCAST v16.0)

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

/// Exporta el estado clínico actual o el informe diagnóstico in vitro a un archivo CSV en data/interactive/
fn export_translational_snapshot(
    phase: SimulationPhase,
    tick_invivo: usize,
    tick_invitro: usize,
    config: &OrganismConfig,
    current_ab_idx: usize,
    initial_g: u8,
    use_rescue_panel: bool,
    biopsy_distribution: &[f64; 5],
    halo_diams: &[f64; 4],
) -> Result<(), Box<dyn std::error::Error>> {
    create_dir_all("data/interactive")?;
    let ab = &config.antibiotics[current_ab_idx.min(config.antibiotics.len() - 1)];
    let phase_str = if phase == SimulationPhase::ClinicalInVivo { "invivo" } else { "invitro" };
    let file_path = format!(
        "data/interactive/reporte_traslacional_{}_{}_{}_g{}.csv",
        config.species.replace(' ', "_"),
        ab.code,
        phase_str,
        initial_g
    );
    let mut writer = File::create(&file_path)?;

    writeln!(
        writer,
        "phase,tick,species,antibiotic,initial_genotype,rescue_panel,biopsy_g0,biopsy_g1,biopsy_g2,biopsy_g3,biopsy_g4,halo_ab1_mm,halo_ab2_mm,halo_ab3_mm,halo_ab4_mm"
    )?;

    writeln!(
        writer,
        "{},{},{},{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.2},{:.2},{:.2},{:.2}",
        phase_str,
        if phase == SimulationPhase::ClinicalInVivo { tick_invivo } else { tick_invitro },
        config.species,
        ab.code,
        initial_g,
        use_rescue_panel,
        biopsy_distribution[0],
        biopsy_distribution[1],
        biopsy_distribution[2],
        biopsy_distribution[3],
        biopsy_distribution[4],
        halo_diams[0],
        halo_diams[1],
        halo_diams[2],
        halo_diams[3]
    )?;

    println!("¡Reporte traslacional exportado exitosamente a {}!", file_path);
    Ok(())
}

fn get_capillary_network() -> Vec<CapillaryPoint> {
    let mut capillaries = Vec::with_capacity(64);
    let cols = 8;
    let rows = 8;
    let margin = 24.0;
    let step_x = (TISSUE_W as f64 - 2.0 * margin) / (cols as f64 - 1.0);
    let step_y = (TISSUE_H as f64 - 2.0 * margin) / (rows as f64 - 1.0);

    for r in 0..rows {
        for c in 0..cols {
            let cx = (TISSUE_X as f64 + margin + c as f64 * step_x).round() as usize;
            let cy = (TISSUE_Y as f64 + margin + r as f64 * step_y).round() as usize;
            capillaries.push(CapillaryPoint { x: cx, y: cy });
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
    initial_genotype: u8,
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
            let near_vessel = capillaries.iter().any(|c| {
                let dx = x as isize - c.x as isize;
                let dy = y as isize - c.y as isize;
                dx * dx + dy * dy <= 16
            });
            if !near_vessel && rng.random_bool(0.78) {
                cell_grid.current.set(x, y, Cell::new(initial_genotype, 0, 100));
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
    panel_offset: usize,
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

    // Inicializar reservorios en los 4 sensidiscos del panel activo (Primario o Rescate)
    for (d, pos) in disk_positions.iter().enumerate() {
        let ab_idx = panel_offset + d;
        if ab_idx < config.antibiotics.len() {
            let ab = &config.antibiotics[ab_idx];
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
    let plot_bottom = gy + gh - 15;
    let plot_top = gy + 28;
    plot_bottom - (norm * (plot_bottom - plot_top) as f64) as usize
}

fn draw_plot_segment(fb: &mut [u32], x: usize, y0: usize, y1: usize, color: u32) {
    let y_min = y0.min(y1);
    let y_max = y0.max(y1);
    for y in y_min..=y_max {
        if y < HEIGHT && x < WIDTH {
            fb[y * WIDTH + x] = color;
        }
    }
}

fn main() {
    let mut window = Window::new(
        "Tesis RAM - Ciclo Traslacional Bedside-to-Bench (Clinica -> Laboratorio)",
        WIDTH,
        HEIGHT,
        WindowOptions::default(),
    )
    .expect("Fallo al inicializar minifb");

    window.set_target_fps(60);

    let mut fb = vec![0u32; WIDTH * HEIGHT];
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(2026);

    let config_files = [
        "config/ecoli_atcc25922.toml",
        "config/klebsiella_blee.toml",
        "config/pseudomonas_pao1.toml",
    ];
    let mut current_cfg_idx = 0usize;
    let mut initial_genotype = 0u8;
    let mut use_rescue_panel = false;
    let mut mouse_was_down = false;

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
        diffusion_coeff: 0.85,
        clearance_rate: 0.00010,
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

    init_infection(&mut tissue_cells, &mut tissue_drug, &capillaries, initial_genotype, &mut rng);

    let mut phase = SimulationPhase::ClinicalInVivo;
    let mut tick_invivo = 0usize;
    let mut tick_invitro = 0usize;
    let mut is_paused = false;

    let mut pk_history = Vec::with_capacity(500);
    let mut ca_history = Vec::with_capacity(500);
    let mut time_above_mic_count = 0usize;
    let mut total_monitored_ticks = 0usize;

    let mut biopsy_counts = [0usize; 5];
    let mut biopsy_distribution = [1.0, 0.0, 0.0, 0.0, 0.0];
    let mut biopsy_minute = 0usize;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let mut reload_simulation = false;

        if window.is_key_pressed(Key::E, minifb::KeyRepeat::No) {
            // Evaluamos los diámetros de halo actuales según la fase para incluirlos en el reporte
            let mut current_halos = [0.0; 4];
            let num_active_disks = if use_rescue_panel && config.antibiotics.len() >= 8 { 4 } else { config.antibiotics.len().min(4) };
            for d in 0..num_active_disks {
                current_halos[d] = measure_halo_diameter(&petri_cells.current, disk_positions[d].x, disk_positions[d].y);
            }

            if let Err(e) = export_translational_snapshot(
                phase,
                tick_invivo,
                tick_invitro,
                &config,
                current_ab_idx,
                initial_genotype,
                use_rescue_panel,
                &biopsy_distribution,
                &current_halos,
            ) {
                println!("Error al exportar reporte traslacional: {}", e);
            }
        }

        if window.is_key_pressed(Key::Key1, minifb::KeyRepeat::No) {
            current_cfg_idx = 0;
            current_ab_idx = 0;
            use_rescue_panel = false;
            reload_simulation = true;
        }
        if window.is_key_pressed(Key::Key2, minifb::KeyRepeat::No) {
            current_cfg_idx = 1;
            current_ab_idx = 0;
            use_rescue_panel = false;
            reload_simulation = true;
        }
        if window.is_key_pressed(Key::Key3, minifb::KeyRepeat::No) {
            current_cfg_idx = 2;
            current_ab_idx = 0;
            use_rescue_panel = false;
            reload_simulation = true;
        }

        if window.is_key_pressed(Key::A, minifb::KeyRepeat::No) && phase == SimulationPhase::ClinicalInVivo {
            current_ab_idx = (current_ab_idx + 1) % config.antibiotics.len();
            reload_simulation = true;
        }

        if window.is_key_pressed(Key::G, minifb::KeyRepeat::No) && phase == SimulationPhase::ClinicalInVivo {
            initial_genotype = (initial_genotype + 1) % 5;
            reload_simulation = true;
        }

        if window.is_key_pressed(Key::T, minifb::KeyRepeat::No) && phase == SimulationPhase::ClinicalInVivo {
            current_reg_idx = (current_reg_idx + 1) % 3;
            reload_simulation = true;
        }

        if window.is_key_pressed(Key::O, minifb::KeyRepeat::No) && phase == SimulationPhase::ClinicalInVivo {
            pk.toggle_next_dose(tick_invivo);
        }

        // Clic izquierdo sobre la placa de Petri en Fase 2 para alternar entre Panel Primario y Panel Rescate
        let mouse_down = window.get_mouse_down(MouseButton::Left);
        if mouse_down && !mouse_was_down && phase == SimulationPhase::LaboratoryInVitro {
            if let Some((mx, my)) = window.get_mouse_pos(MouseMode::Clamp) {
                let dx = mx as isize - DISH_CENTER_X as isize;
                let dy = my as isize - DISH_CENTER_Y as isize;
                if dx * dx + dy * dy <= (DISH_RADIUS * DISH_RADIUS) as isize
                    && config.antibiotics.len() >= 8
                {
                    use_rescue_panel = !use_rescue_panel;
                    let panel_offset = if use_rescue_panel { 4 } else { 0 };
                    init_petri_dish_from_biopsy(
                        &mut petri_cells,
                        &mut petri_drugs,
                        &disk_positions,
                        &config,
                        panel_offset,
                        &biopsy_distribution,
                        &mut rng,
                    );
                    tick_invitro = 0;
                }
            }
        }
        mouse_was_down = mouse_down;

        // [B] Extraer biopsia clínica y sembrar placa de Petri (o regresar)
        if window.is_key_pressed(Key::B, minifb::KeyRepeat::No) {
            match phase {
                SimulationPhase::ClinicalInVivo => {
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

                        // Si el fármaco activo en clínica ya era del panel de rescate (>=4), iniciar en ese panel
                        use_rescue_panel = current_ab_idx >= 4 && config.antibiotics.len() >= 8;
                        let panel_offset = if use_rescue_panel { 4 } else { 0 };

                        init_petri_dish_from_biopsy(
                            &mut petri_cells,
                            &mut petri_drugs,
                            &disk_positions,
                            &config,
                            panel_offset,
                            &biopsy_distribution,
                            &mut rng,
                        );
                        tick_invitro = 0;
                        phase = SimulationPhase::LaboratoryInVitro;
                    }
                }
                SimulationPhase::LaboratoryInVitro => {
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
            init_infection(&mut tissue_cells, &mut tissue_drug, &capillaries, initial_genotype, &mut rng);
            pk_history.clear();
            ca_history.clear();
            time_above_mic_count = 0;
            total_monitored_ticks = 0;
            tick_invivo = 0;
            tick_invitro = 0;
            phase = SimulationPhase::ClinicalInVivo;
        }

        let ab_active = &config.antibiotics[current_ab_idx.min(config.antibiotics.len() - 1)];
        let active_g = initial_genotype.min(4) as usize;
        let mic_active = ab_active.mic_genotypes[active_g];
        let mpc_active = ab_active.mpc_genotypes[active_g];

        // --- Ejecución según la Fase Activa ---
        match phase {
            SimulationPhase::ClinicalInVivo => {
                // Parámetros leídos estrictamente desde el perfil TOML
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
                    let c_plasma = pk.concentration_at(tick_invivo);
                    if c_plasma >= mic_active {
                        time_above_mic_count += 1;
                    }
                    total_monitored_ticks += 1;

                    // 4 sub-pasos de perfusión microvascular por minuto
                    for _ in 0..4 {
                        for cap in &capillaries {
                            for dy in -2..=2isize {
                                for dx in -2..=2isize {
                                    if dx * dx + dy * dy <= 4 {
                                        let px = (cap.x as isize + dx) as usize;
                                        let py = (cap.y as isize + dy) as usize;
                                        if px < WIDTH && py < HEIGHT {
                                            tissue_drug.current.set(px, py, c_plasma);
                                        }
                                    }
                                }
                            }
                        }

                        step_diffusion_2d_biofilm(
                            &tissue_drug.current,
                            &mut tissue_drug.next,
                            &tissue_cells.current,
                            &tissue_diff_params,
                            ab_active.enzymatic_deg_rate,
                        );
                        tissue_drug.swap();
                    }

                    for cap in &capillaries {
                        for dy in -2..=2isize {
                            for dx in -2..=2isize {
                                if dx * dx + dy * dy <= 4 {
                                    let px = (cap.x as isize + dx) as usize;
                                    let py = (cap.y as isize + dy) as usize;
                                    if px < WIDTH && py < HEIGHT {
                                        tissue_drug.current.set(px, py, c_plasma);
                                    }
                                }
                            }
                        }
                    }

                    for y in TISSUE_Y..(TISSUE_Y + TISSUE_H) {
                        for x in TISSUE_X..(TISSUE_X + TISSUE_W) {
                            let is_vessel_core = capillaries.iter().any(|c| {
                                let dx = x as isize - c.x as isize;
                                let dy = y as isize - c.y as isize;
                                dx * dx + dy * dy <= 9
                            });

                            if is_vessel_core {
                                tissue_cells.next.set(x, y, Cell::EMPTY);
                                continue;
                            }

                            let local_c = *tissue_drug.current.get(x, y);
                            let cell = *tissue_cells.current.get(x, y);
                            let density = tissue_cells.current.chamfer_density_5_7(x, y);

                            if cell.is_alive() {
                                let updated = update_living_cell(cell, local_c, density, false, &bio_params, 1.0, &mut rng);
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
                    if pk_history.len() > 430 {
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
                fb.fill(0x0A1118);

                draw_text(&mut fb, WIDTH, 18, 12, "UNIVERSIDAD MAYOR DE SAN ANDRES - INFORMATICA : TESIS RAM", 0x38BDF8);
                draw_text(&mut fb, WIDTH, 18, 28, "FASE 1: TRATAMIENTO CLINICO IN VIVO (PRESIONAR [B] PARA BIOPSIA)", 0xFBBF24);

                let current_dose_num = ((tick_invivo / pk.tau_minutes) + 1).min(pk.total_doses);
                let status_str = format!(
                    "MIN:{:04} (Dia {:.1}) | {} (g{}) | {}: {} | DOSIS:{}/{} | ADH:{:.0}%",
                    tick_invivo,
                    tick_invivo as f64 / 1440.0,
                    config.species,
                    initial_genotype,
                    ab_active.code,
                    ab_active.name,
                    current_dose_num,
                    pk.total_doses,
                    pk.adherence_pct(tick_invivo)
                );
                draw_text(&mut fb, WIDTH, 18, 46, &status_str, 0x4ADE80);
                draw_text(
                    &mut fb,
                    WIDTH,
                    18,
                    64,
                    "[B] BIOPSIA  [O] Omitir/Tomar  [A] Farmaco (1..8)  [G] Inoculo  [T] Pauta  [1..3] Cepa",
                    0x94A3B8,
                );

                draw_rect(&mut fb, WIDTH, 0, 84, WIDTH, 2, 0x1E3A5F);

                // Calendario de pastillero
                draw_text(&mut fb, WIDTH, 18, 97, "CALENDARIO:", 0xE2E8F0);

                let box_w = 24usize;
                let box_h = 16usize;
                let start_box_x = 118usize;
                let next_idx_opt = pk.next_dose_index(tick_invivo);
                let current_elapsed_idx = tick_invivo / pk.tau_minutes;

                for d in 0..pk.total_doses {
                    let bx = start_box_x + d * (box_w + 6);
                    let by = 94;

                    let is_past = d <= current_elapsed_idx;
                    let is_next = Some(d) == next_idx_opt;
                    let is_taken = pk.doses_taken[d];

                    let bg_col = if is_past {
                        if is_taken { 0x166534 } else { 0x991B1B }
                    } else if is_next {
                        if is_taken { 0x0284C7 } else { 0xDC2626 }
                    } else {
                        0x1E293B
                    };

                    draw_rect(&mut fb, WIDTH, bx, by, box_w, box_h, bg_col);
                    let num_str = format!("{:>2}", d + 1);
                    let txt_col = if is_next || is_past { 0xFFFFFF } else { 0x94A3B8 };
                    draw_text(&mut fb, WIDTH, bx + 5, by + 4, &num_str, txt_col);
                }

                if let Some(n_idx) = next_idx_opt {
                    let next_minute = n_idx * pk.tau_minutes;
                    let mins_left = next_minute.saturating_sub(tick_invivo);
                    let (label, col) = if pk.doses_taken[n_idx] {
                        (format!("Prox: D{} en {}m [TOMAR]", n_idx + 1, mins_left), 0x38BDF8)
                    } else {
                        (format!("Prox: D{} en {}m [OMITIDA!]", n_idx + 1, mins_left), 0xEF4444)
                    };
                    draw_text(&mut fb, WIDTH, start_box_x + pk.total_doses * 30 + 12, 97, &label, col);
                }

                // Tejido vascular
                draw_rect(&mut fb, WIDTH, TISSUE_X - 1, TISSUE_Y - 1, TISSUE_W + 2, TISSUE_H + 2, 0x1E293B);

                for y in TISSUE_Y..(TISSUE_Y + TISSUE_H) {
                    for x in TISSUE_X..(TISSUE_X + TISSUE_W) {
                        let cell = *tissue_cells.current.get(x, y);
                        let conc = *tissue_drug.current.get(x, y);

                        let mut is_halo = false;
                        let mut is_core = false;
                        for cap in &capillaries {
                            let dx = x as isize - cap.x as isize;
                            let dy = y as isize - cap.y as isize;
                            let dist2 = dx * dx + dy * dy;
                            if dist2 <= 4 {
                                is_core = true;
                                break;
                            } else if dist2 <= 49 {
                                is_halo = true;
                            }
                        }

                        if is_core {
                            fb[y * WIDTH + x] = 0xF43F5E;
                        } else if is_halo {
                            fb[y * WIDTH + x] = 0x2A1526;
                        } else if cell.is_alive() {
                            fb[y * WIDTH + x] = match cell.genotype() {
                                0 => 0x22C55E,
                                1 => 0xEAB308,
                                2 => 0xF97316,
                                3 => 0xDC2626,
                                _ => 0xA855F7,
                            };
                        } else {
                            let norm = (conc / ab_active.c_max_plasma).clamp(0.0, 1.0);
                            let val = (norm * 80.0) as u32;
                            fb[y * WIDTH + x] = ((val / 2) << 16) | ((val / 3) << 8) | val;
                        }
                    }
                }
                draw_text(&mut fb, WIDTH, TISSUE_X + 10, TISSUE_Y + TISSUE_H - 18, "LECHO VASCULAR TISULAR (64 CAPILARES)", 0xFFFFFF);

                // Gráfica PK
                let gx = 515;
                let gy = 120;
                let gw = 466;
                let gh = 180;

                draw_rect(&mut fb, WIDTH, gx, gy, gw, gh, 0x050C14);
                draw_rect(&mut fb, WIDTH, gx, gy, gw, 1, 0x1E3A5F);
                draw_rect(&mut fb, WIDTH, gx, gy + gh - 1, gw, 1, 0x1E3A5F);
                draw_rect(&mut fb, WIDTH, gx, gy, 1, gh, 0x1E3A5F);
                draw_rect(&mut fb, WIDTH, gx + gw - 1, gy, 1, gh, 0x1E3A5F);

                let line_type = if current_ab_idx >= 4 { "RESCATE" } else { "1ra LINEA" };
                let title_pk = format!("PK C(t) [{} - {}] C_act: {:.2} ug/mL", ab_active.code, line_type, c_plasma_now);
                draw_text(&mut fb, WIDTH, gx + 15, gy + 12, &title_pk, 0x38BDF8);

                let y_mic = log_pk_to_y(mic_active, gy, gh);
                let y_mpc = log_pk_to_y(mpc_active, gy, gh);

                for sx in (gx + 1)..(gx + gw - 1) {
                    fb[y_mic * WIDTH + sx] = 0xF59E0B;
                    fb[y_mpc * WIDTH + sx] = 0xEF4444;
                }

                let mic_label = format!("CMI: {:.3}", mic_active);
                let mpc_label = format!("MPC: {:.3}", mpc_active);
                draw_text(&mut fb, WIDTH, gx + gw - 88, y_mic - 9, &mic_label, 0xF59E0B);
                draw_text(&mut fb, WIDTH, gx + gw - 88, y_mpc - 9, &mpc_label, 0xEF4444);

                if pk_history.len() > 1 {
                    for i in 1..pk_history.len() {
                        let px0 = gx + 15 + (i - 1);
                        let px1 = gx + 15 + i;
                        if px1 < gx + gw - 12 {
                            let py0 = log_pk_to_y(pk_history[i - 1], gy, gh);
                            let py1 = log_pk_to_y(pk_history[i], gy, gh);
                            draw_plot_segment(&mut fb, px0, py0, py1, 0x38BDF8);
                        }
                    }
                }

                // Carga bacteriana
                let dy_g = 320;
                let dh_g = 150;

                draw_rect(&mut fb, WIDTH, gx, dy_g, gw, dh_g, 0x050C14);
                draw_rect(&mut fb, WIDTH, gx, dy_g, gw, 1, 0x1E3A5F);
                draw_rect(&mut fb, WIDTH, gx, dy_g + dh_g - 1, gw, 1, 0x1E3A5F);
                draw_rect(&mut fb, WIDTH, gx, dy_g, 1, dh_g, 0x1E3A5F);
                draw_rect(&mut fb, WIDTH, gx + gw - 1, dy_g, 1, dh_g, 0x1E3A5F);

                draw_text(&mut fb, WIDTH, gx + 15, dy_g + 12, "CARGA BACTERIANA INTERSTICIAL", 0xE2E8F0);

                let max_pop = 160000.0;
                let plot_bot = dy_g + dh_g - 15;
                let plot_top = dy_g + 28;

                let evolved_mutants: usize = ((active_g + 1)..5).map(|idx| counts[idx]).sum();

                if ca_history.len() > 1 {
                    for i in 1..ca_history.len() {
                        let px0 = gx + 15 + (i - 1);
                        let px1 = gx + 15 + i;
                        if px1 < gx + gw - 12 {
                            let n0 = (ca_history[i - 1] as f64 / max_pop).clamp(0.0, 1.0);
                            let n1 = (ca_history[i] as f64 / max_pop).clamp(0.0, 1.0);
                            let py0 = plot_bot - (n0 * (plot_bot - plot_top) as f64) as usize;
                            let py1 = plot_bot - (n1 * (plot_bot - plot_top) as f64) as usize;

                            let pop_col = if evolved_mutants > counts[active_g] {
                                0xEF4444
                            } else {
                                0x4ADE80
                            };
                            draw_plot_segment(&mut fb, px0, py0, py1, pop_col);
                        }
                    }
                }

                // Diagnóstico y aviso para biopsia
                let info_y = 512;
                let pkpd_str = format!("INDICE PK/PD: %T>CMI = {:.1}% (Pauta: {})", pct_time_above_mic, regimens_labels[current_reg_idx]);
                draw_text(&mut fb, WIDTH, gx, info_y, &pkpd_str, 0x38BDF8);

                let pop_str = format!(
                    "POBLACION: {} celdas (g0:{} g1:{} g2:{} g3:{} g4:{})",
                    total_bacteria, counts[0], counts[1], counts[2], counts[3], counts[4]
                );
                draw_text(&mut fb, WIDTH, gx, info_y + 18, &pop_str, 0xE2E8F0);

                let biopsy_call = if evolved_mutants > counts[active_g] {
                    ("FRACASO RAM DETECTADO: PULSE [B] PARA EXTRAER BIOPSIA", 0xEF4444)
                } else if total_bacteria < 5000 {
                    ("INFECCION CONTROLADA: PULSE [B] PARA CULTIVO", 0x4ADE80)
                } else {
                    ("CURSO CLINICO: PULSE [B] PARA MUESTREAR BIOPSIA", 0xF59E0B)
                };
                draw_text(&mut fb, WIDTH, gx, info_y + 36, biopsy_call.0, biopsy_call.1);
            }

            SimulationPhase::LaboratoryInVitro => {
                let r_sq = (DISH_RADIUS * DISH_RADIUS) as isize;
                let disk_r_sq = (DISK_RADIUS * DISK_RADIUS) as isize;
                let panel_offset = if use_rescue_panel && config.antibiotics.len() >= 8 { 4 } else { 0 };
                let active_antibiotics = &config.antibiotics[panel_offset..(panel_offset + NUM_DISKS).min(config.antibiotics.len())];
                let num_ab = active_antibiotics.len();

                let mut bio_params_list = Vec::with_capacity(num_ab);
                for ab in active_antibiotics {
                    bio_params_list.push(BiophysicalParams {
                        base_division_prob: config.kinetics.base_division_prob,
                        fitness_cost_per_mutation: config.kinetics.fitness_cost_per_mutation,
                        maintenance_cost: config.kinetics.maintenance_cost,
                        division_cost: config.kinetics.division_cost,
                        min_resource_for_division: config.kinetics.min_resource_for_division,
                        hill_e_max: ab.hill_e_max,
                        hill_coefficient: ab.hill_coefficient,
                        mic_by_genotype: ab.mic_genotypes,
                        mpc_by_genotype: ab.mpc_genotypes,
                        sos_base_mutation_prob: 0.0, // Ensayo diagnóstico fenotípico puro de la biopsia
                        sos_max_induction_factor: 1.0,
                        sos_lethal_fraction: config.genetics.sos_lethal_fraction,
                    });
                }

                if !is_paused && tick_invitro < MAX_INCUBATION_TICKS {
                    // 1. Difusión radial isotrópica independiente para cada sensidisco del panel activo
                    for (d, ab) in active_antibiotics.iter().enumerate() {
                        let diff_params = DiffusionParams {
                            diffusion_coeff: ab.diffusion_coeff,
                            clearance_rate: ab.clearance_rate,
                            dx: 1.0,
                            dt: 1.0,
                        };

                        step_diffusion_2d_biofilm(
                            &petri_drugs[d].current,
                            &mut petri_drugs[d].next,
                            &petri_cells.current,
                            &diff_params,
                            ab.enzymatic_deg_rate,
                        );
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

                    // 2. Dinámica del césped sembrado con la biopsia evaluando los 4 discos simultáneamente
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
                                petri_cells.next.set(x, y, Cell::EMPTY);
                                continue;
                            }

                            let cell = *petri_cells.current.get(x, y);
                            let density = petri_cells.current.chamfer_density_5_7(x, y);

                            if cell.is_alive() {
                                let mut current_c = cell;
                                for d in 0..num_ab {
                                    let local_conc = *petri_drugs[d].current.get(x, y);
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
                                petri_cells.next.set(x, y, current_c);
                            } else {
                                let neighbors = petri_cells.current.get_living_neighbors(x, y);
                                let candidate = try_colonize_empty_cell(
                                    x,
                                    y,
                                    density,
                                    &neighbors,
                                    &petri_drugs[closest_d].current,
                                    &bio_params_list[closest_d],
                                    &mut rng,
                                );

                                if candidate.is_alive() {
                                    let g_idx = candidate.genotype() as usize;
                                    let mut inhibited_by_other = false;
                                    for d in 0..num_ab {
                                        let conc_d = *petri_drugs[d].current.get(x, y);
                                        if conc_d >= bio_params_list[d].mic_by_genotype[g_idx] {
                                            inhibited_by_other = true;
                                            break;
                                        }
                                    }
                                    petri_cells.next.set(
                                        x,
                                        y,
                                        if inhibited_by_other { Cell::EMPTY } else { candidate },
                                    );
                                } else {
                                    petri_cells.next.set(x, y, Cell::EMPTY);
                                }
                            }
                        }
                    }
                    petri_cells.swap();
                    tick_invitro += 1;
                }

                let mut halo_diams = [0.0; NUM_DISKS];
                for d in 0..num_ab {
                    halo_diams[d] = measure_halo_diameter(&petri_cells.current, disk_positions[d].x, disk_positions[d].y);
                }

                // --- Renderizado Fase 2 (In Vitro) ---
                fb.fill(0x0A1118);

                draw_text(&mut fb, WIDTH, 18, 12, "UNIVERSIDAD MAYOR DE SAN ANDRES - INFORMATICA : TESIS RAM", 0x38BDF8);
                draw_text(&mut fb, WIDTH, 18, 28, "FASE 2: RE-ANTIBIOGRAMA DIAGNOSTICO POST-BIOPSIA (KIRBY-BAUER)", 0x10B981);

                let panel_name = if use_rescue_panel { "RESCATE (2da)" } else { "PRIMARIO (1ra)" };
                let status_str = format!(
                    "INCUB:{:04}/{} | BIOPSIA MIN:{:04} | {} | PANEL:{}",
                    tick_invitro, MAX_INCUBATION_TICKS, biopsy_minute, config.species, panel_name
                );
                draw_text(&mut fb, WIDTH, 18, 46, &status_str, 0x4ADE80);
                draw_text(
                    &mut fb,
                    WIDTH,
                    18,
                    64,
                    "[B] REGRESAR A CLINICA   [CLICK EN PLACA] Panel Rescate   [ESPACIO] Pausa   [R] Reset",
                    0x94A3B8,
                );

                draw_rect(&mut fb, WIDTH, 0, 84, WIDTH, 2, 0x1E3A5F);

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

                            let cell = petri_cells.current.get(px, py);

                            if cell.is_alive() {
                                fb[py * WIDTH + px] = match cell.genotype() {
                                    0 => 0x22C55E,
                                    1 => 0xEAB308,
                                    2 => 0xF97316,
                                    3 => 0xDC2626,
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

                // Discos de celulosa con etiqueta de antibiótico
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

                // Panel diagnóstico comparativo traslacional
                let px = 515;
                let py = 120;
                let pw = 466;
                let ph = 448;

                draw_rect(&mut fb, WIDTH, px, py, pw, ph, 0x050C14);
                draw_rect(&mut fb, WIDTH, px, py, pw, 1, 0x1E3A5F);
                draw_rect(&mut fb, WIDTH, px, py + ph - 1, pw, 1, 0x1E3A5F);
                draw_rect(&mut fb, WIDTH, px, py, 1, ph, 0x1E3A5F);
                draw_rect(&mut fb, WIDTH, gx_end(px, pw), py, 1, ph, 0x1E3A5F);

                let header_title = if use_rescue_panel {
                    "INFORME TRASLACIONAL [PANEL RESCATE OMS]"
                } else {
                    "INFORME COMPARATIVO TRASLACIONAL"
                };
                draw_text(&mut fb, WIDTH, px + 15, py + 14, header_title, 0x38BDF8);
                draw_rect(&mut fb, WIDTH, px + 15, py + 30, pw - 30, 1, 0x1E3A5F);

                let bio_str = format!(
                    "BIOPSIA: g0:{:.0}% g1:{:.0}% g2:{:.0}% g3:{:.0}% g4:{:.0}%",
                    biopsy_distribution[0] * 100.0,
                    biopsy_distribution[1] * 100.0,
                    biopsy_distribution[2] * 100.0,
                    biopsy_distribution[3] * 100.0,
                    biopsy_distribution[4] * 100.0,
                );
                draw_text(&mut fb, WIDTH, px + 15, py + 38, &bio_str, 0xFBBF24);

                let col_y = py + 58;
                draw_text(&mut fb, WIDTH, px + 15, col_y, "FARMACO", 0x94A3B8);
                draw_text(&mut fb, WIDTH, px + 110, col_y, "CARGA", 0x94A3B8);
                draw_text(&mut fb, WIDTH, px + 195, col_y, "HALO POST", 0x94A3B8);
                draw_text(&mut fb, WIDTH, px + 305, col_y, "CORTE", 0x94A3B8);
                draw_text(&mut fb, WIDTH, px + 395, col_y, "CAT", 0x94A3B8);

                for (d, ab) in active_antibiotics.iter().enumerate() {
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
                    draw_text(&mut fb, WIDTH, px + 195, row_y, &diam_str, 0xF8FAFC);

                    let bp_str = format!(">={:.0}", ab.breakpoint_s);
                    draw_text(&mut fb, WIDTH, px + 305, row_y, &bp_str, 0x94A3B8);

                    draw_rect(&mut fb, WIDTH, px + 390, row_y - 2, 32, 14, cat_bg);
                    draw_text(&mut fb, WIDTH, px + 394, row_y + 1, cat, cat_color);
                }

                draw_rect(&mut fb, WIDTH, px + 15, py + 218, pw - 30, 1, 0x1E3A5F);
                draw_text(&mut fb, WIDTH, px + 15, py + 230, "EVALUACION DE LA HIPOTESIS 2 (MSW):", 0x38BDF8);

                let dominant_g = biopsy_distribution
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);

                if dominant_g > 0 {
                    draw_text(&mut fb, WIDTH, px + 15, py + 250, "-> Escape Adaptativo Confirmado in vitro.", 0xEF4444);
                    draw_text(&mut fb, WIDTH, px + 15, py + 268, "   Click en la placa para alternar entre", 0xFCA5A5);
                    draw_text(&mut fb, WIDTH, px + 15, py + 286, "   1ra Linea (R) y Panel de Rescate OMS.", 0xFCA5A5);
                    let res_note = format!("   Genotipo fijado en paciente: g{} (Resistente)", dominant_g);
                    draw_text(&mut fb, WIDTH, px + 15, py + 304, &res_note, 0xFBBF24);
                } else {
                    draw_text(&mut fb, WIDTH, px + 15, py + 250, "-> Cultivo Sensible (Adherencia Exitosa).", 0x4ADE80);
                    draw_text(&mut fb, WIDTH, px + 15, py + 268, "   El inoculo permanece libre de mutantes;", 0xBBF7D0);
                    draw_text(&mut fb, WIDTH, px + 15, py + 286, "   se conservan los halos de referencia S.", 0xBBF7D0);
                }

                draw_rect(&mut fb, WIDTH, px + 15, py + 332, pw - 30, 1, 0x1E3A5F);
                draw_text(&mut fb, WIDTH, px + 15, py + 346, "DATOS DE LA MUESTRA CLINICA:", 0x94A3B8);
                let total_bio_cells: usize = biopsy_counts.iter().sum();
                let b_str = format!("Celulas viables aisladas: {} parches", total_bio_cells);
                draw_text(&mut fb, WIDTH, px + 15, py + 364, &b_str, 0xCBD5E1);
                draw_text(&mut fb, WIDTH, px + 15, py + 382, "Metodo diagnostico: Difusion en disco Kirby-Bauer", 0x64748B);
                draw_text(&mut fb, WIDTH, px + 15, py + 400, "Estandar: Puntos de corte EUCAST v16.0", 0x64748B);
            }
        }

        window.update_with_buffer(&fb, WIDTH, HEIGHT).unwrap();
    }
}

#[inline(always)]
const fn gx_end(px: usize, pw: usize) -> usize {
    px + pw - 1
}