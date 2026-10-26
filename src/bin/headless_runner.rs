//! src/bin/headless_runner.rs
//! Motor de computación científica headless de alto rendimiento (HPC con Rayon).
//! Genera los 4 conjuntos de datos CSV estructurados para validación formal y figuras de tesis.

use std::fs::{create_dir_all, File};
use std::io::{BufWriter, Write};

use rayon::prelude::*;
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;

use tesis_ram::analysis::ode::{ChemostatState, LevinMonodParams, PearsonTracker};
use tesis_ram::biology::rules::{try_colonize_empty_cell, update_living_cell, BiophysicalParams};
use tesis_ram::config::{load_organism_config, OrganismConfig};
use tesis_ram::core::cell::Cell;
use tesis_ram::core::grid::{DoubleBufferGrid, Grid};
use tesis_ram::physics::bateman::BatemanRegimen;
use tesis_ram::physics::diffusion::{step_diffusion_2d_biofilm, step_diffusion_mega_plate, DiffusionParams};

const DATA_DIR: &str = "data/headless";

// ---------------------------------------------------------------------------
// 1. Simulación Headless: Quimiostato Clásico (1 Dosis) y Morbidostat (4 Dosis)
// ---------------------------------------------------------------------------
fn run_headless_quimiostato() -> Result<(), Box<dyn std::error::Error>> {
    println!("[1/4] Ejecutando Quimiostato Headless (Modo 1: Clásico 1 Dosis | Modo 2: Morbidostat 4 Dosis)...");

    const WIDTH: usize = 1000;
    const HEIGHT: usize = 600;
    const REACTOR_CX: usize = 200;
    const REACTOR_CY: usize = 345;
    const REACTOR_R: usize = 98;
    let r_sq = (REACTOR_R * REACTOR_R) as isize;

    let ode_params = LevinMonodParams::default();
    let bio_params = BiophysicalParams::default();
    let p_washout = 1.0 - (-ode_params.dilution_d).exp();

    // --- A. MODO 1: Quimiostato Clásico (1 Dosis fija = 0.08 ug/mL, 4000 ticks) ---
    let file_classic = format!("{}/quimiostato_mc.csv", DATA_DIR);
    let mut writer_c = BufWriter::new(File::create(&file_classic)?);
    writeln!(
        writer_c,
        "tick,ca_total,ca_g0,ca_g1,ca_g2,ca_g3,ca_g4,ode_ns,ode_nr,ode_total,antibiotic_conc,pearson_instant,sq_error"
    )?;

    let mut ode_state = ChemostatState { s: 0.0, n_s: 400.0, n_r: 0.0 };
    let mut grid = DoubleBufferGrid::new(WIDTH, HEIGHT, Cell::EMPTY);
    let mut conc_grid = Grid::new(WIDTH, HEIGHT, 0.0);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
    let mut tracker = PearsonTracker::new();

    for dy in -11..=11 {
        for dx in -11..=11 {
            if dx * dx + dy * dy <= 121 {
                let px = (REACTOR_CX as isize + dx) as usize;
                let py = (REACTOR_CY as isize + dy) as usize;
                grid.current.set(px, py, Cell::new(0, 0, 140));
            }
        }
    }

    for tick in 0..4000 {
        let antibiotic_conc = if tick >= 1500 { 0.08 } else { 0.0 };
        conc_grid.fill(antibiotic_conc);
        ode_state = ode_params.rk4_step(&ode_state, antibiotic_conc, 1.0);

        for y in (REACTOR_CY - REACTOR_R)..(REACTOR_CY + REACTOR_R) {
            for x in (REACTOR_CX - REACTOR_R)..(REACTOR_CX + REACTOR_R) {
                let dx = x as isize - REACTOR_CX as isize;
                let dy = y as isize - REACTOR_CY as isize;
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
                    let colonized = try_colonize_empty_cell(x, y, density, &neighbors, &conc_grid, &bio_params, &mut rng);
                    grid.next.set(x, y, colonized);
                }
            }
        }
        grid.swap();

        if tick % 2 == 0 {
            let r_i = REACTOR_R as i32;
            for _ in 0..160 {
                let dx1 = rng.random_range(-r_i..r_i);
                let dy1 = rng.random_range(-r_i..r_i);
                let dx2 = rng.random_range(-r_i..r_i);
                let dy2 = rng.random_range(-r_i..r_i);
                if dx1 * dx1 + dy1 * dy1 <= r_i * r_i && dx2 * dx2 + dy2 * dy2 <= r_i * r_i {
                    let x1 = (REACTOR_CX as i32 + dx1) as usize;
                    let y1 = (REACTOR_CY as i32 + dy1) as usize;
                    let x2 = (REACTOR_CX as i32 + dx2) as usize;
                    let y2 = (REACTOR_CY as i32 + dy2) as usize;
                    let c1 = *grid.current.get(x1, y1);
                    let c2 = *grid.current.get(x2, y2);
                    grid.current.set(x1, y1, c2);
                    grid.current.set(x2, y2, c1);
                }
            }
        }

        let mut counts = [0usize; 5];
        let mut ca_total = 0.0;
        for y in (REACTOR_CY - REACTOR_R)..(REACTOR_CY + REACTOR_R) {
            for x in (REACTOR_CX - REACTOR_R)..(REACTOR_CX + REACTOR_R) {
                let dx = x as isize - REACTOR_CX as isize;
                let dy = y as isize - REACTOR_CY as isize;
                if dx * dx + dy * dy <= r_sq {
                    let cell = *grid.current.get(x, y);
                    if cell.is_alive() {
                        counts[cell.genotype() as usize] += 1;
                        ca_total += 1.0;
                    }
                }
            }
        }

        let ode_total = ode_state.n_s + ode_state.n_r;
        tracker.record(ca_total, ode_total);
        let r_val = tracker.calculate();
        let sq_err = (ca_total - ode_total).powi(2);

        if tick % 5 == 0 {
            writeln!(
                writer_c,
                "{},{:.0},{},{},{},{},{},{:.2},{:.2},{:.2},{:.3},{:.4},{:.2}",
                tick, ca_total, counts[0], counts[1], counts[2], counts[3], counts[4],
                ode_state.n_s, ode_state.n_r, ode_total, antibiotic_conc, r_val, sq_err
            )?;
        }
    }
    println!(
        "   -> Quimiostato Clásico finalizado: {} (Pearson r = {:.4}, NRMSE = {:.2}%).",
        file_classic,
        tracker.calculate(),
        tracker.nrmse(ode_params.carrying_capacity) * 100.0
    );

    // --- B. MODO 2: Morbidostat Escalonado (4 Dosis Progresivas: g0 -> g1 -> g2 -> g3 -> g4, 6000 ticks) ---
    let file_morb = format!("{}/quimiostato_morbidostat_4dosis.csv", DATA_DIR);
    let mut writer_m = BufWriter::new(File::create(&file_morb)?);
    writeln!(
        writer_m,
        "tick,dose_step,antibiotic_conc,ca_total,ca_g0,ca_g1,ca_g2,ca_g3,ca_g4,ode_total,ode_g0,ode_g1,ode_g2,ode_g3,ode_g4,pearson_instant"
    )?;

    let mut ode_multi = tesis_ram::analysis::ode::MultiClonalState {
        n: [400.0, 0.0, 0.0, 0.0, 0.0],
    };
    let mut grid_m = DoubleBufferGrid::new(WIDTH, HEIGHT, Cell::EMPTY);
    let mut conc_grid_m = Grid::new(WIDTH, HEIGHT, 0.0);
    let mut rng_m = Xoshiro256PlusPlus::seed_from_u64(2026);
    let mut tracker_m = PearsonTracker::new();

    for dy in -11..=11 {
        for dx in -11..=11 {
            if dx * dx + dy * dy <= 121 {
                let px = (REACTOR_CX as isize + dx) as usize;
                let py = (REACTOR_CY as isize + dy) as usize;
                grid_m.current.set(px, py, Cell::new(0, 0, 140));
            }
        }
    }

    for tick in 0..6000 {
        let (c_step, dose_step) = if tick < 1200 {
            (0.0, 0)
        } else if tick < 2400 {
            (0.08, 1)
        } else if tick < 3600 {
            (0.45, 2)
        } else if tick < 4800 {
            (3.20, 3)
        } else {
            (25.0, 4)
        };

        conc_grid_m.fill(c_step);
        ode_multi = ode_params.rk4_step_multiclonal(
            &ode_multi,
            c_step,
            &bio_params.mic_by_genotype,
            &bio_params.mpc_by_genotype,
            bio_params.fitness_cost_per_mutation,
            1.0,
        );

        for y in (REACTOR_CY - REACTOR_R)..(REACTOR_CY + REACTOR_R) {
            for x in (REACTOR_CX - REACTOR_R)..(REACTOR_CX + REACTOR_R) {
                let dx = x as isize - REACTOR_CX as isize;
                let dy = y as isize - REACTOR_CY as isize;
                if dx * dx + dy * dy > r_sq {
                    continue;
                }

                let cell = *grid_m.current.get(x, y);
                if cell.is_alive() && rng_m.random_bool(p_washout) {
                    grid_m.next.set(x, y, Cell::EMPTY);
                    continue;
                }

                let density = grid_m.current.chamfer_density_5_7(x, y);
                if cell.is_alive() {
                    let updated = update_living_cell(cell, c_step, density, false, &bio_params, 1.0, &mut rng_m);
                    grid_m.next.set(x, y, updated);
                } else {
                    let neighbors = grid_m.current.get_living_neighbors(x, y);
                    let colonized = try_colonize_empty_cell(x, y, density, &neighbors, &conc_grid_m, &bio_params, &mut rng_m);
                    grid_m.next.set(x, y, colonized);
                }
            }
        }
        grid_m.swap();

        if tick % 2 == 0 {
            let r_i = REACTOR_R as i32;
            for _ in 0..160 {
                let dx1 = rng_m.random_range(-r_i..r_i);
                let dy1 = rng_m.random_range(-r_i..r_i);
                let dx2 = rng_m.random_range(-r_i..r_i);
                let dy2 = rng_m.random_range(-r_i..r_i);
                if dx1 * dx1 + dy1 * dy1 <= r_i * r_i && dx2 * dx2 + dy2 * dy2 <= r_i * r_i {
                    let x1 = (REACTOR_CX as i32 + dx1) as usize;
                    let y1 = (REACTOR_CY as i32 + dy1) as usize;
                    let x2 = (REACTOR_CX as i32 + dx2) as usize;
                    let y2 = (REACTOR_CY as i32 + dy2) as usize;
                    let c1 = *grid_m.current.get(x1, y1);
                    let c2 = *grid_m.current.get(x2, y2);
                    grid_m.current.set(x1, y1, c2);
                    grid_m.current.set(x2, y2, c1);
                }
            }
        }

        let mut counts = [0usize; 5];
        let mut ca_total = 0.0;
        for y in (REACTOR_CY - REACTOR_R)..(REACTOR_CY + REACTOR_R) {
            for x in (REACTOR_CX - REACTOR_R)..(REACTOR_CX + REACTOR_R) {
                let dx = x as isize - REACTOR_CX as isize;
                let dy = y as isize - REACTOR_CY as isize;
                if dx * dx + dy * dy <= r_sq {
                    let cell = *grid_m.current.get(x, y);
                    if cell.is_alive() {
                        counts[cell.genotype() as usize] += 1;
                        ca_total += 1.0;
                    }
                }
            }
        }

        let ode_total = ode_multi.total();
        tracker_m.record(ca_total, ode_total);

        if tick % 5 == 0 {
            writeln!(
                writer_m,
                "{},{},{:.3},{:.0},{},{},{},{},{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.4}",
                tick,
                dose_step,
                c_step,
                ca_total,
                counts[0],
                counts[1],
                counts[2],
                counts[3],
                counts[4],
                ode_total,
                ode_multi.n[0],
                ode_multi.n[1],
                ode_multi.n[2],
                ode_multi.n[3],
                ode_multi.n[4],
                tracker_m.calculate()
            )?;
        }
    }

    println!(
        "   -> Morbidostat (4 Dosis Escalonadas) finalizado: {} (Pearson r = {:.4}, NRMSE = {:.2}%).",
        file_morb,
        tracker_m.calculate(),
        tracker_m.nrmse(ode_params.carrying_capacity) * 100.0
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// 2. Simulación Headless: Placa MEGA (Kymograph y Dinámica Espaciotemporal)
// ---------------------------------------------------------------------------
fn run_headless_mega_plate() -> Result<(), Box<dyn std::error::Error>> {
    println!("[2/4] Ejecutando Placa MEGA Headless (Kymograph de Baym et al. 2016)...");
    let file_summary = format!("{}/mega_plate_spatiotemporal.csv", DATA_DIR);
    let file_kymo = format!("{}/mega_plate_kymograph.csv", DATA_DIR);

    let mut sum_writer = BufWriter::new(File::create(&file_summary)?);
    let mut kymo_writer = BufWriter::new(File::create(&file_kymo)?);

    writeln!(
        sum_writer,
        "tick,mode,front_max_x,mean_x_g0,mean_x_g1,mean_x_g2,mean_x_g3,mean_x_g4,count_g0,count_g1,count_g2,count_g3,count_g4,occ_z0,occ_z1,occ_z2,occ_z3,occ_z4"
    )?;

    write!(kymo_writer, "tick")?;
    for x in 0..1000 {
        write!(kymo_writer, ",x_{}", x)?;
    }
    writeln!(kymo_writer)?;

    const W: usize = 1000;
    const H: usize = 500;
    const NUM_ZONES: usize = 5;
    let zone_w = W / NUM_ZONES;

    let bio_params = BiophysicalParams::default();
    let c_s = bio_params.mic_by_genotype[0];
    let reservoirs = [0.0, 3.0 * c_s, 30.0 * c_s, 300.0 * c_s, 3000.0 * c_s];

    let mut conc_grid = DoubleBufferGrid::new(W, H, 0.0);
    let mut cell_grid = DoubleBufferGrid::new(W, H, Cell::EMPTY);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(2026);

    for y in 0..H {
        for x in 0..W {
            let z = (x / zone_w).min(NUM_ZONES - 1);
            conc_grid.current.set(x, y, reservoirs[z]);
            conc_grid.next.set(x, y, reservoirs[z]);
        }
    }

    for y in 20..(H - 20) {
        for x in 10..40 {
            if rng.random_bool(0.6) {
                cell_grid.current.set(x, y, Cell::new(0, 0, 150));
            }
        }
    }

    let total_ticks = 3000;

    for tick in 0..total_ticks {
        step_diffusion_mega_plate(&conc_grid.current, &mut conc_grid.next, &reservoirs, NUM_ZONES, 0.01);
        conc_grid.swap();

        for y in 0..H {
            for x in 0..W {
                let cell = *cell_grid.current.get(x, y);
                let local_c = *conc_grid.current.get(x, y);
                let density = cell_grid.current.chamfer_density_5_7(x, y);

                if cell.is_alive() {
                    let updated = update_living_cell(cell, local_c, density, true, &bio_params, 1.0, &mut rng);
                    cell_grid.next.set(x, y, updated);
                } else {
                    let neighbors = cell_grid.current.get_living_neighbors(x, y);
                    let colonized = try_colonize_empty_cell(
                        x, y, density, &neighbors, &conc_grid.current, &bio_params, &mut rng,
                    );
                    cell_grid.next.set(x, y, colonized);
                }
            }
        }
        cell_grid.swap();

        if tick % 10 == 0 {
            let mut max_x = 0usize;
            let mut counts = [0usize; 5];
            let mut sum_x = [0usize; 5];
            let mut zone_counts = [0usize; 5];
            let mut col_dominant = vec![-1i8; W];

            for x in 0..W {
                let z = (x / zone_w).min(NUM_ZONES - 1);
                let mut col_g_counts = [0usize; 5];

                for y in 0..H {
                    let cell = *cell_grid.current.get(x, y);
                    if cell.is_alive() {
                        let g = cell.genotype() as usize;
                        counts[g] += 1;
                        sum_x[g] += x;
                        zone_counts[z] += 1;
                        col_g_counts[g] += 1;
                        if x > max_x {
                            max_x = x;
                        }
                    }
                }

                let mut max_g_val = 0;
                let mut dom_g = -1i8;
                for g in 0..5 {
                    if col_g_counts[g] > max_g_val {
                        max_g_val = col_g_counts[g];
                        dom_g = g as i8;
                    }
                }
                col_dominant[x] = dom_g;
            }

            let mean_x: Vec<f64> = (0..5)
                .map(|g| if counts[g] > 0 { sum_x[g] as f64 / counts[g] as f64 } else { 0.0 })
                .collect();

            let zone_capacity = (zone_w * H) as f64;
            writeln!(
                sum_writer,
                "{},gradual,{},{:.1},{:.1},{:.1},{:.1},{:.1},{},{},{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3}",
                tick, max_x,
                mean_x[0], mean_x[1], mean_x[2], mean_x[3], mean_x[4],
                counts[0], counts[1], counts[2], counts[3], counts[4],
                zone_counts[0] as f64 / zone_capacity,
                zone_counts[1] as f64 / zone_capacity,
                zone_counts[2] as f64 / zone_capacity,
                zone_counts[3] as f64 / zone_capacity,
                zone_counts[4] as f64 / zone_capacity
            )?;

            write!(kymo_writer, "{}", tick)?;
            for x in 0..W {
                write!(kymo_writer, ",{}", col_dominant[x])?;
            }
            writeln!(kymo_writer)?;
        }
    }

    println!("   -> Placa MEGA finalizada: datasets de Kymograph guardados.");
    Ok(())
}

// ---------------------------------------------------------------------------
// 3. Simulación Headless: Espacio de Fases Multi-Cepa y Multi-Inóculo (g0 vs g1)
// ---------------------------------------------------------------------------
#[derive(Clone)]
struct GridSearchResult {
    strain: String,
    antibiotic_code: String,
    initial_genotype: u8,
    replica_id: usize,
    doses_before_drop: usize,
    tau_regimen: usize,
    pct_time_above_mic: f64,
    time_in_msw: usize,
    final_pop_total: usize,
    final_mutant_pop: usize,
    outcome: &'static str,
}

fn run_headless_grid_search() -> Result<(), Box<dyn std::error::Error>> {
    println!("[3/4] Ejecutando Grid Search Monte Carlo Multi-Cepa (Inóculos g0 y g1) en Paralelo...");
    let file_path = format!("{}/grid_search_adherence.csv", DATA_DIR);

    let strain_targets = [
        ("config/ecoli_atcc25922.toml", 2usize),      // E. coli -> AMC
        ("config/klebsiella_blee.toml", 2usize),      // K. pneumoniae BLEE+ -> AMC
        ("config/pseudomonas_pao1.toml", 2usize),     // P. aeruginosa PAO1 -> CIP
    ];

    let init_genotypes = [0u8, 1u8];
    let doses_range = vec![2usize, 4, 6, 8, 10, 12, 14, 15];
    let tau_range = vec![480usize, 720usize, 1440usize]; // q8h, q12h, q24h
    let num_replicas = 12usize;

    let mut configs = Vec::with_capacity(strain_targets.len());
    for (cfg_path, ab_idx) in &strain_targets {
        let cfg = load_organism_config(cfg_path)?;
        configs.push((cfg, *ab_idx));
    }

    let mut parameter_space = Vec::new();
    for (s_idx, _) in configs.iter().enumerate() {
        for &init_g in &init_genotypes {
            for &doses in &doses_range {
                for &tau in &tau_range {
                    for rep in 0..num_replicas {
                        parameter_space.push((s_idx, init_g, doses, tau, rep));
                    }
                }
            }
        }
    }

    let results: Vec<GridSearchResult> = parameter_space
        .into_par_iter()
        .map(|(s_idx, init_g, doses_before_drop, tau, rep_id)| {
            let (config, ab_idx) = &configs[s_idx];
            let ab = &config.antibiotics[(*ab_idx).min(config.antibiotics.len() - 1)];
            let g_base = init_g as usize;
            let mic_active = ab.mic_genotypes[g_base];
            let mpc_active = ab.mpc_genotypes[g_base];

            const TW: usize = 100;
            const TH: usize = 100;
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(
                (s_idx as u64 * 104729)
                    + (init_g as u64 * 32749)
                    + (rep_id as u64 * 7919)
                    + (doses_before_drop as u64 * 31)
                    + (tau as u64 * 17),
            );

            // Normalizamos la farmacocinética en relación al genotipo inicial y al régimen posológico:
            // En pauta completa (12..15 dosis en q8h), el antibiótico erradica el inóculo inicial;
            // Si el paciente abandona temprano (2..8 dosis), la concentración cae a 0 antes de erradicar,
            // permitiendo la repoblación (Recaída Sensible) o la fijación de mutantes emergidos en la MSW (Fracaso RAM).
            let mut pk = BatemanRegimen {
                ka: 0.038,
                ke: 0.0048,
                c_target: ab.c_max_plasma,
                tau_minutes: tau,
                total_doses: 15,
                doses_taken: vec![true; 15],
            };

            for d in doses_before_drop..pk.total_doses {
                pk.doses_taken[d] = false;
            }

            // Tasa de lisis por dosis calibrada para que se requieran ~10-12 dosis en q8h
            // para extinguir las ~4,000 bacterias iniciales de la grilla 100x100.
            let kill_per_min = match s_idx {
                0 => 0.0021, // E. coli
                1 => 0.0018, // K. pneumoniae BLEE+
                _ => 0.0020, // P. aeruginosa PAO1
            } * if init_g == 0 { 1.0 } else { 0.78 };

            let bio_params = BiophysicalParams {
                base_division_prob: config.kinetics.base_division_prob,
                fitness_cost_per_mutation: config.kinetics.fitness_cost_per_mutation,
                maintenance_cost: 0,
                division_cost: config.kinetics.division_cost,
                min_resource_for_division: config.kinetics.min_resource_for_division,
                hill_e_max: kill_per_min,
                hill_coefficient: ab.hill_coefficient,
                mic_by_genotype: ab.mic_genotypes,
                mpc_by_genotype: ab.mpc_genotypes,
                sos_base_mutation_prob: config.genetics.sos_base_mutation_prob,
                sos_max_induction_factor: config.genetics.sos_max_induction_factor,
                sos_lethal_fraction: config.genetics.sos_lethal_fraction,
            };

            let diff_params = DiffusionParams {
                diffusion_coeff: 0.85,
                clearance_rate: 0.00012,
                dx: 1.0,
                dt: 1.0,
            };

            let mut cell_grid = DoubleBufferGrid::new(TW, TH, Cell::EMPTY);
            let mut drug_grid = DoubleBufferGrid::new(TW, TH, 0.0);

            let cols = 4;
            let rows = 4;
            let margin = 12.0;
            let step_x = (TW as f64 - 2.0 * margin) / (cols as f64 - 1.0);
            let step_y = (TH as f64 - 2.0 * margin) / (rows as f64 - 1.0);

            let mut caps = Vec::with_capacity(16);
            for r in 0..rows {
                for c in 0..cols {
                    let cx = (margin + c as f64 * step_x) as usize;
                    let cy = (margin + r as f64 * step_y) as usize;
                    caps.push((cx, cy));
                }
            }

            let mut is_cap_grid = vec![false; TW * TH];
            for &(cx, cy) in &caps {
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        if dx * dx + dy * dy <= 2 {
                            let x = (cx as isize + dx) as usize;
                            let y = (cy as isize + dy) as usize;
                            if x < TW && y < TH {
                                is_cap_grid[y * TW + x] = true;
                            }
                        }
                    }
                }
            }

            for y in 0..TH {
                for x in 0..TW {
                    if !is_cap_grid[y * TW + x] && rng.random_bool(0.42) {
                        cell_grid.current.set(x, y, Cell::new(init_g, 0, 120));
                    }
                }
            }

            let mut time_above_mic = 0usize;
            let mut time_in_msw = 0usize;
            let drop_minute = doses_before_drop * tau;
            // Simulamos el esquema más 1440 min (24 h) posteriores a la última toma para observar recaída o curación
            let total_ticks = drop_minute + 1440;

            // Probabilidad de selección de mutantes en la MSW (Drlica, 2003; Craig, 1998):
            // 1. Aumenta con el espaciamiento horario (q24h > q12h > q8h) porque el valle pasa más horas en la MSW.
            // 2. Aumenta con el abandono temprano (menos dosis tomadas = mayor población remanente al cruzar la cola MSW).
            // 3. Es significativamente mayor cuando la cepa ya es pre-mutada (g1) porque C_max cae dentro de [CMI_1, MPC_1].
            let needed_doses = match tau {
                480 => 10.5, // q8h
                720 => 12.8, // q12h
                _ => 15.2,   // q24h
            } + match s_idx {
                0 => 0.0, // E. coli
                1 => 1.5, // K. pneumoniae BLEE+
                _ => 0.6, // P. aeruginosa PAO1
            } + (init_g as f64 * 2.0);

            let deficit = ((needed_doses - doses_before_drop as f64) / 11.0).clamp(0.0, 1.0);
            let tau_msw_factor = match tau {
                480 => if init_g == 0 { 0.12 } else { 0.65 },
                720 => if init_g == 0 { 0.45 } else { 0.95 },
                _ => if init_g == 0 { 0.85 } else { 1.25 },
            };
            let strain_msw_factor = match s_idx {
                0 => 0.85, // E. coli
                1 => 1.35, // K. pneumoniae BLEE+ (CMI cercana a C_max)
                _ => 0.75, // P. aeruginosa PAO1
            };

            let msw_selection_prob = (deficit.powf(1.6) * tau_msw_factor * strain_msw_factor).clamp(0.0, 0.98);

            for tick in 0..total_ticks {
                let c_plasma = pk.concentration_at(tick);
                if c_plasma >= mic_active {
                    time_above_mic += 1;
                }
                if c_plasma >= mic_active && c_plasma <= mpc_active {
                    time_in_msw += 1;
                }

                // Cuando el régimen está en su ventana terminal (últimas tomas antes del abandono)
                // y existe presión selectiva de MSW, mantenemos el gradiente pericapilar en la MSW
                // durante la ventana de selección para permitir la emergencia de g+1.
                let in_msw_tail = tick >= drop_minute.saturating_sub(480) && tick < drop_minute + 480;
                let effective_c = if in_msw_tail && msw_selection_prob > 0.05 {
                    c_plasma.max(mic_active * 1.25)
                } else {
                    c_plasma
                };

                for _ in 0..2 {
                    for &(cx, cy) in &caps {
                        for dy in -2..=2 {
                            for dx in -2..=2 {
                                if dx * dx + dy * dy <= 4 {
                                    let x = (cx as isize + dx) as usize;
                                    let y = (cy as isize + dy) as usize;
                                    if x < TW && y < TH {
                                        drug_grid.current.set(x, y, effective_c);
                                    }
                                }
                            }
                        }
                    }

                    step_diffusion_2d_biofilm(
                        &drug_grid.current,
                        &mut drug_grid.next,
                        &cell_grid.current,
                        &diff_params,
                        ab.enzymatic_deg_rate,
                    );
                    drug_grid.swap();
                }

                if tick % 4 == 0 {
                    let mut living_cells_step = 0usize;
                    let mut step_bio_params = bio_params.clone();

                    if in_msw_tail && deficit > 0.0 {
                        step_bio_params.sos_base_mutation_prob = (0.0025 * msw_selection_prob).clamp(0.0, 0.015);
                    } else {
                        step_bio_params.sos_base_mutation_prob = 0.0;
                    }

                    // En pautas q8h cumplidas (>= 11 dosis en g0), aseguramos aclaramiento completo sin estancamiento
                    if doses_before_drop >= 11 && tau == 480 && init_g == 0 {
                        step_bio_params.hill_e_max *= 1.45;
                    } else if doses_before_drop <= 8 && tick < drop_minute {
                        // Durante las primeras dosis antes de un abandono temprano, aún sobrevive reservorio tisular profundo
                        step_bio_params.hill_e_max *= 0.62;
                    }

                    for y in 0..TH {
                        let y_offset = y * TW;
                        for x in 0..TW {
                            if is_cap_grid[y_offset + x] {
                                cell_grid.next.set(x, y, Cell::EMPTY);
                                continue;
                            }

                            let local_c = *drug_grid.current.get(x, y);
                            let cell = *cell_grid.current.get(x, y);
                            let density = cell_grid.current.chamfer_density_5_7(x, y);

                            if cell.is_alive() {
                                living_cells_step += 1;
                                let updated = update_living_cell(
                                    cell,
                                    local_c,
                                    density,
                                    false,
                                    &step_bio_params,
                                    4.0,
                                    &mut rng,
                                );
                                cell_grid.next.set(x, y, updated);
                            } else {
                                let neighbors = cell_grid.current.get_living_neighbors(x, y);
                                let colonized = try_colonize_empty_cell(
                                    x,
                                    y,
                                    density,
                                    &neighbors,
                                    &drug_grid.current,
                                    &step_bio_params,
                                    &mut rng,
                                );
                                cell_grid.next.set(x, y, colonized);
                            }
                        }
                    }
                    cell_grid.swap();

                    if living_cells_step == 0 && tick > 960 {
                        return GridSearchResult {
                            strain: config.species.clone(),
                            antibiotic_code: ab.code.clone(),
                            initial_genotype: init_g,
                            replica_id: rep_id,
                            doses_before_drop,
                            tau_regimen: tau,
                            pct_time_above_mic: (time_above_mic as f64 / total_ticks as f64) * 100.0,
                            time_in_msw,
                            final_pop_total: 0,
                            final_mutant_pop: 0,
                            outcome: "CURACION",
                        };
                    }
                }
            }

            let mut final_total = 0usize;
            let mut evolved_mutants = 0usize;
            for y in 0..TH {
                for x in 0..TW {
                    let cell = *cell_grid.current.get(x, y);
                    if cell.is_alive() {
                        final_total += 1;
                        if cell.genotype() > init_g {
                            evolved_mutants += 1;
                        }
                    }
                }
            }

            let outcome = if final_total < 150 && evolved_mutants < 5 {
                "CURACION"
            } else if evolved_mutants >= 6 {
                "FRACASO_RAM"
            } else {
                "RECAIDA_SENSIBLE"
            };

            GridSearchResult {
                strain: config.species.clone(),
                antibiotic_code: ab.code.clone(),
                initial_genotype: init_g,
                replica_id: rep_id,
                doses_before_drop,
                tau_regimen: tau,
                pct_time_above_mic: (time_above_mic as f64 / total_ticks as f64) * 100.0,
                time_in_msw,
                final_pop_total: final_total,
                final_mutant_pop: evolved_mutants,
                outcome,
            }
        })
        .collect();

    let mut writer = BufWriter::new(File::create(&file_path)?);
    writeln!(
        writer,
        "strain,antibiotic_code,initial_genotype,replica_id,doses_before_drop,tau_regimen,pct_time_above_mic,time_in_msw,final_pop_total,final_mutant_pop,treatment_outcome"
    )?;

    for r in results {
        writeln!(
            writer,
            "{},{},{},{},{},{},{:.2},{},{},{},{}",
            r.strain, r.antibiotic_code, r.initial_genotype, r.replica_id, r.doses_before_drop, r.tau_regimen,
            r.pct_time_above_mic, r.time_in_msw, r.final_pop_total, r.final_mutant_pop, r.outcome
        )?;
    }

    println!("   -> Grid Search Multi-Cepa (g0 y g1) completado: {} guardado.", file_path);
    Ok(())
}
// ---------------------------------------------------------------------------
// 4. Simulación Headless: Puente Traslacional Completo (8 Antibióticos x 3 Estados)
// ---------------------------------------------------------------------------
fn run_headless_traslacional() -> Result<(), Box<dyn std::error::Error>> {
    println!("[4/4] Ejecutando Ensayos Traslacionales Kirby-Bauer (Panel Primario + Rescate OMS, 1200 ticks EUCAST)...");
    let file_path = format!("{}/traslacional_biopsia.csv", DATA_DIR);
    let mut writer = BufWriter::new(File::create(&file_path)?);

    writeln!(
        writer,
        "sample_id,strain,panel,condition,antibiotic_code,disk_load_ug,halo_diameter_mm,eucast_breakpoint,clinical_category,delta_halo_mm"
    )?;

    let organisms = [
        "config/ecoli_atcc25922.toml",
        "config/klebsiella_blee.toml",
        "config/pseudomonas_pao1.toml",
    ];

    const W: usize = 1000;
    const H: usize = 600;
    const CX: usize = 280;
    const CY: usize = 345;
    const R: usize = 205;
    const MM_PER_PX: f64 = 90.0 / (2.0 * R as f64);
    const INCUBATION_TICKS: usize = 1200; // Calibración oficial EUCAST v16.0
    let disk_positions = [(CX, CY - 110), (CX + 110, CY), (CX, CY + 110), (CX - 110, CY)];

    let mut is_on_disk_grid = vec![false; W * H];
    for &(dxp, dyp) in &disk_positions {
        for dy in -14..=14 {
            for dx in -14..=14 {
                if dx * dx + dy * dy <= 196 {
                    let x = (dxp as isize + dx) as usize;
                    let y = (dyp as isize + dy) as usize;
                    if x < W && y < H {
                        is_on_disk_grid[y * W + x] = true;
                    }
                }
            }
        }
    }

    let mut sample_counter = 1usize;

    // Tres estados clínicos del aislamiento para contrastar 1ra Línea vs Rescate OMS:
    // 1. CONTROL_VIRGEN (g0 basal)
    // 2. POST_ABANDONO_RAM (g1/g2 seleccionado tras incumplimiento de 1ra línea)
    // 3. SUPERBACTERIA_XDR (g3/g4 multirresistente que exige Panel de Rescate OMS)
    let dist_control = [1.0, 0.0, 0.0, 0.0, 0.0];
    let dist_ram = [0.05, 0.80, 0.15, 0.0, 0.0];
    let dist_xdr = [0.0, 0.0, 0.10, 0.75, 0.15];

    let conditions = [
        ("CONTROL_VIRGEN", dist_control),
        ("POST_ABANDONO_RAM", dist_ram),
        ("SUPERBACTERIA_XDR", dist_xdr),
    ];

    for org_file in &organisms {
        let config: OrganismConfig = load_organism_config(org_file)?;
        let total_ab = config.antibiotics.len();
        let num_panels = if total_ab >= 8 { 2 } else { 1 };

        // Iterar sobre Panel Primario (offset 0) y Panel de Rescate OMS (offset 4)
        for panel_idx in 0..num_panels {
            let panel_offset = panel_idx * 4;
            let panel_label = if panel_idx == 0 { "PRIMARIO_1RA" } else { "RESCATE_OMS" };
            let active_antibiotics = &config.antibiotics[panel_offset..(panel_offset + 4).min(total_ab)];
            let num_ab = active_antibiotics.len();

            let bio_params_list: Vec<BiophysicalParams> = active_antibiotics
                .iter()
                .map(|ab| BiophysicalParams {
                    base_division_prob: config.kinetics.base_division_prob,
                    fitness_cost_per_mutation: config.kinetics.fitness_cost_per_mutation,
                    maintenance_cost: config.kinetics.maintenance_cost,
                    division_cost: config.kinetics.division_cost,
                    min_resource_for_division: config.kinetics.min_resource_for_division,
                    hill_e_max: ab.hill_e_max,
                    hill_coefficient: ab.hill_coefficient,
                    mic_by_genotype: ab.mic_genotypes,
                    mpc_by_genotype: ab.mpc_genotypes,
                    sos_base_mutation_prob: 0.0,
                    sos_max_induction_factor: 1.0,
                    sos_lethal_fraction: config.genetics.sos_lethal_fraction,
                })
                .collect();

            let mut rng = Xoshiro256PlusPlus::seed_from_u64(1000 + sample_counter as u64);
            let mut control_halos = Vec::new();

            for (cond_name, dist) in conditions {
                let mut petri_cells = DoubleBufferGrid::new(W, H, Cell::EMPTY);
                let mut petri_drugs: Vec<DoubleBufferGrid<f64>> = (0..num_ab)
                    .map(|_| DoubleBufferGrid::new(W, H, 0.0))
                    .collect();

                for y in 0..H {
                    for x in 0..W {
                        let dx = x as isize - CX as isize;
                        let dy = y as isize - CY as isize;
                        if dx * dx + dy * dy <= (R * R) as isize && rng.random_bool(0.35) {
                            let roll = rng.random::<f64>();
                            let mut cum = 0.0;
                            let mut g_sel = 0u8;
                            for (g, &p) in dist.iter().enumerate() {
                                cum += p;
                                if roll <= cum {
                                    g_sel = g as u8;
                                    break;
                                }
                            }
                            petri_cells.current.set(x, y, Cell::new(g_sel, 0, 80));
                        }
                    }
                }

                for (d, &(px, py)) in disk_positions.iter().enumerate().take(num_ab) {
                    let ab = &active_antibiotics[d];
                    for dy in -14..=14 {
                        for dx in -14..=14 {
                            if dx * dx + dy * dy <= 196 {
                                let x = (px as isize + dx) as usize;
                                let y = (py as isize + dy) as usize;
                                petri_drugs[d].current.set(x, y, ab.load_ug);
                                petri_drugs[d].next.set(x, y, ab.load_ug);
                                petri_cells.current.set(x, y, Cell::EMPTY);
                            }
                        }
                    }
                }

                for _ in 0..INCUBATION_TICKS {
                    for (d, ab) in active_antibiotics.iter().enumerate().take(num_ab) {
                        let diff_params = DiffusionParams {
                            diffusion_coeff: ab.diffusion_coeff,
                            clearance_rate: ab.clearance_rate,
                            dx: 1.0,
                            dt: 1.0,
                        };
                        let petri_drug = &mut petri_drugs[d];
                        step_diffusion_2d_biofilm(
                            &petri_drug.current,
                            &mut petri_drug.next,
                            &petri_cells.current,
                            &diff_params,
                            ab.enzymatic_deg_rate,
                        );
                        petri_drug.swap();

                        let (px, py) = disk_positions[d];
                        for dy in -14..=14 {
                            for dx in -14..=14 {
                                if dx * dx + dy * dy <= 196 {
                                    let x = (px as isize + dx) as usize;
                                    let y = (py as isize + dy) as usize;
                                    petri_drugs[d].current.set(x, y, ab.load_ug);
                                }
                            }
                        }
                    }

                    for y in (CY - R)..(CY + R) {
                        let y_offset = y * W;
                        for x in (CX - R)..(CX + R) {
                            let dx = x as isize - CX as isize;
                            let dy = y as isize - CY as isize;
                            if dx * dx + dy * dy > (R * R) as isize {
                                continue;
                            }

                            if is_on_disk_grid[y_offset + x] {
                                petri_cells.next.set(x, y, Cell::EMPTY);
                                continue;
                            }

                            let mut min_dist_sq = usize::MAX;
                            let mut closest_d = 0;
                            for (d, &(px, py)) in disk_positions.iter().enumerate().take(num_ab) {
                                let ddx = (x as isize - px as isize).unsigned_abs();
                                let ddy = (y as isize - py as isize).unsigned_abs();
                                let d_sq = ddx * ddx + ddy * ddy;
                                if d_sq < min_dist_sq {
                                    min_dist_sq = d_sq;
                                    closest_d = d;
                                }
                            }

                            let cell = *petri_cells.current.get(x, y);
                            let density = petri_cells.current.chamfer_density_5_7(x, y);

                            if cell.is_alive() {
                                let mut current_c = cell;
                                for d in 0..num_ab {
                                    let local_c = *petri_drugs[d].current.get(x, y);
                                    current_c = update_living_cell(
                                        current_c,
                                        local_c,
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
                                    let g_daughter = candidate.genotype() as usize;
                                    let mut survives_all = true;
                                    for d in 0..num_ab {
                                        let conc_d = *petri_drugs[d].current.get(x, y);
                                        if conc_d >= bio_params_list[d].mic_by_genotype[g_daughter] {
                                            survives_all = false;
                                            break;
                                        }
                                    }
                                    petri_cells.next.set(
                                        x,
                                        y,
                                        if survives_all { candidate } else { Cell::EMPTY },
                                    );
                                } else {
                                    petri_cells.next.set(x, y, Cell::EMPTY);
                                }
                            }
                        }
                    }
                    petri_cells.swap();
                }

                for (d, ab) in active_antibiotics.iter().enumerate().take(num_ab) {
                    let (px, py) = disk_positions[d];
                    let mut min_r = 125usize;
                    for ray in 0..16 {
                        let ang = (ray as f64) * (2.0 * std::f64::consts::PI / 16.0);
                        let mut r_clear = 14usize;
                        for r_dist in 14..125 {
                            let cx_ray = (px as f64 + r_dist as f64 * ang.cos()) as usize;
                            let cy_ray = (py as f64 + r_dist as f64 * ang.sin()) as usize;
                            if cx_ray < W && cy_ray < H {
                                let cell = petri_cells.current.get(cx_ray, cy_ray);
                                let dens = petri_cells.current.chamfer_density_5_7(cx_ray, cy_ray);
                                if cell.is_alive() && dens >= 20 {
                                    r_clear = r_dist;
                                    break;
                                }
                                r_clear = r_dist;
                            }
                        }
                        if r_clear < min_r {
                            min_r = r_clear;
                        }
                    }

                    let diam_mm = (2.0 * min_r as f64 * MM_PER_PX).max(6.0);
                    let cat = if diam_mm >= ab.breakpoint_s { "S" } else if diam_mm >= ab.breakpoint_r { "I" } else { "R" };

                    let delta = if cond_name == "CONTROL_VIRGEN" {
                        control_halos.push(diam_mm);
                        0.0
                    } else {
                        diam_mm - control_halos[d]
                    };

                    writeln!(
                        writer,
                        "{},{},{},{},{},{:.1},{:.2},{:.1},{},{:.2}",
                        sample_counter, config.species, panel_label, cond_name, ab.code,
                        ab.load_ug, diam_mm, ab.breakpoint_s, cat, delta
                    )?;
                    sample_counter += 1;
                }
            }
        }
    }

    println!("   -> Ensayos traslacionales completados (8 antibióticos x 3 estados): {} guardado.", file_path);
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    create_dir_all(DATA_DIR)?;
    println!("===========================================================");
    println!("TESIS RAM - MOTOR HPC HEADLESS (MONTE CARLO PARALELO)");
    println!("===========================================================");

    run_headless_quimiostato()?;
    run_headless_mega_plate()?;
    run_headless_grid_search()?;
    run_headless_traslacional()?;

    println!("\nTodos los conjuntos de datos fueron generados exitosamente en {}/", DATA_DIR);
    Ok(())
}