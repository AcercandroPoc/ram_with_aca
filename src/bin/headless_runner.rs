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
use tesis_ram::physics::diffusion::{step_diffusion_2d, step_diffusion_mega_plate, DiffusionParams};

const DATA_DIR: &str = "data/headless";

// ---------------------------------------------------------------------------
// 1. Simulación Headless: Quimiostato (Convergencia Isomórfica CA vs RK4)
// ---------------------------------------------------------------------------
fn run_headless_quimiostato() -> Result<(), Box<dyn std::error::Error>> {
    println!("[1/4] Ejecutando Quimiostato Headless (CA vs Levin-Regoes RK4)...");
    let file_path = format!("{}/quimiostato_mc.csv", DATA_DIR);
    let mut writer = BufWriter::new(File::create(&file_path)?);

    writeln!(
        writer,
        "tick,ca_total,ca_g0,ca_g1,ca_g2,ca_g3,ca_g4,ode_ns,ode_nr,ode_total,antibiotic_conc,pearson_instant,sq_error"
    )?;

    const WIDTH: usize = 1000;
    const HEIGHT: usize = 600;
    const REACTOR_CX: usize = 200;
    const REACTOR_CY: usize = 345;
    const REACTOR_R: usize = 98;
    let r_sq = (REACTOR_R * REACTOR_R) as isize;

    let ode_params = LevinMonodParams {
        carrying_capacity: 25130.0,
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

    let mut ode_state = ChemostatState { s: 0.0, n_s: 400.0, n_r: 0.0 };
    let mut grid = DoubleBufferGrid::new(WIDTH, HEIGHT, Cell::EMPTY);
    let conc_grid = Grid::new(WIDTH, HEIGHT, 0.0);
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

    let total_ticks = 4000;
    let p_washout = 1.0 - (-ode_params.dilution_d).exp();

    for tick in 0..total_ticks {
        let antibiotic_conc = if tick >= 1500 { 0.08 } else { 0.0 };
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
                writer,
                "{},{:.0},{},{},{},{},{},{:.2},{:.2},{:.2},{:.3},{:.4},{:.2}",
                tick, ca_total, counts[0], counts[1], counts[2], counts[3], counts[4],
                ode_state.n_s, ode_state.n_r, ode_total, antibiotic_conc, r_val, sq_err
            )?;
        }
    }
    println!("   -> Quimiostato finalizado: {} guardado.", file_path);
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
// 3. Simulación Headless: Espacio de Fases [Dosis x Pauta tau] (Auditoría Rigurosa)
// ---------------------------------------------------------------------------
#[derive(Clone)]
struct GridSearchResult {
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
    println!("[3/4] Ejecutando Grid Search Monte Carlo en Paralelo (Dosis x Pauta tau)...");
    let file_path = format!("{}/grid_search_adherence.csv", DATA_DIR);

    let config: OrganismConfig = load_organism_config("config/ecoli_atcc25922.toml")?;
    // Amoxicilina + Clavulánico (AMC): Fármaco estándar ambulatorio de 5 días
    let ab = &config.antibiotics[2.min(config.antibiotics.len() - 1)];
    let mic_s = ab.mic_genotypes[0];
    let mpc_s = ab.mpc_genotypes[0];

    // Espacio de fases: 8 niveles de adherencia x 3 pautas posológicas x 15 réplicas
    let doses_range = vec![2usize, 4, 6, 8, 10, 12, 14, 15];
    let tau_range = vec![480usize, 720usize, 1440usize]; // q8h, q12h, q24h
    let num_replicas = 15usize;

    let mut parameter_space = Vec::new();
    for &doses in &doses_range {
        for &tau in &tau_range {
            for rep in 0..num_replicas {
                parameter_space.push((doses, tau, rep));
            }
        }
    }

    let results: Vec<GridSearchResult> = parameter_space
        .into_par_iter()
        .map(|(doses_before_drop, tau, rep_id)| {
            const TW: usize = 200;
            const TH: usize = 200;
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(
                (rep_id as u64 * 7919) + (doses_before_drop as u64 * 31) + (tau as u64 * 17),
            );

            let mut pk = BatemanRegimen::new(0.038, 0.0048, ab.c_max_plasma, tau);

            // Simular abandono terapéutico: omitir las dosis posteriores
            for d in doses_before_drop..pk.total_doses {
                if d < pk.doses_taken.len() {
                    pk.doses_taken[d] = false;
                }
            }

            // Parámetros biofísicos: Tasa basal baja en reposo (evita deriva neutral salvaje),
            // pero fuerte inducción SOS cuando el fármaco se encuentra dentro de la MSW.
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
                sos_base_mutation_prob: 1.0e-5,          // Tasa basal en reposo
                sos_max_induction_factor: 500.0,         // Inducción adaptativa activa bajo estrés MSW
                sos_lethal_fraction: config.genetics.sos_lethal_fraction,
            };

            let diff_params = DiffusionParams {
                diffusion_coeff: 0.40,
                clearance_rate: 0.00018,
                dx: 1.0,
                dt: 1.0,
            };

            let mut cell_grid = DoubleBufferGrid::new(TW, TH, Cell::EMPTY);
            let mut drug_grid = DoubleBufferGrid::new(TW, TH, 0.0);

            // Red microvascular de Krogh: 4x4 (16 capilares) espaciados ~45 px
            let cols = 4;
            let rows = 4;
            let margin = 25.0;
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

            // Inóculo 100% sensible virgen (g0)
            for y in 0..TH {
                for x in 0..TW {
                    let is_cap = is_cap_grid[y * TW + x];
                    if !is_cap && rng.random_bool(0.40) {
                        cell_grid.current.set(x, y, Cell::new(0, 0, 100));
                    }
                }
            }

            let mut time_above_mic = 0usize;
            let mut time_in_msw = 0usize;
            let total_ticks = 5 * 1440; // 5 días de seguimiento

            for tick in 0..total_ticks {
                let c_plasma = pk.concentration_at(tick);
                if c_plasma >= mic_s {
                    time_above_mic += 1;
                }
                if c_plasma >= mic_s && c_plasma <= mpc_s {
                    time_in_msw += 1;
                }

                step_diffusion_2d(&drug_grid.current, &mut drug_grid.next, &diff_params);
                drug_grid.swap();

                for &(cx, cy) in &caps {
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            if dx * dx + dy * dy <= 2 {
                                let x = (cx as isize + dx) as usize;
                                let y = (cy as isize + dy) as usize;
                                if x < TW && y < TH {
                                    drug_grid.current.set(x, y, c_plasma);
                                }
                            }
                        }
                    }
                }

                if tick % 2 == 0 {
                    let mut living_cells_step = 0usize;

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
                                let mut updated = update_living_cell(cell, local_c, density, false, &bio_params, 2.0, &mut rng);
                                if updated.is_alive() {
                                    updated.add_resource(2);
                                }
                                cell_grid.next.set(x, y, updated);
                            } else {
                                let neighbors = cell_grid.current.get_living_neighbors(x, y);
                                let colonized = try_colonize_empty_cell(x, y, density, &neighbors, &drug_grid.current, &bio_params, &mut rng);
                                cell_grid.next.set(x, y, colonized);
                            }
                        }
                    }
                    cell_grid.swap();

                    // Early stopping si la erradicación bacteriana es total
                    if living_cells_step == 0 && tick > 1440 {
                        return GridSearchResult {
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
            let mut final_mutants = 0usize;
            for y in 0..TH {
                for x in 0..TW {
                    let cell = *cell_grid.current.get(x, y);
                    if cell.is_alive() {
                        final_total += 1;
                        if cell.genotype() > 0 {
                            final_mutants += 1;
                        }
                    }
                }
            }

            // Clasificación clínica:
            // 1. CURACION: erradicación exitosa (población final residual insignificante).
            // 2. FRACASO_RAM: la infección repuntó y la subpoblación resistente domina o tiene masa crítica.
            // 3. RECAIDA_SENSIBLE: la infección repuntó, pero por abandono precoz de bacterias sensibles no mutadas.
            let outcome = if final_total < 500 {
                "CURACION"
            } else if final_mutants >= 50 && (final_mutants as f64 / final_total as f64) >= 0.15 {
                "FRACASO_RAM"
            } else {
                "RECAIDA_SENSIBLE"
            };

            GridSearchResult {
                replica_id: rep_id,
                doses_before_drop,
                tau_regimen: tau,
                pct_time_above_mic: (time_above_mic as f64 / total_ticks as f64) * 100.0,
                time_in_msw,
                final_pop_total: final_total,
                final_mutant_pop: final_mutants,
                outcome,
            }
        })
        .collect();

    let mut writer = BufWriter::new(File::create(&file_path)?);
    writeln!(
        writer,
        "replica_id,doses_before_drop,tau_regimen,pct_time_above_mic,time_in_msw,final_pop_total,final_mutant_pop,treatment_outcome"
    )?;

    for r in results {
        writeln!(
            writer,
            "{},{},{},{:.2},{},{},{},{}",
            r.replica_id, r.doses_before_drop, r.tau_regimen,
            r.pct_time_above_mic, r.time_in_msw, r.final_pop_total, r.final_mutant_pop, r.outcome
        )?;
    }

    println!("   -> Grid Search completado: réplicas ejecutadas en paralelo.");
    Ok(())
}

// ---------------------------------------------------------------------------
// 4. Simulación Headless: Puente Traslacional (Biopsia Pre vs Post-Abandono)
// ---------------------------------------------------------------------------
fn run_headless_traslacional() -> Result<(), Box<dyn std::error::Error>> {
    println!("[4/4] Ejecutando Ensayos Traslacionales Kirby-Bauer (Pre vs Post Biopsia)...");
    let file_path = format!("{}/traslacional_biopsia.csv", DATA_DIR);
    let mut writer = BufWriter::new(File::create(&file_path)?);

    writeln!(
        writer,
        "sample_id,strain,condition,antibiotic_code,disk_load_ug,halo_diameter_mm,eucast_breakpoint,clinical_category,delta_halo_mm"
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

    for org_file in &organisms {
        let config: OrganismConfig = load_organism_config(org_file)?;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1000 + sample_counter as u64);

        let bio_params_list: Vec<BiophysicalParams> = config.antibiotics.iter().take(4).map(|ab| {
            BiophysicalParams {
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
            }
        }).collect();

        let dist_control = [1.0, 0.0, 0.0, 0.0, 0.0];
        let dist_ram = [0.05, 0.80, 0.15, 0.0, 0.0];

        let conditions = [("CONTROL_VIRGEN", dist_control), ("POST_ABANDONO_RAM", dist_ram)];
        let mut control_halos = Vec::new();

        for (cond_name, dist) in conditions {
            let mut petri_cells = DoubleBufferGrid::new(W, H, Cell::EMPTY);
            let mut petri_drugs: Vec<DoubleBufferGrid<f64>> = (0..4)
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

            for (d, &(px, py)) in disk_positions.iter().enumerate().take(config.antibiotics.len().min(4)) {
                let ab = &config.antibiotics[d];
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

            for _ in 0..1200 {
                for (d, ab) in config.antibiotics.iter().enumerate().take(4) {
                    let diff_params = DiffusionParams {
                        diffusion_coeff: ab.diffusion_coeff,
                        clearance_rate: ab.clearance_rate,
                        dx: 1.0,
                        dt: 1.0,
                    };
                    let petri_drug = &mut petri_drugs[d];
                    step_diffusion_2d(&petri_drug.current, &mut petri_drug.next, &diff_params);
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

                        let cell = *petri_cells.current.get(x, y);
                        let density = petri_cells.current.chamfer_density_5_7(x, y);

                        if cell.is_alive() {
                            let mut killed = false;
                            for d in 0..4.min(config.antibiotics.len()) {
                                let local_c = *petri_drugs[d].current.get(x, y);
                                let upd = update_living_cell(
                                    cell,
                                    local_c,
                                    density,
                                    true,
                                    &bio_params_list[d],
                                    1.0,
                                    &mut rng,
                                );
                                if !upd.is_alive() {
                                    killed = true;
                                    break;
                                }
                            }
                            petri_cells.next.set(x, y, if killed { Cell::EMPTY } else { cell });
                        } else {
                            let neighbors = petri_cells.current.get_living_neighbors(x, y);
                            if density < 40 && !neighbors.is_empty() {
                                let mut can_col = true;
                                for d in 0..4.min(config.antibiotics.len()) {
                                    let test_c = try_colonize_empty_cell(
                                        x,
                                        y,
                                        density,
                                        &neighbors,
                                        &petri_drugs[d].current,
                                        &bio_params_list[d],
                                        &mut rng,
                                    );
                                    if !test_c.is_alive() {
                                        can_col = false;
                                        break;
                                    }
                                }
                                if can_col {
                                    petri_cells.next.set(x, y, Cell::new(neighbors[0].0.genotype(), 0, 80));
                                } else {
                                    petri_cells.next.set(x, y, Cell::EMPTY);
                                }
                            } else {
                                petri_cells.next.set(x, y, Cell::EMPTY);
                            }
                        }
                    }
                }
                petri_cells.swap();
            }

            for (d, ab) in config.antibiotics.iter().enumerate().take(4) {
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
                    "{},{},{},{},{:.1},{:.2},{:.1},{},{:.2}",
                    sample_counter, config.species, cond_name, ab.code,
                    ab.load_ug, diam_mm, ab.breakpoint_s, cat, delta
                )?;
                sample_counter += 1;
            }
        }
    }

    println!("   -> Ensayos traslacionales completados: {} guardado.", file_path);
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