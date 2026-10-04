//! src/bin/4_tratamiento_clinico.rs
//! Simulación in vivo: Farmacocinética Tisular (Bateman), Malla Vascular y Pastillero Virtual Interactivo.

use minifb::{Key, Window, WindowOptions};
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;

use tesis_ram::biology::rules::{try_colonize_empty_cell, update_living_cell, BiophysicalParams};
use tesis_ram::config::{load_organism_config, OrganismConfig};
use tesis_ram::core::cell::Cell;
use tesis_ram::core::grid::DoubleBufferGrid;
use tesis_ram::physics::bateman::BatemanRegimen;
use tesis_ram::physics::diffusion::{step_diffusion_2d, DiffusionParams};
use tesis_ram::visualizer::osd::{draw_rect, draw_text};

const WIDTH: usize = 1000;
const HEIGHT: usize = 600;
const OSD_HEIGHT: usize = 90;

const TISSUE_X: usize = 30;
const TISSUE_Y: usize = 122;
const TISSUE_W: usize = 440;
const TISSUE_H: usize = 448;

#[derive(Clone, Copy)]
struct CapillaryPoint {
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

fn log_pk_to_y(conc: f64, gy: usize, gh: usize) -> usize {
    let log_min = -2.3f64; // ~0.005 ug/mL
    let log_max = 2.60f64; // ~400 ug/mL
    let log_c = conc.max(0.005).log10();
    let norm = ((log_c - log_min) / (log_max - log_min)).clamp(0.0, 1.0);
    (gy + gh) - (norm * (gh - 45) as f64) as usize
}

fn main() {
    let mut window = Window::new(
        "Tesis RAM - Farmacocinetica Tisular In Vivo y Rebrote (Bateman / Hill)",
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
    let mut current_ab_idx = 0usize;

    let regimens_tau = [480usize, 720usize, 1440usize];
    let regimens_labels = ["q8h (cada 8 h)", "q12h (cada 12 h)", "q24h (cada 24 h)"];
    let mut current_reg_idx = 0usize;

    let ab_init = &config.antibiotics[current_ab_idx];
    let mut pk = BatemanRegimen::new(0.038, 0.0048, ab_init.c_max_plasma, regimens_tau[current_reg_idx]);

    let diff_params = DiffusionParams {
        diffusion_coeff: 0.50,
        clearance_rate: 0.00018,
        dx: 1.0,
        dt: 1.0,
    };

    let mut cell_grid = DoubleBufferGrid::new(WIDTH, HEIGHT, Cell::EMPTY);
    let mut drug_grid = DoubleBufferGrid::new(WIDTH, HEIGHT, 0.0);

    init_infection(&mut cell_grid, &mut drug_grid, &capillaries, &mut rng);

    let mut tick = 0usize;
    let mut is_paused = false;
    let mut speed_multiplier = 1usize;

    let mut pk_history = Vec::with_capacity(450);
    let mut ca_history = Vec::with_capacity(450);
    let mut time_above_mic_count = 0usize;
    let mut total_monitored_ticks = 0usize;

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

        if window.is_key_pressed(Key::A, minifb::KeyRepeat::No) {
            current_ab_idx = (current_ab_idx + 1) % config.antibiotics.len();
            reload_simulation = true;
        }

        if window.is_key_pressed(Key::T, minifb::KeyRepeat::No) {
            current_reg_idx = (current_reg_idx + 1) % 3;
            reload_simulation = true;
        }

        // [O] Alternar omisión de la próxima dosis programada
        if window.is_key_pressed(Key::O, minifb::KeyRepeat::No) {
            pk.toggle_next_dose(tick);
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
            init_infection(&mut cell_grid, &mut drug_grid, &capillaries, &mut rng);
            pk_history.clear();
            ca_history.clear();
            time_above_mic_count = 0;
            total_monitored_ticks = 0;
            tick = 0;
        }

        if window.is_key_pressed(Key::Equal, minifb::KeyRepeat::No) {
            speed_multiplier = (speed_multiplier + 1).min(5);
        }
        if window.is_key_pressed(Key::Minus, minifb::KeyRepeat::No) {
            speed_multiplier = speed_multiplier.saturating_sub(1).max(1);
        }

        let ab = &config.antibiotics[current_ab_idx.min(config.antibiotics.len() - 1)];
        let mic_s = ab.mic_genotypes[0];
        let mpc_s = ab.mpc_genotypes[0];

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

        // --- Ciclo de Simulación Tisular ---
        if !is_paused {
            for _ in 0..speed_multiplier {
                let c_plasma = pk.concentration_at(tick);

                if c_plasma >= mic_s {
                    time_above_mic_count += 1;
                }
                total_monitored_ticks += 1;

                // 1. Difusión desde los capilares
                step_diffusion_2d(&drug_grid.current, &mut drug_grid.next, &diff_params);
                drug_grid.swap();

                for cap in &capillaries {
                    drug_grid.current.set(cap.x, cap.y, c_plasma);
                }

                // 2. Microdinámica bacteriana intersticial
                for y in TISSUE_Y..(TISSUE_Y + TISSUE_H) {
                    for x in TISSUE_X..(TISSUE_X + TISSUE_W) {
                        let is_vessel = capillaries.iter().any(|c| c.x == x && c.y == y);
                        if is_vessel {
                            cell_grid.next.set(x, y, Cell::EMPTY);
                            continue;
                        }

                        let local_c = *drug_grid.current.get(x, y);
                        let cell = *cell_grid.current.get(x, y);
                        let density = cell_grid.current.chamfer_density_5_7(x, y);

                        if cell.is_alive() {
                            let mut updated = update_living_cell(cell, local_c, density, false, &bio_params, 1.0, &mut rng);
                            if updated.is_alive() {
                                updated.add_resource(1);
                            }
                            cell_grid.next.set(x, y, updated);
                        } else {
                            let neighbors = cell_grid.current.get_living_neighbors(x, y);
                            let colonized = try_colonize_empty_cell(
                                x,
                                y,
                                density,
                                &neighbors,
                                &drug_grid.current,
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

        let mut counts = [0usize; 5];
        let mut total_bacteria = 0usize;
        for y in TISSUE_Y..(TISSUE_Y + TISSUE_H) {
            for x in TISSUE_X..(TISSUE_X + TISSUE_W) {
                let cell = *cell_grid.current.get(x, y);
                if cell.is_alive() {
                    counts[cell.genotype() as usize] += 1;
                    total_bacteria += 1;
                }
            }
        }

        let c_plasma_now = pk.concentration_at(tick);
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

        // --- Renderizado OSD y Gráficas ---
        fb.fill(0x0F172A);

        // Cabecera OSD Superior
        draw_rect(&mut fb, WIDTH, 0, 0, WIDTH, OSD_HEIGHT, 0x1E293B);
        draw_rect(&mut fb, WIDTH, 0, OSD_HEIGHT - 2, WIDTH, 2, 0x38BDF8);

        draw_text(&mut fb, WIDTH, 20, 10, "UNIVERSIDAD MAYOR DE SAN ANDRES - INFORMATICA | TESIS RAM", 0x38BDF8);
        draw_text(&mut fb, WIDTH, 20, 26, "FARMACOCINETICA TISULAR IN VIVO: MODELO BATEMAN Y REBROTE RAM", 0xE2E8F0);

        let current_dose_num = ((tick / pk.tau_minutes) + 1).min(pk.total_doses);
        let status_str = format!(
            "MIN: {:04} (Dia {:.1}) | {} | {}: {} | DOSIS: {}/{} | ADH: {:.0}%",
            tick,
            tick as f64 / 1440.0,
            config.species,
            ab.code,
            ab.name,
            current_dose_num,
            pk.total_doses,
            pk.adherence_pct(tick)
        );
        draw_text(&mut fb, WIDTH, 20, 44, &status_str, 0x4ADE80);

        draw_text(
            &mut fb,
            WIDTH,
            20,
            62,
            "[O] Omitir/Tomar Prox  [A] Farmaco  [T] Pauta  [1..3] Cepa  [ESPACIO] Pausa  [R] Reset",
            0x94A3B8,
        );

        // --- Pastillero Virtual / Calendario de Dosis (y = 92..116) ---
        draw_rect(&mut fb, WIDTH, 0, 90, WIDTH, 28, 0x111827);
        draw_rect(&mut fb, WIDTH, 0, 117, WIDTH, 1, 0x334155);

        draw_text(&mut fb, WIDTH, 20, 99, "CALENDARIO:", 0x94A3B8);

        let box_w = 26usize;
        let box_h = 16usize;
        let start_box_x = 120usize;
        let next_idx_opt = pk.next_dose_index(tick);
        let current_elapsed_idx = tick / pk.tau_minutes;

        for d in 0..pk.total_doses {
            let bx = start_box_x + d * (box_w + 5);
            let by = 96;

            let is_past = d <= current_elapsed_idx;
            let is_next = Some(d) == next_idx_opt;
            let is_taken = pk.doses_taken[d];

            let (bg_col, text_col) = if is_past {
                if is_taken {
                    (0x166534, 0xDCFCE7) // Verde oscuro: Tomada con éxito
                } else {
                    (0x991B1B, 0xFEE2E2) // Rojo oscuro: Omitida en el pasado
                }
            } else if is_next {
                if is_taken {
                    (0x0284C7, 0xF0F9FF) // Azul cian: Próxima dosis (programada)
                } else {
                    (0xDC2626, 0xFFFFFF) // Rojo vivo: Próxima dosis (marcada para omitir)
                }
            } else {
                (0x1F2937, 0x6B7280) // Gris oscuro: Futura pendiente
            };

            draw_rect(&mut fb, WIDTH, bx, by, box_w, box_h, bg_col);

            // Borde resaltado para la próxima dosis
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

        // Indicador textual del estado de la próxima toma
        if let Some(n_idx) = next_idx_opt {
            let next_minute = n_idx * pk.tau_minutes;
            let mins_left = next_minute.saturating_sub(tick);
            let next_state_str = if pk.doses_taken[n_idx] {
                format!("Prox: D{} en {}m [TOMAR]", n_idx + 1, mins_left)
            } else {
                format!("Prox: D{} en {}m [OMITIDA!]", n_idx + 1, mins_left)
            };
            let state_col = if pk.doses_taken[n_idx] { 0x38BDF8 } else { 0xEF4444 };
            draw_text(&mut fb, WIDTH, start_box_x + pk.total_doses * 31 + 10, 99, &next_state_str, state_col);
        } else {
            draw_text(&mut fb, WIDTH, start_box_x + pk.total_doses * 31 + 10, 99, "TRATAMIENTO COMPLETADO", 0x4ADE80);
        }

        // 1. Lecho Tisular Microvascular (Izquierda)
        draw_rect(&mut fb, WIDTH, TISSUE_X - 2, TISSUE_Y - 2, TISSUE_W + 4, TISSUE_H + 4, 0x334155);

        for y in TISSUE_Y..(TISSUE_Y + TISSUE_H) {
            for x in TISSUE_X..(TISSUE_X + TISSUE_W) {
                let is_vessel = capillaries.iter().any(|c| c.x == x && c.y == y);
                if is_vessel {
                    fb[y * WIDTH + x] = 0xEF4444; // Capilar sanguíneo
                    continue;
                }

                let cell = *cell_grid.current.get(x, y);
                if cell.is_alive() {
                    fb[y * WIDTH + x] = match cell.genotype() {
                        0 => 0x22C55E, // Verde: Salvaje
                        1 => 0xEAB308, // Amarillo: g1
                        2 => 0xF97316, // Naranja: g2
                        3 => 0xDC2626, // Rojo: g3
                        _ => 0xA855F7, // Púrpura: g4
                    };
                } else {
                    let conc = *drug_grid.current.get(x, y);
                    let norm = (conc / ab.c_max_plasma).clamp(0.0, 1.0);
                    let intensity = (norm * 75.0) as u32;
                    let r_tis = 0x30 + intensity / 2;
                    let g_tis = 0x1A + intensity / 3;
                    let b_tis = 0x25 + intensity;
                    fb[y * WIDTH + x] = (r_tis << 16) | (g_tis << 8) | b_tis;
                }
            }
        }
        draw_text(&mut fb, WIDTH, TISSUE_X + 10, TISSUE_Y + TISSUE_H - 18, "LECHO VASCULAR TISULAR (64 CAPILARES)", 0xF1F5F9);

        // 2. Curva Farmacocinética Semi-Logarítmica (Bateman)
        let gx = 510;
        let gy = 122;
        let gw = 460;
        let gh = 180;

        draw_rect(&mut fb, WIDTH, gx, gy, gw, gh, 0x020617);
        draw_rect(&mut fb, WIDTH, gx, gy, gw, 1, 0x475569);
        draw_rect(&mut fb, WIDTH, gx, gy + gh, gw, 1, 0x475569);
        draw_rect(&mut fb, WIDTH, gx, gy, 1, gh, 0x475569);
        draw_rect(&mut fb, WIDTH, gx + gw, gy, 1, gh + 1, 0x475569);

        let title_pk = format!("FARMACOCINETICA C(t) [{}]", ab.code);
        draw_text(&mut fb, WIDTH, gx + 15, gy + 12, &title_pk, 0x38BDF8);

        let c_act_label = format!("C_act: {:.2} ug/mL", c_plasma_now);
        draw_text(&mut fb, WIDTH, gx + 230, gy + 12, &c_act_label, 0x4ADE80);

        let y_mic = log_pk_to_y(mic_s, gy, gh);
        let y_mpc = log_pk_to_y(mpc_s, gy, gh);

        // Franja sombreada de la MSW
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

        // Rótulos numéricos explícitos
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

        // 3. Carga Bacteriana Intersticial
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

        // 4. Panel de Diagnóstico Clínico
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

        let diagnosis = if total_bacteria < 5000 {
            ("INFECCION CONTROLADA (CURACION CLINICA)", 0x4ADE80)
        } else if counts[1] + counts[2] + counts[3] + counts[4] > counts[0] {
            ("FRACASO TERAPEUTICO: REBROTE POR SELECCION RAM", 0xEF4444)
        } else if pct_time_above_mic < 20.0 {
            ("INEFICAZ: CONCENTRACION INSUFICIENTE (RESISTENCIA)", 0xEF4444)
        } else {
            ("CURSO CLINICO EN TRATAMIENTO", 0xF59E0B)
        };
        draw_text(&mut fb, WIDTH, gx + 15, info_y + 46, diagnosis.0, diagnosis.1);

        window.update_with_buffer(&fb, WIDTH, HEIGHT).unwrap();
    }
}