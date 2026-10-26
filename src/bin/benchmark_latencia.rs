//! src/bin/benchmark_latencia.rs
//! Benchmark formal de alta precisión para la Hipótesis 2 de la tesis.
//! Evalúa una malla de 10^6 nodos (1000x1000 celdas) a lo largo de 1000 iteraciones.
//!
//! Uso:
//!     cargo run --release --bin benchmark_latencia

use std::time::Instant;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

// Importaciones de tu biblioteca central
use tesis_ram::core::cell::Cell;
use tesis_ram::core::grid::{DoubleBufferGrid, Grid};
use tesis_ram::physics::diffusion::{step_diffusion_2d, DiffusionParams};
use tesis_ram::biology::rules::{update_living_cell, BiophysicalParams};
use tesis_ram::config::OrganismConfig;

const WIDTH: usize = 1000;
const HEIGHT: usize = 1000;
const TOTAL_NODES: usize = WIDTH * HEIGHT; // 1,000,000 nodos
const ITERACIONES: usize = 1000;
const WARMUP_TICKS: usize = 50;

fn main() {
    println!("===============================================================");
    println!("  BENCHMARK FORMAL HPC: EVALUACIÓN DE LATENCIA (HIPÓTESIS 2)");
    println!("  Malla: {} x {} ({} nodos)", WIDTH, HEIGHT, TOTAL_NODES);
    println!("  Iteraciones medidas: {} (tras {} ticks de warmup)", ITERACIONES, WARMUP_TICKS);
    println!("===============================================================\n");

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
    let mut cell_buffer = DoubleBufferGrid::new(WIDTH, HEIGHT, Cell::EMPTY);
    let mut drug_current = Grid::new(WIDTH, HEIGHT, 0.0f64);
    let mut drug_next = Grid::new(WIDTH, HEIGHT, 0.0f64);

    // Carga de configuración
    let config_path = "config/ecoli_atcc25922.toml";
    let org: OrganismConfig = toml::from_str(
        &std::fs::read_to_string(config_path)
            .unwrap_or_else(|_| panic!("No se pudo leer {}", config_path))
    ).expect("Error al deserializar TOML");

    let ab = &org.antibiotics[0];

    // Parámetros físicos estructurados según tu biblioteca
    let diff_params = DiffusionParams {
        diffusion_coeff: ab.diffusion_coeff,
        clearance_rate: ab.clearance_rate,
        dt: 1.0,
        dx: 1.0,
    };

    // Parámetros biofísicos consolidados para update_living_cell
    let bio_params = BiophysicalParams::default();

    println!("[*] Inicializando población confluente (~700,000 bacterias)...");
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let i = y * WIDTH + x;
            if (x + y) % 3 != 0 {
                cell_buffer.current.data[i] = Cell::new(0, 0, 50);
            }
            drug_current.data[i] = if x > 500 { ab.c_max_plasma * 0.5 } else { 0.0 };
        }
    }

    println!("[*] Ejecutando fase de calentamiento...");
    for _ in 0..WARMUP_TICKS {
        step_diffusion_2d(&drug_current, &mut drug_next, &diff_params);
        std::mem::swap(&mut drug_current, &mut drug_next);
        cell_buffer.swap();
    }

    let mut tiempos_difusion = Vec::with_capacity(ITERACIONES);
    let mut tiempos_celular = Vec::with_capacity(ITERACIONES);
    let mut tiempos_swap = Vec::with_capacity(ITERACIONES);
    let mut tiempos_metricas = Vec::with_capacity(ITERACIONES);
    let mut tiempos_totales = Vec::with_capacity(ITERACIONES);

    println!("[*] Midiendo {} iteraciones consecutivas...", ITERACIONES);

    for _ in 0..ITERACIONES {
        let t_inicio_global = Instant::now();

        // 1. Difusión Continua 2D
        let t0 = Instant::now();
        step_diffusion_2d(&drug_current, &mut drug_next, &diff_params);
        std::mem::swap(&mut drug_current, &mut drug_next);
        let d_difusion = t0.elapsed();

        // 2. Cinética Estocástica Celular
        let t1 = Instant::now();
        for y in 1..HEIGHT - 1 {
            let row = y * WIDTH;
            for x in 1..WIDTH - 1 {
                let idx = row + x;
                let c = cell_buffer.current.data[idx];
                if c.is_alive() {
                    // Cálculo de densidad local Chamfer (pesos 7 ortogonal, 5 diagonal)
                    let mut density: u16 = 0;
                    if cell_buffer.current.data[idx - 1].is_alive() { density += 7; }
                    if cell_buffer.current.data[idx + 1].is_alive() { density += 7; }
                    if cell_buffer.current.data[idx - WIDTH].is_alive() { density += 7; }
                    if cell_buffer.current.data[idx + WIDTH].is_alive() { density += 7; }
                    if cell_buffer.current.data[idx - WIDTH - 1].is_alive() { density += 5; }
                    if cell_buffer.current.data[idx - WIDTH + 1].is_alive() { density += 5; }
                    if cell_buffer.current.data[idx + WIDTH - 1].is_alive() { density += 5; }
                    if cell_buffer.current.data[idx + WIDTH + 1].is_alive() { density += 5; }

                    let conc = drug_current.data[idx];
                    cell_buffer.next.data[idx] = update_living_cell(c, conc, density, true, &bio_params, 1.0, &mut rng);
                } else {
                    cell_buffer.next.data[idx] = Cell::EMPTY;
                }
            }
        }
        let d_celular = t1.elapsed();

        // 3. Conmutación Doble Búfer O(1)
        let t2 = Instant::now();
        cell_buffer.swap();
        let d_swap = t2.elapsed();

        // 4. Agregación de Métricas
        let t3 = Instant::now();
        let mut total_vivas = 0usize;
        let mut mutantes = 0usize;
        for cell in &cell_buffer.current.data {
            if cell.is_alive() {
                total_vivas += 1;
                if cell.genotype() > 0 {
                    mutantes += 1;
                }
            }
        }
        let d_metricas = t3.elapsed();

        let d_total = t_inicio_global.elapsed();

        tiempos_difusion.push(d_difusion.as_micros() as f64);
        tiempos_celular.push(d_celular.as_micros() as f64);
        tiempos_swap.push(d_swap.as_micros() as f64);
        tiempos_metricas.push(d_metricas.as_micros() as f64);
        tiempos_totales.push(d_total.as_micros() as f64);

        std::hint::black_box((total_vivas, mutantes));
    }

    fn stats(v: &[f64]) -> (f64, f64) {
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (v.len() - 1) as f64;
        (mean / 1000.0, var.sqrt() / 1000.0)
    }

    let (m_dif, s_dif) = stats(&tiempos_difusion);
    let (m_cel, s_cel) = stats(&tiempos_celular);
    let (m_swp, s_swp) = stats(&tiempos_swap);
    let (m_met, s_met) = stats(&tiempos_metricas);
    let (m_tot, s_tot) = stats(&tiempos_totales);

    println!("\n===============================================================");
    println!("                     RESULTADOS CONSOLIDADOS                   ");
    println!("===============================================================");
    println!("{:<32} | {:>10} | {:>10} | {:>8}", "Fase Algorítmica", "Media (ms)", "StdDev (ms)", "% Ciclo");
    println!("---------------------------------------------------------------");
    println!("{:<32} | {:>10.2} | {:>10.2} | {:>7.1}%", "Difusión 2D (Fick / Chamfer)", m_dif, s_dif, (m_dif / m_tot) * 100.0);
    println!("{:<32} | {:>10.2} | {:>10.2} | {:>7.1}%", "Reglas Estocásticas (Hill/SOS)", m_cel, s_cel, (m_cel / m_tot) * 100.0);
    println!("{:<32} | {:>10.4} | {:>10.4} | {:>7.1}%", "Conmutación Doble Búfer O(1)", m_swp, s_swp, (m_swp / m_tot) * 100.0);
    println!("{:<32} | {:>10.2} | {:>10.2} | {:>7.1}%", "Agregación de Métricas OSD", m_met, s_met, (m_met / m_tot) * 100.0);
    println!("---------------------------------------------------------------");
    println!("{:<32} | {:>10.2} | {:>10.2} | {:>7.1}%", "LATENCIA TOTAL POR PASO", m_tot, s_tot, 100.0);
    println!("===============================================================");

    println!("\n[VEREDICTO HIPÓTESIS 2]");
    if m_tot <= 500.0 {
        println!("  CUMPLE: Latencia media de {:.2} ms <= 500.0 ms (Factor de aceleración: {:.1}x)", m_tot, 500.0 / m_tot);
    } else {
        println!("  FALLA: La latencia supera los 500 ms.");
    }
}