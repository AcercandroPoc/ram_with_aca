//! src/biology/rules.rs
//! Dinámica celular basada en primeros principios: Inhibición de replicación por CMI,
//! Inducción SOS (Cirz et al., 2005) y Barrera estricta MPC (Drlica, 2007).

use crate::core::cell::Cell;
use crate::core::grid::Grid;
use rand::prelude::*;

pub const CROWDING_THRESHOLD_CHAMFER: u16 = 40;

#[derive(Clone, Debug)]
pub struct BiophysicalParams {
    pub base_division_prob: f64,
    pub fitness_cost_per_mutation: f64,
    pub maintenance_cost: u8,
    pub division_cost: u8,
    pub min_resource_for_division: u8,
    pub hill_e_max: f64,
    pub hill_coefficient: f64,
    pub mic_by_genotype: [f64; 5],
    pub mpc_by_genotype: [f64; 5],
    pub sos_base_mutation_prob: f64,
    pub sos_max_induction_factor: f64,
    pub sos_lethal_fraction: f64,
}

impl Default for BiophysicalParams {
    fn default() -> Self {
        Self {
            base_division_prob: 0.08,
            fitness_cost_per_mutation: 0.08, // 8% penalización por mutación (Andersson, 2006)
            maintenance_cost: 1,
            division_cost: 20,
            min_resource_for_division: 35,
            hill_e_max: 0.35,
            hill_coefficient: 2.0,
            // Calibración isomórfica con Baym et al. (2016)
            mic_by_genotype: [0.035, 0.200, 1.500, 12.00, 140.0],
            mpc_by_genotype: [0.280, 1.600, 12.00, 96.00, 1120.0], // ~8x CMI (Drlica, 2007)
            sos_base_mutation_prob: 0.002,
            sos_max_induction_factor: 50.0,
            sos_lethal_fraction: 0.50,
        }
    }
}

#[inline]
pub fn eval_hill_death_prob(conc: f64, genotype: u8, params: &BiophysicalParams, dt: f64) -> f64 {
    let mic = params.mic_by_genotype[genotype.min(4) as usize];
    if conc < mic {
        return 0.0;
    }
    let c_h = conc.powf(params.hill_coefficient);
    let mic_h = mic.powf(params.hill_coefficient);
    let psi_kill = params.hill_e_max * (c_h / (mic_h + c_h));
    1.0 - (-psi_kill * dt).exp()
}

pub fn update_living_cell<R: Rng + ?Sized>(
    mut cell: Cell,
    local_conc: f64,
    density: u16,
    is_solid_agar: bool,
    params: &BiophysicalParams,
    dt: f64,
    rng: &mut R,
) -> Cell {
    let g = cell.genotype();
    let mic = params.mic_by_genotype[g as usize];
    let mpc = params.mpc_by_genotype[g as usize];
    let steps = (dt.round() as u8).max(1);

    // Inducción SOS dentro de la Ventana de Selección de Mutantes (MSW = [CMI, MPC])
    if local_conc >= mic && local_conc < mpc {
        for _ in 0..steps {
            cell.inc_stress();
        }
    } else if local_conc < mic {
        for _ in 0..steps {
            cell.dec_stress();
        }
    }

    // Lisis bactericida cuando C >= CMI
    let p_death = eval_hill_death_prob(local_conc, g, params, dt);
    if rng.random_bool(p_death.clamp(0.0, 1.0)) {
        return Cell::EMPTY;
    }

    if density >= CROWDING_THRESHOLD_CHAMFER {
        return cell;
    }

    if is_solid_agar {
        cell.add_resource(steps);
        return cell;
    }

    let total_maint = params.maintenance_cost.saturating_mul(steps);
    if local_conc < mic {
        cell.add_resource(steps);
    } else if !cell.consume_resource(total_maint) {
        return Cell::EMPTY;
    }

    cell
}

/// Colonización estocástica evaluando viabilidad del progenitor y de la célula hija
pub fn try_colonize_empty_cell<R: Rng + ?Sized>(
    target_x: usize,
    target_y: usize,
    density: u16,
    neighbors: &[(Cell, usize, usize)],
    conc_grid: &Grid<f64>,
    params: &BiophysicalParams,
    rng: &mut R,
) -> Cell {
    if density >= CROWDING_THRESHOLD_CHAMFER || neighbors.is_empty() {
        return Cell::EMPTY;
    }

    let target_conc = *conc_grid.get(target_x, target_y);

    // Filtrar progenitores viables: deben tener energía Y no estar por debajo de su CMI
    let mut eligible = [(Cell::EMPTY, 0.0f64); 8];
    let mut elegible_len = 0usize;

    for &(nb, nx, ny) in neighbors {
        if nb.resource >= params.min_resource_for_division {
            let parent_conc = *conc_grid.get(nx, ny);
            let mic_parent = params.mic_by_genotype[nb.genotype() as usize];
            // Principio CLSI/EUCAST: Si C >= CMI, la replicación está bloqueada
            if parent_conc < mic_parent && elegible_len < 8 {
                eligible[elegible_len] = (nb, parent_conc);
                elegible_len += 1;
            }
        }
    }

    if elegible_len == 0 {
        return Cell::EMPTY;
    }

    // Selección aleatoria entre progenitores no inhibidos
    let (parent, parent_conc) = eligible[rng.random_range(0..elegible_len)];
    let g_parent = parent.genotype();
    let mpc_parent = params.mpc_by_genotype[g_parent as usize];

    // Probabilidad de división celular con penalización de fitness
    let fitness = (1.0 - params.fitness_cost_per_mutation * (g_parent as f64)).max(0.1);
    let p_div = (params.base_division_prob * fitness).clamp(0.0, 1.0);

    if !rng.random_bool(p_div) {
        return Cell::EMPTY;
    }

    // Tasa de mutación inducida por la ruta SOS
    let stress = parent.stress();
    let induction = 1.0 + (params.sos_max_induction_factor - 1.0) * (stress as f64 / 15.0);
    let p_mut = (params.sos_base_mutation_prob * induction).clamp(0.0, 1.0);

    let mut daughter_genotype = g_parent;
    let mut daughter_stress = stress / 2;

    if rng.random_bool(p_mut) {
        // Si el entorno materno ya superó la MPC, la mutagénesis viable es nula
        if parent_conc < mpc_parent {
            if rng.random_bool(params.sos_lethal_fraction) {
                return Cell::EMPTY; // Catástrofe de error
            } else {
                daughter_genotype = (g_parent + 1).min(4); // Salto escalonado unitario
                daughter_stress = 0;
            }
        } else {
            return Cell::EMPTY;
        }
    }

    // La célula hija debe tolerar la concentración del parche destino
    let mic_daughter = params.mic_by_genotype[daughter_genotype as usize];
    if target_conc >= mic_daughter {
        return Cell::EMPTY; // La dosis en el parche destino es letal para la progenie
    }

    Cell::new(daughter_genotype, daughter_stress, 80)
}