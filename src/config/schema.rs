//! src/config/schema.rs
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct OrganismConfig {
    pub name: String,
    pub species: String,
    pub strain_code: String,
    pub gram_stain: String,
    pub clinical_description: String,
    pub kinetics: KineticConfig,
    pub genetics: GeneticsConfig,
    pub antibiotics: Vec<AntibioticConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct KineticConfig {
    pub base_division_prob: f64,
    pub fitness_cost_per_mutation: f64,
    pub maintenance_cost: u8,
    pub division_cost: u8,
    pub min_resource_for_division: u8,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GeneticsConfig {
    pub sos_base_mutation_prob: f64,
    pub sos_max_induction_factor: f64,
    pub sos_lethal_fraction: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AntibioticConfig {
    pub code: String,
    pub name: String,
    pub load_ug: f64,
    pub c_max_plasma: f64, // Concentración pico en sangre in vivo (ug/mL)
    pub diffusion_coeff: f64,
    pub clearance_rate: f64,
    pub hill_e_max: f64,
    pub hill_coefficient: f64,
    pub mic_genotypes: [f64; 5],
    pub mpc_genotypes: [f64; 5],
    pub breakpoint_s: f64,
    pub breakpoint_r: f64,
}