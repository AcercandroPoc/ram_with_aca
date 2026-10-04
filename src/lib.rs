//! tesis_ram - Simulador Estocástico Multiescala de Resistencia Antimicrobiana
//! Biblioteca núcleo para computación científica, modelado biofísico y validación de tesis.
pub mod config;
pub mod core {
    pub mod cell;
    pub mod grid;
}

pub mod physics {
    pub mod bateman;
    pub mod diffusion;
}

pub mod biology {
    pub mod rules;
}

pub mod analysis {
    pub mod ode;
}

pub mod visualizer {
    pub mod font;
    pub mod osd;
}

// Re-exportaciones de alto nivel para ergonomía en binarios y tests
pub use config::{load_organism_config, OrganismConfig};
pub use core::cell::Cell;
pub use core::grid::{DoubleBufferGrid, Grid};
pub use biology::rules::{BiophysicalParams, update_living_cell, try_colonize_empty_cell};
pub use physics::bateman::BatemanRegimen;
pub use physics::diffusion::{step_diffusion_2d, step_diffusion_mega_plate};
pub use analysis::ode::{ChemostatState, LevinMonodParams, PearsonTracker};