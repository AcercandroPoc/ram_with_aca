//! src/config/mod.rs
//! Cargador de perfiles fenotípicos y farmacológicos TOML.

pub mod schema;
pub use schema::OrganismConfig;

use std::fs;
use std::path::Path;

pub fn load_organism_config<P: AsRef<Path>>(path: P) -> Result<OrganismConfig, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(path)?;
    let config: OrganismConfig = toml::from_str(&content)?;
    Ok(config)
}