//! src/physics/diffusion.rs
//! Transporte bidimensional de Fick: Difusión continua estándar y difusión interfacial para Placa MEGA.

use crate::core::grid::Grid;

#[derive(Clone, Debug)]
pub struct DiffusionParams {
    pub diffusion_coeff: f64, // D (cm^2/min o px^2/tick)
    pub clearance_rate: f64,  // k_e (degradación pasiva / aclaramiento tisular)
    pub dx: f64,              // Espaciado de malla h
    pub dt: f64,              // Paso temporal dt
}

impl Default for DiffusionParams {
    fn default() -> Self {
        Self {
            diffusion_coeff: 0.12,
            clearance_rate: 0.001,
            dx: 1.0,
            dt: 1.0,
        }
    }
}

/// 1. Difusión continua estándar de Fick en 2D (Kirby-Bauer radial y tejido vascularizado)
/// Cumple estrictamente el criterio CFL: alpha = (D * dt) / h^2 <= 0.24 para estabilidad numérica.
pub fn step_diffusion_2d(current: &Grid<f64>, next: &mut Grid<f64>, params: &DiffusionParams) {
    let w = current.width;
    let h = current.height;
    let alpha = ((params.diffusion_coeff * params.dt) / (params.dx * params.dx)).min(0.24);
    let decay = (-params.clearance_rate * params.dt).exp();

    for y in 0..h {
        let y_idx = y * w;
        let y_top = if y > 0 { (y - 1) * w } else { y_idx };
        let y_bot = if y + 1 < h { (y + 1) * w } else { y_idx };

        for x in 0..w {
            let x_left = if x > 0 { x - 1 } else { x };
            let x_right = if x + 1 < w { x + 1 } else { x };

            let c_center = current.data[y_idx + x];
            let laplacian = current.data[y_top + x]
                + current.data[y_bot + x]
                + current.data[y_idx + x_left]
                + current.data[y_idx + x_right]
                - 4.0 * c_center;

            next.data[y_idx + x] = ((c_center + alpha * laplacian) * decay).max(0.0);
        }
    }
}

/// 2. Difusión interfacial amortiguada para la Placa MEGA (Baym et al., 2016)
/// Preserva la concentración nuclear de cada meseta y restringe la difusión a +/- 2 px en las fronteras.
pub fn step_diffusion_mega_plate(
    current: &Grid<f64>,
    next: &mut Grid<f64>,
    reservoirs: &[f64],
    num_zones: usize,
    interface_damping: f64,
) {
    let w = current.width;
    let h = current.height;
    let zone_w = w / num_zones;

    for y in 0..h {
        let y_idx = y * w;
        let y_top = if y > 0 { (y - 1) * w } else { y_idx };
        let y_bot = if y + 1 < h { (y + 1) * w } else { y_idx };

        for x in 0..w {
            let zone_idx = (x / zone_w).min(num_zones - 1);
            let dist_to_boundary = (x % zone_w).min(zone_w - (x % zone_w));

            if dist_to_boundary > 2 {
                next.data[y_idx + x] = reservoirs[zone_idx];
                continue;
            }

            let x_left = if x > 0 { x - 1 } else { x };
            let x_right = if x + 1 < w { x + 1 } else { x };

            let c_center = current.data[y_idx + x];
            let laplacian = current.data[y_top + x]
                + current.data[y_bot + x]
                + current.data[y_idx + x_left]
                + current.data[y_idx + x_right]
                - 4.0 * c_center;

            next.data[y_idx + x] = (c_center + interface_damping * laplacian).max(0.0);
        }
    }
}