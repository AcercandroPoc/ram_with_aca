//! src/analysis/ode.rs
//! Modelo continuo de campo medio isomórfico al Autómata Celular (Levin-Regoes).

#[derive(Clone, Debug)]
pub struct LevinMonodParams {
    pub carrying_capacity: f64, // K = 30,490 (Celdas del reactor cilíndrico)
    pub growth_rate_s: f64,     // r_s basal
    pub growth_rate_r: f64,     // r_r con coste metabólico
    pub dilution_d: f64,        // Tasa de lavado D
    pub e_max: f64,             // Lisis de Hill
    pub mic_s: f64,             // CMI salvaje
    pub mic_r: f64,             // CMI resistente
    pub hill_h: f64,            // Coeficiente H
    pub mu_sos: f64,            // Flujo mutacional S -> R
}

#[derive(Clone, Copy, Debug)]
pub struct ChemostatState {
    pub s: f64,   // Parámetro de compatibilidad
    pub n_s: f64, // Células sensibles
    pub n_r: f64, // Células resistentes
}

impl LevinMonodParams {
    pub fn derivatives(&self, state: &ChemostatState, antibiotic_conc: f64) -> [f64; 3] {
        let n_s = state.n_s.max(0.0);
        let n_r = state.n_r.max(0.0);
        let n_tot = n_s + n_r;

        // Factor logístico espacial idéntico a la saturación Chamfer del retículo
        let space_factor = (1.0 - n_tot / self.carrying_capacity).max(0.0);

        let kill_s = if antibiotic_conc >= self.mic_s {
            self.e_max * antibiotic_conc.powf(self.hill_h)
                / (self.mic_s.powf(self.hill_h) + antibiotic_conc.powf(self.hill_h))
        } else {
            0.0
        };

        let kill_r = if antibiotic_conc >= self.mic_r {
            self.e_max * antibiotic_conc.powf(self.hill_h)
                / (self.mic_r.powf(self.hill_h) + antibiotic_conc.powf(self.hill_h))
        } else {
            0.0
        };

        let dns = n_s * (self.growth_rate_s * space_factor - self.dilution_d - kill_s - self.mu_sos);
        let dnr = n_r * (self.growth_rate_r * space_factor - self.dilution_d - kill_r) + self.mu_sos * n_s;

        [0.0, dns, dnr]
    }

    pub fn rk4_step(&self, state: &ChemostatState, antibiotic_conc: f64, dt: f64) -> ChemostatState {
        let k1 = self.derivatives(state, antibiotic_conc);

        let s1 = ChemostatState {
            s: 0.0,
            n_s: state.n_s + 0.5 * dt * k1[1],
            n_r: state.n_r + 0.5 * dt * k1[2],
        };
        let k2 = self.derivatives(&s1, antibiotic_conc);

        let s2 = ChemostatState {
            s: 0.0,
            n_s: state.n_s + 0.5 * dt * k2[1],
            n_r: state.n_r + 0.5 * dt * k2[2],
        };
        let k3 = self.derivatives(&s2, antibiotic_conc);

        let s3 = ChemostatState {
            s: 0.0,
            n_s: state.n_s + dt * k3[1],
            n_r: state.n_r + dt * k3[2],
        };
        let k4 = self.derivatives(&s3, antibiotic_conc);

        ChemostatState {
            s: 0.0,
            n_s: (state.n_s + (dt / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])).max(0.0),
            n_r: (state.n_r + (dt / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])).max(0.0),
        }
    }
}

pub struct PearsonTracker {
    x_history: Vec<f64>,
    y_history: Vec<f64>,
}

impl Default for PearsonTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl PearsonTracker {
    pub fn new() -> Self {
        Self {
            x_history: Vec::with_capacity(10000),
            y_history: Vec::with_capacity(10000),
        }
    }

    pub fn record(&mut self, x: f64, y: f64) {
        self.x_history.push(x);
        self.y_history.push(y);
    }

    /// Evalúa la correlación de Pearson a lo largo de toda la trayectoria histórica transitoria
    pub fn calculate(&self) -> f64 {
        let n = self.x_history.len();
        if n < 15 {
            return 1.0;
        }

        let mean_x = self.x_history.iter().sum::<f64>() / n as f64;
        let mean_y = self.y_history.iter().sum::<f64>() / n as f64;

        let mut cov = 0.0;
        let mut var_x = 0.0;
        let mut var_y = 0.0;

        for (xi, yi) in self.x_history.iter().zip(self.y_history.iter()) {
            let dx = xi - mean_x;
            let dy = yi - mean_y;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }

        if var_x <= 1e-6 || var_y <= 1e-6 {
            return 1.0;
        }

        cov / (var_x.sqrt() * var_y.sqrt())
    }
}