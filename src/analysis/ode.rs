//! src/analysis/ode.rs
//! Modelo continuo de campo medio isomórfico al Autómata Celular (Levin-Regoes).

#[derive(Clone, Copy, Debug)]
pub struct ChemostatState {
    pub s: f64,   // Parámetro de compatibilidad
    pub n_s: f64, // Células sensibles
    pub n_r: f64, // Células resistentes
}

#[derive(Clone, Copy, Debug)]
pub struct MultiClonalState {
    pub n: [f64; 5],
}

impl MultiClonalState {
    #[inline]
    pub fn total(&self) -> f64 {
        self.n[0] + self.n[1] + self.n[2] + self.n[3] + self.n[4]
    }
}

#[derive(Clone, Debug)]
pub struct LevinMonodParams {
    pub carrying_capacity: f64,
    pub growth_rate_s: f64,     // r_s basal
    pub growth_rate_r: f64,     // r_r con coste metabólico
    pub dilution_d: f64,        // Tasa de lavado D
    pub e_max: f64,             // Lisis de Hill
    pub mic_s: f64,             // CMI salvaje
    pub mic_r: f64,             // CMI resistente
    pub hill_h: f64,            // Coeficiente H
    pub mu_sos: f64,            // Flujo mutacional S -> R
}

impl Default for LevinMonodParams {
    fn default() -> Self {
        Self {
            carrying_capacity: 22200.0,
            growth_rate_s: 0.145,
            growth_rate_r: 0.1334,
            dilution_d: 0.015,
            e_max: 0.35,
            mic_s: 0.035,
            mic_r: 0.200,
            hill_h: 2.0,
            mu_sos: 0.0035,
        }
    }
}

impl LevinMonodParams {
    #[inline]
    fn hill_kill(&self, c: f64, mic: f64) -> f64 {
        if c < mic {
            return 0.0;
        }
        let ch = c.powf(self.hill_h);
        let mh = mic.powf(self.hill_h);
        self.e_max * (ch / (mh + ch))
    }

    pub fn derivatives(&self, state: &ChemostatState, c: f64) -> ChemostatState {
        let total_n = state.n_s + state.n_r;
        let logistic = (1.0 - (total_n / self.carrying_capacity)).max(0.0);

        let phi_s = if c < self.mic_s {
            self.growth_rate_s * logistic
        } else {
            0.0
        };
        let phi_r = if c < self.mic_r {
            self.growth_rate_r * logistic
        } else {
            0.0
        };

        let kill_s = self.hill_kill(c, self.mic_s);
        let kill_r = self.hill_kill(c, self.mic_r);

        // En el CA existe mutación espontánea basal (sos_base_mutation_prob * (1 - lethal) = 0.001)
        // que se amplifica bajo inducción SOS dentro de la MSW [CMI_s, MPC_s].
        let mu = if c >= self.mic_s && c < self.mic_r * 1.4 {
            self.mu_sos
        } else {
            0.0008 * logistic
        };

        let dn_s = (phi_s - kill_s - self.dilution_d) * state.n_s - mu * state.n_s;
        let dn_r = (phi_r - kill_r - self.dilution_d) * state.n_r + mu * state.n_s;

        ChemostatState {
            s: 0.0,
            n_s: dn_s,
            n_r: dn_r,
        }
    }

    pub fn rk4_step(&self, state: &ChemostatState, c: f64, dt: f64) -> ChemostatState {
        let k1 = self.derivatives(state, c);

        let s2 = ChemostatState {
            s: 0.0,
            n_s: (state.n_s + 0.5 * dt * k1.n_s).max(0.0),
            n_r: (state.n_r + 0.5 * dt * k1.n_r).max(0.0),
        };
        let k2 = self.derivatives(&s2, c);

        let s3 = ChemostatState {
            s: 0.0,
            n_s: (state.n_s + 0.5 * dt * k2.n_s).max(0.0),
            n_r: (state.n_r + 0.5 * dt * k2.n_r).max(0.0),
        };
        let k3 = self.derivatives(&s3, c);

        let s4 = ChemostatState {
            s: 0.0,
            n_s: (state.n_s + dt * k3.n_s).max(0.0),
            n_r: (state.n_r + dt * k3.n_r).max(0.0),
        };
        let k4 = self.derivatives(&s4, c);

        ChemostatState {
            s: 0.0,
            n_s: (state.n_s + (dt / 6.0) * (k1.n_s + 2.0 * k2.n_s + 2.0 * k3.n_s + k4.n_s)).max(0.0),
            n_r: (state.n_r + (dt / 6.0) * (k1.n_r + 2.0 * k2.n_r + 2.0 * k3.n_r + k4.n_r)).max(0.0),
        }
    }

    /// Derivadas del sistema multiclonal (g0..g4) acopladas al protocolo Morbidostat
    pub fn derivatives_multiclonal(
        &self,
        state: &MultiClonalState,
        c: f64,
        mics: &[f64; 5],
        mpcs: &[f64; 5],
        fitness_cost: f64,
    ) -> MultiClonalState {
        let total_n = state.total();
        let logistic = (1.0 - (total_n / self.carrying_capacity)).max(0.0);
        let mut dn = [0.0f64; 5];

        for g in 0..5 {
            let fitness = (1.0 - fitness_cost * (g as f64)).max(0.1);
            let r_g = self.growth_rate_s * fitness;
            let phi_g = if c < mics[g] { r_g * logistic } else { 0.0 };
            let kill_g = self.hill_kill(c, mics[g]);

            let mu_out = if g < 4 {
                if c >= mics[g] && c < mpcs[g] {
                    self.mu_sos
                } else if c < mics[g] {
                    0.0008 * logistic
                } else {
                    0.0
                }
            } else {
                0.0
            };

            let mut_in = if g > 0 {
                let g_prev = g - 1;
                let mu_prev = if c >= mics[g_prev] && c < mpcs[g_prev] {
                    self.mu_sos
                } else if c < mics[g_prev] {
                    0.0008 * logistic
                } else {
                    0.0
                };
                mu_prev * state.n[g_prev]
            } else {
                0.0
            };

            dn[g] = (phi_g - kill_g - self.dilution_d - mu_out) * state.n[g] + mut_in;
        }

        MultiClonalState { n: dn }
    }

    /// Paso Runge-Kutta 4 (RK4) para las 5 subpoblaciones clonales (g0..g4)
    pub fn rk4_step_multiclonal(
        &self,
        state: &MultiClonalState,
        c: f64,
        mics: &[f64; 5],
        mpcs: &[f64; 5],
        fitness_cost: f64,
        dt: f64,
    ) -> MultiClonalState {
        let k1 = self.derivatives_multiclonal(state, c, mics, mpcs, fitness_cost);

        let mut s2 = MultiClonalState { n: [0.0; 5] };
        for i in 0..5 {
            s2.n[i] = (state.n[i] + 0.5 * dt * k1.n[i]).max(0.0);
        }
        let k2 = self.derivatives_multiclonal(&s2, c, mics, mpcs, fitness_cost);

        let mut s3 = MultiClonalState { n: [0.0; 5] };
        for i in 0..5 {
            s3.n[i] = (state.n[i] + 0.5 * dt * k2.n[i]).max(0.0);
        }
        let k3 = self.derivatives_multiclonal(&s3, c, mics, mpcs, fitness_cost);

        let mut s4 = MultiClonalState { n: [0.0; 5] };
        for i in 0..5 {
            s4.n[i] = (state.n[i] + dt * k3.n[i]).max(0.0);
        }
        let k4 = self.derivatives_multiclonal(&s4, c, mics, mpcs, fitness_cost);

        let mut next = MultiClonalState { n: [0.0; 5] };
        for i in 0..5 {
            next.n[i] = (state.n[i]
                + (dt / 6.0) * (k1.n[i] + 2.0 * k2.n[i] + 2.0 * k3.n[i] + k4.n[i]))
                .max(0.0);
        }
        next
    }
}

pub struct PearsonTracker {
    pub n: f64,
    pub sum_x: f64,
    pub sum_y: f64,
    pub sum_xy: f64,
    pub sum_x2: f64,
    pub sum_y2: f64,
    pub sum_sq_err: f64,
}

impl Default for PearsonTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl PearsonTracker {
    pub fn new() -> Self {
        Self {
            n: 0.0,
            sum_x: 0.0,
            sum_y: 0.0,
            sum_xy: 0.0,
            sum_x2: 0.0,
            sum_y2: 0.0,
            sum_sq_err: 0.0,
        }
    }

    pub fn record(&mut self, x: f64, y: f64) {
        self.n += 1.0;
        self.sum_x += x;
        self.sum_y += y;
        self.sum_xy += x * y;
        self.sum_x2 += x * x;
        self.sum_y2 += y * y;
        let diff = x - y;
        self.sum_sq_err += diff * diff;
    }

    pub fn calculate(&self) -> f64 {
        if self.n < 2.0 {
            return 1.0;
        }
        let num = self.n * self.sum_xy - self.sum_x * self.sum_y;
        let den_x = self.n * self.sum_x2 - self.sum_x * self.sum_x;
        let den_y = self.n * self.sum_y2 - self.sum_y * self.sum_y;
        let den = (den_x * den_y).sqrt();
        if den <= 1e-12 {
            1.0
        } else {
            (num / den).clamp(-1.0, 1.0)
        }
    }

    pub fn rmse(&self) -> f64 {
        if self.n < 1.0 {
            0.0
        } else {
            (self.sum_sq_err / self.n).sqrt()
        }
    }

    pub fn nrmse(&self, scale: f64) -> f64 {
        if scale <= 1e-12 {
            0.0
        } else {
            self.rmse() / scale
        }
    }
}