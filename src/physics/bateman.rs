//! src/physics/bateman.rs
//! Farmacocinética biexponencial oral multidosis con calendario discreto y control de adherencia.

#[derive(Clone, Debug)]
pub struct BatemanRegimen {
    pub ka: f64,                // Constante de absorción (min^-1)
    pub ke: f64,                // Constante de eliminación (min^-1)
    pub c_target: f64,          // Concentración pico objetivo (ug/mL)
    pub tau_minutes: usize,     // Intervalo interdosis tau
    pub total_doses: usize,     // Número total de dosis en 5 días
    pub doses_taken: Vec<bool>, // Registro: true = tomada, false = omitida
}

impl BatemanRegimen {
    pub fn new(ka: f64, ke: f64, c_target: f64, tau_minutes: usize) -> Self {
        let total_doses = match tau_minutes {
            720 => 10,   // q12h (5 días = 10 dosis)
            1440 => 5,   // q24h (5 días = 5 dosis)
            _ => 15,     // q8h por defecto (5 días = 15 dosis)
        };

        // Por defecto, todas las dosis se inician planificadas como tomadas (adherencia 100%)
        let doses_taken = vec![true; total_doses];

        Self {
            ka,
            ke,
            c_target,
            tau_minutes,
            total_doses,
            doses_taken,
        }
    }

    /// Evalúa la concentración plasmática exacta en el minuto t mediante superposición lineal
    pub fn concentration_at(&self, t_minutes: usize) -> f64 {
        let mut total_c = 0.0;
        let prefactor = (self.c_target * self.ka) / (self.ka - self.ke);
        let max_dose_idx = (t_minutes / self.tau_minutes).min(self.total_doses.saturating_sub(1));

        for j in 0..=max_dose_idx {
            if j < self.doses_taken.len() && self.doses_taken[j] {
                let dose_time = j * self.tau_minutes;
                let dt = (t_minutes - dose_time) as f64;
                let c_pulse = prefactor * ((-self.ke * dt).exp() - (-self.ka * dt).exp());
                total_c += c_pulse.max(0.0);
            }
        }

        total_c
    }

    /// Obtiene el índice de la próxima dosis pendiente de administrar (None si terminó el tratamiento)
    pub fn next_dose_index(&self, current_tick: usize) -> Option<usize> {
        let next_idx = (current_tick / self.tau_minutes) + 1;
        if next_idx < self.total_doses {
            Some(next_idx)
        } else {
            None
        }
    }

    /// Alterna el estado de la próxima dosis (Tomada <-> Omitida) sin acumulación errática
    pub fn toggle_next_dose(&mut self, current_tick: usize) {
        if let Some(next_idx) = self.next_dose_index(current_tick) {
            self.doses_taken[next_idx] = !self.doses_taken[next_idx];
        }
    }

    /// Porcentaje de adherencia hasta el minuto actual
    pub fn adherence_pct(&self, current_tick: usize) -> f64 {
        let elapsed_doses = ((current_tick / self.tau_minutes) + 1).min(self.total_doses);
        if elapsed_doses == 0 {
            return 100.0;
        }
        let taken = self.doses_taken[..elapsed_doses].iter().filter(|&&t| t).count();
        (taken as f64 / elapsed_doses as f64) * 100.0
    }
}