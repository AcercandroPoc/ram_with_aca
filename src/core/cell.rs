//! Representación de bajo nivel de la celda bacteriana individual.
//! Empaquetado estricto en 16 bits para optimización de caché L1/L2.

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cell {
    /// Bits 0..=2 (0x07): Genotipo de resistencia (0..=4)
    /// Bit  3     (0x08): Flag de Persistencia Fenotípica (0 = Activa, 1 = Durmiente/Persister)
    /// Bits 4..=7 (0xF0): Acumulador de Estrés SOS (0..=15)
    pub state: u8,
    /// Pool de recursos bioenergéticos internos (0..=255)
    pub resource: u8,
}

impl Cell {
    pub const EMPTY: Self = Cell { state: 0, resource: 0 };

    const GENOTYPE_MASK: u8 = 0x07;
    const PERSISTER_MASK: u8 = 0x08;
    const STRESS_MASK: u8 = 0xF0;

    #[inline(always)]
    pub fn new(genotype: u8, stress: u8, resource: u8) -> Self {
        let g = genotype.min(4) & Self::GENOTYPE_MASK;
        let s = (stress.min(15) & 0x0F) << 4;
        Cell {
            state: s | g,
            resource,
        }
    }

    #[inline(always)]
    pub fn is_alive(&self) -> bool {
        self.resource > 0
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.resource == 0
    }

    #[inline(always)]
    pub fn genotype(&self) -> u8 {
        self.state & Self::GENOTYPE_MASK
    }

    #[inline(always)]
    pub fn stress(&self) -> u8 {
        (self.state >> 4) & 0x0F
    }

    #[inline(always)]
    pub fn is_persister(&self) -> bool {
        (self.state & Self::PERSISTER_MASK) != 0
    }

    #[inline(always)]
    pub fn set_persister(&mut self, persister: bool) {
        if persister {
            self.state |= Self::PERSISTER_MASK;
        } else {
            self.state &= !Self::PERSISTER_MASK;
        }
    }

    #[inline(always)]
    pub fn set_stress(&mut self, s: u8) {
        let clean_s = (s.min(15) & 0x0F) << 4;
        self.state = (self.state & !Self::STRESS_MASK) | clean_s;
    }

    #[inline(always)]
    pub fn set_genotype(&mut self, g: u8) {
        let clean_g = g.min(4) & Self::GENOTYPE_MASK;
        self.state = (self.state & !Self::GENOTYPE_MASK) | clean_g;
    }

    #[inline(always)]
    pub fn inc_stress(&mut self) {
        let current = self.stress();
        if current < 15 {
            self.set_stress(current + 1);
        }
    }

    #[inline(always)]
    pub fn dec_stress(&mut self) {
        let current = self.stress();
        if current > 0 {
            self.set_stress(current - 1);
        }
    }

    #[inline(always)]
    pub fn consume_resource(&mut self, cost: u8) -> bool {
        if self.resource >= cost {
            self.resource -= cost;
            true
        } else {
            self.resource = 0;
            false
        }
    }

    #[inline(always)]
    pub fn add_resource(&mut self, amount: u8) {
        self.resource = self.resource.saturating_add(amount);
    }
}