//! Malla bidimensional contigua y doble búfer para sincronía temporal estricta.
//! Implementa la métrica Chamfer 5-7-11 para erradicar anisotropías reticulares.

use crate::core::cell::Cell;

#[derive(Clone, Debug)]
pub struct Grid<T: Clone> {
    pub width: usize,
    pub height: usize,
    pub data: Vec<T>,
}

impl<T: Clone> Grid<T> {
    pub fn new(width: usize, height: usize, default_val: T) -> Self {
        Self {
            width,
            height,
            data: vec![default_val; width * height],
        }
    }

    #[inline(always)]
    pub fn get(&self, x: usize, y: usize) -> &T {
        &self.data[y * self.width + x]
    }

    #[inline(always)]
    pub fn get_mut(&mut self, x: usize, y: usize) -> &mut T {
        &mut self.data[y * self.width + x]
    }

    #[inline(always)]
    pub fn set(&mut self, x: usize, y: usize, val: T) {
        self.data[y * self.width + x] = val;
    }

    #[inline(always)]
    pub fn idx(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }
}

pub struct DoubleBufferGrid<T: Clone> {
    pub current: Grid<T>,
    pub next: Grid<T>,
}

impl<T: Clone> DoubleBufferGrid<T> {
    pub fn new(width: usize, height: usize, default_val: T) -> Self {
        Self {
            current: Grid::new(width, height, default_val.clone()),
            next: Grid::new(width, height, default_val),
        }
    }

    #[inline(always)]
    pub fn swap(&mut self) {
        std::mem::swap(&mut self.current, &mut self.next);
    }
}

impl Grid<Cell> {
    /// Evalúa la densidad Chamfer 5-7 local alrededor de (x, y) con fronteras de Neumann.
    /// Peso ortogonal = 7, Peso diagonal = 5.
    #[inline]
    pub fn chamfer_density_5_7(&self, x: usize, y: usize) -> u16 {
        let w = self.width as isize;
        let h = self.height as isize;
        let xi = x as isize;
        let yi = y as isize;

        let mut density = 0u16;

        // Vecinos ortogonales (peso 7)
        let ortogonales = [(0, -1), (-1, 0), (1, 0), (0, 1)];
        for (dx, dy) in ortogonales {
            let nx = (xi + dx).clamp(0, w - 1) as usize;
            let ny = (yi + dy).clamp(0, h - 1) as usize;
            if self.get(nx, ny).is_alive() {
                density += 7;
            }
        }

        // Vecinos diagonales (peso 5)
        let diagonales = [(-1, -1), (1, -1), (-1, 1), (1, 1)];
        for (dx, dy) in diagonales {
            let nx = (xi + dx).clamp(0, w - 1) as usize;
            let ny = (yi + dy).clamp(0, h - 1) as usize;
            if self.get(nx, ny).is_alive() {
                density += 5;
            }
        }

        density
    }

    /// Obtiene los vecinos vivos para competencia local y selección clonal.
    #[inline]
    pub fn get_living_neighbors(&self, x: usize, y: usize) -> Vec<(Cell, usize, usize)> {
        let w = self.width as isize;
        let h = self.height as isize;
        let xi = x as isize;
        let yi = y as isize;

        let mut neighbors = Vec::with_capacity(8);
        for dy in -1..=1 {
            for dx in -1..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let nx = (xi + dx).clamp(0, w - 1) as usize;
                let ny = (yi + dy).clamp(0, h - 1) as usize;
                let cell = *self.get(nx, ny);
                if cell.is_alive() {
                    neighbors.push((cell, nx, ny));
                }
            }
        }
        neighbors
    }
}