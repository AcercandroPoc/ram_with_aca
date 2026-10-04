//! src/visualizer/mod.rs
//! Motor de renderizado en búfer para OSD (On-Screen Display) en entornos WSL/Linux.
//! Incluye rasterizador de fuentes bitmap 8x8, cajas, líneas y barras de progreso.

pub mod font;

pub fn draw_rect(fb: &mut [u32], pitch: usize, x: usize, y: usize, w: usize, h: usize, color: u32) {
    let max_y = (y + h).min(fb.len() / pitch);
    let max_x = (x + w).min(pitch);
    for py in y..max_y {
        let offset = py * pitch;
        for px in x..max_x {
            fb[offset + px] = color;
        }
    }
}

pub fn draw_char(fb: &mut [u32], pitch: usize, x: usize, y: usize, c: char, color: u32) {
    let ascii = c as usize;
    if ascii < 32 || ascii > 126 {
        return;
    }
    let glyph = font::get_glyph_8x8(ascii);
    for row in 0..8 {
        let py = y + row;
        if py >= fb.len() / pitch {
            break;
        }
        let byte = glyph[row];
        for col in 0..8 {
            let px = x + col;
            if px >= pitch {
                break;
            }
            if (byte >> (7 - col)) & 1 == 1 {
                fb[py * pitch + px] = color;
            }
        }
    }
}

pub fn draw_text(fb: &mut [u32], pitch: usize, x: usize, y: usize, text: &str, color: u32) {
    let mut cursor_x = x;
    for c in text.chars() {
        if c == '\n' {
            break;
        }
        draw_char(fb, pitch, cursor_x, y, c, color);
        cursor_x += 8;
        if cursor_x + 8 > pitch {
            break;
        }
    }
}