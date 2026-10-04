//! src/visualizer/osd.rs
//! Primitivas gráficas para la interfaz en pantalla (OSD) sobre el framebuffer de minifb.

use super::font::get_glyph_8x8;

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

pub fn draw_text(fb: &mut [u32], pitch: usize, x: usize, y: usize, text: &str, color: u32) {
    let mut cx = x;
    for ch in text.chars() {
        let ascii = ch as usize;
        let glyph = get_glyph_8x8(ascii);
        for row in 0..8 {
            let py = y + row;
            if py >= fb.len() / pitch {
                break;
            }
            let byte = glyph[row];
            for col in 0..8 {
                let px = cx + col;
                if px >= pitch {
                    break;
                }
                if (byte >> (7 - col)) & 1 == 1 {
                    fb[py * pitch + px] = color;
                }
            }
        }
        cx += 8;
        if cx + 8 > pitch {
            break;
        }
    }
}