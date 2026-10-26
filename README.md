# 🦠 RAM con ACA — Simulador Biofísico Multiescala de Resistencia a los Antimicrobianos

[![Rust](https://img.shields.io/badge/Rust-2024_Edition-orange?logo=rust)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![HPC](https://img.shields.io/badge/HPC-Rayon_Parallel-success)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

Un simulador biofísico multiescala de alto rendimiento escrito en **Rust** que modela la aparición y fijación espaciotemporal de la **Resistencia a los Antimicrobianos (RAM / AMR)**.

Combina ecuaciones diferenciales parciales macroscópicas continuas (difusión de Fick atenuada por biopelícula en 2D + farmacocinética normalizada de Bateman) con un **Autómata Celular Estocástico Adaptativo (ACA)** en 2D síncrono y discreto, impulsado por celdas empaquetadas en bits de 16 bits, esténciles isotrópicos Chamfer 5-7-11 y multihilos sin bloqueos (lock-free).

---

## Características Principales

- **Diseño de Alto Rendimiento Orientado a Datos:** Cada célula bacteriana se empaqueta en **16 bits (2 bytes)** (`state: u8` para el genotipo `g en 0..=4` y estrés SOS `s en 0..=15`; `resource: u8` para el fondo metabólico). Una cuadrícula de 1,000,000 de nodos cabe enteramente dentro de la caché L3 de la CPU, logrando **~8.5 ms por paso** con búferes dobles de intercambio de punteros en `O(1)`.
- **Biofísica Multiescala:**
  - **Difusión en Biopelícula 2D:** Solucionador explícito de diferencias finitas que incorpora atenuación por tortuosidad de la biopelícula (factor de `0.30`) e hidrólisis enzimática local por betalactamasa (`enzymatic_deg_rate`).
  - **Farmacocinética Exacta de Bateman:** Absorción oral de dosis múltiples y dinámicas de eliminación sistémica normalizadas en `t_max` para coincidir con las concentraciones plasmáticas máximas clínicas (`C_max`).
  - **Regulón SOS y Ventana de Selección Mutantes (MSW):** Mutagénesis adaptativa dependiente de la concentración delimitada por los umbrales MIC y MPC de EUCAST v16.0 / CLSI M100, acoplada con lisis bactericida de Hill y quiescencia por contacto.
- **Reportes de Cascada y AWaRe de la OMS:** Evalúa **3 patógenos prioritarios ESKAPE** (*E. coli* ATCC 25922, *K. pneumoniae* ESBL+, *P. aeruginosa* PAO1) a través de **8 antibióticos por cepa**, divididos en un **Panel Primario (1.ª Línea)** y un **Panel de Rescate de Reserva de la OMS (2.ª Línea)**.
- **5 Entornos de Prueba Interactivos en Tiempo Real + 1 Motor HPC Desatendido:** Visualización en vivo a 60 FPS (`minifb`) con pastilleros virtuales interactivos, extracción de biopsias, cambio de paneles de cascada y exportación de telemetría en CSV bajo demanda.

---

## Inicio Rápido

### 1. Prerrequisitos

- **Herramientas de Rust:** `rustc >= 1.80` (se recomienda la edición 2024).
- **Dependencias del Sistema (solo Linux / WSL):** Requeridas por `minifb` para el renderizado de ventanas X11:

```bash
sudo apt-get update
sudo apt-get install -y build-essential libx11-dev libxext-dev
```

- **Python 3.10+ (Para postprocesamiento estadístico y gráficos):**

```bash
pip install numpy pandas matplotlib seaborn qrcode[pil]
```

### 2. Compilación (Perfil Release)

Compile siempre con `--release` para habilitar la vectorización automática SIMD y la optimización de enlace de tiempo completo (`lto = "fat"`):

```bash
git clone [https://github.com/AcercandroPoc/ram_with_aca.git](https://github.com/AcercandroPoc/ram_with_aca.git)
cd ram_with_aca
cargo build --release
```

---

## Suite de Simulación Interactiva

Ejecute cualquiera de los 5 simuladores gráficos directamente a través de Cargo:

### 1. Difusión en Disco Kirby-Bauer y Reportes en Cascada de la OMS

Simula una placa de Petri de 90 mm con 4 campos de difusión independientes para prevenir artefactos en los límites de Voronoi.

```bash
cargo run --release --bin 1_antibiograma
```

- **`[1]` `[2]` `[3]`**: Cambiar la cepa bacteriana (*E. coli*, *K. pneumoniae* ESBL+, *P. aeruginosa* PAO1).
- **`[G]`**: Alternar el genotipo del inóculo inicial (`g0` a `g4`).
- **`Clic Izquierdo` (en la placa de Petri)**: Alternar entre el **Panel Primario de 1.ª Línea** y el **Panel de Rescate de Reserva de la OMS** (`CZA`, `AMK`, `TGC`/`ATM`, `COL`).

### 2. Quimiostato Continuo y Morbidostato de 4 Pasos (Validación ODE)

Compara el autómata espacial con un solucionador de Runge-Kutta de 4.° orden (RK4) del sistema ODE multiclon de Levin-Regoes, rastreando la `r` de Pearson y el `NRMSE (%)` en tiempo real.

```bash
cargo run --release --bin 2_quimiostato_ode
```

- **`[M]`**: Alternar entre **Modo 1 (Quimiostato clásico de 1 dosis, `g0 -> g1`)** y **Modo 2 (Morbidostato escalonado de 4 dosis, `g0 -> g4`)**.

### 3. MEGA-Placa de Harvard (Evolución Espaciotemporal)

Emula el *Microbial Evolution and Growth Arena* (Baym et al., *Science* 2016) a través de pasos logarítmicos de antibióticos (`0x` a `3000x MIC_0`).

```bash
cargo run --release --bin 3_placa_mega
```

- **`[1]`**: Modo 1 — Escalera gradual completa (`0x -> 3x -> 30x -> 300x -> 3000x`).
- **`[2]` `[3]` `[4]`**: Modos 2 a 4 — Controles de barrera supercrítica abrupta (`C > MPC`).

### 4. Tejido Vascularizado In Vivo y Adherencia del Paciente

Modela un lecho tisular de `448 x 448` irrigado por 64 capilares de Krogh bajo dosificación oral intermitente.

```bash
cargo run --release --bin 4_tratamiento_clinico
```

- **`[1]` `[2]` `[3]`**: Seleccionar la especie bacteriana.
- **`[A]`**: Recorrer los 8 antibióticos disponibles.
- **`[T]`**: Cambiar el intervalo de dosificación (`q8h`, `q12h`, `q24h`).
- **`[G]`**: Alternar el genotipo de infección inicial (`g0` silvestre frente a `g1` preexpuesto).
- **`[O]`**: Omitir la siguiente dosis programada en el pastillero virtual.
- **`[E]`**: Exportar la serie temporal clínica actual a `data/interactive/*.csv`.

### 5. Ciclo Traslacional de la Cama al Laboratorio

Máquina de estados de dos fases que conecta el fracaso terapéutico in vivo con las pruebas de susceptibilidad diagnóstica in vitro.

```bash
cargo run --release --bin 5_clinica_a_laboratorio
```

- **`[B]`**: Extraer una biopsia de tejido estocástica de la Fase 1 (*In Vivo*) e inocular la Fase 2 (*Antibiograma In Vitro*).
- **`Clic Izquierdo` (en la Fase 2)**: Cambiar entre el Panel Primario y el Panel de Rescate de la OMS.
- **`[E]`**: Exportar la instantánea traslacional (`invivo` o `invitro`) a `data/interactive/*.csv`.

> **Controles Globales (Todos los Simuladores):** **`[Espacio]`** Pausa / Reanudar · **`[R]`** Reiniciar simulación · **`[Esc]`** Salir de la ventana.

---

## Ejecutor HPC Desatendido y Canal de Gráficos

Para ejecutar las exploraciones factoriales completas de Monte Carlo en paralelo a través de todos los núcleos de la CPU utilizando **Rayon** (incluida la búsqueda en cuadrícula de adherencia multi-cepa de 1,728 ejecuciones):

```bash
cargo run --release --bin headless_runner
```

Esto genera los conjuntos de datos principales en `data/headless/`:
- `quimiostato_mc.csv` y `quimiostato_morbidostat_4dosis.csv`
- `mega_plate_kymograph.csv` y `mega_plate_spatiotemporal.csv`
- `grid_search_adherence.csv`
- `traslacional_biopsia.csv`

### Generar Figuras de Alta Resolución (300 DPI PDF/PNG)

Ejecute los scripts de postprocesamiento de Python para construir todos los gráficos analíticos dentro de `plots/`:

```bash
python scripts/plot_quimiostato.py
python scripts/plot_kymograph.py
python scripts/plot_grid_search.py
python scripts/plot_antibiograma.py
python scripts/plot_caso_clinico.py
```

---

## Estructura del Proyecto

```text
ram_with_aca/
├── Cargo.toml               # Dependencias y optimizaciones de compilación (LTO fat, opt-level 3)
├── config/                  # Perfiles TOML declarativos (3 cepas ESKAPE x 8 antibióticos)
│   ├── ecoli_atcc25922.toml
│   ├── klebsiella_blee.toml
│   └── pseudomonas_pao1.toml
├── src/
│   ├── lib.rs               # Exportaciones de la librería principal
│   ├── core/                # Celdas de 16 bits, Grid<T> contiguo, DoubleBuffer O(1), Chamfer 5-7-11
│   ├── physics/             # Difusión en biopelícula 2D + solucionador PK normalizado de Bateman
│   ├── biology/             # Lisis de Hill, quiescencia por contacto y reglas de mutagénesis SOS
│   ├── config/              # Esquemas de deserialización Serde TOML
│   ├── analysis/            # Solucionadores ODE RK4 biclonal y multiclon + Pearson/NRMSE en línea
│   ├── visualizer/          # Rasterizador de software y telemetría OSD (minifb)
│   └── bin/                 # 5 entornos interactivos + headless_runner
├── scripts/                 # Scripts de visualización y trazado estadístico en Python
├── data/                    # Conjuntos de datos CSV generados (headless/ y interactive/)
└── plots/                   # Figuras de salida listas para publicación (PDF/PNG)
```
