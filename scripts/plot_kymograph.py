#!/usr/bin/env python3
"""
scripts/plot_kymograph.py
Genera el Kymograph Espaciotemporal de la Placa MEGA (Baym et al., Science 2016).
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

os.makedirs("plots", exist_ok=True)
data_file = "./../data/headless/mega_plate_kymograph.csv"

if not os.path.exists(data_file):
    print(f"Error: {data_file} no encontrado. Ejecuta cargo run --release --bin headless_runner primero.")
    exit(1)

df = pd.read_csv(data_file)
ticks = df["tick"].values
matrix_cols = [c for c in df.columns if c.startswith("x_")]
kymo_matrix = df[matrix_cols].values # Matriz [tiempos x 1000 columnas]

# Colores de genotipo: -1 = Agar sin bacterias, 0 = g0 (verde), 1 = g1 (amarillo), 2 = g2 (naranja), 3 = g3 (rojo), 4 = g4 (púrpura)
cmap_colors = ["#0F172A", "#22C55E", "#EAB308", "#F97316", "#EF4444", "#A855F7"]
cmap = ListedColormap(cmap_colors)

plt.figure(figsize=(12, 6), dpi=300)
im = plt.imshow(
    kymo_matrix,
    aspect="auto",
    origin="lower",
    cmap=cmap,
    vmin=-1,
    vmax=4,
    extent=[0, 1000, ticks[0], ticks[-1]]
)

# Líneas divisorias de las mesetas de concentración
for zone_x in [200, 400, 600, 800]:
    plt.axvline(x=zone_x, color="#64748B", linestyle="--", linewidth=1.2, alpha=0.8)

# Rótulos de mesetas
plt.text(100, ticks[-1] * 0.95, "0x CMI", color="white", ha="center", weight="bold")
plt.text(300, ticks[-1] * 0.95, "3x CMI", color="white", ha="center", weight="bold")
plt.text(500, ticks[-1] * 0.95, "30x CMI", color="white", ha="center", weight="bold")
plt.text(700, ticks[-1] * 0.95, "300x CMI", color="white", ha="center", weight="bold")
plt.text(900, ticks[-1] * 0.95, "3000x CMI", color="white", ha="center", weight="bold")

plt.xlabel("Coordenada Espacial en Placa MEGA (px)", fontsize=11, fontweight="bold")
plt.ylabel("Tiempo de Incubación (ticks / minutos)", fontsize=11, fontweight="bold")
plt.title("Radiación Adaptativa Espaciotemporal (Kymograph de Baym et al. 2016)", fontsize=13, fontweight="bold", pad=12)

# Leyenda clonal
patches = [
    mpatches.Patch(color="#22C55E", label="g0 (Salvaje)"),
    mpatches.Patch(color="#EAB308", label="g1 (1° Salto)"),
    mpatches.Patch(color="#F97316", label="g2 (2° Salto)"),
    mpatches.Patch(color="#EF4444", label="g3 (3° Salto)"),
    mpatches.Patch(color="#A855F7", label="g4 (Resistente 3000x)"),
]
plt.legend(handles=patches, loc="upper left", framealpha=0.9, facecolor="#1E293B", labelcolor="white")

plt.tight_layout()
out_pdf = "plots/figura_1_mega_plate_kymograph.pdf"
out_png = "plots/figura_1_mega_plate_kymograph.png"
plt.savefig(out_pdf)
plt.savefig(out_png)
plt.close()
print(f"Kymograph guardado: {out_pdf} y {out_png}")