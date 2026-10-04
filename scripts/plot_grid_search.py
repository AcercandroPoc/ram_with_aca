#!/usr/bin/env python3
"""
scripts/plot_grid_search.py
Mapa de calor de probabilidad de fracaso terapéutico (P_RAM) por falta de adherencia.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("plots", exist_ok=True)
data_file = "./../data/headless/grid_search_adherence.csv"

if not os.path.exists(data_file):
    print(f"Error: {data_file} no encontrado.")
    exit(1)

df = pd.read_csv(data_file)

# P_RAM = Fracción de réplicas que terminan en FRACASO_RAM
df["is_ram"] = (df["treatment_outcome"] == "FRACASO_RAM").astype(float)

heatmap_data = df.groupby(["prior_cycles", "doses_before_drop"])["is_ram"].mean().unstack()

plt.figure(figsize=(10, 4.8), dpi=300)
ax = sns.heatmap(
    heatmap_data,
    cmap="YlOrRd",
    vmin=0.0,
    vmax=1.0,
    annot=True,
    fmt=".2f",
    cbar_kws={'label': 'Probabilidad de Emergencia RAM ($P_{\\mathrm{RAM}}$)'},
    linewidths=0.6,
    linecolor="white"
)

plt.xlabel("Dosis Tomadas Antes del Abandono Terapéutico", fontsize=11, fontweight="bold")
plt.ylabel("Ciclos Previos de Recidiva / Automedicación", fontsize=11, fontweight="bold")
plt.title("Espacio de Fases de Adherencia: Emergencia de Resistencia in vivo", fontsize=12, fontweight="bold", pad=12)

# Invertir eje Y para que Ciclo 0 quede en la base
plt.gca().invert_yaxis()

plt.tight_layout()
out_pdf = "plots/figura_3_grid_search_adherencia.pdf"
out_png = "plots/figura_3_grid_search_adherencia.png"
plt.savefig(out_pdf)
plt.savefig(out_png)
plt.close()
print(f"Heatmap de adherencia guardado: {out_pdf} y {out_png}")