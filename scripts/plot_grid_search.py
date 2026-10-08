#!/usr/bin/env python3
"""
scripts/plot_grid_search.py
Mapa de calor de probabilidad de fracaso terapéutico (P_RAM) por pauta y adherencia.
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

# P_RAM = Fracción de réplicas que concluyen en FRACASO_RAM
df["is_ram"] = (df["treatment_outcome"] == "FRACASO_RAM").astype(float)

# Agrupación por Pauta Posológica (tau) vs Dosis Tomadas Antes del Abandono
heatmap_data = df.groupby(["tau_regimen", "doses_before_drop"])["is_ram"].mean().unstack()

# Etiquetado clínico formal de las pautas
regimen_labels = {
    480: "q8h (Cada 8 horas)",
    720: "q12h (Cada 12 horas)",
    1440: "q24h (Cada 24 horas)"
}
heatmap_data.index = [regimen_labels.get(i, f"{i} min") for i in heatmap_data.index]

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
plt.ylabel(r"Pauta Posológica Interdosis ($\tau$)", fontsize=11, fontweight="bold")
plt.title("Espacio de Fases: Emergencia de Resistencia (Amoxicilina + Clavulánico)", fontsize=12, fontweight="bold", pad=12)

# Colocar q8h en la base y q24h en la parte superior
plt.gca().invert_yaxis()

plt.tight_layout()
out_pdf = "plots/figura_3_grid_search_adherencia.pdf"
out_png = "plots/figura_3_grid_search_adherencia.png"
plt.savefig(out_pdf)
plt.savefig(out_png)
plt.close()
print(f"Heatmap guardado exitosamente: {out_pdf} y {out_png}")