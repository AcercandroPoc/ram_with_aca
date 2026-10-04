#!/usr/bin/env python3
"""
scripts/plot_antibiograma.py
Comparativa traslacional: Diámetros de inhibición Kirby-Bauer Pre vs Post-Biopsia.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("plots", exist_ok=True)
data_file = "./../data/headless/traslacional_biopsia.csv"

if not os.path.exists(data_file):
    print(f"Error: {data_file} no encontrado.")
    exit(1)

df = pd.read_csv(data_file)

# Filtrar Escherichia coli para el gráfico representativo
df_ecoli = df[df["strain"] == "Escherichia coli"].copy()

antibiotics = df_ecoli["antibiotic_code"].unique()
x = np.arange(len(antibiotics))
width = 0.35

halos_pre = []
halos_post = []
breakpoints = []

for ab in antibiotics:
    pre_val = df_ecoli[(df_ecoli["antibiotic_code"] == ab) & (df_ecoli["condition"] == "CONTROL_VIRGEN")]["halo_diameter_mm"].values[0]
    post_val = df_ecoli[(df_ecoli["antibiotic_code"] == ab) & (df_ecoli["condition"] == "POST_ABANDONO_RAM")]["halo_diameter_mm"].values[0]
    bp_val = df_ecoli[df_ecoli["antibiotic_code"] == ab]["eucast_breakpoint"].values[0]
    halos_pre.append(pre_val)
    halos_post.append(post_val)
    breakpoints.append(bp_val)

fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

rects1 = ax.bar(x - width/2, halos_pre, width, label="Control Virgen (Pre-Tratamiento)", color="#10B981")
rects2 = ax.bar(x + width/2, halos_post, width, label="Post-Abandono RAM (Biopsia)", color="#EF4444")

# Marcar puntos de corte de EUCAST
for i, bp in enumerate(breakpoints):
    ax.hlines(bp, x[i] - width*0.8, x[i] + width*0.8, colors="#1E293B", linestyles="--", linewidth=1.5)
    ax.text(x[i], bp + 0.8, f"Corte: {bp} mm", ha="center", fontsize=8, color="#1E293B", weight="bold")

ax.set_ylabel("Diámetro del Halo de Inhibición (mm)", fontsize=10, fontweight="bold")
ax.set_title("Ciclo Traslacional: Contracción de Halos por Selección in vivo (E. coli)", fontsize=11, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(antibiotics, fontsize=10, fontweight="bold")
ax.grid(axis="y", linestyle=":", alpha=0.7)
ax.set_ylim(0, 42)
ax.legend(loc="upper right", framealpha=0.9)

plt.tight_layout()
out_pdf = "plots/figura_4_traslacional_kirby_bauer.pdf"
out_png = "plots/figura_4_traslacional_kirby_bauer.png"
plt.savefig(out_pdf)
plt.savefig(out_png)
plt.close()
print(f"Gráfico traslacional guardado: {out_pdf} y {out_png}")