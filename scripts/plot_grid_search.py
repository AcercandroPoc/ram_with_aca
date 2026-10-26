#!/usr/bin/env python3
"""
scripts/plot_grid_search.py
Mapas de calor del espacio de fases [Pauta tau x Dosis antes del abandono]:
1. Figura 3: E. coli individual.
2. Figura 3b: Comparativa Multi-Cepa y efecto del genotipo inicial (Inóculo Virgen g0 vs Pre-expuesto g1).
3. Figura 3c: Desglose Clínico en g0: Fracaso Terapéutico Total (Recaída + RAM) vs Selección de Mutantes (RAM).
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("plots", exist_ok=True)
candidates = ["data/headless/grid_search_adherence.csv", "./../data/headless/grid_search_adherence.csv"]
data_file = next((p for p in candidates if os.path.exists(p)), candidates[0])

if not os.path.exists(data_file):
    print(f"Error: {data_file} no encontrado.")
    exit(1)

df = pd.read_csv(data_file)
df["is_ram"] = (df["treatment_outcome"] == "FRACASO_RAM").astype(float)
df["is_clinical_failure"] = (df["treatment_outcome"] != "CURACION").astype(float)

if "initial_genotype" not in df.columns:
    df["initial_genotype"] = 0

regimen_labels = {
    480: "q8h (Cada 8 h)",
    720: "q12h (Cada 12 h)",
    1440: "q24h (Cada 24 h)"
}

# 1. Figura 3 Clásica: E. coli (Promedio de riesgo evolutivo)
df_ecoli = df[df["strain"] == "Escherichia coli"].copy()
hm_ecoli = df_ecoli.groupby(["tau_regimen", "doses_before_drop"])["is_ram"].mean().unstack()
hm_ecoli.index = [regimen_labels.get(i, f"{i} min") for i in hm_ecoli.index]

plt.figure(figsize=(10, 4.8), dpi=300)
sns.heatmap(
    hm_ecoli,
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
plt.title("Espacio de Fases: Emergencia de Resistencia en E. coli (Amoxicilina + Clavulánico)", fontsize=12, fontweight="bold", pad=12)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("plots/figura_3_grid_search_adherencia.pdf")
plt.savefig("plots/figura_3_grid_search_adherencia.png")
plt.close()

# 2. Figura 3b Multi-Cepa (2 Filas x 3 Columnas):
#    Fila Superior: Inóculo Virgen (g0 -> g1)
#    Fila Inferior: Inóculo Pre-Mutado de 1er Escalón (g1 -> g2, efecto escalera MSW)
strains = df["strain"].unique()
fig, axes = plt.subplots(2, len(strains), figsize=(16, 8.2), dpi=300, sharey=True)

row_configs = [
    (0, "Inóculo Virgen ($g_0 \\to g_1$)", "is_ram"),
    (1, "Cepa Pre-Expuesta ($g_1 \\to g_2$ en MSW)", "is_ram"),
]

for row_idx, (g_val, row_title, metric_col) in enumerate(row_configs):
    df_g = df[df["initial_genotype"] == g_val]
    if df_g.empty:
        df_g = df
    for col_idx, strain in enumerate(strains):
        ax = axes[row_idx, col_idx]
        df_sub = df_g[df_g["strain"] == strain]
        ab_code = df_sub["antibiotic_code"].iloc[0] if "antibiotic_code" in df_sub.columns else ""
        hm = df_sub.groupby(["tau_regimen", "doses_before_drop"])[metric_col].mean().unstack()
        hm.index = [regimen_labels.get(i, f"{i} min") for i in hm.index]

        sns.heatmap(
            hm,
            ax=ax,
            cmap="YlOrRd",
            vmin=0.0,
            vmax=1.0,
            annot=True,
            fmt=".2f",
            cbar=(col_idx == len(strains) - 1),
            cbar_kws={'label': '$P_{\\mathrm{RAM}}$'} if col_idx == len(strains) - 1 else None,
            linewidths=0.5,
            linecolor="white"
        )
        ax.set_title(f"{strain} [{ab_code}]\n{row_title}", fontsize=10, fontweight="bold", pad=6)
        ax.invert_yaxis()
        if row_idx == 1:
            ax.set_xlabel("Dosis Tomadas Antes del Abandono", fontsize=10, fontweight="bold")
        else:
            ax.set_xlabel("")
        if col_idx == 0:
            ax.set_ylabel(r"Pauta Posológica ($\tau$)", fontsize=10, fontweight="bold")
        else:
            ax.set_ylabel("")

fig.suptitle(
    "Espacio de Fases Multi-Cepa: Selección de Resistencia en Cepa Virgen ($g_0$) vs. Pre-Expuesta ($g_1$)",
    fontsize=13,
    fontweight="bold",
    y=1.02
)
plt.tight_layout()
plt.savefig("plots/figura_3b_grid_search_multicepa.pdf", bbox_inches="tight")
plt.savefig("plots/figura_3b_grid_search_multicepa.png", bbox_inches="tight")
plt.close()

# 3. Figura 3c: Contraste en Cepa Virgen (g0) entre Fracaso Clínico (Recaída + RAM) vs Selección RAM
df_g0 = df[df["initial_genotype"] == 0]
fig, axes = plt.subplots(2, len(strains), figsize=(16, 8.2), dpi=300, sharey=True)

for col_idx, strain in enumerate(strains):
    df_sub = df_g0[df_g0["strain"] == strain]
    ab_code = df_sub["antibiotic_code"].iloc[0]

    # Fila 0: Fracaso Clínico Global (Recaída Sensible + Selección RAM)
    hm_fail = df_sub.groupby(["tau_regimen", "doses_before_drop"])["is_clinical_failure"].mean().unstack()
    hm_fail.index = [regimen_labels.get(i, f"{i} min") for i in hm_fail.index]
    sns.heatmap(
        hm_fail, ax=axes[0, col_idx], cmap="Oranges", vmin=0.0, vmax=1.0, annot=True, fmt=".2f",
        cbar=(col_idx == len(strains) - 1),
        cbar_kws={'label': '$P_{\\mathrm{Fracaso}}$ (Recaída+RAM)'} if col_idx == len(strains) - 1 else None,
        linewidths=0.5, linecolor="white"
    )
    axes[0, col_idx].set_title(f"{strain} [{ab_code}]\nFracaso Clínico (Recaída + RAM)", fontsize=10, fontweight="bold", pad=6)
    axes[0, col_idx].invert_yaxis()
    axes[0, col_idx].set_xlabel("")

    # Fila 1: Fracaso por Mutación Genotípica (FRACASO_RAM)
    hm_ram = df_sub.groupby(["tau_regimen", "doses_before_drop"])["is_ram"].mean().unstack()
    hm_ram.index = [regimen_labels.get(i, f"{i} min") for i in hm_ram.index]
    sns.heatmap(
        hm_ram, ax=axes[1, col_idx], cmap="YlOrRd", vmin=0.0, vmax=1.0, annot=True, fmt=".2f",
        cbar=(col_idx == len(strains) - 1),
        cbar_kws={'label': '$P_{\\mathrm{RAM}}$ (Mutación)'} if col_idx == len(strains) - 1 else None,
        linewidths=0.5, linecolor="white"
    )
    axes[1, col_idx].set_title(f"{strain} [{ab_code}]\nEmergencia de Mutantes ($g_0 \\to g_1$)", fontsize=10, fontweight="bold", pad=6)
    axes[1, col_idx].invert_yaxis()
    axes[1, col_idx].set_xlabel("Dosis Tomadas Antes del Abandono", fontsize=10, fontweight="bold")

    if col_idx == 0:
        axes[0, col_idx].set_ylabel(r"Pauta ($\tau$)", fontsize=10, fontweight="bold")
        axes[1, col_idx].set_ylabel(r"Pauta ($\tau$)", fontsize=10, fontweight="bold")
    else:
        axes[0, col_idx].set_ylabel("")
        axes[1, col_idx].set_ylabel("")

fig.suptitle(
    "Infección por Cepa Virgen ($g_0$): Recaída Clínica por Abandono Temprano vs. Selección de Mutantes RAM",
    fontsize=13, fontweight="bold", y=1.02
)
plt.tight_layout()
plt.savefig("plots/figura_3c_recaida_vs_ram.pdf", bbox_inches="tight")
plt.savefig("plots/figura_3c_recaida_vs_ram.png", bbox_inches="tight")
plt.close()

print("Mapas de calor (Figuras 3, 3b y 3c) guardados exitosamente en plots/")