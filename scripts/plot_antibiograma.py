#!/usr/bin/env python3
"""
scripts/plot_antibiograma.py
Comparativa traslacional Kirby-Bauer (EUCAST v16.0):
1. Figura 4: E. coli (Panel Primario: Control Virgen vs Post-Abandono).
2. Figura 4b: Panel Completo Multi-Cepa con los 8 Antibióticos (1ra Línea vs Rescate OMS)
   evaluados en g0 (Virgen), g1/g2 (Post-Abandono) y g3/g4 (Superbacteria XDR).
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("plots", exist_ok=True)
candidates = ["data/headless/traslacional_biopsia.csv", "./../data/headless/traslacional_biopsia.csv"]
data_file = next((p for p in candidates if os.path.exists(p)), candidates[0])

if not os.path.exists(data_file):
    print(f"Error: {data_file} no encontrado.")
    exit(1)

df = pd.read_csv(data_file)

# 1. Figura 4 Clásica: E. coli (Panel Primario: Control Virgen vs Post-Abandono RAM)
df_ecoli = df[df["strain"] == "Escherichia coli"].copy()
if "panel" in df_ecoli.columns:
    df_ecoli_prim = df_ecoli[df_ecoli["panel"] == "PRIMARIO_1RA"].copy()
else:
    df_ecoli_prim = df_ecoli.copy()

antibiotics = df_ecoli_prim["antibiotic_code"].unique()
x = np.arange(len(antibiotics))
width = 0.35

halos_pre = [df_ecoli_prim[(df_ecoli_prim["antibiotic_code"] == ab) & (df_ecoli_prim["condition"] == "CONTROL_VIRGEN")]["halo_diameter_mm"].values[0] for ab in antibiotics]
halos_post = [df_ecoli_prim[(df_ecoli_prim["antibiotic_code"] == ab) & (df_ecoli_prim["condition"] == "POST_ABANDONO_RAM")]["halo_diameter_mm"].values[0] for ab in antibiotics]
breakpoints = [df_ecoli_prim[df_ecoli_prim["antibiotic_code"] == ab]["eucast_breakpoint"].values[0] for ab in antibiotics]

fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
ax.bar(x - width/2, halos_pre, width, label="Control Virgen ($g_0$)", color="#10B981")
ax.bar(x + width/2, halos_post, width, label="Post-Abandono RAM ($g_1/g_2$)", color="#EF4444")

for i, bp in enumerate(breakpoints):
    ax.hlines(bp, x[i] - width*0.8, x[i] + width*0.8, colors="#1E293B", linestyles="--", linewidth=1.5)
    ax.text(x[i], bp + 0.8, f"Corte: {bp} mm", ha="center", fontsize=8, color="#1E293B", weight="bold")

ax.set_ylabel("Diámetro del Halo de Inhibición (mm)", fontsize=10, fontweight="bold")
ax.set_title("Ciclo Traslacional: Contracción de Halos por Selección in vivo (E. coli)", fontsize=11, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(antibiotics, fontsize=10, fontweight="bold")
ax.grid(axis="y", linestyle=":", alpha=0.7)
ax.set_ylim(0, 38)
ax.legend(loc="upper right", framealpha=0.9)

plt.tight_layout()
plt.savefig("plots/figura_4_traslacional_kirby_bauer.pdf")
plt.savefig("plots/figura_4_traslacional_kirby_bauer.png")
plt.close()

# 2. Figura 4b Multi-Cepa Completa: Los 8 Antibióticos (1ra Línea + Rescate OMS) en g0, g1/g2 y g3/g4
strains = df["strain"].unique()
fig, axes = plt.subplots(1, len(strains), figsize=(18, 5.5), dpi=300, sharey=True)

if len(strains) == 1:
    axes = [axes]

has_xdr = "SUPERBACTERIA_XDR" in df["condition"].values
bar_w = 0.26 if has_xdr else 0.35

for idx, strain in enumerate(strains):
    ax = axes[idx]
    df_s = df[df["strain"] == strain].copy()
    abs_s = df_s["antibiotic_code"].unique()
    xs = np.arange(len(abs_s))

    pre_s = [df_s[(df_s["antibiotic_code"] == ab) & (df_s["condition"] == "CONTROL_VIRGEN")]["halo_diameter_mm"].values[0] for ab in abs_s]
    post_s = [df_s[(df_s["antibiotic_code"] == ab) & (df_s["condition"] == "POST_ABANDONO_RAM")]["halo_diameter_mm"].values[0] for ab in abs_s]
    bps_s = [df_s[df_s["antibiotic_code"] == ab]["eucast_breakpoint"].values[0] for ab in abs_s]

    if has_xdr:
        xdr_s = [df_s[(df_s["antibiotic_code"] == ab) & (df_s["condition"] == "SUPERBACTERIA_XDR")]["halo_diameter_mm"].values[0] for ab in abs_s]
        ax.bar(xs - bar_w, pre_s, bar_w, label="Virgen ($g_0$)", color="#10B981")
        ax.bar(xs, post_s, bar_w, label="Post-Abandono ($g_1/g_2$)", color="#F59E0B")
        ax.bar(xs + bar_w, xdr_s, bar_w, label="Multirresistente ($g_3/g_4$)", color="#EF4444")
    else:
        ax.bar(xs - bar_w/2, pre_s, bar_w, label="Control Virgen ($g_0$)", color="#10B981")
        ax.bar(xs + bar_w/2, post_s, bar_w, label="Post-Abandono ($g_1/g_2$)", color="#EF4444")

    # Línea vertical separadora entre Panel Primario (0..3) y Panel de Rescate OMS (4..7)
    if len(abs_s) > 4:
        ax.axvline(x=3.5, color="#38BDF8", linestyle="-.", linewidth=1.5, alpha=0.85)
        ax.text(1.5, 35.0, "1ra Línea", ha="center", fontsize=8.5, color="#334155", weight="bold")
        ax.text(5.5, 35.0, "Panel Rescate OMS", ha="center", fontsize=8.5, color="#0284C7", weight="bold")

    for i, bp in enumerate(bps_s):
        ax.hlines(bp, xs[i] - 0.38, xs[i] + 0.38, colors="#1E293B", linestyles="--", linewidth=1.3)
        ax.text(xs[i], bp + 0.7, f">={bp:.0f}", ha="center", fontsize=7, color="#1E293B", weight="bold")

    ax.set_title(f"{strain}", fontsize=11.5, fontweight="bold", style="italic")
    ax.set_xticks(xs)
    ax.set_xticklabels(abs_s, fontsize=9, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.7)
    ax.set_ylim(0, 38)
    if idx == 0:
        ax.set_ylabel("Diámetro del Halo de Inhibición (mm)", fontsize=10.5, fontweight="bold")
    if idx == len(strains) - 1:
        ax.legend(loc="upper right", fontsize=8, framealpha=0.92)

fig.suptitle(
    "Informe en Cascada EUCAST v16.0: Fracaso del Panel Primario en $g_3/g_4$ y Eficacia del Panel de Rescate OMS",
    fontsize=13, fontweight="bold", y=1.02
)
plt.tight_layout()
plt.savefig("plots/figura_4b_traslacional_multicepa.pdf", bbox_inches="tight")
plt.savefig("plots/figura_4b_traslacional_multicepa.png", bbox_inches="tight")
plt.close()

print("Gráficos traslacionales (Figuras 4 y 4b con los 8 antibióticos) guardados en plots/")