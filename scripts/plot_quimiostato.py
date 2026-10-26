#!/usr/bin/env python3
"""
scripts/plot_quimiostato.py
1. Figura 2: Validación Cruzada Clásica (1 Dosis): Autómata Celular vs EDO RK4 (Sargent, 2013).
2. Figura 2b: Quimiostato Escalonado / Morbidostat (Toprak et al., 2012): 4 Dosis progresivas
   mostrando la sucesión evolutiva de los 5 genotipos (g0 -> g1 -> g2 -> g3 -> g4) y convergencia RK4.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("plots", exist_ok=True)
candidates_c = ["data/headless/quimiostato_mc.csv", "./../data/headless/quimiostato_mc.csv"]
data_file_c = next((p for p in candidates_c if os.path.exists(p)), candidates_c[0])

if not os.path.exists(data_file_c):
    print(f"Error: {data_file_c} no encontrado.")
    exit(1)

df = pd.read_csv(data_file_c)
r_final = df["pearson_instant"].iloc[-1]
rmse = np.sqrt(np.mean((df["ca_total"] - df["ode_total"]) ** 2))
nrmse_pct = (rmse / 25130.0) * 100.0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

# Panel A: Serie Temporal (Modo 1)
ax1.plot(df["tick"], df["ca_total"], label="Autómata Celular (Discreto)", color="#10B981", linewidth=1.8)
ax1.plot(df["tick"], df["ode_total"], label="Levin-Regoes (Continuo RK4)", color="#0284C7", linestyle="--", linewidth=1.8)
ax1.axvline(x=1500, color="#EF4444", linestyle=":", label="Inyección Dosis (0.08 ug/mL)")

ax1.set_xlabel("Tiempo (minutos / ticks)", fontsize=10, fontweight="bold")
ax1.set_ylabel("Población Bacteriana Total (Celdas)", fontsize=10, fontweight="bold")
ax1.set_title("A: Convergencia Dinámica Temporal (1 Dosis)", fontsize=11, fontweight="bold")
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(loc="lower right", framealpha=0.9)

# Panel B: Diagrama de Dispersión vs Recta Identidad
ax2.scatter(df["ode_total"], df["ca_total"], color="#6366F1", alpha=0.4, s=12, label="Datos de Trayectoria")
max_val = max(df["ode_total"].max(), df["ca_total"].max())
ax2.plot([0, max_val], [0, max_val], color="#EF4444", linestyle="-", label="Identidad Teórica (y = x)")

ax2.set_xlabel("Población EDO Continua (RK4)", fontsize=10, fontweight="bold")
ax2.set_ylabel("Población Autómata Celular (CA)", fontsize=10, fontweight="bold")
ax2.set_title(f"B: Correlación Isomórfica (r = {r_final:.4f} | NRMSE = {nrmse_pct:.2f}%)", fontsize=11, fontweight="bold")
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(loc="upper left", framealpha=0.9)

plt.tight_layout()
plt.savefig("plots/figura_2_quimiostato_validacion.pdf")
plt.savefig("plots/figura_2_quimiostato_validacion.png")
plt.close()

# --- 2. Figura 2b: Morbidostat Escalonado (4 Dosis Progresivas: g0 -> g1 -> g2 -> g3 -> g4) ---
candidates_m = [
    "data/headless/quimiostato_morbidostat_4dosis.csv",
    "./../data/headless/quimiostato_morbidostat_4dosis.csv",
]
data_file_m = next((p for p in candidates_m if os.path.exists(p)), None)

if data_file_m and os.path.exists(data_file_m):
    df_m = pd.read_csv(data_file_m)
    r_morb = df_m["pearson_instant"].iloc[-1]
    rmse_m = np.sqrt(np.mean((df_m["ca_total"] - df_m["ode_total"]) ** 2))
    nrmse_m = (rmse_m / 25130.0) * 100.0

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 7.5), dpi=300, sharex=True)

    # Panel Superior: Población Total CA vs RK4 Multi-Clonal con los 4 escalones marcados
    ax_top.plot(df_m["tick"], df_m["ca_total"], label="Autómata Celular Total (CA)", color="#10B981", linewidth=2.0)
    ax_top.plot(df_m["tick"], df_m["ode_total"], label="Sistema EDO Multi-Clonal (RK4)", color="#0284C7", linestyle="--", linewidth=1.8)

    dose_events = [
        (1200, "Dosis 1\n(0.08 ug/mL)"),
        (2400, "Dosis 2\n(0.45 ug/mL)"),
        (3600, "Dosis 3\n(3.20 ug/mL)"),
        (4800, "Dosis 4\n(25.0 ug/mL)"),
    ]
    for t_ev, label_ev in dose_events:
        ax_top.axvline(x=t_ev, color="#EF4444", linestyle=":", linewidth=1.4)
        ax_top.text(t_ev + 40, 22500, label_ev, fontsize=8.5, color="#B91C1C", weight="bold")

    ax_top.set_ylabel("Población Total (Celdas)", fontsize=10.5, fontweight="bold")
    ax_top.set_title(
        f"A: Quimiostato Escalonado / Morbidostat (Toprak et al., 2012) — Convergencia CA vs RK4 (r = {r_morb:.4f}, NRMSE = {nrmse_m:.2f}%)",
        fontsize=11.5,
        fontweight="bold",
    )
    ax_top.set_ylim(0, 26500)
    ax_top.grid(True, linestyle=":", alpha=0.6)
    ax_top.legend(loc="lower right", framealpha=0.92)

    # Panel Inferior: Reemplazo Clonal Sucesivo (g0 -> g1 -> g2 -> g3 -> g4)
    colors_g = ["#22C55E", "#EAB308", "#F97316", "#EF4444", "#A855F7"]
    labels_g = [
        "g0 (Salvaje, CMI=0.035)",
        "g1 (1° Salto, CMI=0.20)",
        "g2 (2° Salto, CMI=1.50)",
        "g3 (3° Salto, CMI=12.0)",
        "g4 (Superbacteria, CMI=140.0)",
    ]
    for g in range(5):
        ax_bot.plot(df_m["tick"], df_m[f"ca_g{g}"], label=f"CA {labels_g[g]}", color=colors_g[g], linewidth=1.8)
        ax_bot.plot(df_m["tick"], df_m[f"ode_g{g}"], color=colors_g[g], linestyle="--", linewidth=1.2, alpha=0.75)

    for t_ev, _ in dose_events:
        ax_bot.axvline(x=t_ev, color="#64748B", linestyle=":", linewidth=1.2)

    ax_bot.set_xlabel("Tiempo de Evolución Continua (minutos / ticks)", fontsize=10.5, fontweight="bold")
    ax_bot.set_ylabel("Subpoblación por Genotipo", fontsize=10.5, fontweight="bold")
    ax_bot.set_title("B: Sucesión Evolutiva Escalonada (Línea sólida: Autómata Celular | Línea punteada: EDO RK4)", fontsize=11, fontweight="bold")
    ax_bot.set_ylim(0, 24000)
    ax_bot.grid(True, linestyle=":", alpha=0.6)
    ax_bot.legend(loc="upper left", ncol=2, fontsize=8.5, framealpha=0.92)

    plt.tight_layout()
    plt.savefig("plots/figura_2b_morbidostat_4dosis.pdf")
    plt.savefig("plots/figura_2b_morbidostat_4dosis.png")
    plt.close()

print("Gráficos de Quimiostato (Figura 2 Clásica y Figura 2b Morbidostat 4 Dosis) guardados en plots/")