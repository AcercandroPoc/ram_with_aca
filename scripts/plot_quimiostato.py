#!/usr/bin/env python3
"""
scripts/plot_quimiostato.py
Validación Cruzada: Autómata Celular vs EDO Continua (Levin-Regoes RK4).
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("plots", exist_ok=True)
data_file = "./../data/headless/quimiostato_mc.csv"

if not os.path.exists(data_file):
    print(f"Error: {data_file} no encontrado.")
    exit(1)

df = pd.read_csv(data_file)
r_final = df["pearson_instant"].iloc[-1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

# Panel A: Serie Temporal
ax1.plot(df["tick"], df["ca_total"], label="Autómata Celular (Discreto)", color="#10B981", linewidth=1.8)
ax1.plot(df["tick"], df["ode_total"], label="Levin-Regoes (Continuo RK4)", color="#0284C7", linestyle="--", linewidth=1.8)
ax1.axvline(x=1500, color="#EF4444", linestyle=":", label="Inyección Dosis (0.08 ug/mL)")

ax1.set_xlabel("Tiempo (minutos / ticks)", fontsize=10, fontweight="bold")
ax1.set_ylabel("Población Bacteriana Total (Celdas)", fontsize=10, fontweight="bold")
ax1.set_title("A: Convergencia Dinámica Temporal", fontsize=11, fontweight="bold")
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(loc="lower right", framealpha=0.9)

# Panel B: Diagrama de Dispersión vs Recta Identidad
ax2.scatter(df["ode_total"], df["ca_total"], color="#6366F1", alpha=0.4, s=12, label="Datos de Trayectoria")
max_val = max(df["ode_total"].max(), df["ca_total"].max())
ax2.plot([0, max_val], [0, max_val], color="#EF4444", linestyle="-", label="Identidad Teórica (y = x)")

ax2.set_xlabel("Población EDO Continua (RK4)", fontsize=10, fontweight="bold")
ax2.set_ylabel("Población Autómata Celular (CA)", fontsize=10, fontweight="bold")
ax2.set_title(f"B: Correlación Isomórfica (Pearson r = {r_final:.4f})", fontsize=11, fontweight="bold")
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(loc="upper left", framealpha=0.9)

plt.tight_layout()
out_pdf = "plots/figura_2_quimiostato_validacion.pdf"
out_png = "plots/figura_2_quimiostato_validacion.png"
plt.savefig(out_pdf)
plt.savefig(out_png)
plt.close()
print(f"Gráfico de Quimiostato guardado: {out_pdf} y {out_png}")