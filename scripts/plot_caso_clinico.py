#!/usr/bin/env python3
"""
scripts/plot_caso_clinico.py
Grafica un caso clínico interactivo exportado desde 4_tratamiento_clinico.rs con la tecla [E].
Muestra la farmacocinética plasmática C(t) frente a CMI/MPC y la dinámica clonal (g0..g4) en el tiempo.
"""
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("plots", exist_ok=True)

# Buscar archivos exportados en data/interactive/
search_path = "data/interactive/caso_clinico_*.csv"
files = glob.glob(search_path)

if not files:
    # Buscar también en ruta relativa anterior por si se ejecuta desde scripts/
    search_path = "./../data/interactive/caso_clinico_*.csv"
    files = glob.glob(search_path)

if not files:
    print(f"Error: No se encontraron archivos de casos clínicos exportados en data/interactive/.")
    print("Ejecuta 4_tratamiento_clinico.rs y presiona la tecla [E] para exportar un caso.")
    exit(1)

# Tomar el archivo más reciente exportado
latest_file = max(files, key=os.path.getmtime)
print(f"Procesando caso clínico: {latest_file}")

df = pd.read_csv(latest_file)
species = df["species"].iloc[0]
ab_code = df["antibiotic"].iloc[0]
init_g = df["initial_genotype"].iloc[0]
regimen = df["regimen"].iloc[0]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.5), dpi=300, sharex=True)

# Panel Superior: Farmacocinética Bateman C(t)
ax1.plot(df["minute"], df["plasma_conc_ug_ml"], label=f"Concentración Plasmática [{ab_code}]", color="#38BDF8", linewidth=2.0)
ax1.set_ylabel("Concentración ($ug/mL$)", fontsize=10.5, fontweight="bold")
ax1.set_title(f"Estudio de Caso Clínico: {species} ({regimen}, Inóculo inicial: $g_{{ {init_g} }})$", fontsize=12, fontweight="bold")
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(loc="upper right", framealpha=0.9)

# Panel Inferior: Dinámica Clonal (g0 a g4)
# Ejemplo de mejora robusta para graficar genotipos dinámicos sin vacíos ni errores de leyenda
colors_g = ["#22C55E", "#EAB308", "#F97316", "#EF4444", "#A855F7"]
labels_g = ["g0 (Salvaje)", "g1 (1° Salto)", "g2 (2° Salto)", "g3 (3° Salto)", "g4 (Resistente Máximo)"]

lines_added = False
for g in range(5):
    col_name = f"g{g}"
    # Verificamos que la columna exista y que al menos tenga una célula viva en toda la serie
    if col_name in df.columns and df[col_name].sum() > 0:
        ax2.plot(df["minute"], df[col_name], label=labels_g[g], color=colors_g[g], linewidth=1.8)
        lines_added = True

# Solo invocamos la leyenda si realmente se agregaron líneas con etiqueta
if lines_added:
    ax2.legend(loc="upper left", framealpha=0.9)
else:
    # Si todo está en cero (infección limpia sin mutantes), añadimos un texto informativo o evitamos el error
    ax2.text(0.5, 0.5, "Población clonal puramente sensible (g0)\nSin emergencia de mutantes RAM", 
             transform=ax2.transAxes, ha="center", va="center", fontsize=10, color="#64748B", style="italic")

ax2.set_xlabel("Tiempo Clínico (minutos)", fontsize=10.5, fontweight="bold")
ax2.set_ylabel("Población Celular por Genotipo", fontsize=10.5, fontweight="bold")
ax2.set_title("Evolución Clonal y Rebrote por Selección RAM", fontsize=11, fontweight="bold")
ax2.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
base_name = os.path.basename(latest_file).replace(".csv", "")
out_pdf = f"plots/{base_name}.pdf"
out_png = f"plots/{base_name}.png"
plt.savefig(out_pdf)
plt.savefig(out_png)
plt.close()

print(f"Gráfico de caso clínico guardado en: {out_pdf} y {out_png}")