import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

# =============================================================================
# 1. CONFIGURACIÓN Y GENERACIÓN DE DATOS (BASADO EN TU CÓDIGO)
# =============================================================================
# Configuración visual
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# Fijar semilla para reproducibilidad
np.random.seed(42)
n = 200

# Generación de variables con relaciones lógicas
visitas_web = np.random.randint(1, 50, n)
compras = 0.35 * visitas_web + np.random.normal(2, 2, n)
monto_total = 75 * compras + np.random.normal(0, 150, n)

# Limpieza técnica e integridad (Tus procesos de limpieza)
compras = np.clip(compras, 0, None).round().astype(int)
devoluciones = np.array([np.random.randint(0, c + 1) for c in compras])
devoluciones = np.clip(devoluciones, 0, (compras / 2).astype(int))
reseñas = np.random.choice([1, 2, 3, 4, 5], size=n, p=[0.05, 0.05, 0.1, 0.5, 0.3])
monto_total = np.clip(monto_total, 0, None)
genero = np.random.choice(["Masculino", "Femenino"], size=n)

# Creación del DataFrame
data = pd.DataFrame({
    "cliente_id": range(1, n + 1),
    "visitas_web": visitas_web,
    "Compras": compras,
    "monto_total": monto_total,
    "devoluciones": devoluciones,
    "reseñas": reseñas,
    "genero": genero
})
data.set_index("cliente_id", inplace=True)
data["monto_total"] = data["monto_total"].round(2)

# =============================================================================
# 2. ANÁLISIS DE VARIABILIDAD (PARA EL INFORME TÉCNICO)
# =============================================================================
media_gasto = data["monto_total"].mean()
std_gasto = data["monto_total"].std()
umbral_high_ticket = media_gasto + std_gasto

print(f"--- Estadísticas de Valor ---")
print(f"Promedio de Gasto: ${media_gasto:.2f}")
print(f"Desviación Estándar: ${std_gasto:.2f}")
print(f"Umbral High-Ticket (+1 std): ${umbral_high_ticket:.2f}")

# =============================================================================
# 3. VISUALIZACIONES PROFESIONALES (CON LABELS LIMPIOS)
# =============================================================================

# --- A. Matriz de Correlación (Heatmap) ---
plt.figure(figsize=(8, 6))
correlacion = data.corr(numeric_only=True)
nombres_heatmap = ["Tráfico Web", "Frecuencia", "Monto Total ($)", "Devoluciones", "Reseñas"]
sns.heatmap(correlacion, annot=True, cmap="RdYlGn", xticklabels=nombres_heatmap, yticklabels=nombres_heatmap)
plt.title("Matriz de Correlación: Interdependencia de Variables")
plt.savefig("grafico_matriz_correlacion.png", dpi=300, bbox_inches="tight")
plt.show()

# --- B. Regresión: Frecuencia vs Gasto (Jointplot) ---
grafico_reg = sns.jointplot(data=data, x="Compras", y="monto_total", kind="reg", color="teal", height=7)
grafico_reg.set_axis_labels("Frecuencia de Compra", "Monto Total de Gasto ($)", fontsize=11)
grafico_reg.fig.suptitle("Análisis de Regresión: Motor de Ingresos", y=1.02)
grafico_reg.savefig("grafico_regresion_compras_monto.png", dpi=300, bbox_inches="tight")
plt.show()

# --- C. Distribución y Segmentación High-Ticket (KDE) ---
plt.figure(figsize=(10, 6))
sns.kdeplot(data["monto_total"], fill=True, color="slategray", alpha=0.3)
plt.axvline(media_gasto, color='red', linestyle='--', label=f'Media: ${media_gasto:.2f}')
plt.axvspan(umbral_high_ticket, data["monto_total"].max(), color='gold', alpha=0.3, label='Segmento High-Ticket')
plt.title("Distribución de Gasto: Identificación de Clientes de Alto Valor")
plt.xlabel("Monto Total ($)")
plt.ylabel("Densidad de Clientes")
plt.legend()
plt.savefig("grafico_densidad_high_ticket.png", dpi=300, bbox_inches="tight")
plt.show()

# --- D. Segmentación por Género (Violin Plot) ---
plt.figure(figsize=(8, 5))
sns.violinplot(data=data, x="genero", y="monto_total", palette="muted", inner="quartile")
plt.title("Distribución de Gasto por Género")
plt.xlabel("Género del Cliente")
plt.ylabel("Monto Total ($)")
plt.savefig("grafico_segmentacion_genero.png", dpi=300, bbox_inches="tight")
plt.show()

# =============================================================================
# 4. MODELO DE REGRESIÓN LINEAL (OLS) Y VALIDACIÓN
# =============================================================================
X = sm.add_constant(data["Compras"])
y = data["monto_total"]
modelo = sm.OLS(y, X).fit()

print("\n--- RESULTADOS DEL MODELO OLS ---")
print(modelo.summary())

# Validación: QQ-Plot de Residuos (Tu gráfico de validación)
plt.figure(figsize=(6, 6))
sm.qqplot(modelo.resid, line="45", fit=True)
plt.title("Validación de Normalidad: QQ-Plot de Residuos")
plt.savefig("grafico_validacion_qqplot.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nAnálisis completado. Todos los gráficos han sido exportados para el informe.")