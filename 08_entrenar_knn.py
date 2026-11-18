import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    LeaveOneGroupOut, StratifiedKFold, cross_val_predict
)
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

import vocales_comun as vc

RUTA_CSV = vc.RUTA_DATASET
RUTA_MODELO = vc.RUTA_MODELO

CEPSTRALES = [f"C{i}" for i in range(1, vc.N_CEPSTRALES + 1)]

CONJUNTOS = {
    "F1+F2": ["F1", "F2"],
    "F1+F2+F3": ["F1", "F2", "F3"],
    "F1/F3+F2/F3": ["R1", "R2"],
    "F1+F2+L21": ["F1", "F2", "L21"],
    "F1+F2+F3+L21+L31": ["F1", "F2", "F3", "L21", "L31"],
    "LPCC": CEPSTRALES,
    "F1+F2+LPCC": ["F1", "F2"] + CEPSTRALES,
}
VALORES_K = [1, 3, 5, 7, 9, 11, 13, 15]
ETIQUETAS = ["a", "e", "i", "o", "u"]

if not RUTA_CSV.exists():
    raise FileNotFoundError(
        f"No existe {RUTA_CSV}. Primero ejecuta 07_construir_dataset.py."
    )

df = pd.read_csv(RUTA_CSV, encoding="utf-8-sig")
y = df["vocal"].values
grupos = df["hablante"].values
n_hablantes = len(np.unique(grupos))


def crear_modelo(k):
    # El StandardScaler es indispensable: F2 (Hz) es mucho mayor que F1
    # y dominaría las distancias del KNN.
    return make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=k, weights="distance"),
    )


def validar(columnas, k):
    X = df[columnas].values
    modelo = crear_modelo(k)

    if n_hablantes >= 2:
        # Un hablante completo se deja fuera en cada vuelta
        pred = cross_val_predict(
            modelo, X, y, cv=LeaveOneGroupOut(), groups=grupos
        )
    else:
        n_splits = min(5, pd.Series(y).value_counts().min())
        if n_splits < 2:
            raise ValueError("Se necesitan al menos 2 muestras por vocal.")
        cv = StratifiedKFold(n_splits, shuffle=True, random_state=0)
        pred = cross_val_predict(modelo, X, y, cv=cv)

    return accuracy_score(y, pred), pred


if n_hablantes < 2:
    print("[AVISO] Solo hay 1 hablante: la precisión será optimista.")
    print("        Graba al menos 3 hablantes para una validación real.\n")

mejor = {"acc": -1}
tabla = []
print(f"{'Características':<12} {'k':>3} {'Precisión':>10}")

for nombre, columnas in CONJUNTOS.items():
    if not all(c in df.columns for c in columnas):
        print(f"[AVISO] {nombre}: faltan columnas en el dataset; "
              f"vuelve a ejecutar 07_construir_dataset.py.")
        continue
    for k in VALORES_K:
        if k > max(1, len(df) // 2):
            continue
        acc, pred = validar(columnas, k)
        print(f"{nombre:<12} {k:>3} {acc:>10.1%}")
        tabla.append({"caracteristicas": nombre, "k": k,
                      "precision": round(float(acc), 4)})

        if acc > mejor["acc"]:
            mejor = {"acc": acc, "pred": pred, "cols": columnas,
                     "k": k, "nombre": nombre}

print(f"\nMejor: {mejor['nombre']}, k={mejor['k']}, "
      f"precisión = {mejor['acc']:.1%}")

# Tabla de validación -> carpeta modelos
ruta_tabla = vc.CARPETA_MODELOS / "validacion_knn.csv"
pd.DataFrame(tabla).to_csv(ruta_tabla, index=False, encoding="utf-8-sig")
print(f"Tabla de validación guardada en {ruta_tabla}")

# Matriz de confusión -> carpeta graficas
ConfusionMatrixDisplay.from_predictions(
    y, mejor["pred"], labels=ETIQUETAS, cmap="Blues"
)
plt.title("Matriz de confusión (validación)")
ruta_matriz = vc.CARPETA_GRAFICAS / "matriz_confusion.png"
plt.savefig(ruta_matriz, dpi=150, bbox_inches="tight")
print(f"Gráfica guardada en {ruta_matriz}")

# Carta de vocales F1-F2 -> carpeta graficas
plt.figure(figsize=(7, 6))
for v in ETIQUETAS:
    sub = df[df["vocal"] == v]
    plt.scatter(sub["F2"], sub["F1"], label=v.upper(), alpha=0.7)
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.xlabel("F2 (Hz)")
plt.ylabel("F1 (Hz)")
plt.title("Vocales en el plano F1-F2")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
ruta_carta = vc.CARPETA_GRAFICAS / "carta_vocales.png"
plt.savefig(ruta_carta, dpi=150, bbox_inches="tight")
print(f"Gráfica guardada en {ruta_carta}")

# Modelo final entrenado con todos los datos -> carpeta modelos
modelo_final = crear_modelo(mejor["k"])
modelo_final.fit(df[mejor["cols"]].values, y)

joblib.dump({"modelo": modelo_final, "columnas": mejor["cols"]}, RUTA_MODELO)
print(f"Modelo guardado en {RUTA_MODELO}")

plt.show()