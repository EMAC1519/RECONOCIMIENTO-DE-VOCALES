import csv
import numpy as np

import vocales_comun as vc

CARPETA_CAPTURAS = vc.CARPETA_VOCES
SALIDA = vc.RUTA_DATASET

ultimos = {}   # (hablante, vocal, repetición) → fila; se queda la más reciente
errores = []

for archivo in sorted(CARPETA_CAPTURAS.glob("*.npz")):
    try:
        with np.load(archivo, allow_pickle=False) as d:
            fs = int(d["fs"])
            senal = np.asarray(d["normalizada"], dtype=np.float64)
            vocal = str(d["vocal"].item())
            hablante = str(d["hablante"].item())
            repeticion = int(d["repeticion"].item())

        car = vc.extraer_caracteristicas(senal, fs)

    except (ValueError, KeyError, np.linalg.LinAlgError) as error:
        errores.append((archivo.name, str(error)))
        continue

    ultimos[(hablante, vocal, repeticion)] = {
        "archivo": archivo.name,
        "hablante": hablante,
        "vocal": vocal,
        "repeticion": repeticion,
        **car,
    }

filas = list(ultimos.values())

if not filas:
    raise RuntimeError(
        f"No se pudo procesar ninguna grabación en {CARPETA_CAPTURAS}."
    )

with open(SALIDA, "w", newline="", encoding="utf-8-sig") as f:
    escritor = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
    escritor.writeheader()
    escritor.writerows(filas)

print(f"Dataset guardado en {SALIDA}: {len(filas)} muestras")
print(f"Hablantes: {sorted({f['hablante'] for f in filas})}")

print("\nPromedios por vocal (Hz):")
for v in "aeiou":
    sub = [f for f in filas if f["vocal"] == v]
    if sub:
        print(f"  {v.upper()}: n={len(sub):2d}  "
              f"F1={np.mean([s['F1'] for s in sub]):6.0f}  "
              f"F2={np.mean([s['F2'] for s in sub]):6.0f}  "
              f"F3={np.mean([s['F3'] for s in sub]):6.0f}")

if errores:
    print(f"\n[AVISO] {len(errores)} archivos con problema:")
    for nombre, motivo in errores:
        print(f"  {nombre}: {motivo}")