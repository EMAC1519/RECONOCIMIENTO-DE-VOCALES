import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows, freqz, lfilter

import vocales_comun as vc

# ============================================================
# DIAGNÓSTICO DE FORMANTES POR VOCAL
# Muestra, grabación por grabación, qué formantes se estimaron,
# qué polos LPC existían y cuáles se descartaron.
# ============================================================

# Zonas amplias esperadas (Hz) para voces adultas: (F1 min, F1 max, F2 min, F2 max).
# Son una referencia orientativa; las voces agudas pueden salirse un poco.
ZONAS = {
    "a": (550, 1100, 900, 1900),
    "e": (330, 700, 1500, 2600),
    "i": (200, 450, 1900, 3100),
    "o": (330, 700, 600, 1300),
    "u": (200, 520, 500, 1200),
}
BW_ANCHO = 400.0      # polos más anchos que esto se consideran sospechosos
MAX_GRAFICAS = 12

vocal = input("Vocal a diagnosticar (a/e/i/o/u) [u]: ").strip().lower() or "u"
if vocal not in ZONAS:
    raise ValueError("Vocal no válida.")
hablante = input("Hablante (ENTER = todos): ").strip().upper()

patron = f"{hablante or '*'}_{vocal}_rep*_*.npz"
ultimos = {}
for archivo in sorted(vc.CARPETA_VOCES.glob(patron)):
    ultimos[tuple(archivo.name.split("_")[:3])] = archivo

if not ultimos:
    raise FileNotFoundError(f"No hay grabaciones con el patrón {patron}.")

f1_min, f1_max, f2_min, f2_max = ZONAS[vocal]
filas, casos, errores = [], [], []

for archivo in ultimos.values():
    with np.load(archivo, allow_pickle=False) as d:
        fs = int(d["fs"])
        senal = np.asarray(d["normalizada"], dtype=np.float64)

    try:
        car = vc.extraer_caracteristicas(senal, fs)
    except (ValueError, np.linalg.LinAlgError) as error:
        errores.append((archivo.name, str(error)))
        continue

    # Frame central del tramo estable
    i0 = int(car["inicio_s"] * fs)
    i1 = int(car["fin_s"] * fs)
    lon = int(vc.DURACION_FRAME * fs)
    centro = (i0 + i1) // 2 - lon // 2
    x = lfilter([1.0, -vc.PREENFASIS], [1.0], senal - np.mean(senal))
    frame = x[centro:centro + lon]
    frame = (frame - np.mean(frame)) * windows.hamming(lon, sym=False)

    a = vc.calcular_lpc(frame, vc.orden_lpc(fs))
    raices = np.roots(a)
    raices = raices[np.imag(raices) > 0]
    frec = np.angle(raices) * fs / (2 * np.pi)
    bw = -fs / np.pi * np.log(np.abs(raices))
    orden = np.argsort(frec)
    frec, bw = frec[orden], bw[orden]

    fuera = not (f1_min <= car["F1"] <= f1_max and f2_min <= car["F2"] <= f2_max)
    # Polo por debajo de F1 que existía pero era demasiado ancho
    perdido = [(f, b) for f, b in zip(frec, bw)
               if 150 <= f < car["F1"] - 80 and b > BW_ANCHO]
    # Polo ancho entre F1 y 1300 Hz (posible F2 descartado por su ancho)
    ancho_f2 = [(f, b) for f, b in zip(frec, bw)
                if car["F1"] + 80 < f < 1300 and b > BW_ANCHO]
    # No hay ningún polo entre F1 y 1300 Hz: F1 y F2 posiblemente fusionados
    hueco = not any(car["F1"] + 150 < f < 1300 for f in frec)

    nota = ""
    if perdido:
        nota = "posible F1 descartado por ancho de banda"
    elif fuera and ancho_f2:
        nota = "posible F2 descartado por ancho de banda"
    elif fuera and hueco and car["F2"] > 1500:
        nota = "posible fusión F1-F2 (sin polo entre F1 y 1300 Hz)"
    elif fuera:
        nota = "fuera de la zona esperada"

    filas.append({
        "archivo": archivo.name, "F1": car["F1"], "F2": car["F2"],
        "F3": car["F3"], "B1": car["B1"], "sd_F1": car["sd_F1"],
        "sd_F2": car["sd_F2"], "n_frames": car["n_frames"],
        "sospechosa": int(bool(nota)), "nota": nota,
    })
    casos.append({"archivo": archivo.name, "frame": frame, "fs": fs,
                  "car": car, "frec": frec, "bw": bw, "a": a,
                  "sospechosa": bool(nota), "nota": nota})

# ---------- Resumen en consola ----------
print(f"\nDIAGNÓSTICO DE LA VOCAL {vocal.upper()} ({len(filas)} grabaciones)")
print(f"Zona esperada: F1 {f1_min}-{f1_max} Hz, F2 {f2_min}-{f2_max} Hz\n")
print(f"{'Archivo':<34}{'F1':>6}{'F2':>7}{'F3':>7}{'sdF2':>7}  Estado")
for f in filas:
    estado = f["nota"] if f["nota"] else "OK"
    print(f"{f['archivo'][:33]:<34}{f['F1']:>6.0f}{f['F2']:>7.0f}"
          f"{f['F3']:>7.0f}{f['sd_F2']:>7.0f}  {estado}")

n_sosp = sum(f["sospechosa"] for f in filas)
print(f"\nSospechosas: {n_sosp} de {len(filas)}")
for motivo in ("posible F1 descartado", "posible F2 descartado",
               "posible fusión F1-F2", "fuera de la zona"):
    cuantas = sum(1 for f in filas if f["nota"].startswith(motivo))
    print(f"  {motivo}: {cuantas}")
for nombre, motivo in errores:
    print(f"[ERROR] {nombre}: {motivo}")

ruta_csv = vc.CARPETA_GRAFICAS / f"diagnostico_{vocal}.csv"
with open(ruta_csv, "w", newline="", encoding="utf-8-sig") as f:
    escritor = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
    escritor.writeheader()
    escritor.writerows(filas)
print(f"Tabla guardada en {ruta_csv}")

# ---------- Gráficas: primero las sospechosas ----------
casos.sort(key=lambda c: not c["sospechosa"])
casos = casos[:MAX_GRAFICAS]
n = len(casos)
cols = 3
filas_fig = int(np.ceil(n / cols))
fig, ejes = plt.subplots(filas_fig, cols, figsize=(15, 3.2 * filas_fig),
                         squeeze=False)

for ax in ejes.ravel():
    ax.axis("off")

for ax, c in zip(ejes.ravel(), casos):
    ax.axis("on")
    n_fft = 2048
    frec_fft = np.fft.rfftfreq(n_fft, d=1.0 / c["fs"])
    esp = 20 * np.log10(np.abs(np.fft.rfft(c["frame"], n=n_fft)) + 1e-12)
    _, h = freqz(1.0, c["a"], worN=frec_fft, fs=c["fs"])
    env = 20 * np.log10(np.abs(h) + 1e-12)
    env += esp.max() - env.max()

    ax.plot(frec_fft, esp, linewidth=0.6, color="#9ecae1")
    ax.plot(frec_fft, env, linewidth=1.8, color="#08519c")
    for nombre in ("F1", "F2", "F3"):
        ax.axvline(c["car"][nombre], color="red", linestyle="--", alpha=0.8)
    for f, b in zip(c["frec"], c["bw"]):
        if f < 4000:
            ancho = b > BW_ANCHO
            ax.plot(f, esp.min() + 4, "x" if ancho else "o",
                    color="darkorange" if ancho else "black", markersize=6)
    ax.set_xlim(0, 4000)
    ax.set_title(f"{c['archivo'][:22]}  F1={c['car']['F1']:.0f} "
                 f"F2={c['car']['F2']:.0f}"
                 + ("  [SOSPECHOSA]" if c["sospechosa"] else ""),
                 fontsize=9, color="red" if c["sospechosa"] else "black")
    ax.set_xlabel("Hz (líneas rojas = F1, F2, F3; × naranja = polo ancho)",
                  fontsize=7)
    ax.grid(alpha=0.3)

fig.tight_layout()
ruta_png = vc.CARPETA_GRAFICAS / f"diagnostico_{vocal}.png"
fig.savefig(ruta_png, dpi=110, bbox_inches="tight")
print(f"Gráfica guardada en {ruta_png}")
plt.show()