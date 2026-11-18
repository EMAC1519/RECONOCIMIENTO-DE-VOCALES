import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows, freqz, lfilter

import vocales_comun as vc

CARPETA_CAPTURAS = vc.CARPETA_VOCES

hablante = input("Identificador del hablante (ej. H01): ").strip().upper()
vocal = input("Vocal que se analizará (a/e/i/o/u): ").strip().lower()
repeticion = int(input("Número de repetición: ").strip())

patron = f"{hablante}_{vocal}_rep{repeticion:02d}_*.npz"
archivos = sorted(CARPETA_CAPTURAS.glob(patron))

if not archivos:
    raise FileNotFoundError(
        f"No se encontró: {patron} en {CARPETA_CAPTURAS}"
    )

archivo = archivos[-1]
print(f"Grabación: {archivo.name}")

with np.load(archivo, allow_pickle=False) as d:
    fs = int(d["fs"])
    senal = np.asarray(d["normalizada"], dtype=np.float64)

car = vc.extraer_caracteristicas(senal, fs)
print(f"Tramo estable: {car['inicio_s']} s a {car['fin_s']} s")
print(f"F1={car['F1']}  F2={car['F2']}  F3={car['F3']} Hz "
      f"({car['n_frames']} frames)")

# Frame central del tramo estable
i0 = int(car["inicio_s"] * fs)
i1 = int(car["fin_s"] * fs)
lon = int(vc.DURACION_FRAME * fs)
centro = (i0 + i1) // 2 - lon // 2

x = lfilter([1.0, -vc.PREENFASIS], [1.0], senal - np.mean(senal))
frame = x[centro:centro + lon]
frame = (frame - np.mean(frame)) * windows.hamming(lon, sym=False)

# Espectro FFT
n_fft = 2048
frec = np.fft.rfftfreq(n_fft, d=1.0 / fs)
espectro_db = 20 * np.log10(np.abs(np.fft.rfft(frame, n=n_fft)) + 1e-12)

# Envolvente LPC
a = vc.calcular_lpc(frame, vc.orden_lpc(fs))
_, h = freqz(1.0, a, worN=frec, fs=fs)
envolvente_db = 20 * np.log10(np.abs(h) + 1e-12)
envolvente_db += espectro_db.max() - envolvente_db.max()

plt.figure(figsize=(12, 5))
plt.plot(frec, espectro_db, linewidth=0.7, label="Espectro FFT")
plt.plot(frec, envolvente_db, linewidth=2, label="Envolvente LPC")

for nombre in ("F1", "F2", "F3"):
    plt.axvline(car[nombre], color="red", linestyle="--", alpha=0.7)
    plt.text(car[nombre], espectro_db.max() + 2, nombre,
             color="red", ha="center")

plt.xlim(0, 4000)
plt.title(f"Vocal {vocal.upper()} - {hablante}, repetición {repeticion}")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud (dB)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

ruta_grafica = vc.CARPETA_GRAFICAS / (
    f"espectro_{hablante}_{vocal}_rep{repeticion:02d}.png"
)
plt.savefig(ruta_grafica, dpi=150, bbox_inches="tight")
print(f"Gráfica guardada en {ruta_grafica}")

plt.show()