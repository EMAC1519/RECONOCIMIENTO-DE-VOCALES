import time
import joblib
import numpy as np
import sounddevice as sd

import vocales_comun as vc

FS = 16000
DURACION = 3
UMBRAL_DISTANCIA = 3.0   # orientativo, en unidades estandarizadas

if not vc.RUTA_MODELO.exists():
    raise FileNotFoundError(
        f"No existe {vc.RUTA_MODELO}. Primero ejecuta 08_entrenar_knn.py."
    )

paquete = joblib.load(vc.RUTA_MODELO)
modelo = paquete["modelo"]
columnas = paquete["columnas"]

input("Presiona ENTER y luego pronuncia una vocal sostenida...")

for n in (3, 2, 1):
    print(n)
    time.sleep(0.6)

print("¡HABLA AHORA!")
audio = sd.rec(int(FS * DURACION), samplerate=FS, channels=1,
               dtype="float32")
sd.wait()

try:
    x = vc.preprocesar(audio[:, 0], FS)
    car = vc.extraer_caracteristicas(x, FS)
except ValueError as error:
    print(f"No se pudo analizar la grabación: {error}")
    raise SystemExit

X = np.array([[car[c] for c in columnas]])

prediccion = modelo.predict(X)[0]
probabilidades = modelo.predict_proba(X)[0]
distancia, _ = modelo[-1].kneighbors(modelo[:-1].transform(X))
distancia_media = float(distancia.mean())

print(f"\nF1={car['F1']:.0f} Hz  F2={car['F2']:.0f} Hz  F3={car['F3']:.0f} Hz")
print("\nProbabilidades:")
for clase, p in sorted(zip(modelo.classes_, probabilidades),
                       key=lambda t: -t[1]):
    print(f"  {clase.upper()}: {p:.0%}")

print(f"\nVOCAL IDENTIFICADA: {prediccion.upper()}")

if distancia_media > UMBRAL_DISTANCIA:
    print("[AVISO] El sonido está lejos de los datos de entrenamiento; "
          "la predicción puede no ser confiable.")