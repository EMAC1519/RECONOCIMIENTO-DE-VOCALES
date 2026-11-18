# Sistema para reconocimiento de vocales

Sistema en Python que reconoce las cinco vocales del español (/a/, /e/, /i/, /o/, /u/) a partir de la voz. Captura el audio con el micrófono, estima los formantes con Predicción Lineal (LPC) y clasifica la vocal con el algoritmo de los K vecinos más cercanos (KNN). Incluye una interfaz gráfica para capturar datos, entrenar el modelo y reconocer vocales en tiempo cuasi-real.

Proyecto desarrollado en la Facultad de Ingeniería y Ciencias de la Universidad Autónoma de Tamaulipas (UAT).

## Video demostrativo

[[Ver el video](URL_DEL_VIDEO)](https://youtube.com/shorts/tPjRCrwHxP4)

## Características

- Captura de voz a 16 kHz, un canal, 3 segundos.
- Preprocesamiento: eliminación de componente continua, filtro Butterworth pasaaltas (70 Hz) y normalización.
- Detección automática del tramo estable de la vocal (sin ventanas de tiempo fijas).
- Estimación de formantes F1, F2 y F3 con LPC (orden 18, preénfasis 0.97) usando la mediana de varios frames.
- Características adicionales para separar mejor las vocales: LPCC, razones F1/F3 y F2/F3, y ancho de banda por formante.
- Clasificador KNN con `StandardScaler` y votación ponderada por distancia.
- Validación *leave-one-speaker-out* (se excluye un hablante completo en cada vuelta).
- Interfaz gráfica en Tkinter con tres pestañas: **Reconocer**, **Capturar datos** y **Entrenamiento**.
- Muestra la vocal identificada, los formantes medidos, la probabilidad de cada vocal y la posición en la carta de vocales.

## Resultados

| Dato | Valor |
|---|---|
| Muestras del dataset | 100 (10 hablantes, 20 por vocal) |
| Solo formantes | 79 – 84 % de precisión |
| Mejor modelo (F1 + F2 + LPCC, k = 7) | 91 % de precisión |
| Validación | Dejando un hablante fuera |

Con solo formantes, la vocal /u/ se confundía por la cercanía entre F1 y F2. Al añadir LPCC y las demás características, la precisión mejoró.

## Estructura del proyecto

```text
Proyecto/
├── vocales_comun.py            # Funciones comunes: rutas, preprocesamiento, LPC, formantes
├── 00_verificar_librerias.py   # Verifica Python, librerías, micrófono y archivos
├── 01b_capturaVocal.py         # Captura por consola (opcional)
├── 03c_espectro_lpc.py         # Espectro FFT, envolvente LPC y formantes (opcional)
├── 07_construir_dataset.py     # Procesa las grabaciones y genera el CSV
├── 08_entrenar_knn.py          # Valida, entrena y guarda el modelo
├── 09_reconocer_vocal.py       # Reconocimiento por consola
├── 10_interfaz_vocales.py      # Interfaz gráfica (Tkinter)
├── voces/                      # Grabaciones de entrenamiento (.npz)
├── dataset/                    # dataset_vocales.csv
├── modelos/                    # modelo_vocales.joblib y validacion_knn.csv
├── graficas/                   # Matriz de confusión, carta de vocales, espectros
└── reconocimientos/            # Un .wav por reconocimiento e historial.csv
```

Las carpetas de datos se crean automáticamente junto a `vocales_comun.py`.

## Requisitos

- Python 3.9 o superior
- Micrófono funcional
- Bibliotecas: NumPy, SciPy, Matplotlib, sounddevice, pandas, joblib y scikit-learn (Tkinter viene incluido con Python)

## Instalación

```bash
git clone URL_DEL_REPOSITORIO
cd NOMBRE_DEL_REPOSITORIO

pip install numpy scipy matplotlib sounddevice pandas joblib scikit-learn

python 00_verificar_librerias.py
```

El script `00_verificar_librerias.py` indica qué falta y ofrece instalarlo.

## Uso

### Con la interfaz gráfica

```bash
python 10_interfaz_vocales.py
```

1. **Capturar datos:** escribe el identificador del hablante (por ejemplo, `H01`), elige la vocal y graba. Revisa la vista previa y pulsa *Guardar* o *Descartar*.
2. **Entrenamiento:** pulsa el botón de entrenamiento. Se ejecutan `07_construir_dataset.py` y `08_entrenar_knn.py` y el modelo se recarga solo.
3. **Reconocer:** pulsa grabar y pronuncia una vocal sostenida. Se muestra la vocal, los formantes y la probabilidad de cada clase.

### Desde la consola

```bash
python 07_construir_dataset.py    # genera dataset/dataset_vocales.csv
python 08_entrenar_knn.py         # entrena y guarda modelos/modelo_vocales.joblib
python 09_reconocer_vocal.py      # graba y reconoce una vocal
```

Para validar con *leave-one-speaker-out* se necesitan al menos dos hablantes en `voces/`. Con uno solo, el sistema usa validación cruzada y avisa que la precisión es optimista.

## Metodología

1. Captura de la señal de voz.
2. Validación de la señal (silencio, saturación).
3. Preprocesamiento.
4. Detección del tramo estable de la vocal.
5. Extracción de características con LPC.
6. Construcción del dataset.
7. Entrenamiento y validación del KNN.
8. Reconocimiento de vocales nuevas.

## Tecnologías

| Tecnología | Uso |
|---|---|
| Python 3 | Lenguaje del sistema |
| NumPy / SciPy | Cálculo numérico, filtros y solución del sistema LPC |
| sounddevice | Captura de audio |
| scikit-learn | `StandardScaler`, KNN, `Pipeline` y validación |
| pandas / joblib | Dataset CSV y serialización del modelo |
| Matplotlib | Gráficas e integración en Tkinter |
| Tkinter | Interfaz gráfica |

## Autores

Alumnos de la Facultad de Ingeniería y Ciencias, Universidad Autónoma de Tamaulipas

- Avalos Martínez Ericm Guadalupe
- Carrera Contreras Diego
- Carrizales Castro José Alberto
- Cepeda Hernandez Marco Antonio
- Herrera Martínez Saul Emmanuel
- Zamarripa Villanueva Danna Itzel

Cd. Victoria, Tamaulipas, México
