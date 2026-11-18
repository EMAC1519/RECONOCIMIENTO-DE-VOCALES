import numpy as np
from pathlib import Path
from scipy.linalg import solve_toeplitz
from scipy.signal import butter, sosfiltfilt, lfilter, windows, freqz

# ============================================================
# FUNCIONES Y RUTAS COMPARTIDAS PARA ANÁLISIS DE VOCALES
# ============================================================

# ------------------------------------------------------------
# Carpetas del proyecto (se crean solas junto a este archivo)
# ------------------------------------------------------------

BASE = Path(__file__).resolve().parent

CARPETA_VOCES = BASE / "voces"
CARPETA_DATASET = BASE / "dataset"
CARPETA_MODELOS = BASE / "modelos"
CARPETA_GRAFICAS = BASE / "graficas"
CARPETA_RECONOCIMIENTOS = BASE / "reconocimientos"

RUTA_DATASET = CARPETA_DATASET / "dataset_vocales.csv"
RUTA_MODELO = CARPETA_MODELOS / "modelo_vocales.joblib"
RUTA_HISTORIAL = CARPETA_RECONOCIMIENTOS / "historial.csv"

for _carpeta in (CARPETA_VOCES, CARPETA_DATASET, CARPETA_MODELOS,
                 CARPETA_GRAFICAS, CARPETA_RECONOCIMIENTOS):
    _carpeta.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Parámetros de análisis
# ------------------------------------------------------------

PREENFASIS = 0.97
DURACION_FRAME = 0.030          # 30 ms
SALTO_FRAME = 0.010             # 10 ms
F_MIN = 90.0                    # Hz
BW_MAX = 800.0                  # Hz, filtro general de candidatos
# Ancho de banda máximo por formante (Hz). F1 es más permisivo: en vocales
# cerradas y posteriores (u, o) su polo LPC suele salir muy ancho y, si se
# descarta, F1 y F2 quedan desplazados hacia F2 y F3.
BW_MAX_FORMANTE = [800.0, 500.0, 600.0]
N_CEPSTRALES = 12               # coeficientes cepstrales LPC (LPCC)
SEPARACION_MIN = 100.0          # Hz mínima entre formantes consecutivos

# Rangos de búsqueda para F1, F2 y F3 (Hz)
RANGOS_FORMANTES = [(200, 1100), (500, 3000), (1500, 4000)]


def orden_lpc(fs):
    """Regla práctica: 2 + fs/1000 (18 para 16 kHz)."""
    return int(2 + fs / 1000)


def preprocesar(x, fs, f_corte=70, orden=4):
    """Quita DC, filtra pasaaltas y normaliza (igual que 01b)."""
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)
    sos = butter(orden, f_corte, btype="highpass", fs=fs, output="sos")
    x = sosfiltfilt(sos, x)
    pico = np.max(np.abs(x))
    if pico < 1e-8:
        raise ValueError("Señal insuficiente después del filtrado.")
    return x / pico


def detectar_segmento_estable(x, fs, umbral_rel=0.25):
    """
    Busca el tramo más largo con energía alta y descarta el 20 %
    de cada extremo (ataque y caída de la vocal).
    Devuelve (índice_inicio, índice_fin) en muestras.
    """
    lon = int(0.025 * fs)
    salto = int(0.010 * fs)
    n = 1 + (len(x) - lon) // salto

    rms = np.array([
        np.sqrt(np.mean(x[i * salto:i * salto + lon] ** 2))
        for i in range(n)
    ])

    activo = rms > umbral_rel * np.percentile(rms, 95)

    mejor = (0, 0)
    inicio = None
    for i, a in enumerate(list(activo) + [False]):
        if a and inicio is None:
            inicio = i
        elif not a and inicio is not None:
            if i - inicio > mejor[1] - mejor[0]:
                mejor = (inicio, i)
            inicio = None

    i0, i1 = mejor
    if i1 - i0 < 15:
        raise ValueError("No se detectó una vocal sostenida.")

    margen = int(0.2 * (i1 - i0))
    i0 += margen
    i1 -= margen

    return i0 * salto, (i1 - 1) * salto + lon


def calcular_lpc(frame, orden):
    """LPC por autocorrelación (Yule-Walker) con regularización."""
    r = np.correlate(frame, frame, mode="full")
    r = r[len(frame) - 1:len(frame) + orden].copy()

    if r[0] <= 1e-12:
        raise ValueError("Energía insuficiente en el fragmento.")

    r[0] *= 1.0001
    a = solve_toeplitz(r[:orden], r[1:orden + 1])
    return np.concatenate(([1.0], -a))


def cepstrum_lpc(a, n):
    """
    Coeficientes cepstrales LPC (c1..cn) a partir del polinomio
    A(z) = 1 - sum(alfa_k z^-k):  c_m = alfa_m + sum_{k<m} (k/m) c_k alfa_{m-k}
    """
    alfa = -np.asarray(a[1:], dtype=np.float64)
    n = min(n, len(alfa))
    c = np.zeros(n + 1)
    for m in range(1, n + 1):
        acum = alfa[m - 1]
        for k in range(1, m):
            acum += (k / m) * c[k] * alfa[m - k - 1]
        c[m] = acum
    return c[1:]


def candidatos_formantes(a, fs):
    """(frecuencia, ancho de banda) de las raíces del polinomio LPC."""
    raices = np.roots(a)
    raices = raices[np.imag(raices) > 0]

    frec = np.angle(raices) * fs / (2 * np.pi)
    bw = -fs / np.pi * np.log(np.abs(raices))

    candidatos = [
        (float(f), float(b))
        for f, b in zip(frec, bw)
        if F_MIN <= f <= fs / 2 - 100 and 0 < b <= BW_MAX
    ]
    return sorted(candidatos)


def asignar_formantes(candidatos):
    """Elige F1, F2, F3 en orden creciente, cada uno dentro de su rango."""
    elegidos = []
    previo = 0.0

    for (f_min, f_max), bw_max in zip(RANGOS_FORMANTES, BW_MAX_FORMANTE):
        for f, b in candidatos:
            if (f > previo + SEPARACION_MIN and f_min <= f <= f_max
                    and b <= bw_max):
                elegidos.append((f, b))
                previo = f
                break
        else:
            return None

    return elegidos


def extraer_caracteristicas(senal, fs):
    """
    Detecta el tramo estable, analiza todos sus frames con LPC y
    devuelve la mediana de F1, F2, F3 (y datos de calidad).
    """
    x = np.asarray(senal, dtype=np.float64)
    x = x - np.mean(x)

    i0, i1 = detectar_segmento_estable(x, fs)

    x_pre = lfilter([1.0, -PREENFASIS], [1.0], x)
    seg = x_pre[i0:i1]

    lon = int(DURACION_FRAME * fs)
    salto = int(SALTO_FRAME * fs)
    ventana = windows.hamming(lon, sym=False)
    orden = orden_lpc(fs)

    filas = []
    cepstrales = []
    niveles = []
    for ini in range(0, len(seg) - lon + 1, salto):
        frame = seg[ini:ini + lon]
        frame = (frame - np.mean(frame)) * ventana

        try:
            a = calcular_lpc(frame, orden)
        except (ValueError, np.linalg.LinAlgError):
            continue

        cepstrales.append(cepstrum_lpc(a, N_CEPSTRALES))

        sel = asignar_formantes(candidatos_formantes(a, fs))
        if sel:
            filas.append([s[0] for s in sel] + [s[1] for s in sel])

            # Nivel (dB) de la envolvente LPC en F2 y F3 respecto a F1.
            # Si F1 y F2 se fusionaron, el "F2" hallado es en realidad F3
            # y su nivel relativo sale muy bajo.
            _, h = freqz(1.0, a, worN=[s[0] for s in sel], fs=fs)
            nivel = 20 * np.log10(np.abs(h) + 1e-12)
            niveles.append([nivel[1] - nivel[0], nivel[2] - nivel[0]])

    if len(filas) < 3:
        raise ValueError("Muy pocos frames válidos para estimar formantes.")

    filas = np.array(filas)
    med = np.median(filas, axis=0)
    desv = np.std(filas[:, :3], axis=0)

    ceps = np.mean(cepstrales, axis=0)
    niv = np.median(niveles, axis=0)

    resultado = {
        "F1": round(float(med[0]), 1),
        "F2": round(float(med[1]), 1),
        "F3": round(float(med[2]), 1),
        "B1": round(float(med[3]), 1),
        "B2": round(float(med[4]), 1),
        "B3": round(float(med[5]), 1),
        "sd_F1": round(float(desv[0]), 1),
        "sd_F2": round(float(desv[1]), 1),
        "sd_F3": round(float(desv[2]), 1),
        "n_frames": int(len(filas)),
        "inicio_s": round(i0 / fs, 3),
        "fin_s": round(i1 / fs, 3),
    }

    # Razones respecto a F3: reducen el efecto del tamaño del tracto vocal
    resultado["R1"] = round(resultado["F1"] / resultado["F3"], 4)
    resultado["R2"] = round(resultado["F2"] / resultado["F3"], 4)

    # Niveles relativos de F2 y F3 respecto a F1 (detectan fusión F1-F2)
    resultado["L21"] = round(float(niv[0]), 1)
    resultado["L31"] = round(float(niv[1]), 1)

    for i, valor in enumerate(ceps, start=1):
        resultado[f"C{i}"] = round(float(valor), 4)

    return resultado