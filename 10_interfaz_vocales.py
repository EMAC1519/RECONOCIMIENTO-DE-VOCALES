import os
import sys
import csv
import time
import queue
import threading
import subprocess
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import pandas as pd
import joblib
import sounddevice as sd
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import vocales_comun as vc

BASE = Path(__file__).resolve().parent
os.chdir(BASE)

FS = 16000
DURACION = 3
VOCALES = ["a", "e", "i", "o", "u"]
UMBRAL_DISTANCIA = 3.0

CARPETA_CAPTURAS = vc.CARPETA_VOCES
RUTA_MODELO = vc.RUTA_MODELO
RUTA_DATASET = vc.RUTA_DATASET

BG        = "#070d1a"
TARJETA   = "#0d1626"
TARJETA2  = "#101c31"
ENTRADA   = "#0a1220"
BORDE     = "#22344f"
BORDE_F   = "#16233a"
TEXTO     = "#e9effc"
MUTED     = "#8296b8"
ACENTO    = "#00e5ff"
ACENTO2   = "#8ff4ff"
ACENTO_F  = "#0e3a45"
VERDE     = "#2ee6a8"
ROJO      = "#ff5f7a"
AMBAR     = "#ffc857"
VIOLETA   = "#a78bfa"

FUENTE  = "Segoe UI"
MONO    = "Cascadia Mono"

COLORES = {
    "a": "#ff6b81", "e": "#ffc857", "i": "#2ee6a8",
    "o": "#00e5ff", "u": "#a78bfa",
}

SOS_PASAALTAS = butter(4, 70, btype="highpass", fs=FS, output="sos")


def grabar_audio():
    audio = sd.rec(int(FS * DURACION), samplerate=FS,
                   channels=1, dtype="float32")
    sd.wait()
    return audio[:, 0].astype(np.float64)


def abrir_carpeta(ruta):
    ruta.mkdir(parents=True, exist_ok=True)
    if sys.platform.startswith("win"):
        os.startfile(ruta)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(ruta)])
    else:
        subprocess.Popen(["xdg-open", str(ruta)])

class Boton(tk.Button):
    def __init__(self, master, texto, comando, tipo="secundario", **kw):
        estilos = {
            "primario":   (ACENTO,  BG,     ACENTO2),
            "secundario": (BORDE,   TEXTO,  "#2c4266"),
            "peligro":    ("#341626", ROJO, "#542033"),
        }
        self._base, fg, self._hover = estilos[tipo]
        super().__init__(master, text=texto, command=comando,
                         bg=self._base, fg=fg,
                         activebackground=self._hover,
                         activeforeground=fg,
                         bd=0, relief="flat", padx=16, pady=9,
                         font=(FUENTE, 10, "bold"),
                         cursor="hand2", **kw)
        self.bind("<Enter>", lambda e: self.config(bg=self._hover))
        self.bind("<Leave>", lambda e: self.config(bg=self._base))


class BarraProb(tk.Canvas):
    def __init__(self, master, color, **kw):
        super().__init__(master, height=16, bg=TARJETA,
                         highlightthickness=0, **kw)
        self.color = color
        self.valor = 0.0
        self.ganador = False
        self.bind("<Configure>", lambda e: self._dibujar())

    def set(self, valor, ganador=False):
        self.valor = max(0.0, min(1.0, float(valor)))
        self.ganador = ganador
        self._dibujar()

    def _dibujar(self):
        self.delete("todo")
        w = max(self.winfo_width(), 40)
        h = max(self.winfo_height(), 12)
        r = h / 2.0
        self._rect(1, 1, w - 1, h - 1, r, fill=ENTRADA, tags="todo")
        if self.valor > 0.004:
            fin = 1 + (w - 2) * self.valor
            color = VERDE if self.ganador else self.color
            self._rect(1, 1, max(fin, 2 * r), h - 1, r, fill=color, tags="todo")
        self.create_text(w - 9, h / 2, text=f"{self.valor:.0%}",
                         anchor="e", fill=TEXTO,
                         font=(FUENTE, 8, "bold"), tags="todo")

    def _rect(self, x1, y1, x2, y2, r, **kw):
        pts = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r,
               x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2,
               x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]
        self.create_polygon(pts, smooth=True, width=0, **kw)


class OndaEncabezado(tk.Canvas):
    def __init__(self, master, alto=116, **kw):
        super().__init__(master, height=alto, bg=BG,
                         highlightthickness=0, **kw)
        self.fase = 0.0
        self.bind("<Configure>", lambda e: self._dibujar())
        self._animar()

    def _animar(self):
        self.fase += 0.085
        self._dibujar()
        self.after(45, self._animar)

    def _dibujar(self):
        self.delete("todo")
        w = self.winfo_width()
        h = self.winfo_height()
        if w < 20 or h < 20:
            return
        for x in range(0, w, 46):
            self.create_line(x, 0, x, h, fill=BORDE_F, tags="todo")
        for y in range(0, h, 29):
            self.create_line(0, y, w, y, fill=BORDE_F, tags="todo")
        self.create_line(0, h / 2, w, h / 2, fill=BORDE, tags="todo")
        olas = ((0.20, 0.055, ACENTO_F, 1),
                (0.34, 0.024, ACENTO,  2))
        for amp, frec, color, grosor in olas:
            pts = []
            for x in range(0, w + 6, 6):
                env = np.sin(0.006 * x + self.fase * 0.55)
                y = h / 2 + h * amp * np.sin(frec * x + self.fase) * env
                pts += [x, float(y)]
            self.create_line(*pts, fill=color, width=grosor,
                             tags="todo")
        # linea de barrido
        xb = int((self.fase * 60) % w)
        self.create_line(xb, 0, xb, h, fill=ACENTO2, tags="todo")


class Chip(tk.Frame):
    def __init__(self, master, texto):
        super().__init__(master, bg=TARJETA2,
                         highlightbackground=BORDE,
                         highlightthickness=1, bd=0)
        tk.Label(self, text=texto, bg=TARJETA2, fg=ACENTO2,
                 font=(MONO, 9)).pack(padx=10, pady=4)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Reconocimiento de vocales")
        self.geometry("1280x800")
        self.minsize(1080, 700)
        self.configure(bg=BG)

        self.cola = queue.Queue()
        self.modelo = None
        self.columnas = None
        self.k = None
        self.df = None
        self.captura = None

        self._estilos()
        self._construir()
        self._cargar_dataset()
        self._cargar_modelo()

        x0 = np.zeros(FS // 10)
        self._dibujar_onda(self.ax_onda, self.canvas_rec,
                           x0, titulo="Senal grabada")
        self._dibujar_espectro(self.ax_fft, self.canvas_rec, x0)
        self._dibujar_carta(self.ax_carta, self.canvas_rec)
        self._dibujar_onda(self.ax_cap, self.canvas_cap,
                           None, titulo="Vista previa de la captura")
        self._refrescar_campos()
        self.after(80, self._procesar_cola)

    def _estilos(self):
        s = ttk.Style(self)
        s.theme_use("clam")

        s.configure(".", background=BG, foreground=TEXTO,
                    font=(FUENTE, 10), bordercolor=BORDE)
        s.configure("TNotebook", background=BG, borderwidth=0)
        s.configure("TNotebook.Tab", background=TARJETA, foreground=MUTED,
                    padding=(22, 10), font=(FUENTE, 10, "bold"),
                    borderwidth=0)
        s.map("TNotebook.Tab",
              background=[("selected", ACENTO_F)],
              foreground=[("selected", ACENTO2)])

        s.configure("Horizontal.TProgressbar", troughcolor=ENTRADA,
                    background=ACENTO, bordercolor=BG, lightcolor=ACENTO,
                    darkcolor=ACENTO, thickness=8)

        for estilo in ("TEntry", "TSpinbox", "TCombobox"):
            s.configure(estilo, fieldbackground=ENTRADA, foreground=TEXTO,
                        insertcolor=TEXTO, arrowcolor=TEXTO,
                        background=ENTRADA, bordercolor=BORDE,
                        borderwidth=1, padding=6)
        s.map("TCombobox",
              fieldbackground=[("readonly", ENTRADA)],
              foreground=[("readonly", TEXTO)])
        s.configure("Vertical.TScrollbar", background=BORDE,
                    troughcolor=ENTRADA, arrowcolor=MUTED,
                    bordercolor=BG, darkcolor=BORDE, lightcolor=BORDE)

        self.option_add("*TCombobox*Listbox.background", ENTRADA)
        self.option_add("*TCombobox*Listbox.foreground", TEXTO)
        self.option_add("*TCombobox*Listbox.selectBackground", ACENTO_F)
        self.option_add("*TCombobox*Listbox.selectForeground", TEXTO)

    def _tarjeta(self, padre, **pack):
        marco = tk.Frame(padre, bg=BORDE, bd=0)
        dentro = tk.Frame(marco, bg=TARJETA)
        dentro.pack(fill="both", expand=True, padx=1, pady=1)
        marco.pack(**pack)
        return dentro

    def _seccion(self, padre, texto):
        tk.Label(padre, text=texto, bg=TARJETA, fg=ACENTO,
                 font=(FUENTE, 9, "bold")).pack(anchor="w", pady=(0, 6))

    def _campo(self, padre, etiqueta):
        tk.Label(padre, text=etiqueta, bg=TARJETA, fg=MUTED,
                 font=(FUENTE, 9)).pack(anchor="w")
        cont = tk.Frame(padre, bg=BORDE, bd=0)
        dentro = tk.Frame(cont, bg=ENTRADA)
        dentro.pack(fill="both", expand=True, padx=1, pady=1)
        cont.pack(fill="x", pady=(3, 12))
        return dentro

    def _construir(self):
        self._encabezado()

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=22, pady=(4, 18))

        t1 = tk.Frame(self.nb, bg=BG)
        t2 = tk.Frame(self.nb, bg=BG)
        t3 = tk.Frame(self.nb, bg=BG)
        self.nb.add(t1, text="  Reconocer  ")
        self.nb.add(t2, text="  Capturar datos  ")
        self.nb.add(t3, text="  Entrenamiento  ")

        self._tab_reconocer(t1)
        self._tab_capturar(t2)
        self._tab_entrenar(t3)

    def _encabezado(self):
        cab = tk.Frame(self, bg=BG)
        cab.pack(fill="x", padx=22, pady=(14, 6))
        self.onda = OndaEncabezado(cab)
        self.onda.pack(fill="x")
        self.onda.update_idletasks()

        tk.Label(cab, text="RECONOCIMIENTO DE VOCALES",
                 bg=BG, fg=TEXTO,
                 font=(FUENTE, 20, "bold")).place(x=30, y=16)
        tk.Label(cab, text="Formantes LPC  ·  Clasificador KNN  ·  "
                           "Carta F1-F2",
                 bg=BG, fg=MUTED,
                 font=(FUENTE, 10)).place(x=31, y=56)

        chips = tk.Frame(cab, bg=BG)
        chips.place(relx=1.0, x=-22, y=24, anchor="ne")
        for texto in ("fs = 16 kHz", "LPC · KNN", "3 s / toma"):
            Chip(chips, texto).pack(side="right", padx=(8, 0))

    def _tab_reconocer(self, padre):
        padre.columnconfigure(1, weight=1)
        padre.rowconfigure(0, weight=1)

        izq = self._tarjeta(padre)
        izq.grid(row=0, column=0, sticky="ns", padx=(0, 12))
        izq.configure(padx=20, pady=18)

        self._seccion(izq, "RESULTADO")
        self.lbl_vocal = tk.Label(izq, text="-", bg=TARJETA,
                                  fg=ACENTO, font=(FUENTE, 96, "bold"))
        self.lbl_vocal.pack()

        self.lbl_formantes = tk.Label(
            izq, text="F1 -    F2 -    F3 -", bg=TARJETA, fg=TEXTO,
            font=(MONO, 10))
        self.lbl_formantes.pack(pady=(0, 12))

        fila_rec = tk.Frame(izq, bg=TARJETA)
        fila_rec.pack(fill="x")
        self.punto_rec = tk.Label(fila_rec, text="●", bg=TARJETA,
                                  fg=BORDE, font=(FUENTE, 10))
        self.punto_rec.pack(side="left")
        self.lbl_estado_rec = tk.Label(
            fila_rec, text="Presiona Grabar y di una vocal sostenida.",
            bg=TARJETA, fg=MUTED, font=(FUENTE, 9),
            wraplength=250, justify="left")
        self.lbl_estado_rec.pack(side="left", padx=(6, 0))

        self.btn_reconocer = Boton(izq, "●   GRABAR VOCAL",
                                   self._reconocer, "primario")
        self.btn_reconocer.pack(fill="x", pady=(14, 8))
        self.barra_rec = ttk.Progressbar(izq, maximum=100)
        self.barra_rec.pack(fill="x")

        sep = tk.Frame(izq, bg=BORDE, height=1)
        sep.pack(fill="x", pady=(16, 12))

        self._seccion(izq, "PROBABILIDADES")
        self.barras_prob = {}
        for v in VOCALES:
            fila = tk.Frame(izq, bg=TARJETA)
            fila.pack(fill="x", pady=3)
            tk.Label(fila, text=v.upper(), bg=TARJETA, fg=COLORES[v],
                     font=(FUENTE, 10, "bold"), width=2).pack(side="left")
            barra = BarraProb(fila, COLORES[v])
            barra.pack(side="left", fill="x", expand=True, padx=(6, 0))
            self.barras_prob[v] = barra

        self.lbl_modelo = tk.Label(izq, text="", bg=TARJETA, fg=MUTED,
                                   font=(FUENTE, 9), wraplength=250,
                                   justify="left")
        self.lbl_modelo.pack(side="bottom", pady=(12, 0), anchor="w")

        der = self._tarjeta(padre)
        der.grid(row=0, column=1, sticky="nsew")
        self.fig_rec = Figure(figsize=(6.2, 6.4), facecolor=TARJETA)
        self.ax_onda = self.fig_rec.add_subplot(3, 1, 1)
        self.ax_fft = self.fig_rec.add_subplot(3, 1, 2)
        self.ax_carta = self.fig_rec.add_subplot(3, 1, 3)
        self.fig_rec.subplots_adjust(hspace=0.85, left=0.09,
                                     right=0.97, top=0.93, bottom=0.09)
        self.canvas_rec = FigureCanvasTkAgg(self.fig_rec, master=der)
        self.canvas_rec.get_tk_widget().pack(fill="both", expand=True,
                                             padx=10, pady=10)

        self.ctx_rec = {"boton": self.btn_reconocer,
                        "barra": self.barra_rec,
                        "estado": self.lbl_estado_rec,
                        "punto": self.punto_rec}

    def _tab_capturar(self, padre):
        padre.columnconfigure(1, weight=1)
        padre.rowconfigure(0, weight=1)

        izq = self._tarjeta(padre)
        izq.grid(row=0, column=0, sticky="ns", padx=(0, 12))
        izq.configure(padx=20, pady=18)

        self._seccion(izq, "NUEVA CAPTURA")

        self.var_hablante = tk.StringVar(value="H01")
        self.var_vocal = tk.StringVar(value="a")
        self.var_rep = tk.StringVar(value="1")

        c1 = self._campo(izq, "HABLANTE")
        self.ent_hablante = ttk.Entry(c1, textvariable=self.var_hablante)
        self.ent_hablante.pack(fill="x", padx=8, pady=5)
        self.ent_hablante.bind("<FocusOut>", self._refrescar_campos)

        c2 = self._campo(izq, "VOCAL")
        cb = ttk.Combobox(c2, textvariable=self.var_vocal,
                          values=VOCALES, state="readonly")
        cb.pack(fill="x", padx=8, pady=5)
        cb.bind("<<ComboboxSelected>>", self._refrescar_campos)

        c3 = self._campo(izq, "REPETICION")
        ttk.Spinbox(c3, from_=1, to=99,
                    textvariable=self.var_rep).pack(fill="x",
                                                    padx=8, pady=5)

        fila_rec = tk.Frame(izq, bg=TARJETA)
        fila_rec.pack(fill="x", pady=(2, 0))
        self.punto_cap = tk.Label(fila_rec, text="●", bg=TARJETA,
                                  fg=BORDE, font=(FUENTE, 10))
        self.punto_cap.pack(side="left")
        self.lbl_estado_cap = tk.Label(
            fila_rec, text="Configura los datos y graba.",
            bg=TARJETA, fg=MUTED, font=(FUENTE, 9),
            wraplength=250, justify="left")
        self.lbl_estado_cap.pack(side="left", padx=(6, 0))

        self.btn_grabar_cap = Boton(izq, "●   GRABAR",
                                    self._grabar_captura, "primario")
        self.btn_grabar_cap.pack(fill="x", pady=(12, 8))
        self.barra_cap = ttk.Progressbar(izq, maximum=100)
        self.barra_cap.pack(fill="x")

        self.lbl_formantes_cap = tk.Label(izq, text="", bg=TARJETA,
                                          fg=TEXTO, font=(MONO, 10))
        self.lbl_formantes_cap.pack(pady=(12, 8))

        fila = tk.Frame(izq, bg=TARJETA)
        fila.pack(fill="x")
        self.btn_guardar = Boton(fila, "GUARDAR", self._guardar_captura,
                                 "secundario", state="disabled")
        self.btn_guardar.pack(side="left", expand=True, fill="x",
                              padx=(0, 5))
        self.btn_descartar = Boton(fila, "DESCARTAR", self._descartar_captura,
                                   "peligro", state="disabled")
        self.btn_descartar.pack(side="left", expand=True, fill="x",
                                padx=(5, 0))

        self.lbl_conteo = tk.Label(izq, text="", bg=TARJETA, fg=MUTED,
                                   font=(MONO, 9), justify="left")
        self.lbl_conteo.pack(pady=(16, 0), anchor="w")

        der = self._tarjeta(padre)
        der.grid(row=0, column=1, sticky="nsew")
        self.fig_cap = Figure(figsize=(6.2, 4.4), facecolor=TARJETA)
        self.ax_cap = self.fig_cap.add_subplot(1, 1, 1)
        self.fig_cap.subplots_adjust(left=0.07, right=0.97,
                                     top=0.9, bottom=0.13)
        self.canvas_cap = FigureCanvasTkAgg(self.fig_cap, master=der)
        self.canvas_cap.get_tk_widget().pack(fill="both", expand=True,
                                             padx=10, pady=10)

        self.ctx_cap = {"boton": self.btn_grabar_cap,
                        "barra": self.barra_cap,
                        "estado": self.lbl_estado_cap,
                        "punto": self.punto_cap}


    def _tab_entrenar(self, padre):
        padre.columnconfigure(0, weight=1)
        padre.rowconfigure(0, weight=1)

        tarjeta = self._tarjeta(padre)
        tarjeta.grid(row=0, column=0, sticky="nsew")
        tarjeta.configure(padx=22, pady=20)
        padre.columnconfigure(0, weight=1)
        padre.rowconfigure(0, weight=1)

        self._seccion(tarjeta, "PIPELINE DE ENTRENAMIENTO")
        tk.Label(
            tarjeta,
            text="Reconstruye el dataset con todas las capturas y entrena "
                 "el KNN (ejecuta 07_construir_dataset.py y "
                 "08_entrenar_knn.py).",
            bg=TARJETA, fg=MUTED, font=(FUENTE, 10),
            wraplength=820, justify="left"
        ).pack(anchor="w", pady=(0, 12))

        self.btn_entrenar = Boton(tarjeta, "RECONSTRUIR DATASET Y ENTRENAR",
                                  self._entrenar, "primario")
        self.btn_entrenar.pack(anchor="w", pady=(0, 14))

        marco = tk.Frame(tarjeta, bg=BORDE, bd=0)
        dentro = tk.Frame(marco, bg=ENTRADA)
        dentro.pack(fill="both", expand=True, padx=1, pady=1)
        marco.pack(fill="both", expand=True)

        self.log = tk.Text(dentro, bg=ENTRADA, fg=TEXTO,
                           insertbackground=TEXTO, relief="flat",
                           font=(MONO, 10), padx=12, pady=10,
                           wrap="word", spacing3=2)
        scroll = ttk.Scrollbar(dentro, command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.log.pack(side="left", fill="both", expand=True)

        accesos = tk.Frame(tarjeta, bg=TARJETA)
        accesos.pack(fill="x", pady=(14, 0))
        for texto, ruta in (
            ("Voces", vc.CARPETA_VOCES),
            ("Dataset", vc.CARPETA_DATASET),
            ("Modelos", vc.CARPETA_MODELOS),
            ("Graficas", vc.CARPETA_GRAFICAS),
            ("Reconocimientos", vc.CARPETA_RECONOCIMIENTOS),
        ):
            Boton(accesos, f"Abrir {texto}",
                  lambda r=ruta: abrir_carpeta(r),
                  "secundario").pack(side="left", padx=(0, 8))

    def _estilo_ax(self, ax):
        ax.set_facecolor(ENTRADA)
        for borde in ax.spines.values():
            borde.set_color(BORDE)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(True, color=BORDE, alpha=0.55, linewidth=0.6)

    def _dibujar_onda(self, ax, canvas, x, car=None, titulo="Senal"):
        ax.clear()
        self._estilo_ax(ax)
        if x is not None:
            t = np.arange(len(x)) / FS
            ax.plot(t, x, color=ACENTO, linewidth=0.6)
            ax.fill_between(t, x, -1, color=ACENTO, alpha=0.08)
            if car:
                ax.axvspan(car["inicio_s"], car["fin_s"], color=VERDE,
                           alpha=0.22, label="Tramo analizado")
                leg = ax.legend(facecolor=TARJETA, edgecolor=BORDE,
                                fontsize=8, loc="upper right")
                for t_ in leg.get_texts():
                    t_.set_color(TEXTO)
        ax.set_xlim(0, DURACION)
        ax.set_ylim(-1.15, 1.15)
        ax.set_title(titulo, color=TEXTO, fontsize=10, pad=8)
        ax.set_xlabel("Tiempo (s)", color=MUTED, fontsize=8)
        canvas.draw_idle()

    def _dibujar_espectro(self, ax, canvas, x):
        ax.clear()
        self._estilo_ax(ax)
        if x is not None and len(x) > 16:
            ventana = np.hanning(len(x))
            mag = np.abs(np.fft.rfft(x * ventana))
            freqs = np.fft.rfftfreq(len(x), 1 / FS)
            mascara = freqs <= 5000
            mag = mag[mascara]
            freqs = freqs[mascara]
            if mag.max() > 0:
                mag = mag / mag.max()
            ax.plot(freqs, mag, color=VIOLETA, linewidth=0.8)
            ax.fill_between(freqs, mag, 0, color=VIOLETA, alpha=0.10)
        ax.set_xlim(0, 5000)
        ax.set_ylim(0, 1.1)
        ax.set_title("Espectro (FFT)", color=TEXTO, fontsize=10, pad=8)
        ax.set_xlabel("Frecuencia (Hz)", color=MUTED, fontsize=8)
        ax.set_ylabel("Magnitud", color=MUTED, fontsize=8)
        canvas.draw_idle()

    def _dibujar_carta(self, ax, canvas, punto=None):
        ax.clear()
        self._estilo_ax(ax)
        hay_datos = self.df is not None and len(self.df) > 0

        if hay_datos:
            for v in VOCALES:
                sub = self.df[self.df["vocal"] == v]
                ax.scatter(sub["F2"], sub["F1"], s=20, color=COLORES[v],
                           alpha=0.5, label=v.upper())
        else:
            ax.text(0.5, 0.5, "Sin dataset todavia", color=MUTED,
                    ha="center", va="center", transform=ax.transAxes)

        if punto is not None:
            ax.scatter([punto[1]], [punto[0]], s=240, marker="*",
                       color="white", edgecolor=ACENTO, linewidth=1.4,
                       zorder=5, label="Tu voz")

        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.set_title("Carta de vocales (F1 - F2)", color=TEXTO,
                     fontsize=10, pad=8)
        ax.set_xlabel("F2 (Hz)", color=MUTED, fontsize=8)
        ax.set_ylabel("F1 (Hz)", color=MUTED, fontsize=8)
        if hay_datos or punto is not None:
            leg = ax.legend(facecolor=TARJETA, edgecolor=BORDE,
                            fontsize=8, ncol=3, loc="best")
            for t in leg.get_texts():
                t.set_color(TEXTO)
        canvas.draw_idle()

    def _cargar_dataset(self):
        try:
            self.df = pd.read_csv(RUTA_DATASET, encoding="utf-8-sig")
        except (OSError, ValueError):
            self.df = None

    def _cargar_modelo(self):
        try:
            paquete = joblib.load(RUTA_MODELO)
            self.modelo = paquete["modelo"]
            self.columnas = paquete["columnas"]
            self.k = self.modelo[-1].n_neighbors
            self.lbl_modelo.config(
                text=f"MODELO CARGADO: {' + '.join(self.columnas)}, "
                     f"k = {self.k}")
        except (OSError, KeyError, ValueError):
            self.modelo = None
            self.lbl_modelo.config(
                text="Sin modelo. Entrenalo en la pestana Entrenamiento.")

    def _grabar(self, ctx, al_terminar):
        ctx["boton"].config(state="disabled")
        ctx["barra"]["value"] = 0
        ctx["estado"].config(text="Preparate...")
        ctx["punto"].config(fg=ROJO)

        def trabajo():
            try:
                time.sleep(0.6)
                self.cola.put(("inicio", ctx))
                x = grabar_audio()
                self.cola.put(("fin", (ctx, al_terminar, x)))
            except Exception as error:
                self.cola.put(("error", (ctx, str(error))))

        threading.Thread(target=trabajo, daemon=True).start()

    def _animar_barra(self, ctx, t0):
        t = time.time() - t0
        ctx["barra"]["value"] = min(100, 100 * t / DURACION)
        if t < DURACION:
            self.after(50, lambda: self._animar_barra(ctx, t0))

    def _procesar_cola(self):
        try:
            while True:
                tipo, dato = self.cola.get_nowait()

                if tipo == "inicio":
                    dato["estado"].config(text="Habla ahora...")
                    self._animar_barra(dato, time.time())

                elif tipo == "fin":
                    ctx, callback, x = dato
                    ctx["barra"]["value"] = 100
                    ctx["boton"].config(state="normal")
                    ctx["punto"].config(fg=BORDE)
                    callback(x)

                elif tipo == "error":
                    ctx, mensaje = dato
                    ctx["boton"].config(state="normal")
                    ctx["punto"].config(fg=BORDE)
                    ctx["estado"].config(text="Error de captura.")
                    messagebox.showerror("Error al grabar", mensaje)

                elif tipo == "log":
                    self.log.insert("end", dato)
                    self.log.see("end")

                elif tipo == "recargar":
                    self._cargar_dataset()
                    self._cargar_modelo()
                    self._dibujar_carta(self.ax_carta, self.canvas_rec)
                    self.btn_entrenar.config(state="normal")

                elif tipo == "fin_entrenamiento":
                    self.btn_entrenar.config(state="normal")
        except queue.Empty:
            pass
        self.after(80, self._procesar_cola)

    def _reconocer(self):
        if self.modelo is None:
            messagebox.showinfo(
                "Sin modelo",
                "Primero entrena el modelo en la pestana Entrenamiento.")
            return
        self.lbl_vocal.config(text="...", fg=MUTED)
        self._mostrar_probabilidades()
        self._grabar(self.ctx_rec, self._resultado_reconocer)

    def _mostrar_probabilidades(self, proba=None, ganador=None):
        for v in VOCALES:
            p = 0.0 if proba is None else proba.get(v, 0.0)
            self.barras_prob[v].set(p, ganador=(v == ganador))

    def _resultado_reconocer(self, x):
        estado = self.lbl_estado_rec

        try:
            xn = vc.preprocesar(x, FS)
        except ValueError as error:
            self.lbl_vocal.config(text="?", fg=ROJO)
            estado.config(text=f"Senal invalida: {error}")
            return

        try:
            car = vc.extraer_caracteristicas(xn, FS)
        except (ValueError, np.linalg.LinAlgError) as error:
            self._dibujar_onda(self.ax_onda, self.canvas_rec, xn,
                               titulo="Senal grabada")
            self._dibujar_espectro(self.ax_fft, self.canvas_rec, xn)
            self.lbl_vocal.config(text="?", fg=ROJO)
            estado.config(text=f"No se pudo analizar: {error}")
            return

        X = np.array([[car[c] for c in self.columnas]])
        pred = self.modelo.predict(X)[0]
        proba = dict(zip(self.modelo.classes_,
                         self.modelo.predict_proba(X)[0]))
        distancia, _ = self.modelo[-1].kneighbors(
            self.modelo[:-1].transform(X))
        lejos = float(distancia.mean()) > UMBRAL_DISTANCIA

        self._guardar_reconocimiento(xn, car, pred, proba[pred], lejos)

        self.lbl_vocal.config(text=pred.upper(),
                              fg=AMBAR if lejos
                              else COLORES.get(pred, ACENTO))
        self.lbl_formantes.config(
            text=f"F1 {car['F1']:>5.0f}   F2 {car['F2']:>5.0f}   "
                 f"F3 {car['F3']:>5.0f} Hz")
        self._mostrar_probabilidades(proba, pred)

        if lejos:
            estado.config(text="Aviso: el sonido esta lejos de los datos "
                               "de entrenamiento; puede no ser confiable.")
        else:
            estado.config(text="Listo. Puedes grabar otra vocal.")

        self._dibujar_onda(self.ax_onda, self.canvas_rec, xn, car,
                           titulo="Senal grabada")
        self._dibujar_espectro(self.ax_fft, self.canvas_rec, xn)
        self._dibujar_carta(self.ax_carta, self.canvas_rec,
                            punto=(car["F1"], car["F2"]))

    def _guardar_reconocimiento(self, xn, car, pred, confianza, lejos):
        try:
            marca = datetime.now().strftime("%Y%m%d_%H%M%S")
            archivo = (vc.CARPETA_RECONOCIMIENTOS
                       / f"reconocimiento_{marca}_{pred}.wav")
            audio16 = (np.clip(xn, -1, 1) * 32767).astype(np.int16)
            wavfile.write(str(archivo), FS, audio16)

            nuevo = not vc.RUTA_HISTORIAL.exists()
            with open(vc.RUTA_HISTORIAL, "a", newline="",
                      encoding="utf-8") as f:
                escritor = csv.writer(f)
                if nuevo:
                    escritor.writerow([
                        "fecha_hora", "archivo", "prediccion", "confianza",
                        "F1_Hz", "F2_Hz", "F3_Hz", "fuera_de_rango"
                    ])
                escritor.writerow([
                    marca, archivo.name, pred, round(float(confianza), 3),
                    car["F1"], car["F2"], car["F3"], int(lejos)
                ])
        except OSError as error:
            print(f"[AVISO] No se pudo guardar el reconocimiento: {error}")

    def _siguiente_repeticion(self):
        h = self.var_hablante.get().strip().upper()
        v = self.var_vocal.get()
        numeros = []
        for f in CARPETA_CAPTURAS.glob(f"{h}_{v}_rep*_*.npz"):
            try:
                numeros.append(int(f.stem.split("_")[2][3:]))
            except (IndexError, ValueError):
                pass
        return max(numeros, default=0) + 1

    def _refrescar_campos(self, _evento=None):
        self.var_rep.set(str(self._siguiente_repeticion()))
        h = self.var_hablante.get().strip().upper()
        partes = []
        for v in VOCALES:
            n = len(list(CARPETA_CAPTURAS.glob(f"{h}_{v}_rep*_*.npz")))
            partes.append(f"{v.upper()}: {n}")
        self.lbl_conteo.config(text=f"Capturas de {h or '-'}\n"
                                    + "  ".join(partes))

    def _grabar_captura(self):
        self.captura = None
        self.btn_guardar.config(state="disabled")
        self.btn_descartar.config(state="disabled")
        self.lbl_formantes_cap.config(text="")
        self._grabar(self.ctx_cap, self._resultado_captura)

    def _resultado_captura(self, x):
        avisos = []
        pico = float(np.max(np.abs(x)))
        rms = float(np.sqrt(np.mean(x ** 2)))

        if not np.all(np.isfinite(x)):
            self.lbl_estado_cap.config(text="La senal contiene valores "
                                            "invalidos. Graba de nuevo.")
            return
        if rms < 1e-4:
            avisos.append("senal muy debil")
        if pico >= 0.99:
            avisos.append("posible saturacion")

        x_dc = x - np.mean(x)
        filtrada = sosfiltfilt(SOS_PASAALTAS, x_dc)
        pico_f = float(np.max(np.abs(filtrada)))
        if pico_f < 1e-8:
            self.lbl_estado_cap.config(text="Senal insuficiente. "
                                            "Revisa el microfono.")
            return
        norm = filtrada / pico_f

        try:
            car = vc.extraer_caracteristicas(norm, FS)
            self.lbl_formantes_cap.config(
                text=f"F1 {car['F1']:>5.0f}   F2 {car['F2']:>5.0f}   "
                     f"F3 {car['F3']:>5.0f} Hz")
        except (ValueError, np.linalg.LinAlgError):
            car = None
            avisos.append("no se detecto una vocal sostenida")

        self._dibujar_onda(self.ax_cap, self.canvas_cap, norm, car,
                           titulo="Vista previa de la captura")

        if car is None:
            self.lbl_estado_cap.config(
                text="Aviso: " + ", ".join(avisos) + ". Graba de nuevo.")
            return

        self.captura = {
            "original": x, "filtrada": filtrada, "normalizada": norm,
            "pico": pico, "rms": rms,
            "rms_filtrada": float(np.sqrt(np.mean(filtrada ** 2))),
        }
        self.btn_guardar.config(state="normal")
        self.btn_descartar.config(state="normal")
        texto = "Revisa la senal y guarda si se ve bien."
        if avisos:
            texto = "Aviso: " + ", ".join(avisos) + ".\n" + texto
        self.lbl_estado_cap.config(text=texto)

    def _guardar_captura(self):
        c = self.captura
        if not c:
            return

        hablante = self.var_hablante.get().strip().upper() or "H01"
        vocal = self.var_vocal.get()
        try:
            repeticion = int(self.var_rep.get())
            if repeticion < 1:
                raise ValueError
        except ValueError:
            messagebox.showwarning("Repeticion invalida",
                                   "La repeticion debe ser un entero "
                                   "positivo.")
            return

        marca = datetime.now().strftime("%Y%m%d_%H%M%S")
        ruta = CARPETA_CAPTURAS / (
            f"{hablante}_{vocal}_rep{repeticion:02d}_{marca}.npz")

        np.savez(
            ruta,
            fs=FS, vocal=vocal, hablante=hablante, repeticion=repeticion,
            fecha_hora=marca,
            original=c["original"], filtrada=c["filtrada"],
            normalizada=c["normalizada"],
            pico_original=c["pico"], rms_original=c["rms"],
            rms_filtrada=c["rms_filtrada"],
        )

        self.captura = None
        self.btn_guardar.config(state="disabled")
        self.btn_descartar.config(state="disabled")
        self.lbl_estado_cap.config(text=f"Guardado en voces/: {ruta.name}")
        self.lbl_formantes_cap.config(text="")
        self._refrescar_campos()

    def _descartar_captura(self):
        self.captura = None
        self.btn_guardar.config(state="disabled")
        self.btn_descartar.config(state="disabled")
        self.lbl_estado_cap.config(text="Captura descartada. Graba de nuevo.")
        self.lbl_formantes_cap.config(text="")
        self._dibujar_onda(self.ax_cap, self.canvas_cap, None,
                           titulo="Vista previa de la captura")

    def _entrenar(self):
        self.btn_entrenar.config(state="disabled")
        self.log.delete("1.0", "end")

        def trabajo():
            entorno = dict(os.environ, PYTHONIOENCODING="utf-8")
            for script in ("07_construir_dataset.py", "08_entrenar_knn.py"):
                self.cola.put(("log", f">>> {script}\n"))
                try:
                    proceso = subprocess.Popen(
                        [sys.executable, "-u", script],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, encoding="utf-8", errors="replace",
                        env=entorno, cwd=BASE)
                except OSError as error:
                    self.cola.put(("log", f"[ERROR] {error}\n"))
                    break

                for linea in proceso.stdout:
                    self.cola.put(("log", linea))
                    if "Modelo guardado" in linea:
                        self.cola.put(("recargar", None))
                proceso.wait()

                if proceso.returncode != 0:
                    self.cola.put(("log", f"\n[ERROR] {script} termino "
                                          f"con errores.\n"))
                    break
                self.cola.put(("log", "\n"))

            self.cola.put(("fin_entrenamiento", None))

        threading.Thread(target=trabajo, daemon=True).start()


if __name__ == "__main__":
    App().mainloop()