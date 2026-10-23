"""
main.py
=======
Interfaz gráfica del proyecto: red neuronal multicapa entrenada con
BACKPROPAGATION, para cualquier número de entradas, capas ocultas, neuronas
por capa y clases.

Ejecutar:  python main.py      (en VS Code: F5)

La interfaz usa tkinter (incluido en Python) y muestra las gráficas de
matplotlib dentro de la ventana. Todos los cálculos de la red están en
formulas.py y red_neuronal.py, sin librerías.

Autor: Carlos Daniel Gómez Murcia — Computacional I
"""

import threading
import tkinter as tk
from tkinter import ttk, messagebox

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import experimentos as ex
import graficas as gr
from red_neuronal import RedNeuronal, num
from aleatorio import DISTRIBUCIONES, centros_automaticos
from formulas import (media, minimo, maximo, es_finito, ajuste_profesor,
                      ajuste_momento_clasico)

AZUL = "#1F3A5F"
FONDO = "#F4F6FA"
MAX_ENTRADAS = 10
MAX_CLASES = 10

# Escenarios de prueba. Los de nubes son los mismos de la entrega 1 (3 entradas);
# XOR y Esfera necesitan capa oculta: ningún plano los separa.
ESCENARIOS = {
    "Separadas": {"tipo": "nubes", "centros": [[2, 2, 2], [8, 2, 5], [5, 8, 8]],
                  "desv": 1.0, "dist": ["normal", "normal", "normal"],
                  "ocultas": "5", "tasa": "0,1", "epocas": "1000"},
    "Cercanas": {"tipo": "nubes", "centros": [[2, 2, 2], [4.5, 2, 3], [3, 4.5, 4]],
                 "desv": 1.2, "dist": ["normal", "normal", "normal"],
                 "ocultas": "5", "tasa": "0,1", "epocas": "1500"},
    "Solapadas": {"tipo": "nubes", "centros": [[2, 2, 2], [3, 2, 2.5], [2.5, 3, 3]],
                  "desv": 1.5, "dist": ["normal", "exponencial", "normal"],
                  "ocultas": "6", "tasa": "0,05", "epocas": "1500"},
    "XOR 3D": {"tipo": "xor", "ocultas": "8", "tasa": "0,1", "epocas": "1500"},
    "Esfera 3D": {"tipo": "esfera", "ocultas": "6", "tasa": "0,1", "epocas": "2000"},
}
TEXTO_AJUSTE = {"epoca": "por época (lotes)", "patron": "por patrón"}
TEXTO_MOMENTO = {"profesor": "del profesor (Peso·Momento)", "clasico": "clásico (α·ΔW anterior)"}


# ----------------------------------------------------------------------
# Lectura de los campos (acepta coma o punto decimal)
# ----------------------------------------------------------------------
def leer_float(var, nombre):
    """Lee un número; rechaza 'nan', 'inf' y números que no caben (M8)."""
    texto = var.get().strip().replace(",", ".")
    try:
        valor = float(texto)
    except ValueError:
        raise ValueError("'%s' debe ser un número (se escribió: '%s')." % (nombre, var.get()))
    if not es_finito(valor):
        raise ValueError("'%s' debe ser un número finito (se escribió: '%s')." % (nombre, var.get()))
    return valor


def leer_positivo(var, nombre, cero=False):
    valor = leer_float(var, nombre)
    if valor < 0 or (valor == 0 and not cero):
        raise ValueError("'%s' debe ser mayor que cero." % nombre if not cero
                         else "'%s' no puede ser negativo." % nombre)
    return valor


def leer_int(var, nombre, minimo_valor=None, maximo_valor=None):
    texto = var.get().strip()
    try:
        real = float(texto.replace(",", "."))
    except ValueError:
        raise ValueError("'%s' debe ser un número entero (se escribió: '%s')." % (nombre, texto))
    if not es_finito(real):
        raise ValueError("'%s' debe ser un número entero finito." % nombre)
    valor = int(real)
    if minimo_valor is not None and valor < minimo_valor:
        raise ValueError("'%s' debe ser al menos %d." % (nombre, minimo_valor))
    if maximo_valor is not None and valor > maximo_valor:
        raise ValueError("'%s' debe ser como máximo %d." % (nombre, maximo_valor))
    return valor


def leer_vector(texto, nombre, n=None):
    """Lee valores separados por punto y coma: '2; 3,5; 1'."""
    partes = [p.strip() for p in texto.replace("\n", " ").split(";") if p.strip() != ""]
    valores = []
    for p in partes:
        try:
            v = float(p.replace(",", "."))
        except ValueError:
            raise ValueError("'%s': '%s' no es un número. Separe los valores con punto y coma (;)."
                             % (nombre, p))
        if not es_finito(v):
            raise ValueError("'%s' tiene un valor que no es finito." % nombre)
        valores.append(v)
    if n is not None and len(valores) != n:
        raise ValueError("'%s' debe tener %d valor%s separados por punto y coma (tiene %d)."
                         % (nombre, n, "" if n == 1 else "es", len(valores)))
    return valores


def leer_ocultas(texto):
    """'5; 3' -> [5, 3] (dos capas ocultas de 5 y 3 neuronas). Vacío -> sin capa oculta."""
    capas = []
    for p in texto.replace(",", ";").split(";"):
        p = p.strip()
        if p == "":
            continue
        try:
            n = int(p)
        except ValueError:
            raise ValueError("Neuronas por capa oculta: '%s' no es un entero. Ejemplo: 5; 3" % p)
        if n < 1:
            raise ValueError("Cada capa oculta debe tener al menos 1 neurona.")
        capas.append(n)
    return capas


def leer_semilla(var):
    texto = var.get().strip()
    if texto == "":
        return None
    try:
        return int(texto)
    except ValueError:
        raise ValueError("La semilla debe ser un entero o quedar vacía (aleatoria).")


def fmt(v):
    return ("%g" % v).replace(".", ",")


def fmt_vector(valores):
    return "; ".join(fmt(v) for v in valores)


# ----------------------------------------------------------------------
class MarcoDesplazable(ttk.Frame):
    """Marco con barra de desplazamiento vertical (para pantallas pequeñas)."""

    def __init__(self, padre):
        super().__init__(padre)
        self.lienzo = tk.Canvas(self, highlightthickness=0, background=FONDO)
        barra = ttk.Scrollbar(self, orient="vertical", command=self.lienzo.yview)
        self.interior = ttk.Frame(self.lienzo, padding=8)
        self.interior.bind("<Configure>", lambda e: self.lienzo.configure(
            scrollregion=self.lienzo.bbox("all")))
        self._ventana = self.lienzo.create_window((0, 0), window=self.interior, anchor="nw")
        self.lienzo.bind("<Configure>", lambda e: self.lienzo.itemconfigure(
            self._ventana, width=e.width))
        self.lienzo.configure(yscrollcommand=barra.set)
        self.lienzo.pack(side="left", fill="both", expand=True)
        barra.pack(side="right", fill="y")
        self.interior.bind("<Enter>", lambda e: self.lienzo.bind_all("<MouseWheel>", self._rueda))
        self.interior.bind("<Leave>", lambda e: self.lienzo.unbind_all("<MouseWheel>"))

    def _rueda(self, evento):
        self.lienzo.yview_scroll(int(-evento.delta / 120), "units")


class PestanaGrafica:
    """Pestaña con una figura de matplotlib y su barra de herramientas
    (zoom, mover, guardar imagen). En 3D se puede rotar con el mouse."""

    def __init__(self, cuaderno, titulo):
        self.marco = ttk.Frame(cuaderno)
        cuaderno.add(self.marco, text=titulo)
        self.fig = Figure(figsize=(9, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.marco)
        self.barra = NavigationToolbar2Tk(self.canvas, self.marco, pack_toolbar=False)
        self.barra.update()
        self.barra.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

    def vista_3d(self):
        for ax in self.fig.axes:
            if hasattr(ax, "elev"):
                return ax.elev, ax.azim
        return None


class PestanaTexto:
    def __init__(self, cuaderno, titulo):
        self.marco = ttk.Frame(cuaderno)
        cuaderno.add(self.marco, text=titulo)
        self.texto = tk.Text(self.marco, wrap="none", font=("Consolas", 10),
                             background="white", padx=8, pady=8)
        sy = ttk.Scrollbar(self.marco, orient="vertical", command=self.texto.yview)
        sx = ttk.Scrollbar(self.marco, orient="horizontal", command=self.texto.xview)
        self.texto.configure(yscrollcommand=sy.set, xscrollcommand=sx.set)
        sy.pack(side="right", fill="y")
        sx.pack(side="bottom", fill="x")
        self.texto.pack(side="left", fill="both", expand=True)

    def mostrar(self, contenido):
        self.texto.configure(state="normal")
        self.texto.delete("1.0", "end")
        self.texto.insert("1.0", contenido)
        self.texto.configure(state="disabled")


# ----------------------------------------------------------------------
class Aplicacion:

    def __init__(self, raiz):
        self.raiz = raiz
        raiz.title("Red neuronal con Backpropagation | Computacional I")
        ancho = minimo([1500, raiz.winfo_screenwidth() - 20])
        alto = minimo([900, raiz.winfo_screenheight() - 80])
        raiz.geometry("%dx%d+0+0" % (ancho, alto))
        raiz.minsize(minimo([1100, ancho]), minimo([600, alto]))
        try:  # en Windows la ventana abre maximizada
            raiz.state("zoomed")
        except tk.TclError:
            pass
        raiz.configure(background=FONDO)
        self._estilos()

        self.res = None
        self.epoca = None
        self.animando = False
        self.pendientes = set()
        self.entrenando = False
        self.parar = False
        self.avance = None

        encabezado = tk.Frame(raiz, background=AZUL)
        encabezado.pack(side="top", fill="x")
        tk.Label(encabezado, text="Red neuronal multicapa con Backpropagation",
                 font=("Segoe UI", 15, "bold"), fg="white", bg=AZUL).pack(side="left", padx=14, pady=8)
        tk.Label(encabezado, text="Carlos Daniel Gómez Murcia — Computacional I — Entrega 2",
                 font=("Segoe UI", 10), fg="#D6E0F0", bg=AZUL).pack(side="right", padx=14)
        self._ver_codigo(tk.Button(encabezado, text="⟲  Reiniciar todo", command=self.reiniciar,
                                   font=("Segoe UI", 10, "bold"), fg="white", bg="#C0392B",
                                   activebackground="#E74C3C", activeforeground="white", relief="flat",
                                   cursor="hand2", padx=12, pady=3),
                         self.reiniciar).pack(side="right", padx=6)
        raiz.bind("<Control-r>", lambda e: self.reiniciar())

        panel = ttk.PanedWindow(raiz, orient="horizontal")
        panel.pack(fill="both", expand=True, padx=6, pady=6)
        izquierda = ttk.Frame(panel, width=600)
        izquierda.pack_propagate(False)
        derecha = ttk.Frame(panel)
        panel.add(izquierda, weight=0)
        panel.add(derecha, weight=1)

        self.nb_config = ttk.Notebook(izquierda)
        self.nb_config.pack(fill="both", expand=True)
        self._pestana_red()
        self._pestana_ejemplo()

        self._barra_epocas(derecha)
        self.nb_res = ttk.Notebook(derecha)
        self.nb_res.pack(fill="both", expand=True)
        self.g_espacio = PestanaGrafica(self.nb_res, "  Espacio  ")
        self.g_errores = PestanaGrafica(self.nb_res, "  Errores  ")
        self.g_capas = PestanaGrafica(self.nb_res, "  Pesos por capa  ")
        self.g_todos = PestanaGrafica(self.nb_res, "  Todos los pesos  ")
        self.g_red = PestanaGrafica(self.nb_res, "  Red  ")
        self.t_proceso = PestanaTexto(self.nb_res, "  Proceso paso a paso  ")
        self.t_resumen = PestanaTexto(self.nb_res, "  Resumen y fórmulas  ")
        self.graficas = [("espacio", self.g_espacio, gr.dibujar_espacio),
                         ("errores", self.g_errores, gr.dibujar_errores),
                         ("capas", self.g_capas, gr.dibujar_pesos_capas),
                         ("todos", self.g_todos, gr.dibujar_todos_pesos),
                         ("red", self.g_red, gr.dibujar_red)]
        self.nb_res.bind("<<NotebookTabChanged>>", lambda e: self._redibujar_visible())
        self.nb_res.bind("<Button-3>", self._codigo_pestana)
        self.t_resumen.mostrar("Configure los datos y la red a la izquierda y presione ENTRENAR.")

    # ------------------------------------------------------------------
    def _estilos(self):
        estilo = ttk.Style()
        estilo.theme_use("clam")
        estilo.configure(".", background=FONDO, font=("Segoe UI", 10))
        estilo.configure("TNotebook.Tab", padding=(10, 5))
        estilo.configure("TLabelframe.Label", font=("Segoe UI", 10, "bold"), foreground=AZUL)
        estilo.configure("Accion.TButton", font=("Segoe UI", 11, "bold"), foreground="white",
                         background=AZUL, padding=8)
        estilo.map("Accion.TButton", background=[("active", "#2E5A8C"), ("disabled", "#8899AA")])
        estilo.configure("Encabezado.TLabel", font=("Segoe UI", 9, "bold"))
        estilo.configure("Nota.TLabel", foreground="#555")
        estilo.configure("Arquitectura.TLabel", font=("Segoe UI", 11, "bold"), foreground=AZUL)

    def _campo(self, marco, fila, texto, valor, ancho=10, columna=0, codigo=None):
        etiqueta = ttk.Label(marco, text=texto)
        etiqueta.grid(row=fila, column=columna, sticky="w", pady=2)
        var = tk.StringVar(value=valor)
        caja = ttk.Entry(marco, textvariable=var, width=ancho)
        caja.grid(row=fila, column=columna + 1, sticky="w", padx=4, pady=2)
        if codigo:
            self._ver_codigo(etiqueta, codigo)
            self._ver_codigo(caja, codigo)
        return var, caja

    def _lista(self, marco, fila, texto, opciones, valor, ancho=12, columna=0, codigo=None):
        etiqueta = ttk.Label(marco, text=texto)
        etiqueta.grid(row=fila, column=columna, sticky="w", pady=2)
        var = tk.StringVar(value=valor)
        lista = ttk.Combobox(marco, textvariable=var, values=opciones, state="readonly", width=ancho)
        lista.grid(row=fila, column=columna + 1, sticky="w", padx=4, pady=2)
        if codigo:
            self._ver_codigo(etiqueta, codigo)
            self._ver_codigo(lista, codigo)
        return var, lista

    # ------------------------------------------------------------------
    # Clic derecho: ver en VS Code el código de cada control
    # ------------------------------------------------------------------
    def _ver_codigo(self, widget, funcion):
        """Un clic derecho sobre el control abre en VS Code la función que hace
        su trabajo. Es solo una ayuda para leer el código."""
        widget.bind("<Button-3>", lambda evento: self._abrir_codigo(funcion))
        return widget

    def _abrir_codigo(self, funcion):
        ruta = str(self.raiz.tk.call("file", "normalize", funcion.__code__.co_filename))
        linea = funcion.__code__.co_firstlineno
        self.estado.configure(text="Código de %s: %s, línea %d"
                              % (funcion.__qualname__, ruta.split("/")[-1], linea))
        if self.raiz.tk.call("tk", "windowingsystem") != "win32":
            return
        try:  # Windows abre la dirección vscode://file/ruta:línea con VS Code
            self.raiz.tk.call("exec", "cmd", "/c", "start", "",
                              "vscode://file/%s:%d" % (ruta, linea), "&")
        except tk.TclError:
            pass

    def _codigo_pestana(self, evento):
        try:
            indice = self.nb_res.index("@%d,%d" % (evento.x, evento.y))
        except tk.TclError:
            return
        funciones = [gr.dibujar_espacio, gr.dibujar_errores, gr.dibujar_pesos_capas,
                     gr.dibujar_todos_pesos, gr.dibujar_red, RedNeuronal.entrenar, ex.resumen]
        self._abrir_codigo(funciones[indice])

    # ------------------------------------------------------------------
    # Pestaña RED NEURONAL
    # ------------------------------------------------------------------
    def _pestana_red(self):
        cfg = ex.config_por_defecto()
        m = MarcoDesplazable(self.nb_config)
        self.nb_config.add(m, text=" Red neuronal ")
        f = m.interior

        # ---------------- Datos ----------------
        datos = ttk.LabelFrame(f, text="Datos simulados", padding=8)
        datos.pack(fill="x", pady=4)
        fila = ttk.Frame(datos)
        fila.pack(fill="x")
        ttk.Label(fila, text="Entradas:").pack(side="left")
        self.v_nentradas = tk.StringVar(value=str(cfg["n_entradas"]))
        sp = ttk.Spinbox(fila, from_=1, to=MAX_ENTRADAS, width=4, textvariable=self.v_nentradas,
                         command=self._cambio_entradas)
        sp.pack(side="left", padx=(4, 12))
        sp.bind("<Return>", lambda e: self._cambio_entradas())
        sp.bind("<FocusOut>", lambda e: self._cambio_entradas())
        self._ver_codigo(sp, ex.generar_datos)
        ttk.Label(fila, text="Clases:").pack(side="left")
        self.v_nclases = tk.StringVar(value=str(cfg["n_clases"]))
        self.sp_clases = ttk.Spinbox(fila, from_=2, to=MAX_CLASES, width=4, textvariable=self.v_nclases,
                                     command=self._cambio_clases)
        self.sp_clases.pack(side="left", padx=(4, 12))
        self.sp_clases.bind("<Return>", lambda e: self._cambio_clases())
        self.sp_clases.bind("<FocusOut>", lambda e: self._cambio_clases())
        self._ver_codigo(self.sp_clases, ex.salidas_deseadas)
        ttk.Label(fila, text="Puntos por clase:").pack(side="left")
        self.v_npc = tk.StringVar(value=str(cfg["n_por_clase"]))
        self._ver_codigo(ttk.Entry(fila, textvariable=self.v_npc, width=6),
                         ex.generar_datos).pack(side="left", padx=4)

        tipo = ttk.Frame(datos)
        tipo.pack(fill="x", pady=(8, 4))
        ttk.Label(tipo, text="Tipo de datos:").pack(side="left")
        self.v_tipo = tk.StringVar(value=cfg["tipo_datos"])
        for valor, texto in (("nubes", "Nubes (una por clase)"), ("xor", "XOR"), ("esfera", "Esfera")):
            self._ver_codigo(ttk.Radiobutton(tipo, text=texto, value=valor, variable=self.v_tipo,
                                             command=self._cambio_tipo),
                             ex.generar_datos).pack(side="left", padx=5)

        # Tabla de las nubes (se reconstruye al cambiar entradas o clases)
        self.marco_nubes = ttk.Frame(datos)
        self.marco_tabla = ttk.Frame(self.marco_nubes)
        self.marco_tabla.pack(fill="x")
        self.filas_clase = []
        for k in range(cfg["n_clases"]):
            c = cfg["clases"][k]
            self.filas_clase.append({"centro": tk.StringVar(value=fmt_vector(c["centro"])),
                                     "desv": tk.StringVar(value=fmt(c["desviacion"])),
                                     "dist": tk.StringVar(value=c["distribucion"])})
        botones = ttk.Frame(self.marco_nubes)
        botones.pack(fill="x", pady=(6, 0))
        self._ver_codigo(ttk.Button(botones, text="Centros automáticos", command=self._centros_auto),
                         centros_automaticos).pack(side="left", padx=2)
        for texto, comando in (("Acercar", lambda: self._mover_medias(0.75)),
                               ("Alejar", lambda: self._mover_medias(1.33)),
                               ("σ +", lambda: self._escalar_sigma(1.3)),
                               ("σ −", lambda: self._escalar_sigma(1 / 1.3))):
            self._ver_codigo(ttk.Button(botones, text=texto, width=8, command=comando),
                             self._mover_medias).pack(side="left", padx=2)

        # Parámetros del XOR y de la esfera
        self.marco_xor = ttk.Frame(datos)
        self.v_xor_desv, _ = self._campo(self.marco_xor, 0, "Desviación de cada esquina",
                                         fmt(cfg["xor_desviacion"]), 8, codigo=ex.generar_xor)
        self.v_xor_lado, _ = self._campo(self.marco_xor, 1, "Lado del cubo [0, lado]",
                                         fmt(cfg["xor_lado"]), 8, codigo=ex.generar_xor)
        self.v_xor_dist, _ = self._lista(self.marco_xor, 2, "Distribución", DISTRIBUCIONES,
                                         cfg["distribucion_xor"], codigo=ex.GeneradorAleatorio.muestra)
        ttk.Label(self.marco_xor, style="Nota.TLabel", justify="left",
                  text="Una nube en cada esquina del cubo; la clase es la paridad de la esquina.\n"
                       "Con 2 entradas es la tabla del XOR. Ningún plano lo separa.").grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))
        self.marco_esfera = ttk.Frame(datos)
        self.v_esf_centro, _ = self._campo(self.marco_esfera, 0, "Centro (en cada eje)",
                                           fmt(cfg["esfera_centro"]), 8, codigo=ex.generar_esfera)
        self.v_esf_r1, _ = self._campo(self.marco_esfera, 1, "Radio de la esfera (clase 1)",
                                       fmt(cfg["esfera_radio_interno"]), 8, codigo=ex.generar_esfera)
        self.v_esf_r2, _ = self._campo(self.marco_esfera, 2, "Radio externo de la cáscara (clase 2)",
                                       fmt(cfg["esfera_radio_externo"]), 8, codigo=ex.generar_esfera)
        ttk.Label(self.marco_esfera, style="Nota.TLabel", justify="left",
                  text="Clase 1 dentro de la esfera; clase 2 en una cáscara alrededor.\n"
                       "La frontera es una superficie cerrada.").grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))

        esc = ttk.Frame(datos)
        esc.pack(side="bottom", fill="x", pady=(8, 0))
        ttk.Label(esc, text="Escenarios de prueba (3 entradas):").pack(anchor="w")
        fila_esc = ttk.Frame(esc)
        fila_esc.pack(fill="x")
        for nombre in ESCENARIOS:
            self._ver_codigo(ttk.Button(fila_esc, text=nombre, width=10,
                                        command=lambda n=nombre: self._escenario(n)),
                             self._escenario).pack(side="left", padx=1)

        # ---------------- Red ----------------
        red = ttk.LabelFrame(f, text="Red neuronal y entrenamiento", padding=8)
        red.pack(fill="x", pady=4)
        self.v_ocultas, _ = self._campo(red, 0, "Neuronas por capa oculta (separadas por ;)",
                                        "; ".join(str(n) for n in cfg["ocultas"]), 12,
                                        codigo=RedNeuronal.__init__)
        self.lbl_arquitectura = ttk.Label(red, style="Arquitectura.TLabel")
        self.lbl_arquitectura.grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 6))
        self.v_tasa, _ = self._campo(red, 2, "Tasa de aprendizaje", fmt(cfg["tasa"]),
                                     codigo=RedNeuronal._aplicar)
        self.v_ajuste, _ = self._lista(red, 3, "Ajuste de los pesos", list(TEXTO_AJUSTE.values()),
                                       TEXTO_AJUSTE[cfg["ajuste"]], 18, codigo=RedNeuronal.entrenar)
        self.v_tipo_momento, _ = self._lista(red, 4, "Momento", list(TEXTO_MOMENTO.values()),
                                             TEXTO_MOMENTO[cfg["tipo_momento"]], 24,
                                             codigo=RedNeuronal._aplicar)
        self.v_momento, self.e_momento = self._campo(red, 5, "   Momento (Peso·Momento)", fmt(cfg["momento"]),
                                                     codigo=ajuste_profesor)
        self.v_alfa, self.e_alfa = self._campo(red, 6, "   α del momento clásico", fmt(cfg["alfa"]),
                                               codigo=ajuste_momento_clasico)
        self.v_epocas, _ = self._campo(red, 7, "Número máximo de épocas", str(cfg["max_epocas"]),
                                       codigo=RedNeuronal.entrenar)
        self.v_ecm, _ = self._campo(red, 8, "ECM objetivo (se detiene al llegar)", fmt(cfg["ecm_objetivo"]),
                                    codigo=RedNeuronal.entrenar)
        self.v_sesgo = tk.BooleanVar(value=cfg["usar_sesgo"])
        self._ver_codigo(ttk.Checkbutton(red, text="Usar sesgo θ con entrada fija x0 =",
                                         variable=self.v_sesgo),
                         RedNeuronal.__init__).grid(row=9, column=0, sticky="w")
        self.v_x0 = tk.StringVar(value=str(cfg["x0"]))
        ttk.Combobox(red, textvariable=self.v_x0, values=("-1", "+1"), state="readonly",
                     width=4).grid(row=9, column=1, sticky="w", padx=4)
        self.v_normalizar = tk.BooleanVar(value=cfg["normalizar"])
        self._ver_codigo(ttk.Checkbutton(red, text="Normalizar las entradas a [0, 1] (F16)",
                                         variable=self.v_normalizar),
                         ex.entrada_de_la_red).grid(row=10, column=0, columnspan=2, sticky="w")
        self.v_barajar = tk.BooleanVar(value=cfg["barajar"])
        self._ver_codigo(ttk.Checkbutton(red, text="Barajar los patrones (solo en el ajuste por patrón)",
                                         variable=self.v_barajar),
                         ex.GeneradorAleatorio.barajar).grid(row=11, column=0, columnspan=2, sticky="w")
        self.v_semilla, _ = self._campo(red, 12, "Semilla (vacía = aleatoria)", "",
                                        codigo=ex.GeneradorAleatorio.__init__)
        self.v_registrar = tk.BooleanVar(value=cfg["registrar"])
        self._ver_codigo(ttk.Checkbutton(red, text="Registrar el proceso paso a paso; épocas:",
                                         variable=self.v_registrar),
                         RedNeuronal._anotar_patron).grid(row=13, column=0, sticky="w")
        self.v_epocas_reg = tk.StringVar(value=str(cfg["epocas_registro"]))
        ttk.Entry(red, textvariable=self.v_epocas_reg, width=5).grid(row=13, column=1, sticky="w", padx=4)
        ttk.Label(red, style="Nota.TLabel", justify="left",
                  text="Con el ajuste por época, G suma los N patrones: si la tasa es grande\n"
                       "el error oscila; con muchos puntos conviene una tasa pequeña.").grid(
            row=14, column=0, columnspan=2, sticky="w", pady=(6, 0))

        acciones = ttk.Frame(f)
        acciones.pack(fill="x", pady=10)
        self.btn_entrenar = self._ver_codigo(
            ttk.Button(acciones, text="▶  ENTRENAR", style="Accion.TButton", command=self.entrenar),
            ex.ejecutar)
        self.btn_entrenar.pack(side="left", fill="x", expand=True)
        self.btn_parar = ttk.Button(acciones, text="■ Detener", command=self.detener_entrenamiento,
                                    state="disabled")
        self.btn_parar.pack(side="left", padx=(6, 0), ipady=6)

        prueba = ttk.LabelFrame(f, text="Clasificar un punto nuevo (después de entrenar)", padding=8)
        prueba.pack(fill="x", pady=4)
        ttk.Label(prueba, text="x1; x2; ...").pack(side="left")
        self.v_punto = tk.StringVar(value="5; 5; 5")
        ttk.Entry(prueba, textvariable=self.v_punto, width=22).pack(side="left", padx=6)
        self._ver_codigo(ttk.Button(prueba, text="Clasificar", command=self.clasificar_punto),
                         ex.clasificar_punto).pack(side="left")

        for var in (self.v_ocultas, self.v_nentradas, self.v_nclases, self.v_tipo):
            var.trace_add("write", lambda *a: self._actualizar_arquitectura())
        self.v_tipo_momento.trace_add("write", lambda *a: self._activar_momento())
        self._construir_tabla()
        self._cambio_tipo()
        self._activar_momento()

    # --- tabla de clases ---
    def _n_entradas(self):
        try:
            return maximo([1, minimo([MAX_ENTRADAS, int(self.v_nentradas.get())])])
        except ValueError:
            return 3

    def _n_clases(self):
        if self.v_tipo.get() != "nubes":
            return 2
        try:
            return maximo([2, minimo([MAX_CLASES, int(self.v_nclases.get())])])
        except ValueError:
            return 2

    def _construir_tabla(self):
        for hijo in self.marco_tabla.winfo_children():
            hijo.destroy()
        n = self._n_entradas()
        nombres = "; ".join("x%d" % (j + 1) for j in range(minimo([n, 4]))) + ("; ..." if n > 4 else "")
        for col, txt in enumerate(["Clase", "Media (%s)" % nombres, "Desv.", "Distribución"]):
            ttk.Label(self.marco_tabla, text=txt, style="Encabezado.TLabel").grid(
                row=0, column=col, padx=2, sticky="w")
        for k, fila in enumerate(self.filas_clase):
            tk.Label(self.marco_tabla, text=" %d " % (k + 1), bg=gr.color_clase(k), fg="white",
                     font=("Segoe UI", 9, "bold")).grid(row=k + 1, column=0, pady=2)
            widgets = [ttk.Entry(self.marco_tabla, textvariable=fila["centro"], width=26),
                       ttk.Entry(self.marco_tabla, textvariable=fila["desv"], width=6),
                       ttk.Combobox(self.marco_tabla, textvariable=fila["dist"], values=DISTRIBUCIONES,
                                    state="readonly", width=11)]
            for col, wdg in enumerate(widgets):
                wdg.grid(row=k + 1, column=col + 1, padx=2, pady=2, sticky="w")
                self._ver_codigo(wdg, ex.generar_nubes if not isinstance(wdg, ttk.Combobox)
                                 else ex.GeneradorAleatorio.muestra)
        self._actualizar_arquitectura()

    def _cambio_entradas(self):
        n = self._n_entradas()
        self.v_nentradas.set(str(n))
        actuales = [len(leer_vector_seguro(f["centro"].get())) for f in self.filas_clase]
        if any(a != n for a in actuales):
            centros = ex.centros_por_defecto(len(self.filas_clase), n)
            for fila, c in zip(self.filas_clase, centros):
                fila["centro"].set(fmt_vector(c))
            self.v_punto.set(fmt_vector([5] * n))
        self._construir_tabla()

    def _cambio_clases(self):
        if self.v_tipo.get() != "nubes":
            return
        k = self._n_clases()
        self.v_nclases.set(str(k))
        if k == len(self.filas_clase):
            return
        centros = ex.centros_por_defecto(k, self._n_entradas())
        while len(self.filas_clase) < k:
            j = len(self.filas_clase)
            self.filas_clase.append({"centro": tk.StringVar(value=fmt_vector(centros[j])),
                                     "desv": tk.StringVar(value="1"),
                                     "dist": tk.StringVar(value="normal")})
        del self.filas_clase[k:]
        self._construir_tabla()

    def _cambio_tipo(self):
        tipo = self.v_tipo.get()
        for marco in (self.marco_nubes, self.marco_xor, self.marco_esfera):
            marco.pack_forget()
        {"nubes": self.marco_nubes, "xor": self.marco_xor, "esfera": self.marco_esfera}[tipo].pack(
            fill="x", pady=(2, 0))
        self.sp_clases.configure(state="normal" if tipo == "nubes" else "disabled")
        if tipo != "nubes":
            self.v_nclases.set("2")
        else:
            self.v_nclases.set(str(len(self.filas_clase)))
        self._actualizar_arquitectura()

    def _actualizar_arquitectura(self):
        try:
            ocultas = leer_ocultas(self.v_ocultas.get())
        except ValueError:
            self.lbl_arquitectura.configure(text="Arquitectura: (revise las neuronas por capa)")
            return
        k = self._n_clases()
        salidas = 1 if k == 2 else k
        capas = [self._n_entradas()] + ocultas + [salidas]
        texto = "Arquitectura: " + " → ".join(str(c) for c in capas)
        texto += "   (%d capa%s oculta%s, %d salida%s)" % (
            len(ocultas), "" if len(ocultas) == 1 else "s", "" if len(ocultas) == 1 else "s",
            salidas, "" if salidas == 1 else "s")
        self.lbl_arquitectura.configure(text=texto)

    def _activar_momento(self):
        clasico = self.v_tipo_momento.get() == TEXTO_MOMENTO["clasico"]
        self.e_momento.configure(state="disabled" if clasico else "normal")
        self.e_alfa.configure(state="normal" if clasico else "disabled")

    def _centros_auto(self):
        centros = centros_automaticos(len(self.filas_clase), self._n_entradas())
        for fila, c in zip(self.filas_clase, centros):
            fila["centro"].set(fmt_vector(c))

    def _escenario(self, nombre):
        e = ESCENARIOS[nombre]
        self.v_tipo.set(e["tipo"])
        self.v_nentradas.set("3")
        if e["tipo"] == "nubes":
            k = minimo([3, maximo([2, len(self.filas_clase)])])
            self.v_nclases.set(str(k))
            self.filas_clase = [{"centro": tk.StringVar(value=fmt_vector(e["centros"][j])),
                                 "desv": tk.StringVar(value=fmt(e["desv"])),
                                 "dist": tk.StringVar(value=e["dist"][j])} for j in range(k)]
        self.v_ocultas.set(e["ocultas"])
        self.v_tasa.set(e["tasa"])
        self.v_epocas.set(e["epocas"])
        self.v_punto.set("5; 5; 5")
        self._construir_tabla()
        self._cambio_tipo()
        self.estado.configure(text="Escenario '%s' cargado: presione ENTRENAR." % nombre)

    def _mover_medias(self, factor):
        """Acerca (factor < 1) o aleja (factor > 1) cada centro respecto al centro
        común de las clases:  c_nuevo = centro_comun + factor * (c - centro_comun)."""
        try:
            n = self._n_entradas()
            centros = [leer_vector(f["centro"].get(), "Media de la clase %d" % (k + 1), n)
                       for k, f in enumerate(self.filas_clase)]
        except ValueError as error:
            messagebox.showerror("Dato no válido", str(error))
            return
        comun = [media([c[j] for c in centros]) for j in range(n)]                # F22
        for fila, c in zip(self.filas_clase, centros):
            fila["centro"].set(fmt_vector([float("%.3g" % (comun[j] + factor * (c[j] - comun[j])))
                                           for j in range(n)]))
        self.entrenar()

    def _escalar_sigma(self, factor):
        try:
            for k, fila in enumerate(self.filas_clase):
                s = leer_positivo(fila["desv"], "Desviación de la clase %d" % (k + 1), cero=True) * factor
                fila["desv"].set(fmt(float("%.3g" % s)))
        except ValueError as error:
            messagebox.showerror("Dato no válido", str(error))
            return
        self.entrenar()

    def _config(self):
        cfg = ex.config_por_defecto()
        n = leer_int(self.v_nentradas, "Entradas", 1, MAX_ENTRADAS)
        cfg["n_entradas"] = n
        cfg["tipo_datos"] = self.v_tipo.get()
        cfg["n_por_clase"] = leer_int(self.v_npc, "Puntos por clase", 1)
        if cfg["tipo_datos"] == "nubes":
            cfg["n_clases"] = len(self.filas_clase)
            cfg["clases"] = []
            for k, fila in enumerate(self.filas_clase):
                cfg["clases"].append({
                    "centro": leer_vector(fila["centro"].get(), "Media de la clase %d" % (k + 1), n),
                    "desviacion": leer_positivo(fila["desv"], "Desviación de la clase %d" % (k + 1), cero=True),
                    "distribucion": fila["dist"].get()})
        elif cfg["tipo_datos"] == "xor":
            cfg["n_clases"] = 2
            cfg["xor_desviacion"] = leer_positivo(self.v_xor_desv, "Desviación de cada esquina", cero=True)
            cfg["xor_lado"] = leer_positivo(self.v_xor_lado, "Lado del cubo")
            cfg["distribucion_xor"] = self.v_xor_dist.get()
            if 2 ** n > 2 * cfg["n_por_clase"]:
                raise ValueError("El XOR con %d entradas tiene %d esquinas: use al menos %d puntos por clase."
                                 % (n, 2 ** n, 2 ** (n - 1)))
        else:
            cfg["n_clases"] = 2
            cfg["esfera_centro"] = leer_float(self.v_esf_centro, "Centro de la esfera")
            cfg["esfera_radio_interno"] = leer_positivo(self.v_esf_r1, "Radio de la esfera")
            cfg["esfera_radio_externo"] = leer_positivo(self.v_esf_r2, "Radio externo")
            if cfg["esfera_radio_externo"] <= 1.3 * cfg["esfera_radio_interno"]:
                raise ValueError("El radio externo debe ser mayor que 1,3 veces el radio de la esfera\n"
                                 "(entre las dos clases se deja un espacio vacío).")
        cfg["ocultas"] = leer_ocultas(self.v_ocultas.get())
        cfg["tasa"] = leer_positivo(self.v_tasa, "Tasa de aprendizaje")
        cfg["ajuste"] = "patron" if self.v_ajuste.get() == TEXTO_AJUSTE["patron"] else "epoca"
        cfg["tipo_momento"] = "clasico" if self.v_tipo_momento.get() == TEXTO_MOMENTO["clasico"] else "profesor"
        cfg["momento"] = leer_float(self.v_momento, "Momento")
        cfg["alfa"] = leer_positivo(self.v_alfa, "α del momento clásico", cero=True)
        if cfg["tipo_momento"] == "clasico" and cfg["alfa"] >= 1:
            raise ValueError("α debe ser menor que 1: con α >= 1 los cambios se acumulan sin control.")
        cfg["max_epocas"] = leer_int(self.v_epocas, "Número máximo de épocas", 1)
        cfg["ecm_objetivo"] = leer_positivo(self.v_ecm, "ECM objetivo", cero=True)
        cfg["usar_sesgo"] = self.v_sesgo.get()
        cfg["x0"] = int(self.v_x0.get())
        cfg["normalizar"] = self.v_normalizar.get()
        cfg["barajar"] = self.v_barajar.get()
        cfg["semilla"] = leer_semilla(self.v_semilla)
        cfg["registrar"] = self.v_registrar.get()
        cfg["epocas_registro"] = leer_int(self.v_epocas_reg, "Épocas a registrar", 1)
        return cfg

    def clasificar_punto(self):
        if self.res is None or self.res["tipo"] != "clasificacion":
            messagebox.showinfo("Primero entrene", "Entrene la red antes de probar un punto.")
            return
        n = len(self.res["X"][0])
        try:
            x = leer_vector(self.v_punto.get(), "Punto nuevo", n)
        except ValueError as error:
            messagebox.showerror("Dato no válido", str(error))
            return
        clase = ex.clasificar_punto(self.res, x)
        y = ex.salida_en_punto(self.res, x)
        self.res["punto"] = (x, clase)
        self.pendientes.add("espacio")
        self.nb_res.select(0)
        self._redibujar_visible()
        self.estado.configure(text="El punto (%s) se clasifica como %s — salidas de la red y = (%s)"
                              % (fmt_vector(x), self.res["nombres"][clase],
                                 ", ".join("%.4f" % v for v in y)))

    # ------------------------------------------------------------------
    # Pestaña EJEMPLO DE CLASE
    # ------------------------------------------------------------------
    def _pestana_ejemplo(self):
        m = MarcoDesplazable(self.nb_config)
        self.nb_config.add(m, text=" Ejemplo de clase ")
        f = m.interior
        ttk.Label(f, justify="left", text=(
            "Reproduce el ejemplo de las diapositivas (XOR con una red 2 → 3 → 1)\n"
            "para comprobar que el código hace las mismas cuentas.")).pack(anchor="w", pady=(0, 8))
        tabla = ttk.Treeview(f, columns=("x1", "x2", "c"), show="headings", height=4)
        for col, txt in zip(("x1", "x2", "c"), ("x1", "x2", "C (deseada)")):
            tabla.heading(col, text=txt)
            tabla.column(col, width=90, anchor="center")
        for x, c in zip(ex.EJEMPLO_X, ex.EJEMPLO_C):
            tabla.insert("", "end", values=(int(x[0]), int(x[1]), c))
        tabla.pack(anchor="w")
        ttk.Label(f, justify="left", text=(
            "\nPesos iniciales (diapositivas):\n"
            "   x1 → n1, n2, n3:  −0,424   −0,740   −0,961\n"
            "   x2 → n1, n2, n3:   0,358   −0,577   −0,469\n"
            "   n1, n2, n3 → y:   −0,017   −0,893    0,148\n"
            "Tasa de aprendizaje = 0,3     Momento = 1     ajuste por época\n\n"
            "Resultado esperado en la época 1 (sin sesgo):\n"
            "   Salidas y = 0,406   0,432   0,437   0,458\n"
            "   DeltaSalida = −0,098   0,139   0,139   −0,114\n"
            "   Pesos nuevos hacia la salida: −0,007   −0,886   0,154\n"
            "   Pesos nuevos de la capa oculta: −0,424  0,358  −0,743  −0,581  −0,961  −0,468\n")
        ).pack(anchor="w")
        opciones = ttk.LabelFrame(f, text="Opciones", padding=8)
        opciones.pack(fill="x", pady=4)
        self.v_ej_sesgo = tk.BooleanVar(value=False)
        ttk.Radiobutton(opciones, text="Sin sesgo (exactamente como en las diapositivas)",
                        value=False, variable=self.v_ej_sesgo).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Radiobutton(opciones, text="Con sesgo θ (inicial 0, x0 = −1)",
                        value=True, variable=self.v_ej_sesgo).grid(row=1, column=0, columnspan=2, sticky="w")
        self.v_ej_epocas, _ = self._campo(opciones, 2, "Número máximo de épocas", "3000",
                                          codigo=ex.ejecutar_ejemplo_clase)
        self.v_ej_ecm, _ = self._campo(opciones, 3, "ECM objetivo", "0,01",
                                       codigo=ex.ejecutar_ejemplo_clase)
        ttk.Label(opciones, style="Nota.TLabel", justify="left", text=(
            "Sin sesgo, la entrada (0, 0) siempre da 0,5 en las neuronas ocultas y la red\n"
            "se queda en 75 % de exactitud. Con sesgo aprende el XOR completo.")).grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self._ver_codigo(ttk.Button(f, text="▶  REPRODUCIR EJEMPLO", style="Accion.TButton",
                                    command=self.reproducir_ejemplo),
                         ex.ejecutar_ejemplo_clase).pack(fill="x", pady=12)
        ttk.Label(f, style="Nota.TLabel", justify="left", text=(
            "Después de reproducirlo: mueva la barra 'Época mostrada' a 1 y abra la\n"
            "pestaña Red para ver los pesos nuevos, o 'Proceso paso a paso' para ver\n"
            "las cuentas de cada patrón y del ajuste.")).pack(anchor="w")

    # ------------------------------------------------------------------
    # Barra de épocas (animación)
    # ------------------------------------------------------------------
    def _barra_epocas(self, padre):
        self.estado = ttk.Label(padre, text="", foreground=AZUL, font=("Segoe UI", 10, "bold"),
                                padding=(6, 0, 6, 4))
        self.estado.pack(side="bottom", fill="x")
        barra = ttk.Frame(padre, padding=(4, 6))
        barra.pack(side="bottom", fill="x")
        ttk.Label(barra, text="Época mostrada:").pack(side="left")
        self.escala = tk.Scale(barra, from_=0, to=1, orient="horizontal", resolution=1,
                               showvalue=True, length=260, background=FONDO,
                               highlightthickness=0, command=self._mover_epoca)
        self.escala.pack(side="left", padx=6)
        self._ver_codigo(self.escala, RedNeuronal.pesos_en_epoca)
        self._ver_codigo(ttk.Button(barra, text="▶ Animar entrenamiento", command=self.animar),
                         self._paso_animacion).pack(side="left", padx=3)
        ttk.Button(barra, text="■ Detener", command=self.detener).pack(side="left", padx=3)
        self._ver_codigo(ttk.Button(barra, text="Resultado final", command=self.mostrar_final),
                         self.mostrar_final).pack(side="left", padx=3)
        self.estado.configure(text="Consejo: clic derecho sobre un botón, un campo o una pestaña "
                                   "de resultados abre su código en VS Code.")

    def _max_epocas(self):
        return self.res["red"].epocas

    def _mover_epoca(self, valor):
        if self.res is None:
            return
        nueva = int(float(valor))
        if nueva >= self._max_epocas() and not self.animando:
            nueva = None
        if nueva == self.epoca:
            return
        self.epoca = nueva
        self.pendientes = {clave for clave, _, _ in self.graficas}
        self._redibujar_visible()

    def animar(self):
        if self.res is None or self.entrenando:
            return
        self.animando = True
        self.escala.set(0)
        self._paso_animacion(0)

    def _paso_animacion(self, k):
        """Avanza la animación; con muchas épocas salta de a varias para que
        dure unos 60 cuadros."""
        if not self.animando or self.res is None:
            return
        total = self._max_epocas()
        self.escala.set(k)
        self._mover_epoca(k)
        self.raiz.update_idletasks()
        if k < total:
            salto = maximo([1, total // 60])
            siguiente = k + salto if k + salto < total else total
            self.raiz.after(60, lambda: self._paso_animacion(siguiente))
        else:
            self.animando = False

    def detener(self):
        self.animando = False

    def mostrar_final(self):
        self.animando = False
        if self.res is None:
            return
        self.escala.set(self._max_epocas())
        self.epoca = None
        self.pendientes = {clave for clave, _, _ in self.graficas}
        self._redibujar_visible()

    # ------------------------------------------------------------------
    # Reiniciar: valores por defecto y resultados en blanco
    # ------------------------------------------------------------------
    def reiniciar(self, preguntar=True):
        if self.entrenando:
            return
        if preguntar and not messagebox.askyesno(
                "Reiniciar todo",
                "Se borrarán los resultados y todos los parámetros volverán a sus "
                "valores iniciales.\n\n¿Desea continuar?"):
            return
        self.animando = False
        cfg = ex.config_por_defecto()
        self.v_tipo.set(cfg["tipo_datos"])
        self.v_nentradas.set(str(cfg["n_entradas"]))
        self.v_nclases.set(str(cfg["n_clases"]))
        self.v_npc.set(str(cfg["n_por_clase"]))
        self.filas_clase = [{"centro": tk.StringVar(value=fmt_vector(c["centro"])),
                             "desv": tk.StringVar(value=fmt(c["desviacion"])),
                             "dist": tk.StringVar(value=c["distribucion"])} for c in cfg["clases"]]
        self.v_xor_desv.set(fmt(cfg["xor_desviacion"]))
        self.v_xor_lado.set(fmt(cfg["xor_lado"]))
        self.v_xor_dist.set(cfg["distribucion_xor"])
        self.v_esf_centro.set(fmt(cfg["esfera_centro"]))
        self.v_esf_r1.set(fmt(cfg["esfera_radio_interno"]))
        self.v_esf_r2.set(fmt(cfg["esfera_radio_externo"]))
        self.v_ocultas.set("; ".join(str(n) for n in cfg["ocultas"]))
        self.v_tasa.set(fmt(cfg["tasa"]))
        self.v_ajuste.set(TEXTO_AJUSTE[cfg["ajuste"]])
        self.v_tipo_momento.set(TEXTO_MOMENTO[cfg["tipo_momento"]])
        self.v_momento.set(fmt(cfg["momento"]))
        self.v_alfa.set(fmt(cfg["alfa"]))
        self.v_epocas.set(str(cfg["max_epocas"]))
        self.v_ecm.set(fmt(cfg["ecm_objetivo"]))
        self.v_sesgo.set(cfg["usar_sesgo"])
        self.v_x0.set(str(cfg["x0"]))
        self.v_normalizar.set(cfg["normalizar"])
        self.v_barajar.set(cfg["barajar"])
        self.v_semilla.set("")
        self.v_registrar.set(cfg["registrar"])
        self.v_epocas_reg.set(str(cfg["epocas_registro"]))
        self.v_punto.set("5; 5; 5")
        self.v_ej_sesgo.set(False)
        self.v_ej_epocas.set("3000")
        self.v_ej_ecm.set("0,01")
        self._construir_tabla()
        self._cambio_tipo()
        self._activar_momento()

        self.res = None
        self.epoca = None
        self.pendientes = set()
        for _, pestana, _ in self.graficas:
            pestana.fig.clear()
            pestana.barra.set_message("")
            pestana.canvas.draw_idle()
        self.t_proceso.mostrar("")
        self.t_resumen.mostrar("Configure los datos y la red a la izquierda y presione ENTRENAR.")
        self.escala.configure(to=1)
        self.escala.set(0)
        self.estado.configure(text="Todo reiniciado: valores por defecto y sin resultados.")
        self.nb_config.select(0)
        self.nb_res.select(0)

    # ------------------------------------------------------------------
    # Entrenamiento (en un hilo aparte para que la ventana no se congele)
    # ------------------------------------------------------------------
    def _progreso(self, epoca, total, ecm):
        self.avance = (epoca, total, ecm)
        return self.parar

    def _ejecutar(self, funcion, despues=None):
        """funcion(progreso) entrena y devuelve el resultado. Corre en otro hilo;
        aquí solo se revisa cada 150 ms cómo va."""
        if self.entrenando:
            return
        self.animando = False
        self.entrenando = True
        self.parar = False
        self.avance = None
        self.btn_entrenar.configure(state="disabled")
        self.btn_parar.configure(state="normal")
        self.raiz.configure(cursor="watch")
        resultado = {}

        def trabajo():
            try:
                resultado["res"] = funcion(self._progreso)
            except Exception as error:          # se muestra en la ventana
                resultado["error"] = error

        hilo = threading.Thread(target=trabajo, daemon=True)
        hilo.start()

        def revisar():
            if hilo.is_alive():
                if self.avance:
                    ep, total, ecm = self.avance
                    self.estado.configure(text="Entrenando... época %d de %d   ECM = %s"
                                          % (ep, total, num(ecm)))
                else:
                    self.estado.configure(text="Generando datos y entrenando...")
                self.raiz.after(150, revisar)
                return
            self.entrenando = False
            self.btn_entrenar.configure(state="normal")
            self.btn_parar.configure(state="disabled")
            self.raiz.configure(cursor="")
            if "error" in resultado:
                messagebox.showerror("Error", str(resultado["error"]))
                self.estado.configure(text="No se pudo entrenar: " + str(resultado["error"]))
                return
            self._mostrar(resultado["res"])
            if despues:
                despues()

        self.raiz.after(150, revisar)

    def detener_entrenamiento(self):
        self.parar = True

    def _mostrar(self, res):
        self.res = res
        self.epoca = None
        self.escala.configure(to=maximo([self._max_epocas(), 1]))
        self.escala.set(self._max_epocas())
        self.epoca = None
        self.pendientes = {clave for clave, _, _ in self.graficas}
        self._redibujar_visible()
        registro = res.get("registro")
        if registro:
            self.t_proceso.mostrar("\n".join(registro))
        else:
            self.t_proceso.mostrar("Active 'Registrar el proceso paso a paso' antes de entrenar para ver,\n"
                                   "patrón por patrón, las salidas de cada capa, el error, DeltaSalida,\n"
                                   "DeltaCamada y el ajuste de cada peso (como en las diapositivas).\n"
                                   "Recomendado con pocos puntos, por ejemplo 5 por clase.")
        self.t_resumen.mostrar(ex.resumen(res))
        por_clase = ", ".join("%s: %.0f %%" % (n, e["exactitud"])
                              for n, e in zip(res["nombres"], res["estadisticas"]))
        self.estado.configure(text="Red %s %s  |  ECM %s  |  exactitud %.1f %% (%s)"
                              % (res["red"].arquitectura(), res["red"].estado(),
                                 num(res["red"].historial_ecm[-1]), res["exactitud"], por_clase))

    def entrenar(self):
        if self.entrenando:
            return
        try:
            cfg = self._config()
        except ValueError as error:
            messagebox.showerror("Dato no válido", str(error))
            self.estado.configure(text="Dato no válido: corrija el campo y vuelva a entrenar.")
            return
        self._ejecutar(lambda progreso: ex.ejecutar(cfg, progreso))

    def reproducir_ejemplo(self):
        if self.entrenando:
            return
        try:
            epocas = leer_int(self.v_ej_epocas, "Número máximo de épocas", 1)
            ecm = leer_positivo(self.v_ej_ecm, "ECM objetivo", cero=True)
        except ValueError as error:
            messagebox.showerror("Dato no válido", str(error))
            return
        con_sesgo = self.v_ej_sesgo.get()
        self._ejecutar(lambda progreso: ex.ejecutar_ejemplo_clase(con_sesgo, epocas, ecm, progreso),
                       lambda: self.nb_res.select(self.t_proceso.marco))

    def _redibujar_visible(self):
        if self.res is None:
            return
        actual = self.nb_res.index(self.nb_res.select())
        if actual >= len(self.graficas):
            return
        clave, pestana, dibujo = self.graficas[actual]
        if clave not in self.pendientes:
            return
        self.pendientes.discard(clave)
        if clave == "espacio":
            dibujo(pestana.fig, self.res, self.epoca, pestana.vista_3d())
        else:
            dibujo(pestana.fig, self.res, self.epoca)
        if clave in ("errores", "capas"):
            pestana.fig.tight_layout(rect=(0, 0, 1, 0.95))
        elif clave == "todos":
            pestana.fig.tight_layout()
        pestana.canvas.draw_idle()


def leer_vector_seguro(texto):
    try:
        return leer_vector(texto, "")
    except ValueError:
        return []


def main():
    try:  # pantallas de alta resolución en Windows: texto nítido
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    raiz = tk.Tk()
    Aplicacion(raiz)
    raiz.mainloop()


if __name__ == "__main__":
    main()
