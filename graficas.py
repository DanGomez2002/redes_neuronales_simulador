"""
graficas.py
===========
Todas las funciones reciben una figura (matplotlib.figure.Figure) y dibujan
dentro de ella, para poder mostrarlas en la interfaz. matplotlib solo se usa
para DIBUJAR: la superficie de separación se calcula aquí a mano evaluando la
red en una rejilla y buscando dónde cambia de clase (marching tetrahedra).

Pestañas:
  1. Espacio      3 entradas: puntos 3D y superficie de separación (F15)
                  2 entradas: plano con las regiones de cada clase y su frontera
  2. Errores      error de la época (F10), ECM (F11) y exactitud (F12) por época
  3. Pesos por capa   la evolución de los pesos, una gráfica por capa
  4. Todos los pesos  todos los pesos de la red en una sola gráfica
  5. Red          el diagrama de la red con los pesos de la época elegida
"""

from matplotlib.colors import to_rgb, ListedColormap
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from formulas import (media, minimo, maximo, valor_absoluto, raiz_cuadrada,
                      funcion_frontera, clase_de_la_salida, es_finito)
import experimentos as ex

COLORES_CLASE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                 "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"]
COLOR_FRONTERA = "#6a3d9a"
FAMILIAS_CAPA = [  # tonos para distinguir las capas en "Todos los pesos"
    ["#08306b", "#2171b5", "#4292c6", "#6baed6", "#9ecae1"],
    ["#7f2704", "#d94801", "#f16913", "#fd8d3c", "#fdae6b"],
    ["#00441b", "#238b45", "#41ab5d", "#74c476", "#a1d99b"],
    ["#3f007d", "#6a51a3", "#807dba", "#9e9ac8", "#bcbddc"],
    ["#67000d", "#cb181d", "#ef3b2c", "#fb6a4a", "#fc9272"],
]
RESOLUCION_3D = 18
RESOLUCION_2D = 90


def color_clase(k):
    return COLORES_CLASE[k % len(COLORES_CLASE)]


# ----------------------------------------------------------------------
# Utilidades (sin librerías)
# ----------------------------------------------------------------------
def _rango(valores, margen=0.1):
    lo, hi = minimo(valores), maximo(valores)
    extra = (hi - lo) * margen if hi > lo else 1.0
    return lo - extra, hi + extra


def _rejilla(lo, hi, n):
    """n+1 valores igualmente espaciados entre lo y hi."""
    paso = (hi - lo) / n
    return [lo + i * paso for i in range(n + 1)]


def _epoca_real(res, epoca):
    red = res["red"]
    return red.epocas if epoca is None or epoca > red.epocas else epoca


def _titulo(res, epoca):
    red = res["red"]
    if epoca is None or epoca >= red.epocas:
        return "%s — %s" % (res["titulo"], red.estado())
    return "%s — época %d" % (res["titulo"], epoca)


def _limites(res):
    puntos = res["X"] + ([res["punto"][0]] if res.get("punto") else [])
    n = len(puntos[0])
    return [_rango([p[j] for p in puntos]) for j in range(n)]


def _salidas_en_rejilla(res, epoca, ejes):
    """Evalúa la red en todos los puntos de la rejilla (en unidades originales).
    El resultado se guarda en res['cache'] para no repetir el cálculo al
    cambiar de pestaña."""
    clave = ("rejilla", epoca, len(ejes[0]))
    if clave in res["cache"]:
        return res["cache"][clave]
    red, mins, maxs = res["red"], res["mins"], res["maxs"]
    pesos, sesgos = red.pesos_en_epoca(epoca)
    from red_neuronal import propagar_con
    salidas = {}
    if len(ejes) == 3:
        for i, a in enumerate(ejes[0]):
            for j, b in enumerate(ejes[1]):
                for k, c in enumerate(ejes[2]):
                    x = ex.entrada_de_la_red(mins, maxs, [a, b, c])
                    salidas[(i, j, k)] = propagar_con(pesos, sesgos, red.x0, x)[-1]
    else:
        for i, a in enumerate(ejes[0]):
            for j, b in enumerate(ejes[1]):
                x = ex.entrada_de_la_red(mins, maxs, [a, b])
                salidas[(i, j)] = propagar_con(pesos, sesgos, red.x0, x)[-1]
    if len(res["cache"]) > 60:      # no guardar demasiadas épocas en memoria
        res["cache"].clear()
    res["cache"][clave] = salidas
    return salidas


# ----------------------------------------------------------------------
# Superficie de separación en 3D: marching tetrahedra
# ----------------------------------------------------------------------
# Esquinas de un cubo de la rejilla y su división en 6 tetraedros que comparten
# la diagonal 0-6. En cada tetraedro la función frontera (F15) se aproxima por
# un plano, así que la superficie f = 0 lo corta en un triángulo o en un
# cuadrilátero (dos triángulos).
_ESQUINAS = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
             (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
_TETRAEDROS = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
               (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)]


def _corte(p, q, fp, fq):
    """Punto de la arista p-q donde f = 0 (interpolación lineal):
       t = fp / (fp - fq),   punto = p + t (q - p)"""
    t = fp / (fp - fq)
    return [p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]), p[2] + t * (q[2] - p[2])]


def triangulos_superficie(ejes, valor, incluir=None):
    """Triángulos de la superficie valor = 0.
    ejes: 3 listas con las coordenadas de la rejilla; valor(i, j, k): f en el nodo.
    incluir(i, j, k): si se da, solo se revisan los cubos con al menos una
    esquina donde incluir es verdadero."""
    n0, n1, n2 = len(ejes[0]) - 1, len(ejes[1]) - 1, len(ejes[2]) - 1
    triangulos = []
    for i in range(n0):
        for j in range(n1):
            for k in range(n2):
                if incluir is not None:
                    alguna = False
                    for a, b, c in _ESQUINAS:
                        if incluir(i + a, j + b, k + c):
                            alguna = True
                            break
                    if not alguna:
                        continue
                f = [valor(i + a, j + b, k + c) for a, b, c in _ESQUINAS]
                positivos = 0
                for v in f:
                    if v > 0:
                        positivos += 1
                if positivos == 0 or positivos == 8:
                    continue                    # el cubo no toca la superficie
                p = [[ejes[0][i + a], ejes[1][j + b], ejes[2][k + c]] for a, b, c in _ESQUINAS]
                for tetra in _TETRAEDROS:
                    pos = [v for v in tetra if f[v] > 0]
                    neg = [v for v in tetra if f[v] <= 0]
                    if len(pos) == 0 or len(neg) == 0:
                        continue
                    if len(pos) == 1 or len(neg) == 1:
                        solo, otros = (pos[0], neg) if len(pos) == 1 else (neg[0], pos)
                        triangulos.append([_corte(p[solo], p[o], f[solo], f[o]) for o in otros])
                    else:
                        a, b = pos
                        c, d = neg
                        q = [_corte(p[a], p[c], f[a], f[c]), _corte(p[a], p[d], f[a], f[d]),
                             _corte(p[b], p[d], f[b], f[d]), _corte(p[b], p[c], f[b], f[c])]
                        triangulos.append([q[0], q[1], q[2]])
                        triangulos.append([q[0], q[2], q[3]])
    return triangulos


def _sombreado(triangulos, color, alfa, escala):
    """Color de cada triángulo según hacia dónde mira (producto cruz = normal),
    para que la superficie se vea con volumen. escala = tamaño de cada eje,
    para que la normal no dependa de las unidades."""
    base = to_rgb(color)
    luz = (0.4, 0.3, 0.87)
    colores = []
    for t in triangulos:
        u = [(t[1][m] - t[0][m]) / escala[m] for m in range(3)]
        v = [(t[2][m] - t[0][m]) / escala[m] for m in range(3)]
        n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
        largo = raiz_cuadrada(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])
        if largo == 0:
            intensidad = 0.8
        else:
            intensidad = 0.55 + 0.45 * valor_absoluto(
                (n[0] * luz[0] + n[1] * luz[1] + n[2] * luz[2]) / largo)
        colores.append((base[0] * intensidad, base[1] * intensidad, base[2] * intensidad, alfa))
    return colores


# ----------------------------------------------------------------------
# 1. Espacio
# ----------------------------------------------------------------------
def dibujar_espacio(fig, res, epoca=None, vista=None):
    n = len(res["X"][0])
    if n == 3:
        return _espacio_3d(fig, res, epoca, vista)
    if n == 2:
        return _espacio_2d(fig, res, epoca)
    fig.clear()
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.5, 0.55, "Con %d entrada%s cada punto vive en %d dimensi%s:\n"
                       "no se puede dibujar el espacio ni la superficie de separación."
            % (n, "" if n == 1 else "s", n, "ón" if n == 1 else "ones"),
            ha="center", va="center", fontsize=13, transform=ax.transAxes)
    ax.text(0.5, 0.35, "La red se entrenó igual con todas las entradas.\n"
                       "Las pestañas Errores, Pesos y Red muestran el entrenamiento.\n"
                       "Para ver la superficie use 3 entradas (o 2 para ver la frontera en el plano).",
            ha="center", va="center", fontsize=11, color="#555", transform=ax.transAxes)
    return ax


def _espacio_3d(fig, res, epoca, vista):
    fig.clear()
    ax = fig.add_subplot(111, projection="3d")
    if vista:
        ax.view_init(elev=vista[0], azim=vista[1])
    X, etiquetas = res["X"], res["etiquetas"]
    limites = _limites(res)
    ep = _epoca_real(res, epoca)

    # --- Superficie de separación (F15) ---
    ejes = [_rejilla(limites[m][0], limites[m][1], RESOLUCION_3D) for m in range(3)]
    salidas = _salidas_en_rejilla(res, ep, ejes)
    escala = [limites[m][1] - limites[m][0] for m in range(3)]
    n_salidas = len(salidas[(0, 0, 0)])
    leyenda_extra = []
    if n_salidas == 1:
        tri = triangulos_superficie(ejes, lambda i, j, k: funcion_frontera(salidas[(i, j, k)], 0))
        if tri:
            ax.add_collection3d(Poly3DCollection(tri, facecolors=_sombreado(tri, COLOR_FRONTERA, 0.45, escala),
                                                 edgecolors="none"))
            leyenda_extra.append(Line2D([], [], color=COLOR_FRONTERA, linewidth=8, alpha=0.5,
                                        label="Superficie de separación (y = 0,5)"))
    else:
        # Varias salidas: una superficie por cada par de clases vecinas (a, b),
        # donde y_a = y_b y esas dos son las salidas más altas (F15).
        for a in range(n_salidas):
            for b in range(a + 1, n_salidas):
                def diferencia(i, j, k, a=a, b=b):
                    y = salidas[(i, j, k)]
                    return y[a] - y[b]

                def son_las_dos_mayores(i, j, k, a=a, b=b):
                    y = salidas[(i, j, k)]
                    menor_par = y[a] if y[a] < y[b] else y[b]
                    for m in range(len(y)):
                        if m != a and m != b and y[m] >= menor_par:
                            return False
                    return True

                tri = triangulos_superficie(ejes, diferencia, son_las_dos_mayores)
                # Se conserva cada triángulo solo si en su centro (promedio de
                # los 3 vértices) a y b siguen siendo las dos salidas mayores:
                # así la superficie termina donde empieza la de otro par.
                filtrados = []
                for t in tri:
                    centro = [(t[0][m] + t[1][m] + t[2][m]) / 3.0 for m in range(3)]
                    y = ex.salida_en_punto(res, centro, ep)
                    menor_par = y[a] if y[a] < y[b] else y[b]
                    if all(y[m] < menor_par for m in range(len(y)) if m != a and m != b):
                        filtrados.append(t)
                tri = filtrados
                if tri:
                    ca, cb = to_rgb(color_clase(a)), to_rgb(color_clase(b))
                    mezcla = tuple((ca[m] + cb[m]) / 2 for m in range(3))
                    ax.add_collection3d(Poly3DCollection(tri, facecolors=_sombreado(tri, mezcla, 0.4, escala),
                                                         edgecolors="none"))
                    leyenda_extra.append(Line2D([], [], color=mezcla, linewidth=8, alpha=0.5,
                                                label="Superficie %s | %s" % (res["nombres"][a],
                                                                               res["nombres"][b])))

    # --- Puntos y centroides (el centroide solo tiene sentido en las nubes) ---
    con_centroides = res["config"] is not None and res["config"]["tipo_datos"] == "nubes"
    for c, nombre in enumerate(res["nombres"]):
        pts = [x for x, e in zip(X, etiquetas) if e == c]
        if not pts:
            continue
        ax.scatter([q[0] for q in pts], [q[1] for q in pts], [q[2] for q in pts],
                   color=color_clase(c), s=12, label=nombre, depthshade=False)
        if con_centroides:
            centro = [media([q[j] for q in pts]) for j in range(3)]               # F22
            ax.scatter([centro[0]], [centro[1]], [centro[2]], marker="X", s=150,
                       color=color_clase(c), edgecolors="black", linewidths=1.1, depthshade=False)
    punto = res.get("punto")
    if punto:
        x, clase = punto
        ax.scatter([x[0]], [x[1]], [x[2]], marker="*", s=320, color=color_clase(clase),
                   edgecolors="black", linewidths=1.2, depthshade=False,
                   label="Punto nuevo → " + res["nombres"][clase])
    manejadores, textos = ax.get_legend_handles_labels()
    manejadores += leyenda_extra
    if con_centroides:
        manejadores.append(Line2D([], [], marker="X", linestyle="none", markersize=9,
                                  markerfacecolor="white", markeredgecolor="black", label="Centroide de cada nube"))
    ax.set_xlim(*limites[0]); ax.set_ylim(*limites[1]); ax.set_zlim(*limites[2])
    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("x3")
    ax.set_title(_titulo(res, epoca), fontsize=11)
    ax.legend(handles=manejadores, loc="upper left", fontsize=8)
    return ax


def segmentos_frontera(ejes, valor):
    """Versión en 2D (marching triangles): cada cuadro de la rejilla se parte en
    dos triángulos y en cada uno la frontera f = 0 es un segmento."""
    segmentos = []
    for i in range(len(ejes[0]) - 1):
        for j in range(len(ejes[1]) - 1):
            esquinas = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
            for tri in ((0, 1, 2), (0, 2, 3)):
                nodos = [esquinas[t] for t in tri]
                f = [valor(a, b) for a, b in nodos]
                p = [[ejes[0][a], ejes[1][b]] for a, b in nodos]
                pos = [m for m in range(3) if f[m] > 0]
                neg = [m for m in range(3) if f[m] <= 0]
                if not pos or not neg:
                    continue
                solo, otros = (pos[0], neg) if len(pos) == 1 else (neg[0], pos)
                seg = []
                for o in otros:
                    t = f[solo] / (f[solo] - f[o])
                    seg.append((p[solo][0] + t * (p[o][0] - p[solo][0]),
                                p[solo][1] + t * (p[o][1] - p[solo][1])))
                segmentos.append(seg)
    return segmentos


def _espacio_2d(fig, res, epoca):
    fig.clear()
    ax = fig.add_subplot(111)
    limites = _limites(res)
    if res["tipo"] == "ejemplo":
        limites = [(-0.5, 1.5), (-0.5, 1.5)]
    ep = _epoca_real(res, epoca)
    ejes = [_rejilla(limites[m][0], limites[m][1], RESOLUCION_2D) for m in range(2)]
    salidas = _salidas_en_rejilla(res, ep, ejes)
    n_clases = res["n_clases"]
    # Región de cada clase (F14) como fondo de color claro
    regiones = [[clase_de_la_salida(salidas[(i, j)]) for i in range(len(ejes[0]))]
                for j in range(len(ejes[1]))]
    claros = [tuple(0.75 + 0.25 * v for v in to_rgb(color_clase(k))) for k in range(n_clases)]
    ax.pcolormesh(ejes[0], ejes[1], regiones, cmap=ListedColormap(claros), vmin=-0.5,
                  vmax=n_clases - 0.5, shading="nearest")
    # Frontera (F15)
    n_salidas = len(salidas[(0, 0)])
    for c in range(1 if n_salidas == 1 else n_salidas):
        seg = segmentos_frontera(ejes, lambda i, j, c=c: funcion_frontera(salidas[(i, j)], c))
        ax.add_collection(LineCollection(seg, colors=COLOR_FRONTERA,
                                         linewidths=2))
    for c, nombre in enumerate(res["nombres"]):
        pts = [x for x, e in zip(res["X"], res["etiquetas"]) if e == c]
        tam = 140 if res["tipo"] == "ejemplo" else 14
        ax.scatter([q[0] for q in pts], [q[1] for q in pts], color=color_clase(c), s=tam,
                   edgecolors="black" if res["tipo"] == "ejemplo" else "none", label=nombre, zorder=3)
    if res["tipo"] == "ejemplo":
        for x in res["X"]:
            y = ex.salida_en_punto(res, x, ep)[0]
            ax.annotate("y = %.3f" % y, (x[0], x[1]), textcoords="offset points",
                        xytext=(10, 8), fontsize=10)
    punto = res.get("punto")
    if punto:
        x, clase = punto
        ax.scatter([x[0]], [x[1]], marker="*", s=320, color=color_clase(clase), edgecolors="black",
                   zorder=4, label="Punto nuevo → " + res["nombres"][clase])
    ax.plot([], [], color=COLOR_FRONTERA, linewidth=2, label="Frontera de separación")
    ax.set_xlim(*limites[0]); ax.set_ylim(*limites[1])
    ax.set_xlabel("x1"); ax.set_ylabel("x2")
    ax.set_title(_titulo(res, epoca), fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    return ax


# ----------------------------------------------------------------------
# 2. Errores
# ----------------------------------------------------------------------
def dibujar_errores(fig, res, epoca=None):
    fig.clear()
    red = res["red"]
    epocas = list(range(len(red.historial_error)))
    marcador = "o" if len(epocas) <= 60 else None
    datos = [(red.historial_error, "E = ½ Σ (d − y)²", "Error de la época (F10)", "#1f3a5f"),
             (red.historial_ecm, "ECM = Σ (d − y)² / (N·m)", "Error cuadrático medio (F11)", "#c0392b"),
             (red.historial_exactitud, "Exactitud (%)", "Exactitud (F12)", "#2ca02c")]
    for n, (valores, etiqueta_y, titulo, color) in enumerate(datos):
        ax = fig.add_subplot(1, 3, n + 1)
        finitos = [v if es_finito(v) else None for v in valores]
        ax.plot(epocas, finitos, color=color, linewidth=1.5, marker=marcador, markersize=3)
        if epoca is not None and epoca < red.epocas:
            ax.axvline(epoca, color="black", linestyle="--", linewidth=1)
            if es_finito(valores[epoca]):
                ax.plot([epoca], [valores[epoca]], marker="o", color="black")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        ax.set_xlabel("Época")
        ax.set_ylabel(etiqueta_y)
        ax.set_title(titulo, fontsize=10)
        ax.grid(alpha=0.3)
        ax.text(0.98, 0.95, "final: %s" % ("%.5g" % valores[-1]), transform=ax.transAxes,
                ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    fig.suptitle("Errores por época — %s" % red.estado(), fontsize=11)


# ----------------------------------------------------------------------
# 3 y 4. Pesos
# ----------------------------------------------------------------------
def _series_capa(red, c, hasta):
    """Lista de series: la evolución de cada peso de la capa c, época a época."""
    filas = [red.valores_capa(c, k) for k in range(hasta + 1)]
    return [[fila[m] for fila in filas] for m in range(len(filas[0]))]


def _es_sesgo(nombre):
    return nombre.startswith("θ")


def dibujar_pesos_capas(fig, res, epoca=None):
    """Una gráfica por capa con la evolución de todos sus pesos."""
    fig.clear()
    red = res["red"]
    L = red.n_capas
    columnas = L if L <= 3 else 2
    filas = (L + columnas - 1) // columnas
    epocas = list(range(red.epocas + 1))
    for c in range(L):
        ax = fig.add_subplot(filas, columnas, c + 1)
        nombres = red.nombres_pesos(c)
        series = _series_capa(red, c, red.epocas)
        for m, (nombre, serie) in enumerate(zip(nombres, series)):
            ax.plot(epocas, serie, linewidth=1.2, linestyle="--" if _es_sesgo(nombre) else "-",
                    color=COLORES_CLASE[m % len(COLORES_CLASE)], label=nombre)
        if epoca is not None and epoca < red.epocas:
            ax.axvline(epoca, color="black", linestyle="--", linewidth=1)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        ax.set_xlabel("Época")
        ax.set_ylabel("Valor del peso")
        ax.set_title("%s (%d pesos)" % (red.nombre_capa(c), len(nombres)), fontsize=10)
        ax.grid(alpha=0.3)
        if len(nombres) <= 12:
            ax.legend(fontsize=7, ncol=2 if len(nombres) > 6 else 1, loc="best")
        else:
            ax.text(0.02, 0.02, "%d pesos: los nombres están en la pestaña Red\ny en 'Resumen y fórmulas'"
                    % len(nombres), transform=ax.transAxes, fontsize=7, color="#555", va="bottom")
    fig.suptitle("Evolución de los pesos capa por capa (F8 / F9) — línea punteada: sesgo θ", fontsize=11)


def dibujar_todos_pesos(fig, res, epoca=None):
    """Todos los pesos de la red en la misma gráfica; cada capa con su familia de color."""
    fig.clear()
    ax = fig.add_subplot(111)
    red = res["red"]
    epocas = list(range(red.epocas + 1))
    leyenda = []
    for c in range(red.n_capas):
        familia = FAMILIAS_CAPA[c % len(FAMILIAS_CAPA)]
        nombres = red.nombres_pesos(c)
        for m, (nombre, serie) in enumerate(zip(nombres, _series_capa(red, c, red.epocas))):
            ax.plot(epocas, serie, linewidth=1.1, color=familia[m % len(familia)],
                    linestyle="--" if _es_sesgo(nombre) else "-")
        leyenda.append(Line2D([], [], color=familia[1], linewidth=3,
                              label="%s (%d pesos)" % (red.nombre_capa(c), len(nombres))))
    if red.usar_sesgo:
        leyenda.append(Line2D([], [], color="gray", linestyle="--", label="Sesgos θ"))
    if epoca is not None and epoca < red.epocas:
        ax.axvline(epoca, color="black", linestyle="--", linewidth=1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax.set_xlabel("Época")
    ax.set_ylabel("Valor del peso")
    ax.set_title("Todos los pesos de la red al mismo tiempo — %s" % red.arquitectura(), fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(handles=leyenda, fontsize=9, loc="best")


# ----------------------------------------------------------------------
# 5. Diagrama de la red
# ----------------------------------------------------------------------
def dibujar_red(fig, res, epoca=None):
    fig.clear()
    ax = fig.add_subplot(111)
    ax.axis("off")
    red = res["red"]
    ep = _epoca_real(res, epoca)
    pesos, sesgos = red.pesos_en_epoca(ep)
    estructura = red.estructura
    posiciones = []
    alto = maximo(estructura) - 1
    for c, n in enumerate(estructura):
        # la neurona 1 arriba, como en las diapositivas; cada capa centrada
        posiciones.append([(float(c), alto - (alto - (n - 1)) / 2.0 - i) for i in range(n)])
    todos = [valor_absoluto(w) for capa in pesos for fila in capa for w in fila]
    mayor = maximo(todos) if todos else 1.0
    if mayor == 0:
        mayor = 1.0
    n_aristas = len(todos)
    for c in range(len(pesos)):
        for j in range(len(pesos[c])):
            for i in range(len(pesos[c][j])):
                w = pesos[c][j][i]
                (xa, ya), (xb, yb) = posiciones[c][i], posiciones[c + 1][j]
                ax.plot([xa, xb], [ya, yb], color="#1f77b4" if w >= 0 else "#d62728",
                        linewidth=0.4 + 3.0 * valor_absoluto(w) / mayor, alpha=0.75, zorder=1)
                if n_aristas <= 24:
                    t = 0.25
                    ax.text(xa + t * (xb - xa), ya + t * (yb - ya), "%.3f" % w, fontsize=7,
                            ha="center", va="center", zorder=3,
                            bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="none", alpha=0.8))
    tam = 900 if maximo(estructura) <= 8 else 350
    for c, capa in enumerate(posiciones):
        ax.scatter([p[0] for p in capa], [p[1] for p in capa], s=tam, zorder=4, linewidths=1.2,
                   color="#dbe7f5" if c == 0 else "#ffffff", edgecolors="#1f3a5f")
        for i, (x, y) in enumerate(capa):
            if c == 0:
                texto = "x%d" % (i + 1)
            else:
                texto = "n%d" % (i + 1) if c < len(posiciones) - 1 else "y%d" % (i + 1)
            ax.text(x, y, texto, ha="center", va="center", fontsize=8, zorder=5)
            if c > 0 and red.usar_sesgo and maximo(estructura) <= 10:
                ax.text(x, y - 0.2, "θ=%.3f" % sesgos[c - 1][i], ha="center",
                        va="top", fontsize=7, color="#333", zorder=6,
                        bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="none", alpha=0.85))
    for c in range(len(estructura)):
        nombre = "Entradas" if c == 0 else red.nombre_capa(c - 1)
        ax.text(float(c), alto + 0.7, nombre, ha="center", fontsize=10,
                fontweight="bold", color="#1f3a5f")
    ax.set_xlim(-0.4, len(estructura) - 0.6)
    ax.set_ylim(-0.9, alto + 1.1)
    ax.plot([], [], color="#1f77b4", linewidth=2, label="peso positivo")
    ax.plot([], [], color="#d62728", linewidth=2, label="peso negativo")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Red %s — pesos de la época %d (grosor ∝ |peso|)" % (red.arquitectura(), ep),
                 fontsize=11)

