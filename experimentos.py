"""
experimentos.py
===============
Une la simulación de datos (aleatorio.py) con la red neuronal
(red_neuronal.py). No dibuja nada ni usa librerías: devuelve los resultados
para que la interfaz (main.py) los muestre.
"""

import copy

from aleatorio import (GeneradorAleatorio, generar_nubes, generar_xor,
                       generar_esfera, centros_automaticos)
from red_neuronal import RedNeuronal, num
from formulas import (normalizar, exactitud, exactitud_por_clase, media,
                      desviacion_estandar, minimo, maximo, clase_de_la_salida)

# ======================================================================
# CONFIGURACIÓN POR DEFECTO
# ======================================================================
CONFIG = {
    # --- datos ---
    "n_entradas": 3,
    "n_clases": 3,
    "n_por_clase": 100,           # mismo número de puntos en todas las clases
    "tipo_datos": "nubes",        # "nubes", "xor" o "esfera"
    "clases": [
        {"centro": [2.0, 2.0, 2.0], "desviacion": 1.0, "distribucion": "normal"},
        {"centro": [8.0, 2.0, 5.0], "desviacion": 1.0, "distribucion": "normal"},
        {"centro": [5.0, 8.0, 8.0], "desviacion": 1.0, "distribucion": "normal"},
    ],
    "xor_desviacion": 0.7,        # nubes de las esquinas del XOR
    "xor_lado": 6.0,
    "esfera_centro": 5.0,
    "esfera_radio_interno": 2.0,
    "esfera_radio_externo": 5.0,
    "distribucion_xor": "normal",
    # --- red ---
    "ocultas": [5],               # neuronas de cada capa oculta
    "tasa": 0.1,
    "momento": 1.0,               # F8 (fórmula del profesor)
    "tipo_momento": "profesor",   # "profesor" (F8) o "clasico" (F9)
    "alfa": 0.9,                  # F9
    "ajuste": "epoca",            # "epoca" (lotes, el del profesor) o "patron"
    "max_epocas": 1000,
    "ecm_objetivo": 0.001,
    "usar_sesgo": True,
    "x0": -1,
    "normalizar": True,
    "barajar": True,              # solo en el ajuste por patrón
    "semilla": None,
    "registrar": False,
    "epocas_registro": 2,
}

# Ejemplo de las diapositivas: XOR con red 2-3-1 sin sesgo
EJEMPLO_X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
EJEMPLO_C = [0, 1, 1, 0]
EJEMPLO_PESOS = [
    [[-0.424, 0.358], [-0.740, -0.577], [-0.961, -0.469]],   # x1, x2 -> neuronas ocultas 1, 2, 3
    [[-0.017, -0.893, 0.148]],                               # ocultas -> salida
]
EJEMPLO_TASA = 0.3
EJEMPLO_MOMENTO = 1.0


def config_por_defecto():
    return copy.deepcopy(CONFIG)


def nombres_clases(n):
    return ["Clase %d" % (k + 1) for k in range(n)]


def salidas_deseadas(etiquetas, n_clases):
    """Con 2 clases hay UNA neurona de salida: d = 0 (clase 1) o d = 1 (clase 2),
    como en el XOR de clase. Con 3 o más clases hay una neurona por clase y la
    salida deseada es 1 en la neurona de su clase y 0 en las demás."""
    if n_clases == 2:
        return [[float(c)] for c in etiquetas]
    return [[1.0 if k == c else 0.0 for k in range(n_clases)] for c in etiquetas]


# ======================================================================
def generar_datos(config, gen):
    n = config["n_entradas"]
    tipo = config["tipo_datos"]
    if tipo == "xor":
        X, etiquetas = generar_xor(gen, config["n_por_clase"], n, config["xor_desviacion"],
                                   config["distribucion_xor"], config["xor_lado"])
        return X, etiquetas, 2
    if tipo == "esfera":
        X, etiquetas = generar_esfera(gen, config["n_por_clase"], n, config["esfera_centro"],
                                      config["esfera_radio_interno"],
                                      config["esfera_radio_externo"])
        return X, etiquetas, 2
    clases = config["clases"][:config["n_clases"]]
    X, etiquetas = generar_nubes(gen, config["n_por_clase"], clases)
    return X, etiquetas, config["n_clases"]


def ejecutar(config, progreso=None):
    gen = GeneradorAleatorio(config["semilla"])
    X, etiquetas, n_clases = generar_datos(config, gen)
    n = config["n_entradas"]

    # Normalización mín-máx de cada entrada (F16): la sigmoide trabaja mejor
    # con entradas entre 0 y 1. Las gráficas se dibujan en las unidades originales.
    if config["normalizar"]:
        mins = [minimo([x[j] for x in X]) for j in range(n)]
        maxs = [maximo([x[j] for x in X]) for j in range(n)]
    else:
        mins, maxs = None, None
    Xr = [entrada_de_la_red(mins, maxs, x) for x in X]
    D = salidas_deseadas(etiquetas, n_clases)

    estructura = [n] + list(config["ocultas"]) + [1 if n_clases == 2 else n_clases]
    red = RedNeuronal(estructura, gen, config["tasa"], config["momento"],
                      config["tipo_momento"], config["alfa"], config["ajuste"],
                      config["usar_sesgo"], config["x0"])
    registro = [] if config["registrar"] else None
    red.entrenar(Xr, D, etiquetas, config["max_epocas"], config["ecm_objetivo"],
                 config["barajar"], registro, config["epocas_registro"], progreso)

    predicciones = [red.clasificar(x) for x in Xr]                             # F14
    estadisticas = []
    for k in range(n_clases):
        pts = [x for x, e in zip(X, etiquetas) if e == k]
        estadisticas.append({
            "n": len(pts),
            "media": [media([q[j] for q in pts]) for j in range(n)],              # F22
            "desviacion": media([desviacion_estandar([q[j] for q in pts]) for j in range(n)]),  # F23
            "exactitud": exactitud_por_clase(predicciones, etiquetas, k)})       # F13

    titulos = {"nubes": "Nubes", "xor": "XOR en %d dimensiones" % n, "esfera": "Esfera y cáscara"}
    return {"tipo": "clasificacion", "X": X, "etiquetas": etiquetas, "n_clases": n_clases,
            "nombres": nombres_clases(n_clases), "red": red, "mins": mins, "maxs": maxs,
            "punto": None, "registro": registro, "estadisticas": estadisticas,
            "exactitud": exactitud(predicciones, etiquetas),                       # F12
            "titulo": "%s — red %s" % (titulos[config["tipo_datos"]], red.arquitectura()),
            "config": copy.deepcopy(config), "cache": {}}


def ejecutar_ejemplo_clase(con_sesgo=False, max_epocas=3000, ecm_objetivo=0.01, progreso=None):
    """Reproduce el ejemplo de las diapositivas (XOR, red 2-3-1, tasa 0,3,
    momento 1, ajuste por época). Sin sesgo da exactamente los números de clase:
    salida 0,406 / 0,432 / 0,437 / 0,458, DeltaS = -0,098 / 0,139 / 0,139 / -0,114,
    pesos nuevos de la salida -0,007 / -0,886 / 0,154."""
    gen = GeneradorAleatorio(1)
    red = RedNeuronal([2, 3, 1], gen, EJEMPLO_TASA, EJEMPLO_MOMENTO, "profesor", 0.9,
                      "epoca", usar_sesgo=con_sesgo, x0=-1, nombre="Ejemplo de clase")
    red.poner_pesos(EJEMPLO_PESOS, [[0.0, 0.0, 0.0], [0.0]])
    registro = []
    red.entrenar(EJEMPLO_X, salidas_deseadas(EJEMPLO_C, 2), EJEMPLO_C, max_epocas,
                 ecm_objetivo, False, registro, 2, progreso)
    predicciones = [red.clasificar(x) for x in EJEMPLO_X]
    return {"tipo": "ejemplo", "X": EJEMPLO_X, "etiquetas": EJEMPLO_C, "n_clases": 2,
            "nombres": ["C = 0", "C = 1"], "red": red, "mins": None, "maxs": None,
            "punto": None, "registro": registro,
            "estadisticas": [{"n": 2, "exactitud": exactitud_por_clase(predicciones, EJEMPLO_C, k)}
                             for k in range(2)],
            "exactitud": exactitud(predicciones, EJEMPLO_C),
            "titulo": "Ejemplo de clase: XOR, red 2 → 3 → 1 %s" % ("con sesgo" if con_sesgo else "sin sesgo"),
            "config": None, "cache": {}}


# ======================================================================
def entrada_de_la_red(mins, maxs, x):
    """Punto en unidades originales -> entrada que recibe la red (F16)."""
    if mins is None:
        return list(x)
    return [normalizar(x[j], mins[j], maxs[j]) for j in range(len(x))]


def salida_en_punto(res, x, epoca=None):
    return res["red"].salida(entrada_de_la_red(res["mins"], res["maxs"], x), epoca)


def clasificar_punto(res, x):
    return clase_de_la_salida(salida_en_punto(res, x))                          # F14


# ======================================================================
def resumen(res):
    """Texto con los resultados y las fórmulas que se usaron."""
    red = res["red"]
    L = ["RESULTADOS: " + res["titulo"], "=" * 70, ""]
    ne, no, ns = red.estructura[0], red.n_capas - 1, red.estructura[-1]
    L.append("Arquitectura: %s   (%d entrada%s, %d capa%s oculta%s, %d salida%s)" % (
        red.arquitectura(), ne, "" if ne == 1 else "s", no, "" if no == 1 else "s",
        "" if no == 1 else "s", ns, "" if ns == 1 else "s"))
    if red.tipo_momento == "clasico":
        regla = "momento clásico (F9), tasa = %s, alfa = %s" % (num(red.tasa), num(red.alfa))
    else:
        regla = "fórmula del profesor (F8), tasa = %s, momento = %s" % (num(red.tasa), num(red.momento))
    L.append("Ajuste: por %s, %s" % ("época (lotes)" if red.ajuste == "epoca" else "patrón", regla))
    L.append("Sesgo: %s" % ("sí, entrada fija x0 = %d" % red.x0 if red.usar_sesgo else "no"))
    if res["mins"] is not None:
        L.append("Entradas normalizadas a [0, 1] con F16 (mín-máx de cada entrada)")
    L.append("")
    L.append("Entrenamiento: %s" % red.estado())
    L.append("   Error de la época E (F10) inicial = %s   final = %s"
             % (num(red.historial_error[0]), num(red.historial_error[-1])))
    L.append("   ECM (F11) inicial = %s   final = %s"
             % (num(red.historial_ecm[0]), num(red.historial_ecm[-1])))
    L.append("   Exactitud global (F12) = %.2f %%" % res["exactitud"])
    for nombre, est in zip(res["nombres"], res["estadisticas"]):
        L.append("      %s: exactitud de la clase (F13) = %.2f %%" % (nombre, est["exactitud"]))
    if res["tipo"] == "clasificacion" and res["config"]["tipo_datos"] == "nubes":
        L.append("")
        L.append("Datos obtenidos en cada clase (comprobación de la simulación):")
        for nombre, clase, est in zip(res["nombres"], res["config"]["clases"], res["estadisticas"]):
            L.append("   %s: %d puntos | media pedida (%s) -> obtenida (%s)"
                     % (nombre, est["n"], "; ".join("%.2f" % v for v in clase["centro"]),
                        "; ".join("%.2f" % v for v in est["media"])))
            L.append("      desviación pedida %.2f -> obtenida %.2f (F23)"
                     % (clase["desviacion"], est["desviacion"]))
    L.append("")
    L.append("PESOS FINALES")
    for c in range(red.n_capas):
        L.append("   " + red.nombre_capa(c) + ":")
        nombres = red.nombres_pesos(c)
        valores = red.valores_capa(c, red.epocas)
        for nombre, v in zip(nombres, valores):
            L.append("      %-12s = %s" % (nombre, num(v)))
    L += ["", "FÓRMULAS UTILIZADAS (formulas.py)", "-" * 70,
          "F1  S = Σ w*x + θ*x0                 F2  y = 1 / (1 + e^(-S))",
          "F3  y' = y (1 - y)                   F4  e = d - y",
          "F5  DeltaSalida = e * y'             F6  DeltaCamada = y' * Σ(peso * Delta siguiente)",
          "F7  G = Σ_patrones entrada * delta",
          "F8  Peso(n+1) = Peso(n)*Momento + G*Tasa                 (profesor)",
          "F9  DeltaW(n) = Tasa*G + alfa*DeltaW(n-1); Peso(n+1) = Peso(n) + DeltaW(n)   (clásico)",
          "F10 E = ½ Σ e²                       F11 ECM = Σ e² / (N*m)",
          "F12 exactitud = 100*aciertos/N       F13 exactitud de la clase k",
          "F14 clase: 1 salida -> y >= 0,5 ; varias -> mayor y",
          "F15 superficie: 1 salida -> y = 0,5 ; varias -> y_k = max(y_j, j != k)",
          "F16 x' = (x - min) / (max - min)",
          "M10 e^x = 2^k * e^r (serie de Taylor), sin librerías"]
    return "\n".join(L)


def centros_por_defecto(n_clases, n_entradas):
    """Centros para la tabla de la interfaz: con 3 entradas y hasta 3 clases se
    usan los de la entrega 1; en otro caso se reparten automáticamente."""
    if n_entradas == 3 and n_clases <= 3:
        return [c["centro"][:] for c in CONFIG["clases"][:n_clases]]
    return centros_automaticos(n_clases, n_entradas)
