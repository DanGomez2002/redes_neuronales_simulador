"""
aleatorio.py
============
Simulación de datos SIN librerías de números aleatorios (no se usa random ni
numpy). Las fórmulas vienen de formulas.py (F17 a F21).

Solo se importa `time`, y únicamente para tomar una semilla distinta cada vez
que el usuario deja la semilla en blanco (no hace ningún cálculo matemático).

Todos los conjuntos tienen el MISMO número de puntos por clase y sirven para
cualquier número de entradas (dimensiones).
"""

import time

from formulas import (congruencial_lineal, uniforme_01, normal_box_muller,
                      uniforme_media_desviacion, exponencial_inversa,
                      binomial_estandarizada, raiz_cuadrada, redondear, seno,
                      potencia_entera)

DISTRIBUCIONES = ("normal", "uniforme", "exponencial", "binomial")
MODELOS_REGRESION = ("lineal", "cuadratico", "senoidal")


class GeneradorAleatorio:
    """Generador congruencial lineal (F17) con las constantes de
    'Numerical Recipes':  a = 1664525,  c = 1013904223,  m = 2^32."""

    A = 1664525
    C = 1013904223
    M = 4294967296            # 2^32

    def __init__(self, semilla=None):
        if semilla is None:
            semilla = time.time_ns()
        self.semilla = int(semilla) % self.M
        self.X = self.semilla

    def uniforme01(self):
        """F17: X(k+1) = (a X(k) + c) mod m ;  U = (X + 0.5) / m"""
        self.X = congruencial_lineal(self.X, self.A, self.C, self.M)
        return uniforme_01(self.X, self.M)

    def uniforme(self, a, b):
        """Número uniforme en (a, b):  a + (b - a) * U"""
        return a + (b - a) * self.uniforme01()

    def entero(self, n):
        """Entero en {0, 1, ..., n-1}"""
        return int(self.uniforme01() * n)

    def barajar(self, lista):
        """Algoritmo de Fisher-Yates: desordena la lista en el mismo sitio."""
        for i in range(len(lista) - 1, 0, -1):
            j = self.entero(i + 1)
            lista[i], lista[j] = lista[j], lista[i]

    # --- distribuciones (todas por media mu y desviación sigma) ---------
    def normal(self, mu, sigma):
        return normal_box_muller(self.uniforme01(), self.uniforme01(), mu, sigma)   # F18

    def uniforme_mu_sigma(self, mu, sigma):
        return uniforme_media_desviacion(self.uniforme01(), mu, sigma)              # F19

    def exponencial(self, mu, sigma):
        return exponencial_inversa(self.uniforme01(), mu, sigma)                    # F20

    def binomial(self, mu, sigma, ensayos=20, p=0.5):
        exitos = 0
        for _ in range(ensayos):
            if self.uniforme01() < p:
                exitos += 1
        return binomial_estandarizada(exitos, ensayos, p, mu, sigma)               # F21

    def muestra(self, distribucion, mu, sigma):
        if distribucion == "normal":
            return self.normal(mu, sigma)
        if distribucion == "uniforme":
            return self.uniforme_mu_sigma(mu, sigma)
        if distribucion == "exponencial":
            return self.exponencial(mu, sigma)
        if distribucion == "binomial":
            return self.binomial(mu, sigma)
        raise ValueError("Distribución desconocida: " + str(distribucion))


# ----------------------------------------------------------------------
# Conjuntos de datos
# ----------------------------------------------------------------------
def generar_nubes(generador, n_por_clase, clases):
    """Una nube por clase.  clases: lista de {centro, desviacion, distribucion};
    el centro tiene tantas coordenadas como entradas. Cada coordenada de cada
    punto sale de la distribución elegida con media = centro y la desviación dada.
    Devuelve X (puntos) y etiquetas (0, 1, 2, ... = índice de la clase)."""
    X, etiquetas = [], []
    for k, clase in enumerate(clases):
        for _ in range(n_por_clase):
            X.append([generador.muestra(clase["distribucion"], c, clase["desviacion"])
                      for c in clase["centro"]])
            etiquetas.append(k)
    return X, etiquetas


def generar_xor(generador, n_por_clase, n_entradas, desviacion, distribucion, lado=6.0):
    """XOR en n dimensiones (2 clases): una nube pequeña en cada esquina del cubo
    [0, lado]^n. La clase de una esquina es la paridad de sus coordenadas
    (cuántas valen 'lado'): par -> clase 1, impar -> clase 2. Con 2 entradas es
    exactamente la tabla del XOR de la clase. Ningún plano lo separa: hace falta
    una capa oculta (por eso el perceptrón no puede)."""
    esquinas = [[], []]
    for numero in range(int(potencia_entera(2.0, n_entradas))):   # 2^n esquinas
        esquina, unos, resto = [], 0, numero
        for _ in range(n_entradas):
            bit = resto % 2
            resto = resto // 2
            esquina.append(lado * bit)
            unos += bit
        esquinas[unos % 2].append(esquina)
    X, etiquetas = [], []
    for k in range(2):
        for p in range(n_por_clase):
            centro = esquinas[k][p % len(esquinas[k])]     # se reparten por igual
            X.append([generador.muestra(distribucion, c, desviacion) for c in centro])
            etiquetas.append(k)
    return X, etiquetas


def centros_automaticos(n_clases, n_entradas, semilla=12345, separacion=6.0):
    """Centros repartidos para n clases y n entradas: cada coordenada al azar en
    [0, 10], repitiendo hasta que cada centro quede a más de 'separacion' de los
    anteriores (o se agoten los intentos). Usa el mismo generador congruencial."""
    gen = GeneradorAleatorio(semilla)
    centros = []
    for _ in range(n_clases):
        mejor, mejor_dist = None, -1.0
        for _ in range(200):
            # coordenada con un decimal:  redondeo(10*u) / 10
            c = [redondear(10.0 * gen.uniforme(0.0, 10.0)) / 10.0 for _ in range(n_entradas)]
            dist = 1e9
            for otro in centros:          # distancia al centro más cercano
                suma = 0.0
                for a, b in zip(c, otro):
                    suma = suma + (a - b) * (a - b)
                d = raiz_cuadrada(suma)
                if d < dist:
                    dist = d
            if dist > mejor_dist:
                mejor, mejor_dist = c, dist
            if dist >= separacion:
                break
        centros.append(mejor)
    return centros


def generar_datos_regresion(generador, n, centro_x, desviacion_x, distribucion_x,
                            coeficientes, sesgo, modelo, desviacion_ruido, distribucion_ruido):
    """n puntos con tantas entradas como coordenadas tenga centro_x.
    Modelo lineal:      y = a1*x1 + a2*x2 + ... + b + ruido
    Modelo cuadrático:  y = a1*x1^2 + a2*x2^2 + ... + b + ruido
    Modelo senoidal:    y = a1*sen(x1) + a2*sen(x2) + ... + b + ruido   (F27)
    El cuadrático y el senoidal no son planos: el perceptrón de la entrega 1 no
    podía ajustarlos y la red con capa oculta sí."""
    X, y = [], []
    for _ in range(n):
        x = [generador.muestra(distribucion_x, c, desviacion_x) for c in centro_x]
        base = 0.0
        for a, xj in zip(coeficientes, x):
            if modelo == "cuadratico":
                base = base + a * xj * xj
            elif modelo == "senoidal":
                base = base + a * seno(xj)                                   # F27
            else:
                base = base + a * xj
        ruido = generador.muestra(distribucion_ruido, 0.0, desviacion_ruido)
        X.append(x)
        y.append(base + sesgo + ruido)
    return X, y
