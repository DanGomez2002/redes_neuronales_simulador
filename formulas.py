"""
formulas.py
===========
Todas las fórmulas del proyecto escritas a mano, sin librerías.

    RED NEURONAL (BACKPROPAGATION)
    F1  Suma ponderada de una neurona .. S = w1*x1 + w2*x2 + ... + wn*xn + theta*x0
    F2  Función sigmoide ............... y = 1 / (1 + e^(-S))
    F3  Derivada de la sigmoide ........ y' = y * (1 - y)
    F4  Error de una salida ............ e = d - y
    F5  Delta de la salida ............. DeltaS = Error * Derivada_FA = e * y'
    F6  Delta de la camada (oculta) .... DeltaC = Derivada_FA * sum(peso * Delta siguiente)
    F7  Entrada * Delta ................ G = sum sobre los patrones de (entrada * delta)
    F8  Ajuste del profesor ............ Peso(n+1) = Peso(n) * Momento + Entrada*Delta*Tasa
    F9  Ajuste con momento clásico ..... DeltaW(n) = Tasa*Entrada*Delta + alfa*DeltaW(n-1)
                                         Peso(n+1) = Peso(n) + DeltaW(n)
    F10 Error de la época .............. E = 1/2 * sum sobre patrones y salidas de e^2
    F11 Error cuadrático medio ......... ECM = (1 / (N*m)) * sum de e^2
    F12 Exactitud ...................... 100 * aciertos / N
    F13 Exactitud por clase ............ 100 * aciertos de la clase k / N_k
    F14 Clase de la red ................ 1 salida: clase 2 si y >= 0,5 ; varias: la de mayor y
    F15 Superficie de separación ....... 1 salida: y = 0,5 ; varias: y_k = max(y_j, j != k)
    F16 Normalización mín-máx .......... x' = (x - min) / (max - min)

    SIMULACIÓN DE DATOS (igual que en la entrega 1)
    F17 Generador congruencial ......... X(k+1) = (a*X(k) + c) mod m ;  U = (X + 0.5) / m
    F18 Normal (Box-Muller) ............ Z = sqrt(-2 ln U1) * cos(2 pi U2) ; X = mu + sigma*Z
    F19 Uniforme ....................... X = mu - a + 2a*U ,  a = sigma*sqrt(3)
    F20 Exponencial (inversa) .......... X = (mu - sigma) - sigma * ln(U)
    F21 Binomial estandarizada ......... X = mu + sigma * (B - n p) / sqrt(n p (1 - p))
    F22 Media .......................... media = (1/N) * sum(x)
    F23 Desviación estándar ............ sigma = sqrt( (1/N) * sum((x - media)^2) )

    FUNCIONES MATEMÁTICAS BÁSICAS
    M1  Valor absoluto
    M2  Raíz cuadrada (Newton-Raphson)
    M3  Arcotangente (serie de Taylor)  ->  M4  pi (fórmula de Machin)
    M5  Logaritmo natural (serie de atanh)
    M6  Coseno (serie de Taylor)
    M7  Mínimo y máximo de una lista (recorrido simple)
    M8  Número finito (ni infinito ni NaN)
    M9  Potencia entera (cuadrados sucesivos)
    M10 Exponencial e^x (reducción de rango + serie de Taylor)
    M11 Redondeo al entero más cercano
"""

# =====================================================================
# FUNCIONES MATEMÁTICAS BÁSICAS (sin librerías)
# =====================================================================
def valor_absoluto(v):
    """M1:  |v| = v si v >= 0,  -v si v < 0"""
    if v >= 0:
        return v
    return -v


def raiz_cuadrada(a):
    """M2: Newton-Raphson para r^2 = a:
           r(k+1) = 1/2 * ( r(k) + a / r(k) )
    Se repite hasta que r deja de cambiar."""
    if a < 0:
        raise ValueError("No existe raíz cuadrada real de un número negativo")
    if a == 0:
        return 0.0
    r = a if a >= 1 else 1.0
    for _ in range(1100):
        r_nuevo = 0.5 * (r + a / r)
        if valor_absoluto(r_nuevo - r) <= 1e-16 * r_nuevo:
            return r_nuevo
        r = r_nuevo
    return r


def arcotangente(z):
    """M3: serie de Taylor (válida para |z| < 1):
           atan(z) = z - z^3/3 + z^5/5 - z^7/7 + ..."""
    suma = 0.0
    potencia = z
    n = 0
    while True:
        termino = potencia / (2 * n + 1)
        if n % 2 == 1:
            termino = -termino
        suma += termino
        if valor_absoluto(termino) < 1e-18:
            return suma
        potencia = potencia * z * z
        n += 1


# M4: número pi con la fórmula de Machin:  pi = 16*atan(1/5) - 4*atan(1/239)
PI = 16.0 * arcotangente(1.0 / 5.0) - 4.0 * arcotangente(1.0 / 239.0)


def _serie_ln(m):
    """ln(m) = 2 * [ z + z^3/3 + z^5/5 + ... ],  z = (m - 1) / (m + 1)"""
    z = (m - 1.0) / (m + 1.0)
    z2 = z * z
    potencia = z
    suma = 0.0
    n = 0
    while True:
        termino = potencia / (2 * n + 1)
        suma += termino
        if valor_absoluto(termino) < 1e-18:
            return 2.0 * suma
        potencia = potencia * z2
        n += 1


LN2 = _serie_ln(2.0)   # ln(2) calculado con la misma serie


def logaritmo_natural(a):
    """M5: se escribe a = m * 2^k con 1 <= m < 2, entonces
           ln(a) = k * ln(2) + ln(m)"""
    if a <= 0:
        raise ValueError("El logaritmo solo existe para números positivos")
    k = 0
    m = a
    while m >= 2.0:
        m = m / 2.0
        k += 1
    while m < 1.0:
        m = m * 2.0
        k -= 1
    return k * LN2 + _serie_ln(m)


def coseno(angulo):
    """M6: serie de Taylor:  cos(t) = 1 - t^2/2! + t^4/4! - t^6/6! + ...
    Antes se lleva el ángulo al intervalo [-pi, pi] restando vueltas completas."""
    vuelta = 2.0 * PI
    t = angulo - int(angulo / vuelta) * vuelta
    if t > PI:
        t -= vuelta
    elif t < -PI:
        t += vuelta
    suma = 0.0
    termino = 1.0
    n = 0
    while valor_absoluto(termino) > 1e-18:
        suma += termino
        termino = -termino * t * t / ((2 * n + 1) * (2 * n + 2))
        n += 1
    return suma


def minimo(valores):
    """M7: el menor valor de la lista, comparando uno por uno."""
    menor = valores[0]
    for v in valores:
        if v < menor:
            menor = v
    return menor


def maximo(valores):
    """M7: el mayor valor de la lista, comparando uno por uno."""
    mayor = valores[0]
    for v in valores:
        if v > mayor:
            mayor = v
    return mayor


INFINITO = float("inf")


def es_finito(v):
    """M8: v es finito si no es NaN (el único valor distinto de sí mismo)
    ni infinito."""
    if v != v:
        return False
    return v != INFINITO and v != -INFINITO


def potencia_entera(base, k):
    """M9: base^k para k entero, por cuadrados sucesivos:
           si k es par   ->  base^k = (base^2)^(k/2)
           si k es impar ->  base^k = base * base^(k-1)
    Para k negativo:  base^k = 1 / base^(-k)."""
    if k < 0:
        return 1.0 / potencia_entera(base, -k)
    resultado = 1.0
    b = base
    while k > 0:
        if k % 2 == 1:
            resultado = resultado * b
        b = b * b
        k = k // 2
    return resultado


def redondear(v):
    """M11: entero más cercano (los .5 se alejan del cero)."""
    if v >= 0:
        return int(v + 0.5)
    return -int(-v + 0.5)


def exponencial(x):
    """M10:  e^x  sin librerías.
    1) Reducción de rango:  x = k*ln(2) + r,  con k = redondeo(x / ln 2) y |r| <= 0.35
       entonces  e^x = 2^k * e^r.
    2) e^r por la serie de Taylor:  e^r = 1 + r + r^2/2! + r^3/3! + ...
       (con |r| <= 0.35 bastan unos 15 términos).
    Fuera de [-745, 709] el resultado no cabe en un número de punto flotante:
    se devuelve 0 o infinito."""
    if x > 709.0:
        return INFINITO
    if x < -745.0:
        return 0.0
    k = redondear(x / LN2)
    r = x - k * LN2
    suma = 1.0
    termino = 1.0
    n = 1
    while True:
        termino = termino * r / n
        suma += termino
        if valor_absoluto(termino) < 1e-17:
            break
        n += 1
    return suma * potencia_entera(2.0, k)


# =====================================================================
# FÓRMULAS DE LA RED NEURONAL
# =====================================================================
def suma_ponderada(w, x, theta, x0):
    """F1:  S = w1*x1 + w2*x2 + ... + wn*xn + theta*x0
    x0 = -1 (convención de la clase):  S = sum(wi*xi) - theta
    Sin sesgo: theta = 0 y la suma queda solo con las entradas."""
    S = 0.0
    for i in range(len(w)):
        S = S + w[i] * x[i]
    return S + theta * x0


def sigmoide(S):
    """F2:  y = 1 / (1 + e^(-S))
    Para S negativo se usa la forma equivalente  y = e^S / (1 + e^S)
    (se multiplica arriba y abajo por e^S): así nunca se calcula e de un número
    grande, que no cabría en la computadora."""
    if S >= 0:
        return 1.0 / (1.0 + exponencial(-S))
    e = exponencial(S)
    return e / (1.0 + e)


def derivada_sigmoide(y):
    """F3:  y' = y * (1 - y)   (y es la salida que ya dio la sigmoide)"""
    return y * (1.0 - y)


def error_salida(d, y):
    """F4:  e = d - y   (d = salida deseada, y = salida obtenida)"""
    return d - y


def delta_salida(e, y):
    """F5:  DeltaSalida = Error * Derivada_FA = e * y(1 - y)"""
    return e * derivada_sigmoide(y)


def delta_camada(y, pesos_siguientes, deltas_siguientes):
    """F6:  DeltaCamada = Derivada_FA * sum_k( peso_k * Delta_k )
    y: salida de la neurona oculta.
    pesos_siguientes[k]: peso que une esta neurona con la neurona k de la capa
    siguiente; deltas_siguientes[k]: delta de esa neurona k.
    Con una sola neurona adelante queda  Derivada * peso * DeltaS  (diapositivas)."""
    suma = 0.0
    for k in range(len(deltas_siguientes)):
        suma = suma + pesos_siguientes[k] * deltas_siguientes[k]
    return derivada_sigmoide(y) * suma


def entrada_por_delta(entrada, delta):
    """F7 (un patrón):  entrada * delta.  En el ajuste por época se suma sobre
    todos los patrones:  G = sum( entrada_p * delta_p )."""
    return entrada * delta


def ajuste_profesor(peso, momento, G, tasa):
    """F8:  Peso(n+1) = (Peso(n) * Momento) + (Entrada * Delta * TasaDeAprendizaje)
    Con Momento = 1 (valor de la clase) queda  Peso(n+1) = Peso(n) + Tasa * G."""
    return peso * momento + G * tasa


def ajuste_momento_clasico(peso, G, tasa, alfa, cambio_anterior):
    """F9:  DeltaW(n) = Tasa * G + alfa * DeltaW(n-1)
            Peso(n+1) = Peso(n) + DeltaW(n)
    El cambio anterior le da "inercia" al peso: sigue moviéndose en la dirección
    en que venía. Devuelve (peso nuevo, cambio de este paso)."""
    cambio = tasa * G + alfa * cambio_anterior
    return peso + cambio, cambio


def error_epoca(errores):
    """F10:  E = 1/2 * sum(e^2)   (errores de todos los patrones y salidas)"""
    suma = 0.0
    for e in errores:
        suma = suma + e * e
    return 0.5 * suma


def error_cuadratico_medio(errores):
    """F11:  ECM = (1 / (N*m)) * sum(e^2)
    N*m = número de errores (N patrones por m salidas)."""
    suma = 0.0
    for e in errores:
        suma = suma + e * e
    return suma / len(errores)


def exactitud(predicciones, reales):
    """F12:  exactitud (%) = 100 * aciertos / N"""
    aciertos = 0
    for p, r in zip(predicciones, reales):
        if p == r:
            aciertos += 1
    return 100.0 * aciertos / len(reales)


def exactitud_por_clase(predicciones, reales, clase):
    """F13:  exactitud de la clase k (%) = 100 * aciertos de la clase k / N_k"""
    aciertos = 0
    total = 0
    for p, r in zip(predicciones, reales):
        if r == clase:
            total += 1
            if p == r:
                aciertos += 1
    if total == 0:
        return 0.0
    return 100.0 * aciertos / total


def clase_de_la_salida(y):
    """F14: con una sola neurona de salida, clase 2 (índice 1) si y >= 0,5 y
    clase 1 (índice 0) si no. Con varias salidas, la clase de la neurona que
    dio la salida más grande."""
    if len(y) == 1:
        return 1 if y[0] >= 0.5 else 0
    mejor = 0
    for k in range(1, len(y)):
        if y[k] > y[mejor]:
            mejor = k
    return mejor


def funcion_frontera(y, clase):
    """F15: función que vale 0 justo sobre la superficie de separación.
    1 salida:  f = y - 0,5            (la red duda entre las dos clases)
    m salidas: f = y_k - max(y_j, j != k)  (la clase k empata con la mejor
               de las demás). f > 0 dentro de la región de la clase k."""
    if len(y) == 1:
        return y[0] - 0.5
    otras = [y[j] for j in range(len(y)) if j != clase]
    return y[clase] - maximo(otras)


def normalizar(v, vmin, vmax):
    """F16:  x' = (x - min) / (max - min)  ->  lleva cada entrada a [0, 1].
    Si todos los valores son iguales no hay rango y se deja en 0,5."""
    if vmax == vmin:
        return 0.5
    return (v - vmin) / (vmax - vmin)


# =====================================================================
# FÓRMULAS DE LA SIMULACIÓN DE DATOS
# =====================================================================
def congruencial_lineal(X, a, c, m):
    """F17:  X(k+1) = (a * X(k) + c) mod m"""
    return (a * X + c) % m


def uniforme_01(X, m):
    """F17:  U = (X + 0.5) / m   ->  número en el intervalo abierto (0, 1)"""
    return (X + 0.5) / m


def normal_box_muller(U1, U2, mu, sigma):
    """F18:  Z = sqrt(-2 ln U1) * cos(2 pi U2)   ~ Normal(0, 1)
             X = mu + sigma * Z"""
    Z = raiz_cuadrada(-2.0 * logaritmo_natural(U1)) * coseno(2.0 * PI * U2)
    return mu + sigma * Z


def uniforme_media_desviacion(U, mu, sigma):
    """F19:  X = (mu - a) + 2a * U,   a = sigma * sqrt(3)"""
    a = sigma * raiz_cuadrada(3.0)
    return (mu - a) + 2.0 * a * U


def exponencial_inversa(U, mu, sigma):
    """F20:  E = -sigma * ln(U)  tiene media sigma y desviación sigma.
             X = (mu - sigma) + E   ->  media mu y desviación sigma."""
    E = -sigma * logaritmo_natural(U)
    return (mu - sigma) + E


def binomial_estandarizada(exitos, ensayos, p, mu, sigma):
    """F21:  X = mu + sigma * (B - n p) / sqrt(n p (1 - p))"""
    z = (exitos - ensayos * p) / raiz_cuadrada(ensayos * p * (1.0 - p))
    return mu + sigma * z


def media(valores):
    """F22:  media = (x1 + x2 + ... + xN) / N"""
    suma = 0.0
    for v in valores:
        suma = suma + v
    return suma / len(valores)


def desviacion_estandar(valores):
    """F23:  sigma = sqrt( (1/N) * [ (x1 - media)^2 + ... + (xN - media)^2 ] )"""
    m = media(valores)
    suma = 0.0
    for v in valores:
        suma = suma + (v - m) * (v - m)
    return raiz_cuadrada(suma / len(valores))
