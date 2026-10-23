"""
red_neuronal.py
===============
Red neuronal multicapa entrenada con BACKPROPAGATION, escrita a mano con las
fórmulas de formulas.py. Sirve para cualquier cantidad de entradas, de capas
ocultas, de neuronas en cada capa y de salidas.

    estructura = [n_entradas, neuronas_oculta_1, ..., neuronas_oculta_h, n_salidas]

Notación de los pesos (la misma de las gráficas):
    w{c}(j,i)  peso de la capa c que llega a la neurona j desde la neurona i
               de la capa anterior (en la capa 1, i es la entrada x_i).
    θ{c}(j)    sesgo de la neurona j de la capa c (su entrada fija es x0).

Pasos de una época (ajuste por época, el del profesor):
    1. Para cada patrón:
         hacia adelante:  S = sum(w*x) + θ*x0   (F1),   y = sigmoide(S)   (F2)
         error:           e = d - y                                     (F4)
         hacia atrás:     DeltaS = e * y(1-y)                            (F5)
                          DeltaC = y(1-y) * sum(peso * Delta siguiente)  (F6)
         acumular:        G = G + entrada * delta                        (F7)
    2. Al terminar TODOS los patrones, una sola vez:
         Peso(n+1) = Peso(n)*Momento + G*Tasa                            (F8)
       (o el momento clásico, F9, si se elige esa opción)

Con el ajuste por patrón el paso 2 se hace después de CADA patrón.

REGRESIÓN: la red es la misma; solo cambia la neurona de salida, que puede ser
lineal (y = S, F24, con DeltaSalida = e, F25) para que la red entregue
cualquier número y no solo valores entre 0 y 1. Las capas ocultas siguen
siendo sigmoides: ellas son las que permiten ajustar curvas.
"""

from formulas import (suma_ponderada, sigmoide, error_salida, delta_salida,
                      delta_camada, entrada_por_delta, ajuste_profesor,
                      ajuste_momento_clasico, error_epoca, error_cuadratico_medio,
                      exactitud, clase_de_la_salida, valor_absoluto, es_finito,
                      funcion_lineal, delta_salida_lineal, coeficiente_determinacion,
                      minimo)

MAX_LINEAS_REGISTRO = 6000


def propagar_con(pesos, sesgos, x0, x, salida_lineal=False):
    """Propagación hacia adelante con unos pesos dados.
    Devuelve la lista de salidas de cada capa: a[0] = x (entradas),
    a[1] = salidas de la capa oculta 1, ..., a[-1] = salidas de la red.
    salida_lineal = True: la última capa usa y = S (regresión, F24)."""
    a = [x]
    ultima = len(pesos) - 1
    for c in range(len(pesos)):
        anterior = a[-1]
        salidas = []
        for j in range(len(pesos[c])):
            S = suma_ponderada(pesos[c][j], anterior, sesgos[c][j], x0)       # F1
            if c == ultima and salida_lineal:
                salidas.append(funcion_lineal(S))                             # F24
            else:
                salidas.append(sigmoide(S))                                   # F2
        a.append(salidas)
    return a


class RedNeuronal:

    def __init__(self, estructura, generador, tasa=0.3, momento=1.0,
                 tipo_momento="profesor", alfa=0.9, ajuste="epoca",
                 usar_sesgo=True, x0=-1, nombre="Red neuronal", salida="sigmoide",
                 regresion=False):
        if len(estructura) < 2:
            raise ValueError("La red necesita al menos la capa de entrada y la de salida.")
        self.estructura = list(estructura)
        self.generador = generador
        self.tasa = tasa
        self.momento = momento              # F8 (fórmula del profesor)
        self.tipo_momento = tipo_momento    # "profesor" (F8) o "clasico" (F9)
        self.alfa = alfa                    # F9
        self.ajuste = ajuste                # "epoca" (lotes) o "patron"
        self.usar_sesgo = usar_sesgo
        self.x0 = x0 if usar_sesgo else 0
        self.nombre = nombre
        self.salida_lineal = salida == "lineal"   # F24 en la capa de salida
        self.regresion = regresion                # mide R² en vez de exactitud
        # Pesos y sesgos iniciales aleatorios en (-1, 1)
        self.pesos, self.sesgos = [], []
        for c in range(1, len(estructura)):
            capa_w, capa_t = [], []
            for _ in range(estructura[c]):
                capa_w.append([generador.uniforme(-1.0, 1.0) for _ in range(estructura[c - 1])])
                capa_t.append(generador.uniforme(-1.0, 1.0) if usar_sesgo else 0.0)
            self.pesos.append(capa_w)
            self.sesgos.append(capa_t)
        self._reiniciar_cambios()
        self.historial_error = []      # F10 de cada época (índice k = pesos tras k ajustes)
        self.historial_ecm = []        # F11
        self.historial_exactitud = []  # F12 (clasificación)
        self.historial_r2 = []         # F26 (regresión)
        self.historial_pesos = []      # copia de (pesos, sesgos) en cada época
        self.epocas = 0
        self.convergio = False
        self.divergio = False
        self.detenido = False
        self.estancado = False

    # ------------------------------------------------------------------
    @property
    def n_capas(self):
        """Número de capas con pesos (ocultas + salida)."""
        return len(self.pesos)

    def nombre_capa(self, c):
        if c == self.n_capas - 1:
            return "Capa de salida"
        return "Capa oculta %d" % (c + 1)

    def arquitectura(self):
        return " → ".join(str(n) for n in self.estructura)

    def _reiniciar_cambios(self):
        """DeltaW(n-1) de F9: empieza en cero para todos los pesos y sesgos."""
        self.cambio_w = [[[0.0] * len(fila) for fila in capa] for capa in self.pesos]
        self.cambio_t = [[0.0] * len(capa) for capa in self.sesgos]

    def copiar_pesos(self):
        return ([[fila[:] for fila in capa] for capa in self.pesos],
                [capa[:] for capa in self.sesgos])

    def poner_pesos(self, pesos, sesgos):
        self.pesos = [[fila[:] for fila in capa] for capa in pesos]
        self.sesgos = [capa[:] for capa in sesgos]
        self._reiniciar_cambios()

    def pesos_en_epoca(self, k):
        """(pesos, sesgos) después de k ajustes (k = 0: pesos iniciales)."""
        if k is None or k > self.epocas:
            k = self.epocas
        return self.historial_pesos[k]

    # ------------------------------------------------------------------
    def propagar(self, x, epoca=None):
        if epoca is None:
            return propagar_con(self.pesos, self.sesgos, self.x0, x, self.salida_lineal)
        pesos, sesgos = self.pesos_en_epoca(epoca)
        return propagar_con(pesos, sesgos, self.x0, x, self.salida_lineal)

    def salida(self, x, epoca=None):
        return self.propagar(x, epoca)[-1]

    def clasificar(self, x, epoca=None):
        return clase_de_la_salida(self.salida(x, epoca))                     # F14

    def retropropagar(self, a, d):
        """Hacia atrás: errores de la salida y deltas de todas las capas.
        deltas[c][j] = delta de la neurona j de la capa c (c = 0 es la primera
        capa oculta, la última es la de salida)."""
        L = self.n_capas
        y = a[-1]
        errores = [error_salida(d[j], y[j]) for j in range(len(y))]          # F4
        deltas = [None] * L
        if self.salida_lineal:
            deltas[L - 1] = [delta_salida_lineal(e) for e in errores]                # F25
        else:
            deltas[L - 1] = [delta_salida(errores[j], y[j]) for j in range(len(y))]   # F5
        for c in range(L - 2, -1, -1):
            siguiente = self.pesos[c + 1]
            capa = []
            for j in range(len(a[c + 1])):
                pesos_hacia_adelante = [siguiente[k][j] for k in range(len(siguiente))]
                capa.append(delta_camada(a[c + 1][j], pesos_hacia_adelante,
                                         deltas[c + 1]))                      # F6
            deltas[c] = capa
        return errores, deltas

    def _gradientes_vacios(self):
        G = [[[0.0] * len(fila) for fila in capa] for capa in self.pesos]
        Gt = [[0.0] * len(capa) for capa in self.sesgos]
        return G, Gt

    def _acumular(self, G, Gt, a, deltas):
        """F7:  G = G + entrada * delta   (la entrada del sesgo es x0)."""
        for c in range(self.n_capas):
            entradas = a[c]
            for j in range(len(deltas[c])):
                dj = deltas[c][j]
                fila = G[c][j]
                for i in range(len(entradas)):
                    fila[i] += entrada_por_delta(entradas[i], dj)
                Gt[c][j] += entrada_por_delta(self.x0, dj)

    def _aplicar(self, G, Gt):
        """Ajusta todos los pesos con F8 (profesor) o F9 (momento clásico)."""
        for c in range(self.n_capas):
            for j in range(len(self.pesos[c])):
                fila = self.pesos[c][j]
                for i in range(len(fila)):
                    if self.tipo_momento == "clasico":
                        fila[i], self.cambio_w[c][j][i] = ajuste_momento_clasico(
                            fila[i], G[c][j][i], self.tasa, self.alfa, self.cambio_w[c][j][i])
                    else:
                        fila[i] = ajuste_profesor(fila[i], self.momento, G[c][j][i], self.tasa)
                if self.usar_sesgo:
                    if self.tipo_momento == "clasico":
                        self.sesgos[c][j], self.cambio_t[c][j] = ajuste_momento_clasico(
                            self.sesgos[c][j], Gt[c][j], self.tasa, self.alfa, self.cambio_t[c][j])
                    else:
                        self.sesgos[c][j] = ajuste_profesor(self.sesgos[c][j], self.momento,
                                                            Gt[c][j], self.tasa)

    # ------------------------------------------------------------------
    def _metrica(self, D, salidas, etiquetas):
        """Clasificación: exactitud (F12) con la clase de cada salida (F14).
        Regresión: R² (F26) entre lo deseado y lo que dio la red."""
        if self.regresion:
            reales, predichos = [], []
            for d, y in zip(D, salidas):
                reales.extend(d)
                predichos.extend(y)
            for v in predichos:
                if not es_finito(v):
                    return v
            return coeficiente_determinacion(reales, predichos)
        return exactitud([clase_de_la_salida(y) for y in salidas], etiquetas)

    def evaluar(self, X, D, etiquetas):
        """Pasa todo el conjunto hacia adelante con los pesos actuales y
        devuelve (E de F10, ECM de F11, exactitud F12 o R² F26)."""
        errores, salidas = [], []
        for x, d in zip(X, D):
            y = propagar_con(self.pesos, self.sesgos, self.x0, x, self.salida_lineal)[-1]
            for j in range(len(y)):
                errores.append(error_salida(d[j], y[j]))
            salidas.append(y)
        return (error_epoca(errores), error_cuadratico_medio(errores),
                self._metrica(D, salidas, etiquetas))

    def _registrar_estado(self, E, ecm, metrica):
        self.historial_error.append(E)
        self.historial_ecm.append(ecm)
        if self.regresion:
            self.historial_r2.append(metrica)
        else:
            self.historial_exactitud.append(metrica)
        self.historial_pesos.append(self.copiar_pesos())

    def nombre_metrica(self):
        return "R²" if self.regresion else "exactitud"

    def texto_metrica(self, v):
        if self.regresion:
            return "R² = %s" % num(v)
        return "exactitud = %.2f %%" % v

    def entrenar(self, X, D, etiquetas, max_epocas=1000, ecm_objetivo=0.0,
                 barajar=False, registro=None, epocas_registro=2, progreso=None,
                 tolerancia=None, paciencia=50):
        """Entrena por épocas.  D[p] = salidas deseadas del patrón p.
        Se detiene cuando ECM <= ecm_objetivo (convergió), cuando los pesos
        dejan de ser números finitos (divergió) o al llegar a max_epocas.
        Si 'registro' es una lista, se anotan las cuentas de las primeras
        'epocas_registro' épocas (como las tablas de las diapositivas).
        progreso(epoca, max_epocas, ecm): función opcional que la interfaz usa
        para mostrar el avance; si devuelve True el entrenamiento se detiene.
        tolerancia: si en las últimas 'paciencia' épocas el ECM no mejoró más que
        esa proporción respecto al mejor anterior, se detiene (se estabilizó).
        Sirve en regresión, donde el ruido no deja que el ECM llegue a cero.
        etiquetas = None en regresión."""
        self.historial_error, self.historial_ecm = [], []
        self.historial_exactitud, self.historial_pesos = [], []
        self.historial_r2 = []
        self.convergio = self.divergio = self.detenido = self.estancado = False
        self.epocas = 0
        self._reiniciar_cambios()
        orden = list(range(len(X)))

        for epoca in range(0, max_epocas + 1):
            anotar = registro is not None and epoca < epocas_registro
            _anotar(registro if anotar else None,
                    "\n%s\nÉPOCA %d  (ajuste por %s)\n%s"
                    % ("=" * 78, epoca + 1, "época" if self.ajuste == "epoca" else "patrón", "=" * 78))

            if self.ajuste == "epoca":
                # Una pasada con los pesos actuales: sirve a la vez para medir el
                # error de la época y para acumular G = Σ entrada*delta (F7).
                G, Gt = self._gradientes_vacios()
                todos_errores, salidas = [], []
                for p in orden:
                    a = propagar_con(self.pesos, self.sesgos, self.x0, X[p], self.salida_lineal)
                    errores, deltas = self.retropropagar(a, D[p])
                    self._acumular(G, Gt, a, deltas)
                    todos_errores.extend(errores)
                    salidas.append(a[-1])
                    if anotar:
                        self._anotar_patron(registro, p, a, D[p], errores, deltas)
                E = error_epoca(todos_errores)                                    # F10
                ecm = error_cuadratico_medio(todos_errores)                       # F11
                exa = self._metrica(D, salidas, etiquetas)                        # F12 / F26
            else:
                E, ecm, exa = self.evaluar(X, D, etiquetas)
            self._registrar_estado(E, ecm, exa)
            _anotar(registro if anotar else None,
                    "\n  -> Error de la época (pesos antes del ajuste): E = ½Σe² = %s   "
                    "ECM = %s   %s" % (num(E), num(ecm), self.texto_metrica(exa)))

            if not es_finito(E):
                self.divergio = True
                break
            if ecm <= ecm_objetivo:
                self.convergio = True
                _anotar(registro if anotar else None,
                        "  -> ECM <= %s: se alcanzó el objetivo y no se ajusta más." % num(ecm_objetivo))
                break
            if epoca == max_epocas:
                break
            h = self.historial_ecm
            if tolerancia is not None and len(h) > paciencia:
                mejor_antes = minimo(h[:-paciencia])
                mejor_reciente = minimo(h[-paciencia:])
                if mejor_reciente >= (1.0 - tolerancia) * mejor_antes:
                    self.estancado = True
                    break
            if progreso is not None and progreso(epoca, max_epocas, ecm):
                self.detenido = True
                break

            if self.ajuste == "epoca":
                antes = self.copiar_pesos() if anotar else None
                self._aplicar(G, Gt)                                              # F8 / F9
                if anotar:
                    self._anotar_ajuste(registro, G, Gt, antes)
            else:
                if barajar:
                    self.generador.barajar(orden)
                for p in orden:
                    a = propagar_con(self.pesos, self.sesgos, self.x0, X[p], self.salida_lineal)
                    errores, deltas = self.retropropagar(a, D[p])
                    G, Gt = self._gradientes_vacios()
                    self._acumular(G, Gt, a, deltas)
                    antes = self.copiar_pesos() if anotar else None
                    self._aplicar(G, Gt)
                    if anotar:
                        self._anotar_patron(registro, p, a, D[p], errores, deltas)
                        self._anotar_ajuste(registro, G, Gt, antes)
            self.epocas = epoca + 1
        return self.historial_error

    def estado(self):
        e = self.epocas
        s = "" if e == 1 else "s"
        if self.divergio:
            return "DIVERGIÓ en la época %d" % e
        if self.convergio:
            return "alcanzó el ECM objetivo en %d época%s" % (e, s)
        if self.detenido:
            return "detenido por el usuario en la época %d" % e
        if self.estancado:
            return "se estabilizó en %d época%s (el ECM dejó de bajar)" % (e, s)
        return "terminó las %d época%s" % (e, s)

    # ------------------------------------------------------------------
    # Nombres y valores de los pesos para las gráficas
    # ------------------------------------------------------------------
    def nombres_pesos(self, c):
        """Nombres de los pesos de la capa c en el mismo orden de valores_capa."""
        nombres = []
        origen = "x" if c == 0 else "n"
        for j in range(len(self.pesos[c])):
            for i in range(len(self.pesos[c][j])):
                nombres.append("w%d(%d,%s%d)" % (c + 1, j + 1, origen, i + 1))
            if self.usar_sesgo:
                nombres.append("θ%d(%d)" % (c + 1, j + 1))
        return nombres

    def valores_capa(self, c, k):
        """Valores de los pesos de la capa c después de k ajustes."""
        pesos, sesgos = self.historial_pesos[k]
        valores = []
        for j in range(len(pesos[c])):
            valores.extend(pesos[c][j])
            if self.usar_sesgo:
                valores.append(sesgos[c][j])
        return valores

    # ------------------------------------------------------------------
    # Registro del proceso (las tablas de las diapositivas)
    # ------------------------------------------------------------------
    def _anotar_patron(self, registro, p, a, d, errores, deltas):
        _anotar(registro, "\nPatrón %d:  entradas x = (%s)   deseada d = (%s)"
                % (p + 1, lista(a[0]), lista(d)))
        for c in range(self.n_capas):
            _anotar(registro, "   %-16s salidas y = (%s)" % (self.nombre_capa(c) + ":", lista(a[c + 1])))
        _anotar(registro, "   Error e = d - y            = (%s)" % lista(errores))
        if self.salida_lineal:
            _anotar(registro, "   DeltaSalida = e * 1 (lineal) = (%s)" % lista(deltas[-1]))
        else:
            _anotar(registro, "   DeltaSalida = e * y(1-y)   = (%s)" % lista(deltas[-1]))
        for c in range(self.n_capas - 2, -1, -1):
            _anotar(registro, "   DeltaCamada %d = y(1-y)*Σ(peso*Delta) = (%s)"
                    % (c + 1, lista(deltas[c])))

    def _anotar_ajuste(self, registro, G, Gt, antes):
        pesos_antes, sesgos_antes = antes
        if self.ajuste == "epoca":
            _anotar(registro, "\nAJUSTE AL FINAL DE LA ÉPOCA (de atrás hacia adelante)")
        else:
            _anotar(registro, "   Ajuste después de este patrón:")
        if self.tipo_momento == "clasico":
            regla = "Peso(n+1) = Peso(n) + Tasa*G + alfa*DeltaW(n-1)   (Tasa = %s, alfa = %s)" % (
                num(self.tasa), num(self.alfa))
        else:
            regla = "Peso(n+1) = Peso(n)*Momento + G*Tasa   (Momento = %s, Tasa = %s)" % (
                num(self.momento), num(self.tasa))
        _anotar(registro, "   " + regla + "    G = Σ entrada*delta")
        for c in range(self.n_capas - 1, -1, -1):
            _anotar(registro, "   " + self.nombre_capa(c) + ":")
            origen = "x" if c == 0 else "n"
            for j in range(len(self.pesos[c])):
                for i in range(len(self.pesos[c][j])):
                    _anotar(registro, "      w%d(%d,%s%d): G = %s   %s -> %s"
                            % (c + 1, j + 1, origen, i + 1, num(G[c][j][i]),
                               num(pesos_antes[c][j][i]), num(self.pesos[c][j][i])))
                if self.usar_sesgo:
                    _anotar(registro, "      θ%d(%d):     G = %s   %s -> %s   (entrada x0 = %d)"
                            % (c + 1, j + 1, num(Gt[c][j]), num(sesgos_antes[c][j]),
                               num(self.sesgos[c][j]), self.x0))


# ----------------------------------------------------------------------
def num(v):
    """Número corto: decimales, o notación científica si es enorme."""
    if not es_finito(v):
        return str(v)
    if valor_absoluto(v) < 1e6:
        return "%.5f" % v
    return "%.3e" % v


def lista(valores):
    return ", ".join(num(v) if isinstance(v, float) else str(v) for v in valores)


def _anotar(registro, texto):
    if registro is None:
        return
    if len(registro) < MAX_LINEAS_REGISTRO:
        registro.append(texto)
    elif len(registro) == MAX_LINEAS_REGISTRO:
        registro.append("... (se omiten las demás líneas; reduzca patrones o épocas registradas)")
