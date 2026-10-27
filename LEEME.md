RED NEURONAL CON BACKPROPAGATION - Entrega 2 (Carlos Daniel Gómez Murcia)
==========================================================================
Clasificación (n clases) y regresión con una red multicapa de n entradas,
n capas ocultas y n neuronas por capa.

Archivos
--------
  main.py          Interfaz gráfica (ejecutar este).
  formulas.py      TODAS las fórmulas escritas a mano: la red (F1-F16), la
                   simulación de datos (F17-F23), la regresión (F24-F27: salida
                   lineal, su delta, R² y seno) y las funciones básicas (M1-M11),
                   incluida la exponencial e^x de la sigmoide (serie de Taylor),
                   la raíz, el logaritmo, el coseno, pi, el mínimo y el máximo.
                   Sin librerías.
  red_neuronal.py  Clase RedNeuronal: propagación hacia adelante, deltas de la
                   salida y de cada camada, ajuste por época (lotes) o por
                   patrón, momento del profesor y momento clásico, historial de
                   errores y de pesos, registro paso a paso.
                   Sirve para n entradas, n capas ocultas, n neuronas y n salidas.
  aleatorio.py     Generador congruencial, distribuciones (normal, uniforme,
                   exponencial, binomial) y conjuntos de datos: nubes por clase,
                   XOR en n dimensiones y datos de
                   regresión (modelo lineal, cuadrático o senoidal + ruido).
  experimentos.py  Configuración por defecto, ejecución, ejemplo de clase,
                   clasificación de puntos nuevos y resumen de resultados.
  graficas.py      Dibuja con matplotlib. La superficie de separación se calcula
                   a mano (marching tetrahedra sobre una rejilla).

Requisitos
----------
  Python 3.8 o superior (tkinter viene incluido)
  matplotlib  ->  pip install -r requirements.txt

Ejecución en Visual Studio Code
-------------------------------
  1. Archivo > Abrir carpeta... > redes_neuronales_simulador
  2. Si la carpeta .venv no existe: Terminal > Ejecutar tarea > "Preparar entorno"
     (crea el entorno e instala matplotlib).
  3. Elegir el intérprete ".venv" si VS Code lo pregunta.
  4. Abrir main.py y presionar F5 (o Ctrl+Shift+B).
  Sin VS Code:  pip install -r requirements.txt   y luego   python main.py

Uso de la interfaz
------------------
  Pestaña "Clasificación" (izquierda):
    - Entradas (1 a 10), clases (2 a 10) y puntos por clase (el mismo número
      en todas las clases).
    - Tipo de datos: nubes (media "x1; x2; ...", desviación y distribución por
      clase) o XOR. Botones: centros automáticos, acercar, alejar, σ+, σ−.
    - Escenarios de prueba con 3 entradas: Separadas, Cercanas, Solapadas
      y XOR 3D.
    - Neuronas por capa oculta separadas por punto y coma: "5" es una capa de
      5 neuronas; "6; 4" son dos capas; vacío = sin capa oculta.
      Con 2 clases hay 1 neurona de salida (d = 0 o 1); con k clases hay k
      salidas (d = 1 en la de su clase y 0 en las demás).
    - Tasa de aprendizaje, ajuste por época (lotes, el del profesor) o por
      patrón, momento del profesor (Peso·Momento) o clásico (α·ΔW anterior),
      épocas, ECM objetivo, sesgo con x0 = −1 o +1, normalizar entradas,
      semilla y registro paso a paso.
    - "Detener" corta el entrenamiento y deja los resultados hasta esa época.
    - Clasificar un punto nuevo: "x1; x2; ...".
  Pestaña "Regresión":
    - Entradas (1 a 10), puntos, modelo real (lineal, cuadrático o senoidal),
      coeficientes "a1; a2; ...", término b, media y desviación de las entradas,
      ruido. La fórmula del modelo se muestra debajo.
    - La misma red; la neurona de salida puede ser lineal (y = S, recomendada)
      o sigmoide (y escalada a [0, 1]). Entradas y salida se normalizan (F16).
    - Se detiene si el ECM deja de mejorar ("Mejora mínima del ECM" en N épocas).
    - Espacio: 1 entrada -> curva de la red; 2 entradas -> superficie 3D;
      3 entradas -> corte en x3 = media con los puntos corregidos; siempre la
      gráfica "real vs. predicho" y el R² (F26). Errores: E, ECM y R² por época.
    - Escenarios Lineal, Cuadrática y Senoidal: cargan valores que funcionan
      (momento clásico α = 0,9 y semilla 2, para repetir el mismo resultado).
    - Predecir y para un punto nuevo "x1; x2; ...".
  Pestaña "Ejemplo de clase": reproduce el XOR de las diapositivas (red 2-3-1,
    tasa 0,3, momento 1) con y sin sesgo.
  Resultados (derecha):
    - Espacio: con 3 entradas, puntos 3D y superficie de separación (se rota
      con el mouse); con 2 entradas, regiones y frontera en el plano; con más
      entradas no se dibuja (las demás gráficas sí).
    - Errores: error de la época E = ½Σe², ECM y exactitud, época a época.
    - Pesos por capa: una gráfica por capa (línea punteada = sesgo θ).
    - Todos los pesos: todos al mismo tiempo, un color por capa.
    - Red: diagrama con el valor de cada peso en la época elegida.
    - Proceso paso a paso: salidas, error, DeltaSalida, DeltaCamada y ajuste
      de cada peso (activar "Registrar el proceso").
    - Resumen y fórmulas.
  Abajo: barra "Época mostrada" y "Animar entrenamiento".
  Debajo de la tasa, en las dos pestañas, una línea indica el paso por época
    ≈ tasa × N (y tasa × N / (1 − α) con momento clásico): con el ajuste por
    época G suma los N patrones, así que al aumentar los puntos hay que bajar
    la tasa. Se pone en rojo cuando el paso es alto.
  Si el entrenamiento sale mal (divergencia, exactitud igual al azar, ECM que
    no baja o R² bajo) aparece un aviso con la causa probable.
  "Resumen y fórmulas" incluye "Dónde está cada paso en el código": archivo,
    función y línea de cada fórmula y de cada paso del backpropagation.
  Clic derecho sobre cualquier botón, campo, lista, casilla o pestaña (también
    las pestañas Clasificación / Regresión / Ejemplo de clase): abre en VS Code
    la función que hace ese trabajo, en la línea exacta. En la pestaña Espacio
    abre la función de la gráfica que se está mostrando (3D, 2D o regresión).
  Los decimales se pueden escribir con coma o con punto.

Qué se envía
------------
  Informe en Word y una carpeta comprimida con los archivos .py, LEEME.md,
  requirements.txt y la carpeta .vscode. La carpeta .venv NO se envía.
