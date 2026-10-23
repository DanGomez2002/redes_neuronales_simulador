# redes_neuronales_simulador — Computacional I, entrega 2

Simulador en Python (tkinter + matplotlib) de una red neuronal multicapa entrenada con backpropagation, para clasificación y regresión (igual que la entrega 1). Autor: Carlos Daniel Gómez Murcia. Se entrega el miércoles 28 de octubre de 2026 junto con un informe (misma plantilla APA de la entrega 1) y una app de estudio para la sustentación.

## Regla principal
Nada de librerías para la matemática ni para el algoritmo: todo se escribe a mano en `formulas.py` (exponencial, raíz, logaritmo, coseno, mínimo, máximo, etc.) y `red_neuronal.py`. Las librerías solo se usan para lo visual (tkinter, matplotlib). No usar `math`, `random`, `numpy`, ni `sum`/`min`/`max` en los cálculos.

## Lo que pidió el profesor
- Backpropagation que funcione para n entradas, n capas ocultas, n neuronas por capa y n clases.
- Con 3 entradas: gráfica 3D de los puntos y la superficie de separación.
- Siempre: gráficas de error, error cuadrático medio y pesos época a época (capa a capa y todos juntos).
- Mismo número de puntos por clase (100 por clase → 100 × N_clases).
- Fórmulas de clase: sigmoide y' = y(1−y); DeltaSalida = Error·Derivada; DeltaCamada = Derivada·peso·DeltaS; Peso(n+1) = Peso(n)·Momento + Entrada·Delta·Tasa.

## Decisiones tomadas con Daniel
- Se usa sesgo con x0 = −1 (como en la entrega 1); se puede desactivar.
- Momento: la fórmula del profesor por defecto (momento = 1) y el momento clásico como opción.
- Ajuste por época (lotes), como en el ejemplo de clase; el ajuste por patrón queda como opción.
- 2 clases → 1 neurona de salida; k clases → k salidas (one-hot).
- Regresión: misma red con 1 salida lineal (y = S, DeltaSalida = e) o sigmoide; entradas y salida normalizadas; métrica R²; modelos lineal, cuadrático y senoidal para n entradas.
- El ejemplo XOR 2-3-1 de las diapositivas se reproduce exacto (sin sesgo) en la pestaña "Ejemplo de clase".

## Archivos
formulas.py (F1–F27, M1–M11), aleatorio.py, red_neuronal.py, experimentos.py, graficas.py, main.py.
