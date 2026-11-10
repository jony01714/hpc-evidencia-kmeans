# K-Means paralelo: memoria compartida vs. memoria distribuida

Evidencia de producto · Cómputo de Alto Rendimiento · Ingeniería en Datos e IA (7.º cuatrimestre)
**Proyecto elegido: P1 – K-Means paralelo desde cero**

**Integrantes:** Jonathan Israel Gonzalez Ruiz

## Qué hace

Implementa el algoritmo de Lloyd (K-Means) en tres versiones que comparten **el mismo núcleo de
cálculo** (`comun/kmeans_comun.py::bloque`), los mismos datos sintéticos y la misma inicialización:

| Versión | Carpeta | Modelo | Cómo reparte el trabajo |
|---|---|---|---|
| Secuencial (referencia, `T1`) | `secuencial/` | 1 hilo | — |
| Paralela A | `paralelo_memoria_compartida/` | **Memoria compartida** (hilos, numba `prange`) | Bloques estáticos de filas por hilo; acumuladores privados por hilo y reducción final (sin locks) |
| Paralela B | `paralelo_memoria_distribuida/` | **Memoria distribuida** (procesos, `mpi4py`) | `Scatterv` de bloques de filas desde el rank 0; una `Allreduce` de `k·d + k` valores por iteración |

Cada iteración tiene dos fases: **asignación** (cada punto al centroide más cercano, costo `O(n·k·d)`,
paralelizable) y **actualización** de centroides (`O(k·d)`, secuencial). Esa segunda fase, junto con la
reducción/comunicación, es la fracción serial que fija el límite de la Ley de Amdahl.

## Estructura

```
comun/                           núcleo compartido, generación de datos e inicialización
secuencial/                      K-Means secuencial
paralelo_memoria_compartida/     K-Means con hilos (numba)
paralelo_memoria_distribuida/    K-Means con MPI (mpi4py)
experimentos/                    verificación, ejecución de experimentos, análisis y specs
resultados/                      CSV, tabla en Markdown, gráficas y verificación (se generan al correr)
```

## Instalación

Ubuntu/Debian:
```bash
sudo apt install mpich libmpich-dev build-essential     # o openmpi-bin libopenmpi-dev
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Con Docker (entorno idéntico en cualquier máquina):
```bash
docker build -t kmeans-paralelo .
docker run --rm -v "$PWD/resultados:/app/resultados" kmeans-paralelo
```

## Cómo ejecutar

```bash
# 1) Correctitud: las versiones paralelas deben dar lo mismo que la secuencial
python experimentos/verificar_correctitud.py

# 2) Experimentos completos (1, 2, 4, 8 hilos/procesos; 5 repeticiones; genera tablas y gráficas)
python experimentos/ejecutar_experimentos.py
#    prueba rápida:  python experimentos/ejecutar_experimentos.py --n 300000 --iter 10 --reps 3
#    más núcleos:    python experimentos/ejecutar_experimentos.py --ps 1,2,4,8,12

# 3) Solo regenerar tablas/gráficas a partir de resultados_crudos.json
python experimentos/analizar.py
```

Para mediciones limpias: laptop conectada a la corriente, sin otras cargas pesadas y sin tocar la
máquina mientras corre. Las especificaciones del equipo se guardan solas en `resultados/specs_equipo.json`.

## Metodología de medición

- **Carga fija:** 25 iteraciones exactas (`tol=0`), así todas las versiones hacen el mismo trabajo.
- **Qué se cronometra:** desde que los datos ya existen hasta tener los centroides finales. La compilación JIT se
  calienta antes y no cuenta. En MPI el tiempo **incluye** el `Scatterv` inicial (repartir datos es un costo real)
  y es el del proceso más lento (`Barrier` antes y después).
- **Repeticiones:** ≥ 3 por configuración (por defecto 5); se reporta media y desviación estándar.
- **Base de comparación:** `S(p) = T1 / Tp` con `T1` = versión secuencial (no la paralela con p=1).
- **Fórmulas:** `S(p) = T1/Tp`, `E(p) = S(p)/p`, Amdahl `S(p) ≤ 1 / ((1-f) + f/p)`; `f` se estima por punto,
  `f(p) = (1-1/S)/(1-1/p)`, y de forma global por mínimos cuadrados usando solo `1 < p ≤ núcleos físicos`
  (leídos de `resultados/specs_equipo.json`); los p mayores son hilos SMT y quedan fuera del ajuste.
- En MPI cada proceso usa 1 hilo de numba, para que el paralelismo venga solo de MPI.

## Verificación de correctitud

`experimentos/verificar_correctitud.py` compara centroides (`rtol=1e-9`), etiquetas (igualdad exacta), inercia
e iteraciones contra la versión secuencial, con 1, 2, 3, 4 y 8 hilos y 1, 2, 3, 4 y 7 procesos MPI (cantidades que
no dividen `n`, para probar el reparto desigual), además de una comparación independiente con scikit-learn. Las
diferencias esperables (~1e-14) vienen solo del orden de las sumas en coma flotante. Resultado en
`resultados/verificacion_correctitud.txt`.

## Resultados y análisis

### Equipo de cómputo

Fuente: `resultados/specs_equipo.json`.

| Componente | Valor |
|---|---|
| CPU | 11th Gen Intel Core i5-1135G7 @ 2.40 GHz (turbo máx. 4200 MHz), laptop |
| Núcleos | **4 núcleos físicos / 8 hilos lógicos** (1 socket, 2 hilos por núcleo, Hyper-Threading) |
| Caché L3 | 8 MiB (compartida) |
| RAM | 10.9 GB |
| GPU | Intel Iris Xe Graphics (TigerLake-LP GT2), integrada; **sin GPU NVIDIA ni CUDA** (no se usa GPU) |
| Sistema | Linux 6.14.0-37-generic x86_64, glibc 2.41 |
| Software | Python 3.13.3, NumPy 2.5.4, Numba 0.68.1, mpi4py 4.1.2, MPICH 4.2.1, scikit-learn 1.9.1 |

### Configuración del experimento

`n = 2,000,000` puntos, `d = 16`, `k = 16`, 25 iteraciones fijas, 5 repeticiones por configuración,
`p ∈ {1, 2, 4, 8}`. Tiempo secuencial de referencia: **T1 = 18.043 ± 0.043 s**.

### Tabla de resultados

Fuente: `resultados/resumen.md` / `resultados/resumen_metricas.csv` (media ± desviación estándar de 5 repeticiones).

| Modelo | p | Tiempo (s) | Speedup S(p) | Eficiencia E(p) | f por punto |
|---|---|---|---|---|---|
| Mem. compartida (numba) | 1 | 18.546 ± 0.586 | 0.97 ± 0.03 | 0.97 | — |
| Mem. compartida (numba) | 2 | 9.794 ± 0.312 | 1.84 ± 0.06 | 0.92 | 0.914 |
| Mem. compartida (numba) | 4 | 4.773 ± 0.093 | 3.78 ± 0.07 | 0.94 | 0.981 |
| Mem. compartida (numba) | 8 (SMT) | 4.440 ± 0.203 | 4.06 ± 0.19 | 0.51 | 0.862 |
| Mem. distribuida (MPI) | 1 | 18.514 ± 0.563 | 0.97 ± 0.03 | 0.97 | — |
| Mem. distribuida (MPI) | 2 | 9.485 ± 0.108 | 1.90 ± 0.02 | 0.95 | 0.949 |
| Mem. distribuida (MPI) | 4 | 4.736 ± 0.118 | 3.81 ± 0.10 | 0.95 | 0.983 |
| Mem. distribuida (MPI) | 8 (SMT) | 4.193 ± 0.085 | 4.30 ± 0.09 | 0.54 | 0.877 |

**(SMT)** = hilos lógicos: con p = 8 hay más hilos/procesos que núcleos físicos (4), así que cada par comparte un
núcleo mediante Hyper-Threading. Esos puntos se muestran en la tabla y en las gráficas, pero no entran al ajuste de
Amdahl.

Gráficas: `resultados/grafica_tiempo_vs_nucleos.png`, `resultados/grafica_speedup_real_vs_ideal.png` (la zona
sombreada marca p > 4, hilos lógicos) y `resultados/grafica_eficiencia.png`.

### Ley de Amdahl

`f` es la fracción paralelizable; el speedup máximo teórico es `S(∞) = 1 / (1 − f)`. El valor global de `f` se ajusta
por mínimos cuadrados **solo con los puntos 1 < p ≤ 4 (núcleos físicos)**, es decir, con p = 2 y p = 4. Por
transparencia también se reporta el ajuste con todos los p > 1 (incluido p = 8).

| Modelo | f (p = 2, 4; núcleos físicos) | S máx. teórico | f (p = 2, 4, 8; solo referencia) | S máx. teórico |
|---|---|---|---|---|
| Mem. compartida (numba, hilos) | **0.9786** | **46.7** | 0.8824 | 8.5 |
| Mem. distribuida (MPI, procesos) | **0.9823** | **56.5** | 0.8944 | 9.5 |

**Por qué se excluye p = 8 del ajuste.** Amdahl supone que la parte paralela se reparte entre `p` procesadores
idénticos e independientes, de modo que su tiempo baja como `f/p`. Con p = 8 en un equipo de 4 núcleos físicos eso
no ocurre: los 8 hilos/procesos son 4 pares de hilos SMT y cada par comparte las unidades de cálculo y la caché
L1/L2 de un mismo núcleo. La caída de rendimiento en p = 8 viene del hardware, no de código serial. Si se incluye
en el ajuste, Amdahl la interpreta como fracción serial y subestima `f`. Eso pasaba con el ajuste anterior
(f = 0.8824 / 0.8944): predecía S(4) = 2.96 y 3.04 cuando se midió 3.78 y 3.81, y la curva quedaba claramente por
debajo de los puntos en p = 2 y p = 4.

Con el ajuste solo sobre núcleos físicos, Amdahl predice S(2) = 1.96 / 1.97 y S(4) = 3.76 / 3.80 (numba / MPI), muy
cerca de lo medido (1.84 / 1.90 y 3.78 / 3.81). Extendida hasta p = 8, la curva predice 6.96 / 7.12, pero se midió
4.06 / 4.30. Esa brecha es precisamente lo que cuesta usar hilos SMT en lugar de núcleos reales.

**Cómo leer los topes de 46.7 y 56.5.** Son el límite teórico de Amdahl si existieran infinitos núcleos *físicos*
y solo importara la fracción serial del algoritmo (`f ≈ 0.98`, es decir, ~2 % serial). Están calculados con solo
dos puntos (p = 2 y 4), así que son una extrapolación muy lejana con mucha incertidumbre. Además, no consideran el
ancho de banda de memoria ni el costo de comunicación, que crecerían con más núcleos. En este equipo el límite real
lo pone el hardware: 4 núcleos físicos, es decir, S ≈ 4.

### Interpretación

**p = 1 (costo de la paralelización).** Ambas versiones paralelas con un solo hilo/proceso tardan ~3 % más que la
secuencial (S = 0.97). Esto incluye la sobrecarga de la infraestructura paralela (planificación de `prange`,
acumuladores por hilo y reducción en numba; `Scatterv`, `Allreduce` y `Barrier` en MPI aunque haya un solo
proceso). Parte de esa diferencia también se debe a variación térmica (ver anomalías).

**p = 2 y p = 4 (núcleos físicos).** Las dos versiones escalan casi linealmente: eficiencia de 0.92–0.95 y S(4) de
3.78 (numba) y 3.81 (MPI), lo que coincide con `f ≈ 0.98` (~2 % serial). Se alejan del ideal por:
- *Fracción serial:* la actualización de centroides (`O(k·d)` por iteración) y la reducción final de acumuladores
  la hace un solo hilo/proceso, mientras que la asignación (`O(n·k·d)`) sí se reparte. Con `n = 2·10⁶` y
  `k·d = 256`, esa parte es pequeña, lo que es consistente con el ~2 % serial del ajuste.
- *Comunicación en MPI:* el `Scatterv` inicial de `X` (2·10⁶ × 16 valores, incluido en el tiempo) y una `Allreduce`
  de `k·d + k = 272` valores por iteración (25 en total). Como el mensaje por iteración es pequeño y todo corre en
  la misma máquina (memoria compartida como transporte), este costo resultó bajo: MPI no fue más lento que numba en
  ningún p.
- *Ancho de banda de memoria:* cada iteración recorre los 2·10⁶ × 16 valores de `X`, mucho más que los 8 MiB de
  L3, así que se leen desde RAM. Todos los núcleos comparten el mismo bus de memoria, y conforme crece `p` ese
  recurso común limita la ganancia.
- *Turbo y temperatura de la laptop:* con 1 núcleo activo el i5-1135G7 puede llegar a 4.2 GHz, pero con todos los
  núcleos activos la frecuencia baja por límites de potencia y temperatura. Esto favorece a T1 y castiga a p
  grande, por lo que reduce el speedup medido.

**p = 8 (hilos SMT / Hyper-Threading).** Pasar de 4 a 8 apenas mejora el tiempo (numba 4.773 → 4.440 s, MPI
4.736 → 4.193 s) y la eficiencia cae a 0.51 y 0.54. Hyper-Threading solo aporta lo que logra aprovechar mientras el
otro hilo del mismo núcleo espera memoria; aquí fue un 7 % (numba) y un 13 % (MPI) sobre p = 4, muy lejos del 6.96 /
7.12 que predeciría Amdahl con 8 núcleos reales. Por eso la eficiencia en p = 8 no indica que el código sea serial:
indica que se agotaron los núcleos físicos.

**Memoria compartida vs. memoria distribuida.** **La diferencia entre MPI y numba está dentro del ruido de
medición.** MPI tiene un tiempo medio algo menor en p = 2, 4 y 8, pero la diferencia (0.309 s, 0.037 s y 0.247 s)
es comparable a la desviación estándar combinada de ambas mediciones (0.33 s, 0.15 s y 0.22 s): ≈ 0.9, 0.25 y 1.1
veces esa desviación, con solo 5 repeticiones. Con estos datos no se puede afirmar que un modelo sea más rápido que
el otro en este equipo. Lo que sí se puede afirmar es que el costo de comunicación de MPI (`Scatterv` +
`Allreduce`) no lo hizo más lento que la versión con hilos.

**Conclusión.** En esta laptop ambos modelos alcanzan un speedup de ~3.8 con los 4 núcleos físicos (f ≈ 0.98) y de
4.06 (numba) / 4.30 (MPI) usando los 8 hilos lógicos, sin diferencia significativa entre ellos. El cuello de
botella dominante no es el código serial sino el hardware: el número de núcleos físicos, el ancho de banda de
memoria y la frecuencia bajo carga.

### Anomalías y valores a tomar con cautela

- **Eficiencia > 1 o f > 1:** no se presentaron (máximo E = 0.97, máximo f = 0.983).
- **Desviación estándar en p = 1 mucho mayor que la secuencial:** T1 tiene ±0.043 s (0.2 %), pero numba p = 1
  tiene ±0.586 s y MPI p = 1 ±0.563 s (~3 %). En `resultados_crudos.json` se ve que no es ruido uniforme: en numba
  p = 1 las dos últimas repeticiones (19.336 s y 19.009 s) son ~7 % más lentas que la primera (18.048 s), y en MPI
  p = 1 la primera repetición (19.319 s) es la más lenta. Lo más probable es calentamiento o cambio de frecuencia de
  la laptop durante la corrida. Esto también infla el "~3 % de sobrecarga" de p = 1, que quizá sea menor.
- **numba p = 2:** ±0.312 s (3.2 %), con repeticiones entre 9.439 y 10.134 s; probablemente es la causa de que su f
  por punto (0.914) quede por debajo de la de p = 4 (0.981), algo que Amdahl no permite si f fuera constante.
- **numba p = 8:** la última repetición (4.774 s) es ~10 % más lenta que las primeras (~4.30 s); posible efecto
  térmico.
- **Ajuste de Amdahl con solo dos puntos:** f = 0.9786 / 0.9823 se obtiene con p = 2 y 4. Encaja bien con esos
  puntos, pero los topes de 46.7 / 56.5 son una extrapolación y no deben leerse como speedups alcanzables.
