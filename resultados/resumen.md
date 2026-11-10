# Resumen de resultados

Problema: K-Means, n=2,000,000 puntos, d=16, k=16, 25 iteraciones fijas, 5 repeticiones por configuracion (media ± desv. estandar).

Nucleos fisicos del equipo: 4. Los p > 4 son hilos logicos (SMT / Hyper-Threading) que comparten nucleo fisico; se marcan con (SMT) y no entran al ajuste de Amdahl.

Tiempo secuencial T1 = 18.043 ± 0.043 s

| Modelo | p | Tiempo (s) | Speedup | Eficiencia | f (Amdahl, punto) |
|---|---|---|---|---|---|
| Mem. compartida | 1 | 18.546 ± 0.586 | 0.97 ± 0.03 | 0.97 | - |
| Mem. compartida | 2 | 9.794 ± 0.312 | 1.84 ± 0.06 | 0.92 | 0.914 |
| Mem. compartida | 4 | 4.773 ± 0.093 | 3.78 ± 0.07 | 0.94 | 0.981 |
| Mem. compartida | 8 (SMT) | 4.440 ± 0.203 | 4.06 ± 0.19 | 0.51 | 0.862 |
| Mem. distribuida | 1 | 18.514 ± 0.563 | 0.97 ± 0.03 | 0.97 | - |
| Mem. distribuida | 2 | 9.485 ± 0.108 | 1.90 ± 0.02 | 0.95 | 0.949 |
| Mem. distribuida | 4 | 4.736 ± 0.118 | 3.81 ± 0.10 | 0.95 | 0.983 |
| Mem. distribuida | 8 (SMT) | 4.193 ± 0.085 | 4.30 ± 0.09 | 0.54 | 0.877 |

## Ajuste global de Amdahl (minimos cuadrados)

| Modelo | f (1 < p ≤ 4, nucleos fisicos) | S max. teorico | f (todos los p > 1, solo referencia) | S max. teorico |
|---|---|---|---|---|
| Memoria compartida (numba, hilos) | 0.9786 | 46.7 | 0.8824 | 8.5 |
| Memoria distribuida (MPI, procesos) | 0.9823 | 56.5 | 0.8944 | 9.5 |

El valor oficial es el de nucleos fisicos. El ajuste con todos los p incluye hilos SMT, que no son procesadores independientes, y se reporta solo por transparencia.
