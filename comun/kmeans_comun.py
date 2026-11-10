"""Utilidades compartidas por TODAS las versiones de K-Means (secuencial y paralelas).

Que las tres versiones usen exactamente el mismo nucleo de calculo (`bloque`),
los mismos datos y la misma inicializacion es lo que hace valida la comparacion
de rendimiento y la verificacion de correctitud.
"""
from __future__ import annotations

import numpy as np
from numba import njit


def generar_datos(n: int, d: int, k: int, semilla: int = 42) -> np.ndarray:
    """Genera `n` puntos en R^d como mezcla de `k` gaussianas (float64, C-contiguo)."""
    rng = np.random.default_rng(semilla)
    centros = rng.uniform(-10.0, 10.0, size=(k, d))
    asignacion = rng.integers(0, k, size=n)
    X = centros[asignacion] + rng.normal(0.0, 1.5, size=(n, d))
    return np.ascontiguousarray(X, dtype=np.float64)


def inicializar_centroides(X: np.ndarray, k: int, semilla: int = 7) -> np.ndarray:
    """Elige k filas distintas de X como centroides iniciales (determinista)."""
    rng = np.random.default_rng(semilla)
    idx = rng.choice(X.shape[0], size=k, replace=False)
    return np.ascontiguousarray(X[idx].copy())


@njit(cache=True)
def bloque(X, lo, hi, C, sumas, cuentas, etiquetas):
    """Paso de asignacion + acumulacion para las filas [lo, hi) de X.

    - Asigna cada punto al centroide mas cercano (distancia euclidiana al cuadrado).
    - Acumula en `sumas`/`cuentas` los puntos de cada cluster (para el nuevo centroide).
    - Escribe la etiqueta de la fila i en `etiquetas[i]` (indice local de `X`: con MPI,
      `X` es el bloque propio del proceso y lo=0, hi=filas locales).
    Devuelve la inercia parcial (suma de distancias^2) del bloque.
    """
    k = C.shape[0]
    d = C.shape[1]
    inercia = 0.0
    for i in range(lo, hi):
        mejor = 0
        mejor_dist = np.inf
        for c in range(k):
            dist = 0.0
            for j in range(d):
                diff = X[i, j] - C[c, j]
                dist += diff * diff
            if dist < mejor_dist:
                mejor_dist = dist
                mejor = c
        etiquetas[i] = mejor
        inercia += mejor_dist
        cuentas[mejor] += 1
        for j in range(d):
            sumas[mejor, j] += X[i, j]
    return inercia


def actualizar_centroides(C, sumas, cuentas):
    """Nuevo centroide = media del cluster; un cluster vacio conserva su centroide previo.

    Devuelve (C_nuevo, desplazamiento_maximo).
    """
    C_nuevo = C.copy()
    llenos = cuentas > 0
    C_nuevo[llenos] = sumas[llenos] / cuentas[llenos][:, None]
    desplazamiento = float(np.max(np.linalg.norm(C_nuevo - C, axis=1)))
    return C_nuevo, desplazamiento
