"""K-Means PARALELO en MEMORIA COMPARTIDA (hilos con numba `prange`, estilo OpenMP).

Reparto de trabajo: los n puntos se dividen en `hilos` bloques contiguos y estaticos.
Cada hilo asigna sus puntos y acumula sumas/cuentas/inercia en acumuladores PRIVADOS
(sin carreras de datos ni locks). Despues se hace una reduccion secuencial de los
acumuladores (k*d valores por hilo, despreciable frente a n*k*d del paso principal).
Esa reduccion y la actualizacion de centroides son la parte secuencial de Amdahl.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from numba import njit, prange
import numba

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from comun.kmeans_comun import actualizar_centroides, bloque  # noqa: E402


@njit(parallel=True, cache=True)
def _paso_paralelo(X, C, sumas_l, cuentas_l, inercia_l, etiquetas, nt):
    n = X.shape[0]
    for t in prange(nt):
        lo = t * n // nt
        hi = (t + 1) * n // nt
        inercia_l[t] = bloque(X, lo, hi, C, sumas_l[t], cuentas_l[t], etiquetas)


def kmeans_numba(X, C0, hilos, max_iter=25, tol=0.0):
    """Devuelve (centroides, etiquetas, inercia, iteraciones) usando `hilos` hilos."""
    maximo = numba.config.NUMBA_NUM_THREADS
    if hilos > maximo:
        raise ValueError(
            f"hilos={hilos} supera NUMBA_NUM_THREADS={maximo}; "
            f"exporta NUMBA_NUM_THREADS>={hilos} antes de importar numba."
        )
    numba.set_num_threads(hilos)

    n = X.shape[0]
    k, d = C0.shape
    C = C0.copy()
    etiquetas = np.empty(n, dtype=np.int32)
    sumas_l = np.zeros((hilos, k, d))
    cuentas_l = np.zeros((hilos, k), dtype=np.int64)
    inercia_l = np.zeros(hilos)

    iteraciones = 0
    for _ in range(max_iter):
        sumas_l[:] = 0.0
        cuentas_l[:] = 0
        _paso_paralelo(X, C, sumas_l, cuentas_l, inercia_l, etiquetas, hilos)
        # Reduccion de los acumuladores privados (parte secuencial)
        C, desplazamiento = actualizar_centroides(C, sumas_l.sum(axis=0), cuentas_l.sum(axis=0))
        iteraciones += 1
        if desplazamiento <= tol:
            break

    sumas_l[:] = 0.0
    cuentas_l[:] = 0
    _paso_paralelo(X, C, sumas_l, cuentas_l, inercia_l, etiquetas, hilos)
    return C, etiquetas, float(inercia_l.sum()), iteraciones


def calentar_jit(hilos, d=4, k=2):
    """Compila el nucleo paralelo con datos diminutos (fuera de la medicion)."""
    X = np.random.default_rng(0).normal(size=(256, d))
    kmeans_numba(X, X[:k].copy(), hilos, max_iter=2)
