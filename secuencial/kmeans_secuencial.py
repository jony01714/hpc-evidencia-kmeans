"""K-Means (Lloyd) SECUENCIAL: referencia de correctitud y de tiempo (T1).

Usa el mismo nucleo compilado (`bloque`) que las versiones paralelas, ejecutado en un
solo hilo. Asi, cualquier diferencia de tiempo se debe al paralelismo y no a que la
base sea un codigo mas lento o distinto.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from comun.kmeans_comun import actualizar_centroides, bloque  # noqa: E402


def kmeans_secuencial(X, C0, max_iter=25, tol=0.0):
    """Devuelve (centroides, etiquetas, inercia, iteraciones).

    Con tol=0 se ejecutan exactamente `max_iter` iteraciones (carga de trabajo fija,
    util para medir tiempos); con tol>0 se detiene al converger.
    """
    n = X.shape[0]
    k, d = C0.shape
    C = C0.copy()
    etiquetas = np.empty(n, dtype=np.int32)
    sumas = np.zeros((k, d))
    cuentas = np.zeros(k, dtype=np.int64)

    iteraciones = 0
    for _ in range(max_iter):
        sumas[:] = 0.0
        cuentas[:] = 0
        bloque(X, 0, n, C, sumas, cuentas, etiquetas)
        C, desplazamiento = actualizar_centroides(C, sumas, cuentas)
        iteraciones += 1
        if desplazamiento <= tol:
            break

    # Pasada final: etiquetas e inercia respecto a los centroides finales.
    sumas[:] = 0.0
    cuentas[:] = 0
    inercia = bloque(X, 0, n, C, sumas, cuentas, etiquetas)
    return C, etiquetas, float(inercia), iteraciones


def calentar_jit(d=4, k=2):
    """Compila el nucleo con datos diminutos para que la compilacion no cuente en los tiempos."""
    X = np.random.default_rng(0).normal(size=(64, d))
    kmeans_secuencial(X, X[:k].copy(), max_iter=2)
