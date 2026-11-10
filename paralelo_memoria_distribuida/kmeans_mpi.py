"""K-Means PARALELO en MEMORIA DISTRIBUIDA (procesos MPI con mpi4py).

Reparto de trabajo: el proceso 0 genera/posee el conjunto de datos y lo reparte por
bloques de filas con `Scatterv`; ningun otro proceso ve el resto de los datos.
En cada iteracion cada proceso asigna SUS puntos y acumula sumas/cuentas locales; una
unica `Allreduce` suma esos acumuladores (k*d + k valores) y todos los procesos calculan
el mismo nuevo centroide, asi que no hace falta un broadcast adicional.
Costos de comunicacion: el Scatterv inicial (O(n*d)) y una Allreduce por iteracion.

Ejecucion:  mpiexec -n P python kmeans_mpi.py ...   (ver experimentos/)
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
from mpi4py import MPI

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from comun.kmeans_comun import actualizar_centroides, bloque  # noqa: E402


def _limites(n, p):
    """Filas [lo, hi) de cada rank (bloques contiguos, balanceados a +-1 fila)."""
    return [(r * n // p, (r + 1) * n // p) for r in range(p)]


def kmeans_mpi(comm, X, C0, max_iter=25, tol=0.0, devolver_etiquetas=False):
    """K-Means distribuido. `X` solo es necesario en rank 0 (en los demas puede ser None).

    Devuelve en TODOS los procesos (centroides, inercia, iteraciones); en rank 0 ademas
    las etiquetas globales si `devolver_etiquetas` (None en los demas).
    """
    rank, p = comm.Get_rank(), comm.Get_size()
    k, d = C0.shape

    # ---- Distribucion de datos (comunicacion O(n*d), se incluye en el tiempo medido) ----
    n = comm.bcast(X.shape[0] if rank == 0 else None, root=0)
    lims = _limites(n, p)
    filas = np.array([hi - lo for lo, hi in lims], dtype=np.int64)
    cuentas_env = filas * d
    desplaz = np.array([lo * d for lo, _ in lims], dtype=np.int64)
    X_local = np.empty((filas[rank], d), dtype=np.float64)
    comm.Scatterv([X, cuentas_env, desplaz, MPI.DOUBLE] if rank == 0 else None, X_local, root=0)

    C = C0.copy()
    etiquetas = np.empty(filas[rank], dtype=np.int32)
    sumas = np.zeros((k, d))
    cuentas = np.zeros(k, dtype=np.int64)
    # Buffer unico para una sola Allreduce por iteracion: [sumas (k*d) | cuentas (k)]
    buf_local = np.zeros(k * d + k)
    buf_global = np.zeros(k * d + k)

    def paso(C):
        sumas[:] = 0.0
        cuentas[:] = 0
        inercia_local = bloque(X_local, 0, X_local.shape[0], C, sumas, cuentas, etiquetas)
        buf_local[: k * d] = sumas.ravel()
        buf_local[k * d :] = cuentas
        comm.Allreduce(buf_local, buf_global, op=MPI.SUM)
        return inercia_local, buf_global[: k * d].reshape(k, d), buf_global[k * d :].astype(np.int64)

    iteraciones = 0
    for _ in range(max_iter):
        _, sumas_g, cuentas_g = paso(C)
        C, desplazamiento = actualizar_centroides(C, sumas_g, cuentas_g)
        iteraciones += 1
        if desplazamiento <= tol:
            break

    inercia_local, _, _ = paso(C)
    inercia = comm.allreduce(inercia_local, op=MPI.SUM)

    etiquetas_globales = None
    if devolver_etiquetas:
        etiquetas_globales = np.empty(n, dtype=np.int32) if rank == 0 else None
        comm.Gatherv(etiquetas, [etiquetas_globales, filas, np.array([lo for lo, _ in lims]), MPI.INT]
                     if rank == 0 else None, root=0)
    return C, float(inercia), iteraciones, etiquetas_globales


def calentar_jit(comm, d=4, k=2):
    """Compila el nucleo en cada proceso con datos diminutos (fuera de la medicion)."""
    X = np.random.default_rng(0).normal(size=(256, d)) if comm.Get_rank() == 0 else None
    kmeans_mpi(comm, X, np.random.default_rng(0).normal(size=(k, d)), max_iter=2)
