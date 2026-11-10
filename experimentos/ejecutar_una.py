"""Ejecuta UNA configuracion (modelo, p) y escribe una linea JSON con los tiempos.

Lo invoca `ejecutar_experimentos.py` (uno por proceso para aislar la memoria y las
variables de entorno), pero tambien sirve para pruebas manuales:

    python experimentos/ejecutar_una.py --modelo secuencial
    python experimentos/ejecutar_una.py --modelo numba --p 4
    mpiexec -n 4 python experimentos/ejecutar_una.py --modelo mpi

Que se mide: desde que los datos ya existen hasta tener los centroides finales
(el JIT se calienta antes y no cuenta). En MPI el tiempo INCLUYE el Scatterv inicial,
porque repartir los datos es un costo real de memoria distribuida.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

RAIZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, RAIZ)


def _argumentos():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", choices=["secuencial", "numba", "mpi"], required=True)
    ap.add_argument("--p", type=int, default=1, help="hilos (numba). En MPI se usa el tamano del comunicador")
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--iter", type=int, default=25, dest="iteraciones")
    ap.add_argument("--tol", type=float, default=0.0, help="0 = iteraciones fijas (carga constante)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--semilla", type=int, default=42)
    ap.add_argument("--etiquetas-npz", default=None,
                    help="si se indica, guarda centroides/etiquetas/inercia (verificacion de correctitud)")
    return ap.parse_args()


def _guardar_npz(ruta, C, etiquetas, inercia, iteraciones):
    import numpy as np
    np.savez(ruta, centroides=C, etiquetas=etiquetas, inercia=inercia, iteraciones=iteraciones)


def main():
    a = _argumentos()
    import numpy as np
    from comun.kmeans_comun import generar_datos, inicializar_centroides

    tiempos, resultado = [], None

    if a.modelo == "secuencial":
        from secuencial.kmeans_secuencial import calentar_jit, kmeans_secuencial
        X = generar_datos(a.n, a.d, a.k, a.semilla)
        C0 = inicializar_centroides(X, a.k)
        calentar_jit(a.d, a.k)
        for _ in range(a.reps):
            t0 = time.perf_counter()
            resultado = kmeans_secuencial(X, C0, a.iteraciones, a.tol)
            tiempos.append(time.perf_counter() - t0)
        C, etiquetas, inercia, it = resultado
        p = 1

    elif a.modelo == "numba":
        from paralelo_memoria_compartida.kmeans_numba import calentar_jit, kmeans_numba
        X = generar_datos(a.n, a.d, a.k, a.semilla)
        C0 = inicializar_centroides(X, a.k)
        calentar_jit(a.p, a.d, a.k)
        for _ in range(a.reps):
            t0 = time.perf_counter()
            resultado = kmeans_numba(X, C0, a.p, a.iteraciones, a.tol)
            tiempos.append(time.perf_counter() - t0)
        C, etiquetas, inercia, it = resultado
        p = a.p

    else:  # mpi
        from mpi4py import MPI
        from paralelo_memoria_distribuida.kmeans_mpi import calentar_jit, kmeans_mpi
        comm = MPI.COMM_WORLD
        rank, p = comm.Get_rank(), comm.Get_size()
        X = C0 = None
        if rank == 0:
            X = generar_datos(a.n, a.d, a.k, a.semilla)
            C0 = inicializar_centroides(X, a.k)
        C0 = comm.bcast(C0, root=0)  # la inicializacion no se mide
        calentar_jit(comm, a.d, a.k)
        pedir = a.etiquetas_npz is not None
        for r in range(a.reps):
            comm.Barrier()
            t0 = time.perf_counter()
            resultado = kmeans_mpi(comm, X, C0, a.iteraciones, a.tol, devolver_etiquetas=(pedir and r == a.reps - 1))
            comm.Barrier()  # el tiempo es el del proceso MAS LENTO
            tiempos.append(time.perf_counter() - t0)
        C, inercia, it, etiquetas = resultado
        if rank != 0:
            return

    if a.etiquetas_npz:
        _guardar_npz(a.etiquetas_npz, C, etiquetas, inercia, it)
    print(json.dumps({
        "modelo": a.modelo, "p": p, "n": a.n, "d": a.d, "k": a.k,
        "iter_max": a.iteraciones, "tol": a.tol, "iteraciones": it,
        "tiempos": tiempos, "inercia": inercia,
    }))


if __name__ == "__main__":
    main()
