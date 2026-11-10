"""Verifica que las versiones paralelas producen el mismo resultado que la secuencial.

Compara (centroides, etiquetas, inercia, iteraciones) de:
  - secuencial (referencia)
  - memoria compartida (numba) con 1, 2, 3, 4 y 8 hilos
  - memoria distribuida (MPI) con 1, 2, 3, 4 y 7 procesos (cantidades NO divisibles
    a proposito, para probar el reparto desigual de filas)
  - scikit-learn (verificacion independiente; opcional, si esta instalado)

Tolerancia: las sumas en coma flotante se hacen en distinto orden segun el reparto,
por eso se exige igualdad exacta de etiquetas y cercania numerica (rtol=1e-9) en el resto.

Uso:  python experimentos/verificar_correctitud.py
Salida: resultados/verificacion_correctitud.txt  (codigo de salida 1 si algo falla)
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np

RAIZ = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, RAIZ)
os.environ.setdefault("NUMBA_NUM_THREADS", str(max(8, os.cpu_count() or 1)))

from comun.kmeans_comun import generar_datos, inicializar_centroides  # noqa: E402
from experimentos.utilidades import comando_trabajo, mpiexec_disponible  # noqa: E402
from paralelo_memoria_compartida.kmeans_numba import kmeans_numba  # noqa: E402
from secuencial.kmeans_secuencial import kmeans_secuencial  # noqa: E402

N, D, K, ITER = 50_001, 8, 7, 300  # N impar a proposito
RTOL = 1e-9


def comparar(nombre, ref, otro, lineas):
    C_r, e_r, in_r, it_r = ref
    C_o, e_o, in_o, it_o = otro
    ok_c = np.allclose(C_r, C_o, rtol=RTOL, atol=1e-9)
    difer = int(np.sum(e_r != e_o))
    ok_e = difer == 0
    ok_i = abs(in_r - in_o) <= RTOL * abs(in_r)
    ok_it = it_r == it_o
    ok = ok_c and ok_e and ok_i and ok_it
    lineas.append(
        f"[{'OK ' if ok else 'FALLA'}] {nombre:<28} max|dC|={np.max(np.abs(C_r - C_o)):.2e}  "
        f"etiquetas distintas={difer}  dInercia_rel={abs(in_r - in_o) / abs(in_r):.2e}  iter={it_o}"
    )
    return ok


def main():
    X = generar_datos(N, D, K)
    C0 = inicializar_centroides(X, K)
    ref = kmeans_secuencial(X, C0, ITER, tol=1e-6)  # converge antes de ITER: prueba el criterio de paro
    lineas = [f"Datos: n={N}, d={D}, k={K}; referencia: secuencial ({ref[3]} iteraciones, tol=1e-6)"]
    todo_ok = True

    for hilos in (1, 2, 3, 4, 8):
        res = kmeans_numba(X, C0, hilos, ITER, tol=1e-6)
        todo_ok &= comparar(f"memoria compartida, {hilos} hilos", ref, res, lineas)

    if mpiexec_disponible():
        for p in (1, 2, 3, 4, 7):
            with tempfile.TemporaryDirectory() as tmp:
                ruta = os.path.join(tmp, "r.npz")
                cmd, env = comando_trabajo("mpi", p, [
                    "--n", str(N), "--d", str(D), "--k", str(K), "--iter", str(ITER),
                    "--tol", "1e-6", "--reps", "1", "--etiquetas-npz", ruta])
                subprocess.run(cmd, env=env, check=True, capture_output=True)
                z = np.load(ruta)
                res = (z["centroides"], z["etiquetas"], float(z["inercia"]), int(z["iteraciones"]))
            todo_ok &= comparar(f"memoria distribuida, {p} procs", ref, res, lineas)
    else:
        lineas.append("[OMITIDO] MPI: mpiexec no encontrado")
        todo_ok = False

    try:  # verificacion independiente con scikit-learn (iteraciones fijas, mismos centroides iniciales)
        from sklearn.cluster import KMeans
        fijo = kmeans_secuencial(X, C0, 15, tol=0.0)
        km = KMeans(n_clusters=K, init=C0, n_init=1, max_iter=15, tol=0.0, algorithm="lloyd").fit(X)
        ok = np.allclose(np.sort(fijo[0], axis=0), np.sort(km.cluster_centers_, axis=0), rtol=1e-6, atol=1e-6)
        lineas.append(f"[{'OK ' if ok else 'FALLA'}] secuencial vs scikit-learn (15 iter fijas): "
                      f"inercia {fijo[2]:.6f} vs {km.inertia_:.6f}")
        todo_ok &= ok
    except ImportError:
        lineas.append("[OMITIDO] scikit-learn no instalado (verificacion independiente)")

    lineas.append("RESULTADO GLOBAL: " + ("CORRECTO" if todo_ok else "HAY DIFERENCIAS"))
    texto = "\n".join(lineas)
    print(texto)
    os.makedirs(os.path.join(RAIZ, "resultados"), exist_ok=True)
    with open(os.path.join(RAIZ, "resultados", "verificacion_correctitud.txt"), "w") as f:
        f.write(texto + "\n")
    sys.exit(0 if todo_ok else 1)


if __name__ == "__main__":
    main()
