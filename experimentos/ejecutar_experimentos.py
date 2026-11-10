"""Corre TODOS los experimentos: secuencial + memoria compartida + memoria distribuida.

Para cada p en --ps (por defecto 1,2,4,8) mide `--reps` repeticiones (>=3 exigido por la
rubrica) de cada version paralela y las guarda en resultados/resultados_crudos.json.
Despues ejecuta `analizar.py` para generar tablas y graficas.

Recomendaciones para mediciones limpias: laptop conectada a la corriente, cerrar
navegador/IDE pesados y no tocar la maquina mientras corre.

Uso:
    python experimentos/ejecutar_experimentos.py                  # configuracion completa
    python experimentos/ejecutar_experimentos.py --n 300000 --iter 10 --reps 3   # prueba rapida
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

RAIZ = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, RAIZ)
from experimentos.utilidades import comando_trabajo, mpiexec_disponible  # noqa: E402
from experimentos.specs_equipo import main as guardar_specs  # noqa: E402


def correr(modelo, p, extra):
    cmd, env = comando_trabajo(modelo, p, extra)
    t0 = time.time()
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Fallo {modelo} p={p}:\n{r.stderr[-2000:]}")
    linea = [l for l in r.stdout.splitlines() if l.startswith("{")][-1]
    res = json.loads(linea)
    print(f"  {modelo:<10} p={p:<2} media={sum(res['tiempos']) / len(res['tiempos']):8.3f}s "
          f"(reps={len(res['tiempos'])}, total {time.time() - t0:5.1f}s)", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ps", default="1,2,4,8", help="lista de hilos/procesos separados por coma")
    ap.add_argument("--n", type=int, default=2_000_000)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--iter", type=int, default=25, dest="iteraciones")
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    if a.reps < 3:
        sys.exit("La rubrica exige al menos 3 repeticiones por configuracion (--reps >= 3).")
    ps = [int(x) for x in a.ps.split(",")]
    extra = ["--n", str(a.n), "--d", str(a.d), "--k", str(a.k),
             "--iter", str(a.iteraciones), "--reps", str(a.reps), "--tol", "0"]

    guardar_specs()
    print(f"\nExperimentos: n={a.n}, d={a.d}, k={a.k}, {a.iteraciones} iteraciones fijas, "
          f"{a.reps} repeticiones, p={ps}\n")
    resultados = [correr("secuencial", 1, extra)]
    for p in ps:
        resultados.append(correr("numba", p, extra))
    if mpiexec_disponible():
        for p in ps:
            resultados.append(correr("mpi", p, extra))
    else:
        print("AVISO: no hay mpiexec; se omite la version de memoria distribuida.")

    # Chequeo de correctitud embebido: todas las inercias deben coincidir con la secuencial.
    ref = resultados[0]["inercia"]
    for r in resultados[1:]:
        dif = abs(r["inercia"] - ref) / abs(ref)
        if dif > 1e-9:
            print(f"  ADVERTENCIA: {r['modelo']} p={r['p']} difiere de la secuencial (inercia rel {dif:.2e})")

    os.makedirs(os.path.join(RAIZ, "resultados"), exist_ok=True)
    with open(os.path.join(RAIZ, "resultados", "resultados_crudos.json"), "w") as f:
        json.dump({"config": vars(a), "resultados": resultados}, f, indent=2)
    print("\nGuardado: resultados/resultados_crudos.json\nGenerando tablas y graficas...")
    subprocess.run([sys.executable, os.path.join(RAIZ, "experimentos", "analizar.py")], check=True)


if __name__ == "__main__":
    main()
