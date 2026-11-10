"""Funciones auxiliares para lanzar procesos de medicion (numba y MPI)."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

RAIZ = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SCRIPT_UNA = os.path.join(RAIZ, "experimentos", "ejecutar_una.py")


def mpiexec_disponible() -> str | None:
    return shutil.which("mpiexec") or shutil.which("mpirun")


def comando_mpi(p: int) -> list[str]:
    """Prefijo `mpiexec -n p` con los flags que necesite la implementacion instalada."""
    exe = mpiexec_disponible()
    if exe is None:
        raise RuntimeError("No se encontro mpiexec/mpirun. Instala MPICH u OpenMPI (ver README).")
    version = subprocess.run([exe, "--version"], capture_output=True, text=True).stdout
    cmd = [exe, "-n", str(p)]
    if "Open MPI" in version or "OpenRTE" in version:
        cmd.insert(1, "--oversubscribe")  # permite p > nucleos
        if os.geteuid() == 0:
            cmd.insert(1, "--allow-run-as-root")  # solo ocurre dentro de Docker
    return cmd


def comando_trabajo(modelo: str, p: int, args_extra: list[str]) -> tuple[list[str], dict]:
    """Devuelve (comando, entorno) para ejecutar una configuracion en un proceso aparte."""
    entorno = dict(os.environ)
    entorno["PYTHONPATH"] = RAIZ + os.pathsep + entorno.get("PYTHONPATH", "")
    base = [sys.executable, SCRIPT_UNA, "--modelo", modelo] + args_extra
    if modelo == "mpi":
        # Un hilo de numba por proceso MPI: el paralelismo viene SOLO de MPI.
        entorno["NUMBA_NUM_THREADS"] = "1"
        entorno["OMP_NUM_THREADS"] = "1"
        return comando_mpi(p) + base, entorno
    if modelo == "numba":
        entorno["NUMBA_NUM_THREADS"] = str(p)
        return base + ["--p", str(p)], entorno
    entorno["NUMBA_NUM_THREADS"] = "1"
    return base, entorno
