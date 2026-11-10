"""Registra las especificaciones del equipo de computo (requisito de la evidencia).

Uso:  python experimentos/specs_equipo.py     ->  resultados/specs_equipo.json
Es Linux-first (lee /proc y usa lscpu); en otros sistemas rellena lo que pueda.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys

RAIZ = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def _cmd(*args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return ""


def obtener_specs() -> dict:
    specs = {
        "sistema": platform.platform(),
        "python": sys.version.split()[0],
        "nucleos_logicos": os.cpu_count(),
    }
    lscpu = _cmd("lscpu")
    for clave, patron in {
        "cpu_modelo": r"Model name:\s*(.+)",
        "hilos_por_nucleo": r"Thread\(s\) per core:\s*(\d+)",
        "nucleos_por_socket": r"Core\(s\) per socket:\s*(\d+)",
        "sockets": r"Socket\(s\):\s*(\d+)",
        "cache_L3": r"L3 cache:\s*(.+)",
        "frecuencia_max_mhz": r"CPU max MHz:\s*([\d.]+)",
    }.items():
        m = re.search(patron, lscpu)
        if m:
            specs[clave] = m.group(1).strip()
    try:
        with open("/proc/meminfo") as f:
            kb = int(re.search(r"MemTotal:\s+(\d+)", f.read()).group(1))
        specs["ram_gb"] = round(kb / 1024 / 1024, 1)
    except Exception:
        pass
    if shutil.which("nvidia-smi"):
        specs["gpu"] = _cmd("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader").strip() or "sin GPU NVIDIA"
    else:
        lspci = _cmd("lspci")
        gpus = [l.split(": ", 1)[-1] for l in lspci.splitlines() if re.search(r"VGA|3D|Display", l)]
        specs["gpu"] = "; ".join(gpus) if gpus else "no detectada"
    versiones = {}
    for mod in ("numpy", "numba", "mpi4py", "matplotlib", "sklearn"):
        try:
            versiones[mod] = __import__(mod).__version__
        except Exception:
            versiones[mod] = "no instalado"
    specs["versiones"] = versiones
    mpi = shutil.which("mpiexec")
    if mpi:
        specs["mpi"] = _version_mpi(_cmd(mpi, "--version"))
    return specs


def _version_mpi(salida: str) -> str:
    """Implementacion y version de MPI a partir de `mpiexec --version`.

    - MPICH (lanzador Hydra): "HYDRA build details:" + "    Version:   4.2.1"
    - Intel MPI (tambien Hydra): "Intel(R) MPI Library for Linux* OS, Version 2021.x ..."
    - Open MPI: "mpiexec (Open MPI) 4.1.6"
    """
    lineas = [l.strip() for l in salida.splitlines() if l.strip()]
    if not lineas:
        return "desconocida"
    m = re.search(r"Open MPI\)?\s*([\d.]+)", salida)
    if m:
        return f"Open MPI {m.group(1)}"
    intel = next((l for l in lineas if "Intel(R) MPI" in l), None)
    if intel:
        return intel
    m = re.search(r"^\s*Version:\s*(\S+)", salida, re.MULTILINE)
    if m:
        impl = "MPICH" if "HYDRA" in salida.upper() else "MPI"
        return f"{impl} {m.group(1)} (Hydra, Version: {m.group(1)})"
    return lineas[0]


def main():
    specs = obtener_specs()
    os.makedirs(os.path.join(RAIZ, "resultados"), exist_ok=True)
    with open(os.path.join(RAIZ, "resultados", "specs_equipo.json"), "w") as f:
        json.dump(specs, f, indent=2, ensure_ascii=False)
    print(json.dumps(specs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
