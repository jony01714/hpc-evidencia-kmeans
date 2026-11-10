"""Calcula metricas (tiempo, speedup, eficiencia, Amdahl) y genera las graficas.

Entrada : resultados/resultados_crudos.json
Salida  : resultados/resumen_metricas.csv, resultados/resumen.md,
          resultados/grafica_tiempo_vs_nucleos.png,
          resultados/grafica_speedup_real_vs_ideal.png,
          resultados/grafica_eficiencia.png

Definiciones (T1 = tiempo medio de la version SECUENCIAL, Tp = tiempo medio con p):
    S(p) = T1 / Tp                      E(p) = S(p) / p
    Amdahl: S(p) <= 1 / ((1-f) + f/p)   con f = fraccion paralelizable.
    f por punto: f(p) = (1 - 1/S) / (1 - 1/p)  (p>1)
    f global   : minimos cuadrados (busqueda en malla) SOLO sobre 1 < p <= nucleos fisicos.
                 Los p mayores son hilos logicos (SMT/Hyper-Threading) que comparten nucleo fisico,
                 lo que rompe el supuesto de Amdahl de p procesadores identicos: se muestran en tabla y
                 graficas pero no entran al ajuste. Por transparencia tambien se reporta f con todos los p.
La desviacion estandar de S(p) se propaga desde las de T1 y Tp (cociente de medias).
"""
from __future__ import annotations

import csv
import json
import os
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RAIZ = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RES = os.path.join(RAIZ, "resultados")

# Paleta categorica (slots 1 y 2, adyacentes validados) + tintas neutras
SUPERFICIE, TINTA, TINTA_2, REJILLA = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2dd"
MODELOS = {
    "numba": {"nombre": "Memoria compartida (numba, hilos)", "corto": "Mem. compartida", "color": "#2a78d6", "marca": "o"},
    "mpi": {"nombre": "Memoria distribuida (MPI, procesos)", "corto": "Mem. distribuida", "color": "#eb6834", "marca": "s"},
}


def nucleos_fisicos() -> int:
    """nucleos_por_socket * sockets de specs_equipo.json; si falta, os.cpu_count() // 2."""
    try:
        with open(os.path.join(RES, "specs_equipo.json")) as f:
            specs = json.load(f)
        return int(specs["nucleos_por_socket"]) * int(specs["sockets"])
    except Exception:
        return max(1, (os.cpu_count() or 2) // 2)


def cargar():
    with open(os.path.join(RES, "resultados_crudos.json")) as f:
        datos = json.load(f)
    seq = next(r for r in datos["resultados"] if r["modelo"] == "secuencial")
    return datos, seq


def ajustar_amdahl(ps, speedups, p_max=None):
    """f que minimiza el error cuadratico de S(p)=1/((1-f)+f/p) para 1 < p <= p_max."""
    pares = [(p, s) for p, s in zip(ps, speedups) if p > 1 and (p_max is None or p <= p_max)]
    if not pares:
        return float("nan")
    mejor_f, mejor_err = 0.0, float("inf")
    for f in np.linspace(0, 1, 10001):
        err = sum((1.0 / ((1 - f) + f / p) - s) ** 2 for p, s in pares)
        if err < mejor_err:
            mejor_f, mejor_err = float(f), err
    return mejor_f


def tabla(datos, seq, nf):
    t1 = st.mean(seq["tiempos"])
    s1 = st.stdev(seq["tiempos"]) if len(seq["tiempos"]) > 1 else 0.0
    filas = []
    for modelo in MODELOS:
        for r in sorted((x for x in datos["resultados"] if x["modelo"] == modelo), key=lambda x: x["p"]):
            tp = st.mean(r["tiempos"])
            sp = st.stdev(r["tiempos"]) if len(r["tiempos"]) > 1 else 0.0
            speed = t1 / tp
            sd_speed = speed * ((s1 / t1) ** 2 + (sp / tp) ** 2) ** 0.5
            p = r["p"]
            f_p = (1 - 1 / speed) / (1 - 1 / p) if p > 1 else float("nan")
            filas.append({"modelo": modelo, "p": p, "reps": len(r["tiempos"]), "tiempo_medio_s": tp,
                          "tiempo_std_s": sp, "speedup": speed, "speedup_std": sd_speed,
                          "eficiencia": speed / p, "fraccion_paralela_f": f_p,
                          "hilos_logicos_smt": p > nf})
    return t1, s1, filas


def _estilo(ax, titulo, xlabel, ylabel, ps):
    ax.set_facecolor(SUPERFICIE)
    ax.set_title(titulo, loc="left", fontsize=12, color=TINTA, pad=12, fontweight="bold")
    ax.set_xlabel(xlabel, color=TINTA_2)
    ax.set_ylabel(ylabel, color=TINTA_2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ps)
    ax.set_xticklabels([str(p) for p in ps])
    ax.minorticks_off()
    ax.grid(True, color=REJILLA, linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color("#c9c8c2")
    ax.tick_params(colors=TINTA_2)


def _figura():
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=150)
    fig.patch.set_facecolor(SUPERFICIE)
    return fig, ax


def _guardar(fig, nombre):
    fig.tight_layout()
    fig.savefig(os.path.join(RES, nombre), facecolor=SUPERFICIE)
    plt.close(fig)


def graficas(datos, t1, filas, fs, nf):
    ps = sorted({f["p"] for f in filas})

    # 1) Tiempo vs nucleos (barras de error = desviacion estandar)
    fig, ax = _figura()
    _estilo(ax, "Tiempo de ejecucion vs. numero de nucleos", "Hilos / procesos (p)", "Tiempo medio (s)", ps)
    ax.axhline(t1, color=TINTA_2, linestyle="--", linewidth=1.2)
    ax.text(ps[-1], t1, f" secuencial {t1:.2f}s", color=TINTA_2, va="bottom", ha="right", fontsize=9)
    for m, cfg in MODELOS.items():
        fm = [f for f in filas if f["modelo"] == m]
        if not fm:
            continue
        ax.errorbar([f["p"] for f in fm], [f["tiempo_medio_s"] for f in fm], yerr=[f["tiempo_std_s"] for f in fm],
                    color=cfg["color"], marker=cfg["marca"], markersize=8, linewidth=2, capsize=3,
                    markeredgecolor=SUPERFICIE, markeredgewidth=1.5, label=cfg["nombre"])
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, labelcolor=TINTA, loc="upper right", bbox_to_anchor=(1, 0.88))
    _guardar(fig, "grafica_tiempo_vs_nucleos.png")

    # 2) Speedup real vs ideal (+ ajuste de Amdahl solo con nucleos fisicos)
    fig, ax = _figura()
    _estilo(ax, "Speedup real vs. ideal", "Hilos / procesos (p)", "Speedup  S(p) = T1 / Tp", ps)
    ax.set_title("Speedup real vs. ideal", loc="left", fontsize=12, color=TINTA, pad=24, fontweight="bold")
    ax.text(0, 1.015, f"Amdahl ajustada con 1 < p \u2264 {nf} (nucleos fisicos); p > {nf} = hilos logicos SMT, "
            "fuera del ajuste", transform=ax.transAxes, fontsize=8.5, color=TINTA_2, va="bottom")
    if ps[-1] > nf:
        ax.axvspan(nf, ps[-1] * 1.15, color=REJILLA, alpha=0.45, linewidth=0, zorder=0)
        ax.set_xlim(ps[0] / 1.15, ps[-1] * 1.15)
        ax.text(ps[-1], 1.0, "hilos logicos (SMT)\ncomparten nucleo fisico", color=TINTA_2, fontsize=8,
                ha="right", va="bottom")
    ax.plot(ps, ps, color=TINTA_2, linestyle="--", linewidth=1.5, label="Ideal  S = p")
    pf = np.linspace(ps[0], ps[-1], 100)
    for m, cfg in MODELOS.items():
        fm = [f for f in filas if f["modelo"] == m]
        if not fm:
            continue
        f_glob = fs[m]
        ax.plot(pf, 1 / ((1 - f_glob) + f_glob / pf), color=cfg["color"], linestyle=":", linewidth=1.5, alpha=0.9,
                label=f"Amdahl ajustada ({cfg['corto']}): f={f_glob:.3f}, {nf} nucleos fisicos")
        ax.errorbar([f["p"] for f in fm], [f["speedup"] for f in fm], yerr=[f["speedup_std"] for f in fm],
                    color=cfg["color"], marker=cfg["marca"], markersize=8, linewidth=2, capsize=3,
                    markeredgecolor=SUPERFICIE, markeredgewidth=1.5, label=f"Real: {cfg['nombre']}")
    ax.legend(frameon=False, labelcolor=TINTA, loc="upper left", fontsize=8.5)
    _guardar(fig, "grafica_speedup_real_vs_ideal.png")

    # 3) Eficiencia
    fig, ax = _figura()
    _estilo(ax, "Eficiencia paralela  E(p) = S(p) / p", "Hilos / procesos (p)", "Eficiencia", ps)
    ax.axhline(1.0, color=TINTA_2, linestyle="--", linewidth=1.2)
    for m, cfg in MODELOS.items():
        fm = [f for f in filas if f["modelo"] == m]
        if not fm:
            continue
        ax.plot([f["p"] for f in fm], [f["eficiencia"] for f in fm], color=cfg["color"], marker=cfg["marca"],
                markersize=8, linewidth=2, markeredgecolor=SUPERFICIE, markeredgewidth=1.5, label=cfg["nombre"])
    ax.set_ylim(0, 1.1)
    ax.legend(frameon=False, labelcolor=TINTA, loc="lower left")
    _guardar(fig, "grafica_eficiencia.png")


def main():
    datos, seq = cargar()
    nf = nucleos_fisicos()
    t1, s1, filas = tabla(datos, seq, nf)
    fs, fs_todos = {}, {}
    for m in MODELOS:
        fm = [f for f in filas if f["modelo"] == m]
        ps_m, sp_m = [f["p"] for f in fm], [f["speedup"] for f in fm]
        fs[m] = ajustar_amdahl(ps_m, sp_m, p_max=nf) if fm else float("nan")
        fs_todos[m] = ajustar_amdahl(ps_m, sp_m) if fm else float("nan")

    with open(os.path.join(RES, "resumen_metricas.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)

    def tope(fv):
        return float("inf") if fv >= 1 else 1 / (1 - fv)

    c = datos["config"]
    lineas = [
        "# Resumen de resultados\n",
        f"Problema: K-Means, n={c['n']:,} puntos, d={c['d']}, k={c['k']}, {c['iteraciones']} iteraciones fijas, "
        f"{c['reps']} repeticiones por configuracion (media ± desv. estandar).\n",
        f"Nucleos fisicos del equipo: {nf}. Los p > {nf} son hilos logicos (SMT / Hyper-Threading) que comparten "
        "nucleo fisico; se marcan con (SMT) y no entran al ajuste de Amdahl.\n",
        f"Tiempo secuencial T1 = {t1:.3f} ± {s1:.3f} s\n",
        "| Modelo | p | Tiempo (s) | Speedup | Eficiencia | f (Amdahl, punto) |",
        "|---|---|---|---|---|---|",
    ]
    for f in filas:
        fp = "-" if f["p"] == 1 else f"{f['fraccion_paralela_f']:.3f}"
        etiqueta_p = f"{f['p']} (SMT)" if f["hilos_logicos_smt"] else str(f["p"])
        lineas.append(f"| {MODELOS[f['modelo']]['corto']} | {etiqueta_p} | {f['tiempo_medio_s']:.3f} ± {f['tiempo_std_s']:.3f} "
                      f"| {f['speedup']:.2f} ± {f['speedup_std']:.2f} | {f['eficiencia']:.2f} | {fp} |")
    lineas += [
        "",
        "## Ajuste global de Amdahl (minimos cuadrados)\n",
        f"| Modelo | f (1 < p ≤ {nf}, nucleos fisicos) | S max. teorico | f (todos los p > 1, solo referencia) | S max. teorico |",
        "|---|---|---|---|---|",
    ]
    for m, cfg in MODELOS.items():
        if np.isfinite(fs[m]):
            lineas.append(f"| {cfg['nombre']} | {fs[m]:.4f} | {tope(fs[m]):.1f} | {fs_todos[m]:.4f} | {tope(fs_todos[m]):.1f} |")
    lineas += [
        "",
        f"El valor oficial es el de nucleos fisicos. El ajuste con todos los p incluye hilos SMT, que no son "
        "procesadores independientes, y se reporta solo por transparencia.",
    ]
    with open(os.path.join(RES, "resumen.md"), "w") as f:
        f.write("\n".join(lineas) + "\n")
    print("\n".join(lineas))
    graficas(datos, t1, filas, fs, nf)
    print("\nGraficas guardadas en resultados/")


if __name__ == "__main__":
    main()
