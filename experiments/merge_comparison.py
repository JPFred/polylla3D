# Compares PolyllaEdge without the merge phase of single tetrahedra and with it,
# once per merge criterion, on quality meshes of 1000, 5000 and 10000 points.

import contextlib
import io
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from newMesh import EdgeTetrahedronMesh  # noqa: E402
from PolyllaEdge import PolyllaEdge  # noqa: E402
from merge_criteria import ConvexityCriterion, LargestFaceCriterion, EquilateralFaceCriterion, KernelCriterion  # noqa: E402
from metrics import convexity_index, edge_ratio, kernel_ratio, solidity  # noqa: E402
from generate_quality_meshes import quality_mesh  # noqa: E402

# Generated meshes go to data/experiments/ (ignored by git)
OUT = ROOT / "data" / "experiments" / "merge_comparison"
RESULTS = ROOT / "experiments" / "results" / "merge_comparison.json"

# name: maximum tetrahedron volume (-a), same as neighbor_study.py
QUALITY = {"1000quality": 0.00038, "5000quality": 0.0000695, "10000quality": 0.000033}
EPS = 1e-9
CRITERIA = [ConvexityCriterion(), LargestFaceCriterion(), EquilateralFaceCriterion(), KernelCriterion()]


# Mean of each metric among the polyhedra with the same number of tetrahedra,
# to compare criteria without the effect of the polyhedron size
def by_size(sizes, sol, ker, er):
    groups = defaultdict(list)
    for i, n in enumerate(sizes):
        groups[n].append(i)
    return {n: dict(poliedros=len(idx),
                    convexos_pct=100 * sum(sol[i] >= 1 - EPS for i in idx) / len(idx),
                    solidez_media=statistics.mean(sol[i] for i in idx),
                    kernel_rate=100 * sum(ker[i] > 0 for i in idx) / len(idx),
                    kernel_ratio_media=statistics.mean(ker[i] for i in idx),
                    edge_ratio_media=statistics.mean(er[i] for i in idx))
            for n, idx in sorted(groups.items())}


def run(mesh, criterion):
    t0 = time.time()
    edge = PolyllaEdge(mesh, merge_criterion=criterion)
    elapsed = time.time() - t0
    with contextlib.redirect_stdout(io.StringIO()) as log:
        errors = edge.validate()
    non_manifold = int(log.getvalue().split("non-manifold edges:")[1].split()[0])
    polys = edge.polyhedral_mesh
    sizes = [len(p.tetras) for p in polys]
    sol = [float(solidity(mesh, p.tetras)) for p in polys]
    area = [float(convexity_index(mesh, p.tetras)) for p in polys]
    ker = [float(kernel_ratio(mesh, p.tetras)) for p in polys]
    er = [float(edge_ratio(mesh, p.tetras)) for p in polys]
    return dict(
        criterio=criterion.name if criterion else "sin union",
        poliedros=len(polys), solos=sizes.count(1),
        tets_por_poliedro_media=statistics.mean(sizes), tets_por_poliedro_max=max(sizes),
        tamanos=dict(sorted(Counter(sizes).items())),
        convexos=sum(s >= 1 - EPS for s in sol), convexos_pct=100 * sum(s >= 1 - EPS for s in sol) / len(polys),
        convexos_multi=sum(s >= 1 - EPS for s, n in zip(sol, sizes) if n > 1),
        solidez_media=statistics.mean(sol), solidez_mediana=statistics.median(sol), solidez_min=min(sol),
        indice_area_media=statistics.mean(area), indice_area_min=min(area), indice_area_max=max(area),
        kernel_rate=100 * sum(k > 0 for k in ker) / len(polys),
        kernel_ratio_media=statistics.mean(ker), kernel_ratio_min=min(ker),
        edge_ratio_media=statistics.mean(er), edge_ratio_min=min(er),
        por_tamano=by_size(sizes, sol, ker, er),
        errores_validate=errors, aristas_no_manifold=non_manifold,
        unidos=edge.merged_singles, en_pareja=edge.paired_singles, sin_unir=edge.unmerged_singles,
        tiempo_s=elapsed,
    )


def report(results):
    cols = [("poliedros", "{:>9}"), ("solos", "{:>6}"), ("tets_por_poliedro_media", "{:>8.2f}"),
            ("tets_por_poliedro_max", "{:>6}"), ("convexos_multi", "{:>8}"), ("convexos_pct", "{:>8.1f}"),
            ("solidez_media", "{:>8.4f}"), ("solidez_min", "{:>8.4f}"), ("indice_area_media", "{:>8.4f}"),
            ("kernel_rate", "{:>7.1f}"), ("kernel_ratio_media", "{:>8.4f}"), ("edge_ratio_media", "{:>8.4f}"),
            ("errores_validate", "{:>5}"), ("aristas_no_manifold", "{:>5}"), ("unidos", "{:>6}"),
            ("en_pareja", "{:>7}"), ("tiempo_s", "{:>7.1f}")]
    head = ["polys", "solos", "tet/pol", "max", "conv>1", "%conv", "sol.med", "sol.min", "area", "kern%", "ker.med", "er.med", "err", "nm",
            "unid", "pareja", "t(s)"]
    widths = [9, 6, 8, 6, 8, 8, 8, 8, 8, 7, 8, 8, 5, 5, 6, 7, 7]
    print(f"{'malla':<14}{'criterio':<18}" + "".join(f"{h:>{w}}" for h, w in zip(head, widths)))
    for r in results:
        for c in r["corridas"]:
            print(f"{r['malla']:<14}{c['criterio']:<18}" + "".join(fmt.format(c[k]) for k, fmt in cols))


# Metrics by polyhedron size, one row per criterion; groups with fewer than
# min_group polyhedra are skipped because their means are not reliable
def report_by_size(results, min_group=30):
    print(f"\n{'malla':<14}{'tets':>5}  {'criterio':<18}{'polys':>7}{'%conv':>7}{'sol.med':>8}"
          f"{'kern%':>7}{'ker.med':>8}{'er.med':>8}")
    for r in results:
        for n in sorted({n for c in r["corridas"] for n in c["por_tamano"]}):
            for c in r["corridas"]:
                g = c["por_tamano"].get(n)
                if g is None or g["poliedros"] < min_group:
                    continue
                print(f"{r['malla']:<14}{n:>5}  {c['criterio']:<18}{g['poliedros']:>7}{g['convexos_pct']:>7.1f}"
                      f"{g['solidez_media']:>8.4f}{g['kernel_rate']:>7.1f}{g['kernel_ratio_media']:>8.4f}"
                      f"{g['edge_ratio_media']:>8.4f}")


if __name__ == "__main__":
    results = []
    for name, vol in QUALITY.items():
        base = quality_mesh(name, vol, OUT)
        with contextlib.redirect_stdout(io.StringIO()):
            mesh = EdgeTetrahedronMesh(f"{base}.node", f"{base}.face", f"{base}.ele", f"{base}.edge")
        runs = [run(mesh, None)] + [run(mesh, c) for c in CRITERIA]
        results.append(dict(malla=name, params={"q": 2.0, "a": vol}, puntos=mesh.n_nodes,
                            tetraedros=mesh.n_tetrahedrons, corridas=runs))
        print("listo", name, flush=True)
    with open(RESULTS, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    report(results)
    report_by_size(results)
