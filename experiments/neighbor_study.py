# Estudio de vecinos de los tetraedros solos de PolyllaEdge
# (reunion 17-11): vecinos por cara y por arista de cada poliedro de un
# solo tetraedro, clasificados en poliedro / tetraedro solo / borde.
import contextlib
import io
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from newMesh import EdgeTetrahedronMesh  # noqa: E402
from PolyllaEdge import PolyllaEdge  # noqa: E402

TETGEN = ROOT / "tetgen" / "build" / "tetgen.exe"
# Las mallas generadas van a data/experiments/ (ignorado por git)
OUT = ROOT / "data" / "experiments" / "neighbor_study"
OUT.mkdir(parents=True, exist_ok=True)
RESULTS = ROOT / "experiments" / "results" / "neighbor_study.json"

BOX = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], dtype=float)

# Parametros (los del notebook 3D-samplegeneration para cada tamano)
RANDOM = {1000: 0.1, 5000: 0.05, 10000: 0.05}             # tolerancia de borde
QUALITY = {1000: 0.00038, 5000: 0.0000695, 10000: 0.000033}  # volumen max (-a)
SEED = 1


def tetgen(args, path):
    subprocess.run([str(TETGEN), args, str(path)], check=True, capture_output=True)


def random_mesh(n, tol):
    rng = np.random.default_rng(SEED)
    pts = np.vstack([rng.random((n - 8, 3)), BOX])
    pts[pts <= tol] = 0.0
    pts[pts >= 1.0 - tol] = 1.0
    pts = np.unique(pts, axis=0)
    name = f"{n}random_tol{int(tol * 100)}_s{SEED}"
    with open(OUT / f"{name}.node", "w") as f:
        f.write(f"{len(pts)} 3 0 0\n")
        for i, p in enumerate(pts):
            f.write(f"{i} {p[0]:.17g} {p[1]:.17g} {p[2]:.17g}\n")
    tetgen("-fzennQ", OUT / f"{name}.node")
    return OUT / f"{name}.1"


def quality_mesh(n, vol):
    # Cubo unitario como PLC; TetGen con calidad por defecto (-q, radio-arista 2.0)
    name = f"{n}quality"  # sin puntos en el nombre: TetGen corta en el primer '.'
    with open(OUT / f"{name}.poly", "w") as f:
        f.write("8 3 0 0\n")
        for i, p in enumerate(BOX):
            f.write(f"{i} {p[0]:g} {p[1]:g} {p[2]:g}\n")
        quads = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
        f.write("6 0\n")
        for q in quads:
            f.write("1\n4 " + " ".join(map(str, q)) + "\n")
        f.write("0\n0\n")
    tetgen(f"-pqa{vol:.10f}fzennQ", OUT / f"{name}.poly")
    return OUT / f"{name}.1"


def study(base, label, params):
    with contextlib.redirect_stdout(io.StringIO()):
        mesh = EdgeTetrahedronMesh(f"{base}.node", f"{base}.face", f"{base}.ele", f"{base}.edge")
        edge = PolyllaEdge(mesh)
    size = {}
    poly_of = {}
    for k, p in enumerate(edge.polyhedral_mesh):
        size[k] = len(p.tetras)
        for t in p.tetras:
            poly_of[t] = k
    singles = [p.tetras[0] for p in edge.polyhedral_mesh if len(p.tetras) == 1]
    single_set = set(singles)

    def kind(t):
        if t == -1:
            return "borde"
        return "solo" if t in single_set else "poliedro"

    # --- Vecinos por cara ---
    face_kinds = Counter()
    combos = Counter()           # (poliedro, solo, borde) por tetraedro
    distinct_polys = Counter()   # cuantos poliedros distintos lo rodean por cara
    for t in singles:
        c = Counter()
        polys = set()
        for f in mesh.tetra_list[t].faces:
            fc = mesh.face_list[f]
            other = fc.n2 if fc.n1 == t else fc.n1
            k = kind(other)
            c[k] += 1
            if k == "poliedro":
                polys.add(poly_of[other])
        face_kinds.update(c)
        combos[(c["poliedro"], c["solo"], c["borde"])] += 1
        distinct_polys[len(polys)] += 1

    # --- Vecinos por arista (cada arista por separado, por rango de largo) ---
    # Por arista: los otros tetraedros de su anillo son todos solos / mixto /
    # todos en poliedros / ninguno (arista de un solo tetraedro)
    rank_class = {r: Counter() for r in range(1, 7)}
    tet_with_all_single_edge = 0
    for t in singles:
        es = sorted(mesh.tetra_list[t].edges, key=edge.edge_key, reverse=True)
        found = False
        for r, e in enumerate(es, start=1):
            others = [o for o in mesh.edge_list[e].tetrahedrons if o != t]
            if not others:
                cls = "sin otros"
            elif all(o in single_set for o in others):
                cls = "todos solos"
                found = True
            elif any(o in single_set for o in others):
                cls = "mixto"
            else:
                cls = "todos poliedros"
            rank_class[r][cls] += 1
        tet_with_all_single_edge += found

    # Aristas de la malla cuyo anillo completo son solo tetraedros solos (>= 2)
    cand_edges = [e for e in range(mesh.n_edges)
                  if len(mesh.edge_list[e].tetrahedrons) >= 2
                  and all(o in single_set for o in mesh.edge_list[e].tetrahedrons)]
    covered = {o for e in cand_edges for o in mesh.edge_list[e].tetrahedrons}
    closed_cands = sum(1 for e in cand_edges
                       if len(mesh.edge_list[e].faces) == len(mesh.edge_list[e].tetrahedrons))

    return dict(
        malla=label, params=params,
        puntos=mesh.n_nodes, tetraedros=mesh.n_tetrahedrons, caras=mesh.n_faces,
        aristas=mesh.n_edges, poliedros=len(edge.polyhedral_mesh), solos=len(singles),
        colgantes=edge.polyhedra_with_hanging,
        cara=dict(face_kinds),
        combos={f"{p}P {s}S {b}B": v for (p, s, b), v in sorted(combos.items(), key=lambda kv: -kv[1])},
        poliedros_distintos=dict(sorted(distinct_polys.items())),
        arista_por_rango={r: dict(c) for r, c in rank_class.items()},
        solos_con_arista_todos_solos=tet_with_all_single_edge,
        aristas_candidatas=len(cand_edges), aristas_candidatas_cerradas=closed_cands,
        solos_cubiertos_por_candidatas=len(covered),
    )


def pct(a, b):
    return f"{100 * a / b:5.1f}%" if b else "  -  "


def report(rs):
    print("\n## Tabla 1: mallas")
    print(f"{'malla':<24}{'puntos':>8}{'tets':>8}{'caras':>8}{'aristas':>8}{'polys':>8}{'solos':>7}{'%polys':>8}{'%tets':>8}")
    for r in rs:
        print(f"{r['malla']:<24}{r['puntos']:>8}{r['tetraedros']:>8}{r['caras']:>8}{r['aristas']:>8}"
              f"{r['poliedros']:>8}{r['solos']:>7}{pct(r['solos'], r['poliedros']):>8}{pct(r['solos'], r['tetraedros']):>8}")

    print("\n## Tabla 2: vecinos por cara (sobre las 4*solos caras)")
    print(f"{'malla':<24}{'poliedro':>10}{'solo':>10}{'borde':>10}")
    for r in rs:
        n = 4 * r['solos']
        c = r['cara']
        print(f"{r['malla']:<24}{pct(c.get('poliedro', 0), n):>10}{pct(c.get('solo', 0), n):>10}{pct(c.get('borde', 0), n):>10}")

    print("\n## Tabla 3: combinaciones por tetraedro solo (P=poliedro, S=solo, B=borde)")
    for r in rs:
        s = r['solos']
        txt = ", ".join(f"{k}: {pct(v, s).strip()}" for k, v in list(r['combos'].items())[:8])
        print(f"{r['malla']:<24}{txt}")

    print("\n## Tabla 4: poliedros distintos vecinos por cara")
    for r in rs:
        s = r['solos']
        txt = ", ".join(f"{k}: {pct(v, s).strip()}" for k, v in r['poliedros_distintos'].items())
        print(f"{r['malla']:<24}{txt}")

    print("\n## Tabla 5: vecinos por arista, por rango (1 = arista mas larga). % 'todos solos' / 'mixto' / 'todos poliedros' / 'sin otros'")
    for r in rs:
        s = r['solos']
        parts = []
        for rank in range(1, 7):
            c = r['arista_por_rango'][rank]
            parts.append(f"{rank}: {pct(c.get('todos solos', 0), s).strip()}/{pct(c.get('mixto', 0), s).strip()}/"
                         f"{pct(c.get('todos poliedros', 0), s).strip()}/{pct(c.get('sin otros', 0), s).strip()}")
        print(f"{r['malla']}")
        for p in parts:
            print(f"    rango {p}")

    print("\n## Tabla 6: posibles uniones por arista")
    print(f"{'malla':<24}{'solos c/ arista todo-solos':>28}{'aristas cand.':>15}{'cerradas':>10}{'solos cubiertos':>17}")
    for r in rs:
        s = r['solos']
        print(f"{r['malla']:<24}{r['solos_con_arista_todos_solos']:>8} ({pct(r['solos_con_arista_todos_solos'], s)}){'':>8}"
              f"{r['aristas_candidatas']:>15}{r['aristas_candidatas_cerradas']:>10}"
              f"{r['solos_cubiertos_por_candidatas']:>8} ({pct(r['solos_cubiertos_por_candidatas'], s)})")


if __name__ == "__main__":
    results = []
    for n, vol in QUALITY.items():
        results.append(study(quality_mesh(n, vol), f"{n}quality", {"q": 2.0, "a": vol}))
        print("listo", results[-1]["malla"], flush=True)
    for n, tol in RANDOM.items():
        results.append(study(random_mesh(n, tol), f"{n}random", {"tolerancia": tol, "semilla": SEED}))
        print("listo", results[-1]["malla"], flush=True)
    with open(RESULTS, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    report(results)
