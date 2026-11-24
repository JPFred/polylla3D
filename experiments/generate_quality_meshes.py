# Generates quality meshes of the unit cube with TetGen (-pq: radius-edge
# ratio 2.0, -a: maximum tetrahedron volume) and writes them to data/input/.

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TETGEN = ROOT / "tetgen" / "build" / "tetgen.exe"
OUT = ROOT / "data" / "input"

# name: maximum tetrahedron volume (-a). The name has no dots: TetGen cuts it at the first '.'
QUALITY = {"43quality": 0.02}

CUBE_VERTICES = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
                 (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
CUBE_FACES = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]


def write_cube_poly(path):
    with open(path, "w") as f:
        f.write("8 3 0 0\n")
        for i, (x, y, z) in enumerate(CUBE_VERTICES):
            f.write(f"{i} {x} {y} {z}\n")
        f.write("6 0\n")
        for quad in CUBE_FACES:
            f.write("1\n4 " + " ".join(map(str, quad)) + "\n")
        f.write("0\n0\n")


def quality_mesh(name, vol, out=OUT):
    out.mkdir(parents=True, exist_ok=True)
    poly = out / f"{name}.poly"
    write_cube_poly(poly)
    subprocess.run([str(TETGEN), f"-pqa{vol:.10f}fzennQ", str(poly)], check=True, capture_output=True)
    return out / f"{name}.1"


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, vol in QUALITY.items():
        base = quality_mesh(name, vol)
        with open(f"{base}.node") as f:
            n_points = int(f.readline().split()[0])
        print(f"{base.name}: {n_points} points (a = {vol})")
