"""Fit check for every part, in the Wrist_Roll_Pitch frame, against the imported pocket part (ref/).

  python fit_check.py          -> table on stdout + fit_check_report.md + screenshots in fit_check/

Checks: watertight; interference volume with the pocket part (must be 0); minimum gap; block slides out along +X;
each cradle drops out along -Z (it is fitted from below after the block is in); cradle vs block overlap (0);
clamp jaw vs fixed jaw (0). Screenshots: sections at key heights (part blue, block red, cradle orange).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402

part = trimesh.load("ref/Wrist_Roll_Pitch_SO101.stl")
os.makedirs("fit_check", exist_ok=True)
PARTS = {"Spoon_Block_SO101": "spoon_block_inplace.stl", "Cradle_S": "Cradle_S_inplace.stl", "Cradle_M": "Cradle_M_inplace.stl",
         "Cradle_L": "Cradle_L_inplace.stl", "Cradle_Clamp": "Cradle_Clamp_inplace.stl", "Cradle_Clamp_Jaw": "Cradle_Clamp_Jaw_inplace.stl"}
M = {k: trimesh.load(v) for k, v in PARTS.items()}


def inter(a, b):
    return round(float(a.intersection(b, engine="manifold").volume), 3)


def sweep(m, d, steps=(0.5, 1, 2, 4, 8, 16, 32, 60)):
    return max(inter(m.copy().apply_translation(np.asarray(d, float) * s), part) for s in steps)


def min_gap(m):
    pts = m.sample(5000)
    pts = pts[(pts[:, 2] > -40)]
    if not len(pts):
        return np.nan
    return round(float(-trimesh.proximity.signed_distance(part, pts).max()), 3)


rows = []
blk = M["Spoon_Block_SO101"]
for name, m in M.items():
    r = {"part": name, "watertight": m.is_watertight, "volume_cm3": round(m.volume / 1000, 1),
         "interference_mm3": inter(m, part), "min_gap_mm": min_gap(m)}
    if name == "Spoon_Block_SO101":
        r["assembly path"] = f"slide out +X: max {sweep(m, (1, 0, 0))} mm3"
    else:
        r["assembly path"] = f"drop out -Z: max {sweep(m, (0, 0, -1))} mm3"
        r["vs block_mm3"] = inter(m, blk)
    rows.append(r)
rows.append({"part": "Clamp jaw vs fixed jaw", "interference_mm3": inter(M["Cradle_Clamp_Jaw"], M["Cradle_Clamp"])})

cols = ["part", "watertight", "volume_cm3", "interference_mm3", "min_gap_mm", "vs block_mm3", "assembly path"]
lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
for r in rows:
    lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
table = "\n".join(lines)
print(table)
open("fit_check_report.md", "w").write("# Fit check (Wrist_Roll_Pitch frame)\n\n" + table + "\n")


def draw(ax, m, z, color):
    s = m.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    if s is None:
        return
    p, _ = s.to_2D(to_2D=np.eye(4))
    for g in p.polygons_full:
        xs, ys = g.exterior.xy
        ax.fill(xs, ys, alpha=0.35, color=color); ax.plot(xs, ys, color=color, lw=0.8)
        for h in g.interiors:
            hx, hy = h.xy; ax.plot(hx, hy, color=color, lw=0.6)


for tag, others in [("block", []), ("block_cradle_M", ["Cradle_M"]), ("block_cradle_L", ["Cradle_L"]),
                    ("block_clamp", ["Cradle_Clamp", "Cradle_Clamp_Jaw"])]:
    zs = [-2, -15, -28, -30.8, -32.5, -34.4, -36, -45]
    fig, axs = plt.subplots(2, 4, figsize=(19, 8.5))
    ov = [inter(blk, part)] + [inter(M[o], part) for o in others]
    for ax, z in zip(axs.flat, zs):
        draw(ax, part, z, "#2a78d6"); draw(ax, blk, z, "#d62728")
        for o in others:
            draw(ax, M[o], z, "#ff9f1c")
        ax.set_title(f"Z = {z}", fontsize=10); ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_xlim(-30, 38); ax.set_ylim(-20, 18)
    fig.suptitle(f"Sections: Wrist_Roll_Pitch (blue), block (red){', ' + ', '.join(others) + ' (orange)' if others else ''}. "
                 f"Interference with the pocket part (mm3): {ov}", fontsize=11)
    fig.tight_layout(); fig.savefig(f"fit_check/sections_{tag}.png", dpi=110); plt.close(fig)

trimesh.util.concatenate([part, blk]).export("fit_check/context_block.stl")
trimesh.util.concatenate([part, blk, M["Cradle_M"]]).export("fit_check/context_block_cradleM.stl")
trimesh.util.concatenate([part, blk, M["Cradle_Clamp"], M["Cradle_Clamp_Jaw"]]).export("fit_check/context_block_clamp.stl")
print("screenshots in fit_check/")
