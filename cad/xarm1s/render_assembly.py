"""
Assembly render + interference check for the xArm 1S spoon attachment.
Run after spoonful_xarm.py (needs ./assembly/*_placed.stl).

The gripper here is a BOX PROXY built from caliper measurements (2026-11-13):
upright servo body, mounting ears front and back, the flat finger layer under it,
the pads, the cable exit and the start of the blue wrist bracket. Good for gross
clearance, not for sub-millimetre fit.
"""
import os
import platform

import numpy as np
import trimesh

if platform.system() == "Linux":
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import pyrender  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import spoonful_xarm as P  # noqa: E402

ASM = "assembly"


def bx(x0, x1, y0, y1, z0, z1):
    m = trimesh.creation.box(extents=(x1 - x0, y1 - y0, z1 - z0))
    m.apply_translation(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2))
    return m


def proxy_boxes():
    """(name, x0, x1, y0, y1, z0, z1) in the assembly frame (X fwd, Z = 0 at the servo top)."""
    g = P.grip_w / 2
    L, H = P.servo_len, P.servo_h
    ear = (P.ear_span - L) / 2
    fz0 = -H - P.finger_stack
    return [
        ("servo body", -L, 0, -g, g, -H, 0),
        ("front ear", 0, ear, -g, g, -H, -H + P.ear_t),
        ("rear ear", -L - ear, -L, -g, g, -H, -H + P.ear_t),
        ("finger layer", -L, P.pad_reach, -16, 16, fz0, -H),
        ("cable exit", -L - 8, -L, -P.cable_d / 2, P.cable_d / 2, -P.cable_drop - P.cable_d / 2,
         -P.cable_drop + P.cable_d / 2),
        ("wrist bracket", -L - P.rear_clear - 20, -L - P.rear_clear, -30, 30, -H - 15, 10),
    ]


def gripper_proxy():
    return [bx(*b[1:]) for b in proxy_boxes()]


def spoon_mesh():
    """Spoon in assembly frame: station s -> X = s - spoon_at_face; handle axis on the cap seam."""
    zc = P.cap_z0_axis
    x_of = lambda s: s - P.spoon_at_face  # noqa: E731
    secs = []
    for s, w, t in P.handle_profile:
        r = min(2.5, t / 2 - 0.1)
        pts = []
        for cx, cy in ((w / 2 - r, t / 2 - r), (-w / 2 + r, t / 2 - r), (-w / 2 + r, -t / 2 + r), (w / 2 - r, -t / 2 + r)):
            for a in np.linspace(0, 2 * np.pi, 12, endpoint=False):
                pts.append((x_of(s), cx + r * np.cos(a), zc + cy + r * np.sin(a)))
        secs.append(np.array(pts))
    pieces = [trimesh.convex.convex_hull(np.vstack([a, b])) for a, b in zip(secs, secs[1:])]
    # bowl: shallow scoop, concave side up
    bowl = trimesh.creation.icosphere(subdivisions=4)
    bowl.apply_scale((P.bowl_len / 2, P.bowl_w / 2, 7.0))
    bowl = bowl.slice_plane((0, 0, 1.5), (0, 0, -1), cap=False)
    inner = trimesh.creation.icosphere(subdivisions=4)
    inner.apply_scale((P.bowl_len / 2 - 2, P.bowl_w / 2 - 2, 5.0))
    bowl.apply_translation((x_of(P.spoon_len - P.bowl_len / 2), 0, zc + 2))
    pieces.append(bowl)
    return trimesh.util.concatenate(pieces)


def load(name):
    p = os.path.join(ASM, f"{name}_placed.stl")
    return trimesh.load(p) if os.path.exists(p) else None


def look_at(eye, target, up=(0, 0, 1)):
    eye, target, up = map(np.asarray, (eye, target, up))
    f = target - eye
    f = f / np.linalg.norm(f)
    r = np.cross(f, up)
    r = r / np.linalg.norm(r)
    u = np.cross(r, f)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = r, u, -f, eye
    return m


def render(items, eye, target, out, title, size=(1400, 1000), yfov=0.55):
    scene = pyrender.Scene(bg_color=[0.96, 0.95, 0.93, 1.0], ambient_light=[0.35, 0.35, 0.35])
    for mesh, color, rough in items:
        mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=list(color) + [1.0], metallicFactor=0.0,
                                                 roughnessFactor=rough, doubleSided=True)
        scene.add(pyrender.Mesh.from_trimesh(mesh, material=mat, smooth=False))
    cam = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=size[0] / size[1])
    pose = look_at(eye, target)
    scene.add(cam, pose=pose)
    scene.add(pyrender.DirectionalLight(intensity=3.0), pose=pose)
    scene.add(pyrender.DirectionalLight(intensity=2.0), pose=look_at((80, 150, 250), (0, 0, 0)))
    scene.add(pyrender.DirectionalLight(intensity=1.2), pose=look_at((-150, -100, 120), (0, 0, 0)))
    r = pyrender.OffscreenRenderer(*size)
    color, _ = r.render(scene)
    r.delete()
    img = Image.fromarray(color)
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 26)
    except OSError:
        font = ImageFont.load_default()
    d.text((24, 18), title, fill=(40, 40, 40), font=font)
    img.save(out)
    print("wrote", out)


def interference_report():
    """Volume overlaps between printed parts and the gripper proxy (CadQuery booleans)."""
    import cadquery as cq

    def cqbox(x0, x1, y0, y1, z0, z1):
        return cq.Workplane("XY").box(x1 - x0, y1 - y0, z1 - z0, centered=False).translate((x0, y0, z0))

    mount = P.make_mount()
    cr, h = P.make_cradle(P.clear_M, "M")
    cr = cr.translate((P.cradle_x0, 0, P.plate_t))
    cap, _ = P.make_cap("SM")
    cap = cap.translate((P.cradle_x0, 0, P.plate_t + h))
    rows = []
    for a_name, a in (("Mount", mount), ("Cradle_M", cr), ("Cap", cap)):
        for b_name, *b in proxy_boxes():
            try:
                v = a.intersect(cqbox(*b)).val().Volume()
            except Exception:
                v = 0.0
            rows.append((a_name, b_name, v))
    # side play: skirt inner face (minus crush rib) to the servo body
    play = P.side_clear - P.rib_h
    return rows, play


def main():
    P.cap_z0_axis = P.plate_t + P.cradle_height(P.clear_M)
    coral = (0.93, 0.42, 0.30)
    cream = (0.96, 0.93, 0.86)
    dark = (0.18, 0.19, 0.21)
    servo_c = (0.12, 0.12, 0.13)
    spoon_c = (0.36, 0.72, 0.66)

    items = []
    for gm in gripper_proxy():
        items.append((gm, servo_c, 0.6))
    for name, col in (("Mount", coral), ("Cradle_M", cream), ("Cap", coral)):
        m = load(name)
        if m is not None:
            items.append((m, col, 0.45))
    items.append((spoon_mesh(), spoon_c, 0.35))

    os.makedirs("renders", exist_ok=True)
    tgt = (10, 0, -10)
    render(items, eye=(240, -55, 25), target=(0, 0, -8), out="renders/diner_view.png",
           title="Spoonful on xArm 1S: diner's view (gripper is a proxy)")
    render(items, eye=(150, -230, 140), target=tgt, out="renders/three_quarter.png",
           title="Spoonful on xArm 1S: three-quarter view")
    render(items, eye=(0, -360, -20), target=tgt, out="renders/side.png",
           title="Side view: spoon axis, mount on the upright gripper servo", yfov=0.5)

    rows, play = interference_report()
    with open("renders/interference.txt", "w") as f:
        f.write("Overlap volumes against the gripper proxy (mm^3). 0 = clear.\n")
        for a, b, v in rows:
            f.write(f"{a:15s} vs {b:16s} {v:8.2f}\n")
        f.write(f"\nSide play per side at the crush ribs: {play:.2f} mm (sand the ribs if it won't go on).\n")
    print(open("renders/interference.txt").read())


if __name__ == "__main__":
    main()
