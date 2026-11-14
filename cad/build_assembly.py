"""One-file views of the Spoonful wrist on the whole SO-101 arm.

    python build_assembly.py            -> Spoonful_Arm_Assembly.glb   (whole arm, spoon fitted, posed)
                                           Spoonful_Wrist_Exploded.glb (wrist only, parts pulled apart along
                                                                        the directions they go together)
                                           assembly_*.png               (renders of both)

The arm comes from TheRobotStudio/SO-ARM100 Simulation/SO101/so101_new_calib.urdf and its meshes (ref/assets/):
every link mesh is placed by forward kinematics, exactly as the robot model defines it. Removed: servo 5 (wrist roll),
servo 6 + wrist-roll follower (gripper link) and the moving jaw. Added, in the Wrist_Roll_Pitch part frame, which the
URDF places at wrist_link * T(0, -0.028, 0.0181) * RPY(-pi/2, -pi/2, 0): the block, a cradle, pins, screws, inserts,
magnets and a PLACEHOLDER spoon (13 x 5.5 mm handle, 90 mm long; not a real spoon).
GLB is in millimetres, Z up. Open in macOS Preview / Quick Look, Blender, or https://gltf-viewer.donmccurdy.com.
"""
import argparse
import math
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from trimesh.transformations import euler_matrix, translation_matrix, rotation_matrix

import interface as itf

URDF = "ref/so101_new_calib.urdf"
ASSETS = "ref/assets"
REMOVE = {("wrist_link", "sts3215_03a_no_horn_v1.stl")}            # servo 5, replaced by the block
REMOVE_LINKS = {"gripper_link", "moving_jaw_so101_v1_link"}        # gripper servo + follower + jaw
T_LINK_PART = translation_matrix([0, -0.028, 0.0181]) @ euler_matrix(-1.5708, -1.5708, 0, axes="sxyz")

COLORS = {"3d_printed": [225, 225, 220, 255], "sts3215": [45, 45, 50, 255], "block": [240, 140, 40, 255],
          "cradle": [30, 160, 150, 255], "steel": [190, 195, 200, 255], "screw": [25, 25, 25, 255],
          "brass": [200, 160, 60, 255], "magnet": [120, 125, 135, 255], "spoon": [90, 150, 230, 255],
          "wrist_part": [205, 215, 230, 255]}

# Poses (radians). "rest": the URDF zero (wrist level, spoon pointing straight out, bowl up).
POSES = {"rest": {}, "present": {"shoulder_lift": -0.35, "elbow_flex": 0.55, "wrist_flex": -0.2}}


def tf(o):
    xyz = [float(v) for v in o.get("xyz", "0 0 0").split()]
    rpy = [float(v) for v in o.get("rpy", "0 0 0").split()]
    return translation_matrix(xyz) @ euler_matrix(*rpy, axes="sxyz")


def link_transforms(q):
    """World transform of every link for joint angles q (FK over the URDF tree)."""
    root = ET.parse(URDF).getroot()
    joints = root.findall("joint")
    child = {j.find("child").get("link"): j for j in joints}
    T = {}

    def get(link):
        if link in T:
            return T[link]
        if link not in child:
            T[link] = np.eye(4)
            return T[link]
        j = child[link]
        M = get(j.find("parent").get("link")) @ tf(j.find("origin"))
        if j.get("type") == "revolute":
            ax = [float(v) for v in j.find("axis").get("xyz").split()]
            M = M @ rotation_matrix(q.get(j.get("name"), 0.0), ax)
        T[link] = M
        return M
    for l in root.findall("link"):
        get(l.get("name"))
    return root, T


def colored(mesh, key):
    m = mesh.copy()
    m.visual = trimesh.visual.ColorVisuals(m, face_colors=np.tile(COLORS[key], (len(m.faces), 1)))
    return m


def arm_meshes(q):
    """(name, mesh in mm) for every kept URDF visual, plus the part->world transform (mm)."""
    root, T = link_transforms(q)
    out = []
    for l in root.findall("link"):
        name = l.get("name")
        if name in REMOVE_LINKS:
            continue
        for v in l.findall("visual"):
            mesh_f = v.find("geometry/mesh").get("filename").split("/")[-1]
            if (name, mesh_f) in REMOVE:
                continue
            mat = v.find("material").get("name") if v.find("material") is not None else "3d_printed"
            m = trimesh.load(f"{ASSETS}/{mesh_f}")
            m.apply_transform(T[name] @ tf(v.find("origin")))
            m.apply_scale(1000.0)
            out.append((f"{name}:{mesh_f[:-4]}", colored(m, "sts3215" if "sts3215" in mesh_f else "3d_printed")))
    Twp = T["wrist_link"] @ T_LINK_PART
    S = np.diag([1000.0, 1000.0, 1000.0, 1.0])
    part_to_world_mm = S @ Twp @ np.diag([0.001, 0.001, 0.001, 1.0])
    return out, part_to_world_mm


# ---------------- new parts + hardware, in the Wrist_Roll_Pitch part frame (mm) ----------------
def zcyl(x, y, z0, z1, d, sections=24):
    c = trimesh.creation.cylinder(radius=d / 2, height=z1 - z0, sections=sections)
    c.apply_translation([x, y, (z0 + z1) / 2])
    return c


def placeholder_spoon(cx, cy, z_top, handle_w=13.0, handle_t=5.5, total=90.0, bowl_len=32.0, bowl_w=24.0, bowl_d=8.0):
    """Handle along -Z from the end stop, bowl opening toward -Y (up when presenting)."""
    h_len = total - bowl_len
    handle = trimesh.creation.box(extents=[handle_w, handle_t, h_len])
    handle.apply_translation([cx, cy, z_top - h_len / 2])
    outer = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    outer.apply_scale([bowl_w / 2, bowl_d, bowl_len / 2])
    inner = outer.copy(); inner.apply_scale([0.86, 0.9, 0.9]); inner.apply_translation([0, -2.2, 0])
    half = trimesh.creation.box(extents=[bowl_w * 2, bowl_d * 2, bowl_len * 2]); half.apply_translation([0, -bowl_d, 0])
    bowl = outer.difference(inner, engine="manifold").difference(half, engine="manifold")
    bowl.apply_translation([cx, cy - handle_t / 2 + bowl_d * 0.95, z_top - h_len - bowl_len / 2 + 2])
    return trimesh.util.concatenate([handle, bowl])


def wrist_parts(cradle_name="Cradle_M"):
    """Every new part and fastener in the part frame. Returns {name: (mesh, color, explode_vector_mm)}."""
    import cradle as cr
    import spoon_block as sb
    parts = {}
    parts["Spoon_Block_SO101"] = (trimesh.load("spoon_block_inplace.stl"), "block", (55, 0, 0))
    for i, (x, y) in enumerate(sb.m2_pts):
        parts[f"M2x8 screw {i + 1}"] = (trimesh.util.concatenate([zcyl(x, y, -34.4, -26.4, 2.0), zcyl(x, y, -34.4, -33.0, 3.8)]),
                                        "screw", (0, 0, -30))
    for i, (x, y) in enumerate(itf.SCREWS):
        parts[f"M3 insert {i + 1}"] = (zcyl(x, y, itf.MATE_Z, itf.MATE_Z + 5.7, 4.0), "brass", (55, 0, -14))
    cs, info = cr.fitted_cradle(*cr.VARIANTS[cradle_name])
    import cadquery as cq  # noqa: F401
    cradle_mesh = trimesh.load(f"{cradle_name}_inplace.stl")
    parts[cradle_name] = (cradle_mesh, "cradle", (0, 0, -45))
    for i, (x, y) in enumerate(itf.PINS):
        parts[f"3x8 dowel pin {i + 1}"] = (zcyl(x, y, itf.MATE_Z - itf.PIN_PLATE_DEPTH, itf.MATE_Z - itf.PIN_PLATE_DEPTH + itf.PIN_LEN, 3.0),
                                           "steel", (0, 0, -25))
    for i, (x, y) in enumerate(itf.SCREWS):
        head = trimesh.creation.cone(radius=3.0, height=1.7, sections=24)
        head.apply_transform(rotation_matrix(math.pi, [1, 0, 0])); head.apply_translation([x, y, itf.END_STOP_Z + 1.7])
        parts[f"M3x10 flat-head screw {i + 1}"] = (trimesh.util.concatenate([zcyl(x, y, itf.END_STOP_Z, itf.END_STOP_Z + 10, 3.0), head]),
                                                   "screw", (0, 0, -95))
    w, t = cr.VARIANTS[cradle_name]
    in_t = t + 0.2
    cy = ((in_t / 2 + cr.LIP_T) - (in_t / 2 + cr.FLOOR_T)) / 2 + itf.CY
    face_y = cy + in_t / 2
    for i, z in enumerate(info["magnets_z"]):
        mag = trimesh.creation.cylinder(radius=5.0, height=2.0, sections=32)
        mag.apply_transform(rotation_matrix(math.pi / 2, [1, 0, 0])); mag.apply_translation([itf.CX, face_y + 1.0, z])
        parts[f"10x2 magnet {i + 1}"] = (mag, "magnet", (0, 18, -45))
    parts["PLACEHOLDER spoon (13 x 5.5 handle, 90 mm)"] = (placeholder_spoon(itf.CX, cy, itf.END_STOP_Z, w, t), "spoon", (0, -45, -60))
    return parts


def render(scene, path, eye, target, up=(0, 0, 1), size=(1400, 1050), yfov=0.6):
    import pyrender   # macOS: native offscreen context (do not set PYOPENGL_PLATFORM)
    from PIL import Image
    s = pyrender.Scene(bg_color=[245, 245, 248, 255], ambient_light=[0.35, 0.35, 0.35])
    for g in scene.geometry.values():
        s.add(pyrender.Mesh.from_trimesh(g, smooth=False))
    eye, target, up = map(lambda a: np.asarray(a, float), (eye, target, up))
    f = target - eye; f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    pose = np.eye(4); pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = r, u, -f, eye
    s.add(pyrender.PerspectiveCamera(yfov=yfov), pose=pose)
    s.add(pyrender.DirectionalLight(intensity=3.0), pose=pose)
    s.add(pyrender.DirectionalLight(intensity=1.5), pose=rotation_matrix(-0.8, [1, 0, 0]))
    rnd = pyrender.OffscreenRenderer(*size)
    img, _ = rnd.render(s)
    rnd.delete()
    Image.fromarray(img).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", default="rest", choices=list(POSES))
    ap.add_argument("--cradle", default="Cradle_M")
    a = ap.parse_args()

    # 1) whole arm, assembled
    arm, P2W = arm_meshes(POSES[a.pose])
    parts = wrist_parts(a.cradle)
    sc = trimesh.Scene()
    for n, m in arm:
        sc.add_geometry(m, node_name=n, geom_name=n)
    for n, (m, c, _) in parts.items():
        mm = colored(m, c); mm.apply_transform(P2W)
        sc.add_geometry(mm, node_name=n, geom_name=n)
    sc.export("Spoonful_Arm_Assembly.glb")
    b = sc.bounds
    ctr = b.mean(0)
    wrist = (P2W @ [itf.CX, 0, -40, 1])[:3]
    render(sc, "assembly_arm.png", eye=ctr + [260, -420, 260], target=ctr)
    render(sc, "assembly_wrist_closeup.png", eye=wrist + [90, -150, 70], target=wrist)

    # 2) wrist only, exploded, in the part frame
    ex = trimesh.Scene()
    wp = colored(trimesh.load(f"{ASSETS}/wrist_roll_pitch_so101_v2.stl").apply_scale(1000.0), "wrist_part")
    ex.add_geometry(wp, node_name="Wrist_Roll_Pitch (existing arm part)", geom_name="wrist")
    for n, (m, c, v) in parts.items():
        mm = colored(m, c); mm.apply_translation(v)
        ex.add_geometry(mm, node_name=n, geom_name=n)
    ex.export("Spoonful_Wrist_Exploded.glb")
    render(ex, "assembly_wrist_exploded.png", eye=[230, -330, 40], target=[15, -10, -85])
    print("wrote Spoonful_Arm_Assembly.glb, Spoonful_Wrist_Exploded.glb, assembly_*.png")
    print(f"arm bbox (mm): {np.round(b[0])} .. {np.round(b[1])}; wrist interface centre in world (mm): {np.round(wrist, 1)}")


if __name__ == "__main__":
    main()
