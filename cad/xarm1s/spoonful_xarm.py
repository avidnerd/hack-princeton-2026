"""
Spoonful: modular spoon attachment for the Hiwonder xArm 1S gripper.

    python spoonful_xarm.py            # all STLs into ./stl
    python spoonful_xarm.py --render   # also the assembly + diner-view renders

Modules
  A  Mount (saddle over the upright gripper servo; rear open for the cable)
  B  Cradle_M, Cradle_L (spoon handle trough)
  C  Cap (6 x 2 discs; the 17.8 mm discs are far too strong for a breakaway and don't fit)
  +  Stop_Housing_24/28, Nameplate, Cable_Clip (x4), Bowl_Ring

Assembly frame (mm), used by every module and the render:
  X  forward along the gripper axis, X = 0 at the FRONT face of the servo block
  Y  across the gripper (jaws close along Y)
  Z  up, Z = 0 at the TOP of the servo block
Each part is exported in its own print orientation (see PRINT notes).

Standard interface (the only thing modules share):
  horizontal 30 x 20 pad, cradle underside <-> mount top
  2 x  3.0 pins, 14 apart along the spoon axis (on the cradle)
       3.3 holes on the mount
  2 x  M3 clearance holes, 10 apart across the axis, countersunk from the
       trough side of the cradle; M3 heat-set inserts in the mount
"""
import math
import os
import sys

import cadquery as cq

# ============================================================
# TOP-LEVEL PARAMETERS (the ones you are most likely to change)
# ============================================================
grip_w = 20.2            # gripper servo body width across the arm (Y), calipers 2026-11-13

# ============================================================
# GRIPPER (calipers, 2026-11-13, photo cad/IMG_7144.heic)
# The gripper servo stands upright on a flat plastic gripper plate; its shaft points down into a flat finger
# linkage layer under it. Mounting ears stick out front and back along X. The cable leaves the rear (wrist) face.
# ============================================================
servo_len = 40.0         # servo body along X (54.5 including the mounting ears)
ear_span = 54.5          # ear tip to ear tip along X
servo_h = 39.6           # servo body height, top to the finger layer
finger_stack = 8.6       # finger linkage layer under the servo
pad_reach = 56.9         # servo front face to the finger pads, fingers fully open
ear_t = 3.4              # ear thickness (calipers); ears sit near the bottom of the body. Servo top is flat.
cable_drop, cable_d = 6.9, 3.9    # cable exit centre below the servo top (rear face), bundle diameter
rear_clear = 22.2        # servo rear face to the blue wrist bracket

# ============================================================
# SPOON
# ============================================================
spoon_len = 159.0
# (distance from handle end, width, thickness). 15..85 measured with calipers 2026-11-13 (silicone, light jaws);
# 0, 110 and 125 are photo estimates and only matter for the render.
handle_profile = [
    (0.0, 15.0, 10.0), (15.0, 15.2, 10.4), (20.0, 15.6, 10.8), (35.0, 16.8, 9.0), (50.0, 16.8, 8.6),
    (65.0, 15.2, 8.9), (75.0, 13.4, 8.6), (85.0, 12.3, 8.8), (110.0, 10.6, 9.5), (125.0, 10.0, 9.0),
]
bowl_len, bowl_w = 34.0, 21.0
cradle_zone = (20.0, 75.0)          # handle section held by the cradle
bowl_tip_ahead = 100.0              # bowl tip ahead of the servo block front face

# ============================================================
# MODULE A: MOUNT
# ============================================================
side_clear = 0.5                    # per side, servo body to skirt (crush ribs take up the rest)
skirt_t = 3.0
skirt_h = 14.0                      # skirt depth below the servo top (ears are ~35 down, well clear)
plate_t = 5.0                       # top plate (brief said 4; +1 so the M3 inserts have meat)
rear_gap = 1.0                      # plate stops this short of the rear face: rear is open for the cable
plate_len = servo_len - rear_gap    # along X
fascia_t = 4.0                      # front wall carrying SPOONFUL
corner_r = 6.0                      # plan-view corners
top_chamfer = 1.0                   # top edges (bed side when printing)

tie_w, tie_t = 5.0, 1.6             # zip tie, with play
tie_x = (-33.0, -26.0)              # two ties: over the plate, down the skirt channels, under the gripper plate
tie_groove_d = 1.5                  # depth of the groove across the top
rib_h, rib_w = 0.35, 2.0            # crush ribs on the skirt inner faces
rib_x = (-30.0, -15.0, -2.0)
tape_recess = 1.0
tape_inset = 3.0

insert_d, pin_hole_d = 4.0, 3.3     # M3 heat-set insert hole, locating pin hole

# ============================================================
# MODULE B / C: CRADLE + CAP
# ============================================================
cradle_len = cradle_zone[1] - cradle_zone[0]   # 55
cradle_w = 48.0                     # constant outer width: magnets live in the walls
floor_t = 4.0
roof_t = 3.0
edge_chamfer = 1.5
clear_M = 0.6                       # total, on width and thickness
clear_L = clear_M + 1.5

pin_d, pin_len = 3.0, 4.5           # brief said 6; plate is 5 thick over a 1 mm tape recess
pin_pitch = 14.0
screw_pitch = 10.0
m3_clear, csk_d = 3.4, 6.6

small_pocket_d, small_pocket_h = 5.9 + 0.3, 2.2     # small discs measured 5.9
mag_x = (12.0, 42.0)                # from the handle end of the cradle
mag_y = 16.4                        # wall pockets, +-Y

# Where the cradle sits on the mount (derived)
spoon_at_face = spoon_len - bowl_tip_ahead     # spoon station at X = 0
cradle_x0 = cradle_zone[0] - spoon_at_face     # cradle handle-end X in assembly
iface_local_x = cradle_len / 2                 # interface centre, cradle-local X
iface_x = cradle_x0 + iface_local_x            # interface centre, assembly X

# ============================================================
# COSMETICS
# ============================================================
stop_box = (45.0, 45.0, 40.0)
stop_wall = 2.5
nameplate = (120.0, 30.0, 3.0)
nameplate_hole_x = 50.0
slot_len = 10.0
ring_id, ring_h, ring_wall = 120.0, 8.0, 3.0

emboss = 0.6
eps = 0.01
OUT = "stl"


# ============================================================
# HELPERS
# ============================================================
def box(x0, x1, y0, y1, z0, z1):
    return cq.Workplane("XY").box(x1 - x0, y1 - y0, z1 - z0, centered=False).translate((x0, y0, z0))


def text_on(plane_origin, x_dir, normal, txt, size, depth):
    wp = cq.Workplane(cq.Plane(origin=plane_origin, xDir=x_dir, normal=normal))
    return wp.text(txt, size, depth, combine=False, halign="center", valign="center", font="Arial",
                   kind="bold")


def handle_wt(s):
    """Interpolated (width, thickness) of the handle at station s."""
    p = handle_profile
    for (s0, w0, t0), (s1, w1, t1) in zip(p, p[1:]):
        if s0 <= s <= s1:
            f = (s - s0) / (s1 - s0)
            return w0 + f * (w1 - w0), t0 + f * (t1 - t0)
    return p[-1][1], p[-1][2]


def slot2d(wp, length, width):
    return wp.slot2D(length, width)


# ============================================================
# MODULE A: MOUNT (assembly frame)
# ============================================================
def make_mount():
    """Saddle over the upright gripper servo: top plate + two side skirts + front fascia. Rear open (cable)."""
    half_in = grip_w / 2 + side_clear                  # 10.6
    half_out = half_in + skirt_t                       # 13.6
    x_rear = -plate_len

    body = box(x_rear, fascia_t, -half_out, half_out, -skirt_h, plate_t)
    body = body.edges("|Z").fillet(corner_r)
    body = body.faces(">Z").edges().chamfer(top_chamfer)
    # servo channel, open at the rear and the bottom
    body = body.cut(box(x_rear - 1, 0, -half_in, half_in, -skirt_h - 1, 0))

    # crush ribs on the skirts: sand them if it's tight
    for x in rib_x:
        for s in (-1, 1):
            y0 = s * half_in
            y1 = y0 - s * rib_h
            body = body.union(box(x - rib_w / 2, x + rib_w / 2, min(y0, y1), max(y0, y1), -skirt_h, 0))

    # tape / hot-glue recess on the underside of the plate
    body = body.cut(box(x_rear + tape_inset, -tape_inset, -half_in + tape_inset, half_in - tape_inset,
                        -1, tape_recess))

    # zip ties: groove across the top, slot through the plate, channel down the inside of each skirt
    for x in tie_x:
        x0, x1 = x - tie_w / 2, x + tie_w / 2
        body = body.cut(box(x0, x1, -half_out - 1, half_out + 1, plate_t - tie_groove_d, plate_t + 1))
        for s in (-1, 1):
            ya, yb = s * (half_in - 0.5), s * (half_in + tie_t)
            body = body.cut(box(x0, x1, min(ya, yb), max(ya, yb), -skirt_h - 1, plate_t + 1))

    # standard interface: inserts + pin holes
    for s in (-1, 1):
        body = body.cut(cq.Workplane("XY").center(iface_x, s * screw_pitch / 2)
                        .circle(insert_d / 2).extrude(plate_t + 2).translate((0, 0, -1)))
        body = body.cut(cq.Workplane("XY").center(iface_x + s * pin_pitch / 2, 0)
                        .circle(pin_hole_d / 2).extrude(plate_t + 2).translate((0, 0, -1)))

    # SPOONFUL on the fascia (faces the diner)
    body = body.union(text_on((fascia_t - eps, 0, (plate_t - skirt_h) / 2), (0, 1, 0), (1, 0, 0),
                              "SPOONFUL", 3.6, emboss + eps))
    return body


# ============================================================
# MODULE B: CRADLE (local: X 0..len handle end -> bowl end, Z 0 = underside)
# ============================================================
def cavity_outline(clear, n=22):
    pts_top, pts_bot = [], []
    for i in range(n + 1):
        x = cradle_len * i / n
        w, _ = handle_wt(cradle_zone[0] + x)
        pts_top.append((x, (w + clear) / 2))
        pts_bot.append((x, -(w + clear) / 2))
    pts = [(-1.0, pts_top[0][1])] + pts_top + [(cradle_len + 1.0, pts_top[-1][1])]
    pts += [(cradle_len + 1.0, pts_bot[-1][1])] + pts_bot[::-1] + [(-1.0, pts_bot[0][1])]
    return pts


def half_depth(x, clear):
    """Depth of each half-trough at cradle-local x: the handle is split at its mid-plane, the mating face."""
    return (handle_wt(cradle_zone[0] + x)[1] + clear) / 2


def max_half_depth(clear, n=110):
    return max(half_depth(cradle_len * i / n, clear) for i in range(n + 1))


def cradle_height(clear):
    return floor_t + max_half_depth(clear)


def cavity(clear, z_face, direction, n=22):
    """Trough following the handle's width (in plan) and thickness (in side view). direction -1: into the
    cradle below the mating face at z_face; +1: into the cap above it."""
    plan = (cq.Workplane("XY", origin=(0, 0, z_face - 20)).polyline(cavity_outline(clear)).close().extrude(40))
    side = [(-1.0, z_face - direction)]
    for i in range(n + 1):
        x = cradle_len * i / n
        side.append((x, z_face + direction * half_depth(x, clear)))
    side[0] = (-1.0, z_face + direction * half_depth(0, clear))
    side.append((cradle_len + 1.0, z_face + direction * half_depth(cradle_len, clear)))
    side += [(cradle_len + 1.0, z_face - direction), (-1.0, z_face - direction)]
    prof = cq.Workplane("XZ").polyline(side).close().extrude(cradle_w, both=True)
    return plan.intersect(prof)


def make_cradle(clear, label):
    h = cradle_height(clear)
    body = box(0, cradle_len, -cradle_w / 2, cradle_w / 2, 0, h).edges().chamfer(edge_chamfer)
    body = body.cut(cavity(clear, h, -1))
    # magnet pockets in the mating face (all four, so both caps fit)
    for x in mag_x:
        for s in (-1, 1):
            body = body.cut(cq.Workplane("XY", origin=(0, 0, h - small_pocket_h)).center(x, s * mag_y)
                            .circle(small_pocket_d / 2).extrude(small_pocket_h + 1))
    # interface: countersunk M3 through the floor, pins below
    xc = iface_local_x
    floor_z = h - half_depth(xc, clear)          # trough floor at the screws
    for s in (-1, 1):
        body = body.cut(cq.Workplane("XY", origin=(0, 0, -1)).center(xc, s * screw_pitch / 2)
                        .circle(m3_clear / 2).extrude(floor_z + 2))
        csk_h = (csk_d - m3_clear) / 2
        body = body.cut(cq.Workplane("XY", origin=(0, 0, floor_z - csk_h)).center(xc, s * screw_pitch / 2)
                        .circle(m3_clear / 2).workplane(offset=csk_h + eps).circle(csk_d / 2 + eps).loft())
        body = body.union(cq.Workplane("XY", origin=(0, 0, -pin_len)).center(xc + s * pin_pitch / 2, 0)
                          .circle(pin_d / 2).extrude(pin_len + eps))
    # size label on the +Y side face
    body = body.cut(text_on((cradle_len / 2, cradle_w / 2 + eps, h / 2), (-1, 0, 0), (0, 1, 0),
                            label, min(5.0, h - 3.5), -0.5))
    return body, h


# ============================================================
# MODULE C: CAP (local: Z 0 = mating face, roof on top)
# ============================================================
def make_cap(variant):
    h = max_half_depth(clear_M) + roof_t
    body = box(0, cradle_len, -cradle_w / 2, cradle_w / 2, 0, h).edges().chamfer(edge_chamfer)
    body = body.cut(cavity(clear_M, 0.0, 1))
    spots = [(x, sv * mag_y) for x in mag_x for sv in (-1, 1)]
    d, dh = small_pocket_d, small_pocket_h
    for x, y in spots:
        body = body.cut(cq.Workplane("XY", origin=(0, 0, -1)).center(x, y).circle(d / 2).extrude(dh + 1))
    # fingernail notch at the bowl end, +Y side, on the mating edge
    body = body.cut(box(cradle_len - 4, cradle_len + 1, mag_y - 5, mag_y + 5, -1, 2.0))
    # N dot over the first magnet, label on top
    body = body.union(cq.Workplane("XY", origin=(0, 0, h - eps)).center(mag_x[0], mag_y)
                      .circle(1.8).extrude(emboss + eps))
    body = body.union(text_on((cradle_len / 2 + 6, 0, h - eps), (1, 0, 0), (0, 0, 1),
                              "v2", 6.0, emboss + eps))
    return body, h


# ============================================================
# COSMETICS
# ============================================================
def make_stop(hole_d):
    bx, by, bz = stop_box
    b = box(-bx / 2, bx / 2, -by / 2, by / 2, 0, bz).edges("|Z").fillet(4).faces(">Z").edges().chamfer(1.0)
    b = b.cut(box(-bx / 2 + stop_wall, bx / 2 - stop_wall, -by / 2 + stop_wall, by / 2 - stop_wall,
                  -1, bz - stop_wall))
    b = b.cut(cq.Workplane("XY").circle(hole_d / 2).extrude(bz + 1))
    # inner tape flange at the base
    # 5 mm flange, 45-degree inner slope so it prints without support when flipped
    ix, iy, fw = bx - 2 * stop_wall, by - 2 * stop_wall, 5.0
    fl = box(-ix / 2, ix / 2, -iy / 2, iy / 2, 0, fw)
    fl = fl.cut(cq.Workplane("XY", origin=(0, 0, -eps)).rect(ix - 2 * fw, iy - 2 * fw)
                .workplane(offset=fw + 2 * eps).rect(ix + 0.02, iy + 0.02).loft())
    b = b.union(fl)
    b = b.cut(box(-5, 5, by / 2 - stop_wall - 6, by / 2 + 1, -1, 6))           # cable slot, rear
    b = b.union(text_on((0, -by / 2 + eps, bz / 2), (1, 0, 0), (0, -1, 0), "STOP", 10.0, emboss + eps))
    # print top-down: flip so the button face is on the bed
    return b.rotate((0, 0, 0), (1, 0, 0), 180).translate((0, 0, bz))


def make_nameplate():
    lx, ly, lz = nameplate
    p = box(-lx / 2, lx / 2, -ly / 2, ly / 2, 0, lz).edges("|Z").fillet(4).faces(">Z").edges().fillet(0.8)
    for s in (-1, 1):
        p = p.cut(cq.Workplane("XY", origin=(0, 0, -1)).center(s * nameplate_hole_x, 0)
                  .slot2D(slot_len, m3_clear).extrude(lz + 2))
    p = p.union(text_on((0, 0, lz - eps), (1, 0, 0), (0, 0, 1), "SPOONFUL", 13.0, emboss + 0.2 + eps))
    return p


def make_clip():
    cable_d, wall, w = 5.0, 1.6, 10.0
    r_o = cable_d / 2 + wall
    base = box(-8, 8, -w / 2, w / 2, 0, 2.0)
    ring = (cq.Workplane("XZ", origin=(0, w / 2, 0)).center(0, 2.0 + cable_d / 2)
            .circle(r_o).circle(cable_d / 2).extrude(w))
    ring = ring.cut(box(-1.75, 1.75, -w, w, 2.0 + cable_d / 2, 20))             # snap opening 3.5
    one = base.union(ring)
    out = one
    for i in range(1, 4):
        out = out.union(one.translate((0, i * (w + 4), 0)))
    return out


def make_ring():
    r_i = ring_id / 2
    r = cq.Workplane("XY").circle(r_i + ring_wall).circle(r_i).extrude(ring_h)
    tab = box(r_i + ring_wall - 1, r_i + ring_wall + 18, -11, 11, 0, 3.0).edges("|Z").fillet(3)
    tab = tab.cut(cq.Workplane("XY", origin=(0, 0, -1)).center(r_i + ring_wall + 9, 0)
                  .slot2D(slot_len, m3_clear, angle=90).extrude(5))
    return r.union(tab)


# ============================================================
# BUILD + EXPORT
# ============================================================
def export(shape, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{name}.stl")
    cq.exporters.export(shape, path, tolerance=0.01, angularTolerance=0.1)
    bb = shape.val().BoundingBox()
    v = shape.val().Volume() / 1000
    print(f"{name:18s} {bb.xlen:6.1f} x {bb.ylen:6.1f} x {bb.zlen:5.1f}  {v:5.1f} cm3  ~{v * 1.24:4.0f} g solid")
    return path


def flip_z(shape, top_z):
    return shape.rotate((0, 0, 0), (1, 0, 0), 180).translate((0, 0, top_z))


if __name__ == "__main__":
    parts = {}  # name -> (print-oriented shape, assembly-placed shape or None)

    mount = make_mount()
    parts["Mount"] = (flip_z(mount, plate_t), mount)          # PRINT: top plate on the bed

    cradle_z0 = plate_t
    for label, clr in (("M", clear_M), ("L", clear_L)):
        cr, h = make_cradle(clr, label)
        placed = cr.translate((cradle_x0, 0, cradle_z0))
        parts[f"Cradle_{label}"] = (flip_z(cr, h), placed if label == "M" else None)   # PRINT: mating face down
        if label == "M":
            cap_z0 = cradle_z0 + h
    cap, h = make_cap("SM")
    parts["Cap"] = (cap, cap.translate((cradle_x0, 0, cap_z0)))  # PRINT as built

    parts["Stop_Housing_24"] = (make_stop(24.0), None)
    parts["Stop_Housing_28"] = (make_stop(28.0), None)
    parts["Nameplate"] = (make_nameplate(), None)
    parts["Cable_Clip"] = (make_clip(), None)
    parts["Bowl_Ring"] = (make_ring(), None)

    for name, (printed, _) in parts.items():
        export(printed, name)

    os.makedirs("assembly", exist_ok=True)
    for name, (_, placed) in parts.items():
        if placed is not None:
            cq.exporters.export(placed, f"assembly/{name}_placed.stl", tolerance=0.02, angularTolerance=0.2)
    print(f"interface centre X = {iface_x:.1f}, cradle X {cradle_x0:.1f}..{cradle_x0 + cradle_len:.1f}, "
          f"handle axis Z = {cap_z0:.2f}")

    if "--render" in sys.argv:
        import render_assembly
        render_assembly.main()
