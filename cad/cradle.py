"""Spoon cradles for the standard interface (interface.py): fitted channel cradles and a universal clamp.

    python cradle.py                       # exports Cradle_S, Cradle_M, Cradle_L, Cradle_Clamp (+ jaw)
    python cradle.py --width 12.3 --thick 4.6 --name Cradle_X    # a fourth, cut to measured numbers

Frame: Wrist_Roll_Pitch part frame (mm); bowl faces -Y, handle runs along -Z from the end stop (plate underside).
Print: plate DOWN (the face that seats on the block on the bed), channel up. No supports. The magnet pockets are then
horizontal, so they are flat-topped teardrops (roof toward the open end = up on the bed).
Pins: two 3 x 8 steel dowels pressed into the plate's top face (printed pins would point into the bed).
Screws: two M3 x 10 flat-head, driven from below UP THE CHANNEL with a 2 mm hex key; heads sit flush in the end stop.
"""
import argparse

import cadquery as cq

import interface as itf

# Defaults (all cradles)
WALL = 2.5            # side walls
FLOOR_T = 3.0         # floor on the +Y side (magnets in it)
LIP_T = 1.2           # lip thickness along Y
LIP_ENTRY = 0.7       # 45 deg lead-in on the lips' outer edge
MAG_D, MAG_DEPTH = 10.2, 2.2
MAG_FROM_STOP, MAG_PITCH = 8.0, 25.0
MAG_TEAR_CAP = 1.1    # teardrop roof truncated at 1.1 r (flat top bridges 3.2 mm); keeps 8 + 25 mm spacing in a 40 mm channel
POCKET_PLATE_EDGE_X = 9.5   # Wrist_Roll_Pitch plate reaches down to Z -34.5 for X >= 9.5
TAB_X0 = 12.0               # small tab X 12..19, Y -15..-9, down to Z -37.8

VARIANTS = {"Cradle_S": (10.0, 4.0), "Cradle_M": (13.0, 5.5), "Cradle_L": (16.0, 7.0)}


def box(x0, x1, y0, y1, z0, z1):
    return cq.Workplane("XY").box(x1 - x0, y1 - y0, z1 - z0, centered=False).translate((x0, y0, z0))


def magnet_pocket(x, z, face_y, toward_plus_y=True, d=MAG_D, depth=MAG_DEPTH, cap=MAG_TEAR_CAP):
    """Disc pocket on a face normal to Y, flat-topped teardrop roof toward -Z (= up when printed plate-down)."""
    r = d / 2
    sgn = -1 if toward_plus_y else 1                      # XZ workplane normal is -Y
    disc = cq.Workplane("XZ").center(x, z).circle(r).extrude(sgn * depth)
    c = r * cap
    roof = (cq.Workplane("XZ").polyline([(x - r * 0.7071, z - r * 0.7071), (x - (1.4142 * r - c), z - c),
                                         (x + (1.4142 * r - c), z - c), (x + r * 0.7071, z - r * 0.7071)])
            .close().extrude(sgn * depth))
    off = -0.01 if toward_plus_y else 0.01
    return disc.union(roof).translate((0, face_y + off, 0))


def fitted_cradle(handle_width, handle_thick, channel_length=40.0, lip=1.0, clear_w=0.4, clear_t=0.2):
    """Plate + channel. Returns (solid, info dict)."""
    in_w, in_t = handle_width + clear_w, handle_thick + clear_t
    cx = itf.CX
    cy = ((in_t / 2 + LIP_T) - (in_t / 2 + FLOOR_T)) / 2 + itf.CY      # centre the SHELL on the interface
    x0, x1 = cx - in_w / 2 - WALL, cx + in_w / 2 + WALL
    y_open, y_floor = cy - in_t / 2 - LIP_T, cy + in_t / 2 + FLOOR_T
    z1, z0 = itf.END_STOP_Z, itf.END_STOP_Z - channel_length
    mag_z = [z1 - MAG_FROM_STOP, z1 - MAG_FROM_STOP - MAG_PITCH]

    assert x1 <= POCKET_PLATE_EDGE_X - 0.3, f"channel reaches X {x1:.1f}: hits the pocket plate; max handle width ~{2 * (POCKET_PLATE_EDGE_X - 0.3 - cx - WALL) - clear_w:.1f}"
    assert in_w >= MAG_D + 0.2, f"inside width {in_w:.1f} mm needs >= {MAG_D + 0.2:.1f} (10.2 mm magnet pocket + 0.1 mm each side): min handle width {MAG_D + 0.2 - clear_w:.1f} mm"
    assert mag_z[-1] - MAG_D / 2 * MAG_TEAR_CAP >= z0 + 1.0, "channel too short for the second magnet"
    assert lip < in_w / 2 - 1.0, "lips close the mouth"
    itf.check_channel_access(cx - in_w / 2, cx + in_w / 2, cy - in_t / 2, cy + in_t / 2)

    solid = itf.plate_solid().union(box(x0, x1, y_open, y_floor, z0, z1))   # touch exactly: an overlap leaves slivers
    # Junction fillet (shell -> plate underside), all concave junction edges in ONE call with one radius, limited by
    # the smallest plate margin around the shell. Wide cradles (shell wider than the 20 mm plate in X) only have the
    # two Y-side junction edges, which end on the plate's outer faces.
    px0, px1 = itf.CX - itf.BOSS_X / 2, itf.CX + itf.BOSS_X / 2
    py0, py1 = itf.CY - itf.BOSS_Y / 2, itf.CY + itf.BOSS_Y / 2
    wide = x0 < px0 or x1 > px1                        # shell overhangs the plate in X: only the Y-side edges are junctions
    margins = [y_open - py0, py1 - y_floor] + ([] if wide else [x0 - px0, px1 - x1])
    r = min(2.0, min(margins) - 0.3)
    if r >= 0.5:
        sel = cq.selectors.BoxSelector((max(x0, px0) - 0.1, y_open - 0.1, z1 - 0.05), (min(x1, px1) + 0.1, y_floor + 0.1, z1 + 0.05))
        edges = solid.edges(sel)
        solid = (edges.edges("|X") if wide else edges).fillet(r)
    channel = box(cx - in_w / 2, cx + in_w / 2, cy - in_t / 2, cy + in_t / 2, z0 - 1, z1)
    h_in, h_out = in_w / 2 - lip, in_w / 2 - lip + LIP_ENTRY
    y_in, y_le = cy - in_t / 2 + 0.01, y_open + LIP_ENTRY
    mouth = (cq.Workplane("XY").polyline([(cx - h_in, y_in), (cx + h_in, y_in), (cx + h_in, y_le), (cx + h_out, y_open),
                                          (cx + h_out, y_open - 1), (cx - h_out, y_open - 1), (cx - h_out, y_open),
                                          (cx - h_in, y_le)]).close()
             .extrude(channel_length + 1).translate((0, 0, z0 - 1)))
    solid = solid.cut(channel).cut(mouth)
    for z in mag_z:
        solid = solid.cut(magnet_pocket(cx, z, cy + in_t / 2, toward_plus_y=True))
    solid = solid.cut(itf.plate_cuts())
    info = {"inside": (round(in_w, 2), round(in_t, 2)), "outer_x": (round(x0, 1), round(x1, 1)),
            "outer_y": (round(y_open, 1), round(y_floor, 1)), "end_stop_z": z1, "bottom_z": z0, "magnets_z": mag_z}
    return solid, info


# ------------------------------------------------------------------------------------------------
# Universal clamp: fixed jaw (U channel on the plate, ribbed floor, magnet lug) + separate moving jaw (bar).
# The bar closes the U's open -Y side, seated on the rail lands. One magnet in the fixed jaw's lug holds the bar via
# a steel M3 washer; a hook on the bar's other end wraps the +X rail, so a knock swings the bar off the magnet
# (breakaway). The M3 screw runs through a captive nut in the bar and presses the handle onto the ribbed floor: it
# sets the grip for 4-10 mm handles; it does not hold the bar on.
# ------------------------------------------------------------------------------------------------
CLAMP = dict(int_w=18.4, int_t=10.6, length=30.0, u_dx=-0.8, lug_w=12.5, lug_len=19.0, bar_t=4.0, hook_t=2.0,
             hook_reach=3.0, hook_clear=0.3, rib_h=1.0, rib_pitch=3.0, nut_af=5.6, nut_depth=2.6, washer_d=7.4,
             washer_depth=0.6)


def rib_rows(x0, x1, face_y, z_top, z_bot, into_minus_y, h, pitch):
    """Triangular ribs (45 deg both faces, printable) running along X on a face normal to Y."""
    ribs = None
    z = z_top - pitch / 2
    while z - h > z_bot:
        s = -1 if into_minus_y else 1
        tri = (cq.Workplane("YZ").polyline([(face_y, z + h), (face_y + s * h, z), (face_y, z - h)]).close()
               .extrude(x1 - x0).translate((x0, 0, 0)))
        ribs = tri if ribs is None else ribs.union(tri)
        z -= pitch
    return ribs


def clamp_cradle():
    p = CLAMP
    cx = itf.CX + p["u_dx"]
    cy = ((p["int_t"] / 2) - (p["int_t"] / 2 + FLOOR_T)) / 2 + itf.CY + 0.0   # centre U (lands..floor) on the interface
    ix0, ix1 = cx - p["int_w"] / 2, cx + p["int_w"] / 2
    land_y, floor_y = cy - p["int_t"] / 2, cy + p["int_t"] / 2
    z1, z0 = itf.END_STOP_Z, itf.END_STOP_Z - p["length"]
    rx0, rx1 = ix0 - WALL, ix1 + WALL
    assert rx1 <= POCKET_PLATE_EDGE_X - 0.3, f"clamp rail at X {rx1:.1f} hits the pocket plate"
    itf.check_channel_access(ix0, ix1, land_y, floor_y)
    lug_x0, lug_x1 = rx0 - p["lug_w"], rx0
    lug_z1, lug_z0 = itf.MATE_Z, itf.MATE_Z - p["lug_len"]      # lug reaches the mating plane: sits on the bed when printed
    mag_x, mag_z = (lug_x0 + lug_x1) / 2, lug_z0 + MAG_D / 2 + 1.5

    # fixed jaw
    fixed = itf.plate_solid()
    fixed = fixed.union(box(lug_x0, itf.CX - itf.BOSS_X / 2 + 0.01, -itf.BOSS_Y / 2, itf.BOSS_Y / 2, itf.END_STOP_Z, itf.MATE_Z))  # web to the lug
    fixed = fixed.union(box(rx0, rx1, land_y, floor_y + FLOOR_T, z0, z1 + 0.01))
    fixed = fixed.union(box(lug_x0, lug_x1 + 0.01, land_y, floor_y + FLOOR_T, lug_z0, lug_z1))
    fixed = fixed.cut(box(ix0, ix1, land_y - 1, floor_y, z0 - 1, z1))
    fixed = fixed.union(rib_rows(ix0, ix1, floor_y, z1, z0, True, p["rib_h"], p["rib_pitch"]))
    fixed = fixed.cut(magnet_pocket(mag_x, mag_z, land_y, toward_plus_y=True))
    fixed = fixed.cut(itf.plate_cuts())

    # moving jaw (bar)
    by0, by1 = land_y - p["bar_t"], land_y
    bx0, bx1 = lug_x0, rx1 + p["hook_clear"] + p["hook_t"]
    bar = box(bx0, bx1, by0, by1, z0, lug_z0 + p["lug_len"] - 2.0 if False else z1 - 2.0)
    bar = box(bx0, bx1, by0, by1, z0, z1 - 2.0)
    hook = box(rx1 + p["hook_clear"], bx1, by1 - 0.01, land_y + p["hook_reach"], z0, z1 - 2.0)
    bar = bar.union(hook)
    bar = bar.union(rib_rows(ix0 + 1, ix1 - 1, by1, z1 - 2.0, z0, False, p["rib_h"] * 0.6, p["rib_pitch"]))
    sx, sz = cx, (z0 + z1) / 2
    screw = cq.Workplane("XZ").center(sx, sz).circle(itf.M3_CLEAR_D / 2).extrude(-(p["bar_t"] + 4)).translate((0, by0 - 1, 0))
    nut = cq.Workplane("XZ").center(sx, sz).polygon(6, p["nut_af"] / 0.866).extrude(-p["nut_depth"]).translate((0, by0 - 0.01, 0))
    washer = cq.Workplane("XZ").center(mag_x, mag_z).circle(p["washer_d"] / 2).extrude(p["washer_depth"]).translate((0, by1 + 0.01, 0))
    bar = bar.cut(screw).cut(nut).cut(washer)
    info = {"interior": (p["int_w"], p["int_t"]), "rails_x": (round(rx0, 1), round(rx1, 1)), "lug_x": (round(lug_x0, 1), round(lug_x1, 1)),
            "magnet": (round(mag_x, 1), round(mag_z, 1)), "screw": (round(sx, 1), round(sz, 1)), "z": (z0, z1)}
    return fixed, bar, info


def export_printed(solid, name):
    """Plate down: rotate 180 about X (part +Z -> bed), drop to Z = 0. STL + STEP."""
    r = solid.rotate((0, 0, 0), (1, 0, 0), 180)
    bb = r.val().BoundingBox()
    r = r.translate((-bb.xmin, -bb.ymin, -bb.zmin))
    cq.exporters.export(r, f"{name}.stl", tolerance=0.01, angularTolerance=0.1)
    cq.exporters.export(r, f"{name}.step")
    cq.exporters.export(solid, f"{name}_inplace.stl", tolerance=0.01, angularTolerance=0.1)


def export_flat_bar(bar, name):
    """Bar prints flat, inner face (ribs, washer recess, hook) up: rotate so -Y (outer face) is on the bed."""
    r = bar.rotate((0, 0, 0), (1, 0, 0), -90)
    bb = r.val().BoundingBox()
    r = r.translate((-bb.xmin, -bb.ymin, -bb.zmin))
    cq.exporters.export(r, f"{name}.stl", tolerance=0.01, angularTolerance=0.1)
    cq.exporters.export(bar, f"{name}_inplace.stl", tolerance=0.01, angularTolerance=0.1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--width", type=float, help="handle width (mm), widest point inside the cradle")
    ap.add_argument("--thick", type=float, help="handle thickness (mm)")
    ap.add_argument("--name", default="Cradle_X")
    ap.add_argument("--length", type=float, default=40.0)
    ap.add_argument("--lip", type=float, default=1.0)
    ap.add_argument("--clear-w", type=float, default=0.4)
    ap.add_argument("--clear-t", type=float, default=0.2)
    a = ap.parse_args()
    jobs = {a.name: (a.width, a.thick)} if a.width else VARIANTS
    for name, (w, t) in jobs.items():
        s, info = fitted_cradle(w, t, a.length, a.lip, a.clear_w, a.clear_t)
        export_printed(s, name)
        print(f"{name}: handle {w} x {t} -> {info}")
    if not a.width:
        fixed, bar, info = clamp_cradle()
        export_printed(fixed, "Cradle_Clamp")
        export_flat_bar(bar, "Cradle_Clamp_Jaw")
        print(f"Cradle_Clamp: {info}")
