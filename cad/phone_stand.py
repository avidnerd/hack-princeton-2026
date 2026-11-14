"""Spoonful phone camera stand: holds the phone's REAR camera facing the diner, landscape, tilted 15 deg up.

Frame: X along the base's 120 mm side (= phone's long side, landscape), -Y = toward the diner (camera side),
Z up, base underside at Z = 0. Print as modelled: base flat on the bed, post vertical; tree supports under the
tilted cradle.

Cradle walls (back_mode):
  "lips" (default, per the 2026-11-05 brief): 6 mm lips on both faces only, so the camera is never covered. The phone
         leans back 15 deg, and the screen-side lip stops it rotating back, so 6 mm is enough to hold it.
  "strip_fingers": screen side gets a 12 mm strip + two 8 mm fingers at the top corners (stiffer hold).
Ends (X) are open. The 14 mm cable notch is cut in the camera-side lip at the charging-port end, derived from
camera_corner (see port_end below): the only place the camera corner matters.
"""
import math

import cadquery as cq

# ============================================================
# PARAMETERS
# ============================================================
# --- Phone IN ITS CASE (PLACEHOLDERS from the brief: 78 x 160 x 10, typical large phone in a case) ---
phone_long = 160.0        # mm, long side (horizontal in landscape)
phone_short = 78.0        # mm, short side (vertical in landscape)
phone_thick = 10.0        # mm, with case
camera_corner = "top-left"   # in landscape, as seen from the SCREEN side (how you hold it)
# For a phone whose rear camera is top-left in portrait (seen from the back), landscape with the camera on top puts it
# top-left seen from the screen = top-RIGHT seen by the diner, and the charging port at the DINER's LEFT end (-X).
port_end = {"top-left": -1, "top-right": +1, "bottom-left": +1, "bottom-right": -1}[camera_corner]   # -1 = -X end
phone_mass_g = 230.0      # g, with case (only for the tipping check)

# --- Base ---
base_l, base_w, base_t = 120.0, 90.0, 5.0     # mm (X, Y, Z)
base_r = 8.0                                  # mm, corner radius
tape_strip_w = 19.0                           # mm, double-sided mounting tape width
tape_strip_l = 104.0                          # mm
tape_recess = 0.6                             # mm deep, two strips at Y = +/- tape_y
tape_y = 20.0

# --- Post ---
post_x, post_y = 20.0, 15.0                   # mm, section
post_h = 150.0                                # mm, H: from the base top to the cradle floor's lowest point
post_fillet = 8.0                             # mm, at the root
post_cy = None                                # mm, post centre Y; None = as far back as the tipping check allows

# --- Cradle ---
tilt_deg = 15.0                               # lens looks up toward a seated face
pocket_clear_long = 1.0                       # mm, pocket length = phone_long + this
pocket_clear_thick = 0.6                      # mm, pocket depth = phone_thick + this
floor_t = 4.0                                 # mm, under the phone
lip_h, lip_t = 6.0, 3.0                       # camera-side lip (above the floor)
back_mode = "lips"                            # "lips" or "strip_fingers"
strip_h = 12.0                                # screen-side strip height above the floor
finger_w = 8.0                                # screen-side fingers at the two top corners
back_t = 4.0                                  # mm, screen-side strip + finger thickness (stiffer than the lip)
finger_top_gap = 4.0                          # fingers stop this far below the phone's top edge
cable_notch_w = 14.0                          # mm, in the camera-side lip at the port end
cable_notch_inset = 4.0                       # mm from the end of the tray to the notch edge

# --- Checks ---
tip_margin_mm = 20.0                          # combined CG must be this far inside the base footprint (no tape);
                                              # 20 mm resists a push on the phone while it is lifted out
pla_density = 1.24e-3                         # g/mm3 (x infill fraction below)
infill_equiv = 0.45                           # rough solid fraction at 3 walls / 20% infill

# ============================================================
# DERIVED
# ============================================================
t = math.radians(tilt_deg)
pk_len = phone_long + pocket_clear_long
pk_dep = phone_thick + pocket_clear_thick
tray_dep = lip_t + pk_dep + back_t            # local y: camera lip | pocket | screen-side wall
tray_len = pk_len                              # ends open

# Cradle local frame (u along X, v across the tray, w up the phone), then rotated by tilt about X so the phone's top
# leans toward +Y. Local origin = bottom of the pocket floor at the camera-side lip's outer face.
def to_world(v, w, z0, y0):
    """local (v across, w up) -> world (y, z), tilt back = top moves toward +Y."""
    return y0 + v * math.cos(t) + w * math.sin(t), z0 - v * math.sin(t) + w * math.cos(t)


# Phone CG in local coords: middle of the pocket, half the phone height above the floor top
cg_v, cg_w = lip_t + pk_dep / 2, floor_t + phone_short / 2


def build(post_cy):
    z0 = base_t + post_h
    y0 = post_cy - tray_dep / 2          # tray centred on the post (before tilt)
    base = (cq.Workplane("XY").box(base_l, base_w, base_t, centered=(True, True, False))
            .edges("|Z").fillet(base_r))
    for sy in (tape_y, -tape_y):
        base = base.cut(cq.Workplane("XY").box(tape_strip_l, tape_strip_w, tape_recess + 0.01, centered=(True, True, False))
                        .translate((0, sy, -0.01)))
    post = cq.Workplane("XY").box(post_x, post_y, post_h + floor_t, centered=(True, True, False)).translate((0, post_cy, base_t - 0.01))
    stand = base.union(post)
    stand = stand.edges(cq.selectors.BoxSelector((-post_x / 2 - 0.1, post_cy - post_y / 2 - 0.1, base_t - 0.1),
                                                 (post_x / 2 + 0.1, post_cy + post_y / 2 + 0.1, base_t + 0.1))).fillet(post_fillet)
    # tray in local coords: XY plane = (u, v), Z = w
    tray = cq.Workplane("XY").box(tray_len, tray_dep, floor_t, centered=(True, False, False))
    tray = tray.union(cq.Workplane("XY").box(tray_len, lip_t, floor_t + lip_h, centered=(True, False, False)))
    back_h = lip_h if back_mode == "lips" else strip_h
    strip = cq.Workplane("XY").box(tray_len, back_t, floor_t + back_h, centered=(True, False, False)).translate((0, lip_t + pk_dep, 0))
    tray = tray.union(strip)
    if back_mode == "strip_fingers":
        fh = floor_t + phone_short - finger_top_gap
        for sx in (-1, 1):
            tray = tray.union(cq.Workplane("XY").box(finger_w, back_t, fh, centered=(True, False, False))
                              .translate((sx * (tray_len / 2 - finger_w / 2), lip_t + pk_dep, 0)))
    if cable_notch_w > 0:
        nx = port_end * (tray_len / 2 - cable_notch_inset - cable_notch_w / 2)
        tray = tray.cut(cq.Workplane("XY").box(cable_notch_w, lip_t + 1, lip_h + 1, centered=(True, False, False))
                        .translate((nx, -0.5, floor_t)))
    # place: rotate about X by +tilt (top toward +Y), move local origin to (y0, z0)
    tray = tray.rotate((0, 0, 0), (1, 0, 0), tilt_deg).translate((0, y0, z0))
    stand = stand.union(tray)
    return stand, z0, y0


def tipping(post_cy):
    """Combined CG (phone + stand) in Y vs the base footprint. Returns (cg_y, margin front, margin rear)."""
    stand, z0, y0 = build(post_cy)
    m_s = stand.val().Volume() * pla_density * infill_equiv
    c = stand.val().Center()
    py, _ = to_world(cg_v, cg_w, z0, y0)
    cg = (m_s * c.y + phone_mass_g * py) / (m_s + phone_mass_g)
    return stand, cg, cg + base_w / 2, base_w / 2 - cg, m_s, py


# Choose the post position: as far back as possible while the CG keeps tip_margin_mm inside the rear edge.
if post_cy is None:
    for cy in [base_w / 2 - post_y / 2 - post_fillet - 1.0 - k * 1.0 for k in range(0, 60)]:   # root fillet needs base behind the post
        stand, cg, m_front, m_rear, m_s, py = tipping(cy)
        if m_rear >= tip_margin_mm and m_front >= tip_margin_mm:
            post_cy = cy
            break
    else:
        raise SystemExit("no post position keeps the CG inside the base: widen the base")
else:
    stand, cg, m_front, m_rear, m_s, py = tipping(post_cy)
    assert m_rear >= tip_margin_mm and m_front >= tip_margin_mm, f"tips: CG {cg:.1f} mm, margins {m_front:.1f}/{m_rear:.1f}"

result = stand

if __name__ == "__main__":
    cq.exporters.export(result, "Phone_Stand.stl", tolerance=0.01, angularTolerance=0.1)
    cq.exporters.export(result, "Phone_Stand.step")
    bb = result.val().BoundingBox()
    z0 = base_t + post_h
    lens_z = to_world(0, floor_t + phone_short / 2, z0, post_cy - tray_dep / 2)[1]
    print(f"bbox {bb.xlen:.1f} x {bb.ylen:.1f} x {bb.zlen:.1f} mm; post centre Y {post_cy:+.1f} (base rear edge +{base_w / 2:g})")
    print(f"stand ~{m_s:.0f} g, phone {phone_mass_g:.0f} g; combined CG Y {cg:+.1f} mm; margin front {m_front:.1f}, rear {m_rear:.1f} mm")
    print(f"phone centre height ~{lens_z:.0f} mm above the table; at 600 mm a 15 deg tilt aims at "
          f"~{lens_z + 600 * math.tan(t):.0f} mm")
