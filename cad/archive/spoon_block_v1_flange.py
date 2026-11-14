"""Spoonful spoon block: replaces the SO-101 wrist-roll servo (servo 5) and carries a magnetic spoon cradle.

Coordinate frame: the Wrist_Roll_Pitch_SO101 part's own frame (mm), exactly as in
TheRobotStudio/SO-ARM100 STL/SO101/Individual/Wrist_Roll_Pitch_SO101.stl. The block is modelled IN PLACE, so it
can be overlaid on the pocket part for a fit check (fit_check.py).

Measured from the reference meshes (cad/ref/, see fit_check.py and ref/overlay.png), not taken from the brief:
  - servo 5 envelope: X -10.2..35.2, Y +/-12.4, Z -34.6..5.0; full rectangular body only from Z -28.6 (X<5.5)
    / -30.2 (X>=9.5); below that a neck X 8.6..34.7, Y +/-7.0 down to Z -31.4 that keys into a recess in the
    pocket plate, and the round output boss (dia 20) + spline at the axis.
  - old wrist-roll (output) axis: X 0.0, Y 0.0, along -Z (URDF joint 'wrist_roll'). The brief's X -1.8 is off.
  - pocket plate: Z -34.5..-30.4 at X >= ~9.5 (arc-shaped inner edge around the axis). Through-holes at
    (28.85, +/-10.25): dia 4.12 counterbore Z -34.5..-33.0, then dia 2.06 to the plate top. M2 only: the plate
    does NOT pass M3. There is no plate hole at X ~ 8.5.
  - wrist-flex axis = part +X, so the spoon pitches in the part YZ plane; world-up in the elbow-up pose is part -Y.
    The cradle therefore opens toward -Y (bowl's concave side faces -Y).

Print orientation: block bottom (cradle) UP, block top face (Z = +0.4) on the bed. Export rotates for this.
"""
import cadquery as cq

# ============================================================
# PARAMETERS
# ============================================================
# --- Spoon (MEASURE AND REPLACE: placeholders for a typical silicone baby spoon) ---
handle_width = 12.0       # mm, widest part of the handle inside the cradle (lies along part X)
handle_thick = 4.0        # mm (lies along part Y)

# --- Block body (servo 5 replica) ---
body_x0, body_x1 = -10.2, 35.2   # mm, servo body extent along X
body_w = 24.6                    # mm, along Y; servo is 24.8, 0.2 under for a slide fit
body_top_z = 0.4                 # mm, top of servo body (rear boss not replicated)
body_bot_z = -30.2               # mm, bottom of the full-width body (plate arms top at -30.4)
edge_chamfer = 1.0               # mm, on the 4 long (X-direction) edges
end_bevel = 3.3                  # mm, 45 deg bevel on the two -X vertical edges (servo has 3.0; pocket walls follow it)

# Neck that keys into the plate recess (servo neck X 8.6..34.7, Y +/-7.0, bottom -31.4)
neck_on = True
neck_clear = 0.3                 # mm, per side and at the bottom
neck_x0, neck_x1 = 8.6 + neck_clear, 34.7 - neck_clear
neck_half_w = 7.0 - neck_clear
neck_bot_z = -31.4 + neck_clear

# --- Mounting: M2 self-tapping from below, through the pocket plate ---
screw_pts = [(28.85, 10.25), (28.85, -10.25)]   # plate hole centres (measured)
pilot_d = 1.7                    # mm, M2 self-tap pilot in PLA
pilot_depth = 8.0                # mm from the block bottom face

# --- Cradle part (separate; bolts to the block's underside from below after the block is in the pocket) ---
# The block must slide in along +X over the pocket plate, and anything hanging below Z -30.4 at X > 9.5 would hit
# the plate. So, like the stock servo horn + wrist-roll follower, the cradle is attached afterwards from below,
# located by two 3 mm dowel pins and clamped by two M3 screws (diagonal pattern).
cradle_cx_nominal = 0.0          # mm, handle centre X: the old output axis. Shifted toward -X only if needed (see DERIVED)
centre_shell_on_axis = True      # shift the handle ~1 mm in Y so the cradle SHELL is centred on Y = 0 (frees room for screw heads)
cradle_len = 40.0                # mm, channel length along -Z below the flange
fit_w = 0.4                      # mm, inside width = handle_width + fit_w
fit_t = 0.2                      # mm, inside depth = handle_thick + fit_t
wall = 2.5                       # mm, side walls
floor_t = 3.0                    # mm, floor (the +Y side; magnets go in it)
lip_over = 1.0                   # mm, lip overhang over each top edge of the handle
lip_t = 1.2                      # mm, lip thickness (along Y)
lip_entry = 0.7                  # mm, 45 deg lead-in on the lips' outer edge so the handle snaps in

flange_x0, flange_x1 = -10.2, 8.6   # mm; stays 0.3 mm off the block neck (X 8.9) and ~0.9 mm off the pocket plate edge (X ~9.5)
flange_half_w = 11.8                # mm, along Y; pocket side arms start at |Y| 12.4 below the block
flange_t = 3.0                      # mm; this is the only overhang the two-part design adds
max_added_overhang = 3.0            # mm, agreed limit

# Fasteners (diagonal): M3 screws from below into the block, 3 mm steel dowel pins (3 x 8) block -> flange
fast_y = 9.4                     # mm, |Y| of all four (hole edge 0.7 mm from the flange edge)
flange_screws = [(-5.5, fast_y), (5.5, -fast_y)]
pins = [(5.5, fast_y), (-5.5, -fast_y)]
m3_clear_d = 3.4                 # mm, through the flange
m3_head_d = 5.5                  # mm, pan/socket head (clearance check only)
block_pilot_m3_d = 2.5           # mm, M3 self-tap pilot in the block; use M3 x 10 (7 mm engagement)
block_pilot_m3_depth = 9.0       # mm
pin_d = 3.0
pin_len = 8.0                    # mm
pin_press_d = 3.0                # mm, block hole (vertical, prints ~0.1 under: press fit)
pin_press_depth = 6.0            # mm into the block
pin_slip_d = 3.1                 # mm, flange hole (close slip fit; locates the cradle)
pin_slip_depth = 2.5             # mm, blind from the flange top face (pin protrudes 2.0)

# Junction fillets (cradle shell -> flange underside). Y-facing R2 (screw heads), -X side limited by the flange margin.
fillet_y = 2.0
fillet_x = 1.0

# Magnets: 10 x 2 N35 discs, pressed + glued into the floor's inside face
mag_d = 10.2
mag_depth = 2.2
mag_from_stop = 7.0              # mm, first magnet centre from the end stop (flange underside)
mag_pitch = 25.0                 # mm
mag_teardrop = True              # pockets are horizontal when printed; teardrop roof avoids an overhang
mag_tear_cap = 1.2               # teardrop apex truncated at this many radii (flat top bridges ~2.6 mm)

# Overhang budget: spoon tip no more than this far below the servo's bottom (Z -34.6), like the stock gripper
spoon_total_length = None        # mm, MEASURE; if longer than the budget, the handle runs up into a slot in the block
tip_limit_below_servo = 90.0     # mm
servo_bottom_z = -34.6

# ============================================================
# DERIVED + CHECKS
# ============================================================
in_w = handle_width + fit_w
in_t = handle_thick + fit_t
cradle_cx = min(cradle_cx_nominal, flange_x1 - in_w / 2 - wall)   # shift toward -X only if the handle is wide
cradle_cy = ((in_t / 2 + lip_t) - (in_t / 2 + floor_t)) / 2 if centre_shell_on_axis else 0.0
cr_x0, cr_x1 = cradle_cx - in_w / 2 - wall, cradle_cx + in_w / 2 + wall
cr_open_y = cradle_cy - in_t / 2 - lip_t          # -Y face (open side, lips)
cr_floor_y = cradle_cy + in_t / 2 + floor_t       # +Y outer face
fl_z1 = body_bot_z                                # flange top = block bottom face
fl_z0 = body_bot_z - flange_t
cr_z1, cr_z0 = fl_z0, fl_z0 - cradle_len          # flange underside = end stop
mag_z = [cr_z1 - mag_from_stop, cr_z1 - mag_from_stop - mag_pitch]

assert flange_t <= max_added_overhang + 1e-9, "flange adds more overhang than agreed"
assert cr_x0 >= flange_x0 - 1e-6 and cr_x1 <= flange_x1 + 1e-6, \
    f"handle too wide: max {flange_x1 - flange_x0 - 2 * wall - fit_w:.1f} mm"
assert -flange_half_w <= cr_open_y and cr_floor_y <= flange_half_w, "cradle deeper than the flange (Y)"
for sx, sy in flange_screws:
    band = (cr_open_y - fillet_y, cr_floor_y + fillet_y)
    gap = min(abs(sy) - m3_head_d / 2 - max(abs(band[0]), abs(band[1])), 99)
    assert gap >= 0.3, f"M3 head at Y={sy} only {gap:.2f} mm from the cradle fillet"
for x, y in flange_screws + pins:
    assert flange_half_w - (abs(y) + m3_clear_d / 2) >= 0.6, "hole too close to the flange edge"
assert pin_len - pin_press_depth < pin_slip_depth, "pin longer than its flange hole"
assert in_w >= mag_d + 1.0, f"inside width {in_w:.1f} too narrow for a {mag_d} magnet pocket"
assert floor_t - mag_depth >= 0.6, "floor too thin behind the magnet pockets"
assert mag_z[-1] - mag_d / 2 * (mag_tear_cap if mag_teardrop else 1) >= cr_z0 + 1.0, "second magnet runs off the channel end"
tip_floor_z = servo_bottom_z - tip_limit_below_servo
max_spoon_len = cr_z1 - tip_floor_z               # handle butting the end stop
slot_depth = 0.0
if spoon_total_length is not None:
    slot_depth = max(0.0, spoon_total_length - max_spoon_len)
    assert slot_depth <= flange_t + 20.0, (
        f"spoon {spoon_total_length} mm needs a {slot_depth:.0f} mm slot; more than the block allows. Shorten the handle.")


def box(x0, x1, y0, y1, z0, z1):
    return cq.Workplane("XY").box(x1 - x0, y1 - y0, z1 - z0, centered=False).translate((x0, y0, z0))


def zcyl(x, y, z0, h, d):
    return cq.Workplane("XY").circle(d / 2).extrude(h).translate((x, y, z0))


# ============================================================
# MODEL, phase 2: features
# ============================================================
# --- block ---
block = box(body_x0, body_x1, -body_w / 2, body_w / 2, body_bot_z, body_top_z)
block = block.edges("|Z").edges("<X").chamfer(end_bevel)
block = block.edges("|X").chamfer(edge_chamfer)
if neck_on:
    block = block.union(box(neck_x0, neck_x1, -neck_half_w, neck_half_w, neck_bot_z, body_bot_z + 0.01))
for x, y in screw_pts:                                  # M2, through the pocket plate
    z_face = neck_bot_z if (neck_on and neck_x0 <= x <= neck_x1 and abs(y) <= neck_half_w) else body_bot_z
    block = block.cut(zcyl(x, y, z_face - 0.01, pilot_depth, pilot_d))
for x, y in flange_screws:
    block = block.cut(zcyl(x, y, body_bot_z - 0.01, block_pilot_m3_depth, block_pilot_m3_d))
for x, y in pins:
    block = block.cut(zcyl(x, y, body_bot_z - 0.01, pin_press_depth, pin_press_d))

# --- cradle part ---
flange = box(flange_x0, flange_x1, -flange_half_w, flange_half_w, fl_z0, fl_z1)
shell = box(cr_x0, cr_x1, cr_open_y, cr_floor_y, cr_z0, cr_z1 + 0.01)
cradle = flange.union(shell)
# junction fillets: select the concave edges where the shell meets the flange underside
sel = lambda x0, x1, y0, y1: cq.selectors.BoxSelector((x0, y0, fl_z0 - 0.05), (x1, y1, fl_z0 + 0.05))  # noqa: E731
cradle = cradle.edges(sel(cr_x0 + 0.1, cr_x1 - 0.1, cr_open_y - 0.1, cr_open_y + 0.1)).fillet(fillet_y)
cradle = cradle.edges(sel(cr_x0 + 0.1, cr_x1 - 0.1, cr_floor_y - 0.1, cr_floor_y + 0.1)).fillet(fillet_y)
if cr_x0 - flange_x0 >= fillet_x + 0.2:
    cradle = cradle.edges(sel(cr_x0 - 0.1, cr_x0 + 0.1, cr_open_y + 0.5, cr_floor_y - 0.5)).fillet(fillet_x)
# channel + mouth with a lead-in (trapezoid: wider at the outer face)
channel = box(cradle_cx - in_w / 2, cradle_cx + in_w / 2, cradle_cy - in_t / 2, cradle_cy + in_t / 2, cr_z0 - 1, cr_z1)
h_in, h_out = in_w / 2 - lip_over, in_w / 2 - lip_over + lip_entry
y_in, y_le, y_out = cradle_cy - in_t / 2 + 0.01, cr_open_y + lip_entry, cr_open_y - 1.0
mouth = (cq.Workplane("XY").polyline([(cradle_cx - h_in, y_in), (cradle_cx + h_in, y_in), (cradle_cx + h_in, y_le),
                                      (cradle_cx + h_out, cr_open_y), (cradle_cx + h_out, y_out),
                                      (cradle_cx - h_out, y_out), (cradle_cx - h_out, cr_open_y),
                                      (cradle_cx - h_in, y_le)]).close()
         .extrude(cradle_len + 1).translate((0, 0, cr_z0 - 1)))
cradle = cradle.cut(channel).cut(mouth)
# magnet pockets in the floor's inside face, axis +Y; teardrop roof points to part -Z (= up when printed)
face_y = cradle_cy + in_t / 2
for z in mag_z:
    r = mag_d / 2
    prof = cq.Workplane("XZ").center(cradle_cx, z).circle(r)
    pocket = prof.extrude(-mag_depth)                     # XZ normal is -Y; negative extrude goes +Y
    if mag_teardrop:
        c = r * mag_tear_cap                                # 45 deg roof from the tangent points, flat at c
        tri = (cq.Workplane("XZ").polyline([(cradle_cx - r * 0.7071, z - r * 0.7071), (cradle_cx - (1.4142 * r - c), z - c),
                                            (cradle_cx + (1.4142 * r - c), z - c), (cradle_cx + r * 0.7071, z - r * 0.7071)])
               .close().extrude(-mag_depth))
        pocket = pocket.union(tri)
    cradle = cradle.cut(pocket.translate((0, face_y - 0.01, 0)))
# fasteners through the flange
for x, y in flange_screws:
    cradle = cradle.cut(zcyl(x, y, fl_z0 - 1, flange_t + 2, m3_clear_d))
for x, y in pins:
    cradle = cradle.cut(zcyl(x, y, fl_z1 - pin_slip_depth, pin_slip_depth + 0.01, pin_slip_d))
# long spoon: slot through the end stop and up into the block
if slot_depth > 0:
    slot = box(cradle_cx - in_w / 2, cradle_cx + in_w / 2, cradle_cy - in_t / 2, cradle_cy + in_t / 2,
               cr_z1 - 0.01, cr_z1 + slot_depth)
    cradle, block = cradle.cut(slot), block.cut(slot)

assembly = block.union(cradle)
result = assembly

# ============================================================
# EXPORT
# ============================================================
if __name__ == "__main__":
    for name, solid in [("spoon_block_inplace.stl", block), ("spoon_cradle_inplace.stl", cradle),
                        ("spoon_assembly_inplace.stl", assembly)]:
        cq.exporters.export(solid, name, tolerance=0.01, angularTolerance=0.1)
    # Print orientation: both parts flipped (rotate 180 deg about X) so the mating face is on the bed:
    #   block  -> its top face (Z +0.4) on the bed? No: the block's BOTTOM (pilot/pin holes) must face up, so the
    #             top face goes on the bed: rotate 180 about X puts Z +0.4 at the bottom.
    #   cradle -> flange top (the face that seats on the block) on the bed, channel pointing up.
    def to_bed(solid):
        r = solid.rotate((0, 0, 0), (1, 0, 0), 180)
        bb = r.val().BoundingBox()
        return r.translate((-bb.xmin, -bb.ymin, -bb.zmin))
    for name, solid in [("Spoon_Block_SO101.stl", block), ("Spoon_Cradle_SO101.stl", cradle)]:
        p = to_bed(solid)
        cq.exporters.export(p, name, tolerance=0.01, angularTolerance=0.1)
        cq.exporters.export(p, name.replace(".stl", ".step"))
    for n, s_ in [("block", block), ("cradle", cradle)]:
        bb = s_.val().BoundingBox()
        print(f"{n:6s} bbox X {bb.xmin:.1f}..{bb.xmax:.1f}  Y {bb.ymin:.1f}..{bb.ymax:.1f}  Z {bb.zmin:.1f}..{bb.zmax:.1f}")
    print(f"cradle centre X {cradle_cx:+.2f} (axis at 0); max handle width that fits: {flange_x1 - flange_x0 - 2 * wall - fit_w:.1f} mm")
    print(f"cradle inside {in_w:.1f} x {in_t:.1f} mm, handle centre ({cradle_cx:+.2f}, {cradle_cy:+.2f}); "
          f"end stop Z {cr_z1:.1f}; channel bottom Z {cr_z0:.1f}; magnets at Z {mag_z}")
    print(f"longest spoon without a slot (tip <= {tip_limit_below_servo:g} mm below the servo bottom): {max_spoon_len:.1f} mm;"
          f" slot into block: {slot_depth:.1f} mm")
