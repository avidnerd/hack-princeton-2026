"""Spoonful spoon block: replaces the SO-101 wrist-roll servo (servo 5). Carries the standard cradle interface
(interface.py) on its bottom face; cradles bolt on after the block is in the pocket.

Coordinate frame: the Wrist_Roll_Pitch_SO101 part's own frame (mm), so the block can be overlaid on the pocket part.

Measured from the reference meshes (cad/ref/, fit_check.py, ref/overlay.png):
  - servo 5 envelope X -10.2..35.2, Y +/-12.4, Z -34.6..5.0; full-width body ends at Z -30.2; below it a neck
    X 8.6..34.7, Y +/-7.0 down to Z -31.4 that keys into a recess in the pocket plate; 3.0 mm 45 deg bevels on the
    two -X vertical edges (the pocket walls follow them).
  - pocket plate Z -34.5..-30.4 at X >= ~9.5. Its screw holes are at (28.85, +/-10.25), dia 4.12 counterbore then
    dia 2.06: M2 only. No plate hole at X ~8.5.
  - old wrist-roll axis X 0.0, Y 0.0 (URDF joint). The interface is centred at X -1.8 as specified; that also keeps
    the 20 mm interface plate 0.4 mm clear of the block neck / pocket plate on the +X side.

Print: bottom face (interface) UP, top face (Z +0.4) on the bed. 4 walls, 40% infill, 0.2 mm layers, no supports.
"""
import cadquery as cq

import interface as itf

# ============================================================
# PARAMETERS
# ============================================================
body_x0, body_x1 = -10.2, 35.2   # mm, servo body extent along X
body_w = 24.6                    # mm, along Y; servo is 24.8, 0.2 under for a slide fit
body_top_z = 0.4                 # mm, top of the servo body (rear boss not replicated)
body_bot_z = itf.BLOCK_BOTTOM_Z  # -30.2, bottom of the full-width body (pocket plate arms top at -30.4)
edge_chamfer = 1.0               # mm, the 4 long (X-direction) edges
end_bevel = 3.3                  # mm, 45 deg on the two -X vertical edges (servo 3.0)

neck_on = True                   # key into the pocket plate's recess (stops rocking)
neck_clear = 0.3
neck_x0, neck_x1 = 8.6 + neck_clear, 34.7 - neck_clear
neck_half_w = 7.0 - neck_clear
neck_bot_z = -31.4 + neck_clear

# M2 self-tap pilots, screws come up through the pocket plate. "measured" = plate hole centres from the mesh;
# "brief" = the brief's (28.4, +/-10), 0.5 mm off the plate holes, so the screw would cut off-centre. Keep measured.
m2_source = "measured"
m2_pts = {"measured": [(28.85, 10.25), (28.85, -10.25)], "brief": [(28.4, 10.0), (28.4, -10.0)]}[m2_source]
m2_optional_pts = [(8.5, 10.0), (8.5, -10.0)]   # requested optional pair; the pocket plate has no holes here
m2_optional_on = True
m2_pilot_d = 1.7                 # mm, M2 self-tap in PLA
m2_pilot_depth = 8.0             # mm; use M2 x 8 (the plate takes ~2.6 mm of a screw)

# ============================================================
# MODEL
# ============================================================
def box(x0, x1, y0, y1, z0, z1):
    return cq.Workplane("XY").box(x1 - x0, y1 - y0, z1 - z0, centered=False).translate((x0, y0, z0))


def zhole(x, y, z0, d, depth):
    return cq.Workplane("XY").circle(d / 2).extrude(depth).translate((x, y, z0 - 0.01))


block = box(body_x0, body_x1, -body_w / 2, body_w / 2, body_bot_z, body_top_z)
block = block.edges("|Z").edges("<X").chamfer(end_bevel)
block = block.edges("|X").chamfer(edge_chamfer)
if neck_on:
    block = block.union(box(neck_x0, neck_x1, -neck_half_w, neck_half_w, neck_bot_z, body_bot_z + 0.01))
if itf.boss_solid() is not None:
    block = block.union(itf.boss_solid())
for x, y in m2_pts + (m2_optional_pts if m2_optional_on else []):
    z_face = neck_bot_z if (neck_on and neck_x0 <= x <= neck_x1 and abs(y) <= neck_half_w) else body_bot_z
    block = block.cut(zhole(x, y, z_face, m2_pilot_d, m2_pilot_depth))
block = block.cut(itf.block_cuts())

result = block

if __name__ == "__main__":
    cq.exporters.export(block, "spoon_block_inplace.stl", tolerance=0.01, angularTolerance=0.1)
    printed = block.rotate((0, 0, 0), (1, 0, 0), 180)          # interface up, top face on the bed
    bb = printed.val().BoundingBox()
    printed = printed.translate((-bb.xmin, -bb.ymin, -bb.zmin))
    cq.exporters.export(printed, "Spoon_Block_SO101.stl", tolerance=0.01, angularTolerance=0.1)
    cq.exporters.export(printed, "Spoon_Block_SO101.step")
    bb = block.val().BoundingBox()
    print(f"block in place: X {bb.xmin:.1f}..{bb.xmax:.1f}  Y {bb.ymin:.1f}..{bb.ymax:.1f}  Z {bb.zmin:.1f}..{bb.zmax:.1f}")
    print(f"interface: boss face Z {itf.MATE_Z}, end stop Z {itf.END_STOP_Z}; inserts: {itf.USE_INSERTS}")
