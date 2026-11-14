"""The one CRADLE INTERFACE every cradle and the block share. Change it here and nothing else.

Part frame: Wrist_Roll_Pitch_SO101 (mm). The interface sits under the block's bottom face (Z = -30.2).

Layout (top view, looking up at the block's underside; X right, Y up):

      +-----------------------------+   boss / plate 20 (X) x 14 (Y), centred at (-1.8, 0)
      |     (P1)                    |   P1, P2  dia 3 locating pins, 14 mm apart (diagonal: dX 10.8, dY 9)
      |        (S1)        (S2)     |   S1, S2  M3 flat-head screws, 6.6 mm apart along X, on the centreline
      |                   (P2)      |
      +-----------------------------+

Why the screws are on the centreline (not 10 mm apart along Y): every cradle body hangs directly under this
20 x 14 area, so a screw driven from below would have its head buried under the cradle. Instead the screws are driven
from INSIDE the channel, up its open tip end, with a 2 mm hex key; flat (countersunk) heads sit flush in the end stop.
The channel interior of the smallest cradle (10.4 x 4.2) passes a 2 mm key, so all cradles share these positions.
"""

# --- placement ---
CX, CY = -1.8, 0.0            # interface centre (the brief's axis value; the URDF axis is X 0.0, see spoon_block.py)
BLOCK_BOTTOM_Z = -30.2        # block bottom face

# --- boss on the block / plate on the cradle ---
BOSS_X, BOSS_Y = 20.0, 14.0
BOSS_H = 0.0                  # NO raised boss: the block slides in along +X with only 0.2 mm above the pocket plate
                              # (plate top Z -30.4, block bottom -30.2); a 2 mm boss collides with the plate on the way in.
                              # The interface is the flat 20 x 14 area of the block's bottom face.
PLATE_T = 4.0
MATE_Z = BLOCK_BOTTOM_Z - BOSS_H          # boss face = plate top
END_STOP_Z = MATE_Z - PLATE_T             # plate underside = the spoon's end stop

# --- pins (3 mm steel dowel 3 x 8, pressed into the plate, slip into the block) ---
PINS = [(CX - 5.4, CY + 4.5), (CX + 5.4, CY - 4.5)]   # 14.06 mm apart on a diagonal. Purely along X they would
                                                     # break out of the block's -X face (block ends at X -10.2):
                                                     # only 13.35 mm fits along X with >= 1 mm walls.
PIN_D = 3.0
PIN_BLOCK_D, PIN_BLOCK_DEPTH = 3.1, 6.0   # in the block, from the boss face
PIN_PLATE_D, PIN_PLATE_DEPTH = 3.0, 3.5   # press fit in the plate, blind from the plate top (1 mm floor below)
PIN_LEN = 8.0

# --- screws: M3 x 10 flat head (DIN 7991 / ISO 10642), driven from below through the channel ---
SCREWS = [(CX - 3.3, CY), (CX + 3.3, CY)]
M3_CLEAR_D = 3.4
M3_CSK_D = 6.4                # countersink diameter at the end-stop face (head 6.0)
M3_CSK_ANGLE = 90.0
USE_INSERTS = True            # True: M3 heat-set insert pockets in the block; False: self-tap pilots
INSERT_D, INSERT_DEPTH = 4.0, 6.0
SELF_TAP_D, SELF_TAP_DEPTH = 2.5, 8.0     # M3 self-tap in PLA (1.7 mm is an M2 pilot: too small for M3)


def boss_solid():
    import cadquery as cq
    if BOSS_H <= 0:
        return None
    return (cq.Workplane("XY").box(BOSS_X, BOSS_Y, BOSS_H + 0.01, centered=(True, True, False))
            .translate((CX, CY, MATE_Z)))


def block_cuts():
    """Holes the block gets in the interface (pin holes + insert pockets / pilots), as one solid to subtract."""
    import cadquery as cq
    cut = None
    for x, y in PINS:
        c = cq.Workplane("XY").circle(PIN_BLOCK_D / 2).extrude(PIN_BLOCK_DEPTH).translate((x, y, MATE_Z - 0.01))
        cut = c if cut is None else cut.union(c)
    d, depth = (INSERT_D, INSERT_DEPTH) if USE_INSERTS else (SELF_TAP_D, SELF_TAP_DEPTH)
    for x, y in SCREWS:
        cut = cut.union(cq.Workplane("XY").circle(d / 2).extrude(depth).translate((x, y, MATE_Z - 0.01)))
    return cut


def plate_solid():
    import cadquery as cq
    return (cq.Workplane("XY").box(BOSS_X, BOSS_Y, PLATE_T, centered=(True, True, False))
            .translate((CX, CY, END_STOP_Z)))


def plate_cuts():
    """Pin press holes (blind from the top) + countersunk M3 clearance holes (countersink at the end-stop face)."""
    import cadquery as cq
    cut = None
    for x, y in PINS:
        c = cq.Workplane("XY").circle(PIN_PLATE_D / 2).extrude(PIN_PLATE_DEPTH + 0.01).translate((x, y, MATE_Z - PIN_PLATE_DEPTH))
        cut = c if cut is None else cut.union(c)
    csk_h = (M3_CSK_D - M3_CLEAR_D) / 2          # 90 deg: depth = radius difference
    for x, y in SCREWS:
        hole = cq.Workplane("XY").circle(M3_CLEAR_D / 2).extrude(PLATE_T + 2).translate((x, y, END_STOP_Z - 1))
        cone = (cq.Workplane("XY").circle(M3_CSK_D / 2).workplane(offset=csk_h).circle(M3_CLEAR_D / 2)
                .loft().translate((x, y, END_STOP_Z - 0.01)))
        cut = cut.union(hole).union(cone)
    return cut


def check_channel_access(interior_x0, interior_x1, interior_y0, interior_y1, key_d=2.0):
    """The hex key must reach both screws up the channel: each screw axis inside the interior with key clearance."""
    for x, y in SCREWS:
        assert interior_x0 + key_d / 2 <= x <= interior_x1 - key_d / 2 and \
            interior_y0 + key_d / 2 <= y <= interior_y1 - key_d / 2, \
            f"screw at ({x}, {y}) not reachable through the channel interior"
