# Spoonful wrist parts: fit guide

All sources are in `cad/` and parametric. Run in `cad/.venv` (`source .venv/bin/activate`).
Fit check: `python fit_check.py`. Every part shows **0 mm³ interference** with the imported Wrist_Roll_Pitch mesh,
0 with the block, and a clear assembly path (`fit_check_report.md`, screenshots in `fit_check/`).

## Parts and hardware

| STL | What | Hardware |
|---|---|---|
| `Spoon_Block_SO101` | replaces servo 5; slides in from +X | 2× M2 × 8 self-tap (up through the wrist plate); 2× M3 heat-set inserts (Ø4 × 6 pockets) |
| `Cradle_S` / `_M` / `_L` | fitted channel cradles | 2× 3 × 8 steel dowels (press into the plate); 2× M3 × 10 **flat head**; 2× 10 × 2 N35 magnets; 2× steel M3 washers on the spoon |
| `Cradle_Clamp` + `Cradle_Clamp_Jaw` | universal fallback | same pins and screws; 1 magnet; 1 steel M3 washer (glued in the jaw recess); 1 M3 nut + M3 × 12 screw |
| `Phone_Stand` | camera stand (placeholder phone 78 × 160 × 10) | 2 strips of 19 mm mounting tape |

## Assembly

1. **Block.** Press in the two M3 inserts. Slide the block into the wrist pocket from +X, then fit the two M2 × 8
   screws up through the plate.
2. **Cradle.** Press the dowels into the cradle plate (flush with its top). Offer the cradle up from below, pins
   into the block. Drive both M3 × 10 flat-heads **up the channel** with a 2 mm hex key; the heads sit flush in the
   end stop.
3. **Spoon.** Glue the magnets into the floor pockets. Epoxy the steel washers to the underside of the handle at
   8 mm and 33 mm from the handle end.

## Which cradle first

Measure the **last 40 mm of the handle** (the part that goes in the cradle, handle end first). A cradle fits when
the width is in its range **and** the thickness is at or below its maximum:

| Cradle | Handle width | Max thickness |
|---|---|---|
| `Cradle_S` | 9.4 – 10.4 mm | 4.2 mm |
| `Cradle_M` | 12.4 – 13.4 mm | 5.7 mm |
| `Cradle_L` | 15.4 – 16.4 mm | 7.2 mm |

- **Upper width limit:** the channel width.
- **Lower width limit:** narrower handles slip past the two 1 mm lips; this assumes at least 0.5 mm of lip engagement per side.
- **No calipers:** try **M** first, then S or L. The spoon should click past the lips under light thumb pressure
  and lift out when the bowl is pushed sideways.
- **Gaps:** a handle 10.5–12.3 or 13.5–15.3 mm wide fits none of the three. Cut a fourth (next section), or use the
  clamp.
- **Length:** the spoon tip must be ≤ 90 mm below the servo's bottom face. With the end stop at Z −34.2, the
  **spoon can be at most 90.4 mm long**. Most baby spoons are longer, so shorten the handle end, or accept more
  overhang.

## Cutting a fourth cradle (about 2 minutes)

```bash
python cradle.py --width 12.3 --thick 4.6 --name Cradle_X   # your measured widest / thickest, in mm
python fit_check.py                                       # optional: re-run the interference check
```

- **Options:** `--lip 1.0` (lip overhang), `--clear-w 0.4 --clear-t 0.2` (fit clearance), `--length 40`
  (channel length).
- **Limits:** the script refuses handles wider than 16.6 mm (the channel would hit the wrist plate) or narrower
  than 10.0 mm (no room for the Ø10.2 magnet).
- **Clamp grip range:** handles 4–10 mm thick and up to 18 mm wide.

## Print settings (PLA, 0.4 nozzle, 0.2 mm layers; exported STLs are already oriented)

| Part | Orientation | Settings |
|---|---|---|
| Block | interface face **up**, top face on the bed | 4 walls, 40% infill, no supports |
| Cradles, clamp | plate **down** (the face that seats on the block on the bed), channel up | 4 walls, 40% infill, no supports |
| Clamp jaw | flat, ribs and hook up | 3 walls, 40% infill |
| Phone stand | base down | 3 walls, 20% infill, **tree supports** under the tilted cradle only |

**Magnet pockets are horizontal in this orientation, not "up".** They are flat-topped teardrops, so they print
without support. Printing them facing up would mean printing the channel on its side, with the lips and pins as
overhangs.

## Where these parts differ from the brief, and why

1. **No raised 2 mm boss.** The block slides in over the wrist plate with only 0.2 mm to spare (plate top at
   Z −30.4, block bottom at −30.2). A boss collided with the plate (up to 100 mm³) during insertion. The interface
   is the flat 20 × 14 area of the block's underside.
2. **M3 screws on the centreline, 6.6 mm apart, driven from below through the channel.** Every cradle body hangs
   directly under the 20 × 14 plate, so a screw head anywhere on it would be buried. Flat heads sit flush in the
   end stop. The pins stay 14 mm apart, on a diagonal: along X they broke out of the block's −X face.
3. **M2 pilots at the measured plate holes (28.85, ±10.25), not (28.4, ±10).** The brief's position is 0.5 mm off
   the hole. The optional pair at (8.5, ±10) is included, but the wrist plate has no holes there.
4. **Self-tap pilot is Ø2.5 if `USE_INSERTS = False`.** Ø1.7 is an M2 size and would split the PLA with an M3.
5. **Pins are steel dowels pressed into the cradle plate.** Printed pins would point into the bed with the plate
   down.
6. **The clamp's magnet acts on a land, not across the handle.** A 10 × 2 N35 disc pulling across a 4–10 mm handle
   gap holds only tens of grams. In this design:
   - The jaw seats on the rail lands and the magnet lug, and a hook on its other end wraps the +X rail.
   - A knock swings the jaw off the magnet (breakaway).
   - The M3 screw through the captive nut presses the handle onto the ribbed floor: it sets the grip only.
7. **Added overhang is 4 mm (the plate).** It was 3 mm with the earlier flange design.
8. **Interface centre is X −1.8 as specified.** The URDF wrist-roll axis is at X 0.0; −1.8 also keeps the plate
   0.4 mm clear of the wrist plate.
