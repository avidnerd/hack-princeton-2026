# Spoonful on xArm 1S: fit guide

Source: `spoonful_xarm.py` (all parameters at the top). To rebuild:
`../.venv/bin/python spoonful_xarm.py --render`
The STLs are in `stl/`, already rotated to print orientation. Renders and the interference report are in `renders/`.

## What to print

| STL | Settings | Notes |
|---|---|---|
| `Mount` | 3 walls, 20% infill, no supports | Prints top-down (top plate on the bed). 8.4 cm³, the quickest part. |
| `Cradle_M` and `Cradle_L` | 4 walls, 40% infill, no supports | Prints with the mating face down and the pins up. The trough roof bridges 19 mm. |
| `Cap` | 4 walls, 40% infill, no supports | Mating face down. |
| `Stop_Housing_24` / `_28`, `Nameplate`, `Cable_Clip` (4 per plate), `Bowl_Ring` | 2 walls, 15% | Cosmetic. |

All parts are PLA with 0.2 mm layers. Volumes run 8 to 21 cm³ for the functional parts. I didn't slice them, so the 45-minute
target isn't checked. The cradles (20 to 21 cm³) are the most likely to go over. If one does, use 0.28 mm layers.

## Hardware

- 2 × M3 heat-set inserts in the mount. Press them in from the top face, flush.
- 2 × **M3 × 8 flat-head** screws, cradle to mount. A socket head sticks up about 1.4 mm into the trough and will hit the
  handle.
- 2 × zip ties (up to 5 mm wide), over the saddle and under the gripper plate.
- Magnets: up to 8 small discs (5.9 × 2), 4 in the cradle and 4 in the cap. Pockets are 6.2 × 2.2. The 17.8 mm discs
  aren't used: a pair would pull several kg, so the cap would never break away, and they don't fit the walls.

## Assembly

1. **Saddle on.** Push the mount down over the top of the upright gripper servo, fascia (`SPOONFUL`) at the front,
   flush with the servo's front face (the fingertip side). The skirts reach 14 mm down the sides, well above the
   mounting ears. The rear is open: the servo lead comes out of the rear face 6.9 mm below the top and is not covered.
2. **Zip ties.** Each tie runs across the top in its groove, down the channel inside each skirt, and under the black
   gripper plate the servo stands on, keeping clear of the finger linkage. Pull both tight. This is the main thing
   holding the mount on. Hot glue in the recess under the top plate if it still rocks.
3. **Cable.** Route the servo lead **down and back** to the wrist, not up in a loop. The spoon handle passes over the
   rear of the servo about 10 to 20 mm above its top, and its end stops about 3 mm short of the blue wrist bracket.
4. **Cradle.** Put the pins into the mount's pin holes and drive the two M3 × 8 flat-heads down through the trough
   floor into the inserts.
5. **Spoon.** Lay the handle in the trough, bowl forward and concave side up. The trough follows the handle's taper
   (narrow at 20 mm from the end, widest at 35, narrow again at 75), so the spoon can't slide either way.
6. **Cap.** Lower it on. The magnets pull it down and centre it.

## Magnets and polarity

```
  CAP (mating face down)          N dot on top marks the first pocket
   [ N ]           [ N ]          Cap: 4 pockets at 12 and 42 mm, both walls
     ^ attract                      the handle taper makes it fit one way only
   [ S ]           [ S ]          CRADLE: 4 pockets at 12 and 42 mm, both walls
   [ S ]           [ S ]          every cradle magnet S face UP
  CRADLE (mating face up)
```

To get the polarity right without thinking about it:
1. Glue all four cradle magnets in with the same face up. A fifth magnet held over each one should be pushed away the
   same way every time.
2. Set a cap magnet on top of each cradle magnet you will use. They snap on with the right face up.
3. Put a drop of super glue on top of each of those magnets, lower the cap onto them and press it shut. Leave it to cure.
   Each magnet ends up glued into the cap pocket above it, facing the right way.

Start with two magnet pairs, both at the 12 mm end. A sideways knock to the bowl should pop the cap off.
- **Holds under 300 g:** add more pairs.
- **Doesn't break away:** leave one pocket empty.

Two small pairs is the expected answer for a 300 to 800 g pull-off; this has not been tested yet.

## If something doesn't fit

| Problem | What to do |
|---|---|
| Mount won't go over the servo | Sand the 0.35 mm crush ribs on the inside of the skirts. If it's still tight, set `side_clear` to 0.7 and reprint the Mount only (about 15 min). |
| Mount rocks | Tape or hot glue in the 1 mm recess under the top plate, and tighten the ties. |
| Handle won't drop into Cradle_M | Use Cradle_L. Same cap, same screws. |
| Handle loose in Cradle_L | A dab of hot glue at the handle end of the trough. |
| Spoon handle end hits the wrist bracket when the wrist moves | Set `bowl_tip_ahead` to 110 (moves the spoon 10 mm forward) and reprint both cradles and caps. |
| New spoon | Edit `handle_profile` and `cradle_zone`, then reprint one cradle (about 20 min). The interface stays the same. |

**Hot glue is fine** in the saddle underside recess, at the handle end of the cradle trough, and on zip-tie tails.
**No hot glue** on magnet faces (it makes a gap and they lose most of their pull), on the interface pins or holes, or
on the cradle-to-cap mating faces.

## What's verified and what isn't

- Measured with calipers (2026-11-13, photo `cad/IMG_7144.heic`): servo body 40.0 along the arm (54.5 with the ears)
  × 20.2 across × 39.6 tall, finger layer 8.6, pads 56.9 ahead of the front face, cable 6.9 below the top
  (3.9 bundle), wrist bracket 22.2 behind the rear face.
- Mount, cradle and cap were checked against a **box model** of that: 0 mm³ overlap (`renders/interference.txt`).
- Also measured: ears 3.4 thick near the bottom of the body; the servo top is flat.
- Spoon handle, calipers with light jaw pressure (silicone squashes), width × thickness from the handle end:
  15: 15.2 × 10.4, 20: 15.6 × 10.8, 35: 16.8 × 9.0, 50: 16.8 × 8.6, 65: 15.2 × 8.9, 75: 13.4 × 8.6, 85: 12.3 × 8.8.
  The trough follows both, so it narrows at each end (the spoon can't slide) and gets shallower toward the bowl.
- Not measured: total spoon length (159 assumed) and whether the handle is straight. Length only changes how far
  the tip reaches, not the fit.
- Assumed, not measured: how high the blue wrist bracket rises (the spoon handle end stops about 3 mm short of it).
- Bowl tip: 100 mm ahead of the servo front face. Handle axis: 13.6 mm above the servo top.

## Changes from the brief

- The interface is **horizontal** (cradle sits on top of the saddle). The brief specified 14 mm pin spacing along the
  spoon axis and screws countersunk from the trough side, and only a horizontal joint allows both. With the spoon tip
  100 mm ahead of the face, the cradle sits over the saddle anyway.
- Plate thickness is 5, not 4, so the inserts have material to hold. Pins are 4.5 long, not 6, so they don't touch the servo.
- The top is flat with R6 corners and a 1 mm top chamfer instead of a dome, because it prints face-down and the
  interface needs a flat seat. `SPOONFUL` is on the front fascia, where the diner sees it.
- **No bite block.** The real gripper's fingers are flat linkages under the servo, so there is nothing for a block to
  sit between, and the mount doesn't need the gripper closed to stay on.
- **The saddle is 27 mm wide, not 72**, sized to the 20.2 mm upright servo, with the rear left open for the cable.
- The cradle has no closed end wall. The handle has to pass through there, and the taper already stops it moving.
- The cradle and cap are 48 mm wide all the way along. Magnet pockets are at 12 and 42 mm from the handle end.
- **Small magnets only, one cap** (was Cap_LG + Cap_SM): the large discs on hand are 17.8 mm, too strong and too big.
- The cradles print mating-face down, not trough-up, so the pins point up and need no support.
