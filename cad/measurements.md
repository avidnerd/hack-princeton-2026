Measure the gripper and spoon first, because the CAD needs them before you print. The arm link lengths can wait until bring-up, as long as the arm is on the table.

1. Gripper (affects the mount)

These values came from photos, so they could be off by about ±3 mm. Open the jaws fully for all of them.

Measure: Width of the gripper
  servo block, side to side, at
  its widest (include screw
  heads and bosses)
CAD variable: grip_w
Current guess: 62
────────────────────────────────
Measure: Length of the servo
  block, front to back
CAD variable: servo_len
Current guess: 45
───────
Meo
  block
CA
Current guess: 45
───────
Mew
  pads, fully open
CA
Current guess: 40
───────
Mefinger
  ervo
CAD variable: finger_stack
Cu
────────────────────────────────
Measure: Distance from the       re
  the pads start               CA
Current guess: 25 (pure guess) ───────                              Me
CAD variable: pad_len          Cu
────────────────────────────────
Measure: Wrist-roll horn face
to
  the fingertip
CA
Current guess: 105             
Also note anything that sticks ouleexit, screw heads, horn) and whnt's side skirts have to clear it.

2. Spoon (affects the cradl
- Total length (spoon_len, 
- Handle width and thickness  0and 125 mm from the handle  kthe points with a pencil    hrod is good for setting ou
- Bowl length and widest wid  tly34 × 21).                 
3. Hardware (affects the pocan
                            Meter×                           
  magnets, since they vary. Cu
  (pocket 10.3 × 3.2)       ───────                           Mesaid                            
Currently assumed: pocket 6.
  2.2                       ───────                           Me
  outer diameter × length   Cu
─────────────────────────────
Measure: Zip-tie width ×    
Currently assumed: slot 5 × 
4. Arm links (affect        co
                            Mecentres, not link edges. Theeasure from the outside edge of onehoe of the next, then subtract one hu
                            Mee of                            
Config key: base_height_mm  Pl
─────────────────────────────
Measure: Shoulder shaft to  el
  shaft                     Co
Placeholder: 98             ───────                           Me
  wrist-pitch shaft         Co
Placeholder: 96             ───────                           Meto
  the front face of the grip
  servo                     Co
Placeholder: used to work ou
                            Sth150 mm, so use a ruler for  anlate tool_mm and tool_offset_mm  (swrist) from the CAD once themoyou'll only need to check thaf
                            Semat, even a photo of notes. I'll upinterference check, re-export thconfig.yaml and the NOTES ta