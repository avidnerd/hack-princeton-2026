# Pitch (2 minutes, Sunday) DRAFT: fill numbers from `scripts/export_stats.py` after the dress rehearsals

**0:00 to 0:15, the problem.** About a million Americans can't lift a spoon to their own mouth (Parkinson's, ALS,
stroke, spinal cord injury) [FILL: cite a source or drop the number]. Every bite waits on another person. The
devices that exist cost around $6,000 and have to be set up for each person.

**0:15 to 0:25, what we built.** Spoonful: a $200 kit arm, a laptop camera, and software we wrote this weekend.

**0:25, the arm first moves.** Judge sits down. Caregiver presses Start; judge says "Spoonful, bite".
- 0:25 to 0:40: it scoops. The spoon stops 10 cm out and says "Ready when you are". Point at the screen: *it's
  waiting for you, and the reason is in big text.*
- 0:40 to 0:55: judge opens their mouth and the arm comes in, adjusting to where their mouth actually is. Then ask
  them to turn their head: the arm backs off right away. Face it again and it comes back.

**0:55 to 1:15, three things it does that a $6,000 device doesn't.**
1. *Adaptive delivery:* a new person sits down and gets fed. No setup.
2. *Consent:* it approaches only while you look at it with your mouth open. It never moves without a permit, and
   every decision is on screen.
3. *Teach by feeding:* the caregiver moves the arm by hand once and it repeats the motion. No programming.

**1:15 to 1:35, safety is the architecture.** Every motion command passes five layers in one place. Physical STOP,
voice "stop", and a magnet breakaway spoon. Over the weekend: [FILL] bites, [FILL] people, [FILL] contacts, and it
backs off within [FILL p95] ms of someone turning away.

**1:35 to 1:50, caregiver.** Learned pace ("every 11 s"), and the caregiver gets a text after each meal (Photon).

**1:50 to 2:00, close.** Independence is getting to choose when the next bite comes. Spoonful.

Rehearsal checklist: preflight green · fresh calibration · yogurt in the ring · rocker within reach · Wi-Fi off once
to prove the offline fallbacks.
