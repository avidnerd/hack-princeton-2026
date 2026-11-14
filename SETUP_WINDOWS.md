# Setting up Spoonful on the Dell Alienware (Windows 11)

This takes you from a fresh Windows laptop to the arm moving. Run every command in **PowerShell**: Start, type
`powershell`, press Enter. Lines starting with `#` are comments. You don't need to type them.

Work through the steps in order. Steps 1 to 4 need no hardware. Step 5 is the arm.

> Use native Windows, **not WSL**. WSL can't see the arm's USB board without extra tools.

---

## 1. Install the tools (about 10 min, once)

```powershell
winget install --id Git.Git -e
winget install --id Python.Python.3.12 -e
winget install --id GitHub.cli -e
winget install --id OpenJS.NodeJS.LTS -e          # optional: only for caregiver texts (Photon)
winget install --id Microsoft.VCRedist.2015+.x64 -e   # C++ runtime that mediapipe/opencv need
```

**Close PowerShell and open a new one** so it picks up the new programs. Check that they installed:

```powershell
py -3.12 --version     # Python 3.12.x
git --version
gh --version
```

## 2. Get the code

The repo is private, so sign in to GitHub first. The command opens a browser.

```powershell
gh auth login                     # GitHub.com -> HTTPS -> Login with a web browser
cd $HOME\Documents
gh repo clone avidnerd/spoonful
cd spoonful
```

You now have three folders:

| Folder | What it is |
|---|---|
| `spoonful\` | The software: arm control, face tracking, web app, tests |
| `cad\` | 3D-printed parts. The STLs to print are in `cad\xarm1s\stl\` |
| `scratch\` | Bring-up scripts for the arm (step 5) |

## 3. Install the software (about 5 min)

```powershell
cd spoonful
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
```

- The face model (`models\face_landmarker.task`) is already in the repo, so there's nothing to download.
- `.env` holds API keys (ElevenLabs, Gemini, Photon). Every key is optional and everything works without them. To
  carry keys over from the Mac, copy them by hand: open `.env` in Notepad (`notepad .env`) and paste them in. **Never
  commit `.env`.**
- Every command here calls `.venv\Scripts\python.exe` directly, so you never need to "activate" the venv. That's
  what normally runs into PowerShell's script-blocking policy.

Check it:

```powershell
.venv\Scripts\python.exe -m pytest -q        # expect "66 passed"
```

## 4. Try it with no robot (2 min)

```powershell
.venv\Scripts\python.exe -m spoonful seed-sim
.venv\Scripts\python.exe -m spoonful --backend sim run --sim-face --open
```

A browser opens. Press `o` for the operator drawer, then **Start session** and **Bite**. On the **Sim** tab, tick
"mouth open". Press Ctrl+C in PowerShell to quit.

**Use your real camera:** run the same command without `--sim-face`. The first time, Windows may ask for camera
permission. If it doesn't and you get no picture: Settings → Privacy & security → Camera → turn on **"Let desktop
apps access your camera"**. Do the same under Microphone if you'll use voice commands.

**Pick the right camera.** Alienwares often have two: the normal one and an infrared Windows Hello camera. The IR
one shows up as a grey, flickering image. Find the colour one:

```powershell
.venv\Scripts\python.exe ..\scratch\check_camera.py --list          # shows which numbers open
.venv\Scripts\python.exe ..\scratch\check_camera.py --index 0 --secs 10   # needs at least 20 FPS
```

Then set `camera: {index: N ...}` in `spoonful\config.yaml`. If the FPS is low, try `--api msmf`. If that's
better, set `api: msmf` in the camera line of `config.yaml`.

**Plug the laptop in and use performance mode.** On battery, Alienwares throttle hard and face tracking drops
frames. Use Alienware Command Center → Performance, or Settings → Power → Best performance.

---

## 5. Connect the arm (Phase 0, about 30 min)

### 5a. Power and USB: no motion

1. Plug the 7.5 V 6 A adapter into the arm's control board, with its switch **OFF**.
2. Connect the board's USB to the laptop. No driver is needed: Windows sees it as a USB HID device.
3. Put the arm in a safe pose: folded, nothing in reach, your hand near the switch. Switch **ON**.
4. Check that Windows sees the board:

   ```powershell
   Get-PnpDevice -PresentOnly | Where-Object InstanceId -match 'VID_0483&PID_5750'
   ```

   One or more lines with `Status OK` means it's connected. Nothing means try another USB port or cable (some cables
   carry power only) and check that the board's light is on.

5. Read the voltage and positions. **This never moves the arm.**

   ```powershell
   cd $HOME\Documents\spoonful\spoonful
   .venv\Scripts\python.exe -m spoonful --backend xarm test-arm
   ```

   You should see the battery voltage (about 7 to 8 V) and six servo positions. If it can't open the device, try
   `--backend raw`.

### 5b. Map the joints: motors off, you move the arm by hand

**Hold the arm before you run this.** It turns the motors off and the arm goes limp.

```powershell
.venv\Scripts\python.exe ..\scratch\bringup_arm.py watch
```

Move **one joint at a time**, slowly, from end stop to end stop: base, shoulder, elbow, wrist pitch, wrist roll,
gripper. The script prints which servo ID changed and its min and max. For each joint, write down in the table in
`spoonful\NOTES.md`:
- which ID moved
- its raw range
- which direction counts as "forward/down"

Press Ctrl+C when you're done.

Then put those numbers into `spoonful\config.yaml`:
- `arm.servo_map`: joint name → servo ID. The current values are guesses.
- `arm.soft_limits_deg`: 5° inside each hard stop.
- `kinematics.sign`: +1 or −1 per joint, so that positive matches the direction you wrote down.

### 5c. Measure the links (ruler, 5 min)

Measure axis to axis (the centre of one joint's rotation to the next):
- base height
- upper arm (shoulder → elbow)
- forearm (elbow → wrist pitch)
- wrist pitch axis → gripper servo's front face

Write them in `NOTES.md` under "Link lengths", then put them in `config.yaml` under `kinematics`.

### 5d. First real motion

```powershell
.venv\Scripts\python.exe ..\scratch\bringup_arm.py jog           # DRY RUN: only prints the moves
.venv\Scripts\python.exe ..\scratch\bringup_arm.py jog --live    # asks y/N before EACH joint, moves +-10 deg and back
```

Keep a hand near the power switch. If a joint moves the wrong way or further than 10°, **switch off** and fix
`kinematics.sign` / `servo_map` before going on.

### 5e. Preflight, then go live

```powershell
.venv\Scripts\python.exe scripts\preflight.py --backend xarm
```

When it's all green, set `arm: backend: xarm` in `config.yaml`. Then run the full demo with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\demo.ps1
```

(`-ExecutionPolicy Bypass` applies only to this one run and changes no settings.) After that, follow
`spoonful\README.md`: calibrate, teach a bite, run a session.

---

## Optional pieces

| Piece | Windows steps |
|---|---|
| **Voice output** | Works out of the box with the built-in Windows voice. `voice: Samantha` in config only applies on a Mac. For the ElevenLabs voice, put `ELEVENLABS_API_KEY` in `.env` and set `speech: backend: elevenlabs`. Lines are cached in `data\tts_cache\` on first run, so later runs need no internet. |
| **Voice commands** ("Spoonful, bite") | Set `voice: enabled: true`. The first run downloads the Whisper `small` model (about 500 MB), **so do it once on good Wi-Fi before the demo.** It runs on the CPU, so the NVIDIA GPU isn't needed. |
| **Caregiver texts** (Photon) | `cd sidecar`, `npm install`, put the `PHOTON_*` keys in `.env`. `demo.ps1` starts it automatically. |
| **ESP32 STOP button** | Flash `firmware\esp32_stop_piezo\esp32_stop_piezo.ino` with the Arduino IDE (board: ESP32S3 Dev Module). An ESP32-S3 on its native USB needs no driver. A board with a CP210x or CH340 chip needs that chip's driver. The port is found automatically. If it isn't, set `esp32: port: COM5` (whatever Device Manager → Ports shows). Without the ESP32, the **spacebar is STOP**. |
| **CAD** (edit or re-render parts) | `cd ..\cad`, `py -3.12 -m venv .venv`, `.venv\Scripts\python.exe -m pip install -r requirements.txt`, then `cd xarm1s` and `..\.venv\Scripts\python.exe spoonful_xarm.py --render`. You only need this to change parts: the STLs are already built. |

## Moving from the Mac

These are **not** in git (they're per machine or secret). Copy them over on a USB stick or by AirDrop to your phone,
if they exist on the Mac:

| From the Mac (`hack-princeton/spoonful/...`) | Why |
|---|---|
| `.env` | API keys |
| `data/calibration/`, `data/trajectories/` | Calibration and taught bites. Empty right now, so nothing to copy yet |
| `data/tts_cache/` | Saves regenerating ElevenLabs lines |

Code changes go through git: `git pull` on one machine after `git push` on the other.

## If something goes wrong

| Problem | Fix |
|---|---|
| `py` not found | Reopen PowerShell after installing Python. Or reinstall from python.org and tick "Add to PATH". |
| `DLL load failed` importing mediapipe or cv2 | Install the C++ runtime (step 1, last line), then reopen PowerShell. |
| pip fails building something | You're on the wrong Python. Check with `.venv\Scripts\python.exe --version` (must be 3.12). Delete `.venv` and redo step 3 with `py -3.12`. |
| Black or grey camera picture | Wrong index (IR camera), or camera privacy is off (step 4). Close Zoom, Teams or the Camera app, since only one program can use the camera. |
| Camera takes 10+ s to start | Make sure `api: auto` (DirectShow) is set, not `msmf`. |
| `test-arm` can't open the device | Run the `Get-PnpDevice` check. Try `--backend raw`. Unplug and replug USB with the arm powered. Close any other program talking to the arm, such as Hiwonder's PC software. |
| Arm twitches or resets when it moves | Weak power: use the 6 A adapter, not USB power. Check the voltage in `test-arm`. |
| No voice | Check Windows sound output and `speech: enabled: true` in config. Test the Windows voice: `Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak("test")` |
| Firewall popup when starting the app | Choose cancel or Private only. The app listens on 127.0.0.1 (this laptop only). |
