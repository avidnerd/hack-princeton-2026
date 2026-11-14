"""OS voice / mp3 player commands per platform (the Windows path can't run on the Mac, so check the argv)."""
from pathlib import Path

from spoonful.brain import speech


def test_windows_uses_powershell_sapi_with_text_in_env(monkeypatch):
    monkeypatch.setattr(speech.shutil, "which", lambda exe: r"C:\ps.exe" if exe == "powershell" else None)
    argv, env = speech.tts_command("I'll wait", "Samantha", 1.2, platform="win32")
    assert argv[0] == r"C:\ps.exe" and "System.Speech" in argv[-1]
    assert "I'll wait" not in " ".join(argv)            # text never goes through PowerShell quoting
    assert env == {"SPOONFUL_TEXT": "I'll wait", "SPOONFUL_RATE": "2"}
    argv, env = speech.play_command(Path("data/x.mp3"), platform="win32")
    assert "MediaPlayer" in argv[-1] and Path(env["SPOONFUL_FILE"]).is_absolute()


def test_mac_and_none(monkeypatch):
    monkeypatch.setattr(speech.shutil, "which", lambda exe: "/usr/bin/" + exe if exe in ("say", "afplay") else None)
    assert speech.tts_command("Paused", "Samantha", 1.0, platform="darwin")[0][:3] == ["say", "-v", "Samantha"]
    assert speech.play_command(Path("a.mp3"), platform="darwin")[0][0] == "afplay"
    monkeypatch.setattr(speech.shutil, "which", lambda exe: None)
    assert speech.tts_command("Paused", "", 1.0, platform="linux") is None
    assert speech.play_command(Path("a.mp3"), platform="linux") is None
