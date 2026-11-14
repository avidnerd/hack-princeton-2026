"""POST the caregiver summary to the Node Photon sidecar (sidecar/photon.js). Down sidecar -> (False, reason),
and the UI shows the summary with a Copy button instead."""
from __future__ import annotations

from ..config import PhotonCfg


def send_text(cfg: PhotonCfg, body: str, to: str | None = None) -> tuple[bool, str]:
    if not cfg.enabled:
        return False, "photon disabled in config"
    to = to or cfg.caregiver_number
    if not to or to.startswith("+1000"):
        return False, "photon.caregiver_number not set"
    try:
        import httpx
        r = httpx.post(f"{cfg.sidecar_url.rstrip('/')}/text", json={"to": to, "body": body}, timeout=8)
        if r.status_code == 200:
            return True, "sent"
        return False, f"sidecar {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return False, f"sidecar unreachable: {e.__class__.__name__}"


def health(cfg: PhotonCfg) -> bool:
    try:
        import httpx
        return httpx.get(f"{cfg.sidecar_url.rstrip('/')}/health", timeout=1).status_code == 200
    except Exception:
        return False
