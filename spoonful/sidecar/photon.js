// Spoonful Photon sidecar: Express on 127.0.0.1:7777.
//   POST /text {to: "+1...", body: "..."}  -> sends an iMessage via spectrum-ts
//   GET  /health                           -> {ok, ready}
// API verified against spectrum-ts 12.16.0 typings (dist/*.d.ts):
//   const app = await Spectrum({ projectId, projectSecret, providers: [imessage.config()] });
//   const space = await imessage(app).space.create(phone);   // E.164 phone or email handle
//   await space.send(text(body));
// Credentials from ../.env: PHOTON_PROJECT_ID, PHOTON_PROJECT_SECRET.
// Without them the sidecar still starts and /text returns 503, so the Python side shows "Copy" instead.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import express from "express";

const here = path.dirname(fileURLToPath(import.meta.url));
const envPath = path.join(here, "..", ".env");
if (fs.existsSync(envPath)) {
  for (const line of fs.readFileSync(envPath, "utf8").split("\n")) {
    const m = line.match(/^\s*([A-Z0-9_]+)\s*=\s*(.*)\s*$/);
    if (m && !(m[1] in process.env)) process.env[m[1]] = m[2].replace(/^["']|["']$/g, "");
  }
}

const PORT = Number(process.env.PHOTON_SIDECAR_PORT || 7777);
let app = null;
let imessage = null;
let text = null;
let initError = null;

async function init() {
  const { PHOTON_PROJECT_ID: projectId, PHOTON_PROJECT_SECRET: projectSecret } = process.env;
  if (!projectId || !projectSecret) {
    initError = "PHOTON_PROJECT_ID / PHOTON_PROJECT_SECRET not set";
    return;
  }
  try {
    const core = await import("spectrum-ts");
    const prov = await import("spectrum-ts/providers/imessage");
    imessage = prov.imessage;
    text = core.text;
    app = await core.Spectrum({ projectId, projectSecret, providers: [imessage.config()] });
  } catch (e) {
    initError = String(e && e.message ? e.message : e);
  }
}

const server = express();
server.use(express.json({ limit: "32kb" }));

server.get("/health", (_req, res) => res.json({ ok: true, ready: !!app, error: initError }));

server.post("/text", async (req, res) => {
  const { to, body } = req.body || {};
  if (typeof to !== "string" || !/^\+?[0-9]{7,15}$|^[^@\s]+@[^@\s]+$/.test(to) || typeof body !== "string" || !body) {
    return res.status(400).json({ ok: false, error: "need {to: E.164 or email, body: string}" });
  }
  if (!app) return res.status(503).json({ ok: false, error: initError || "not ready" });
  try {
    const space = await imessage(app).space.create(to);
    await space.send(text(body.slice(0, 1000)));
    res.json({ ok: true });
  } catch (e) {
    res.status(502).json({ ok: false, error: String(e && e.message ? e.message : e) });
  }
});

await init();
server.listen(PORT, "127.0.0.1", () =>
  console.log(`photon sidecar on 127.0.0.1:${PORT} ready=${!!app}${initError ? " (" + initError + ")" : ""}`));

process.on("SIGINT", async () => { try { await app?.stop(); } finally { process.exit(0); } });
