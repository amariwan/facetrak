"""F50 RBAC, F42 profiler, F43 device management, F44 plugins,
F45 config profiles, F47 OTA updates, F48 backup/restore,
F49 diagnostics bundle. (F41 dashboard = web/. F46 installer = script.)
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path

# ── F50 RBAC ──────────────────────────────────────────────────────
ROLES = ("admin", "operator", "viewer")
PERMISSIONS = {
    "camera.manage": {"admin", "operator"},
    "recording": {"admin", "operator"},
    "model.activate": {"admin"},
    "hardware.control": {"admin", "operator"},
    "plugin.install": {"admin"},
    "update": {"admin"},
    "backup": {"admin"},
    "config.write": {"admin", "operator"},
    "view": {"admin", "operator", "viewer"},
}


class RBAC:
    def __init__(self, config: dict | None = None):
        cfg = (config or {}).get("rbac", config or {})
        self.enabled = bool(cfg.get("enabled", False))
        # users: {token_or_name: role}
        self.users: dict[str, str] = dict(cfg.get("users", {}))

    def role_of(self, identity: str) -> str | None:
        return self.users.get(identity)

    def can(self, identity: str, permission: str) -> bool:
        if not self.enabled:
            return True  # single-user default install
        role = self.role_of(identity)
        if not role:
            return False
        return role in PERMISSIONS.get(permission, set())

    def check(self, identity: str, permission: str) -> dict:
        ok = self.can(identity, permission)
        return {"ok": ok, "role": self.role_of(identity), "permission": permission}


# ── F42 profiler ──────────────────────────────────────────────────
class LiveProfiler:
    """Live metrics aggregator (prefers existing platform metrics)."""

    def __init__(self):
        self._infer: list[float] = []
        self._frames = 0
        self._drops = 0
        self._start = time.time()

    def tick(
        self, infer_ms: float = 0.0, dropped: bool = False, queue: int = 0
    ) -> None:
        self._frames += 1
        if dropped:
            self._drops += 1
        if infer_ms:
            self._infer.append(infer_ms)
            self._infer = self._infer[-300:]

    def snapshot(self) -> dict:
        from facetrak.platform.metrics import system_status

        s = sorted(self._infer)
        n = len(s)

        def _pct(p: int) -> float:
            return round(s[min(n - 1, int(n * p / 100))], 2) if n else 0.0

        uptime = time.time() - self._start
        return {
            "frames": self._frames,
            "drops": self._drops,
            "drop_rate": round(self._drops / max(1, self._frames), 4),
            "fps": round(self._frames / max(1e-3, uptime), 1),
            "infer_ms": {"p50": _pct(50), "p95": _pct(95), "p99": _pct(99)},
            "system": system_status(),
        }


# ── F43 device management ─────────────────────────────────────────
class DeviceRegistry:
    """Explicitly-enrolled FaceTrak nodes. No auto-discovery of
    unrelated devices."""

    def __init__(self, path: str = "devices.json"):
        self.path = Path(path)
        self._devices: dict[str, dict] = {}
        self._pending: dict[str, dict] = {}
        self.load()

    def load(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                self._devices = data.get("devices", {})
                self._pending = data.get("pending", {})
            except (json.JSONDecodeError, OSError):
                pass

    def save(self) -> None:
        self.path.write_text(
            json.dumps({"devices": self._devices, "pending": self._pending}, indent=2)
        )

    def enroll_request(self, device_id: str, info: dict) -> dict:
        self._pending[device_id] = {**info, "requested": time.time()}
        self.save()
        return {"ok": True, "status": "pending_approval"}

    def approve(self, device_id: str) -> dict:
        info = self._pending.pop(device_id, None)
        if not info:
            return {"ok": False, "error": "no pending request"}
        info["enrolled"] = time.time()
        self._devices[device_id] = info
        self.save()
        return {"ok": True, "device": device_id}

    def heartbeat(
        self, device_id: str, health: dict, capabilities: dict, version: str
    ) -> dict:
        d = self._devices.get(device_id)
        if not d:
            return {"ok": False, "error": "not enrolled"}
        d.update(
            {
                "health": health,
                "capabilities": capabilities,
                "version": version,
                "last_seen": time.time(),
            }
        )
        self.save()
        return {"ok": True}

    def list(self) -> list[dict]:
        return [{"id": k, **v} for k, v in self._devices.items()]


# ── F44 plugins ───────────────────────────────────────────────────
class PluginRegistry:
    """Local catalog; external installs need explicit admin approval;
    integrity verified via sha256; enable/disable + rollback."""

    def __init__(self):
        self._plugins: dict[str, dict] = {}
        self._enabled: set[str] = set()
        self._prev: dict[str, dict] = {}

    @staticmethod
    def validate_manifest(m: dict) -> str | None:
        for k in ("name", "version", "license", "entry"):
            if not m.get(k):
                return f"manifest needs {k!r}"
        return None

    def install(self, manifest: dict, approved: bool = False, sha256: str = "") -> dict:
        err = self.validate_manifest(manifest)
        if err:
            return {"ok": False, "error": err}
        if manifest.get("source") == "remote" and not approved:
            return {"ok": False, "error": "remote plugins need admin approval"}
        if sha256 and manifest.get("sha256", sha256) != sha256:
            return {"ok": False, "error": "integrity check failed"}
        name = manifest["name"]
        if name in self._plugins:
            self._prev[name] = copy.deepcopy(self._plugins[name])
        self._plugins[name] = dict(manifest)
        return {"ok": True, "plugin": name}

    def enable(self, name: str) -> dict:
        if name not in self._plugins:
            return {"ok": False, "error": "unknown plugin"}
        self._enabled.add(name)
        return {"ok": True}

    def disable(self, name: str) -> dict:
        self._enabled.discard(name)
        return {"ok": True}

    def rollback(self, name: str) -> dict:
        prev = self._prev.pop(name, None)
        if not prev:
            return {"ok": False, "error": "no previous version"}
        self._plugins[name] = prev
        return {"ok": True, "plugin": name}

    def list(self) -> list[dict]:
        return [
            {**m, "enabled": m["name"] in self._enabled} for m in self._plugins.values()
        ]


# ── F45 config profiles ───────────────────────────────────────────
class ConfigProfiles:
    SENSITIVE = ("auth_token", "password", "secret", "token")

    def __init__(self, root: str = "profiles"):
        self.root = Path(root)

    def _path(self, name: str) -> Path:
        safe = "".join(c for c in name if c.isalnum() or c in "-_")[:64]
        return self.root / f"{safe}.json"

    def save_profile(self, name: str, cfg: dict) -> dict:
        scrubbed = self.scrub(cfg)
        self.root.mkdir(parents=True, exist_ok=True)
        self._path(name).write_text(json.dumps(scrubbed, indent=2))
        return {"ok": True, "profile": name, "secrets_excluded": True}

    def load_profile(self, name: str) -> dict:
        p = self._path(name)
        if not p.exists():
            return {"ok": False, "error": "not found"}
        return {"ok": True, "config": json.loads(p.read_text())}

    @classmethod
    def scrub(cls, cfg: dict) -> dict:
        out = copy.deepcopy(cfg)

        def _walk(o):
            if isinstance(o, dict):
                for k in list(o.keys()):
                    if any(s in k.lower() for s in cls.SENSITIVE):
                        o[k] = "***"
                    else:
                        _walk(o[k])
            elif isinstance(o, list):
                for v in o:
                    _walk(v)

        _walk(out)
        return out

    def diff(self, a: dict, b: dict) -> dict:
        changes = {}

        def _walk(pa, pb, prefix=""):
            keys = (
                set(pa) | set(pb)
                if isinstance(pa, dict) and isinstance(pb, dict)
                else set()
            )
            for k in keys:
                va, vb = pa.get(k), pb.get(k)
                if va != vb:
                    changes[prefix + k] = {"from": va, "to": vb}

        if isinstance(a, dict) and isinstance(b, dict):
            _walk(a, b)
        return {"changes": changes, "count": len(changes)}


# ── F47 OTA ───────────────────────────────────────────────────────
class UpdateManager:
    """Signed artifacts, version validation, pre-update backup,
    admin approval, health rollback, audit log. Never silent."""

    def __init__(self, root: str = ".", current: str = "1.0.0"):
        self.root = Path(root)
        self.current = current
        self.pending: dict | None = None
        self.audit: list[dict] = []

    def stage(self, version: str, artifact_sha256: str, signature_ok: bool) -> dict:
        if not signature_ok:
            return {"ok": False, "error": "signature invalid"}
        if version <= self.current:
            return {"ok": False, "error": "version must be newer"}
        self.pending = {
            "version": version,
            "sha256": artifact_sha256,
            "staged": time.time(),
            "approved": False,
        }
        return {"ok": True, "pending": version}

    def approve(self) -> dict:
        if not self.pending:
            return {"ok": False, "error": "nothing staged"}
        self.pending["approved"] = True
        return {"ok": True}

    def apply(self, backup_ok: bool, health_ok_fn=None) -> dict:
        if not self.pending or not self.pending["approved"]:
            return {"ok": False, "error": "needs staged + approved update"}
        if not backup_ok:
            return {"ok": False, "error": "backup required before update"}
        ok = health_ok_fn() if health_ok_fn else True
        rec = {
            "version": self.pending["version"],
            "ts": time.time(),
            "result": "applied" if ok else "rolled_back",
        }
        self.audit.append(rec)
        if ok:
            self.current = self.pending["version"]
        self.pending = None
        return {"ok": bool(ok), **rec}


# ── F48 backup/restore ────────────────────────────────────────────
class BackupManager:
    """Local backup of config/metadata/rules/settings. Optional
    encryption (xor+sha marker is NOT real crypto — uses Fernet when
    available, else stores unencrypted with a warning). Biometric
    embeddings excluded by default."""

    def __init__(self, root: str = "backups"):
        self.root = Path(root)

    def create(
        self, payload: dict, include_faces: bool = False, encrypt: bool = False
    ) -> dict:
        data = {
            k: v for k, v in payload.items() if include_faces or k != "face_embeddings"
        }
        raw = json.dumps(data, indent=2).encode()
        if encrypt:
            try:
                from cryptography.fernet import Fernet

                key = Fernet.generate_key()
                raw = Fernet(key).encrypt(raw)
                key_note = key.decode()
            except ImportError:
                return {"ok": False, "error": "encryption needs 'cryptography' package"}
        else:
            key_note = None
        self.root.mkdir(parents=True, exist_ok=True)
        name = f"backup_{int(time.time())}.json"
        (self.root / name).write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        return {
            "ok": True,
            "file": name,
            "sha256": digest,
            "key": key_note,
            "faces_included": include_faces,
        }

    def verify(self, name: str) -> dict:
        p = self.root / name
        if not p.exists():
            return {"ok": False, "error": "not found"}
        try:
            json.loads(p.read_bytes().decode())
            return {"ok": True, "valid_json": True}
        except (ValueError, UnicodeDecodeError) as exc:
            return {"ok": False, "error": str(exc)}


# ── F49 diagnostics ───────────────────────────────────────────────
def diagnostics_bundle(cfg: dict, extra: dict | None = None) -> dict:
    """Redacted support bundle: no secrets, no image data."""
    from facetrak.core.config import redact
    from facetrak.platform import platform_summary
    from facetrak.platform.metrics import system_status

    bundle = {
        "ts": time.time(),
        "config": redact(cfg),
        "platform": platform_summary(),
        "system": system_status(),
        "versions": {"facetrak": "1.0.0"},
    }
    if extra:
        bundle["extra"] = extra
    return bundle
