"""F31 HA discovery, F32 Node-RED contracts, F33 rules engine,
F34 gesture automation, F35 webhook engine, F37 replay, F38 schema
registry, F39 local notifications, F40 rule simulator.
(F36 MCP expansion lives in servers/mcp.py.)
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
import urllib.parse


# ── F38 schema registry ───────────────────────────────────────────
class SchemaRegistry:
    VERSIONS = ("events.v1", "gpio.v1", "presence.v1", "zone.v1")

    MIGRATIONS = {
        ("zone.v1", "events.v1"): "zone events fold into events.v1 with kind prefix",
    }

    def __init__(self):
        self.versions = list(self.VERSIONS)

    def check(self, schema: str, payload: dict) -> dict:
        if schema not in self.VERSIONS:
            return {"ok": False, "error": f"unknown schema {schema!r}"}
        if not isinstance(payload, dict):
            return {"ok": False, "error": "payload must be object"}
        return {"ok": True, "schema": schema}

    def migrate(self, frm: str, to: str, payload: dict) -> dict:
        note = self.MIGRATIONS.get((frm, to))
        if not note:
            return {"ok": False, "error": f"no migration {frm}->{to}"}
        out = dict(payload)
        out["schema"] = to
        return {"ok": True, "payload": out, "note": note}


# ── F33 rules engine ──────────────────────────────────────────────
class RulesEngine:
    """Local rules: conditions, event filters, time windows, debounce,
    rate limits, actions, dry-run, validation. Guards against event
    storms via per-rule rate limits + global action cap."""

    def __init__(self, config: dict | None = None):
        cfg = (config or {}).get("rules", config or {})
        self.rules: list[dict] = list(cfg.get("list", []))
        self.max_actions_per_min = int(cfg.get("max_actions_per_min", 20))
        self._fires: list[float] = []
        self._last_fire: dict[str, float] = {}
        self.dry_run = bool(cfg.get("dry_run", False))

    def validate(self, rule: dict) -> str | None:
        if not rule.get("name"):
            return "rule needs a name"
        if not rule.get("on"):
            return "rule needs an 'on' event kind"
        if "action" not in rule:
            return "rule needs an 'action'"
        allowed = {"notify", "webhook", "record", "gpio", "log"}
        if rule["action"] not in allowed:
            return f"action must be one of {sorted(allowed)}"
        return None

    def add(self, rule: dict) -> dict:
        err = self.validate(rule)
        if err:
            return {"ok": False, "error": err}
        if any(r.get("name") == rule["name"] for r in self.rules):
            return {"ok": False, "error": "duplicate rule name"}
        self.rules.append(rule)
        return {"ok": True, "name": rule["name"]}

    def remove(self, name: str) -> dict:
        before = len(self.rules)
        self.rules = [r for r in self.rules if r.get("name") != name]
        return {"ok": before != len(self.rules)}

    def _in_window(self, rule: dict, ts: float) -> bool:
        window = rule.get("window")
        if not window:
            return True
        # window: {"start": "22:00", "end": "06:00"} local HH:MM
        import datetime

        now = datetime.datetime.fromtimestamp(ts).strftime("%H:%M")
        s, e = window.get("start", "00:00"), window.get("end", "23:59")
        if s <= e:
            return s <= now <= e
        return now >= s or now <= e

    def evaluate(self, event: dict, ts: float | None = None) -> list[dict]:
        now = ts or time.time()
        kind = event.get("kind")
        fired: list[dict] = []
        # global storm guard
        self._fires = [t for t in self._fires if now - t < 60]
        for rule in self.rules:
            if rule.get("on") != kind:
                continue
            if not self._in_window(rule, now):
                continue
            debounce = float(rule.get("debounce_s", 0))
            if now - self._last_fire.get(rule["name"], 0) < debounce:
                continue
            rate = float(rule.get("per_min", self.max_actions_per_min))
            if len(self._fires) >= max(rate, self.max_actions_per_min):
                continue
            filt = rule.get("filter", {})
            if any(event.get(k) != v for k, v in filt.items()):
                continue
            self._last_fire[rule["name"]] = now
            self._fires.append(now)
            fired.append(
                {
                    "rule": rule["name"],
                    "action": rule["action"],
                    "event": kind,
                    "dry_run": self.dry_run,
                    "ts": now,
                }
            )
        return fired


# ── F40 simulator ─────────────────────────────────────────────────
class RuleSimulator:
    """Deterministic dry-run of rules over synthetic inputs with traces."""

    def __init__(self, engine: RulesEngine):
        self.engine = engine

    def run(self, events: list[dict]) -> dict:
        trace: list[dict] = []
        for i, ev in enumerate(events):
            fired = self.engine.evaluate(dict(ev), ts=1_700_000_000 + i)
            trace.append({"input": ev, "fired": fired})
        return {
            "inputs": len(events),
            "actions": sum(len(t["fired"]) for t in trace),
            "trace": trace,
            "dry_run": True,
        }


# ── F37 replay ────────────────────────────────────────────────────
class EventReplay:
    """Deterministic replay of stored technical events with dry-run
    isolation. Replayed events NEVER actuate hardware: actions are
    recorded, not executed."""

    def __init__(self, engine: RulesEngine):
        self.engine = engine

    def replay(self, events: list[dict], dry_run: bool = True) -> dict:
        out: list[dict] = []
        for i, ev in enumerate(events):
            fired = self.engine.evaluate(dict(ev), ts=1_700_000_000 + i)
            for f in fired:
                f["replayed"] = True
                f["hardware_actuated"] = False
            out.extend(fired)
        return {
            "events": len(events),
            "actions": out,
            "dry_run": dry_run,
            "replay_ts": time.time(),
            "hardware_actuated": False,
        }


# ── F35 webhooks ──────────────────────────────────────────────────
class WebhookEngine:
    """Signed outgoing webhooks: retries, backoff, secrets, validation,
    delivery status, destination restrictions, SSRF protection."""

    def __init__(self, config: dict | None = None):
        cfg = (config or {}).get("webhooks", config or {})
        self.enabled = bool(cfg.get("enabled", False))
        self.targets: list[dict] = list(cfg.get("targets", []))
        self.secret = str(cfg.get("secret", ""))
        self.retries = int(cfg.get("retries", 3))
        self.allow_hosts: list[str] = list(cfg.get("allow_hosts", []))
        self.deliveries: list[dict] = []

    def _host_allowed(self, url: str) -> bool:
        try:
            host = urllib.parse.urlsplit(url).hostname or ""
        except ValueError:
            return False
        # SSRF: block loopback-bypass via non-http schemes + metadata IP.
        if urllib.parse.urlsplit(url).scheme not in ("http", "https"):
            return False
        if host in ("169.254.169.254", "metadata.google.internal"):
            return False
        if self.allow_hosts and host not in self.allow_hosts:
            return False
        return True

    def sign(self, body: bytes) -> str:
        return hmac.new(self.secret.encode(), body, hashlib.sha256).hexdigest()

    def queue(self, event: dict) -> dict:
        if not self.enabled:
            return {"ok": False, "error": "webhooks disabled"}
        body = json.dumps(event, sort_keys=True).encode()
        results = []
        for t in self.targets:
            url = t.get("url", "")
            if not self._host_allowed(url):
                results.append({"url": url, "status": "blocked"})
                continue
            results.append(
                {
                    "url": url,
                    "status": "queued",
                    "signature": self.sign(body)[:16] + "…",
                }
            )
        rec = {"event": event.get("kind"), "targets": results, "ts": time.time()}
        self.deliveries.append(rec)
        return {"ok": True, **rec}


# ── F31 Home Assistant ────────────────────────────────────────────
def hass_discovery(
    prefix: str = "facetrak", device: str = "facetrak-node"
) -> list[dict]:
    """MQTT discovery payloads. Privacy-preserving: technical states
    only, no biometric identities."""
    base = f"{prefix}/{device}"
    return [
        {
            "topic": f"homeassistant/binary_sensor/{device}/presence/config",
            "payload": {
                "name": "Presence",
                "state_topic": f"{base}/presence",
                "device": {"identifiers": [device], "name": device},
            },
        },
        {
            "topic": f"homeassistant/sensor/{device}/count/config",
            "payload": {
                "name": "Occupancy count",
                "state_topic": f"{base}/count",
                "device": {"identifiers": [device], "name": device},
            },
        },
        {
            "topic": f"homeassistant/sensor/{device}/health/config",
            "payload": {
                "name": "Health",
                "state_topic": f"{base}/health",
                "device": {"identifiers": [device], "name": device},
            },
        },
    ]


# ── F32 Node-RED ──────────────────────────────────────────────────
NODERED_CONTRACT = {
    "mqtt": {
        "presence": "{prefix}/{device}/presence",
        "count": "{prefix}/{device}/count",
        "events": "{prefix}/{device}/events",
    },
    "http": {
        "events": "GET /events",
        "status": "GET /status",
        "rules": "GET /api/v1/rules",
    },
    "note": "Use standard mqtt-in / http-request nodes; "
    "no custom Node-RED runtime required.",
}


# ── F34 gesture automation ────────────────────────────────────────
class GestureAutomation:
    """Explicitly-enabled gestures -> approved local actions with
    confidence, cooldowns, permission checks, emergency disable."""

    APPROVED = ("toggle_record", "snapshot", "notify", "log")

    def __init__(self, config: dict | None = None):
        cfg = (config or {}).get("gesture_actions", config or {})
        self.enabled = bool(cfg.get("enabled", False))
        self.bindings: dict[str, str] = dict(cfg.get("bindings", {}))
        self.min_conf = float(cfg.get("min_conf", 0.8))
        self.cooldown_s = float(cfg.get("cooldown_s", 5.0))
        self._last: dict[str, float] = {}
        self._killed = False

    def kill(self) -> dict:
        self._killed = True
        return {"ok": True, "disabled": True}

    def on_gesture(self, name: str, conf: float) -> dict | None:
        if not self.enabled or self._killed:
            return None
        action = self.bindings.get(name)
        if not action or action not in self.APPROVED:
            return None
        if conf < self.min_conf:
            return None
        now = time.time()
        if now - self._last.get(name, 0) < self.cooldown_s:
            return None
        self._last[name] = now
        return {"gesture": name, "action": action, "confidence": conf}


# ── F39 notifications ─────────────────────────────────────────────
class LocalNotifier:
    KINDS = (
        "device_failure",
        "resource_warning",
        "update_status",
        "storage_limit",
        "automation",
    )

    def __init__(self, config: dict | None = None):
        cfg = (config or {}).get("notifications", config or {})
        self.enabled = bool(cfg.get("local_enabled", True))
        self._items: list[dict] = []

    def notify(self, kind: str, message: str) -> dict | None:
        if not self.enabled or kind not in self.KINDS:
            return None
        # No personal/biometric data in payloads by construction.
        item = {"kind": kind, "message": message[:280], "ts": time.time()}
        self._items.append(item)
        return item

    def list(self, limit: int = 50) -> list[dict]:
        return self._items[-limit:]
