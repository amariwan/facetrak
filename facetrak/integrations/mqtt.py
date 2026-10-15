"""Optional MQTT publisher for anonymized FaceTrak events.

No hard dependency: the client module (paho-mqtt) is imported lazily and
only when enabled. For tests, any object with connect/loop_start/publish/
disconnect methods can be injected.
"""

import json
import logging
import time

logger = logging.getLogger(__name__)


def build_message(kind: str, payload: dict, include_names: bool = False) -> dict:
    """Standardized event schema. Names only when explicitly allowed."""
    body = dict(payload)
    if not include_names:
        body.pop("name", None)
        body.pop("names", None)
    return {"source": "facetrak", "kind": kind, "ts": time.time(), "data": body}


class MQTTPublisher:
    def __init__(self, cfg: dict, client=None):
        m = cfg.get("mqtt", {})
        self.enabled = bool(m.get("enabled", False))
        self.host = m.get("host", "127.0.0.1")
        self.port = int(m.get("port", 1883))
        self.prefix = str(m.get("topic_prefix", "facetrak")).strip("/") or "facetrak"
        self.username = m.get("username", "")
        self.password = m.get("password", "")
        self.tls = bool(m.get("tls", False))
        self.include_names = bool(m.get("include_names", False))
        self._client = client
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    def start(self) -> bool:
        if not self.enabled or self._connected:
            return self._connected
        try:
            if self._client is None:
                import paho.mqtt.client as mqtt  # lazy optional dep

                self._client = mqtt.Client()
                if self.username:
                    self._client.username_pw_set(self.username, self.password)
                if self.tls:
                    self._client.tls_set()
                self._client.reconnect_delay_set(min_delay=1, max_delay=30)
            self._client.connect(self.host, self.port, keepalive=30)
            self._client.loop_start()
            self._connected = True
            logger.info("MQTT connected to %s:%d", self.host, self.port)
        except Exception as exc:
            logger.warning("MQTT connect failed (%s:%d): %s", self.host, self.port, exc)
            self._connected = False
        return self._connected

    def publish(self, kind: str, payload: dict) -> bool:
        if not self.enabled or not self._connected or self._client is None:
            return False
        msg = build_message(kind, payload, self.include_names)
        try:
            self._client.publish(
                f"{self.prefix}/{kind}", json.dumps(msg), qos=0, retain=False
            )
            return True
        except Exception as exc:
            logger.warning("MQTT publish failed: %s", exc)
            self._connected = False
            return False

    def stop(self):
        try:
            if self._client is not None and self._connected:
                self._client.loop_stop()
                self._client.disconnect()
        except Exception:
            pass
        finally:
            self._connected = False
