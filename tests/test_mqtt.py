"""MQTT publisher: anonymized schema, disabled-by-default, fake client."""

from facetrak.integrations.mqtt import MQTTPublisher, build_message


def test_build_message_strips_names_by_default():
    msg = build_message("zone_enter", {"zone": "door", "name": "amy", "track_id": 3})
    assert msg["source"] == "facetrak"
    assert msg["kind"] == "zone_enter"
    assert "name" not in msg["data"]
    assert msg["data"]["track_id"] == 3


def test_build_message_keeps_names_when_allowed():
    msg = build_message("zone_enter", {"name": "amy"}, include_names=True)
    assert msg["data"]["name"] == "amy"


class FakeClient:
    def __init__(self):
        self.published = []
        self.connected_to = None

    def connect(self, host, port, keepalive=30):
        self.connected_to = (host, port)

    def loop_start(self):
        pass

    def publish(self, topic, payload, qos=0, retain=False):
        self.published.append((topic, payload))

    def loop_stop(self):
        pass

    def disconnect(self):
        pass


def _cfg(**kw):
    base = {
        "enabled": True,
        "host": "127.0.0.1",
        "port": 1883,
        "topic_prefix": "facetrak",
        "username": "",
        "password": "",
        "tls": False,
        "include_names": False,
    }
    base.update(kw)
    return {"mqtt": base}


def test_publish_flow_with_fake_client():
    fake = FakeClient()
    pub = MQTTPublisher(_cfg(), client=fake)
    assert pub.start()
    assert pub.connected
    assert pub.publish("zone_enter", {"zone": "door", "name": "amy"})
    topic, payload = fake.published[0]
    assert topic == "facetrak/zone_enter"
    assert '"amy"' not in payload  # anonymized by default
    pub.stop()
    assert not pub.connected


def test_disabled_publisher_is_noop():
    pub = MQTTPublisher({"mqtt": {"enabled": False}}, client=FakeClient())
    assert not pub.start()
    assert not pub.publish("zone_enter", {})


def test_connect_failure_is_safe():
    class BadClient(FakeClient):
        def connect(self, host, port, keepalive=30):
            raise ConnectionRefusedError("no broker")

    pub = MQTTPublisher(_cfg(), client=BadClient())
    assert not pub.start()
    assert not pub.publish("zone_enter", {})
