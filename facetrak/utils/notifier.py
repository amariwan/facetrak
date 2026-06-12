import logging
import subprocess
import datetime

logger = logging.getLogger(__name__)


class Notifier:
    def __init__(self):
        self._notified: set[str] = set()

    def notify(self, name: str):
        if name in self._notified:
            return
        self._notified.add(name)
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        safe = name.replace("\\", "\\\\").replace('"', '""')
        try:
            subprocess.run([
                "osascript", "-e",
                f'display notification "{safe} detected at {ts}" '
                f'with title "FaceTrak" sound name "Tink"'
            ], timeout=2, capture_output=True, check=False)
        except (OSError, subprocess.SubprocessError) as exc:
            logger.debug("Notification failed: %s", exc)

    def reset(self):
        self._notified.clear()
