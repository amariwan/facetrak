"""Dependency-free Linux system metrics (stdlib only).

All readers accept override paths so tests can inject fixtures; on
non-Linux systems they return None fields instead of raising.
"""

import os
import time


def cpu_temp_c(sysfs: str = "/sys/class/thermal/thermal_zone0/temp") -> float | None:
    try:
        with open(sysfs) as f:
            return int(f.read().strip()) / 1000.0
    except (OSError, ValueError):
        return None


def load_avg() -> tuple[float, float, float] | None:
    try:
        return os.getloadavg()
    except OSError:
        return None


def mem_info(proc: str = "/proc/meminfo") -> dict | None:
    try:
        out = {}
        with open(proc) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2 and parts[0].endswith(":"):
                    try:
                        out[parts[0][:-1].lower()] = int(parts[1])
                    except ValueError:
                        pass
        if "memtotal" not in out:
            return None
        return out
    except OSError:
        return None


def self_rss_mb(status: str = "/proc/self/status") -> float | None:
    try:
        with open(status) as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    try:  # fallback: macOS / non-proc systems (ru_maxrss units differ)
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if rss > 0:
            import sys

            return rss / (1024.0 * 1024.0) if sys.platform == "darwin" else rss / 1024.0
    except (ImportError, OSError, ValueError):
        pass
    return None


def throttled(vcgencmd: str = "/usr/bin/vcgencmd") -> str | None:
    """Pi under-voltage/thermal throttle state (None when unavailable)."""
    import subprocess

    try:
        out = subprocess.run(
            [vcgencmd, "get_throttled"],
            capture_output=True,
            timeout=2,
            check=False,
            text=True,
        )
        if out.returncode == 0 and "throttled=" in out.stdout:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def system_status() -> dict:
    mem = mem_info()
    load = load_avg()
    status = {
        "ts": time.time(),
        "cpu_temp_c": cpu_temp_c(),
        "load_1_5_15": list(load) if load else None,
        "self_rss_mb": (round(self_rss_mb(), 1) if self_rss_mb() is not None else None),
        "throttled": throttled(),
    }
    if mem:
        total = mem.get("memtotal", 0)
        avail = mem.get("memavailable", mem.get("memfree", 0))
        status["mem_total_kb"] = total
        status["mem_used_pct"] = (
            round(100.0 * (total - avail) / total, 1) if total else None
        )
    return status
