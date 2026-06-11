"""Microphone audio event detection via sounddevice.

Runs in a background thread and emits events:
  - LOUD       : RMS level above loud_threshold
  - CLAP       : short transient spike (possible hand clap)
  - VOICE      : sustained audio in speech-frequency band
  - SILENCE    : extended quiet period after activity

Install: pip install sounddevice
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)

_SAMPLE_RATE   = 16000
_BLOCK_SIZE    = 512       # ~32 ms per block
_CHANNELS      = 1

_LOUD_DB       = -20.0     # dBFS threshold for LOUD event
_CLAP_DB       = -10.0     # sharp transient for clap
_CLAP_DECAY_S  = 0.15      # clap must fall back below LOUD within this time
_VOICE_HZ_LO   = 300
_VOICE_HZ_HI   = 3400
_VOICE_RATIO   = 0.35      # fraction of energy in voice band to flag VOICE
_SILENCE_S     = 2.0       # seconds of quiet before SILENCE event


class AudioEvent(str, Enum):
    LOUD    = "loud"
    CLAP    = "clap"
    VOICE   = "voice"
    SILENCE = "silence"


@dataclass
class AudioSample:
    event: AudioEvent
    rms_db: float
    timestamp: float = field(default_factory=time.time)


def _rms_db(block: np.ndarray) -> float:
    rms = np.sqrt(np.mean(block.astype(np.float32) ** 2))
    if rms < 1e-10:
        return -96.0
    return 20.0 * np.log10(rms / 32768.0)


def _voice_ratio(block: np.ndarray, sr: int) -> float:
    """Fraction of spectral energy in voice frequency band."""
    fft = np.abs(np.fft.rfft(block.astype(np.float32)))
    freqs = np.fft.rfftfreq(len(block), d=1.0 / sr)
    total = fft.sum() + 1e-10
    voice = fft[(freqs >= _VOICE_HZ_LO) & (freqs <= _VOICE_HZ_HI)].sum()
    return float(voice / total)


class AudioMonitor:
    """Continuously monitors the microphone and emits AudioSample events.

    Usage:
        monitor = AudioMonitor()
        monitor.start()
        event = monitor.poll()   # non-blocking, returns AudioSample | None
        monitor.stop()
    """

    def __init__(self, device: int | str | None = None,
                 loud_db: float = _LOUD_DB,
                 clap_db: float = _CLAP_DB,
                 voice_ratio: float = _VOICE_RATIO,
                 silence_s: float = _SILENCE_S):
        self._device     = device
        self._loud_db    = loud_db
        self._clap_db    = clap_db
        self._voice_ratio = voice_ratio
        self._silence_s  = silence_s

        self._queue: queue.Queue[AudioSample] = queue.Queue(maxsize=64)
        self._stop      = threading.Event()
        self._thread: threading.Thread | None = None
        self._enabled   = False

        self._last_active = 0.0
        self._prev_db     = -96.0

    def start(self) -> bool:
        try:
            import sounddevice  # noqa: F401 — just verify import
        except ImportError:
            logger.warning(
                "sounddevice not installed — audio disabled. "
                "Install with: pip install sounddevice"
            )
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._enabled = True
        logger.info("AudioMonitor started")
        return True

    def _run(self) -> None:
        import sounddevice as sd
        try:
            with sd.InputStream(
                device=self._device,
                samplerate=_SAMPLE_RATE,
                blocksize=_BLOCK_SIZE,
                channels=_CHANNELS,
                dtype="int16",
            ) as stream:
                while not self._stop.is_set():
                    block, _ = stream.read(_BLOCK_SIZE)
                    self._process(block[:, 0])
        except Exception as exc:
            logger.error("AudioMonitor stream error: %s", exc)
            self._enabled = False

    def _process(self, block: np.ndarray) -> None:
        db = _rms_db(block)
        now = time.time()

        # clap: current block very loud AND previous was quiet
        if db >= self._clap_db and self._prev_db < self._loud_db:
            self._emit(AudioEvent.CLAP, db)

        elif db >= self._loud_db:
            self._last_active = now
            vr = _voice_ratio(block, _SAMPLE_RATE)
            if vr >= self._voice_ratio:
                self._emit(AudioEvent.VOICE, db)
            else:
                self._emit(AudioEvent.LOUD, db)

        elif now - self._last_active > self._silence_s and self._last_active > 0:
            self._emit(AudioEvent.SILENCE, db)
            self._last_active = 0.0   # emit once per silence period

        self._prev_db = db

    def _emit(self, event: AudioEvent, db: float) -> None:
        sample = AudioSample(event=event, rms_db=round(db, 1))
        try:
            self._queue.put_nowait(sample)
        except queue.Full:
            pass  # drop oldest implicitly — queue is fixed size

    def poll(self) -> AudioSample | None:
        """Non-blocking: return next event or None."""
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def poll_all(self) -> list[AudioSample]:
        """Drain all pending events."""
        events: list[AudioSample] = []
        while True:
            e = self.poll()
            if e is None:
                break
            events.append(e)
        return events

    @property
    def enabled(self) -> bool:
        return self._enabled

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        self._enabled = False
