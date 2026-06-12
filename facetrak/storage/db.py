import datetime
import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path("facetrak.db")


def _ts() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


@contextmanager
def _conn():
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init():
    with _conn() as con:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS presence (
                id        INTEGER PRIMARY KEY,
                ts        TEXT NOT NULL,
                event     TEXT NOT NULL,
                name      TEXT NOT NULL,
                track_id  INTEGER,
                duration_s REAL,
                blinks    INTEGER
            );
            CREATE TABLE IF NOT EXISTS emotion_log (
                id        INTEGER PRIMARY KEY,
                ts        TEXT NOT NULL,
                name      TEXT,
                emotion   TEXT,
                smile     REAL,
                attentive INTEGER,
                yaw       REAL,
                pitch     REAL
            );
            CREATE TABLE IF NOT EXISTS crowd_samples (
                id    INTEGER PRIMARY KEY,
                ts    TEXT NOT NULL,
                count INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_presence_ts   ON presence(ts);
            CREATE INDEX IF NOT EXISTS idx_emotion_ts    ON emotion_log(ts);
            CREATE INDEX IF NOT EXISTS idx_crowd_ts      ON crowd_samples(ts);
        """)


def log_presence(event: str, name: str, track_id: int,
                 duration_s: float | None = None, blinks: int | None = None):
    with _conn() as con:
        con.execute(
            "INSERT INTO presence(ts,event,name,track_id,duration_s,blinks) "
            "VALUES(?,?,?,?,?,?)",
            (_ts(), event, name, track_id, duration_s, blinks))


def log_emotion(name: str | None, emotion: str, smile: float,
                attentive: bool, yaw: float, pitch: float):
    with _conn() as con:
        con.execute(
            "INSERT INTO emotion_log(ts,name,emotion,smile,attentive,yaw,pitch) "
            "VALUES(?,?,?,?,?,?,?)",
            (_ts(), name, emotion, round(smile, 3),
             int(attentive), round(yaw, 1), round(pitch, 1)))


def log_crowd(count: int):
    with _conn() as con:
        con.execute("INSERT INTO crowd_samples(ts,count) VALUES(?,?)",
                    (_ts(), count))


def query_presence(limit: int = 50) -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM presence ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


def query_emotions(limit: int = 200) -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM emotion_log ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


def query_crowd(limit: int = 3600) -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM crowd_samples ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


def crowd_summary() -> dict:
    with _conn() as con:
        row = con.execute(
            "SELECT COUNT(*) as samples, MAX(count) as peak, "
            "AVG(count) as avg FROM crowd_samples"
        ).fetchone()
        appearances = con.execute(
            "SELECT COUNT(*) FROM presence WHERE event='appeared'"
        ).fetchone()[0]
    if row["samples"] == 0:
        return {"peak": 0, "avg": 0.0, "total_appearances": appearances}
    return {
        "peak": row["peak"],
        "avg": round(row["avg"], 2),
        "total_appearances": appearances,
    }


# ── PresenceLog (formerly events.py) ──

class PresenceLog:
    def appeared(self, name: str | None, track_id: int):
        log_presence("appeared", name or "unknown", track_id)

    def left(self, name: str | None, track_id: int,
             duration: float, blinks: int = 0):
        log_presence("left", name or "unknown", track_id,
                     round(duration, 1), blinks)

    @staticmethod
    def tail(limit: int = 50) -> list[dict]:
        return query_presence(limit)
