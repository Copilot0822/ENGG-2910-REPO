"""SQLite index + compressed Parquet traces; atomic writes, incremental imports."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
import uuid

import pandas as pd


class Store:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        for name in ("points", "segments", "models", "reports"):
            (self.root / name).mkdir(exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS rides (
                    id TEXT PRIMARY KEY, fingerprint TEXT UNIQUE NOT NULL,
                    metadata TEXT NOT NULL, points_path TEXT NOT NULL, segments_path TEXT
                );
                CREATE TABLE IF NOT EXISTS osm_tiles (
                    key TEXT PRIMARY KEY, fetched_at REAL NOT NULL, payload BLOB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.root / "cycling.sqlite", timeout=60)
        db.execute("PRAGMA journal_mode=WAL")
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def contains(self, ride_id, fingerprint):
        with self.connect() as db:
            return (
                db.execute(
                    "SELECT 1 FROM rides WHERE id=? OR fingerprint=?", (str(ride_id), fingerprint)
                ).fetchone()
                is not None
            )

    def _frame_path(self, kind, ride_id):
        safe = hashlib.sha256(str(ride_id).encode()).hexdigest()[:32]
        return self.root / kind / f"{safe}.parquet"

    @staticmethod
    def _write_frame(frame, path):
        temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        try:
            frame.to_parquet(temporary, index=False, compression="zstd")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def add_ride(self, meta, fingerprint, points):
        path = self._frame_path("points", meta["id"])
        self._write_frame(points, path)
        with self.connect() as db:
            db.execute(
                "INSERT INTO rides VALUES (?,?,?,?,NULL)",
                (str(meta["id"]), fingerprint, json.dumps(meta, default=str), str(path)),
            )
        self.invalidate_model()

    def save_segments(self, ride_id, segments):
        path = self._frame_path("segments", ride_id)
        self._write_frame(segments, path)
        with self.connect() as db:
            db.execute("UPDATE rides SET segments_path=? WHERE id=?", (str(path), str(ride_id)))

    def update_metadata(self, ride_id, updates):
        with self.connect() as db:
            row = db.execute("SELECT metadata FROM rides WHERE id=?", (str(ride_id),)).fetchone()
            if row:
                meta = json.loads(row[0])
                meta.update(updates)
                db.execute("UPDATE rides SET metadata=? WHERE id=?", (json.dumps(meta), str(ride_id)))

    def rides(self):
        with self.connect() as db:
            records = db.execute("SELECT metadata FROM rides").fetchall()
        result = pd.DataFrame([json.loads(r[0]) for r in records])
        return result.sort_values("date").reset_index(drop=True) if len(result) else result

    def points(self, ride_id):
        with self.connect() as db:
            row = db.execute("SELECT points_path FROM rides WHERE id=?", (str(ride_id),)).fetchone()
        if not row:
            raise KeyError(ride_id)
        return pd.read_parquet(row[0])

    def segments(self, ride_ids=None):
        with self.connect() as db:
            rows = db.execute(
                "SELECT id, segments_path FROM rides WHERE segments_path IS NOT NULL"
            ).fetchall()
        selected = None if ride_ids is None else set(map(str, ride_ids))
        frames = [
            pd.read_parquet(path) for ride_id, path in rows if selected is None or ride_id in selected
        ]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def set_state(self, key, value):
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO state VALUES (?,?)", (key, json.dumps(value, default=str))
            )

    def get_state(self, key, default=None):
        with self.connect() as db:
            row = db.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def invalidate_model(self):
        self.set_state("model_stale", True)
        self.set_state("journey_stale", True)

    def model_path(self):
        return self.root / "models" / "cycling.joblib"
