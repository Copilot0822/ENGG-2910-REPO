from dataclasses import asdict, dataclass, fields
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Settings:
    data_dir: str = str(ROOT / "data")
    allowed_sports: tuple[str, ...] = ("ride", "gravelride", "mountainbikeride", "ebikeride")
    include_ebikes: bool = False
    moving_threshold_mps: float = 0.8
    max_speed_mps: float = 35.0
    gap_seconds: float = 30.0
    half_life_days: float = 120.0
    optimize_half_life: bool = False
    validation_rides: int = 8
    osm_provider: str = "offline"
    overpass_url: str = "https://overpass-api.de/api/interpreter"
    osm_tile_degrees: float = 0.02
    osm_max_tiles: int = 12
    osm_min_interval_s: float = 3.0
    osm_cache_days: int = 180
    local_osm_path: str = ""
    local_osm_paths: list[str] | None = None
    valhalla_url: str = ""
    map_match_radius_m: float = 40.0
    osm_workers: int = 3
    comparison_workers: int = 3
    weather_enabled: bool = True
    weather_offline: bool = False
    weather_radius_km: float = 65.0
    weather_candidates: int = 4
    weather_min_interval_s: float = 1.0
    weather_max_requests: int = 700
    max_archive_bytes: int = 2_000_000_000
    max_uncompressed_bytes: int = 8_000_000_000
    max_entry_bytes: int = 128_000_000
    max_archive_entries: int = 50_000
    max_points: int = 1_000_000

    @classmethod
    def load(cls, path: Path | None = None):
        path = path or ROOT / "config.local.json"
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        names = {f.name for f in fields(cls)}
        result = cls(**{k: v for k, v in data.items() if k in names})
        if os.environ.get("CYCLING_DATA_DIR"):
            result.data_dir = os.environ["CYCLING_DATA_DIR"]
        return result

    def save(self, path: Path | None = None):
        (path or ROOT / "config.local.json").write_text(
            json.dumps(asdict(self), indent=2), encoding="utf-8"
        )
