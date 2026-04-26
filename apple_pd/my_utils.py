"""
Apple Look Around scraper utilities — args, dataset loading, timing, helpers.

Dataset must include `panoid` and `build_id`. `lat` and `lon` are required so
we can look up camera_metadata via the z=17 coverage tile (cached).
"""
import argparse
import csv
import json
import os
import time

from rich import print


class timer:
    def __enter__(self):
        self.start = time.time()
        self.time_elapsed = None
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        hrs, rem = divmod(self.interval, 3600)
        mins, secs = divmod(rem, 60)
        self.time_elapsed = f"{int(hrs)}h {int(mins)}m {secs:.2f}s"
        return False


def open_dataset(path: str) -> list[dict]:
    """
    Load dataset from CSV or JSON.

    CSV: auto-detects columns (case-insensitive, ignores '_' / '-'):
         panoid (req), buildid (req), lat (req), lon (req),
         heading_deg (opt), country_code (opt), address_label (opt)

    JSON: list of dicts with the same keys, OR list of strings (panoids only —
          but those are useless without build_id, will fail at fetch time).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        with open(path, "r", encoding="utf-8-sig") as f:
            sample = f.read(4096)
            f.seek(0)
            delim = ";" if ";" in sample and sample.count(";") > sample.count(",") else ","
            reader = csv.DictReader(f, delimiter=delim)

            cols = {}
            if reader.fieldnames:
                for field in reader.fieldnames:
                    clean = field.lower().strip().replace("_", "").replace("-", "")
                    if clean == "panoid":
                        cols["panoid"] = field
                    elif clean in ("buildid", "build"):
                        cols["build_id"] = field
                    elif clean in ("lat", "latitude"):
                        cols["lat"] = field
                    elif clean in ("lon", "lng", "longitude"):
                        cols["lon"] = field
                    elif clean in ("headingdeg", "heading", "yaw"):
                        cols["heading_deg"] = field
                    elif clean == "countrycode":
                        cols["country_code"] = field
                    elif clean == "addresslabel":
                        cols["address_label"] = field

            if "panoid" not in cols:
                print(f"[red]Error: CSV missing 'panoid' column. Found: {reader.fieldnames}[/]")
                return []
            if "build_id" not in cols:
                print(f"[red]Error: CSV missing 'build_id' column. Found: {reader.fieldnames}[/]")
                return []
            if "lat" not in cols or "lon" not in cols:
                print(f"[red]Error: CSV needs 'lat' and 'lon' columns to look up metadata. Found: {reader.fieldnames}[/]")
                return []

            records = []
            for row in reader:
                panoid = row.get(cols["panoid"], "").strip()
                build = row.get(cols["build_id"], "").strip()
                if not panoid or not build:
                    continue
                rec = {"panoid": panoid, "build_id": build}
                try:
                    rec["lat"] = float(row[cols["lat"]])
                    rec["lon"] = float(row[cols["lon"]])
                except (ValueError, KeyError):
                    continue
                if "heading_deg" in cols and row.get(cols["heading_deg"]):
                    try:
                        rec["heading_deg"] = float(row[cols["heading_deg"]])
                    except ValueError:
                        pass
                if "country_code" in cols and row.get(cols["country_code"]):
                    rec["country_code"] = row[cols["country_code"]].strip()
                if "address_label" in cols and row.get(cols["address_label"]):
                    rec["address_label"] = row[cols["address_label"]].strip()
                records.append(rec)
            return records

    # JSON
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list) and data and isinstance(data[0], str):
        # Bare panoid list — still legal but reproject will fail without build_id
        return [{"panoid": p} for p in data]
    return data if isinstance(data, list) else []


def parse_args():
    p = argparse.ArgumentParser(
        description="Apple Look Around Panorama Downloader (Stage 2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Basic
    p.add_argument("--zoom", type=int, default=2, help="Zoom level 0-7 (Apple). 0=highest")
    p.add_argument("--dataset", type=str, required=True, help="CSV or JSON of pano records")
    p.add_argument("--max-threads", type=int, default=100, help="Max concurrent panos in flight")
    p.add_argument("--workers", type=int, default=15, help="ThreadPool workers for CPU work")
    p.add_argument("--limit", type=int, default=None, help="Cap number of panos to process")
    p.add_argument("--output", type=str, default=r"D:\GeoAxis\Hypervision\Output\Apple_Images",
                   help="Output directory")

    # Outputs
    p.add_argument("--directional-views", action="store_true", help="Extract perspective views")
    p.add_argument("--keep-pano", action="store_true", help="Save full equirect panorama to disk")

    # View options
    p.add_argument("--view-resolution", type=int, default=512)
    p.add_argument("--view-fov", type=float, default=90.0)
    p.add_argument("--num-views", type=int, default=6)
    p.add_argument("--view-offset", type=float, default=0.0,
                   help="Yaw offset (deg) applied to all views")
    p.add_argument("--global", dest="global_view", action="store_true",
                   help="Extract one random view per pano instead of num_views")
    p.add_argument("--aa-strength", type=float, default=0.8)
    p.add_argument("--interpolation", type=str, default="lanczos", choices=["cubic", "lanczos"])
    p.add_argument("--no-antialias", action="store_true")
    p.add_argument("--jpeg-quality", type=int, default=95)

    # Equirect quality
    p.add_argument("--max-equirect-w", type=int, default=4096,
                   help="Cap on equirect width before extraction (default 4096)")

    return p.parse_args()


def format_size(num_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} PB"
