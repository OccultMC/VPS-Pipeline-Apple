"""
OPTIMIZED Apple Look Around panorama processing pipeline.

Mirrors gsvpd/core_optimized.py:
  - Streaming pipeline (semaphore + asyncio.as_completed, no batch boundaries)
  - Chunk-of-10k dispatch to bound the queue
  - ThreadPoolExecutor for CPU-bound HEIC decode + reprojection + view remap
    (numpy/cv2/pillow_heif release the GIL → no IPC overhead)
  - Coverage-tile metadata cache (apple_fetch._COVERAGE_CACHE)

Per-pano flow:
  1. Async: ensure camera_metadata for this pano is in cache
  2. Async: fetch all 6 face HEICs in parallel (or 4 if no top/bottom needed)
  3. Thread: decode HEICs -> RGB arrays
  4. Thread: reproject 6 faces -> equirectangular RGB array
  5. Thread (optional): extract directional perspective views via cv2.remap
  6. Save outputs (panorama JPG and/or per-view JPGs)
"""
from __future__ import annotations

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Union

import aiohttp
from aiohttp import ClientTimeout
import cv2
import numpy as np
from rich import print

from streetlevel.lookaround.auth import Authenticator

from .apple_fetch import (
    decode_heic_bytes, fetch_all_faces, get_camera_metadata,
    prefetch_coverage_tiles_for_records, get_heic_decoder_name,
)
from .directional_views import DirectionalViewExtractor, DirectionalViewConfig
from .equirect_reproject import reproject_faces_to_equirect
from .progress_bar import ProgressBar


# Authenticator is cheap; one per process is plenty (faces will retry with a
# fresh one on 403 inside fetch_face).
_GLOBAL_AUTH = Authenticator()


def _process_in_thread(
    face_heics: List[Optional[bytes]],
    faces_meta: List[dict],
    panoid: str,
    config: dict,
    heading_deg: Optional[float],
) -> dict:
    """
    Heavy CPU work: decode HEICs, reproject to equirect, optionally extract
    perspective views, JPEG-encode everything.

    Runs inside a ThreadPoolExecutor — numpy/cv2/pillow_heif release the GIL.
    """
    result = {
        "success": False,
        "error": "",
        "size": (0, 0),
        "views": [],
        "view_filenames": [],
        "panorama_bytes": None,
        "timings": {},
    }

    try:
        # ── Decode side HEICs (top/bottom are skipped at fetch time) ──
        t0 = time.perf_counter()
        face_imgs = [decode_heic_bytes(h) for h in face_heics]
        if all(im is None for im in face_imgs[:4]):
            result["error"] = "all 4 side HEICs failed to decode"
            return result
        t1 = time.perf_counter()
        result["timings"]["  decode_heic"] = t1 - t0

        # ── Reproject to equirect ──
        out_w = config.get("max_equirect_w", 4096)
        equirect = reproject_faces_to_equirect(face_imgs, faces_meta, out_w=out_w)
        result["size"] = (equirect.shape[1], equirect.shape[0])
        t2 = time.perf_counter()
        result["timings"]["  reproject"] = t2 - t1

        # OpenCV needs BGR
        equirect_bgr = cv2.cvtColor(equirect, cv2.COLOR_RGB2BGR)

        # ── Optional: directional views ──
        if config.get("create_directional_views"):
            view_extractor = DirectionalViewExtractor()
            view_config = DirectionalViewConfig(
                output_resolution=config.get("view_resolution", 512),
                fov_degrees=config.get("view_fov", 90.0),
                num_views=config.get("num_views", 6),
                global_view=config.get("global_view", False),
                augment=False,
                target_yaw=heading_deg,
                antialias_strength=0.0 if config.get("no_antialias") else config.get("aa_strength", 0.8),
                interpolation=config.get("interpolation", "lanczos"),
                yaw_offset=config.get("view_offset", 0.0),
            )
            t3 = time.perf_counter()
            view_result = view_extractor.extract_views(equirect_bgr, view_config)
            t4 = time.perf_counter()
            result["timings"]["  view_extract"] = t4 - t3

            encode_time = 0.0
            if view_result.success:
                jpeg_q = config.get("jpeg_quality", 95)
                for i, (view, meta) in enumerate(zip(view_result.views, view_result.metadata)):
                    yaw = meta["yaw"]
                    if config.get("global_view") or heading_deg is not None:
                        fname = f"{panoid}_rnd_Y{int(yaw)}.jpg"
                    else:
                        fname = f"{panoid}_zoom{config.get('zoom_level', 2)}_view{i:02d}_{yaw:.0f}deg.jpg"
                    te = time.perf_counter()
                    _, buffer = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
                    encode_time += time.perf_counter() - te
                    result["views"].append(buffer.tobytes())
                    result["view_filenames"].append(fname)
            result["timings"]["  jpeg_encode"] = encode_time

        # ── Optional: keep full equirect ──
        if config.get("keep_panorama"):
            jpeg_q = config.get("jpeg_quality", 95)
            _, buffer = cv2.imencode(".jpg", equirect_bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
            result["panorama_bytes"] = buffer.tobytes()

        del equirect, equirect_bgr, face_imgs
        result["success"] = True

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    return result


async def process_panoid(
    session: aiohttp.ClientSession,
    record: dict,
    sem_pano: asyncio.Semaphore,
    executor: ThreadPoolExecutor,
    config: dict,
) -> dict:
    """Fetch metadata + 6 faces, then offload everything CPU-bound to a thread."""
    panoid = record["panoid"]
    build_id = record.get("build_id")
    lat = record.get("lat")
    lon = record.get("lon")
    heading_deg = record.get("heading_deg")

    out = {
        "pano_id": panoid,
        "zoom": config.get("zoom_level", 2),
        "size": (0, 0),
        "file_size": 0,
        "success": False,
        "error": "",
        "views_created": 0,
        "view_files": [],
    }

    if not build_id:
        out["error"] = "missing build_id (CSV needs build_id column)"
        return out
    if lat is None or lon is None:
        out["error"] = "missing lat/lon (needed to look up camera_metadata)"
        return out

    try:
        async with sem_pano:
            zoom = config.get("zoom_level", 2)

            # 1. camera_metadata via cached coverage tile
            faces_meta = await get_camera_metadata(session, panoid, lat, lon)
            if faces_meta is None or len(faces_meta) < 4:
                out["error"] = "camera_metadata not found in coverage tile"
                return out

            # 2. only the 4 side faces — top/bottom (faces 4, 5) are skipped
            # because directional views never sample beyond the sides' vertical
            # FOV. Saves 33% of network round-trips per pano.
            face_heics = await fetch_all_faces(
                session, panoid, build_id, zoom, _GLOBAL_AUTH, n_faces=4
            )
            n_ok = sum(1 for h in face_heics if h)
            if n_ok < 4:
                out["error"] = f"only {n_ok}/4 sides fetched"
                return out

            # 3. CPU-bound work (decode + reproject + views + encode)
            result = await asyncio.get_running_loop().run_in_executor(
                executor,
                _process_in_thread, face_heics, faces_meta, panoid, config, heading_deg,
            )

            if not result["success"]:
                out["error"] = result["error"]
                return out

            out["size"] = result["size"]
            out["views_created"] = len(result["views"])

            # 4. Save outputs
            output_dir = config.get("output_dir", ".")
            if config.get("keep_panorama") and result["panorama_bytes"]:
                pano_dir = os.path.join(output_dir, f"panos_z{zoom}")
                os.makedirs(pano_dir, exist_ok=True)
                pano_path = os.path.join(pano_dir, f"{panoid}.jpg")
                with open(pano_path, "wb") as f:
                    f.write(result["panorama_bytes"])
                out["file_size"] = len(result["panorama_bytes"])

            if result["views"]:
                views_dir = os.path.join(output_dir, f"views_z{zoom}")
                os.makedirs(views_dir, exist_ok=True)
                for view_bytes, fname in zip(result["views"], result["view_filenames"]):
                    fp = os.path.join(views_dir, fname)
                    with open(fp, "wb") as f:
                        f.write(view_bytes)
                    out["view_files"].append(fname)

            out["success"] = True

    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"

    return out


async def fetch_panos(
    sem_pano: asyncio.Semaphore,
    connector: aiohttp.TCPConnector,
    max_workers: int,
    config: dict,
    records: List[dict],
    output_dir: Optional[str] = None,
) -> tuple[int, int, str]:
    """Stream-process all panoramas concurrently."""
    print("[green]| Running OPTIMIZED Apple Look Around scraper..[/]\n")

    if config.get("create_directional_views"):
        msg = f"[green]| Directional Views: Enabled"
        if config.get("global_view"):
            msg += " [cyan](GLOBAL SINGLE RANDOM VIEW)[/]"
        else:
            msg += f" ({config.get('num_views', 6)} views)"
        msg += f" {config.get('view_resolution', 512)}x{config.get('view_resolution', 512)}[/]"
        print(msg)

    if not config.get("keep_panorama"):
        print("[yellow]| Panorama Storage: In-memory only (not saving full equirect to disk)[/]")

    jpeg_q = config.get("jpeg_quality", 95)
    if jpeg_q != 95:
        print(f"[cyan]| JPEG Quality: {jpeg_q}[/]")
    if config.get("no_antialias"):
        print("[cyan]| Antialiasing: DISABLED (--no-antialias)[/]")
    print(f"[cyan]| Equirect width cap: {config.get('max_equirect_w', 4096)} px[/]")
    print(f"[cyan]| HEIC decoder: {get_heic_decoder_name()}[/]\n")

    if output_dir is None:
        output_dir = os.getcwd()
    config["output_dir"] = output_dir

    progress = ProgressBar(len(records))
    success_count = 0
    fail_count = 0

    print(f"[cyan]Optimization: {max_workers} thread workers for CPU work (no IPC overhead)[/]")
    print(f"[cyan]Optimization: streaming pipeline (max {config.get('max_threads', 100)} concurrent panos)[/]\n")

    session_timeout = ClientTimeout(total=20, connect=8)
    async with aiohttp.ClientSession(connector=connector, timeout=session_timeout) as session:

        # Pre-fetch every coverage tile for the input set so per-pano
        # metadata lookups become free dict hits (no HTTP stall).
        t_pf0 = time.perf_counter()
        n_tiles = await prefetch_coverage_tiles_for_records(
            session, records, concurrency=min(100, config.get("max_threads", 100)),
        )
        t_pf1 = time.perf_counter()
        if n_tiles:
            print(f"[cyan]| Pre-fetched {n_tiles} coverage tiles in {t_pf1 - t_pf0:.2f}s[/]")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:

            CHUNK_SIZE = 10_000
            total = len(records)
            for i in range(0, total, CHUNK_SIZE):
                chunk = records[i : i + CHUNK_SIZE]
                tasks = [
                    process_panoid(session, rec, sem_pano, executor, config)
                    for rec in chunk
                ]
                for coro in asyncio.as_completed(tasks):
                    try:
                        result = await coro
                        if result["success"]:
                            success_count += 1
                            progress.log_success(result, config)
                        else:
                            fail_count += 1
                            progress.log_failure(result)
                        progress.update(success_count, fail_count)
                    except Exception as e:
                        print(f"[red]CRITICAL ERROR in worker: {e}[/]")
                        fail_count += 1
                        progress.update(success_count, fail_count)

    progress.finish()
    return len(records), success_count, output_dir
