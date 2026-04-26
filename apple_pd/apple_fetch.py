"""
Apple Look Around tile/face fetch and HEIC decode.

Replaces gsvpd/core_optimized.fetch_tile (Google's tile API). Each Apple
panorama has 6 lens-projected faces (back/left/front/right/top/bottom)
delivered as HEIC at 8 zoom levels (0=highest).

Authenticated URL signing is identical to streetlevel.lookaround.auth —
imported directly so we stay compatible with whatever ID/build_id we
got from upstream.
"""
from __future__ import annotations

import asyncio
import io
from typing import List, Optional, Tuple

import aiohttp
import numpy as np
from PIL import Image
import pillow_heif
from yarl import URL as YURL

from streetlevel.lookaround import Face, get_coverage_tile_async, get_coverage_tile_by_latlon_async
from streetlevel.lookaround.auth import Authenticator
from streetlevel.geo import wgs84_to_tile_coord


pillow_heif.register_heif_opener()


# HEIC decoder selection. Two options:
#
#   pillow_heif (default): ~10 ms/face on the test fixture. Releases the
#       GIL → scales ~5x across our ThreadPoolExecutor. This is the right
#       choice for the Stage 2 batch pipeline.
#
#   heic2rgb (sk-zk):      ~5 ms/face per call (~2x faster) but DOES NOT
#       release the GIL. In a threaded batch pipeline this serialises all
#       decodes, giving ~half the throughput of pillow_heif. Only worth
#       enabling for single-pano / single-thread workloads (e.g. the
#       LookAround_Scraper debug webapp).
#
# Force selection with the APPLE_HEIC_DECODER environment variable:
#   APPLE_HEIC_DECODER=pillow_heif   (default, recommended for Stage 2)
#   APPLE_HEIC_DECODER=heic2rgb      (only if heic2rgb wheel is installed)
import os as _os

_DECODER_PREF = _os.environ.get("APPLE_HEIC_DECODER", "pillow_heif").lower()

try:
    import heic2rgb as _heic2rgb  # type: ignore
    _HEIC2RGB_AVAILABLE = True
except ImportError:
    _heic2rgb = None
    _HEIC2RGB_AVAILABLE = False

_USE_HEIC2RGB = _HEIC2RGB_AVAILABLE and _DECODER_PREF == "heic2rgb"


def get_heic_decoder_name() -> str:
    if _USE_HEIC2RGB:
        return "heic2rgb (libavcodec, single-threaded - no GIL release)"
    note = ""
    if _HEIC2RGB_AVAILABLE and _DECODER_PREF != "heic2rgb":
        note = "  [heic2rgb available; set APPLE_HEIC_DECODER=heic2rgb to enable]"
    return f"pillow_heif (GIL-releasing, scales across threads){note}"


FACE_NAMES = ["back", "left", "front", "right", "top", "bottom"]
FACE_ENDPOINT = "https://gspe72-ssl.ls.apple.com/mnn_us/"


# Diagnostic counters — first N responses printed in detail
_FETCH_DIAG = {
    "logged": 0, "limit": 20,
    "status_counts": {},
    "non200_logged": 0, "non200_limit": 10,
    "exc_logged": 0, "exc_limit": 5,
}


def _build_face_url(panoid: str, build_id: str, face_idx: int, zoom: int,
                    auth: Authenticator) -> str:
    """Build the authenticated face URL exactly the way streetlevel does."""
    panoid_padded = panoid.zfill(20)
    panoid_split = [panoid_padded[i:i + 4] for i in range(0, 20, 4)]
    panoid_url = "/".join(panoid_split)
    build_id_padded = build_id.zfill(10)
    z = min(7, zoom)
    url = f"{FACE_ENDPOINT}{panoid_url}/{build_id_padded}/t/{face_idx}/{z}"
    return auth.authenticate_url(url)


async def fetch_face(
    session: aiohttp.ClientSession,
    panoid: str,
    build_id: str,
    face_idx: int,
    zoom: int,
    auth: Authenticator,
    retries: int = 2,
    backoff: float = 0.15,
) -> Optional[bytes]:
    """Fetch a single HEIC face. Returns bytes or None."""
    for attempt in range(1, retries + 1):
        try:
            url = _build_face_url(panoid, build_id, face_idx, zoom, auth)
            # encoded=True prevents aiohttp/yarl from double-encoding the
            # already-encoded %2B in the auth accessKey (Apple returns 403
            # otherwise).
            async with session.get(YURL(url, encoded=True)) as response:
                status = response.status
                _FETCH_DIAG["status_counts"][status] = _FETCH_DIAG["status_counts"].get(status, 0) + 1

                if _FETCH_DIAG["logged"] < _FETCH_DIAG["limit"]:
                    _FETCH_DIAG["logged"] += 1
                    cl = response.headers.get("Content-Length", "?")
                    ct = response.headers.get("Content-Type", "?")
                    print(f"[FETCH-DIAG #{_FETCH_DIAG['logged']}] panoid={panoid} "
                          f"face={FACE_NAMES[face_idx]} zoom={zoom} status={status} "
                          f"content-length={cl} content-type={ct}", flush=True)

                if status != 200:
                    if _FETCH_DIAG["non200_logged"] < _FETCH_DIAG["non200_limit"]:
                        _FETCH_DIAG["non200_logged"] += 1
                        try:
                            body = await response.read()
                            preview = body[:300].decode("utf-8", errors="replace")
                        except Exception as e:
                            preview = f"<read failed: {e}>"
                        print(f"[FETCH-DIAG NON-200 #{_FETCH_DIAG['non200_logged']}] "
                              f"status={status} panoid={panoid} face={face_idx} zoom={zoom}\n"
                              f"  body[:300]={preview!r}", flush=True)
                    # Apple returns 403 when auth tokens expire. Retry with
                    # a brand-new Authenticator session_id (cheap).
                    if status == 403 and attempt < retries:
                        auth = Authenticator()
                        await asyncio.sleep(backoff * (2 ** (attempt - 1)))
                        continue
                    return None

                return await response.read()

        except Exception as e:
            if _FETCH_DIAG["exc_logged"] < _FETCH_DIAG["exc_limit"]:
                _FETCH_DIAG["exc_logged"] += 1
                print(f"[FETCH-DIAG EXC #{_FETCH_DIAG['exc_logged']}] "
                      f"panoid={panoid} face={face_idx} zoom={zoom} attempt={attempt}/{retries} "
                      f"{type(e).__name__}: {e}", flush=True)
            if attempt < retries:
                await asyncio.sleep(backoff * (2 ** (attempt - 1)))

    return None


async def fetch_all_faces(
    session: aiohttp.ClientSession,
    panoid: str,
    build_id: str,
    zoom: int,
    auth: Authenticator,
    n_faces: int = 6,
) -> List[Optional[bytes]]:
    """Fetch all `n_faces` (default 6) faces in parallel."""
    tasks = [fetch_face(session, panoid, build_id, i, zoom, auth) for i in range(n_faces)]
    return await asyncio.gather(*tasks)


def decode_heic_bytes(heic: bytes) -> Optional[np.ndarray]:
    """Decode HEIC bytes -> RGB uint8 numpy array. Returns None on failure."""
    if not heic:
        return None
    if _USE_HEIC2RGB:
        try:
            img = _heic2rgb.decode(heic)
            return np.frombuffer(img.data, dtype=np.uint8).reshape(img.height, img.width, 3)
        except Exception:
            # Corrupt HEIC etc — fall through to pillow_heif which is more lenient.
            pass
    try:
        img = Image.open(io.BytesIO(heic)).convert("RGB")
        return np.asarray(img)
    except Exception:
        return None


# ─── Coverage-tile metadata cache ────────────────────────────────────────────
# Multiple panos on the same z=17 tile share the same coverage_tile fetch.
# Cache stores: { (tile_x, tile_y): {pano_id_str: pano_obj_with_camera_metadata} }

_COVERAGE_CACHE: dict = {}
_COVERAGE_LOCK_BY_TILE: dict = {}


async def prefetch_coverage_tiles_for_records(
    session: aiohttp.ClientSession,
    records: list,
    concurrency: int = 50,
) -> int:
    """
    Pre-populate _COVERAGE_CACHE for every unique z=17 tile referenced by
    `records` (each record must have lat + lon). Runs `concurrency`
    fetches in parallel. Returns the number of tiles fetched (cache misses).

    After this returns, every per-pano `get_camera_metadata` call below
    is a pure dict lookup — no extra HTTP round-trip.
    """
    unique_tiles = set()
    for r in records:
        lat = r.get("lat") if isinstance(r, dict) else None
        lon = r.get("lon") if isinstance(r, dict) else None
        if lat is None or lon is None:
            continue
        unique_tiles.add(wgs84_to_tile_coord(lat, lon, 17))
    todo = [t for t in unique_tiles if t not in _COVERAGE_CACHE]
    if not todo:
        return 0

    sem = asyncio.Semaphore(concurrency)

    async def _one(tx, ty):
        async with sem:
            try:
                cov = await get_coverage_tile_async(tx, ty, session)
                _COVERAGE_CACHE[(tx, ty)] = {str(p.id): p for p in cov.panos}
            except Exception:
                _COVERAGE_CACHE[(tx, ty)] = {}

    await asyncio.gather(*[_one(tx, ty) for tx, ty in todo])
    return len(todo)


async def get_camera_metadata(
    session: aiohttp.ClientSession,
    panoid: str,
    lat: float,
    lon: float,
) -> Optional[List[dict]]:
    """
    Look up camera_metadata for a pano via its z=17 coverage tile (cached).
    Returns a list of 6 dicts (yaw, pitch, roll, fov_s, fov_h, cx, cy) or None.
    """
    tile_x, tile_y = wgs84_to_tile_coord(lat, lon, 17)
    key = (tile_x, tile_y)

    panos_dict = _COVERAGE_CACHE.get(key)
    if panos_dict is None:
        # Single in-flight fetch per tile (avoids duplicate API hits when
        # many concurrent panos share the same tile)
        lock = _COVERAGE_LOCK_BY_TILE.setdefault(key, asyncio.Lock())
        async with lock:
            panos_dict = _COVERAGE_CACHE.get(key)
            if panos_dict is None:
                try:
                    cov = await get_coverage_tile_async(tile_x, tile_y, session)
                    panos_dict = {str(p.id): p for p in cov.panos}
                except Exception as e:
                    print(f"[META-FETCH FAIL] tile=({tile_x},{tile_y}) {type(e).__name__}: {e}", flush=True)
                    panos_dict = {}
                _COVERAGE_CACHE[key] = panos_dict

    pano = panos_dict.get(str(panoid))
    if pano is None or not pano.camera_metadata:
        return None

    return [
        {
            "yaw": cm.position.yaw,
            "pitch": cm.position.pitch,
            "roll": cm.position.roll,
            "fov_s": cm.lens_projection.fov_s,
            "fov_h": cm.lens_projection.fov_h,
            "cx": cm.lens_projection.cx,
            "cy": cm.lens_projection.cy,
        }
        for cm in pano.camera_metadata
    ]
