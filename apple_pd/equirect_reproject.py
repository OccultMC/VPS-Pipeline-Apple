"""
Apple Look Around → equirectangular reprojection.

In-memory port of LookAround_Scraper/equirect_reproject.py.

Inputs:
    face_imgs:  list of 4 or 6 numpy arrays (back, left, front, right [, top, bottom])
                in HxWx3 uint8 RGB
    faces_meta: list of dicts with keys yaw, pitch, roll, fov_s, fov_h, cx, cy
                (one per face, indexes match face_imgs)

Output:
    HxWx3 uint8 numpy array (equirectangular panorama)

Caching:
    The per-pixel world-direction grids (rx_w/ry_w/rz_w) only depend on
    out_w + out_h, so we keep the last grid built and re-use it across panos.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import cv2
import numpy as np


# Per-process grid cache: keyed by (out_w, out_h)
_GRID_CACHE: dict = {}

# Per-process LUT cache for the side-face sampling tables. Apple's CAR rig has
# fixed lens metadata, so for a given (out_w, out_h, face dims) every pano
# uses the SAME mapping. Precomputing once saves the per-pano meshgrid +
# masks + np.where work (the bulk of the reprojection cost).
#
# Two cache shapes:
#   _SIDE_LUT_CACHE: list of (in_face, u_pix, v_pix) per side — kept for
#       reference / non-atlas path.
#   _ATLAS_MAP_CACHE: (map_x, map_y) into a horizontally concatenated face
#       atlas. Single cv2.remap call samples all sides at once.
_SIDE_LUT_CACHE: dict = {}
_ATLAS_MAP_CACHE: dict = {}


def _world_dir_grids(out_w: int, out_h: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (yaw_2d, pitch_2d, rx_w, ry_w, rz_w) for an equirect of size out_w×out_h."""
    cached = _GRID_CACHE.get((out_w, out_h))
    if cached is not None:
        return cached
    yaws = (np.arange(out_w, dtype=np.float32) / out_w) * 2 * np.float32(math.pi) - np.float32(math.pi)
    pitches = np.float32(math.pi) / 2 - (np.arange(out_h, dtype=np.float32) / out_h) * np.float32(math.pi)
    yaw_2d, pitch_2d = np.meshgrid(yaws, pitches)
    cos_p = np.cos(pitch_2d)
    rx_w = np.sin(yaw_2d) * cos_p
    ry_w = np.sin(pitch_2d)
    rz_w = np.cos(yaw_2d) * cos_p
    _GRID_CACHE[(out_w, out_h)] = (yaw_2d, pitch_2d, rx_w, ry_w, rz_w)
    return _GRID_CACHE[(out_w, out_h)]


def _render_top_or_bottom(m: dict, img: np.ndarray,
                          rx_w: np.ndarray, ry_w: np.ndarray, rz_w: np.ndarray,
                          out: np.ndarray) -> np.ndarray:
    """
    Same approach as lookmap.eu's LookaroundAdapter for top/bottom faces:
    place the SphereGeometry patch at its default (yaw, fov_s, fov_h, cy)
    equator position, then rotate it to the pole via rotateX(-pitch) and
    rotateZ(-roll), with .scale(-1,-1,1) effectively applied (S below).
    Inverse for backward sampling.
    """
    cp, sp = math.cos(m["pitch"]), math.sin(m["pitch"])
    cr, sr = math.cos(m["roll"]), math.sin(m["roll"])

    rx1 = cr * rx_w - sr * ry_w
    ry1 = sr * rx_w + cr * ry_w
    rz1 = rz_w
    rx2 = rx1
    ry2 = cp * ry1 - sp * rz1
    rz2 = sp * ry1 + cp * rz1
    rx2 = -rx2  # S = diag(-1, -1, 1)
    ry2 = -ry2

    phi = np.arctan2(rz2, -rx2)
    theta = np.arccos(np.clip(ry2, -1.0, 1.0))

    fov_s = m["fov_s"]
    fov_h = m["fov_h"]
    phi_start = m["yaw"] - fov_s / 2 - math.pi / 2
    theta_start = math.pi / 2 - fov_h / 2 - m["cy"]

    phi_norm = (phi - phi_start) % (2 * math.pi)
    in_phi = phi_norm < fov_s
    in_theta = (theta >= theta_start) & (theta <= theta_start + fov_h)
    in_face = in_phi & in_theta
    if not in_face.any():
        return out

    u = phi_norm / fov_s
    v = (theta - theta_start) / fov_h
    u_pix = np.clip((u * img.shape[1]).astype(np.int32), 0, img.shape[1] - 1)
    v_pix = np.clip((v * img.shape[0]).astype(np.int32), 0, img.shape[0] - 1)
    sampled = img[v_pix, u_pix]
    return np.where(in_face[..., None], sampled, out)


def _build_atlas_map(
    sides_meta: List[dict],
    side_widths: List[int],
    side_heights: List[int],
    out_w: int,
    out_h: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (map_x, map_y) for cv2.remap into a horizontally concatenated
    side-face atlas: atlas[:, 0:w0] = back, atlas[:, w0:w0+w1] = left, etc.
    Pixels not covered by any side get map_x = -1 → cv2.remap's
    BORDER_CONSTANT yields black.

    Same projection math as the reference numpy path; identical mapping math
    (yaw_centers normalisation, eff_min/eff_max overlap-removal,
    pitch range with cy offset, fractional u/v).
    """
    yaw_2d, pitch_2d, _, _, _ = _world_dir_grids(out_w, out_h)

    yaw_centers = [m["yaw"] for m in sides_meta]
    for i in range(1, 4):
        while yaw_centers[i] < yaw_centers[i - 1]:
            yaw_centers[i] += 2 * math.pi
    fov_s = [m["fov_s"] for m in sides_meta]
    fov_h = [m["fov_h"] for m in sides_meta]
    cy = [m["cy"] for m in sides_meta]

    eff_min = [yaw_centers[i] - fov_s[i] / 2 for i in range(4)]
    eff_max = [yaw_centers[i] + fov_s[i] / 2 for i in range(4)]
    for i in range(1, 4):
        if eff_max[i - 1] > eff_min[i]:
            eff_max[i - 1] = eff_min[i]

    map_x = np.full((out_h, out_w), -1.0, dtype=np.float32)
    map_y = np.full((out_h, out_w), -1.0, dtype=np.float32)

    # face offset = sum of widths of preceding faces in the atlas
    offsets = [0]
    for w in side_widths[:-1]:
        offsets.append(offsets[-1] + w)

    for i in range(4):
        if side_widths[i] == 0:
            continue
        dyaw = ((yaw_2d - yaw_centers[i] + math.pi) % (2 * math.pi)) - math.pi
        dyaw_min = eff_min[i] - yaw_centers[i]
        dyaw_max = eff_max[i] - yaw_centers[i]
        in_yaw = (dyaw >= dyaw_min) & (dyaw <= dyaw_max)
        pitch_min = -fov_h[i] / 2 + cy[i]
        pitch_max = fov_h[i] / 2 + cy[i]
        in_pitch = (pitch_2d >= pitch_min) & (pitch_2d <= pitch_max)
        in_face = in_yaw & in_pitch
        if not in_face.any():
            continue

        u = (dyaw - dyaw_min) / fov_s[i]
        v = (pitch_max - pitch_2d) / fov_h[i]
        u_pix_in_face = np.clip(u * side_widths[i], 0, side_widths[i] - 1)
        v_pix_in_face = np.clip(v * side_heights[i], 0, side_heights[i] - 1)

        atlas_x = u_pix_in_face + offsets[i]
        # Sides are processed 0..3, later face wins on overlap (matches
        # the reference numpy path's `out = np.where(in_face, sampled, out)`).
        map_x = np.where(in_face, atlas_x.astype(np.float32), map_x)
        map_y = np.where(in_face, v_pix_in_face.astype(np.float32), map_y)

    return map_x, map_y


def _build_side_luts(
    sides_meta: List[dict],
    side_widths: List[int],
    side_heights: List[int],
    out_w: int,
    out_h: int,
) -> List[tuple]:
    """
    Precompute per-side (in_face_mask, u_pix, v_pix) once — these only
    depend on the lens metadata + face dims + output size, all of which
    are constant across panos at a given zoom for the same camera rig.
    """
    yaw_2d, pitch_2d, _, _, _ = _world_dir_grids(out_w, out_h)

    yaw_centers = [m["yaw"] for m in sides_meta]
    for i in range(1, 4):
        while yaw_centers[i] < yaw_centers[i - 1]:
            yaw_centers[i] += 2 * math.pi
    fov_s = [m["fov_s"] for m in sides_meta]
    fov_h = [m["fov_h"] for m in sides_meta]
    cy = [m["cy"] for m in sides_meta]

    eff_min = [yaw_centers[i] - fov_s[i] / 2 for i in range(4)]
    eff_max = [yaw_centers[i] + fov_s[i] / 2 for i in range(4)]
    for i in range(1, 4):
        if eff_max[i - 1] > eff_min[i]:
            eff_max[i - 1] = eff_min[i]

    luts = []
    for i in range(4):
        dyaw = ((yaw_2d - yaw_centers[i] + math.pi) % (2 * math.pi)) - math.pi
        dyaw_min = eff_min[i] - yaw_centers[i]
        dyaw_max = eff_max[i] - yaw_centers[i]
        in_yaw = (dyaw >= dyaw_min) & (dyaw <= dyaw_max)
        pitch_min = -fov_h[i] / 2 + cy[i]
        pitch_max = fov_h[i] / 2 + cy[i]
        in_pitch = (pitch_2d >= pitch_min) & (pitch_2d <= pitch_max)
        in_face = in_yaw & in_pitch
        u = (dyaw - dyaw_min) / fov_s[i]
        v = (pitch_max - pitch_2d) / fov_h[i]
        u_pix = np.clip((u * side_widths[i]).astype(np.int32), 0, side_widths[i] - 1)
        v_pix = np.clip((v * side_heights[i]).astype(np.int32), 0, side_heights[i] - 1)
        luts.append((in_face, u_pix, v_pix))
    return luts


def reproject_faces_to_equirect(
    face_imgs: List[np.ndarray],
    faces_meta: List[dict],
    out_w: int = 4096,
    out_h: int | None = None,
) -> np.ndarray:
    """
    Reproject 4 or 6 lens-projected faces into an equirectangular RGB array.

    `face_imgs` order: back, left, front, right, [top, bottom]
    `faces_meta` order matches.

    Side-face sampling tables are cached per (out_w, out_h, face dims, lens
    metadata signature) so the per-pano cost is just 4 fancy-indexing
    lookups instead of meshgrid + 4 mask-builds + 4 np.wheres.
    """
    if out_h is None:
        out_h = out_w // 2

    sides_imgs = face_imgs[:4]
    sides_meta = faces_meta[:4]
    cap_imgs = face_imgs[4:6]
    cap_meta = faces_meta[4:6]

    out = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    # Top/bottom first — sides will overwrite them in any overlap region.
    # These are RARE in normal pipelines (we skip downloading them by default
    # because views never sample the polar caps), but we still support them.
    if cap_imgs and any(im is not None for im in cap_imgs):
        _, _, rx_w, ry_w, rz_w = _world_dir_grids(out_w, out_h)
        for m, img in zip(cap_meta, cap_imgs):
            if img is not None:
                out = _render_top_or_bottom(m, img, rx_w, ry_w, rz_w, out)

    # Side faces via cv2.remap on a horizontally concatenated atlas.
    # Map cached per (out_w, out_h, dims, lens metadata signature) so
    # each subsequent pano of the same shape just hits the dict.
    side_widths = tuple(im.shape[1] if im is not None else 0 for im in sides_imgs)
    side_heights = tuple(im.shape[0] if im is not None else 0 for im in sides_imgs)
    if not all(h == side_heights[0] for h in side_heights if h):
        # Mixed heights — fall back to the per-side numpy path.
        return _reproject_via_numpy(sides_imgs, sides_meta, side_widths, side_heights,
                                    out_w, out_h, out)

    sig = (
        out_w, out_h, side_widths, side_heights,
        tuple(round(m["yaw"], 4) for m in sides_meta),
        tuple(round(m["fov_s"], 4) for m in sides_meta),
        tuple(round(m["fov_h"], 4) for m in sides_meta),
        tuple(round(m["cy"], 4) for m in sides_meta),
    )
    cached = _ATLAS_MAP_CACHE.get(sig)
    if cached is None:
        cached = _build_atlas_map(sides_meta, list(side_widths), list(side_heights), out_w, out_h)
        _ATLAS_MAP_CACHE[sig] = cached
    map_x, map_y = cached

    # Build the atlas (concatenate sides horizontally; missing faces become
    # zeros at their slice — the matching map_x = -1 keeps them out anyway).
    parts = []
    for im, w in zip(sides_imgs, side_widths):
        if w == 0:
            continue
        parts.append(im if im is not None else np.zeros((side_heights[0], w, 3), dtype=np.uint8))
    atlas = np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]

    sides_out = cv2.remap(
        atlas, map_x, map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    # Composite sides over the (possibly already-painted) top/bottom layer:
    # a side-pixel exists wherever map_x >= 0.
    if cap_imgs and any(im is not None for im in cap_imgs):
        side_mask = (map_x >= 0)
        out = np.where(side_mask[..., None], sides_out, out)
    else:
        out = sides_out

    return out


def _reproject_via_numpy(sides_imgs, sides_meta, side_widths, side_heights,
                         out_w, out_h, out):
    """Fallback for mixed-height face sets — uses the per-side LUT cache."""
    sig = (
        out_w, out_h, tuple(side_widths), tuple(side_heights),
        tuple(round(m["yaw"], 4) for m in sides_meta),
        tuple(round(m["fov_s"], 4) for m in sides_meta),
        tuple(round(m["fov_h"], 4) for m in sides_meta),
        tuple(round(m["cy"], 4) for m in sides_meta),
    )
    luts = _SIDE_LUT_CACHE.get(sig)
    if luts is None:
        luts = _build_side_luts(sides_meta, list(side_widths), list(side_heights), out_w, out_h)
        _SIDE_LUT_CACHE[sig] = luts
    for i, img in enumerate(sides_imgs):
        if img is None:
            continue
        in_face, u_pix, v_pix = luts[i]
        if not in_face.any():
            continue
        sampled = img[v_pix, u_pix]
        out = np.where(in_face[..., None], sampled, out)
    return out
