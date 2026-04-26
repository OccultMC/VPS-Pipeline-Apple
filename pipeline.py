#!/usr/bin/env python3
"""
VPS Pipeline (Apple Look Around): HEIC face fetch → metadata-based equirect →
directional views → MegaLoc feature extraction → R2 upload.

Queue-based distributed worker — pulls pano chunks from a shared Redis queue,
processes each chunk, uploads results to R2, and grabs the next chunk.

Workers self-destruct when the queue is empty and all tasks are done.

Progress is logged to stdout in structured format for vastai logs polling:
    PROGRESS|{instance_id}|{chunk_id}|{processed}|{total}|{status}

Differences from the Google variant:
  - 4 HEIC face fetches per pano (back/left/front/right) from
    gspe72-ssl.ls.apple.com instead of cbk0-3.google.com tiles.
  - HEIC decoded with pillow_heif (or heic2rgb if installed).
  - Equirect built by reproject_faces_to_equirect using each face's
    lens metadata (yaw/fov_s/fov_h/cy) — see apple_pd/equirect_reproject.py.
  - Camera metadata fetched once per z=17 coverage tile (cached) so
    multiple panos in the same tile share one HTTP round-trip.
"""

import asyncio
import csv
import gc
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
import math
from typing import Dict, List, Set, Tuple

import numpy as np

# Fix OpenMP conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration (Hardcoded per spec)
# ═══════════════════════════════════════════════════════════════════════════════

HARDCODED_CONFIG = {
    'zoom_level': 2,
    'max_threads': 150,
    'workers': 8,
    'create_directional_views': True,
    'keep_panorama': False,
    'view_resolution': 322,
    'view_fov': 60.0,
    'num_views': 8,
    'global_view': False,
    'augment': False,
    'no_antialias': True,
    'interpolation': 'cubic',
    'output_dir': None,
    'batch_size': 32,
    'queue_size': 512,
}

# ═══════════════════════════════════════════════════════════════════════════════
# Environment Variables
# ═══════════════════════════════════════════════════════════════════════════════

REDIS_URL = os.environ.get('REDIS_URL', '')
REDIS_TOKEN = os.environ.get('REDIS_TOKEN', '')
REGION = os.environ.get('REGION', '')
CSV_BUCKET_PREFIX = os.environ.get('CSV_BUCKET_PREFIX', 'CSV')
FEATURES_BUCKET_PREFIX = os.environ.get('FEATURES_BUCKET_PREFIX', 'Features')
CITY_NAME = os.environ.get('CITY_NAME', 'Unknown')
INSTANCE_ID = (os.environ.get('INSTANCE_ID', '')
               or os.environ.get('CONTAINER_ID', '')
               or os.environ.get('VAST_CONTAINERLABEL', ''))
VAST_API_KEY = os.environ.get('VAST_API_KEY', '')

MAX_DISK_GB = 100
MIN_FREE_GB = 5
TOTAL_CHUNKS = 0  # Set from Redis metadata at startup


def _chunk_num(chunk_id: str) -> int:
    """Convert Redis chunk ID to 1-based number: 'chunk_0001' → 1."""
    return int(chunk_id.split('_')[1])


def _output_base(city: str, chunk_id: str) -> str:
    """Return '{city}_{N}.{total}' for output filenames."""
    return f"{city}_{_chunk_num(chunk_id)}.{TOTAL_CHUNKS}"


def _redis_retry(fn, *args, retries=5, delay=3, label="redis"):
    """Retry a Redis call with exponential backoff. Returns True on success."""
    for attempt in range(1, retries + 1):
        try:
            fn(*args)
            return True
        except Exception as e:
            print(f"[WARN] {label} attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                time.sleep(delay * attempt)
    print(f"[ERROR] {label} failed after {retries} attempts")
    return False

# ═══════════════════════════════════════════════════════════════════════════════
# Imports: Street View Downloader
# ═══════════════════════════════════════════════════════════════════════════════

import aiohttp
import cv2
from streetlevel.lookaround.auth import Authenticator
from apple_pd.apple_fetch import (
    fetch_all_faces, decode_heic_bytes, get_camera_metadata,
    prefetch_coverage_tiles_for_records, get_heic_decoder_name,
)
from apple_pd.equirect_reproject import reproject_faces_to_equirect
from apple_pd.directional_views import (
    DirectionalViewExtractor, DirectionalViewConfig,
)
from concurrent.futures import ThreadPoolExecutor

# ═══════════════════════════════════════════════════════════════════════════════
# Imports: Feature Extraction
# ═══════════════════════════════════════════════════════════════════════════════

import torch
from torchvision import transforms

# ═══════════════════════════════════════════════════════════════════════════════
# R2 Storage
# ═══════════════════════════════════════════════════════════════════════════════

from r2_storage import R2Client

# ═══════════════════════════════════════════════════════════════════════════════
# Redis Task Queue
# ═══════════════════════════════════════════════════════════════════════════════

from redis_queue import TaskQueue


# ═══════════════════════════════════════════════════════════════════════════════
# View Item & Shared State
# ═══════════════════════════════════════════════════════════════════════════════

_SENTINEL = None

class ViewItem:
    __slots__ = ('panoid', 'view_data', 'lat', 'lng')
    def __init__(self, panoid: str, view_data, lat: float, lng: float):
        self.panoid = panoid
        self.view_data = view_data  # RGB numpy array (uint8 HWC)
        self.lat = lat
        self.lng = lng

class SharedState:
    """Thread-safe writing to memmap + metadata + failures."""
    def __init__(self, features_memmap, metadata_file_path, failed_file_path, start_idx=0):
        self.memmap = features_memmap
        self.write_idx = start_idx
        self.lock = threading.Lock()
        self.metadata_handle = open(metadata_file_path, 'w', encoding='utf-8')
        self.failed_handle = open(failed_file_path, 'w', encoding='utf-8')
        self._batch_count = 0

    def write_batch(self, features_batch: np.ndarray, metadata_batch: List[dict]):
        n = len(features_batch)
        if n == 0:
            return
        with self.lock:
            start = self.write_idx
            end = start + n
            self.memmap[start:end] = features_batch
            for i, meta in enumerate(metadata_batch):
                meta['feature_index'] = start + i
                self.metadata_handle.write(json.dumps(meta) + '\n')
            self.metadata_handle.flush()
            self.write_idx = end
            self._batch_count += 1
            # Flush memmap to disk periodically to keep RSS low
            # Run in background to avoid blocking the GPU thread
            if self._batch_count % 100 == 0:
                mm = self.memmap
                threading.Thread(target=lambda: mm.flush(), daemon=True).start()

    def log_failure(self, panoid: str, reason: str):
        with self.lock:
            entry = {'panoid': panoid, 'reason': str(reason), 'timestamp': time.time()}
            self.failed_handle.write(json.dumps(entry) + '\n')
            self.failed_handle.flush()

    def close(self):
        with self.lock:
            if self.metadata_handle:
                self.metadata_handle.close()
                self.metadata_handle = None
            if self.failed_handle:
                self.failed_handle.close()
                self.failed_handle = None


# ═══════════════════════════════════════════════════════════════════════════════
# GPU Feature Extractor
# ═══════════════════════════════════════════════════════════════════════════════

GPU_INIT_TIMEOUT = int(os.environ.get('GPU_INIT_TIMEOUT', '300'))  # 5 min default


class _InitWatchdog:
    """Watchdog that kills the process if init hangs too long."""
    def __init__(self, timeout_sec: int, stage: str = "unknown"):
        self.timeout = timeout_sec
        self.stage = stage
        self._timer = None

    def start(self, stage: str = None):
        if stage:
            self.stage = stage
        self.cancel()
        self._timer = threading.Timer(self.timeout, self._on_timeout)
        self._timer.daemon = True
        self._timer.start()

    def cancel(self):
        if self._timer:
            self._timer.cancel()
            self._timer = None

    def _on_timeout(self):
        msg = (f"[FATAL] Watchdog timeout after {self.timeout}s during: {self.stage}. "
               f"Process is stuck — forcing exit.")
        print(msg, flush=True)
        try:
            upload_logs_to_r2()
        except Exception:
            pass
        os._exit(1)


def _run_with_timeout(fn, timeout_sec: int, stage: str):
    """Run fn() in a thread with a timeout. Raises TimeoutError if it hangs."""
    result_container = [None]
    error_container = [None]

    def _target():
        try:
            result_container[0] = fn()
        except Exception as e:
            error_container[0] = e

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=timeout_sec)

    if t.is_alive():
        raise TimeoutError(
            f"{stage} timed out after {timeout_sec}s — process is hung. "
            f"This usually means a network stall (model download) or CUDA driver issue."
        )

    if error_container[0] is not None:
        raise error_container[0]

    return result_container[0]


class GpuExtractor:
    def __init__(self):
        t0 = time.time()
        self._watchdog = _InitWatchdog(GPU_INIT_TIMEOUT * 2 + 120, "gpu_init_overall")
        self._watchdog.start()

        try:
            self._init_gpu(t0)
        finally:
            self._watchdog.cancel()

    @staticmethod
    def _load_model():
        """Load MegaLoc model using the same get_trained_model path as inference.

        Primary: torch.hub → get_trained_model (HuggingFace weights, strict=True)
        Fallback: baked model.safetensors if hub download fails (strict=True)
        """
        errors = []

        # ── Primary: torch.hub get_trained_model (same as Flask server) ──
        for attempt in range(1, 4):
            try:
                print(f"[INIT]   torch.hub get_trained_model attempt {attempt}/3...", flush=True)
                model = torch.hub.load(
                    "gmberton/MegaLoc", "get_trained_model", trust_repo=True
                )
                print("[INIT]   Model loaded via torch.hub (matches inference server)", flush=True)
                return model
            except Exception as e:
                print(f"[INIT]   torch.hub attempt {attempt} failed: {type(e).__name__}: {e}", flush=True)
                errors.append(f"torch.hub#{attempt}: {e}")
                if attempt < 3:
                    time.sleep(2 ** attempt)

        # ── Fallback: baked model (strict=True to catch key mismatches) ──
        print("[WARN] torch.hub failed, trying baked model fallback...", flush=True)
        model_path = Path('/app/models/megaloc/model.safetensors')
        if model_path.exists():
            try:
                from safetensors.torch import load_file

                size_mb = model_path.stat().st_size / 1e6
                print(f"[INIT]   Loading baked weights ({size_mb:.1f}MB) from {model_path}", flush=True)
                state_dict = load_file(str(model_path))

                hub_dir = Path(torch.hub.get_dir()) / 'gmberton_MegaLoc_main'
                if not hub_dir.exists():
                    raise FileNotFoundError(f"MegaLoc architecture not found at {hub_dir}")

                sys.path.insert(0, str(hub_dir))
                try:
                    import importlib
                    megaloc_module = importlib.import_module('megaloc_model')
                    model = megaloc_module.MegaLoc()
                    model.load_state_dict(state_dict, strict=True)
                    print("[INIT]   Model loaded from baked weights (strict=True, all keys matched)", flush=True)
                    return model
                finally:
                    sys.path.pop(0)
            except Exception as e:
                errors.append(f"baked model: {e}")
                print(f"[WARN] Baked model fallback failed: {e}", flush=True)
        else:
            errors.append("baked model.safetensors not found")

        raise RuntimeError(
            f"All model sources failed:\n"
            + "\n".join(f"  - {err}" for err in errors)
            + "\n\nEnsure the worker has network access to github.com + huggingface.co, "
            + "or bake a compatible model.safetensors into the Docker image."
        )

    def _init_gpu(self, t0: float):
        # ── Step 1: CUDA check ──
        print(f"[INIT] Step 1/5: Checking CUDA availability...", flush=True)
        cuda_available = torch.cuda.is_available()
        if not cuda_available:
            raise RuntimeError(
                "CUDA is not available! Check nvidia-smi, CUDA drivers, and container GPU passthrough. "
                "torch.cuda.is_available() returned False."
            )
        gpu_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        gpu_mem = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / (1024**3)
        print(f"[INIT]   CUDA OK — GPU: {gpu_name}, VRAM: {gpu_mem:.1f}GB", flush=True)
        print(f"[INIT]   CUDA version: {torch.version.cuda}, PyTorch: {torch.__version__}", flush=True)

        torch.set_float32_matmul_precision('high')
        self.device = torch.device('cuda')

        # ── Step 2: Load MegaLoc model ──
        print(f"[INIT] Step 2/5: Loading MegaLoc model...", flush=True)
        self._watchdog.start("load_model")
        dl_start = time.time()
        try:
            model = _run_with_timeout(
                self._load_model,
                timeout_sec=GPU_INIT_TIMEOUT,
                stage="load_model"
            )
        except TimeoutError:
            raise RuntimeError(
                f"Model loading timed out after {GPU_INIT_TIMEOUT}s. "
                "Network download or model construction is hung."
            )
        print(f"[INIT]   Model ready in {time.time() - dl_start:.1f}s", flush=True)

        # ── Step 3: Move to GPU ──
        self._watchdog.start("model_to_cuda")
        print(f"[INIT] Step 3/5: Moving model to {self.device}...", flush=True)
        move_start = time.time()
        try:
            model = _run_with_timeout(
                lambda: model.to(self.device).eval(),
                timeout_sec=120,
                stage="model_to_cuda"
            )
        except TimeoutError:
            raise RuntimeError(
                "model.to(cuda) timed out after 120s. CUDA driver may be unresponsive. "
                "Check nvidia-smi and dmesg for GPU errors."
            )
        print(f"[INIT]   Model on GPU in {time.time() - move_start:.1f}s", flush=True)

        # ── Step 4: DataParallel / compile ──
        self._watchdog.start("torch_compile")
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            print(f"[INIT] Step 4/5: Wrapping with DataParallel ({gpu_count} GPUs)...", flush=True)
            model = torch.nn.DataParallel(model)

        if hasattr(torch, 'compile'):
            print(f"[INIT] Step 4/5: torch.compile()...", flush=True)
            compile_start = time.time()
            try:
                model = _run_with_timeout(
                    lambda: torch.compile(model),
                    timeout_sec=120,
                    stage="torch_compile"
                )
                print(f"[INIT]   torch.compile() done in {time.time() - compile_start:.1f}s", flush=True)
            except TimeoutError:
                print(f"[WARN] torch.compile() timed out after 120s — running without compilation (this is OK)", flush=True)
            except Exception as e:
                print(f"[WARN] torch.compile() failed: {type(e).__name__}: {e} — running without compilation", flush=True)
        else:
            print(f"[INIT] Step 4/5: torch.compile not available (PyTorch < 2.0), skipping", flush=True)

        # ── Step 5: Warmup inference ──
        self._watchdog.start("warmup_inference")
        print(f"[INIT] Step 5/5: Warmup inference...", flush=True)
        warmup_start = time.time()
        try:
            dummy = torch.randn(1, 3, 322, 322, device=self.device)
            dummy = (dummy - torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)) / \
                    torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
            with torch.no_grad():
                _ = model(dummy)
            del dummy
            torch.cuda.synchronize()
            print(f"[INIT]   Warmup done in {time.time() - warmup_start:.1f}s", flush=True)
        except Exception as e:
            print(f"[WARN] Compiled warmup failed: {type(e).__name__}: {e}", flush=True)
            print(f"[WARN] Falling back to eager mode (disabling torch.compile)...", flush=True)

            try:
                torch._dynamo.reset()
            except Exception:
                pass

            if hasattr(model, '_orig_mod'):
                model = model._orig_mod
                print(f"[WARN] Unwrapped compiled model to original module", flush=True)

            try:
                torch.cuda.empty_cache()
                dummy = torch.randn(1, 3, 322, 322, device=self.device)
                dummy = (dummy - torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)) / \
                        torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
                with torch.no_grad():
                    _ = model(dummy)
                del dummy
                torch.cuda.synchronize()
                print(f"[INIT]   Eager warmup OK in {time.time() - warmup_start:.1f}s", flush=True)
            except Exception as e2:
                raise RuntimeError(
                    f"Warmup inference failed in both compiled AND eager mode: {type(e2).__name__}: {e2}. "
                    f"The model may be incompatible with this GPU or CUDA version."
                )

        self.model = model
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        self.executor = ThreadPoolExecutor(max_workers=16)

        # ── Step 6: Auto batch size probe ──
        self._watchdog.start("batch_size_probe")
        print(f"[INIT] Step 6/6: Probing optimal batch size via VRAM measurement...", flush=True)
        self.batch_size = self._probe_max_batch_size()

        vram_used = torch.cuda.memory_allocated(0) / (1024**3)
        vram_reserved = torch.cuda.memory_reserved(0) / (1024**3)
        print(f"[INIT] GpuExtractor ready — total init: {time.time() - t0:.1f}s, "
              f"VRAM: {vram_used:.2f}GB used / {vram_reserved:.2f}GB reserved, "
              f"batch_size={self.batch_size}", flush=True)

    def _probe_max_batch_size(self) -> int:
        """Run a single-image inference under autocast, measure peak VRAM delta,
        then compute the largest power-of-2 batch that fits in 75% of free VRAM."""
        try:
            torch.cuda.empty_cache()
            baseline = torch.cuda.memory_allocated(0)
            torch.cuda.reset_peak_memory_stats(0)

            dummy = torch.randn(1, 3, 322, 322, device=self.device)
            dummy = (dummy - self.mean) / self.std
            with torch.no_grad():
                with torch.autocast('cuda', dtype=torch.float16):
                    _ = self.model(dummy)
            torch.cuda.synchronize()

            peak = torch.cuda.max_memory_allocated(0)
            del dummy
            torch.cuda.empty_cache()

            per_image = max(peak - baseline, 1)
            free = torch.cuda.mem_get_info(0)[0]
            max_batch = max(8, int(free * 0.75 / per_image))
            max_batch = min(max_batch, 512)
            max_batch = 2 ** int(math.log2(max_batch))

            print(f"[INIT]   Auto batch size: {max_batch} "
                  f"(per-image={per_image / 1e6:.1f} MB, free={free / 1e6:.0f} MB)", flush=True)
            return max_batch
        except Exception as e:
            print(f"[WARN] Batch size probe failed ({e}), defaulting to 32", flush=True)
            return 32

    @staticmethod
    def _decode_item(item: ViewItem):
        """Convert ViewItem RGB numpy array → float32 CHW CPU tensor."""
        try:
            return transforms.functional.to_tensor(item.view_data)
        except Exception as e:
            print(f"[WARN] Decode failed panoid={item.panoid}: {type(e).__name__}: {e}", flush=True)
            return None

    def start_decode(self, items: List[ViewItem]) -> list:
        """Non-blocking: submit numpy→tensor conversion for all items; return list of futures."""
        return [self.executor.submit(self._decode_item, item) for item in items]

    def _run_inference(self, items: List[ViewItem], valid_tensors: list, valid_indices: list):
        """GPU inference with pin_memory transfer, fp16 autocast, and OOM auto-retry."""
        try:
            images = torch.stack(valid_tensors).pin_memory().to(self.device, non_blocking=True)
            if images.shape[-2:] != (322, 322):
                images = torch.nn.functional.interpolate(
                    images, size=(322, 322), mode='bilinear', align_corners=False
                )
            images = (images - self.mean) / self.std

            with torch.no_grad():
                with torch.autocast('cuda', dtype=torch.float16):
                    feats = self.model(images)
            del images

            feats_np = feats.float().cpu().numpy()
            metadata_batch = [
                {'panoid': items[i].panoid, 'lat': items[i].lat, 'lng': items[i].lng}
                for i in valid_indices
            ]
            return feats_np, metadata_batch, valid_indices

        except torch.cuda.OutOfMemoryError:
            half = len(valid_tensors) // 2
            torch.cuda.empty_cache()
            if half == 0:
                print("[WARN] OOM on single image — skipping", flush=True)
                return None, [], []

            new_bs = max(8, half)
            print(f"[WARN] OOM on batch={len(valid_tensors)} → retrying as 2×{half}, "
                  f"shrinking batch_size {self.batch_size} → {new_bs}", flush=True)
            self.batch_size = new_bs

            f1, m1, vi1 = self._run_inference(items, valid_tensors[:half], valid_indices[:half])
            f2, m2, vi2 = self._run_inference(items, valid_tensors[half:], valid_indices[half:])

            if f1 is None and f2 is None:
                return None, [], []
            if f1 is None:
                return f2, m2, vi2
            if f2 is None:
                return f1, m1, vi1
            return np.concatenate([f1, f2], axis=0), m1 + m2, vi1 + vi2

    def infer_prefetched(self, items: List[ViewItem], futures: list):
        """Block on decode futures collected by start_decode(), then run GPU inference."""
        tensors_or_none = [f.result() for f in futures]
        valid_indices = [i for i, t in enumerate(tensors_or_none) if t is not None]
        valid_tensors = [tensors_or_none[i] for i in valid_indices]

        failures = len(items) - len(valid_tensors)
        if failures:
            print(f"[WARN] {failures}/{len(items)} images failed to decode in batch", flush=True)
        if not valid_tensors:
            return None, [], []
        return self._run_inference(items, valid_tensors, valid_indices)

    def extract_batch(self, items: List[ViewItem]):
        """Synchronous decode + GPU inference (no prefetch). Kept for compatibility."""
        tensors_or_none = list(self.executor.map(self._decode_item, items))
        valid_indices = [i for i, t in enumerate(tensors_or_none) if t is not None]
        valid_tensors = [tensors_or_none[i] for i in valid_indices]

        failures = len(items) - len(valid_tensors)
        if failures:
            print(f"[WARN] {failures}/{len(items)} images failed to decode in batch", flush=True)
        if not valid_tensors:
            return None, [], []
        return self._run_inference(items, valid_tensors, valid_indices)


# ═══════════════════════════════════════════════════════════════════════════════
# CSV Loader
# ═══════════════════════════════════════════════════════════════════════════════

def load_csv(csv_path: str) -> Tuple[List[dict], Dict[str, Dict]]:
    records = []
    metadata = {}
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        sample = f.read(4096)
        f.seek(0)
        delimiter = ',' if sample.count(',') >= sample.count(';') else ';'
        reader = csv.DictReader(f, delimiter=delimiter)

        col_map = {}
        if reader.fieldnames:
            for field in reader.fieldnames:
                clean = field.lower().strip().replace('_', '').replace('-', '')
                if clean == 'panoid':
                    col_map['panoid'] = field
                elif clean in ('buildid', 'build'):
                    col_map['build_id'] = field
                elif clean in ('lat', 'latitude'):
                    col_map['lat'] = field
                elif clean in ('lon', 'lng', 'longitude'):
                    col_map['lon'] = field
                elif clean in ('headingdeg', 'heading', 'yaw'):
                    col_map['heading'] = field

        if 'panoid' not in col_map:
            print(f"[ERROR] No panoid column in CSV. Columns: {reader.fieldnames}")
            sys.exit(1)
        if 'build_id' not in col_map:
            print(f"[ERROR] Apple pipeline requires a build_id column. Columns: {reader.fieldnames}")
            sys.exit(1)
        if 'lat' not in col_map or 'lon' not in col_map:
            print(f"[ERROR] Apple pipeline requires lat + lon (for coverage-tile metadata lookup). Columns: {reader.fieldnames}")
            sys.exit(1)

        for row in reader:
            panoid = row.get(col_map['panoid'], '').strip()
            build_id = row.get(col_map['build_id'], '').strip()
            if not panoid or not build_id:
                continue
            try:
                lat = float(row[col_map['lat']])
                lon = float(row[col_map['lon']])
            except (ValueError, KeyError):
                continue
            record = {
                'panoid': panoid,
                'build_id': build_id,
                'lat': lat,
                'lon': lon,
            }
            if 'heading' in col_map and row.get(col_map['heading']):
                try:
                    record['heading_deg'] = float(row[col_map['heading']])
                except ValueError:
                    pass
            records.append(record)
            if 'lat' in col_map and 'lon' in col_map:
                try:
                    lat = float(row.get(col_map['lat'], '').strip())
                    lon = float(row.get(col_map['lon'], '').strip())
                    metadata[panoid] = {'lat': round(lat, 5), 'lng': round(lon, 5)}
                except (ValueError, AttributeError):
                    pass
    return records, metadata


# ═══════════════════════════════════════════════════════════════════════════════
# Async Downloader (Apple Look Around)
# ═══════════════════════════════════════════════════════════════════════════════

# One Authenticator per process; fetch_face will rotate it on transient 403s.
_GLOBAL_AUTH = Authenticator()


def _process_apple_pano(face_heics, faces_meta, panoid, config, heading_deg):
    """
    Run inside ThreadPoolExecutor: decode 4 HEICs, reproject to equirect,
    extract directional perspective views, JPEG-encode each. Returns the
    Google-shaped result dict {success, error, views, view_filenames, ...}.
    """
    out = {'success': False, 'error': '', 'views': [], 'view_filenames': []}
    try:
        imgs = [decode_heic_bytes(h) for h in face_heics]
        if all(im is None for im in imgs[:4]):
            out['error'] = 'all 4 sides failed to decode'
            return out

        equi = reproject_faces_to_equirect(
            imgs, faces_meta, out_w=config.get('max_equirect_w', 4096)
        )
        equi_bgr = cv2.cvtColor(equi, cv2.COLOR_RGB2BGR)

        view_extractor = DirectionalViewExtractor()
        view_config = DirectionalViewConfig(
            output_resolution=config.get('view_resolution', 322),
            fov_degrees=config.get('view_fov', 60.0),
            num_views=config.get('num_views', 8),
            global_view=config.get('global_view', False),
            augment=config.get('augment', False),
            target_yaw=heading_deg,
            antialias_strength=0.0 if config.get('no_antialias') else config.get('aa_strength', 0.8),
            interpolation=config.get('interpolation', 'cubic'),
            yaw_offset=config.get('view_offset', 0.0),
        )
        view_result = view_extractor.extract_views(equi_bgr, view_config)
        if not view_result.success:
            out['error'] = view_result.error or 'view extraction failed'
            return out

        jpeg_q = config.get('jpeg_quality', 95)
        for i, (view, meta) in enumerate(zip(view_result.views, view_result.metadata)):
            yaw = meta['yaw']
            if config.get('global_view') or heading_deg is not None:
                fname = f"{panoid}_rnd_Y{int(yaw)}.jpg"
            else:
                fname = (f"{panoid}_zoom{config.get('zoom_level', 2)}"
                         f"_view{i:02d}_{yaw:.0f}deg.jpg")
            ok, buf = cv2.imencode('.jpg', view, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
            if ok:
                out['views'].append(buf.tobytes())
                out['view_filenames'].append(fname)
        out['success'] = bool(out['views'])
        del equi, equi_bgr, imgs
    except Exception as e:
        out['error'] = f'{type(e).__name__}: {e}'
    return out


async def _download_single_pano(session, record, sem, executor, config, item_queue, metadata, stats, shared_state):
    panoid_str = record['panoid']
    build_id = record.get('build_id')
    lat = record.get('lat')
    lon = record.get('lon')
    heading_deg = record.get('heading_deg')
    zoom_level = config['zoom_level']

    if not build_id or lat is None or lon is None:
        stats['dl_fail'] += 1
        shared_state.log_failure(panoid_str, 'missing build_id/lat/lon')
        return

    retries = 3
    for attempt in range(1, retries + 1):
        try:
            async with sem:
                # Camera metadata via cached coverage tile (free hit if
                # prefetch_coverage_tiles_for_records was called upfront).
                faces_meta = await get_camera_metadata(session, panoid_str, lat, lon)
                if faces_meta is None or len(faces_meta) < 4:
                    if attempt < retries:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    stats['dl_fail'] += 1
                    shared_state.log_failure(panoid_str, 'no_camera_metadata')
                    return

                # 4 side faces in parallel (top/bottom skipped — perspective
                # views never sample beyond the sides' vertical FOV).
                face_heics = await fetch_all_faces(
                    session, panoid_str, build_id, zoom_level, _GLOBAL_AUTH, n_faces=4
                )
                if sum(1 for h in face_heics if h) < 4:
                    if attempt < retries:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    stats['dl_fail'] += 1
                    shared_state.log_failure(panoid_str, 'face_fetch_failed')
                    return

                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    executor, _process_apple_pano,
                    face_heics, faces_meta, panoid_str, config, heading_deg,
                )
                del face_heics

                if not result['success'] or not result['views']:
                    if attempt < retries:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    stats['dl_fail'] += 1
                    shared_state.log_failure(panoid_str, result.get('error') or 'process_failed')
                    return

                meta = metadata.get(panoid_str, {'lat': 0.0, 'lng': 0.0})
                for view_data, _ in zip(result['views'], result['view_filenames']):
                    item = ViewItem(panoid_str, view_data, meta['lat'], meta['lng'])
                    while True:
                        try:
                            item_queue.put(item, timeout=1.0)
                            break
                        except queue.Full:
                            continue

                stats['dl_ok'] += 1
                stats['views_produced'] += len(result['views'])
                del result
                return

        except Exception as e:
            if attempt < retries:
                await asyncio.sleep(2 ** attempt)
            else:
                stats['dl_fail'] += 1
                shared_state.log_failure(panoid_str, f"exception: {e}")

async def _run_downloader(records, config, item_queue, metadata, stats, shared_state):
    from aiohttp import ClientTimeout
    sem = asyncio.Semaphore(config['max_threads'])
    connector = aiohttp.TCPConnector(limit=600, limit_per_host=200, ttl_dns_cache=300)
    timeout = ClientTimeout(total=15, connect=8)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Pre-fetch every unique z=17 coverage tile so per-pano metadata
        # lookups become free dict hits — same trick we use in Stage 2.
        try:
            n_tiles = await prefetch_coverage_tiles_for_records(
                session, records, concurrency=min(100, config['max_threads']),
            )
            if n_tiles:
                print(f"[INFO] Pre-fetched {n_tiles} coverage tiles", flush=True)
        except Exception as e:
            print(f"[WARN] Coverage tile prefetch failed: {e}", flush=True)

        with ThreadPoolExecutor(max_workers=config['workers']) as executor:
            CHUNK = 5000
            for i in range(0, len(records), CHUNK):
                chunk = records[i:i + CHUNK]
                tasks = [
                    _download_single_pano(session, rec, sem, executor, config,
                                          item_queue, metadata, stats, shared_state)
                    for rec in chunk
                ]
                await asyncio.gather(*tasks, return_exceptions=True)

    item_queue.put(_SENTINEL)
    stats['dl_done'] = True

def downloader_thread(records, config, item_queue, metadata, stats, shared_state):
    asyncio.run(_run_downloader(records, config, item_queue, metadata, stats, shared_state))


# ═══════════════════════════════════════════════════════════════════════════════
# Disk Space Management
# ═══════════════════════════════════════════════════════════════════════════════

def get_free_gb(path: str = '/') -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)

def wait_for_disk_space(path: str = '/', min_gb: float = MIN_FREE_GB):
    while get_free_gb(path) < min_gb:
        print(f"[WARN] Only {get_free_gb(path):.1f}GB free, waiting for space (need {min_gb}GB)...")
        time.sleep(60)


# ═══════════════════════════════════════════════════════════════════════════════
# Log Capture & Upload
# ═══════════════════════════════════════════════════════════════════════════════

LOG_FILE = f"/tmp/worker_{INSTANCE_ID or 'unknown'}.log"


class TeeWriter:
    """Writes to both the original stream and a log file."""
    def __init__(self, original, log_file_handle):
        self.original = original
        self.log_file = log_file_handle

    def write(self, data):
        self.original.write(data)
        try:
            self.log_file.write(data)
        except Exception:
            pass

    def flush(self):
        self.original.flush()
        try:
            self.log_file.flush()
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self.original, name)


def _start_log_capture():
    """Tee stdout and stderr to a log file."""
    try:
        fh = open(LOG_FILE, "w", encoding="utf-8", errors="replace")
        sys.stdout = TeeWriter(sys.__stdout__, fh)
        sys.stderr = TeeWriter(sys.__stderr__, fh)
        return fh
    except Exception as e:
        print(f"[WARN] Could not start log capture: {e}")
        return None


def upload_logs_to_r2():
    """Upload the captured log file to R2."""
    try:
        sys.stdout.flush()
        sys.stderr.flush()

        r2 = R2Client()
        log_key = f"Logs/{FEATURES_BUCKET_PREFIX}/worker_{INSTANCE_ID}.log"
        r2.upload_file(LOG_FILE, log_key)
        print(f"[INFO] Uploaded logs to R2: {log_key}")
    except Exception as e:
        print(f"[WARN] Failed to upload logs to R2: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Self-Destruct
# ═══════════════════════════════════════════════════════════════════════════════

def self_destruct():
    """Destroy this Vast.ai instance — retries forever until the instance is gone."""
    instance_id = INSTANCE_ID

    if not instance_id:
        # Fallback: try to detect from vastai CLI
        if VAST_API_KEY:
            try:
                result = subprocess.run(
                    ["vastai", "--api-key", VAST_API_KEY, "show", "instances", "--raw"],
                    capture_output=True, text=True, timeout=30
                )
                if result.returncode == 0 and result.stdout.strip():
                    instances = json.loads(result.stdout.strip())
                    if instances and len(instances) == 1:
                        instance_id = str(instances[0].get('id', ''))
            except Exception as e:
                print(f"[WARN] Instance ID auto-detect failed: {e}")

    if not instance_id:
        print("[WARN] Cannot self-destruct: unable to determine INSTANCE_ID — sleeping forever")
        while True:
            time.sleep(3600)
    if not VAST_API_KEY:
        print("[WARN] Cannot self-destruct: VAST_API_KEY not set — sleeping forever")
        while True:
            time.sleep(3600)

    cmd = ["vastai", "--api-key", VAST_API_KEY, "destroy", "instance", str(instance_id)]
    attempt = 0
    while True:
        attempt += 1
        try:
            print(f"[INFO] Self-destruct attempt {attempt} for instance {instance_id}...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            print(f"[INFO] Self-destruct response: exit={result.returncode} stdout='{stdout}' stderr='{stderr}'")
            if result.returncode == 0:
                print(f"[INFO] Instance {instance_id} destroyed successfully.")
                return
            print(f"[WARN] Self-destruct attempt {attempt} failed (exit {result.returncode}) — retrying in 30s")
        except Exception as e:
            print(f"[WARN] Self-destruct attempt {attempt} exception: {e} — retrying in 30s")
        time.sleep(30)


# ═══════════════════════════════════════════════════════════════════════════════
# Upload with Retry
# ═══════════════════════════════════════════════════════════════════════════════

MAX_EXTENDED_RETRIES = 30


def upload_with_retry(r2, local_path, bucket_key, label="FILE", max_attempts=5):
    """Upload a file to R2 with exponential backoff and extended retry."""
    file_size = os.path.getsize(local_path)

    for attempt in range(1, max_attempts + 1):
        print(f"[INFO] Uploading {label} ({file_size / (1024**2):.1f} MB), attempt {attempt}/{max_attempts}...")
        if r2.upload_file(local_path, bucket_key, max_retries=1):
            return True
        print(f"[WARN] Upload attempt {attempt}/{max_attempts} failed for {bucket_key}")
        if attempt < max_attempts:
            wait = min(2 ** attempt, 120)
            time.sleep(wait)

    # Extended retry with client reset — check if file already landed on R2
    # (upload may succeed server-side but connection resets before response)
    for retry_count in range(1, MAX_EXTENDED_RETRIES + 1):
        try:
            if r2.file_exists(bucket_key):
                print(f"[INFO] File already on R2 (uploaded but response lost): {bucket_key}")
                return True
        except Exception:
            pass
        print(f"[WARN] Extended retry #{retry_count}/{MAX_EXTENDED_RETRIES} for {bucket_key}")
        r2.reset_client()
        time.sleep(60)
        if r2.upload_file(local_path, bucket_key, max_retries=1):
            return True

    # Final existence check before declaring permanent failure
    try:
        if r2.file_exists(bucket_key):
            print(f"[INFO] File already on R2 after retries exhausted: {bucket_key}")
            return True
    except Exception:
        pass

    print(f"[ERROR] Upload permanently failed for {bucket_key}")
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# Process a Single Chunk
# ═══════════════════════════════════════════════════════════════════════════════

def process_chunk(r2, tq: TaskQueue, extractor: GpuExtractor, chunk_id: str, work_dir: Path,
                  preloaded=None, chunks_done_so_far: int = 0, redis_chunk_id: str = None):
    """
    Extract features for a chunk. Returns (features_file, metadata_file, local_csv)
    for async upload, or None if chunk was empty.

    preloaded: optional (local_csv, records, metadata_map) to skip CSV download.
    redis_chunk_id: global chunk ID for Redis ops (batch mode). Defaults to chunk_id.
    """
    _rcid = redis_chunk_id or chunk_id
    city = CITY_NAME

    if preloaded:
        local_csv, records, metadata_map = preloaded
        total_records = len(records)
        print(f"[CHUNK {chunk_id}] Using pre-fetched CSV ({total_records} panos)")
    else:
        csv_filename = f"{city}_{chunk_id}.csv"
        csv_key = f"{CSV_BUCKET_PREFIX}/{csv_filename}"
        local_csv = str(work_dir / csv_filename)
        print(f"[CHUNK {chunk_id}] Downloading {csv_key}...")
        try:
            tq.report_status(REGION, INSTANCE_ID, "DOWNLOADING", chunk_id=_rcid)
        except Exception:
            pass
        if not r2.download_file(csv_key, local_csv, max_retries=3):
            raise RuntimeError(f"Failed to download {csv_key}")
        records, metadata_map = load_csv(local_csv)
        total_records = len(records)

    views_per_pano = HARDCODED_CONFIG['num_views']
    total_views_est = total_records * views_per_pano
    feature_dim = 8448

    print(f"[CHUNK {chunk_id}] {total_records} panos, ~{total_views_est} views expected")

    if total_records == 0:
        print(f"[CHUNK {chunk_id}] Empty chunk, skipping")
        _cleanup_chunk_files(work_dir, chunk_id, local_csv)
        return None

    # ── Step 3: Setup output files ──
    out_base = _output_base(city, chunk_id)
    features_file = str(work_dir / f"{out_base}.npy")
    metadata_file = str(work_dir / f"Metadata_{out_base}.jsonl")
    failed_file = str(work_dir / f"failed_{chunk_id}.jsonl")

    # ~270MB for 1K panos × 8 views × 8448 dim × 4 bytes
    features_memmap = np.lib.format.open_memmap(
        features_file, mode='w+', dtype='float32',
        shape=(total_views_est, feature_dim)
    )

    shared_state = SharedState(features_memmap, metadata_file, failed_file, start_idx=0)

    # ── Step 4: Download + extract ──
    dl_config = dict(HARDCODED_CONFIG)
    # No row-skip optimisation on Apple — faces are independent lens
    # projections, not equirect rows; dropping any of the 4 sides leaves
    # holes in the panorama.
    print(f"[INFO] HEIC decoder: {get_heic_decoder_name()}", flush=True)

    item_queue = queue.Queue(maxsize=dl_config['queue_size'])
    stats = {'dl_ok': 0, 'dl_fail': 0, 'ext_ok': 0, 'views_produced': 0, 'dl_done': False}

    dl_thread = threading.Thread(
        target=downloader_thread,
        args=(records, dl_config, item_queue, metadata_map, stats, shared_state)
    )
    dl_thread.start()

    # ── Extraction loop ──
    loop_start = time.time()
    last_progress_time = time.time()
    last_progress_count = 0
    last_log_time = time.time()
    last_heartbeat_time = time.time()
    batch_times = []
    STALL_TIMEOUT = int(os.environ.get('STALL_TIMEOUT', '600'))
    LOG_INTERVAL = 30
    HEARTBEAT_INTERVAL = 30

    print(f"[CHUNK {chunk_id}] Starting extraction (batch_size={extractor.batch_size})", flush=True)
    try:
        tq.report_status(REGION, INSTANCE_ID, "EXTRACTING", chunk_id=_rcid,
                         total=total_views_est)
    except Exception:
        pass

    pending_batch: List[ViewItem] = []
    pending_futures = None

    try:
        while True:
            wait_for_disk_space(str(work_dir), MIN_FREE_GB)

            now = time.time()

            # ── Redis heartbeat + status report ──
            if now - last_heartbeat_time >= HEARTBEAT_INTERVAL:
                try:
                    tq.heartbeat(REGION, INSTANCE_ID, _rcid)
                    elapsed = now - loop_start
                    _spd = stats['ext_ok'] / elapsed if elapsed > 0 else 0
                    _rem = total_views_est - stats['ext_ok']
                    _eta = _rem / _spd if _spd > 0 else 0
                    tq.report_status(
                        REGION, INSTANCE_ID, "EXTRACTING",
                        chunk_id=_rcid, chunks_done=chunks_done_so_far,
                        processed=stats['ext_ok'], total=total_views_est,
                        speed=_spd, eta=_eta,
                    )
                except Exception as e:
                    print(f"[WARN] Heartbeat failed: {e}", flush=True)
                last_heartbeat_time = now

            # ── Stall detection ──
            since_progress = now - last_progress_time
            if since_progress > STALL_TIMEOUT and stats['ext_ok'] == last_progress_count:
                msg = (f"[FATAL] Chunk {chunk_id} stalled — no progress for {since_progress:.0f}s")
                print(msg, flush=True)
                raise RuntimeError(msg)

            # ── Periodic status log ──
            if now - last_log_time >= LOG_INTERVAL:
                elapsed = now - loop_start
                speed = stats['ext_ok'] / elapsed if elapsed > 0 else 0
                remaining = total_views_est - stats['ext_ok']
                eta = remaining / speed if speed > 0 else 0
                pct = int(stats['ext_ok'] / total_views_est * 100) if total_views_est > 0 else 0
                print(f"PROGRESS|{INSTANCE_ID}|{chunk_id}|{stats['ext_ok']}|{total_views_est}|EXTRACTING", flush=True)
                print(f"[STATS] chunk={chunk_id} | "
                      f"views={stats['ext_ok']:,}/{total_views_est:,} ({pct}%) | "
                      f"speed={speed:.1f} views/s | eta={eta/60:.1f}min | "
                      f"dl_ok={stats['dl_ok']} | dl_fail={stats['dl_fail']} | "
                      f"queue={item_queue.qsize()}", flush=True)
                last_log_time = now

            # ── Fill next batch ──
            current_batch: List[ViewItem] = []
            while len(current_batch) < extractor.batch_size:
                try:
                    item = item_queue.get(timeout=0.01)
                    if item is _SENTINEL:
                        continue
                    current_batch.append(item)
                except queue.Empty:
                    break

            # ── No new items: drain pending then check for exit ──
            if not current_batch:
                if pending_batch and pending_futures is not None:
                    batch_start = time.time()
                    try:
                        feats_np, meta_batch, _ = extractor.infer_prefetched(pending_batch, pending_futures)
                        if feats_np is not None and len(meta_batch) > 0:
                            shared_state.write_batch(feats_np, meta_batch)
                            stats['ext_ok'] += len(meta_batch)
                            last_progress_time = time.time()
                            last_progress_count = stats['ext_ok']
                            del feats_np, meta_batch
                        batch_times.append(time.time() - batch_start)
                        if stats['ext_ok'] % 5000 == 0:
                            gc.collect()
                    except Exception as e:
                        print(f"[ERROR] Batch extraction failed: {type(e).__name__}: {e}", flush=True)
                        import traceback
                        traceback.print_exc()
                    finally:
                        pending_batch = []
                        pending_futures = None
                if not dl_thread.is_alive():
                    break
                continue

            # ── Submit decode of current batch ──
            current_futures = extractor.start_decode(current_batch)

            # ── GPU inference on pending batch ──
            if pending_batch and pending_futures is not None:
                batch_start = time.time()
                try:
                    feats_np, meta_batch, _ = extractor.infer_prefetched(pending_batch, pending_futures)
                    if feats_np is not None and len(meta_batch) > 0:
                        shared_state.write_batch(feats_np, meta_batch)
                        stats['ext_ok'] += len(meta_batch)
                        last_progress_time = time.time()
                        last_progress_count = stats['ext_ok']
                        del feats_np, meta_batch
                    batch_times.append(time.time() - batch_start)
                    if stats['ext_ok'] % 5000 == 0:
                        gc.collect()
                except Exception as e:
                    print(f"[ERROR] Batch extraction failed: {type(e).__name__}: {e}", flush=True)
                    import traceback
                    traceback.print_exc()

            # ── Promote current batch to pending ──
            pending_batch = current_batch
            pending_futures = current_futures

    except KeyboardInterrupt:
        print("[WARN] Interrupted", flush=True)

    dl_thread.join()
    final_count = shared_state.write_idx
    shared_state.close()

    # ── Truncate memmap to actual size ──
    del features_memmap
    gc.collect()

    if final_count == 0 and total_records > 0:
        print(f"[ERROR] Chunk {chunk_id}: 0 features from {total_records} panos!")
        _cleanup_chunk_files(work_dir, chunk_id, local_csv)
        raise RuntimeError(f"Zero features extracted from chunk {chunk_id}")

    if final_count > 0 and final_count < total_views_est:
        print(f"[CHUNK {chunk_id}] Truncating features: {total_views_est} → {final_count}")
        mm = np.lib.format.open_memmap(features_file, mode='r+')
        truncated = mm[:final_count].copy()
        del mm
        np.save(features_file, truncated)
        del truncated

    print(f"[CHUNK {chunk_id}] Extraction complete: {final_count} features")

    # Return file paths for async upload (caller handles upload + cleanup)
    return features_file, metadata_file, local_csv


def _cleanup_chunk_files(work_dir: Path, chunk_id: str, local_csv: str = None):
    """Delete local files for a processed chunk to free disk space."""
    city = CITY_NAME
    out_base = _output_base(city, chunk_id)
    for f in [
        str(work_dir / f"{out_base}.npy"),
        str(work_dir / f"Metadata_{out_base}.jsonl"),
        str(work_dir / f"failed_{chunk_id}.jsonl"),
    ]:
        try:
            os.remove(f)
        except OSError:
            pass
    if local_csv:
        try:
            os.remove(local_csv)
        except OSError:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# Async Upload (parallel NPY + metadata)
# ═══════════════════════════════════════════════════════════════════════════════

def upload_chunk_files(r2, chunk_id: str, features_file: str, metadata_file: str,
                       local_csv: str, work_dir: Path):
    """Upload NPY + metadata to R2 in parallel, then cleanup local files."""
    city = CITY_NAME
    out_base = _output_base(city, chunk_id)
    npy_key = f"{FEATURES_BUCKET_PREFIX}/{out_base}.npy"
    meta_key = f"{FEATURES_BUCKET_PREFIX}/Metadata_{out_base}.jsonl"

    errors = []

    def _upload_npy():
        if os.path.exists(features_file) and os.path.getsize(features_file) > 0:
            if not upload_with_retry(r2, features_file, npy_key, label="NPY"):
                errors.append(f"Failed to upload NPY for chunk {chunk_id}")

    def _upload_meta():
        if os.path.exists(metadata_file) and os.path.getsize(metadata_file) > 0:
            if not upload_with_retry(r2, metadata_file, meta_key, label="META"):
                errors.append(f"Failed to upload metadata for chunk {chunk_id}")

    npy_t = threading.Thread(target=_upload_npy)
    meta_t = threading.Thread(target=_upload_meta)
    npy_t.start()
    meta_t.start()
    npy_t.join()
    meta_t.join()

    if errors:
        raise RuntimeError("; ".join(errors))

    _cleanup_chunk_files(work_dir, chunk_id, local_csv)
    print(f"[CHUNK {chunk_id}] Upload + cleanup complete")


def _do_background_upload(error_ref, r2, chunk_id, features_file, metadata_file,
                          local_csv, work_dir):
    """Thread target: upload chunk files and store any error in error_ref[0]."""
    try:
        upload_chunk_files(r2, chunk_id, features_file, metadata_file, local_csv, work_dir)
    except Exception as e:
        error_ref[0] = e


# ═══════════════════════════════════════════════════════════════════════════════
# Prefetch Next Chunk (background CSV download during extraction)
# ═══════════════════════════════════════════════════════════════════════════════

def _do_prefetch(result_ref, r2, tq, work_dir, skip_prefixes=None):
    """Thread target: claim next chunk + download/parse CSV. Stores result in result_ref[0].
    skip_prefixes: shared set of city CSV prefixes to skip (missing on R2)."""
    try:
        next_id = tq.claim_task(REGION, INSTANCE_ID)
        if next_id is None:
            return

        # Batch mode: look up per-chunk metadata
        _bmeta = None
        try:
            _bmeta = tq.get_chunk_meta(REGION, next_id)
        except Exception:
            pass

        if _bmeta:
            city = _bmeta['city_name']
            csv_prefix = _bmeta['csv_prefix']
            file_cid = f"chunk_{_bmeta['chunk_num']:04d}"
        else:
            city = CITY_NAME
            csv_prefix = CSV_BUCKET_PREFIX
            file_cid = next_id

        # Skip if this city's CSVs are known missing
        if skip_prefixes and csv_prefix in skip_prefixes:
            print(f"[PREFETCH] Skipping {next_id} — city '{city}' CSVs missing on R2")
            try:
                tq.fail_task(REGION, next_id, INSTANCE_ID,
                             f"city_csv_missing:{csv_prefix}")
            except Exception:
                pass
            return

        csv_fn = f"{city}_{file_cid}.csv"
        csv_key = f"{csv_prefix}/{csv_fn}"
        local_csv = str(work_dir / csv_fn)

        print(f"[PREFETCH] Downloading CSV for next chunk {next_id}...")
        if not r2.download_file(csv_key, local_csv, max_retries=3):
            print(f"[PREFETCH] CSV download failed for {next_id}, returning to queue")
            # Blacklist this city prefix
            if skip_prefixes is not None:
                skip_prefixes.add(csv_prefix)
                print(f"[PREFETCH] Blacklisting city prefix '{csv_prefix}' — "
                      f"all future chunks from {city} will be skipped")
            try:
                tq.fail_task(REGION, next_id, INSTANCE_ID,
                             f"city_csv_missing:{csv_prefix}")
            except Exception:
                pass
            return

        records, metadata_map = load_csv(local_csv)
        result_ref[0] = (next_id, local_csv, records, metadata_map, _bmeta)
        print(f"[PREFETCH] Ready: chunk {next_id} ({len(records)} panos)")
    except Exception as e:
        print(f"[PREFETCH] Error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Resume — Reconcile Redis with R2 (source of truth)
# ═══════════════════════════════════════════════════════════════════════════════

def reconcile_with_r2(r2, tq: TaskQueue):
    """
    Scan R2 for already-uploaded NPY+metadata files and mark those chunks done
    in Redis. Also reclaim stale active tasks and recover lost tasks.
    This makes the pipeline fully resumable after worker crashes.
    """
    # Batch mode: skip R2 scan (chunks have mixed paths), just do stale recovery
    try:
        if (tq.redis.hlen(f"job:{REGION}:cmap") or 0) > 0:
            print("[RESUME] Batch mode — skipping R2 reconciliation, using stale recovery only")
            stale = tq.reclaim_stale(REGION)
            if stale:
                print(f"[RESUME] Reclaimed {len(stale)} stale tasks: {stale}")
            lost = tq.recover_lost_tasks(REGION)
            if lost:
                print(f"[RESUME] Recovered {len(lost)} lost tasks: {lost}")
            progress = tq.get_progress(REGION)
            print(f"[RESUME] After recovery — todo: {progress['todo']}, "
                  f"active: {progress['active']}, done: {progress['done']}/{progress['total_chunks']}")
            return
    except Exception:
        pass

    city = CITY_NAME
    prefix = f"{FEATURES_BUCKET_PREFIX}/{city}_"

    print(f"[RESUME] Scanning R2 for existing outputs under '{prefix}'...")

    # List NPY and metadata files already in R2
    npy_keys = set(r2.list_objects(prefix, suffix='.npy'))
    meta_keys = set(r2.list_objects(
        f"{FEATURES_BUCKET_PREFIX}/Metadata_{city}_", suffix='.jsonl'
    ))

    # Extract chunk IDs from NPY filenames: "Features/KansasCity_1.11.npy" → "chunk_0001"
    import re as _re
    npy_pattern = _re.compile(rf'^{_re.escape(city)}_(\d+)\.\d+\.npy$')
    done_chunks = set()
    for key in npy_keys:
        filename = key.rsplit('/', 1)[-1]
        m = npy_pattern.match(filename)
        if not m:
            continue
        chunk_num = int(m.group(1))
        chunk_id = f"chunk_{chunk_num:04d}"
        # Verify metadata also exists
        expected_meta = f"{FEATURES_BUCKET_PREFIX}/Metadata_{city}_{chunk_num}.{TOTAL_CHUNKS}.jsonl"
        if expected_meta in meta_keys:
            done_chunks.add(chunk_id)

    if done_chunks:
        print(f"[RESUME] Found {len(done_chunks)} completed chunks on R2")
        reconciled = tq.reconcile_done(REGION, done_chunks)
        if reconciled > 0:
            print(f"[RESUME] Marked {reconciled} chunks as done in Redis (were stale/orphaned)")

    # Reclaim active tasks from dead workers (stale > 5 min)
    stale = tq.reclaim_stale(REGION)
    if stale:
        print(f"[RESUME] Reclaimed {len(stale)} stale tasks: {stale}")

    # Recover any tasks that fell through the cracks (LPOP succeeded but HSET failed)
    lost = tq.recover_lost_tasks(REGION)
    if lost:
        print(f"[RESUME] Recovered {len(lost)} lost tasks: {lost}")

    progress = tq.get_progress(REGION)
    print(f"[RESUME] After reconciliation — todo: {progress['todo']}, "
          f"active: {progress['active']}, done: {progress['done']}/{progress['total_chunks']}")

    return done_chunks


# ═══════════════════════════════════════════════════════════════════════════════
# Main Pipeline — Pipelined Chunk Queue Loop
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_instance_id():
    """Detect Vast.ai instance ID at runtime."""
    global INSTANCE_ID
    if INSTANCE_ID:
        return
    # Try Vast.ai CLI detection
    if VAST_API_KEY:
        try:
            result = subprocess.run(
                ["vastai", "--api-key", VAST_API_KEY, "show", "instances", "--raw"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                instances = json.loads(result.stdout.strip())
                if instances and len(instances) == 1:
                    INSTANCE_ID = str(instances[0].get('id', ''))
        except Exception:
            pass
    # Final fallback: hostname
    if not INSTANCE_ID:
        import socket
        INSTANCE_ID = socket.gethostname()


def main():
    global CITY_NAME, CSV_BUCKET_PREFIX, FEATURES_BUCKET_PREFIX, TOTAL_CHUNKS

    work_dir = Path('/app/work')
    work_dir.mkdir(parents=True, exist_ok=True)

    _detect_instance_id()
    print(f"[INFO] Worker {INSTANCE_ID} starting")
    print(f"[INFO] City: {CITY_NAME}")
    print(f"[INFO] Region: {REGION}")
    print(f"[INFO] CSV prefix: {CSV_BUCKET_PREFIX}")
    print(f"[INFO] Features prefix: {FEATURES_BUCKET_PREFIX}")
    print(f"[INFO] Redis: {REDIS_URL[:40]}...")

    # ── Init R2 client ──
    r2 = R2Client()

    # ── Init Redis task queue ──
    if not REDIS_URL or not REDIS_TOKEN:
        print("[FATAL] REDIS_URL and REDIS_TOKEN must be set")
        sys.exit(1)
    if not REGION:
        print("[FATAL] REGION must be set (e.g. 'AU/Queensland/Brisbane')")
        sys.exit(1)

    tq = TaskQueue(REDIS_URL, REDIS_TOKEN)
    try:
        tq.report_status(REGION, INSTANCE_ID, "STARTING")
    except Exception:
        pass

    # Verify connection and job exists
    progress = tq.get_progress(REGION)
    print(f"[INFO] Job status: {progress}")
    if progress['total_chunks'] == 0:
        print("[FATAL] No job found in Redis for this region")
        sys.exit(1)

    # Set TOTAL_CHUNKS so output filenames include the total
    global TOTAL_CHUNKS
    TOTAL_CHUNKS = progress['total_chunks']
    print(f"[INFO] Total chunks: {TOTAL_CHUNKS} — output naming: {{city}}_N.{TOTAL_CHUNKS}.npy")

    # ── Resume: reconcile Redis state with R2 reality ──
    # Scans R2 for already-uploaded NPY files, marks them done in Redis,
    # reclaims stale tasks from dead workers, recovers lost tasks.
    reconcile_with_r2(r2, tq)

    # Check if job is already fully done after reconciliation
    if tq.is_complete(REGION):
        print("[INFO] Job already complete (all chunks uploaded to R2). Self-destructing.")
        upload_logs_to_r2()
        self_destruct()
        return

    # ── Init GPU extractor ONCE (expensive — 30-60s) ──
    print("[INFO] Initializing GPU extractor...")
    try:
        tq.report_status(REGION, INSTANCE_ID, "LOADING_MODEL")
    except Exception:
        pass
    extractor = GpuExtractor()

    # ── Pipelined chunk queue loop ──
    # Overlaps: upload of chunk N runs in background while chunk N+1 extracts.
    # Prefetch: next chunk's CSV is downloaded during current extraction.
    chunks_done = 0
    chunks_failed = 0
    idle_cycles = 0
    idle_start_time = None
    MAX_IDLE_CYCLES = 6
    MAX_IDLE_SECONDS = 600  # 10 min hard idle timeout → self-destruct
    skip_city_prefixes = set()  # CSV prefixes with missing R2 files — skip entire city

    # Pipeline state
    pending_upload = None   # (thread, redis_chunk_id, error_ref)
    prefetched = None       # (chunk_id, local_csv, records, metadata_map[, batch_meta])

    while True:
        # ── Drain previous background upload ──
        if pending_upload is not None:
            ul_thread, ul_chunk_id, ul_error = pending_upload
            ul_thread.join()
            if ul_error[0] is not None:
                chunks_failed += 1
                err = ul_error[0]
                print(f"[ERROR] Background upload for {ul_chunk_id} failed: {err}")
                try:
                    tq.fail_task(REGION, ul_chunk_id, INSTANCE_ID,
                                 f"{type(err).__name__}: {str(err)[:200]}")
                except Exception:
                    pass
            else:
                if _redis_retry(tq.complete_task, REGION, ul_chunk_id, INSTANCE_ID,
                                label=f"complete_task({ul_chunk_id})"):
                    chunks_done += 1
                    print(f"[INFO] Completed chunk {ul_chunk_id} ({chunks_done} total)")
                else:
                    # Data is on R2 — reconcile will fix this on next worker startup
                    print(f"[WARN] Could not mark {ul_chunk_id} done in Redis "
                          "(data safe on R2, will reconcile)")
                    chunks_done += 1
            pending_upload = None

        # ── Get next chunk: from prefetch or fresh claim ──
        _bmeta = None  # batch metadata for current chunk
        if prefetched is not None:
            chunk_id = prefetched[0]
            pf_csv, pf_records, pf_meta = prefetched[1], prefetched[2], prefetched[3]
            _bmeta = prefetched[4] if len(prefetched) > 4 else None
            preloaded_data = (pf_csv, pf_records, pf_meta)
            prefetched = None
        else:
            try:
                chunk_id = tq.claim_task(REGION, INSTANCE_ID)
            except Exception as e:
                print(f"[WARN] Redis claim_task failed: {e} — retrying in 10s")
                time.sleep(10)
                continue

            if chunk_id is None:
                try:
                    if tq.is_complete(REGION):
                        print(f"[INFO] Job complete! Processed {chunks_done} chunks. Self-destructing.")
                        break
                except Exception as e:
                    print(f"[WARN] Redis is_complete check failed: {e}")

                idle_cycles += 1
                if idle_start_time is None:
                    idle_start_time = time.time()
                try:
                    tq.report_status(REGION, INSTANCE_ID, "IDLE",
                                     chunks_done=chunks_done)
                except Exception:
                    pass

                # Periodically reclaim stale tasks from dead/stuck workers
                if idle_cycles % MAX_IDLE_CYCLES == 0:
                    try:
                        reclaimed = tq.reclaim_stale(REGION)
                        if reclaimed:
                            print(f"[IDLE] Reclaimed {len(reclaimed)} stale tasks: {reclaimed}")
                    except Exception:
                        pass
                    try:
                        if tq.is_complete(REGION):
                            print(f"[INFO] Job complete after idle wait. Self-destructing.")
                            break
                    except Exception:
                        pass

                # Hard idle timeout — self-destruct if idle too long
                idle_elapsed = time.time() - idle_start_time
                if idle_elapsed >= MAX_IDLE_SECONDS:
                    print(f"[INFO] Idle for {idle_elapsed:.0f}s with no work. Self-destructing.")
                    break
                time.sleep(5)  # Reduced from 30s for faster queue response
                continue

            preloaded_data = None

            # Look up batch metadata for freshly claimed chunk
            try:
                _bmeta = tq.get_chunk_meta(REGION, chunk_id)
            except Exception:
                pass

        idle_cycles = 0
        idle_start_time = None

        # ── Batch mode: override globals with per-chunk paths ──
        _redis_cid = chunk_id  # preserve original for Redis ops
        if _bmeta:
            CITY_NAME = _bmeta['city_name']
            CSV_BUCKET_PREFIX = _bmeta['csv_prefix']
            FEATURES_BUCKET_PREFIX = _bmeta['features_prefix']
            TOTAL_CHUNKS = _bmeta['city_total']
            chunk_id = f"chunk_{_bmeta['chunk_num']:04d}"  # local file chunk ID
            print(f"[BATCH] {_redis_cid} → {CITY_NAME} {chunk_id} "
                  f"(csv={CSV_BUCKET_PREFIX}, feat={FEATURES_BUCKET_PREFIX})")

            # Skip entire city if its CSV prefix previously 404'd
            if CSV_BUCKET_PREFIX in skip_city_prefixes:
                print(f"[SKIP-CITY] {CITY_NAME} — CSVs missing on R2, skipping {_redis_cid}")
                try:
                    tq.fail_task(REGION, _redis_cid, INSTANCE_ID,
                                 f"city_csv_missing:{CSV_BUCKET_PREFIX}")
                except Exception:
                    pass
                prefetched = None
                continue

        # ── Skip if output already exists on R2 (avoids re-extracting) ──
        out_base = _output_base(CITY_NAME, chunk_id)
        npy_key = f"{FEATURES_BUCKET_PREFIX}/{out_base}.npy"
        try:
            if r2.file_exists(npy_key):
                print(f"[SKIP] Chunk {chunk_id} output already on R2 — marking done")
                try:
                    tq.complete_task(REGION, _redis_cid, INSTANCE_ID)
                except Exception:
                    pass
                chunks_done += 1
                prefetched = None
                continue
        except Exception:
            pass  # If check fails, process normally

        print(f"\n{'='*60}")
        print(f"[INFO] {'Pre-fetched' if preloaded_data else 'Claimed'} chunk: {chunk_id} "
              f"(completed so far: {chunks_done})")
        print(f"{'='*60}")

        # ── Prefetch next chunk's CSV in background during extraction ──
        pf_result = [None]
        pf_thread = threading.Thread(
            target=_do_prefetch, args=(pf_result, r2, tq, work_dir, skip_city_prefixes), daemon=True
        )
        pf_thread.start()

        # ── Process current chunk ──
        try:
            result = process_chunk(r2, tq, extractor, chunk_id, work_dir,
                                   preloaded=preloaded_data,
                                   chunks_done_so_far=chunks_done,
                                   redis_chunk_id=_redis_cid)

            # Wait for prefetch to finish before starting upload
            pf_thread.join()
            prefetched = pf_result[0]

            if result is None:
                # Empty chunk — mark complete immediately
                tq.complete_task(REGION, _redis_cid, INSTANCE_ID)
                chunks_done += 1
                print(f"[INFO] Completed empty chunk {chunk_id} ({chunks_done} total)")
            else:
                features_file, metadata_file, local_csv = result

                # Launch background upload (parallel NPY + metadata)
                # chunk_id = local file ID for naming; _redis_cid tracked in pending_upload
                err_ref = [None]
                ul_thread = threading.Thread(
                    target=_do_background_upload,
                    args=(err_ref, r2, chunk_id, features_file, metadata_file,
                          local_csv, work_dir),
                    daemon=True,
                )
                ul_thread.start()
                pending_upload = (ul_thread, _redis_cid, err_ref)
                try:
                    tq.report_status(REGION, INSTANCE_ID, "UPLOADING",
                                     chunk_id=_redis_cid, chunks_done=chunks_done)
                except Exception:
                    pass
                # Loop immediately → start next chunk while upload runs

        except Exception as e:
            chunks_failed += 1
            error_msg = f"{type(e).__name__}: {str(e)[:200]}"
            print(f"[ERROR] Chunk {chunk_id} failed: {error_msg}")
            import traceback
            traceback.print_exc()
            try:
                tq.fail_task(REGION, _redis_cid, INSTANCE_ID, error_msg)
            except Exception as re:
                print(f"[WARN] Failed to return chunk to queue: {re}")
            _cleanup_chunk_files(work_dir, chunk_id)

            # If CSV download failed (404), skip all future chunks from this city
            if "Failed to download" in str(e) and CSV_BUCKET_PREFIX:
                skip_city_prefixes.add(CSV_BUCKET_PREFIX)
                print(f"[SKIP-CITY] Blacklisting city prefix '{CSV_BUCKET_PREFIX}' "
                      f"— all future chunks from {CITY_NAME} will be skipped")

            # Still collect prefetch result
            pf_thread.join()
            prefetched = pf_result[0]

        # Force GC between chunks
        gc.collect()
        torch.cuda.empty_cache()

    # ── Wait for final background upload before exit ──
    if pending_upload is not None:
        print("[INFO] Waiting for final background upload to complete...")
        ul_thread, ul_chunk_id, ul_error = pending_upload
        ul_thread.join()
        if ul_error[0] is not None:
            chunks_failed += 1
            print(f"[ERROR] Final upload failed for {ul_chunk_id}: {ul_error[0]}")
            try:
                tq.fail_task(REGION, ul_chunk_id, INSTANCE_ID,
                             str(ul_error[0])[:200])
            except Exception:
                pass
        else:
            if _redis_retry(tq.complete_task, REGION, ul_chunk_id, INSTANCE_ID,
                            label=f"complete_task({ul_chunk_id})"):
                chunks_done += 1
                print(f"[INFO] Final chunk {ul_chunk_id} completed ({chunks_done} total)")
            else:
                print(f"[WARN] Could not mark final {ul_chunk_id} done in Redis "
                      "(data safe on R2, will reconcile)")
                chunks_done += 1

    # ── Done — self-destruct ──
    try:
        final_progress = tq.get_progress(REGION)
        perm_failed = final_progress.get('failed', 0)
    except Exception:
        perm_failed = 0
    print(f"\n[INFO] Final stats: {chunks_done} chunks completed, {chunks_failed} failed this session, {perm_failed} permanently failed")
    try:
        tq.report_status(REGION, INSTANCE_ID, "DONE", chunks_done=chunks_done)
    except Exception:
        pass
    upload_logs_to_r2()
    self_destruct()


if __name__ == '__main__':
    _log_fh = _start_log_capture()
    try:
        main()
    except Exception as e:
        error_type = type(e).__name__
        print(f"[CRITICAL] {error_type}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        upload_logs_to_r2()
        sys.exit(1)
