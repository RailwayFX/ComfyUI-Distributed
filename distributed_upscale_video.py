# Import statements (original + new)
import torch
from PIL import Image, ImageFilter, ImageDraw
import json
import asyncio, sys

if sys.platform.startswith("win"):
    try:
        # If a loop is already running, policy change may not take effect.
        try:
            asyncio.get_running_loop()
            print("[USDU] Warning: an event loop is already running; "
                  "policy change may not fully apply.")
        except RuntimeError:
            pass  # No running loop yet; safe to switch policy.

        current = asyncio.get_event_loop_policy()
        # Switch only if not already Selector
        if not isinstance(current, asyncio.WindowsSelectorEventLoopPolicy):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            print("[USDU] Applied WindowsSelectorEventLoopPolicy")
    except Exception as e:
        print(f"[USDU] Failed to set WindowsSelectorEventLoopPolicy: {e}")
# ---- Quiet Windows 10054 noise in asyncio ----
import logging

def _is_win_conn_reset(exc: BaseException) -> bool:
    # True for Windows-specific connection reset noise
    if not sys.platform.startswith("win"):
        return False
    if isinstance(exc, ConnectionResetError):
        return True
    if isinstance(exc, OSError) and getattr(exc, "winerror", None) == 10054:
        return True
    # Fallback: textual match (defensive)
    return "10054" in str(exc)

def _install_win_connreset_filter():
    # Install a loop exception handler that ignores benign 10054 during shutdown
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        return  # no loop yet; safe to skip

    default_handler = loop.get_exception_handler()

    def _handler(loop, context):
        exc = context.get("exception")
        if exc and _is_win_conn_reset(exc):
            # Swallow ConnectionResetError(10054) noise
            return
        # Delegate to default handler
        if default_handler is not None:
            default_handler(loop, context)
        else:
            loop.default_exception_handler(context)

    try:
        loop.set_exception_handler(_handler)
        # Optional: keep asyncio logger quiet in INFO for less noise
        logging.getLogger("asyncio").setLevel(logging.WARNING)
    except Exception:
        pass

_install_win_connreset_filter()
# ---- End of quieting block ----

from contextlib import contextmanager
import aiohttp
import io
import base64
import math
import time
import torch.nn as nn
from typing import List, Tuple, Dict
from functools import wraps

import os
from concurrent.futures import ThreadPoolExecutor
import torch.nn.functional as F
import numpy as np
from typing import Optional
from comfy.utils import ProgressBar  # ComfyUI progress bar
from comfy import model_management
import gc
from tqdm.auto import tqdm

# Import ComfyUI modules
import comfy.samplers
import comfy.model_management

# Import shared utilities
from .utils.logging import debug_log, log
from .utils.image import tensor_to_pil, pil_to_tensor
from .utils.network import get_client_session
from .utils.async_helpers import run_async_in_server_loop
from .utils.config import get_worker_timeout_seconds
from .utils.constants import (
    TILE_COLLECTION_TIMEOUT, TILE_WAIT_TIMEOUT,
    TILE_SEND_TIMEOUT, MAX_BATCH
)

# Import for controller support
from .utils.usdu_utils import (
    crop_cond,
    get_crop_region,
    expand_crop,
)
from .utils.usdu_managment import (
    clone_conditioning, ensure_tile_jobs_initialized,
    # Job management functions
    _drain_results_queue,
    _check_and_requeue_timed_out_workers, _get_completed_count, _mark_task_completed,
    _send_heartbeat_to_master, _cleanup_job,
    # Constants
    JOB_COMPLETED_TASKS, JOB_WORKER_STATUS, JOB_PENDING_TASKS,
    MAX_PAYLOAD_SIZE,
    register_final_frames
)
from .utils.usdu_managment import init_dynamic_job, init_static_job_batched

# -----------------------------------------------------------------------------
# JPEG encoding/decoding multiprocess/thread utilities
#
# The following configuration and helper functions implement multi-threaded
# JPEG encoding and decoding. A global thread pool is initialized using a
# fraction of the available CPU cores, controlled by `JPEG_THREAD_FRACTION`.
# The helper functions `_encode_tile_data`, `_encode_np_to_jpeg` and
# `_decode_base64_to_np` are used to perform JPEG operations in parallel
# without blocking the main thread. These functions should be used whenever
# large numbers of tiles or frames need to be encoded or decoded. Because
# PIL's JPEG implementation releases the GIL during compression, using a
# ThreadPoolExecutor can take advantage of multiple cores effectively.

# Fraction of CPU cores to dedicate to JPEG tasks. Can be overridden via
# the environment variable JPEG_THREAD_FRACTION (e.g., "0.75" for 75%).
JPEG_THREAD_FRACTION = float(os.environ.get("JPEG_THREAD_FRACTION", "0.5"))

# Internal ThreadPoolExecutor instance for JPEG encoding/decoding. It is lazily
# instantiated on first use to avoid unnecessary thread creation when JPEG
# operations are not performed. See `get_jpeg_executor()` below.
_JPEG_EXECUTOR: Optional[ThreadPoolExecutor] = None

def get_jpeg_executor() -> ThreadPoolExecutor:
    """
    Return a shared ThreadPoolExecutor configured for JPEG operations.

    The number of worker threads is determined by the available CPU cores
    multiplied by JPEG_THREAD_FRACTION. At least one thread is always
    allocated. This executor is reused across calls to avoid repeatedly
    creating and destroying threads.
    """
    global _JPEG_EXECUTOR
    if _JPEG_EXECUTOR is None:
        # Determine available CPU cores; fallback to 1 if unknown
        try:
            cpu_count = os.cpu_count() or 1
        except Exception:
            cpu_count = 1
        # Compute number of threads based on the configured fraction
        num_workers = max(1, int(cpu_count * JPEG_THREAD_FRACTION))
        _JPEG_EXECUTOR = ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix="jpeg")
    return _JPEG_EXECUTOR

def _encode_tile_data(tile_data: dict) -> dict:
    """
    Encode a single tile into JPEG bytes with associated metadata.

    The input `tile_data` dictionary must contain the keys 'tile_idx', 'x',
    'y', 'extracted_width', 'extracted_height' and optionally 'batch_idx' and
    'global_idx'. It may also contain a pre-rendered PIL image under the key
    'image'. If no 'image' is present, the function falls back to converting
    the provided torch tensor under 'tile' into a PIL image via `tensor_to_pil`.
    The output is a dictionary with keys 'bytes' (JPEG-compressed data) and
    'meta' (original metadata for reconstruction on the master).
    """
    # Prefer the prepared PIL image when available to avoid redundant
    # conversions. If not present, use tensor_to_pil to convert the tensor.
    img = tile_data.get('image')
    if img is None:
        img = tensor_to_pil(tile_data['tile'], 0)

    # Perform JPEG encoding into a byte buffer. High quality is retained to
    # minimize compression artefacts; subsampling and optimize flags mirror the
    # original behaviour in the codebase.
    bio = io.BytesIO()
    img.save(bio, format='JPEG', quality=100, subsampling=0, optimize=False)
    raw = bio.getvalue()

    # Collect metadata for the tile for later reassembly
    meta = {
        'tile_idx': tile_data['tile_idx'],
        'x': tile_data['x'],
        'y': tile_data['y'],
        'extracted_width': tile_data['extracted_width'],
        'extracted_height': tile_data['extracted_height']
    }
    if 'batch_idx' in tile_data:
        meta['batch_idx'] = tile_data['batch_idx']
    if 'global_idx' in tile_data:
        meta['global_idx'] = tile_data['global_idx']
    return {'bytes': raw, 'meta': meta}

def _encode_np_to_jpeg(arr: np.ndarray) -> bytes:
    """
    Encode a numpy array (H×W×3, uint8 [0,255]) into JPEG-compressed bytes.
    """
    bio = io.BytesIO()
    Image.fromarray(arr).save(bio, format='JPEG', quality=100, subsampling=0, optimize=False)
    return bio.getvalue()


def _decode_base64_to_np(fb64: str) -> np.ndarray:
    """
    Decode a base64-encoded JPEG image into a numpy array of shape H×W×3
    with dtype=uint8 in [0,255]. No scaling is performed here to avoid
    forcing float32; scaling happens later only if the target dtype is float.
    """
    pil_img = Image.open(io.BytesIO(base64.b64decode(fb64))).convert('RGB')
    # Keep uint8 to avoid unnecessary FP32 memory usage
    return np.array(pil_img, dtype=np.uint8)



# Note: MAX_BATCH and HEARTBEAT_TIMEOUT are imported from utils.constants
# They can be overridden via environment variables:
# - COMFYUI_MAX_BATCH (default: 20)
# - COMFYUI_HEARTBEAT_TIMEOUT (default: 90)

# Sync wrapper decorator for async methods
def sync_wrapper(async_func):
    """Decorator to wrap async methods for synchronous execution."""
    @wraps(async_func)
    def sync_func(self, *args, **kwargs):
        # Use run_async_in_server_loop for ComfyUI compatibility
        return run_async_in_server_loop(
            async_func(self, *args, **kwargs),
            timeout=600.0  # 10 minute timeout for long operations
        )
    return sync_func

@torch.no_grad()
def upscale_with_model_batched(
    upscale_model,
    images,                                   # <- accepts torch.Tensor OR PIL.Image OR list[PIL.Image]
    per_batch: int = 16,
    clear_cuda_each_batch: bool = False,
    show_tqdm: bool = True,
    tqdm_desc: str = "[UpscalingWithModel]",
    verbose: bool = False,
    log_every_batches: int = 1,
    time_unit: str = "s",                 # "s" or "ms"
    prefer_tiled: bool = False,           # start in tiled mode
    sticky_tiled_after_oom: bool = True,  # once OOM happens -> always tiled
    auto_tiled_vram_threshold: float = 0.90,  # if util>=threshold -> tiled
    direct_clear_cuda_each_batch: Optional[bool] = None,  # default False
    init_tile: int = 512,
    min_tile: int = 128,
    overlap: int = 32,
    remember_tile_across_batches: bool = True,  # remember tile size
    return_pil: bool = True,             # default True since downstream uses PIL
    pil_autoswitch_bytes: int = 1_000_000_000,  # auto-switch to PIL if est. output > ~1GB
):
    """
    Batched upscaling with sticky tiled mode and tile-size memory.

    Input:
      - torch.Tensor [B,H,W,C] in 0..1
      - PIL.Image or list[PIL.Image] (RGB)

    If return_pil=True:
      - Do NOT pre-allocate a giant CPU tensor.
      - Convert each processed batch directly to uint8 PIL frames and append to a list.

    If return_pil=False:
      - Collect CPU BHWC chunks in a list and concatenate at the end.
    """
    # -------- helpers --------
    def _fmt_b(x: int) -> str:
        for u in ["B","KB","MB","GB","TB"]:
            if x < 1024 or u == "TB": return f"{x:.1f}{u}"
            x /= 1024.0

    def _cuda_stats(dev: torch.device):
        try:
            if dev.type == "cuda":
                idx = dev.index or 0
                free_b, total_b = torch.cuda.mem_get_info(idx)
                alloc = torch.cuda.memory_allocated(idx)
                reserv = torch.cuda.memory_reserved(idx)
                used = total_b - free_b
                util = used / float(total_b) if total_b else 0.0
                return {"alloc":alloc,"reserv":reserv,"free":free_b,"total":total_b,"used":used,"util":util}
        except Exception:
            pass
        return None

    def _stats_str(s) -> str:
        if not s: return "n/a"
        return (f"alloc={_fmt_b(s['alloc'])}, reserv={_fmt_b(s['reserv'])}, "
                f"free={_fmt_b(s['free'])}, total={_fmt_b(s['total'])}, util={s['util']*100:.1f}%")

    def _now(): return time.perf_counter()
    def _dt(dt): return dt * (1000.0 if time_unit=="ms" else 1.0)
    def _log(msg):
        if verbose: print(msg, flush=True)

    # -------- normalize/ingest input to BHWC tensor (no forced float32) --------
    model_dtype = None
    try:
        # Prefer the model's parameter dtype to respect current ComfyUI precision (fp16/bf16/etc.)
        p = next(upscale_model.parameters()) if hasattr(upscale_model, "parameters") else None
        if p is not None:
            model_dtype = p.dtype
    except Exception:
        pass
    if model_dtype is None:
        # Fallback to torch.get_default_dtype() if parameters() not available
        model_dtype = torch.get_default_dtype()

    # Convert PIL inputs to a BHWC tensor with model_dtype in 0..1
    pil_in = False
    if isinstance(images, Image.Image):
        pil_in = True
        images = [images]
    if isinstance(images, list) and len(images) > 0 and isinstance(images[0], Image.Image):
        pil_in = True
        # Build a minimal, contiguous BHWC tensor in the model's dtype
        np_list = []
        for im in images:
            if im.mode != "RGB":
                im = im.convert("RGB")
            arr = np.asarray(im, dtype=np.uint8)  # HWC uint8
            np_list.append(arr)
        # Stack as BHWC uint8 then scale to 0..1 in model_dtype
        arr_bhwc = np.stack(np_list, axis=0)  # [B,H,W,C], uint8
        t = torch.from_numpy(arr_bhwc)  # uint8 CPU
        images = t.to(dtype=torch.float32).mul_(1.0/255.0)  # temporary float32 for safe scaling math
        # Cast to model dtype only once, to avoid keeping two copies
        images = images.to(dtype=model_dtype, copy=False)
        del np_list, arr_bhwc, t

    # -------- validate --------
    if not (isinstance(images, torch.Tensor) and images.ndim == 4 and images.shape[-1] == 3):
        raise ValueError("Expected BHWC tensor or PIL.Image/list[PIL.Image] (RGB).")

    if per_batch < 1:
        raise ValueError("per_batch must be >= 1.")

    # -------- setup --------
    dev = model_management.get_torch_device()
    is_cuda = (dev.type == "cuda")
    in_dtype = images.dtype  # keep the current dtype; do not force float32

    x = images.movedim(-1, -3).contiguous()  # BHWC -> BCHW
    B, C, H, W = x.shape
    scale = float(getattr(upscale_model, "scale", 1.0))
    outW, outH = int(W * scale), int(H * scale)

    # Auto-switch to PIL path if the final dense tensor would be huge
    try:
        est_bytes = int(B) * int(outH) * int(outW) * int(C) * int(torch.tensor([], dtype=in_dtype).element_size())
        if est_bytes >= pil_autoswitch_bytes:
            return_pil = True
    except Exception:
        pass

    _log("="*70)
    _log(f"{tqdm_desc} starting")
    _log(f"device={dev}, dtype={x.dtype}, cuda={is_cuda}")
    _log(f"B={B}, per_batch={per_batch}, in={W}x{H}, scale={scale} -> out≈{outW}x{outH}")
    if is_cuda: _log(f"VRAM before: {_stats_str(_cuda_stats(dev))}")
    if clear_cuda_each_batch: _log("clear_cuda_each_batch=True (tiled cleanup).")
    _log(f"direct_clear_cuda_each_batch={bool(direct_clear_cuda_each_batch or False)}")

    pbar = ProgressBar(B)
    total_batches = max(1, math.ceil(B / per_batch))
    tbar = tqdm(total=total_batches, desc=tqdm_desc, unit="it", dynamic_ncols=True, leave=True) if show_tqdm else None

    # Try to free some VRAM before moving model
    t0 = _now()
    try:
        elem_size = images.element_size()
        mem_required = model_management.module_size(getattr(upscale_model, "model", upscale_model))
        mem_required += (512 * 512 * 3) * elem_size * max(scale, 1.0) * 384.0
        mem_required += images.nelement() * elem_size
        model_management.free_memory(mem_required, dev)
        _log(f"reserved/free_memory heuristic={_fmt_b(int(mem_required))} on {dev} (t={_dt(_now()-t0):.1f}{time_unit})")
        if is_cuda: _log(f"VRAM after reserve: {_stats_str(_cuda_stats(dev))}")
    except Exception as e:
        _log(f"reserve/free_memory skipped: {e!r}")

    # Move model
    t1 = _now()
    upscale_model.to(dev)
    if hasattr(upscale_model, "eval"): upscale_model.eval()
    _log(f"model.to({dev}) + eval (t={_dt(_now()-t1):.1f}{time_unit})")

    # -------- mode state --------
    use_tiled = bool(prefer_tiled)
    sticky_tripped = False

    tile_state = int(init_tile)   # last successful tile size
    have_tile_state = False       # becomes True after first successful tiled pass

    batch_times = []
    start_t, done_frames = _now(), 0

    # Output accumulators (no giant pre-allocation)
    out_cpu_chunks: list[torch.Tensor] = []  # used when return_pil=False
    out_pil_frames: list = []                # used when return_pil=True

    try:
        for batch_idx, s in enumerate(range(0, B, per_batch), start=1):
            b_t0 = _now()
            chunk = x[s:s + per_batch]
            n = int(chunk.shape[0])
            if n == 0:
                if tbar: tbar.update(1)
                continue

            _log("-"*70)
            _log(f"Batch {batch_idx}/{total_batches}  frames={s}..{s+n-1}  shape={list(chunk.shape)}")

            # Auto-tiling by VRAM
            if is_cuda and auto_tiled_vram_threshold is not None:
                st = _cuda_stats(dev)
                if st and st["util"] >= auto_tiled_vram_threshold and not use_tiled:
                    use_tiled = True
                    _log(f"auto-tiling engaged (VRAM util {st['util']*100:.1f}% >= {auto_tiled_vram_threshold*100:.0f}%)")

            if is_cuda and verbose:
                torch.cuda.synchronize()
                _log(f"VRAM pre-pass: {_stats_str(_cuda_stats(dev))}")

            # Cleanup policy for this batch
            do_cleanup_after = clear_cuda_each_batch if use_tiled else bool(direct_clear_cuda_each_batch or False)

            # Direct path unless already in tiled mode
            if not use_tiled:
                try:
                    d_t0 = _now()
                    y = upscale_model(chunk.to(dev, non_blocking=True))
                    if is_cuda and verbose: torch.cuda.synchronize()
                    _log(f"direct pass ok (t={_dt(_now()-d_t0):.1f}{time_unit})")
                except model_management.OOM_EXCEPTION:
                    _log("direct pass OOM -> switch to tiled (sticky)")
                    use_tiled = True
                    sticky_tripped = True

            if use_tiled:
                tile = tile_state if (remember_tile_across_batches and have_tile_state) else int(init_tile)
                in_img = chunk.to(dev, non_blocking=True)
                while True:
                    try:
                        if is_cuda and verbose: torch.cuda.synchronize()
                        t_t0 = _now()
                        y = comfy.utils.tiled_scale(
                            in_img,
                            lambda a: upscale_model(a),
                            tile_x=tile, tile_y=tile,
                            overlap=overlap,
                            upscale_amount=scale,
                            pbar=None
                        )
                        if is_cuda and verbose: torch.cuda.synchronize()
                        _log(f"tiled pass ok  tile={tile} overlap={overlap} (t={_dt(_now()-t_t0):.1f}{time_unit})")
                        tile_state = tile
                        have_tile_state = True
                        break
                    except model_management.OOM_EXCEPTION as e:
                        tile //= 2
                        _log(f"tiled pass OOM -> reduce tile to {tile}")
                        if tile < int(min_tile):
                            _log("tile < min_tile -> re-raise OOM")
                            raise e
                        if is_cuda:
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                            torch.cuda.ipc_collect()
                            gc.collect()
                del in_img

            # ---- collect results without giant allocation ----
            y_bhwc = y.permute(0, 2, 3, 1)

            if return_pil:
                y_u8 = (y_bhwc.clamp_(0.0, 1.0) * 255.0 + 0.5).to(torch.uint8).to("cpu", non_blocking=False)
                for i in range(y_u8.shape[0]):
                    arr = y_u8[i].numpy().copy()  # HWC uint8
                    out_pil_frames.append(Image.fromarray(arr, mode="RGB"))
                del y_u8
            else:
                y_cpu = y_bhwc.to("cpu", dtype=in_dtype, non_blocking=False).contiguous()
                y_cpu.clamp_(0.0, 1.0)
                out_cpu_chunks.append(y_cpu)

            pbar.update(n)
            if tbar: tbar.update(1)

            del y, y_bhwc, chunk

            if is_cuda and do_cleanup_after:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                gc.collect()

            b_dt = _now() - b_t0
            batch_times.append(b_dt)
            done_frames += n
            elapsed = _now() - start_t
            fps = done_frames / elapsed if elapsed > 0 else float("inf")
            mpix_done = (W * H * done_frames) / 1e6
            mpix_s = mpix_done / elapsed if elapsed > 0 else float("inf")
            avg_bt = sum(batch_times) / len(batch_times)
            eta = (total_batches - batch_idx) * avg_bt

            _log(f"batch time={_dt(b_dt):.1f}{time_unit}  avg/batch={_dt(avg_bt):.1f}{time_unit}  ETA≈{_dt(eta):.0f}{time_unit}")
            _log(f"throughput: {fps:.2f} fps, {mpix_s:.2f} MPix/s (done {done_frames}/{B})")
            if is_cuda and verbose and (batch_idx % max(1, log_every_batches) == 0):
                _log(f"VRAM post-batch: {_stats_str(_cuda_stats(dev))}")

            if sticky_tripped and sticky_tiled_after_oom:
                use_tiled = True  # keep tiled for all subsequent batches

        total_t = _now() - start_t
        _log("="*70)
        _log(f"{tqdm_desc} done  total={_dt(total_t):.1f}{time_unit}  avg/batch={_dt(sum(batch_times)/max(1,len(batch_times))):.1f}{time_unit}")
        if is_cuda: _log(f"VRAM final: {_stats_str(_cuda_stats(dev))}")

        if return_pil:
            return out_pil_frames  # list[PIL.Image]
        else:
            if len(out_cpu_chunks) == 1:
                return out_cpu_chunks[0]
            return torch.cat(out_cpu_chunks, dim=0)

    finally:
        if tbar:
            try: tbar.close()
            except Exception: pass
        try: upscale_model.cpu()
        except Exception: pass
        del x
        gc.collect()
        if is_cuda:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


@torch.no_grad()
def auto_morph_align_batch_bhwc(
    clean: torch.Tensor,
    image: torch.Tensor,
    blur: int = 0,
    max_disp_px: float = 160.0,      # max displacement clamp at full-res
    lk_win_rad: int = 15,            # LK box window radius
    pyramid_levels: int = 4,         # kept for API compatibility; ignored
    iters_per_level: int = 6,        # iterations at half-res
    stop_at_fine_levels: int = 0,    # kept for API compatibility; ignored
    pre_smooth_rad: int = 0,         # low-pass before gradients
    min_eig_rel: float = 0.001,      # confidence gate on λ_min
    inc_smooth_rad: int = 0          # optional blur of incremental flow
) -> torch.Tensor:
    """
    Dense optical-flow alignment (GPU-only), computed at half resolution:
      1) Downscale (area) to 1/2.
      2) Iteratively estimate flow at half-res.
      3) Upscale flow to full-res (proper vector scaling).
      4) Single inverse warp at full-res.
    Inputs/outputs: BHWC, floating point in [0,1] with the SAME dtype as input.
    """
    # ---- sanity checks ----
    if clean.ndim != 4 or image.ndim != 4:
        raise ValueError("Expected BHWC tensors.")
    if clean.shape != image.shape:
        raise ValueError(f"Shape mismatch: clean {clean.shape} vs image {image.shape}.")
    if not (torch.is_floating_point(clean) and torch.is_floating_point(image)):
        raise TypeError("Expected floating point tensors in [0,1].")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    # Preserve original device and dtype
    orig_device, in_dtype = image.device, image.dtype

    # Move to CUDA without changing dtype
    clean = clean.to("cuda", non_blocking=True)
    image = image.to("cuda", non_blocking=True)

    B, H, W, C = image.shape
    blur = int(max(0, int(blur)))

    # Use machine epsilon scaled for numerical safety in the current dtype
    EPS = torch.finfo(in_dtype).eps * 16.0

    # ----------------- small helpers -----------------
    def _to_gray_bhwc(x: torch.Tensor) -> torch.Tensor:
        # simple luma (keeps dtype)
        r, g, b = x[..., 0:1], x[..., 1:2], x[..., 2:3]
        return (0.2989 * r + 0.5870 * g + 0.1140 * b).clamp(0.0, 1.0)

    def _bhwc_to_nchw(x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 3, 1, 2).contiguous()

    def _nchw_to_bhwc(x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 3, 1).contiguous()

    def _resize_bhwc(x: torch.Tensor, oh: int, ow: int, mode: str = "area") -> torch.Tensor:
        # Use area for downscale; bilinear for upscale
        nchw = _bhwc_to_nchw(x)
        out = F.interpolate(nchw, size=(oh, ow), mode=("area" if mode == "area" else "bilinear"),
                            align_corners=False if mode != "area" else None)
        return _nchw_to_bhwc(out)

    def _avg_blur_1ch_nchw(x: torch.Tensor, rad: int) -> torch.Tensor:
        if rad <= 0:
            return x
        k = 2 * rad + 1
        x = F.pad(x, (rad, rad, rad, rad), mode="replicate")
        return F.avg_pool2d(x, kernel_size=k, stride=1)

    def _sobel_conv_1ch_nchw(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gx = torch.tensor([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3) / 8.0
        gy = torch.tensor([[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3) / 8.0
        xp = F.pad(x, (1, 1, 1, 1), mode="replicate")
        return F.conv2d(xp, gx), F.conv2d(xp, gy)

    def _box_sum_1ch_nchw(x: torch.Tensor, rad: int) -> torch.Tensor:
        if rad <= 0:
            return x
        k = 2 * rad + 1
        w = torch.ones((1, 1, k, k), dtype=x.dtype, device=x.device)
        xp = F.pad(x, (rad, rad, rad, rad), mode="replicate")
        return F.conv2d(xp, w)

    def _box_blur_flow_gpu(flow_nchw: torch.Tensor, rad: int) -> torch.Tensor:
        if rad <= 0:
            return flow_nchw
        k = 2 * rad + 1
        c = flow_nchw.shape[1]
        w = torch.ones((c, 1, k, k), dtype=flow_nchw.dtype, device=flow_nchw.device) / float(k * k)
        x = F.pad(flow_nchw, (rad, rad, rad, rad), mode="replicate")
        return F.conv2d(x, w, groups=c)

    def _warp_by_flow_bhwc(img_bhwc: torch.Tensor, flow_bhwc: torch.Tensor) -> torch.Tensor:
        # Inverse warp: output(x) = img(x - flow)
        N, Hh, Ww, _ = img_bhwc.shape
        img_nchw = _bhwc_to_nchw(img_bhwc)
        ys = torch.linspace(-1.0, 1.0, Hh, device=img_bhwc.device, dtype=img_bhwc.dtype)
        xs = torch.linspace(-1.0, 1.0, Ww, device=img_bhwc.device, dtype=img_bhwc.dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        base = torch.stack((xx, yy), dim=-1).unsqueeze(0)
        fx = flow_bhwc[..., 0] * (2.0 / max(1, (Ww - 1)))
        fy = flow_bhwc[..., 1] * (2.0 / max(1, (Hh - 1)))
        grid = base + torch.stack((-fx, -fy), dim=-1)
        out = F.grid_sample(img_nchw, grid, mode="bilinear", padding_mode="border", align_corners=False)
        return _nchw_to_bhwc(out)

    def _lk_dense_increment(
        img_bhwc: torch.Tensor,
        ref_bhwc: torch.Tensor,
        win_rad: int,
        eps: float,
        clamp_px: float,
        pre_blur_rad: int,
        min_eig_rel_thr: float,
        inc_blur_rad: int
    ) -> torch.Tensor:
        # One-step Lucas–Kanade increment at current scale
        img_g = _to_gray_bhwc(img_bhwc)
        ref_g = _to_gray_bhwc(ref_bhwc)

        img_n = _bhwc_to_nchw(img_g)
        ref_n = _bhwc_to_nchw(ref_g)

        if pre_blur_rad > 0:
            img_n = _avg_blur_1ch_nchw(img_n, pre_blur_rad)
            ref_n = _avg_blur_1ch_nchw(ref_n, pre_blur_rad)

        Ix, Iy = _sobel_conv_1ch_nchw(img_n)
        It = ref_n - img_n

        Sxx = _box_sum_1ch_nchw(Ix * Ix, win_rad)
        Syy = _box_sum_1ch_nchw(Iy * Iy, win_rad)
        Sxy = _box_sum_1ch_nchw(Ix * Iy, win_rad)
        Sxt = _box_sum_1ch_nchw(Ix * It, win_rad)
        Syt = _box_sum_1ch_nchw(Iy * It, win_rad)

        det = (Sxx * Syy - Sxy * Sxy)
        u = (-(Syy * Sxt) + (Sxy * Syt)) / (det + eps)
        v = ((Sxy * Sxt) - (Sxx * Syt)) / (det + eps)

        trace = Sxx + Syy
        sqrt_term = torch.sqrt((Sxx - Syy) * (Sxx - Syy) + 4.0 * Sxy * Sxy + eps)
        lam_min = 0.5 * (trace - sqrt_term)
        k = (2 * win_rad + 1) ** 2
        lam_min_rel = lam_min / max(1.0, float(k))
        mask = (lam_min_rel >= float(min_eig_rel_thr)).to(u.dtype)

        u = (u * mask).clamp(-clamp_px, clamp_px)
        v = (v * mask).clamp(-clamp_px, clamp_px)

        inc = _nchw_to_bhwc(torch.cat([u, v], dim=1))  # [B,H,W,2]

        if inc_blur_rad > 0:
            inc = _nchw_to_bhwc(_box_blur_flow_gpu(_bhwc_to_nchw(inc), inc_blur_rad))

        return inc

    # ----------------- half-res flow, then upscale -----------------
    t0 = time.perf_counter()
    #pbar = ProgressBar(B)
    out_frames = []

    # per-level clamp at half-res so that full-res magnitude <= max_disp_px
    clamp_half = float(max_disp_px) / 2.0

    for i in range(B):
        src_full = image[i:i+1]
        ref_full = clean[i:i+1]

        # build half-res pair (area downscale)
        Hh = max(1, H // 2)
        Wh = max(1, W // 2)
        src_half = _resize_bhwc(src_full, Hh, Wh, mode="area")
        ref_half = _resize_bhwc(ref_full, Hh, Wh, mode="area")

        # initialize zero flow at half-res with input dtype
        flow_h = torch.zeros((1, Hh, Wh, 2), dtype=in_dtype, device=src_half.device)

        # iterative LK at half-res (warp -> increment -> accumulate)
        for _ in range(max(1, int(iters_per_level))):
            warped_h = _warp_by_flow_bhwc(src_half, flow_h)
            inc_h = _lk_dense_increment(
                warped_h, ref_half,
                win_rad=lk_win_rad,
                eps=EPS,
                clamp_px=clamp_half,
                pre_blur_rad=pre_smooth_rad,
                min_eig_rel_thr=min_eig_rel,
                inc_blur_rad=inc_smooth_rad
            )
            flow_h = (flow_h + inc_h).clamp(min=-clamp_half, max=clamp_half)

        # optional smoothing of final half-res flow (legacy 'blur' knob)
        with torch.cuda.amp.autocast(enabled=True):
            flow_h = _nchw_to_bhwc(_box_blur_flow_gpu(_bhwc_to_nchw(flow_h), blur))

        # upscale flow to full-res and scale vector magnitudes
        flow_full = _nchw_to_bhwc(
            F.interpolate(_bhwc_to_nchw(flow_h), size=(H, W), mode="bilinear", align_corners=False)
        )
        flow_full[..., 0] *= (W / max(1.0, float(Wh)))
        flow_full[..., 1] *= (H / max(1.0, float(Hh)))
        flow_full = flow_full.clamp(min=-max_disp_px, max=max_disp_px)

        # single inverse warp at full-res
        aligned = _warp_by_flow_bhwc(src_full, flow_full).clamp(0.0, 1.0)
        out_frames.append(aligned)
        #pbar.update(i + 1)

        # cleanup
        del src_full, ref_full, src_half, ref_half, flow_h, flow_full, aligned

    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"[AutoMorph] {B} frame(s), {dt:.3f} s, {dt/max(1,B):.4f} s/frame", flush=True)

    # Preserve dtype and move back to original device
    result_cuda = torch.cat(out_frames, dim=0).contiguous()
    return result_cuda.to(orig_device, dtype=in_dtype, non_blocking=True)


def sharpen_tiny(
    image: torch.Tensor,
    sharpen_radius: int = 1,     # will be 1
    sigma_x: float = 0.5,
    sigma_y: float = 0.5,
    alpha: float = 1.0,
) -> torch.Tensor:
    """
    Ultra-fast unsharp mask for BHWC float tensors in [0,1] with OOM safety.

    - Uses a single depthwise 3x3 Gaussian conv (separable kernel fused via outer product).
    - Vectorized across batch; dynamically downsizes chunk if CUDA OOM is encountered.
    - Preserves input dtype/device and returns BHWC.
    """

    # Fast exits
    if sharpen_radius <= 0 or alpha == 0.0:
        return image
    if image.ndim != 4 or image.shape[-1] not in (1, 2, 3, 4):
        raise ValueError("Expected BHWC image with 1..4 channels.")

    t0 = time.perf_counter()

    B, H, W, C = image.shape
    dev = image.device
    dt = image.dtype
    is_cuda = dev.type == "cuda"

    # Enable fast CuDNN kernels where applicable
    if is_cuda:
        torch.backends.cudnn.benchmark = True  # safe for inference with varying sizes

    # --- Build 3x3 depthwise Gaussian kernel on the same device/dtype ---
    # For radius=1 we need K=3; keep sigma_x/sigma_y for anisotropic cases.
    k = 2 * sharpen_radius + 1  # -> 3
    # 1D grids: [-1, 0, 1]
    xs = torch.linspace(-sharpen_radius, sharpen_radius, k, device=dev, dtype=dt)
    gx = torch.exp(-(xs**2) / (2.0 * (sigma_x * sigma_x) + 1e-12))
    gy = torch.exp(-(xs**2) / (2.0 * (sigma_y * sigma_y) + 1e-12))
    gx = gx / (gx.sum() + 1e-12)  # [3]
    gy = gy / (gy.sum() + 1e-12)  # [3]

    # Outer product -> 2D kernel [3,3]
    g2d = torch.outer(gy, gx)
    # Depthwise weights: [C, 1, 3, 3] (same kernel per channel)
    weight = g2d.view(1, 1, k, k).expand(C, 1, k, k).contiguous()

    # Pre-allocate output tensor
    out = torch.empty_like(image)

    # Heuristic to choose batch chunk size (in frames) to avoid OOM
    # We estimate activation footprint conservatively and leave a safety margin.
    def estimate_safe_chunk():
        # bytes per element for current dtype
        if dt == torch.float32:
            bpe = 4
        elif dt in (torch.float16, torch.bfloat16):
            bpe = 2
        else:
            # default conservative
            bpe = max(2, torch.finfo(dt).bits // 8)

        # Approx per-frame working set in bytes:
        # input + padded input (~+8%), output, and conv work buffers.
        # Use a conservative multiplier.
        per_frame_bytes = C * H * W * bpe
        conservative_factor = 6.0  # generous headroom for conv workspace
        frame_budget = per_frame_bytes * conservative_factor

        if is_cuda:
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(dev)
                # Keep only 60% of currently free memory for safety
                budget = max(128 * 1024 * 1024, int(free_bytes * 0.60))
                chunk = max(1, min(B, budget // max(1, int(frame_budget))))
                return chunk
            except Exception:
                pass

        # CPU or mem_get_info unavailable: fallback to megapixels heuristic
        # Aim ~8 MP per chunk (safe on most systems)
        mp_target = 8_000_000
        per_frame_mp = max(1, H * W)
        chunk = max(1, min(B, mp_target // per_frame_mp))
        return chunk

    alpha_t = torch.as_tensor(alpha, dtype=dt, device=dev)

    # Process in chunks with automatic backoff on OOM
    remaining = B
    start = 0
    chunk = estimate_safe_chunk()

    # Single-pass depthwise conv; reflect padding keeps edges clean.
    # Use channels_last on CUDA for better throughput.
    while remaining > 0:
        # Make sure chunk is at least 1
        chunk = max(1, min(chunk, remaining))
        try:
            x_bhwc = image[start:start + chunk]  # [N,H,W,C]
            x = x_bhwc.permute(0, 3, 1, 2)       # [N,C,H,W]
            if is_cuda:
                x = x.contiguous(memory_format=torch.channels_last)

            # Reflect pad by 1 pixel and run depthwise conv
            x_pad = F.pad(x, (1, 1, 1, 1), mode="reflect")
            if is_cuda:
                x_pad = x_pad.contiguous(memory_format=torch.channels_last)

            blur = F.conv2d(
                x_pad, weight, bias=None, stride=1, padding=0, groups=C
            )

            # Unsharp mask: sharp = (1+alpha)*x - alpha*blur
            sharp = torch.add(x, x, alpha=alpha_t)   # (1+alpha)*x
            sharp.add_(blur, alpha=-alpha_t)         # subtract alpha*blur
            sharp.clamp_(0.0, 1.0)

            # Write back as BHWC
            out[start:start + chunk].copy_(sharp.permute(0, 2, 3, 1))

            # Advance window
            start += chunk
            remaining -= chunk

            # Try to grow chunk opportunistically to reduce loop overhead
            if is_cuda and remaining > 0:
                chunk = min(B - start, chunk * 2)

            # Explicitly drop large temporaries
            del x_bhwc, x, x_pad, blur, sharp
        except RuntimeError as e:
            # Robust OOM handling: shrink chunk and retry
            if "out of memory" in str(e).lower() and is_cuda:
                torch.cuda.empty_cache()
                # Halve the chunk; if already at 1, re-raise
                if chunk > 1:
                    chunk = max(1, chunk // 2)
                    continue
            raise

    dt_s = time.perf_counter() - t0
    print(f"[SharpenTiny] {B} frame(s), {dt_s:.3f} s, {dt_s / max(1, B):.4f} s/frame", flush=True)
    return out


class UltimateSDUpscaleDistributed:
    """
    Distributed version of Ultimate SD Upscale (No Upscale).

    Now expects and returns: list of RGB PIL frames.

    Supports three processing modes:
    1. Single GPU: No workers available, process everything locally
    2. Static Mode: Small batches, distributes tiles across workers (flattened)
    3. Dynamic Mode: Large batches, assigns whole images to workers dynamically

    Features:
    - Multi-mode batch handling for efficient video/image upscaling
    - Tiled VAE support for memory efficiency
    - Dynamic load balancing for large batches
    - Backward compatible with single-image workflows

    Environment Variables:
    - COMFYUI_MAX_BATCH: Chunk size for tile sending (default 20)
    - COMFYUI_MAX_PAYLOAD_SIZE: Max API payload bytes (default 50MB)

    Threshold: dynamic_threshold input controls mode switch (default 8)
    """

    def __init__(self):
        """Initialize the node and ensure persistent storage exists."""
        ensure_tile_jobs_initialized()
        debug_log("UltimateSDUpscaleDistributed - Node initialized (PIL I/O)")
        self._progress = None
        self._progress_keepalive = False

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Force re-execution."""
        return float("nan")  # Always re-execute

    # ------------ small helpers (PIL-first) ------------
    def _infer_batch_and_size_from_pil(self, images_list):
        """Return (batch, H, W) from list of RGB PIL images."""
        if not isinstance(images_list, list) or len(images_list) == 0:
            raise ValueError("Expected a non-empty list of RGB PIL images")
        w, h = images_list[0].size
        return len(images_list), h, w

    def _ensure_rgb_copies(self, images_list):
        """Ensure every image is RGB and detached from any shared buffer."""
        out = []
        for img in images_list:
            if img.mode != "RGB":
                img = img.convert("RGB")
            out.append(img.copy())
        return out

    # ---------------------------------------------------

    def run(self, upscaled_image, model, positive, negative, vae, seed, steps, cfg,
            sampler_name, scheduler, denoise, tile_width, tile_height, padding,
            mask_blur, mask_expand, force_uniform_tiles, tiled_decode,
            multi_job_id="", is_worker=False, master_url="", enabled_worker_ids="[]",
            worker_id="", tile_indices="", dynamic_threshold=8, auto_morph_output=False, auto_morph_blur=8,
            # Optional high-pass and gradient-preserve blending parameters
            highpass_blend=False, hp_blur_size=100, hp_opacity=0.85, hp_device="auto", hp_store_result_on="cpu", hp_contrast=1.0,
            preserve_gradients_blend=False, pg_highpass_radius=15, pg_downscale_by=0.25, pg_strength=10.0,
            pg_expand=5, pg_mask_blur=25, pg_clamp_blacks=0.05, pg_clamp_whites=0.9,
            pg_temporal_deflicker=False, pg_device="auto", post_sharpen=0.0, pre_sharpen=0.0):
        """Entry point (synchronous). Now expects a list of RGB PIL frames and returns the same."""
        if not isinstance(upscaled_image, list):
            raise TypeError("UltimateSDUpscaleDistributed now expects a list of RGB PIL frames as input")

        batch_size, height, width = self._infer_batch_and_size_from_pil(upscaled_image)

        # Enforce 4n+1 batches globally when batch > 1 (master only)
        if not is_worker and batch_size != 1 and (batch_size % 4 != 1):
            raise ValueError(
                f"Batch size {batch_size} is not of the form 4n+1. "
                "This node requires batch sizes of 1 or 4n+1 (1, 5, 9, 13, ...). "
                "Please adjust the batch size."
            )

        if not multi_job_id:
            # No distributed processing, run single GPU version
            return self.process_single_gpu(upscaled_image, model, positive, negative, vae,
                                           seed, steps, cfg, sampler_name, scheduler, denoise,
                                           tile_width, tile_height, padding, mask_blur, mask_expand,
                                           force_uniform_tiles, tiled_decode, auto_morph_output, auto_morph_blur,
                                           highpass_blend, hp_blur_size, hp_opacity, hp_device, hp_store_result_on, hp_contrast,
                                           preserve_gradients_blend, pg_highpass_radius, pg_downscale_by, pg_strength, pg_expand, pg_mask_blur,
                                           pg_clamp_blacks, pg_clamp_whites, pg_temporal_deflicker, pg_device, post_sharpen, pre_sharpen)

        if is_worker:
            # Worker mode
            return self.process_worker(upscaled_image, model, positive, negative, vae,
                                       seed, steps, cfg, sampler_name, scheduler, denoise,
                                       tile_width, tile_height, padding, mask_blur, mask_expand,
                                       force_uniform_tiles, tiled_decode, multi_job_id, master_url,
                                       worker_id, enabled_worker_ids, dynamic_threshold, auto_morph_output, auto_morph_blur,
                                       highpass_blend, hp_blur_size, hp_opacity, hp_device, hp_store_result_on, hp_contrast,
                                       preserve_gradients_blend, pg_highpass_radius, pg_downscale_by, pg_strength, pg_expand, pg_mask_blur,
                                       pg_clamp_blacks, pg_clamp_whites, pg_temporal_deflicker, pg_device, post_sharpen, pre_sharpen)
        else:
            # Master mode
            return self.process_master(upscaled_image, model, positive, negative, vae,
                                       seed, steps, cfg, sampler_name, scheduler, denoise,
                                       tile_width, tile_height, padding, mask_blur, mask_expand,
                                       force_uniform_tiles, tiled_decode, multi_job_id, enabled_worker_ids,
                                       dynamic_threshold, auto_morph_output, auto_morph_blur,
                                       highpass_blend, hp_blur_size, hp_opacity, hp_device, hp_store_result_on, hp_contrast,
                                       preserve_gradients_blend, pg_highpass_radius, pg_downscale_by, pg_strength, pg_expand, pg_mask_blur,
                                       pg_clamp_blacks, pg_clamp_whites, pg_temporal_deflicker, pg_device, post_sharpen, pre_sharpen)

    def process_worker(self, upscaled_image, model, positive, negative, vae,
                       seed, steps, cfg, sampler_name, scheduler, denoise,
                       tile_width, tile_height, padding, mask_blur, mask_expand,
                       force_uniform_tiles, tiled_decode, multi_job_id, master_url,
                       worker_id, enabled_worker_ids, dynamic_threshold, auto_morph_output, auto_morph_blur,
                       highpass_blend, hp_blur_size, hp_opacity, hp_device, hp_store_result_on, hp_contrast,
                       preserve_gradients_blend, pg_highpass_radius, pg_downscale_by, pg_strength, pg_expand, pg_mask_blur,
                       pg_clamp_blacks, pg_clamp_whites, pg_temporal_deflicker, pg_device, post_sharpen, pre_sharpen):
        """Unified worker processing (PIL at edges, tensors only inside process_tiles_batch)."""
        batch_size, height, width = self._infer_batch_and_size_from_pil(upscaled_image)
        enabled_workers = json.loads(enabled_worker_ids)
        num_workers = len(enabled_workers)
        all_tiles = self.calculate_tiles(width, height,
                                         self.round_to_multiple(tile_width),
                                         self.round_to_multiple(tile_height),
                                         force_uniform_tiles)
        num_tiles_per_image = len(all_tiles)

        mode = self._determine_processing_mode(batch_size, num_workers, dynamic_threshold)
        if num_workers > 0 and num_tiles_per_image > 1:
            mode = "static"

        debug_log(f"USDU Dist Worker - Batch size {batch_size}")

        # Always static batched-per-tile in this implementation
        return self._process_worker_static_sync(upscaled_image, model, positive, negative, vae,
                                                seed, steps, cfg, sampler_name, scheduler, denoise,
                                                tile_width, tile_height, padding, mask_blur, mask_expand,
                                                force_uniform_tiles, tiled_decode, multi_job_id, master_url,
                                                worker_id, enabled_workers, auto_morph_output, auto_morph_blur,
                                                highpass_blend, hp_blur_size, hp_opacity, hp_device, hp_store_result_on, hp_contrast,
                                                preserve_gradients_blend, pg_highpass_radius, pg_downscale_by, pg_strength, pg_expand,
                                                pg_mask_blur, pg_clamp_blacks, pg_clamp_whites, pg_temporal_deflicker, pg_device,
                                                post_sharpen, pre_sharpen)

    def process_master(self, upscaled_image, model, positive, negative, vae,
                       seed, steps, cfg, sampler_name, scheduler, denoise,
                       tile_width, tile_height, padding, mask_blur, mask_expand,
                       force_uniform_tiles, tiled_decode, multi_job_id, enabled_worker_ids,
                       dynamic_threshold, auto_morph_output, auto_morph_blur,
                       highpass_blend, hp_blur_size, hp_opacity, hp_device, hp_store_result_on, hp_contrast,
                       preserve_gradients_blend, pg_highpass_radius, pg_downscale_by, pg_strength, pg_expand, pg_mask_blur,
                       pg_clamp_blacks, pg_clamp_whites, pg_temporal_deflicker, pg_device, post_sharpen, pre_sharpen):
        """Unified master processing (PIL outer, tensors only inside process_tiles_batch)."""
        tile_width = self.round_to_multiple(tile_width)
        tile_height = self.round_to_multiple(tile_height)

        # Fresh progress bar unless explicitly preserved by video wrapper
        if not getattr(self, "_progress_keepalive", False):
            self._progress = None

        batch_size, height, width = self._infer_batch_and_size_from_pil(upscaled_image)

        all_tiles = self.calculate_tiles(width, height, tile_width, tile_height, force_uniform_tiles)
        num_tiles_per_image = len(all_tiles)
        rows = math.ceil(height / tile_height)
        cols = math.ceil(width / tile_width)
        log(f"USDU Dist: Canvas {width}x{height} | Tile {tile_width}x{tile_height} | Grid {rows}x{cols} ({num_tiles_per_image} tiles/image) | Batch {batch_size}")

        enabled_workers = json.loads(enabled_worker_ids)
        num_workers = len(enabled_workers)

        mode = self._determine_processing_mode(batch_size, num_workers, dynamic_threshold)
        if num_workers > 0 and num_tiles_per_image > 1:
            mode = "static"

        log(f"USDU Dist: Workers {num_workers}")

        if mode == "single_gpu":
            return self.process_single_gpu(upscaled_image, model, positive, negative, vae,
                                           seed, steps, cfg, sampler_name, scheduler, denoise,
                                           tile_width, tile_height, padding, mask_blur, mask_expand,
                                           force_uniform_tiles, tiled_decode, auto_morph_output, auto_morph_blur,
                                           highpass_blend, hp_blur_size, hp_opacity, hp_device, hp_store_result_on, hp_contrast,
                                           preserve_gradients_blend, pg_highpass_radius, pg_downscale_by, pg_strength, pg_expand, pg_mask_blur,
                                           pg_clamp_blacks, pg_clamp_whites, pg_temporal_deflicker, pg_device, post_sharpen, pre_sharpen)

        return self._process_master_static_sync(upscaled_image, model, positive, negative, vae,
                                                seed, steps, cfg, sampler_name, scheduler, denoise,
                                                tile_width, tile_height, padding, mask_blur, mask_expand,
                                                force_uniform_tiles, tiled_decode, multi_job_id, enabled_workers,
                                                all_tiles, num_tiles_per_image, auto_morph_output, auto_morph_blur,
                                                highpass_blend, hp_blur_size, hp_opacity, hp_device, hp_store_result_on, hp_contrast,
                                                preserve_gradients_blend, pg_highpass_radius, pg_downscale_by, pg_strength, pg_expand, pg_mask_blur,
                                                pg_clamp_blacks, pg_clamp_whites, pg_temporal_deflicker, pg_device, post_sharpen, pre_sharpen)

    def round_to_multiple(self, value: int, multiple: int = 8) -> int:
        """Round value to nearest multiple."""
        return round(value / multiple) * multiple

    def calculate_tiles(self, image_width: int, image_height: int,
                        tile_width: int, tile_height: int, force_uniform_tiles: bool = True) -> List[Tuple[int, int]]:
        """Calculate tile positions (simple grid)."""
        rows = math.ceil(image_height / tile_height)
        cols = math.ceil(image_width / tile_width)
        tiles: List[Tuple[int, int]] = []
        for yi in range(rows):
            for xi in range(cols):
                tiles.append((xi * tile_width, yi * tile_height))
        return tiles

    def extract_batch_tile_with_padding(self, images_pil: List[Image.Image], x: int, y: int,
                                        tile_width: int, tile_height: int, padding: int,
                                        force_uniform_tiles: bool) -> Tuple[torch.Tensor, int, int, int, int]:
        """
        Extract a tile ROI for the entire batch from a list of RGB PIL images.
        Returns a BHWC tensor batch for process_tiles_batch and the crop info (x1, y1, extracted_w, extracted_h).
        """
        # All images are same size by assumption
        w, h = images_pil[0].size

        # Build mask and compute padded crop region via USDU helpers
        mask = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([x, y, x + tile_width, y + tile_height], fill=255)
        x1, y1, x2, y2 = get_crop_region(mask, padding)

        if force_uniform_tiles:
            target_w = self.round_to_multiple(tile_width + padding, 8)
            target_h = self.round_to_multiple(tile_height + padding, 8)
            (x1, y1, x2, y2), (target_w, target_h) = expand_crop((x1, y1, x2, y2), w, h, target_w, target_h)
        else:
            crop_w = x2 - x1
            crop_h = y2 - y1
            target_w = max(8, math.ceil(crop_w / 8) * 8)
            target_h = max(8, math.ceil(crop_h / 8) * 8)
            (x1, y1, x2, y2), (target_w, target_h) = expand_crop((x1, y1, x2, y2), w, h, target_w, target_h)

        extracted_width = x2 - x1
        extracted_height = y2 - y1

        # Crop and resize each PIL tile; then convert to BHWC tensor and stack
        tensors = []
        for img in images_pil:
            if img.mode != "RGB":
                img = img.convert("RGB")
            tile_pil = img.crop((x1, y1, x2, y2))
            if tile_pil.size != (target_w, target_h):
                tile_pil = tile_pil.resize((target_w, target_h), Image.LANCZOS)
            t = pil_to_tensor(tile_pil)  # -> [1,H,W,C]
            tensors.append(t)

        tile_batch = torch.cat(tensors, dim=0)  # [B,H,W,C]
        return tile_batch, x1, y1, extracted_width, extracted_height

    def create_tile_mask(self, image_width: int, image_height: int,
                         x: int, y: int, tile_width: int, tile_height: int,
                         mask_blur: int, mask_expand: int) -> Image.Image:
        """
        Build a PIL mask for blending a tile.
        """
        W, H = int(image_width), int(image_height)
        x0, y0 = int(x), int(y)
        x1, y1 = x0 + int(tile_width), y0 + int(tile_height)
        e = int(mask_expand)

        if e >= 0:
            rx0 = x0 - e; ry0 = y0 - e; rx1 = x1 + e; ry1 = y1 + e
        else:
            c = -e
            left_shrink  = 0 if x0 <= 0 else c
            top_shrink   = 0 if y0 <= 0 else c
            right_shrink = 0 if x1 >= W else c
            bot_shrink   = 0 if y1 >= H else c
            rx0 = x0 + left_shrink
            ry0 = y0 + top_shrink
            rx1 = x1 - right_shrink
            ry1 = y1 - bot_shrink

        rx0 = max(0, min(W, rx0)); ry0 = max(0, min(H, ry0))
        rx1 = max(0, min(W, rx1)); ry1 = max(0, min(H, ry1))

        if rx1 <= rx0:
            mid = (rx0 + rx1) // 2
            rx0 = max(0, min(W - 1, mid)); rx1 = min(W, rx0 + 1)
        if ry1 <= ry0:
            mid = (ry0 + ry1) // 2
            ry0 = max(0, min(H - 1, mid)); ry1 = min(H, ry0 + 1)

        mask = Image.new('L', (W, H), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([rx0, ry0, rx1, ry1], fill=255)
        if mask_blur > 0:
            mask = mask.filter(ImageFilter.BoxBlur(int(mask_blur)))
        return mask

    def blend_tile(self, base_image: Image.Image, tile_image: Image.Image,
                   x: int, y: int, extracted_size: Tuple[int, int],
                   mask: Image.Image, padding: int) -> Image.Image:
        """In-place PIL blending using paste with local cropped mask."""
        ew, eh = int(extracted_size[0]), int(extracted_size[1])
        if tile_image.mode != "RGB":
            tile_image = tile_image.convert("RGB")
        if tile_image.size != (ew, eh):
            tile_image = tile_image.resize((ew, eh), Image.LANCZOS)
        if base_image.mode != "RGB":
            base_image = base_image.convert("RGB")
        x2, y2 = x + ew, y + eh
        local_mask = mask.crop((x, y, x2, y2))
        base_image.paste(tile_image, (x, y), local_mask)
        return base_image

    # --------- unchanged tensor core for tiles batch (kept by design) ----------
    def process_tiles_batch(self, tile_batch: torch.Tensor, model, positive, negative, vae,
                            seed: int, steps: int, cfg: float, sampler_name: str,
                            scheduler: str, denoise: float, tiled_decode: bool,
                            region: Tuple[int, int, int, int], image_size: Tuple[int, int],
                            auto_morph_output: bool, auto_morph_blur: int,
                            highpass_blend: bool, hp_blur_size: int, hp_opacity: float, hp_device: str, hp_store_result_on: str, hp_contrast: float,
                            preserve_gradients_blend: bool, pg_highpass_radius: int, pg_downscale_by: float, pg_strength: float, pg_expand: int, pg_mask_blur: int,
                            pg_clamp_blacks: float, pg_clamp_whites: float, pg_temporal_deflicker: bool, pg_device: str, post_sharpen: float, pre_sharpen: float
                            ) -> torch.Tensor:
        """
        UNCHANGED INTERNALS: tensor pipeline (encode → sample → decode), optional blends, sharpen.
        """
        from nodes import common_ksampler, VAEEncode, VAEDecode
        try:
            from nodes import VAEEncodeTiled, VAEDecodeTiled
            tiled_vae_available = True
        except ImportError:
            tiled_vae_available = False

        clean = tile_batch.detach()
        if hasattr(clean, 'requires_grad_'):
            clean.requires_grad_(False)

        positive_tile = clone_conditioning(positive, clone_hints=True)
        negative_tile = clone_conditioning(negative, clone_hints=True)

        init_size = image_size
        canvas_size = image_size
        tile_size = (clean.shape[2], clean.shape[1])  # (W,H)
        w_pad = 0; h_pad = 0
        positive_cropped = crop_cond(positive_tile, region, init_size, canvas_size, tile_size, w_pad, h_pad)
        negative_cropped = crop_cond(negative_tile, region, init_size, canvas_size, tile_size, w_pad, h_pad)

        if pre_sharpen != 0.0:
            log(f"Applying pre-sharpen")
            clean = sharpen_tiny(clean, alpha=pre_sharpen)

        log(f"Encoding to latent")
        latent = VAEEncode().encode(vae, clean)[0]
        log(f"Starting ksampler")

        _saved_pb = self._progress  # keep current tile-level bar
        samples = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive_cropped, negative_cropped, latent, denoise=denoise)[0]
        self._progress = _saved_pb  # restore tile-level bar
        if self._progress is not None:
            self._progress.update(0)  # force redraw without changing counts

        log(f"Decoding back to image")
        if tiled_decode and tiled_vae_available:
            image = VAEDecodeTiled().decode(vae, samples, tile_size=512)[0]
        else:
            image = VAEDecode().decode(vae, samples)[0]
        log(f"Decoding is done")

        if auto_morph_output:
            aligned_cpu = auto_morph_align_batch_bhwc(clean, image, blur=int(auto_morph_blur))
            image = aligned_cpu.to(image.device, non_blocking=True)

        if highpass_blend:
            aligned_cpu = HighPassBlend().apply(image, clean, hp_blur_size, hp_opacity, hp_contrast, hp_device, hp_store_result_on, False)[0]
            image = aligned_cpu.to(image.device, non_blocking=True)

        if preserve_gradients_blend:
            aligned_cpu = PreserveGradientsBlend().apply(image, clean, pg_highpass_radius, pg_downscale_by, pg_strength, pg_expand,
                                                         pg_mask_blur, pg_clamp_blacks, pg_clamp_whites, pg_temporal_deflicker, pg_device, False)[0]
            image = aligned_cpu.to(image.device, non_blocking=True)

        if post_sharpen != 0.0:
            log(f"Applying post-sharpen")
            image = sharpen_tiny(image, alpha=post_sharpen)

        return image
    # ---------------------------------------------------------------------------

    def _slice_conditioning(self, positive, negative, batch_idx):
        """Helper to slice conditioning for a specific batch index."""
        positive_sliced = clone_conditioning(positive)
        negative_sliced = clone_conditioning(negative)

        for cond_list in [positive_sliced, negative_sliced]:
            for i in range(len(cond_list)):
                emb, cond_dict = cond_list[i]
                if emb.shape[0] > 1:
                    cond_list[i][0] = emb[batch_idx:batch_idx+1]
                if 'control' in cond_dict:
                    control = cond_dict['control']
                    while control is not None:
                        hint = control.cond_hint_original
                        if hint.shape[0] > 1:
                            control.cond_hint_original = hint[batch_idx:batch_idx+1]
                        control = control.previous_controlnet
                if 'mask' in cond_dict and cond_dict['mask'].shape[0] > 1:
                    cond_dict['mask'] = cond_dict['mask'][batch_idx:batch_idx+1]
        return positive_sliced, negative_sliced

    async def _get_all_completed_tasks(self, multi_job_id):
        """Helper to retrieve all completed tasks from the job data."""
        prompt_server = ensure_tile_jobs_initialized()
        async with prompt_server.distributed_tile_jobs_lock:
            job_data = prompt_server.distributed_pending_tile_jobs.get(multi_job_id)
            if job_data and JOB_COMPLETED_TASKS in job_data:
                return dict(job_data[JOB_COMPLETED_TASKS])
            return {}

    def _process_worker_static_sync(self, upscaled_image_pil_list, model, positive, negative, vae,
                                    seed, steps, cfg, sampler_name, scheduler, denoise,
                                    tile_width, tile_height, padding, mask_blur, mask_expand,
                                    force_uniform_tiles, tiled_decode, multi_job_id, master_url,
                                    worker_id, enabled_workers, auto_morph_output, auto_morph_blur,
                                    highpass_blend, hp_blur_size, hp_opacity, hp_device, hp_store_result_on, hp_contrast,
                                    preserve_gradients_blend, pg_highpass_radius, pg_downscale_by, pg_strength, pg_expand, pg_mask_blur,
                                    pg_clamp_blacks, pg_clamp_whites, pg_temporal_deflicker, pg_device, post_sharpen, pre_sharpen):
        """Worker static mode: pull tile-ids, process batched-per-tile, send PIL results."""
        tile_width = self.round_to_multiple(tile_width)
        tile_height = self.round_to_multiple(tile_height)

        batch_size, height, width = self._infer_batch_and_size_from_pil(upscaled_image_pil_list)
        all_tiles = self.calculate_tiles(width, height, tile_width, tile_height, force_uniform_tiles)
        num_tiles_per_image = len(all_tiles)
        total_tiles = batch_size * num_tiles_per_image

        processed_tiles = []
        log(f"USDU Dist Worker[{worker_id[:8]}]: Canvas {width}x{height} | Tile {tile_width}x{tile_height} | Tiles/image {num_tiles_per_image} | Batch {batch_size}")

        # Wait until job ready
        max_poll_attempts = 1200
        for attempt in range(max_poll_attempts):
            ready = run_async_in_server_loop(self._check_job_status(multi_job_id, master_url), timeout=5.0)
            if ready:
                debug_log(f"Worker[{worker_id[:8]}] job {multi_job_id} ready after {attempt} attempts")
                break
            time.sleep(1.0)
        else:
            log(f"Job {multi_job_id} not ready after {max_poll_attempts} attempts, aborting")
            return (upscaled_image_pil_list,)

        while True:
            tile_idx, estimated_remaining, batched_static = run_async_in_server_loop(
                self._request_tile_from_master(multi_job_id, master_url, worker_id),
                timeout=TILE_WAIT_TIMEOUT
            )
            if tile_idx is None:
                debug_log(f"Worker[{worker_id[:8]}] - No more tiles to process")
                break

            tx, ty = all_tiles[tile_idx]
            tile_batch, x1, y1, ew, eh = self.extract_batch_tile_with_padding(
                upscaled_image_pil_list, tx, ty, tile_width, tile_height, padding, force_uniform_tiles
            )
            region = (x1, y1, x1 + ew, y1 + eh)
            processed_batch = self.process_tiles_batch(
                tile_batch, model, positive, negative, vae,
                seed, steps, cfg, sampler_name, scheduler, denoise, tiled_decode,
                region, (width, height), auto_morph_output, auto_morph_blur,
                highpass_blend, hp_blur_size, hp_opacity, hp_device, hp_store_result_on, hp_contrast,
                preserve_gradients_blend, pg_highpass_radius, pg_downscale_by, pg_strength, pg_expand, pg_mask_blur,
                pg_clamp_blacks, pg_clamp_whites, pg_temporal_deflicker, pg_device, post_sharpen, pre_sharpen
            )

            ready_tiles = []
            for b in range(batch_size):
                t_pil = tensor_to_pil(processed_batch, b)
                if t_pil.size != (ew, eh):
                    t_pil = t_pil.resize((ew, eh), Image.LANCZOS)
                ready_tiles.append(t_pil)

            for b, t_pil in enumerate(ready_tiles):
                processed_tiles.append({
                    'image': t_pil,
                    'tile_idx': tile_idx,
                    'x': x1,
                    'y': y1,
                    'extracted_width': ew,
                    'extracted_height': eh,
                    'padding': padding,
                    'batch_idx': b,
                    'global_idx': b * num_tiles_per_image + tile_idx
                })

            try:
                run_async_in_server_loop(_send_heartbeat_to_master(multi_job_id, master_url, worker_id), timeout=5.0)
            except Exception as e:
                debug_log(f"Worker[{worker_id[:8]}] heartbeat failed: {e}")

            if len(processed_tiles) >= MAX_BATCH:
                run_async_in_server_loop(
                    self.send_tiles_batch_to_master(processed_tiles, multi_job_id, master_url, padding, worker_id),
                    timeout=TILE_SEND_TIMEOUT
                )
                processed_tiles = []

        if processed_tiles:
            run_async_in_server_loop(
                self.send_tiles_batch_to_master(processed_tiles, multi_job_id, master_url, padding, worker_id),
                timeout=TILE_SEND_TIMEOUT
            )

        debug_log(f"Worker {worker_id} completed all assigned and requeued tiles")
        return (upscaled_image_pil_list,)

    async def _async_collect_and_monitor_static(self, multi_job_id, total_tiles, expected_total):
        """Async helper for collection and monitoring in static mode."""
        last_progress_log = time.time()
        progress_interval = 5.0
        last_heartbeat_check = time.time()

        while True:
            if comfy.model_management.processing_interrupted():
                log("Processing interrupted by user")
                raise comfy.model_management.InterruptProcessingException()

            collected_count = await _drain_results_queue(multi_job_id)

            # Update progress for tiles finished on workers
            if collected_count:
                try:
                    if self._progress is not None:
                        self._progress.update(collected_count)
                except Exception:
                    pass

            current_time = time.time()
            if current_time - last_heartbeat_check >= 10.0:
                requeued_count = await self._check_and_requeue_timed_out_workers(multi_job_id, expected_total)
                if requeued_count > 0:
                    log(f"Requeued {requeued_count} tasks from timed-out workers")
                last_heartbeat_check = current_time

            completed_count = await _get_completed_count(multi_job_id)

            if current_time - last_progress_log >= progress_interval:
                log(f"Progress: {completed_count}/{expected_total} tasks completed")
                last_progress_log = current_time

            if completed_count >= expected_total:
                debug_log(f"All {expected_total} tasks completed")
                break

            prompt_server = ensure_tile_jobs_initialized()
            async with prompt_server.distributed_tile_jobs_lock:
                job_data = prompt_server.distributed_pending_tile_jobs.get(multi_job_id)
                if job_data:
                    pending_queue = job_data.get(JOB_PENDING_TASKS)
                    active_workers = list(job_data.get(JOB_WORKER_STATUS, {}).keys())
                    if pending_queue and not pending_queue.empty() and len(active_workers) == 0:
                        log(f"No active workers remaining with {expected_total - completed_count} tasks pending. Returning for local processing.")
                        break

            await asyncio.sleep(0.1)

        return await self._get_all_completed_tasks(multi_job_id)

    def _process_master_static_sync(self, upscaled_image_pil_list, model, positive, negative, vae,
                                    seed, steps, cfg, sampler_name, scheduler, denoise,
                                    tile_width, tile_height, padding, mask_blur, mask_expand,
                                    force_uniform_tiles, tiled_decode, multi_job_id, enabled_workers,
                                    all_tiles, num_tiles_per_image, auto_morph_output, auto_morph_blur,
                                    highpass_blend, hp_blur_size, hp_opacity, hp_device, hp_store_result_on, hp_contrast,
                                    preserve_gradients_blend, pg_highpass_radius, pg_downscale_by, pg_strength, pg_expand, pg_mask_blur,
                                    pg_clamp_blacks, pg_clamp_whites, pg_temporal_deflicker, pg_device, post_sharpen, pre_sharpen):
        """Master static mode (PIL outer; tensors only for process_tiles_batch)."""
        batch_size, height, width = self._infer_batch_and_size_from_pil(upscaled_image_pil_list)
        total_tiles = batch_size * num_tiles_per_image
        # Fresh bar unless a cumulative one was prepared by the video wrapper
        if self._progress is None or not getattr(self, "_progress_keepalive", False):
            self._progress = ProgressBar(total_tiles)
            self._progress_keepalive = False  # default: do not carry over across runs


        result_images = self._ensure_rgb_copies(upscaled_image_pil_list)

        log("USDU Dist: Using tile queue distribution")
        run_async_in_server_loop(
            init_static_job_batched(multi_job_id, batch_size, num_tiles_per_image, enabled_workers),
            timeout=10.0
        )
        debug_log(f"Initialized tile-id queue with {num_tiles_per_image} ids for batch {batch_size}")

        tile_masks = []
        for idx, (tx, ty) in enumerate(all_tiles):
            tile_masks.append(self.create_tile_mask(width, height, tx, ty, tile_width, tile_height, mask_blur, mask_expand))

        processed_count = 0
        consecutive_no_tile = 0
        max_consecutive_no_tile = 2

        while processed_count < total_tiles:
            comfy.model_management.throw_exception_if_processing_interrupted()
            tile_idx = run_async_in_server_loop(self._get_next_tile_index(multi_job_id), timeout=5.0)
            if tile_idx is not None:
                consecutive_no_tile = 0
                processed_count += batch_size
                # Count this tile processed for the whole batch
                try:
                    if self._progress is not None:
                        self._progress.update(batch_size)
                except Exception:
                    pass
                tx, ty = all_tiles[tile_idx]
                tile_batch, x1, y1, ew, eh = self.extract_batch_tile_with_padding(
                    result_images, tx, ty, tile_width, tile_height, padding, force_uniform_tiles
                )
                region = (x1, y1, x1 + ew, y1 + eh)
                processed_batch = self.process_tiles_batch(
                    tile_batch, model, positive, negative, vae,
                    seed, steps, cfg, sampler_name, scheduler, denoise, tiled_decode,
                    region, (width, height), auto_morph_output, auto_morph_blur,
                    highpass_blend, hp_blur_size, hp_opacity, hp_device, hp_store_result_on, hp_contrast,
                    preserve_gradients_blend, pg_highpass_radius, pg_downscale_by, pg_strength, pg_expand, pg_mask_blur,
                    pg_clamp_blacks, pg_clamp_whites, pg_temporal_deflicker, pg_device, post_sharpen, pre_sharpen
                )
                tile_mask = tile_masks[tile_idx]
                for b in range(batch_size):
                    tile_pil = tensor_to_pil(processed_batch, b)
                    if tile_pil.size != (ew, eh):
                        tile_pil = tile_pil.resize((ew, eh), Image.LANCZOS)
                    result_images[b] = self.blend_tile(result_images[b], tile_pil, x1, y1, (ew, eh), tile_mask, padding)
                    global_idx = b * num_tiles_per_image + tile_idx
                    run_async_in_server_loop(_mark_task_completed(multi_job_id, global_idx, {'batch_idx': b, 'tile_idx': tile_idx}), timeout=5.0)
                log(f"USDU Dist: Tiles progress {processed_count}/{total_tiles} (tile {tile_idx})")
            else:
                consecutive_no_tile += 1
                if consecutive_no_tile >= max_consecutive_no_tile:
                    debug_log(f"Master processed {processed_count} tiles, moving to collection phase")
                    break
                time.sleep(0.1)
        master_processed_count = processed_count

        remaining_tiles = total_tiles - master_processed_count
        if remaining_tiles > 0:
            debug_log(f"Master waiting for {remaining_tiles} tiles from workers")
            try:
                collected_tasks = run_async_in_server_loop(
                    self._async_collect_and_monitor_static(multi_job_id, total_tiles, expected_total=total_tiles),
                    timeout=None
                )
            except comfy.model_management.InterruptProcessingException:
                run_async_in_server_loop(_cleanup_job(multi_job_id), timeout=5.0)
                raise

            completed_count = len(collected_tasks)
            if completed_count < total_tiles:
                log(f"Processing remaining {total_tiles - completed_count} tasks locally after worker failures")
                while True:
                    comfy.model_management.throw_exception_if_processing_interrupted()
                    tile_id = run_async_in_server_loop(self._get_next_tile_index(multi_job_id), timeout=5.0)
                    if tile_id is None:
                        break
                    tx, ty = all_tiles[tile_id]
                    tile_batch, x1, y1, ew, eh = self.extract_batch_tile_with_padding(
                        result_images, tx, ty, tile_width, tile_height, padding, force_uniform_tiles
                    )
                    region = (x1, y1, x1 + ew, y1 + eh)
                    processed_batch = self.process_tiles_batch(
                        tile_batch, model, positive, negative, vae,
                        seed, steps, cfg, sampler_name, scheduler, denoise, tiled_decode,
                        region, (width, height), auto_morph_output, auto_morph_blur,
                        highpass_blend, hp_blur_size, hp_opacity, hp_device, hp_store_result_on, hp_contrast,
                        preserve_gradients_blend, pg_highpass_radius, pg_downscale_by, pg_strength, pg_expand, pg_mask_blur,
                        pg_clamp_blacks, pg_clamp_whites, pg_temporal_deflicker, pg_device, post_sharpen, pre_sharpen
                    )
                    tile_mask = tile_masks[tile_id]
                    out_bs = processed_batch.shape[0]
                    for b in range(min(batch_size, out_bs)):
                        tile_pil = tensor_to_pil(processed_batch, b)
                        if tile_pil.size != (ew, eh):
                            tile_pil = tile_pil.resize((ew, eh), Image.LANCZOS)
                        result_images[b] = self.blend_tile(result_images[b], tile_pil, x1, y1, (ew, eh), tile_mask, padding)
                        global_idx = b * num_tiles_per_image + tile_id
                        run_async_in_server_loop(_mark_task_completed(multi_job_id, global_idx, {'batch_idx': b, 'tile_idx': tile_id}), timeout=5.0)
                    # Count locally recovered tile for however many outputs we actually produced
                    try:
                        if self._progress is not None:
                            self._progress.update(min(batch_size, out_bs))
                    except Exception:
                        pass
        else:
            collected_tasks = run_async_in_server_loop(self._get_all_completed_tasks(multi_job_id), timeout=5.0)

        log(f"Blend worker tiles synchronously")

        for global_idx, tile_data in collected_tasks.items():
            if 'tensor' not in tile_data and 'image' not in tile_data:
                continue
            batch_idx = tile_data.get('batch_idx', global_idx // num_tiles_per_image)
            tile_idx = tile_data.get('tile_idx', global_idx % num_tiles_per_image)
            if batch_idx >= batch_size:
                continue
            x = tile_data.get('x', 0)
            y = tile_data.get('y', 0)
            if 'image' in tile_data:
                tile_pil = tile_data['image']
            else:
                tile_tensor = tile_data['tensor']
                tile_pil = tensor_to_pil(tile_tensor, 0)
            orig_x, orig_y = all_tiles[tile_idx]
            tile_mask = tile_masks[tile_idx]
            extracted_width = tile_data.get('extracted_width', tile_width + 2 * padding)
            extracted_height = tile_data.get('extracted_height', tile_height + 2 * padding)
            result_images[batch_idx] = self.blend_tile(result_images[batch_idx], tile_pil,
                                                       x, y, (extracted_width, extracted_height), tile_mask, padding)
        try:
            # Return list of RGB PIL frames (final)
            return (result_images,)
        finally:
            if not getattr(self, "_progress_keepalive", False):
                self._progress = None
            else:
                self._progress_keepalive = False
            run_async_in_server_loop(_cleanup_job(multi_job_id), timeout=5.0)


    async def send_tiles_batch_to_master(self, processed_tiles, multi_job_id, master_url,
                                         padding, worker_id):
        """Send PIL tiles (JPEG-encoded) to master with size-aware chunking."""
        if not processed_tiles:
            return
        total_tiles = len(processed_tiles)
        log(f"Worker[{worker_id[:8]}] - Preparing to send {total_tiles} tiles (size-aware chunks)")

        try:
            executor = get_jpeg_executor()
            encoded = list(executor.map(_encode_tile_data, processed_tiles))
        except Exception:
            encoded = [_encode_tile_data(td) for td in processed_tiles]

        max_bytes = int(MAX_PAYLOAD_SIZE) - (1024 * 1024)
        i = 0
        chunk_index = 0
        while i < total_tiles:
            data = aiohttp.FormData()
            data.add_field('multi_job_id', multi_job_id)
            data.add_field('worker_id', str(worker_id))
            data.add_field('padding', str(padding))

            metadata = []
            used = 0
            j = i
            while j < total_tiles:
                img_bytes = encoded[j]['bytes']
                meta = encoded[j]['meta']
                overhead = 1024
                if used + len(img_bytes) + overhead > max_bytes and j > i:
                    break
                metadata.append(meta)
                data.add_field(f'tile_{j - i}', io.BytesIO(img_bytes), filename=f'tile_{j}.jpg', content_type='image/jpeg')
                used += len(img_bytes) + overhead
                j += 1

            if j == i:
                meta = encoded[j]['meta']
                metadata.append(meta)
                data.add_field('tile_0', io.BytesIO(encoded[j]['bytes']), filename=f'tile_{j}.jpg', content_type='image/jpeg')
                j += 1

            chunk_size = j - i
            is_chunk_last = (j >= total_tiles)
            data.add_field('is_last', str(is_chunk_last))
            data.add_field('batch_size', str(chunk_size))
            data.add_field('tiles_metadata', json.dumps(metadata), content_type='application/json')

            max_retries = 5
            retry_delay = 0.5
            for attempt in range(max_retries):
                try:
                    session = await get_client_session()
                    url = f"{master_url}/distributed/submit_tiles"
                    async with session.post(url, data=data) as response:
                        response.raise_for_status()
                        break
                except Exception as e:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        retry_delay = min(retry_delay * 2, 5.0)
                    else:
                        log(f"UltimateSDUpscale Worker - Failed to send chunk {chunk_index} after {max_retries} attempts: {e}")
                        raise

            log(f"Worker[{worker_id[:8]}] - Sent chunk {chunk_index} ({chunk_size} tiles, ~{used/1e6:.2f} MB)")
            chunk_index += 1
            i = j

    def process_single_gpu(self, upscaled_image_pil_list, model, positive, negative, vae,
                           seed, steps, cfg, sampler_name, scheduler, denoise,
                           tile_width, tile_height, padding, mask_blur, mask_expand,
                           force_uniform_tiles, tiled_decode, auto_morph_output, auto_morph_blur,
                           highpass_blend, hp_blur_size, hp_opacity, hp_device, hp_store_result_on, hp_contrast,
                           preserve_gradients_blend, pg_highpass_radius, pg_downscale_by, pg_strength, pg_expand, pg_mask_blur,
                           pg_clamp_blacks, pg_clamp_whites, pg_temporal_deflicker, pg_device, post_sharpen, pre_sharpen):
        """Single-GPU path with PIL edges and tensor core inside process_tiles_batch."""
        tile_width = self.round_to_multiple(tile_width)
        tile_height = self.round_to_multiple(tile_height)

        # Fresh progress bar unless explicitly preserved by video wrapper
        if not getattr(self, "_progress_keepalive", False):
            self._progress = None

        batch_size, height, width = self._infer_batch_and_size_from_pil(upscaled_image_pil_list)

        all_tiles = self.calculate_tiles(width, height, tile_width, tile_height, force_uniform_tiles)
        rows = math.ceil(height / tile_height)
        cols = math.ceil(width / tile_width)
        log(f"USDU Dist: Single GPU | Canvas {width}x{height} | Tile {tile_width}x{tile_height} | Grid {rows}x{cols} ({len(all_tiles)} tiles/image) | Batch {batch_size}")

        # Local-only progress for single GPU (no workers involved)
        local_pbar = ProgressBar(len(all_tiles) * batch_size)

        result_images = self._ensure_rgb_copies(upscaled_image_pil_list)

        tile_masks = []
        for tx, ty in all_tiles:
            tile_masks.append(self.create_tile_mask(width, height, tx, ty, tile_width, tile_height, mask_blur, mask_expand))

        for tile_idx, (tx, ty) in enumerate(all_tiles):
            tile_batch, x1, y1, ew, eh = self.extract_batch_tile_with_padding(
                result_images, tx, ty, tile_width, tile_height, padding, force_uniform_tiles
            )
            region = (x1, y1, x1 + ew, y1 + eh)
            processed_batch = self.process_tiles_batch(
                tile_batch, model, positive, negative, vae,
                seed, steps, cfg, sampler_name, scheduler, denoise, tiled_decode,
                region, (width, height), auto_morph_output, auto_morph_blur,
                highpass_blend, hp_blur_size, hp_opacity, hp_device, hp_store_result_on, hp_contrast,
                preserve_gradients_blend, pg_highpass_radius, pg_downscale_by, pg_strength, pg_expand, pg_mask_blur,
                pg_clamp_blacks, pg_clamp_whites, pg_temporal_deflicker, pg_device, post_sharpen, pre_sharpen
            )

            tile_mask = tile_masks[tile_idx]
            for b in range(batch_size):
                tile_pil = tensor_to_pil(processed_batch, b)
                if tile_pil.size != (ew, eh):
                    tile_pil = tile_pil.resize((ew, eh), Image.LANCZOS)
                result_images[b] = self.blend_tile(result_images[b], tile_pil, x1, y1, (ew, eh), tile_mask, padding)

            # Count this tile processed for the whole batch
            try:
                local_pbar.update(batch_size)
            except Exception:
                pass

        # Final output: list of RGB PIL frames
        return (result_images,)

    async def _get_next_tile_index(self, multi_job_id):
        """Get next tile index from pending queue for master in static mode."""
        prompt_server = ensure_tile_jobs_initialized()
        async with prompt_server.distributed_tile_jobs_lock:
            job_data = prompt_server.distributed_pending_tile_jobs.get(multi_job_id)
            if not job_data or JOB_PENDING_TASKS not in job_data:
                return None
            try:
                tile_idx = await asyncio.wait_for(job_data[JOB_PENDING_TASKS].get(), timeout=0.1)
                return tile_idx
            except asyncio.TimeoutError:
                return None

    async def _check_and_requeue_timed_out_workers(self, multi_job_id, batch_size):
        """Requeue timed-out workers' tasks (delegated)."""
        return await _check_and_requeue_timed_out_workers(multi_job_id, batch_size)

    async def _request_tile_from_master(self, multi_job_id, master_url, worker_id):
        """Request a tile index from master (reusing dynamic infra)."""
        max_retries = 10
        retry_delay = 0.5
        start_time = time.time()

        for attempt in range(max_retries):
            if time.time() - start_time > 30:
                log(f"Total request timeout after 30s for worker {worker_id}")
                return None, 0, False
            try:
                session = await get_client_session()
                url = f"{master_url}/distributed/request_image"
                async with session.post(url, json={'worker_id': str(worker_id), 'multi_job_id': multi_job_id}) as response:
                    if response.status == 200:
                        data = await response.json()
                        tile_idx = data.get('tile_idx')
                        estimated_remaining = data.get('estimated_remaining', 0)
                        batched_static = data.get('batched_static', False)
                        return tile_idx, estimated_remaining, batched_static
                    elif response.status == 404:
                        text = await response.text()
                        debug_log(f"Job not found (404), will retry: {text}")
                        await asyncio.sleep(1.0)
                    else:
                        text = await response.text()
                        debug_log(f"Request tile failed: {response.status} - {text}")
            except Exception as e:
                if attempt < max_retries - 1:
                    debug_log(f"Retry {attempt + 1}/{max_retries} after error: {e}")
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, 5.0)
                else:
                    log(f"Failed to request tile after {max_retries} attempts: {e}")
                    raise
        return None, 0, False

    async def _check_job_status(self, multi_job_id, master_url):
        """Check if job is ready on the master."""
        try:
            try:
                session = await get_client_session()
                url = f"{master_url}/distributed/job_status?multi_job_id={multi_job_id}"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('ready', False)
                    return False
            except (aiohttp.ClientOSError, aiohttp.ServerDisconnectedError, ConnectionResetError, OSError) as e:
                if _is_win_conn_reset(e):
                    return False
                raise
        except Exception as e:
            debug_log(f"Job status check failed: {e}")
            return False

    def _determine_processing_mode(self, batch_size: int, num_workers: int, dynamic_threshold: int) -> str:
        """Prefer static (tile-based) when any workers are available; else single_gpu."""
        if num_workers == 0:
            return "single_gpu"
        return "static"

# Ensure initialization before registering routes
ensure_tile_jobs_initialized()

# ------------------------ Video wrapper (chunks + crossfade) ------------------------
# Wrapper for per-frame video batches: optional preroll, slicing into 4n+1 chunks,
# calling UltimateSDUpscaleDistributed on each chunk and crossfading at the junctions.

def _vd_log(msg: str) -> None:
    print(f"[VideoUpscaleWrapper] {msg}", flush=True)

def _ensure_4n1(x: int) -> int:
    """Clamp to >=1 and convert to the form 4n+1."""
    x = max(1, int(x))
    r = (x - 1) % 4
    return x if r == 0 else x + (4 - r)

def _ensure_mult4(x: int) -> int:
    """Clamp to >=0 and convert to a multiple of 4 (0, 4, 8, ...)"""
    x = max(0, int(x))
    return x - (x % 4)

def _plan_chunks(B: int, max_len: int, min_len: int, crossfade: int) -> List[Tuple[int, int, int]]:
    """
    Return a list of windows (start, end, head_overlap). The crossfade is placed
    at the beginning of each chunk (except the first). Window lengths are adjusted
    to the 4n+1 format.
    """
    out: List[Tuple[int, int, int]] = []
    i = 0
    pos = 0
    while pos < B:
        first = (i == 0)
        core = min(max_len, B - pos)
        head = 0 if first else min(crossfade, pos)

        s = pos - head
        e = pos + core
        if e > B:
            e = B
            core = e - pos

        total = e - s
        last = (e >= B)

        # The last chunk should not be shorter than min_len — extend it to the left
        if last and total < min_len:
            need = min_len - total
            grow_left = min(need, s)
            s -= grow_left
            head += grow_left
            total += grow_left

        # Adjust the window length to 4n+1 (first extend the head, then the tail)
        rem = (total - 1) % 4
        if rem != 0:
            grow_need = 4 - rem
            if not first and grow_need > 0:
                gl = min(grow_need, s)
                if gl > 0:
                    s -= gl
                    head += gl
                    total += gl
                    grow_need -= gl
            if grow_need > 0:
                gr = min(grow_need, B - e)
                if gr > 0:
                    e += gr
                    total += gr
                    grow_need -= gr
            if grow_need > 0:
                shrink = rem
                if total - shrink >= (min_len if last else 1):
                    e -= shrink
                    total -= shrink
                else:
                    sh2 = min(shrink, head)
                    s += sh2
                    head -= sh2
                    total -= sh2

        out.append((s, e, head))
        pos = e
        i += 1
        if i > 10000:
            raise RuntimeError("Too many chunks (safety break)")
    return out

def _linspace(start: float, end: float, steps: int, device, dtype) -> torch.Tensor:
    return torch.linspace(start, end, steps=steps, device=device, dtype=dtype)

def _crossfade(prev_tail: torch.Tensor, curr_head: torch.Tensor) -> torch.Tensor:
    """
    Linear crossfade along the temporal axis. Input tensor shape is [M,H,W,C].
    The output is computed as:
        out[k] = (1 - alpha_k) * prev[k] + alpha_k * curr[k], alpha_k ∈ (0..1].
    """
    assert prev_tail.shape == curr_head.shape, "Crossfade shapes must match"
    M = prev_tail.shape[0]
    if M == 0:
        return prev_tail
    a = _linspace(1.0 / M, 1.0, M, device=prev_tail.device, dtype=prev_tail.dtype).view(M, 1, 1, 1)
    return prev_tail * (1.0 - a) + curr_head * a

def _empty_cache():
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()

# ------------------------ RIFLEx (built-in minimal) ------------------------
def _rope_riflex(pos: torch.Tensor, dim: int, theta: float, L_test: int, k: int, scale: float) -> torch.Tensor:
    """
    Generate 2×2 RoPE matrices with the RIFLEx modification for the axis with index ``k-1``.

    Parameters:
        pos (torch.Tensor): positions as ``[B, N]`` or ``[N]``.
        dim (int): size of the axis (must be even).
        theta (float): base value.
        L_test (int): sequence length.
        k (int): 1-based frequency index.
        scale (float): scaling factor.

    Returns:
        torch.Tensor: tensor of shape ``[..., dim/2, 2, 2]``.
    """
    assert dim % 2 == 0
    if pos.ndim == 1:
        pos = pos.unsqueeze(0)  # [1,N]
    device = pos.device
    # Determine the numeric dtype based on the input positions.  This allows
    # RoPE matrices to be generated in either float16 or float32 depending on
    # how ComfyUI is configured.  If the dtype is not a floating type
    # (unlikely), fall back to the current default dtype.
    if pos.dtype in (torch.float16, torch.float32):
        dtype = pos.dtype
    else:
        dtype = torch.get_default_dtype()

    scale_vec = torch.linspace(0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64, device=device)
    omega = 1.0 / (theta ** scale_vec)  # [dim/2]

    if k and L_test:
        kk = int(k) - 1
        if 0 <= kk < omega.shape[0]:
            omega = omega.clone()
            omega[kk] = float(scale) * 2.0 * torch.pi / float(L_test)

    # Keep the intermediate computations in the chosen dtype.  Casting the
    # positions and frequency vector ensures that subsequent trigonometric
    # operations do not upcast back to float32 unnecessarily.
    t = pos.to(dtype)[:, :, None] * omega.to(dtype).view(1, 1, -1)  # [B,N,dim/2]
    cos_t = torch.cos(t)
    sin_t = torch.sin(t)
    out = torch.stack(
        (
            torch.stack((cos_t, -sin_t), dim=-1),
            torch.stack((sin_t,  cos_t), dim=-1),
        ),
        dim=-2,
    )  # [B,N,dim/2,2,2]
    return out

class _EmbedND_RIFLEx(nn.Module):
    def __init__(self, dim: int, theta: float, axes_dim: List[int], num_frames: int, k: int, scale: float):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim
        self.num_frames = int(num_frames)
        self.k = int(k)
        self.scale = float(scale)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # ids: [B,N,n_axes]
        n_axes = ids.shape[-1]
        parts = []
        for i in range(n_axes):
            dim_i = int(self.axes_dim[i])
            k_i = self.k if i == 0 else 0  # RIFLEx applies only to the temporal axis
            parts.append(_rope_riflex(ids[..., i], dim_i, self.theta, self.num_frames, k_i, self.scale))
        emb = torch.cat(parts, dim=-3)  # concatenate along the "d" axis
        return emb.unsqueeze(1)

def _riflex_prepare_model(model, *, L_test: int, k: int, scale: float):
    """
    If WAN/Hunyuan is recognized, return a clone of the model with the embedder replaced.
    Otherwise, return the original model and log a no-op.
    """
    try:
        dm = model.model.diffusion_model  # Comfy MODEL -> .model.diffusion_model
    except Exception:
        # Log message remains unchanged to preserve code behavior
        _vd_log("RIFLEx: no-op (нет .model.diffusion_model)")
        return model

    # WAN path
    if hasattr(dm, "rope_embedder") and hasattr(dm, "dim") and hasattr(dm, "num_heads"):
        try:
            d = int(dm.dim) // int(dm.num_heads)
            axes_dim = [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]
            rope_embedder = _EmbedND_RIFLEx(d, 10000.0, axes_dim, L_test, k, scale)
            model_clone = model.clone()
            model_clone.add_object_patch("diffusion_model.rope_embedder", rope_embedder)
            _vd_log(f"RIFLEx: WAN k={k}, L={L_test}, scale={scale}, d={d}")
            return model_clone
        except Exception as ex:
            # Log message remains unchanged to preserve code behavior
            _vd_log(f"RIFLEx: WAN patch failed → no-op ({ex})")
            return model

    # Hunyuan path
    if hasattr(dm, "pe_embedder") and hasattr(dm, "params"):
        params = dm.params
        for attr in ("hidden_size", "num_heads", "theta", "axes_dim"):
            if not hasattr(params, attr):
                break
        else:
            try:
                d = int(params.hidden_size) // int(params.num_heads)
                rope_embedder = _EmbedND_RIFLEx(d, float(params.theta), list(params.axes_dim), L_test, k, scale)
                model_clone = model.clone()
                model_clone.add_object_patch("diffusion_model.pe_embedder", rope_embedder)
                _vd_log(f"RIFLEx: Hunyuan k={k}, L={L_test}, scale={scale}, d={d}")
                return model_clone
            except Exception as ex:
                # Log message remains unchanged to preserve code behavior
                _vd_log(f"RIFLEx: Hunyuan patch failed → no-op ({ex})")
                return model

    # Log message remains unchanged to preserve code behavior
    _vd_log(f"RIFLEx: no-op (embedder not found)")
    return model

# ===== Distributed Final Frames Helpers (module-level, reusable) =====

def finals_b64_size(n: int) -> int:
    # base64 expands ~4/3; keep integer math
    return ((int(n) + 2) // 3) * 4


def iter_finals_payload_chunks(frames_bytes, max_batch: int, max_payload: int):
    # approximate JSON overheads
    overhead = 256
    per_item_overhead = 64
    batch, size = [], 0
    for b in frames_bytes:
        enc = finals_b64_size(len(b)) + per_item_overhead
        if batch and (len(batch) >= int(max_batch) or size + enc + overhead > int(max_payload)):
            yield batch
            batch, size = [], 0
        # if a single item is larger than the limit, still send it alone
        if not batch and enc + overhead > int(max_payload):
            yield [b]
            continue
        batch.append(b)
        size += enc
    if batch:
        yield batch


def dist_send_finals_chunked(multi_job_id: str, frames_bytes, enabled_workers):
    """
    Chunk finals by MAX_BATCH / MAX_PAYLOAD_SIZE and register them on the master.
    Critically: advertise expected_total and seal the last chunk.
    """
    total = len(frames_bytes)
    sent = 0
    # 1MB headroom on request size
    eff_payload_limit = max(1 * 1024 * 1024, int(MAX_PAYLOAD_SIZE) - (1024 * 1024))
    chunk_idx = 0

    for chunk in iter_finals_payload_chunks(frames_bytes, int(MAX_BATCH), eff_payload_limit):
        chunk_idx += 1
        is_last = (sent + len(chunk) == total)
        run_async_in_server_loop(
            register_final_frames(
                multi_job_id=multi_job_id,
                frames=chunk,
                enabled_workers=enabled_workers,
                expected_total=total,
                append=True,
                seal=is_last,  # seal on the last batch
            ),
            timeout=10.0
        )
        sent += len(chunk)
    _vd_log(f"Master registered finals: {sent}/{total}")




async def dist_async_wait_for_final_frames(multi_job_id: str,
                                           master_url: str,
                                           worker_id: str,
                                           dtype,
                                           device):
    """
    Asynchronously poll the master for final frames and assemble them.
    Tracks inbound batch size and approximate payload against MAX_BATCH and MAX_PAYLOAD_SIZE.
    Decodes JPEGs to uint8 and converts to the requested dtype only when needed.
    """
    final_frames = []
    start_time = time.time()
    batch_idx = 0
    total_received = 0
    total_decoded = 0

    while True:
        # Non-fatal heartbeat
        try:
            await _send_heartbeat_to_master(multi_job_id, master_url, worker_id)
        except Exception as _hb_err:
            _vd_log(f"Heartbeat error (ignored): {type(_hb_err).__name__}: {_hb_err}")

        try:
            session = await get_client_session()
            url = f"{master_url}/distributed/request_finals"
            payload = {'multi_job_id': multi_job_id, 'worker_id': str(worker_id)}
            async with session.post(url, json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    frames_b64 = data.get('frames', [])
                    is_last = bool(data.get('is_last', False))

                    # Inbound limits tracking (decode anyway even if exceeded)
                    try:
                        approx_payload = 256 + 64 * len(frames_b64) + sum(len(s or "") for s in frames_b64)
                        eff_payload_limit = max(1 * 1024 * 1024, int(MAX_PAYLOAD_SIZE) - (1024 * 1024))
                        if len(frames_b64) > int(MAX_BATCH) or approx_payload > eff_payload_limit:
                            _vd_log(
                                f"Notice: inbound finals batch #{batch_idx}: "
                                f"count={len(frames_b64)} (limit {int(MAX_BATCH)}), "
                                f"payload≈{approx_payload}B (limit {eff_payload_limit}B)."
                            )
                    except Exception:
                        pass

                    _vd_log(f"Batch {batch_idx}: received {len(frames_b64)} item(s), is_last={is_last}")
                    total_received += len(frames_b64)

                    if frames_b64:
                        # Parallel JPEG decoding using shared executor; fallback to sequential
                        try:
                            executor = get_jpeg_executor()
                            arrs = list(executor.map(_decode_base64_to_np, frames_b64))
                        except Exception:
                            arrs = [_decode_base64_to_np(fb64) for fb64 in frames_b64]

                        # Convert arrays to torch tensors on target device/dtype
                        for j, arr in enumerate(arrs):
                            try:
                                # Start from uint8 on device to minimize host RAM usage
                                t = torch.from_numpy(arr).to(device=device, dtype=torch.uint8)
                                # If requested dtype is floating, convert and normalize in-place
                                if torch.is_floating_point(torch.empty((), dtype=dtype)):
                                    t = t.to(dtype=dtype)
                                    # Scale only for float dtypes; keep integers as-is
                                    t = t.div_(255.0)
                                final_frames.append(t)
                                total_decoded += 1
                            except Exception as e:
                                _vd_log(f"Decode failed at batch={batch_idx}, idx={j}: {type(e).__name__}: {e}")

                        # Help GC release large strings promptly
                        frames_b64 = None
                        arrs = None

                    if is_last:
                        _vd_log(f"Completed: batches={batch_idx+1}, received={total_received}, decoded={total_decoded}")
                        break

                elif resp.status == 404:
                    _vd_log("Master not ready (404), retrying…")
                    await asyncio.sleep(2.0)
                else:
                    _vd_log(f"Unexpected status={resp.status}, retrying…")
                    await asyncio.sleep(2.0)

            batch_idx += 1

        except Exception as req_err:
            _vd_log(f"Request error, retrying: {type(req_err).__name__}: {req_err}")
            await asyncio.sleep(1.0)

        if time.time() - start_time > 900:
            raise RuntimeError("Timeout waiting for final frames from master")

    if final_frames:
        out = torch.stack(final_frames, dim=0).contiguous()
    else:
        out = torch.empty((0,), dtype=dtype, device=device)

    return out



def wait_for_final_frames(multi_job_id: str,
                          master_url: str,
                          worker_id: str,
                          dtype,
                          device):
    """
    Sync wrapper around dist_async_wait_for_final_frames for ComfyUI's sync graph execution.
    """
    return run_async_in_server_loop(
        dist_async_wait_for_final_frames(multi_job_id, master_url, worker_id, dtype, device),
        timeout=960.0
    )


# ------------------------ Main wrapper class ------------------------
class UltimateSDVideoUpscaleDistributed:
    """
    Video wrapper:
      • Optional preroll/postroll.
      • Optional batching into 4n+1-safe chunks with crossfade.
      • RIFLEx (WAN/Hunyuan) optional.
      • Internally calls UltimateSDUpscaleDistributed, now with PIL frames.
      • Master can broadcast assembled finals to workers.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "upscaled_image": ("IMAGE",),
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),

                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),

                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler":    (comfy.samplers.KSampler.SCHEDULERS,),

                "denoise": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "tile_width": ("INT", {"default": 896, "min": 64, "max": 2048, "step": 8}),
                "tile_height": ("INT", {"default": 896, "min": 64, "max": 2048, "step": 8}),
                "padding": ("INT", {"default": 128, "min": 0, "max": 256, "step": 8}),
                "mask_blur": ("INT", {"default": 64, "min": 0, "max": 256}),
                "mask_expand": ("INT", {"default": 0, "min": -256, "max": 256}),
                "force_uniform_tiles": ("BOOLEAN", {"default": True}),
                "tile_auto_size": ("BOOLEAN", {"default": True}),
                "tiled_decode": ("BOOLEAN", {"default": False}),

                "preroll_frames": ("INT", {"default": 8, "min": 0, "max": 100000, "step": 4, "mod": 0}),
                "postroll_frames": ("INT", {"default": 8, "min": 0, "max": 100000, "step": 4, "mod": 0}),

                "split_into_chunks": ("BOOLEAN", {"default": True}),
                "chunk_max_frames": ("INT", {"default": 141, "min": 1, "max": 100000, "step": 4, "mod": 1}),
                "chunk_min_frames": ("INT", {"default": 25, "min": 1, "max": 100000, "step": 4, "mod": 1}),
                "chunk_crossfade_frames": ("INT", {"default": 25, "min": 1, "max": 1000, "step": 4, "mod": 1}),

                "enable_riflex": ("BOOLEAN", {"default": True}),
                "riflex_k": ("INT", {"default": 6, "min": 1, "max": 256}),
                "riflex_frames": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "riflex_scale": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 1.0, "step": 0.01}),

                "auto_morph_output": ("BOOLEAN", {"default": False}),
                "auto_morph_blur": ("INT", {"default": 50, "min": 0, "max": 200, "step": 1}),

                "highpass_blend": ("BOOLEAN", {"default": True}),
                "hp_blur_size": ("INT", {"default": 33, "min": 0, "max": 1000, "step": 1}),
                "hp_opacity": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "hp_device": (["auto", "cuda", "cpu"],),
                "hp_store_result_on": (["cpu", "same_as_input"],),
                "hp_contrast": ("FLOAT", {"default": 0.96, "min": 0.0, "max": 1.0, "step": 0.01}),

                "preserve_gradients_blend": ("BOOLEAN", {"default": True}),
                "pg_highpass_radius": ("INT", {"default": 3, "min": 0, "max": 1000, "step": 1}),
                "pg_downscale_by": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01}),
                "pg_strength": ("FLOAT", {"default": 15.0, "min": 0.0, "max": 32.0, "step": 0.1}),
                "pg_expand": ("INT", {"default": 15, "min": 0, "max": 128, "step": 1}),
                "pg_mask_blur": ("INT", {"default": 40, "min": 0, "max": 400, "step": 1}),
                "pg_clamp_blacks": ("FLOAT", {"default": 0.00, "min": 0.0, "max": 1.0, "step": 0.01}),
                "pg_clamp_whites": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "pg_temporal_deflicker": ("BOOLEAN", {"default": True}),
                "pg_device": (["auto", "cuda", "cpu"],),

                "upscale_with_model": ("BOOLEAN", {"default": True}),
                "upscale_with_model_batch": ("INT", {"default": 2, "min": 1, "max": 32, "step": 1}),
                "upscale_with_model_run_distributed": ("BOOLEAN", {"default": False}),
                "upscale_with_model_force_tiled": ("BOOLEAN", {"default": True}),

                "pre_sharpen": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 10.0, "step": 0.01}),
                "post_sharpen": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 10.0, "step": 0.01}),
                "simple_pre_upscaler_multiplier": ("FLOAT", {"default": 1, "min": 0.01, "max": 10.0, "step": 0.01}),
                "send_finals_to_workers": ("BOOLEAN", {"default": False}),

            },
            "hidden": {
                "multi_job_id": ("STRING", {"default": ""}),
                "is_worker": ("BOOLEAN", {"default": False}),
                "master_url": ("STRING", {"default": ""}),
                "enabled_worker_ids": ("STRING", {"default": "[]"}),
                "worker_id": ("STRING", {"default": ""}),
                "tile_indices": ("STRING", {"default": ""}),
                "dynamic_threshold": ("INT", {"default": 8, "min": 1, "max": 64}),
            },
            "optional": {
                "upscale_model": ("UPSCALE_MODEL",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "image/upscaling"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def __init__(self):
        # Base node from the same file; it accepts PIL
        self.core = UltimateSDUpscaleDistributed()
        self._model_upscaler = None

    # ---------- small PIL helpers (kept inside class, no new imports) ----------

    @staticmethod
    def _tensor_bhwc_to_pil_list(t: torch.Tensor) -> list:
        """Convert [B,H,W,C] float tensor in 0..1 to list of RGB PIL.Image."""
        imgs = []
        for i in range(int(t.shape[0])):
            im = tensor_to_pil(t[i:i+1], 0).convert("RGB")
            imgs.append(im.copy())
        return imgs

    @staticmethod
    def _pil_list_to_bhwc_tensor(imgs: list, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Convert list of PIL.Image back to [B,H,W,C] 0..1 strictly in the target dtype/device (no float32 staging)."""
        if len(imgs) == 0:
            return torch.empty((0,), dtype=dtype, device=device)

        # Infer shape from the first frame
        w0, h0 = imgs[0].size
        B = len(imgs)
        # Pre-allocate final tensor on target device with target dtype
        out = torch.empty((B, h0, w0, 3), dtype=dtype, device=device)

        # Convert per-frame without float32 promotion
        for i, im in enumerate(imgs):
            a = np.asarray(im, dtype=np.uint8)  # HWC, uint8
            t_cpu = torch.from_numpy(a)  # CPU uint8
            # Convert to target dtype on CPU first to keep RAM usage low and avoid float32
            t_cpu = t_cpu.to(dtype=dtype)  # CPU -> target dtype
            # Normalize in-place if floating type
            if torch.is_floating_point(t_cpu):
                t_cpu = t_cpu / 255.0
            # Move to target device (may be CPU or CUDA)
            t = t_cpu.to(device=device, dtype=dtype, non_blocking=False)
            # Write into the preallocated output tensor
            out[i].copy_(t)
            # Drop temporaries early
            del a, t_cpu, t

        return out

    @staticmethod
    def _crossfade_pil(prev_tail: list, curr_head: list) -> list:
        """Linear crossfade between two equal-length PIL lists using PIL.Image.blend."""
        m = len(prev_tail)
        if m == 0:
            return prev_tail
        blended = []
        for k in range(m):
            alpha = float(k + 1) / float(m)
            blended.append(Image.blend(prev_tail[k], curr_head[k], alpha))
        return blended

    def _core_run(self, upscaled_image, model, positive, negative, vae,
                  seed, steps, cfg, sampler_name, scheduler, denoise,
                  tile_width, tile_height, padding, mask_blur, mask_expand,
                  force_uniform_tiles, tiled_decode,
                  multi_job_id, is_worker, master_url,
                  enabled_worker_ids, worker_id, tile_indices,
                  dynamic_threshold, auto_morph_output, auto_morph_blur,
                  highpass_blend, hp_blur_size, hp_opacity, hp_device, hp_store_result_on, hp_contrast,
                  preserve_gradients_blend, pg_highpass_radius, pg_downscale_by, pg_strength, pg_expand, pg_mask_blur,
                  pg_clamp_blacks, pg_clamp_whites, pg_temporal_deflicker, pg_device, post_sharpen, pre_sharpen):
        # Call base node; it accepts PIL and returns list of PIL frames
        return self.core.run(
            upscaled_image, model, positive, negative, vae,
            seed, steps, cfg, sampler_name, scheduler, denoise,
            tile_width, tile_height, padding, mask_blur, mask_expand,
            force_uniform_tiles, tiled_decode,
            multi_job_id, is_worker, master_url,
            enabled_worker_ids, worker_id, tile_indices,
            dynamic_threshold, auto_morph_output, auto_morph_blur,
            highpass_blend, hp_blur_size, hp_opacity, hp_device, hp_store_result_on, hp_contrast,
            preserve_gradients_blend, pg_highpass_radius, pg_downscale_by, pg_strength, pg_expand, pg_mask_blur,
            pg_clamp_blacks, pg_clamp_whites, pg_temporal_deflicker, pg_device, post_sharpen, pre_sharpen
        )

    def run(self, upscaled_image, model, positive, negative, vae, seed, steps, cfg,
            sampler_name, scheduler, denoise, tile_width, tile_height, padding,
            mask_blur, mask_expand, force_uniform_tiles, tiled_decode, tile_auto_size,
            preroll_frames, split_into_chunks,
            enable_riflex, riflex_k, riflex_frames, riflex_scale,
            chunk_max_frames, chunk_min_frames, chunk_crossfade_frames,
            multi_job_id="", is_worker=False, master_url="", enabled_worker_ids="[]",
            worker_id="", tile_indices="", dynamic_threshold=8, auto_morph_output=False, auto_morph_blur=8,
            highpass_blend=False, hp_blur_size=100, hp_opacity=0.85, hp_device="auto", hp_store_result_on="cpu", hp_contrast=1.0,
            preserve_gradients_blend=False, pg_highpass_radius=15, pg_downscale_by=0.25, pg_strength=10.0,
            pg_expand=5, pg_mask_blur=25, pg_clamp_blacks=0.05, pg_clamp_whites=0.9,
            pg_temporal_deflicker=False, pg_device="auto",
            upscale_with_model=False, upscale_model="", upscale_with_model_batch=1, postroll_frames=0,
            post_sharpen=0.0, upscale_with_model_force_tiled=False, simple_pre_upscaler_multiplier=1.0,
            send_finals_to_workers=False, upscale_with_model_run_distributed=True, pre_sharpen=0.0):

        # Preserve original dtype/device for the final IMAGE tensor conversion
        orig_dtype = upscaled_image.dtype
        orig_device = upscaled_image.device

        # Input sanity: IMAGE must be [B,H,W,C] tensor → convert to list[PIL.Image] ASAP
        images: torch.Tensor = upscaled_image
        assert isinstance(images, torch.Tensor) and images.ndim == 4, "IMAGE must be [B,H,W,C]"

        # Convert early to PIL and free tensor
        frames_pil = self._tensor_bhwc_to_pil_list(images)
        del images, upscaled_image
        _empty_cache()

        # Broadcast flag
        broadcast = bool(send_finals_to_workers)

        # ----------------- (1) Optional pre-upscale WITH MODEL (now PIL in/out) -----------------
        if bool(upscale_with_model):
            can_dist = bool(upscale_with_model_run_distributed) and (
                (not bool(is_worker)) or (bool(multi_job_id) and isinstance(enabled_worker_ids, str) and enabled_worker_ids.strip() not in ("", "[]"))
            )

            if can_dist:
                # UpscaleWithModelBatchedTiled now accepts PIL directly; it still RETURNS an IMAGE tensor.
                if self._model_upscaler is None:
                    self._model_upscaler = UpscaleWithModelBatchedTiled()

                model_job_id = f"{multi_job_id}__umodel" if multi_job_id else ""
                model_max_tile = int(max(int(tile_width), int(tile_height)))
                model_overlap  = int(padding)

                result = self._model_upscaler.apply(
                    upscale_model=upscale_model,
                    images=frames_pil,                      # pass PIL list directly
                    per_batch=int(upscale_with_model_batch),
                    force_tiled=bool(upscale_with_model_force_tiled),
                    max_tile_size=model_max_tile,
                    tile_overlap=model_overlap,
                    run_distributed=True,
                    multi_job_id=model_job_id,
                    is_worker=bool(is_worker),
                    master_url=str(master_url),
                    enabled_worker_ids=str(enabled_worker_ids),
                    worker_id=str(worker_id),
                )[0]

                # Normalize to PIL for downstream PIL-only pipeline
                if isinstance(result, torch.Tensor):
                    frames_pil = self._tensor_bhwc_to_pil_list(result)
                    del result
                    _empty_cache()
                elif isinstance(result, list) and (len(result) == 0 or isinstance(result[0], Image.Image)):
                    frames_pil = result
                else:
                    raise RuntimeError("Unexpected result type from distributed model upscaler.")
                _vd_log(f"Distributed pre-upscale via UpscaleWithModelBatchedTiled: B={len(frames_pil)}")
            else:
                # Local PIL→PIL path via updated upscale_with_model_batched
                frames_pil = upscale_with_model_batched(
                    upscale_model=upscale_model,
                    images=frames_pil,                         # PIL list in
                    per_batch=int(upscale_with_model_batch),
                    clear_cuda_each_batch=True,
                    prefer_tiled=upscale_with_model_force_tiled,
                    return_pil=True                            # ensure PIL out
                )
                _vd_log("Local pre-upscale via upscale_with_model_batched (PIL).")
            _empty_cache()

        # ----------------- (2) Optional simple pre-upscaler (PIL path) -----------------
        if simple_pre_upscaler_multiplier != 1.0 and len(frames_pil) > 0:
            from PIL import Image as _PILImage
            mul = float(simple_pre_upscaler_multiplier)
            w0, h0 = frames_pil[0].size

            # Round to multiples of 8 (like original area path)
            def _round8(x: int) -> int:
                return max(8, ((int(round(x)) + 7) // 8) * 8)

            new_w = _round8(w0 * mul)
            new_h = _round8(h0 * mul)

            # BOX ≈ area resampling
            for i in range(len(frames_pil)):
                frames_pil[i] = frames_pil[i].resize((new_w, new_h), resample=_PILImage.Resampling.BOX)
            _vd_log(f"Simple pre-upscale (area/BOX) ×{mul} → {new_w}x{new_h}")
            _empty_cache()

        # ----------------- (3) Already in PIL → compute sizes -----------------
        if len(frames_pil) == 0:
            empty = torch.empty((0, 1, 1, 3), dtype=orig_dtype, device=orig_device)
            return (empty,)

        B = len(frames_pil)
        width, height = frames_pil[0].size

        # Optional auto tile sizing uses width/height above
        if bool(tile_auto_size):
            import math
            _tw_lim = int(tile_width)
            _th_lim = int(tile_height)

            def _up16(x: int) -> int:
                return ((int(x) + 15) // 16) * 16

            def _auto_tile_1d(image_side: int, limit: int) -> int:
                limit = max(16, int(limit))
                tiles = max(1, math.ceil(image_side / limit))
                while True:
                    cand = math.ceil(image_side / tiles)
                    cand16 = _up16(cand)
                    if cand16 <= limit or tiles >= image_side:
                        return min(image_side, max(16, cand16))
                    tiles += 1

            tile_width  = _auto_tile_1d(width, _tw_lim)
            tile_height = _auto_tile_1d(height, _th_lim)

            try:
                _workers = json.loads(enabled_worker_ids) if isinstance(enabled_worker_ids, str) and enabled_worker_ids.strip() else []
            except Exception:
                _workers = []
            participants = 1 + (len(_workers) if isinstance(_workers, (list, tuple)) else 0)

            if participants > 1:
                def _grid_counts(tw: int, th: int) -> tuple[int, int, int]:
                    gx = max(1, math.ceil(width  / max(1, tw)))
                    gy = max(1, math.ceil(height / max(1, th)))
                    return gx, gy, gx * gy

                gx0, gy0, T0 = _grid_counts(tile_width, tile_height)
                if T0 % participants != 0:
                    best = None
                    max_step = 3
                    for dx in range(-max_step, max_step + 1):
                        for dy in range(-max_step, max_step + 1):
                            if dx == 0 and dy == 0:
                                continue
                            tx = max(1, gx0 + dx)
                            ty = max(1, gy0 + dy)
                            tw_c = _up16(math.ceil(width  / tx)) if tx < gx0 else _up16(math.floor(width  / tx))
                            th_c = _up16(math.ceil(height / ty)) if ty < gy0 else _up16(math.floor(height / ty))
                            tw_c = min(max(16, tw_c), _tw_lim)
                            th_c = min(max(16, th_c), _th_lim)
                            gx, gy, T = _grid_counts(tw_c, th_c)
                            if T % participants != 0:
                                continue
                            delta_tiles = abs(T - T0)
                            prefer_penalty = 0 if T >= T0 else 0.5
                            delta_size = abs(tw_c - tile_width) + abs(th_c - tile_height)
                            score = (delta_tiles, prefer_penalty, delta_size)
                            if best is None or score < best[0]:
                                best = (score, T, tw_c, th_c, gx, gy)
                    if best is not None:
                        _, Tn, tw_n, th_n, gx_n, gy_n = best
                        if tw_n != tile_width or th_n != tile_height:
                            _vd_log(f"[Distributed] Balanced tiles: {gx0}x{gy0}={T0} → {gx_n}x{gy_n}={Tn}. Tile {tile_width}x{tile_height} → {tw_n}x{th_n}")
                            tile_width, tile_height = int(tw_n), int(th_n)

        # ----------------- (4) Preroll / Postroll на PIL -----------------
        pre_n  = _ensure_mult4(preroll_frames)
        post_n = _ensure_mult4(postroll_frames)

        if pre_n > 0:
            head = [frames_pil[0]] * pre_n
            frames_pil = head + frames_pil
            _vd_log(f"Preroll: +{pre_n} → {len(frames_pil)}")

        if post_n > 0:
            tail = [frames_pil[-1]] * post_n
            frames_pil = frames_pil + tail
            _vd_log(f"Postroll: +{post_n} → {len(frames_pil)}")

        B = len(frames_pil)

        base_job_id = multi_job_id or ""
        try:
            enabled_workers_list = json.loads(enabled_worker_ids) if enabled_worker_ids else []
        except Exception:
            enabled_workers_list = []

        # ----------------- (5) No-chunk mode (unchanged, works on PIL) -----------------
        if not split_into_chunks:
            pad_left = 0
            target = _ensure_4n1(B)
            pad_left = target - B
            if pad_left > 0:
                pad = [frames_pil[0]] * pad_left
                proc_list = pad + frames_pil
                _vd_log(f"No-chunk: pad_left {pad_left} → {len(proc_list)}")
            else:
                proc_list = frames_pil

            full_job_id = f"{base_job_id}__full" if base_job_id else ""

            model_to_use = model
            if enable_riflex:
                L_test = int(riflex_frames) if int(riflex_frames) > 0 else int(len(proc_list))
                model_to_use = _riflex_prepare_model(model, L_test=L_test, k=int(riflex_k), scale=float(riflex_scale))

            (out_pil,) = self._core_run(
                proc_list, model_to_use, positive, negative, vae,
                seed, steps, cfg, sampler_name, scheduler, denoise,
                tile_width, tile_height, padding, mask_blur, mask_expand,
                force_uniform_tiles, tiled_decode,
                full_job_id, is_worker, master_url,
                enabled_worker_ids, worker_id, tile_indices,
                dynamic_threshold, auto_morph_output, auto_morph_blur,
                highpass_blend, hp_blur_size, hp_opacity, hp_device, hp_store_result_on, hp_contrast,
                preserve_gradients_blend, pg_highpass_radius, pg_downscale_by, pg_strength, pg_expand, pg_mask_blur,
                pg_clamp_blacks, pg_clamp_whites, pg_temporal_deflicker, pg_device, post_sharpen, pre_sharpen
            )
            del proc_list
            _empty_cache()

            drop_left = pad_left + pre_n
            if drop_left > 0:
                out_pil = out_pil[drop_left:]
            if post_n > 0 and len(out_pil) >= post_n:
                out_pil = out_pil[:-post_n]

            if bool(is_worker):
                if broadcast:
                    final_out_tensor = wait_for_final_frames(multi_job_id, master_url, worker_id, orig_dtype, orig_device)
                    _vd_log("Done (no-chunk worker, waited finals).")
                    _empty_cache()
                    return (final_out_tensor,)
                else:
                    out_tensor = self._pil_list_to_bhwc_tensor(out_pil, orig_dtype, orig_device)
                    del out_pil
                    _empty_cache()
                    _vd_log("Done (no-chunk worker, local result).")
                    return (out_tensor,)

            if broadcast and len(enabled_workers_list) > 0:
                u8_list = [np.array(im, copy=False) for im in out_pil]
                try:
                    executor = get_jpeg_executor()
                    frames_bytes = list(executor.map(_encode_np_to_jpeg, u8_list))
                except Exception:
                    frames_bytes = [_encode_np_to_jpeg(a) for a in u8_list]
                try:
                    self._send_finals_chunked(multi_job_id, frames_bytes, enabled_workers_list)
                    _vd_log(f"Master registered {len(frames_bytes)} frames (no-chunk).")
                except Exception as e:
                    debug_log(f"Error registering final frames: {e}")
                finally:
                    del u8_list, frames_bytes
                    _empty_cache()

            out_tensor = self._pil_list_to_bhwc_tensor(out_pil, orig_dtype, orig_device)
            del out_pil
            _vd_log("Done (no-chunk).")
            _empty_cache()
            return (out_tensor,)

        # ----------------- (6) Chunk mode (unchanged logic; operates on PIL) -----------------
        chunk_max_frames = _ensure_4n1(chunk_max_frames)
        chunk_min_frames = _ensure_4n1(chunk_min_frames)
        chunk_crossfade_frames = _ensure_4n1(chunk_crossfade_frames)

        _vd_log(f"Input {B} frames (PIL), W×H={width}×{height}")
        _vd_log(f"Chunks: max={chunk_max_frames}, min={chunk_min_frames}, crossfade={chunk_crossfade_frames}")

        if B <= chunk_max_frames:
            pad_left = 0
            if B < chunk_min_frames:
                pad_left = (chunk_min_frames - B)
            target = _ensure_4n1(B + pad_left)
            pad_left = target - B
            if pad_left > 0:
                pad = [frames_pil[0]] * pad_left
                proc_list = pad + frames_pil
                _vd_log(f"Single chunk: pad_left {pad_left} → {len(proc_list)}")
            else:
                proc_list = frames_pil

            chunk_job_id = f"{multi_job_id}__ch001" if multi_job_id else ""
            model_to_use = model
            if enable_riflex:
                L_test = int(riflex_frames) if int(riflex_frames) > 0 else int(len(proc_list))
                model_to_use = _riflex_prepare_model(model, L_test=L_test, k=int(riflex_k), scale=float(riflex_scale))

            (out_pil,) = self._core_run(
                proc_list, model_to_use, positive, negative, vae,
                seed, steps, cfg, sampler_name, scheduler, denoise,
                tile_width, tile_height, padding, mask_blur, mask_expand,
                force_uniform_tiles, tiled_decode,
                chunk_job_id, is_worker, master_url,
                enabled_worker_ids, worker_id, tile_indices,
                dynamic_threshold, auto_morph_output, auto_morph_blur,
                highpass_blend, hp_blur_size, hp_opacity, hp_device, hp_store_result_on, hp_contrast,
                preserve_gradients_blend, pg_highpass_radius, pg_downscale_by, pg_strength, pg_expand, pg_mask_blur,
                pg_clamp_blacks, pg_clamp_whites, pg_temporal_deflicker, pg_device, post_sharpen, pre_sharpen
            )
            del proc_list
            _empty_cache()

            drop_left = pad_left + pre_n
            if drop_left > 0:
                out_pil = out_pil[drop_left:]
            if post_n > 0 and len(out_pil) >= post_n:
                out_pil = out_pil[:-post_n]

            if bool(is_worker):
                if broadcast:
                    final_out_tensor = wait_for_final_frames(multi_job_id, master_url, worker_id, orig_dtype, orig_device)
                    _vd_log("Done (single-chunk worker, waited finals).")
                    _empty_cache()
                    return (final_out_tensor,)
                else:
                    out_tensor = self._pil_list_to_bhwc_tensor(out_pil, orig_dtype, orig_device)
                    del out_pil
                    _vd_log("Done (single-chunk worker, local).")
                    _empty_cache()
                    return (out_tensor,)

            if broadcast and len(enabled_workers_list) > 0:
                u8_list = [np.array(im, copy=False) for im in out_pil]
                try:
                    executor = get_jpeg_executor()
                    frames_bytes = list(executor.map(_encode_np_to_jpeg, u8_list))
                except Exception:
                    frames_bytes = [_encode_np_to_jpeg(a) for a in u8_list]
                try:
                    self._send_finals_chunked(multi_job_id, frames_bytes, enabled_workers_list)
                    _vd_log(f"Master registered {len(frames_bytes)} frames (single-chunk).")
                except Exception as e:
                    debug_log(f"Error registering final frames: {e}")
                finally:
                    del u8_list, frames_bytes
                    _empty_cache()

            out_tensor = self._pil_list_to_bhwc_tensor(out_pil, orig_dtype, orig_device)
            del out_pil
            _vd_log("Done (single-chunk).")
            _empty_cache()
            return (out_tensor,)

        plan = _plan_chunks(B, chunk_max_frames, chunk_min_frames, chunk_crossfade_frames)
        _vd_log(f"Planned {len(plan)} chunk(s).")
        for i, (s, e, head) in enumerate(plan, 1):
            _vd_log(f"  • #{i}: [{s}:{e}) len={e - s}, head={head}")

        # --- Cumulative progress bar for all chunks (master only) ---
        if not is_worker:
            try:
                from math import ceil
                # tiles per image at current resolution
                tiles_per_image = ceil(height / max(1, tile_height)) * ceil(width / max(1, tile_width))
                # frames processed by core across all chunks (heads included)
                total_frames = sum(e - s for s, e, _ in plan)
                total_tiles = int(tiles_per_image * total_frames)

                from comfy.utils import ProgressBar as _PB
                self.core._progress = _PB(total_tiles)
                self.core._progress_keepalive = True  # keep across chunks
            except Exception:
                self.core._progress = None
                self.core._progress_keepalive = False

        if bool(is_worker):
            if broadcast:
                for i, (s, e, head) in enumerate(plan, 1):
                    sub = frames_pil[s:e]
                    _vd_log(f"[Worker {i}/{len(plan)}] → core.run len={len(sub)}")
                    chunk_job_id = f"{(multi_job_id or '')}__ch{i:03d}"
                    model_to_use = model
                    if enable_riflex:
                        L_test = int(riflex_frames) if int(riflex_frames) > 0 else int(len(sub))
                        model_to_use = _riflex_prepare_model(model, L_test=L_test, k=int(riflex_k), scale=float(riflex_scale))
                    try:
                        _ = self._core_run(
                            sub, model_to_use, positive, negative, vae,
                            seed, steps, cfg, sampler_name, scheduler, denoise,
                            tile_width, tile_height, padding, mask_blur, mask_expand,
                            force_uniform_tiles, tiled_decode,
                            chunk_job_id, is_worker, master_url,
                            enabled_worker_ids, worker_id, tile_indices,
                            dynamic_threshold, auto_morph_output, auto_morph_blur,
                            highpass_blend, hp_blur_size, hp_opacity, hp_device, hp_store_result_on, hp_contrast,
                            preserve_gradients_blend, pg_highpass_radius, pg_downscale_by, pg_strength, pg_expand, pg_mask_blur,
                            pg_clamp_blacks, pg_clamp_whites, pg_temporal_deflicker, pg_device, post_sharpen, pre_sharpen
                        )
                    finally:
                        del sub
                        _empty_cache()
                final_out_tensor = wait_for_final_frames(multi_job_id, master_url, worker_id, orig_dtype, orig_device)
                _vd_log(f"Done (multi-chunk worker, finals). Output: {final_out_tensor.shape[0]}")
                _empty_cache()
                return (final_out_tensor,)
            else:
                assembled = None
                for i, (s, e, head) in enumerate(plan, 1):
                    sub = frames_pil[s:e]
                    _vd_log(f"[Worker {i}/{len(plan)}] → core.run len={len(sub)}")
                    chunk_job_id = f"{(multi_job_id or '')}__ch{i:03d}"
                    model_to_use = model
                    if enable_riflex:
                        L_test = int(riflex_frames) if int(riflex_frames) > 0 else int(len(sub))
                        model_to_use = _riflex_prepare_model(model, L_test=L_test, k=int(riflex_k), scale=float(riflex_scale))
                    (out_chunk,) = self._core_run(
                        sub, model_to_use, positive, negative, vae,
                        seed, steps, cfg, sampler_name, scheduler, denoise,
                        tile_width, tile_height, padding, mask_blur, mask_expand,
                        force_uniform_tiles, tiled_decode,
                        chunk_job_id, is_worker, master_url,
                        enabled_worker_ids, worker_id, tile_indices,
                        dynamic_threshold, auto_morph_output, auto_morph_blur,
                        highpass_blend, hp_blur_size, hp_opacity, hp_device, hp_store_result_on, hp_contrast,
                        preserve_gradients_blend, pg_highpass_radius, pg_downscale_by, pg_strength, pg_expand, pg_mask_blur,
                        pg_clamp_blacks, pg_clamp_whites, pg_temporal_deflicker, pg_device, post_sharpen, pre_sharpen
                    )
                    if assembled is None:
                        assembled = out_chunk
                    else:
                        m = int(head)
                        if m > 0:
                            blended = self._crossfade_pil(assembled[-m:], out_chunk[:m])
                            assembled[-m:] = blended
                            assembled.extend(out_chunk[m:])
                            del blended
                        else:
                            assembled.extend(out_chunk)
                    del sub, out_chunk
                    _empty_cache()

                out_pil = assembled[pre_n:] if pre_n > 0 else assembled
                if post_n > 0 and len(out_pil) >= post_n:
                    out_pil = out_pil[:-post_n]
                out_tensor = self._pil_list_to_bhwc_tensor(out_pil, orig_dtype, orig_device)
                del out_pil, assembled
                _vd_log(f"Done (multi-chunk worker local). Output: {out_tensor.shape[0]}")
                _empty_cache()
                return (out_tensor,)

        stitched = None
        for i, (s, e, head) in enumerate(plan, 1):
            sub = frames_pil[s:e]
            _vd_log(f"[{i}/{len(plan)}] → core.run len={len(sub)}")
            chunk_job_id = f"{(multi_job_id or '')}__ch{i:03d}"
            model_to_use = model
            if enable_riflex:
                L_test = int(riflex_frames) if int(riflex_frames) > 0 else int(len(sub))
                model_to_use = _riflex_prepare_model(model, L_test=L_test, k=int(riflex_k), scale=float(riflex_scale))
            (out_chunk,) = self._core_run(
                sub, model_to_use, positive, negative, vae,
                seed, steps, cfg, sampler_name, scheduler, denoise,
                tile_width, tile_height, padding, mask_blur, mask_expand,
                force_uniform_tiles, tiled_decode,
                chunk_job_id, is_worker, master_url,
                enabled_worker_ids, worker_id, tile_indices,
                dynamic_threshold, auto_morph_output, auto_morph_blur,
                highpass_blend, hp_blur_size, hp_opacity, hp_device, hp_store_result_on, hp_contrast,
                preserve_gradients_blend, pg_highpass_radius, pg_downscale_by, pg_strength, pg_expand, pg_mask_blur,
                pg_clamp_blacks, pg_clamp_whites, pg_temporal_deflicker, pg_device, post_sharpen, pre_sharpen
            )
            if stitched is None:
                stitched = out_chunk
            else:
                m = int(head)
                if m > 0:
                    blended = self._crossfade_pil(stitched[-m:], out_chunk[:m])
                    stitched[-m:] = blended
                    stitched.extend(out_chunk[m:])
                    del blended
                else:
                    stitched.extend(out_chunk)
            del sub, out_chunk
            _empty_cache()

        out_pil = stitched[pre_n:] if pre_n > 0 else stitched
        if post_n > 0 and len(out_pil) >= post_n:
            out_pil = out_pil[:-post_n]
        del stitched
        _empty_cache()

        if broadcast and len(enabled_workers_list) > 0:
            u8_list = [np.array(im, copy=False) for im in out_pil]
            try:
                executor = get_jpeg_executor()
                frames_bytes = list(executor.map(_encode_np_to_jpeg, u8_list))
            except Exception:
                frames_bytes = [_encode_np_to_jpeg(a) for a in u8_list]
            try:
                self._send_finals_chunked(multi_job_id, frames_bytes, enabled_workers_list)
                _vd_log(f"Master registered {len(frames_bytes)} frames (multi-chunk).")
            except Exception as e:
                debug_log(f"Error registering final frames: {e}")
            finally:
                del u8_list, frames_bytes
                _empty_cache()

        out_tensor = self._pil_list_to_bhwc_tensor(out_pil, orig_dtype, orig_device)
        del out_pil
        _vd_log(f"Done. Output frames: {out_tensor.shape[0]}")
        _empty_cache()

        # --- Finalize single cumulative progress bar ---
        if not bool(is_worker):
            try:
                self.core._progress_keepalive = False
                # Ensure no stale handle remains for next runs
                self.core._progress = None
            except Exception:
                pass

        return (out_tensor,)


    # -------- Finals helpers: kept for back-compat and external calls --------
    @staticmethod
    def finals_b64_size(n):
        return ((int(n) + 2) // 3) * 4

    @staticmethod
    def iter_finals_payload_chunks(frames_bytes, max_batch, max_payload):
        overhead = 256
        per_item_overhead = 64
        batch, size = [], 0
        for b in frames_bytes:
            enc = UltimateSDVideoUpscaleDistributed.finals_b64_size(len(b)) + per_item_overhead
            if batch and (len(batch) >= int(max_batch) or size + enc + overhead > int(max_payload)):
                yield batch
                batch, size = [], 0
            if not batch and enc + overhead > int(max_payload):
                yield [b]
                continue
            batch.append(b)
            size += enc
        if batch:
            yield batch

    async def _async_wait_for_final_frames(self, multi_job_id: str, master_url: str, worker_id: str,
                                           dtype: torch.dtype, device: torch.device):
        return await dist_async_wait_for_final_frames(multi_job_id, master_url, worker_id, dtype, device)

    def _wait_for_final_frames(self, multi_job_id: str, master_url: str, worker_id: str,
                               dtype: torch.dtype, device: torch.device):
        return wait_for_final_frames(multi_job_id, master_url, worker_id, dtype, device)

    def _send_finals_chunked(self, multi_job_id, frames_bytes, enabled_workers):
        return dist_send_finals_chunked(multi_job_id, frames_bytes, enabled_workers)


# ------------------------------
# Globals and small utilities
# ------------------------------

# Tiny LRU-ish cache for base grids used by grid_sample (keyed by device+H+W+dtype)
_BASE_GRID_CACHE: Dict[Tuple[str, int, int, torch.dtype], torch.Tensor] = {}
_MAX_GRID_CACHE = 8  # do not let the cache grow indefinitely


def _grid_cache_key(device: torch.device, H: int, W: int, dtype: torch.dtype) -> Tuple[str, int, int, torch.dtype]:
    return (f"{device.type}:{device.index if device.index is not None else -1}", H, W, dtype)


def _get_base_grid(H: int, W: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Returns a normalized base grid of shape [1, H, W, 2] for grid_sample with align_corners=True.
    Cached to avoid re-allocations each call.
    """
    key = _grid_cache_key(device, H, W, dtype)
    grid = _BASE_GRID_CACHE.get(key)
    if grid is None:
        xs = torch.arange(W, device=device, dtype=dtype)
        ys = torch.arange(H, device=device, dtype=dtype)
        if W > 1:
            xs = xs.mul_(2.0 / (W - 1)).add_(-1.0)
        else:
            xs = xs.zero_()
        if H > 1:
            ys = ys.mul_(2.0 / (H - 1)).add_(-1.0)
        else:
            ys = ys.zero_()
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # H,W
        base = torch.stack((xx, yy), dim=-1).unsqueeze(0)  # 1,H,W,2
        if len(_BASE_GRID_CACHE) >= _MAX_GRID_CACHE:
            _BASE_GRID_CACHE.pop(next(iter(_BASE_GRID_CACHE)))
        _BASE_GRID_CACHE[key] = base
        grid = base
    return grid


# ------------------------------
# Format utilities
# ------------------------------

def _to_nchw(x: torch.Tensor) -> torch.Tensor:
    # No forced contiguous; keep lightweight unless ops need it.
    return x.permute(0, 3, 1, 2)


def _to_bhwc(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 3, 1)


def _pick_device(pref: str) -> torch.device:
    pref = (pref or "auto").lower()
    if pref == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pref == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _move(x: torch.Tensor, dev: torch.device) -> torch.Tensor:
    return x if x.device == dev else x.to(dev, non_blocking=True)


# ------------------------------
# Fast Gaussian approx (unified)
# ------------------------------

def _boxes_for_gauss(sigma: float, n: int = 3):
    """
    Compute n box kernel sizes that approximate a Gaussian with std=sigma.
    Note: sigma is used as-is (no division by 3 anywhere).
    """
    s = float(max(sigma, 1e-6))
    ideal = math.sqrt((12 * s * s / n) + 1)
    lower = int(math.floor(ideal))
    if lower % 2 == 0:
        lower -= 1
    upper = lower + 2
    m = round((12 * s * s - n * lower * lower - 4 * n * lower - 3 * n) / (-4 * lower - 4))
    return [lower if i < m else upper for i in range(n)]


def _box_blur_1d_prefixsum_nchw(x: torch.Tensor, r: int, dim: int) -> torch.Tensor:
    """
    Separable 1D blur via prefix sums. Keeps temporaries minimal.
    dim: 3 -> width, 2 -> height
    Clamps reflect padding radius to be valid for the tensor size.
    """
    if r <= 0:
        return x
    if dim == 3:  # width
        W = x.shape[3]
        r_eff = int(min(r, max(W - 1, 0)))
        if r_eff <= 0:
            return x
        ks = 2 * r_eff + 1
        xpad = F.pad(x, (r_eff, r_eff, 0, 0), mode="reflect")
        cs = F.pad(torch.cumsum(xpad, dim=3), (1, 0, 0, 0), value=0.0)
        out = (cs[..., ks:] - cs[..., :-ks]) / ks
        return out
    if dim == 2:  # height
        H = x.shape[2]
        r_eff = int(min(r, max(H - 1, 0)))
        if r_eff <= 0:
            return x
        ks = 2 * r_eff + 1
        xpad = F.pad(x, (0, 0, r_eff, r_eff), mode="reflect")
        cs = F.pad(torch.cumsum(xpad, dim=2), (0, 0, 1, 0), value=0.0)
        out = (cs[:, :, ks:, :] - cs[:, :, :-ks, :]) / ks
        return out
    raise ValueError("dim must be 2 (H) or 3 (W)")


def _box_blur_nchw(x: torch.Tensor, r: int) -> torch.Tensor:
    """2D box blur by two 1D passes."""
    if r <= 0:
        return x
    return _box_blur_1d_prefixsum_nchw(_box_blur_1d_prefixsum_nchw(x, r, dim=3), r, dim=2)


@torch.no_grad()
def _fast_gauss_approx_bhwc(
    x_bhwc: torch.Tensor,
    radius: float,
    frame_chunk: int = 1,
    only_first_channel: bool = False,
    pyramid_target_radius: float = 32.0,
) -> torch.Tensor:
    """
    Unified 3× box approximation of Gaussian blur in BHWC.
    Internal pipeline keeps NCHW to minimize permute overhead.
    """
    B, H, W, C = x_bhwc.shape
    r_full = float(max(radius, 0.0))
    if r_full <= 0.0:
        return x_bhwc[..., :1] if only_first_channel else x_bhwc

    # Choose power-of-two shrink so that working radius ~ pyramid_target_radius
    shrink = 1
    if r_full > pyramid_target_radius:
        shrink = int(2 ** max(0, math.floor(math.log2(r_full / max(pyramid_target_radius, 1e-6)))))

    r_small = r_full / float(shrink)
    ks_list = _boxes_for_gauss(r_small, n=3)
    radii = [max(0, (k - 1) // 2) for k in ks_list]

    csz = max(1, int(frame_chunk))
    outs: List[torch.Tensor] = []

    for i in range(0, B, csz):
        part_bhwc = x_bhwc[i:i + csz]
        x = _to_nchw(part_bhwc)  # N,C,H,W

        # Downscale if needed (area preserves energy)
        if shrink > 1:
            oh = max(1, H // shrink)
            ow = max(1, W // shrink)
            x = F.interpolate(x, size=(oh, ow), mode="area", recompute_scale_factor=False)

        # Channel selection only once
        if only_first_channel and x.shape[1] > 1:
            x = x[:, :1, :, :]

        # 3-pass box blur
        for r in radii:
            x = _box_blur_nchw(x, r)

        # Upscale back if needed
        if shrink > 1:
            x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False, recompute_scale_factor=False)

        outs.append(_to_bhwc(x))

    return torch.cat(outs, dim=0)


# ------------------------------
# Node: HighPassBlend (optimized)
# ------------------------------

class HighPassBlend:
    """
    High-pass-like blend per frame with reduced format churn:
      result1 = 0.5 + 0.5 * (foreground - blur(foreground))
      result2 = blur(background)
      out_ll  = LinearLight(result2, result1)
      out     = lerp(result2, out_ll, opacity) -> clamp to [0..1]
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "foreground_image": ("IMAGE",),
                "background_image": ("IMAGE",),
                "blur_size": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 500.0, "step": 0.1}),
                "opacity":   ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0,  "step": 0.01}),
                "contrast":   ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0,  "step": 0.01}),
                "device":    (["auto", "cuda", "cpu"],),
                "store_result_on": (["cpu", "same_as_input"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "image/blending"

    @torch.no_grad()
    def _process_single_out(self, img1: torch.Tensor, img2: torch.Tensor, blur_size: float, opacity: float, contrast: float) -> torch.Tensor:
        """
        Computes final blended frame directly (keeps intermediates short-lived).
        """
        img1 = img1.clamp(0.0, 1.0)
        img2 = img2.clamp(0.0, 1.0)

        # Unified blur, sigma = blur_size (no /3)
        blur1 = _fast_gauss_approx_bhwc(img1, float(blur_size), frame_chunk=1, only_first_channel=False)
        blur2 = _fast_gauss_approx_bhwc(img2, float(blur_size), frame_chunk=1, only_first_channel=False)

        result1 = 0.5 + 0.5 * (img1 - blur1)

        if opacity <= 0.0:
            out = blur2
        else:
            # Linear Light blending
            out_ll = ((blur2 + 2.0 * result1 - 1.0) - 0.5) * contrast + 0.5
            out = out_ll if opacity >= 1.0 else (blur2 * (1.0 - opacity) + out_ll * opacity)

        return out.clamp(0.0, 1.0)

    def apply(self, foreground_image, background_image, blur_size, opacity, contrast, device="auto", store_result_on="cpu", progress_bar=True):
        if foreground_image.shape != background_image.shape:
            raise ValueError("HighPassBlend: foreground_image and background_image must have the same BHWC shape.")

        tgt = _pick_device(device)
        out_store_dev = torch.device("cpu") if store_result_on == "cpu" else foreground_image.device
        B = int(foreground_image.shape[0])

        with torch.no_grad():
            if B > 1:
                t0 = time.perf_counter()
                if progress_bar: pb = ProgressBar(B)
                outs_cpu: List[torch.Tensor] = []
                for i in range(B):
                    img1 = _move(foreground_image[i:i+1], tgt)
                    img2 = _move(background_image[i:i+1], tgt)
                    out = self._process_single_out(img1, img2, float(blur_size), float(opacity), float(contrast))
                    outs_cpu.append(out.to(torch.device("cpu"), non_blocking=True))
                    if progress_bar: pb.update(1)

                out_batched = torch.cat(outs_cpu, dim=0)
                if out_store_dev.type != "cpu":
                    out_batched = out_batched.to(out_store_dev, non_blocking=True)

                gc.collect()
                if tgt.type == "cuda":
                    torch.cuda.synchronize()

                dt = time.perf_counter() - t0
                print(f"[HighPassBlend] {B} frame(s), {dt:.3f} s, {dt/max(1,B):.4f} s/frame", flush=True)
                return (out_batched,)
            else:
                img1 = _move(foreground_image, tgt)
                img2 = _move(background_image, tgt)
                out = self._process_single_out(img1, img2, float(blur_size), float(opacity), float(contrast))
                out_cpu = out.to(torch.device("cpu"), non_blocking=True)
                if out_store_dev.type != "cpu":
                    out_cpu = out_cpu.to(out_store_dev, non_blocking=True)
                gc.collect()
                if tgt.type == "cuda":
                    torch.cuda.synchronize()
                return (out_cpu,)


# ------------------------------
# Scale helpers
# ------------------------------

def _downscale_bhwc(x_bhwc: torch.Tensor, scale: float) -> torch.Tensor:
    s = float(scale)
    if s >= 0.999:
        return x_bhwc
    n, h, w, _ = x_bhwc.shape
    oh, ow = max(1, int(h * s)), max(1, int(w * s))
    return _to_bhwc(F.interpolate(_to_nchw(x_bhwc), size=(oh, ow), mode="area", recompute_scale_factor=False))


def _upscale_to_bhwc(x_bhwc_small: torch.Tensor, H: int, W: int) -> torch.Tensor:
    return _to_bhwc(F.interpolate(_to_nchw(x_bhwc_small), size=(H, W), mode="bilinear", align_corners=False, recompute_scale_factor=False))


def _morph_dilate_mask(mask_bhwc: torch.Tensor, pixels: int) -> torch.Tensor:
    """Dilation via max-pool; expects 1-channel or will take the first channel.
    Clamps reflect padding to valid range for H and W.
    """
    if pixels <= 0:
        return mask_bhwc
    x = _to_nchw(mask_bhwc[..., :1])  # N,1,H,W
    H, W = x.shape[2], x.shape[3]
    p = int(min(pixels, max(min(H, W) - 1, 0)))
    if p <= 0:
        return mask_bhwc
    k = p * 2 + 1
    y = F.max_pool2d(F.pad(x, (p, p, p, p), mode="reflect"), kernel_size=k, stride=1)
    return _to_bhwc(y)



# ------------------------------
# Optical flow helpers
# ------------------------------

def _to_gray_bhwc(x_bhwc: torch.Tensor) -> torch.Tensor:
    """Efficient RGB->Gray (BHWC). Keeps a single 1-channel output to save memory."""
    if x_bhwc.shape[-1] == 1:
        return x_bhwc
    r = x_bhwc[..., 0:1]
    g = x_bhwc[..., 1:2]
    b = x_bhwc[..., 2:3]
    return (0.299 * r + 0.587 * g + 0.114 * b)


_SOBEL_CACHE: Dict[Tuple[str,int,torch.dtype], Tuple[torch.Tensor,torch.Tensor]] = {}

def _get_sobel_kernels(dtype: torch.dtype, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cache Sobel kernels per (device,dtype)."""
    key = (device.type, device.index if device.index is not None else -1, dtype)
    k = _SOBEL_CACHE.get(key)
    if k is None:
        kx = torch.tensor([[1, 0, -1],
                           [2, 0, -2],
                           [1, 0, -1]], dtype=dtype, device=device).view(1, 1, 3, 3) / 8.0
        ky = torch.tensor([[1, 2, 1],
                           [0, 0, 0],
                           [-1, -2, -1]], dtype=dtype, device=device).view(1, 1, 3, 3) / 8.0
        _SOBEL_CACHE[key] = (kx, ky)
        return kx, ky
    return k

def _sobel_nchw(x_nchw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # x: N,1,H,W -> Ix, Iy
    kx, ky = _get_sobel_kernels(x_nchw.dtype, x_nchw.device)
    Ix = F.conv2d(x_nchw, kx, padding=1)
    Iy = F.conv2d(x_nchw, ky, padding=1)
    return Ix, Iy


def _box_sum_nchw(x: torch.Tensor, k: int) -> torch.Tensor:
    """Sum over k×k window using avg_pool2d (avg * k^2 equals sum)."""
    pad = k // 2
    return F.avg_pool2d(x, kernel_size=k, stride=1, padding=pad) * (k * k)


def _sanitize_unit(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


def _flow_warp_bhwc(x_bhwc: torch.Tensor, flow_bhwc: torch.Tensor) -> torch.Tensor:
    """
    Warp BHWC image by BHWC flow (pixels) using grid_sample with cached base grid.
    Auto-resizes flow to match image HxW if needed (bilinear, align_corners=True, scaled).
    Ensures input and grid dtypes match.
    """
    x_nchw = _to_nchw(x_bhwc)
    N, C, H, W = x_nchw.shape

    # keep flow in same dtype as input to satisfy grid_sample
    dt = x_nchw.dtype
    flow = flow_bhwc[..., :2].to(dt)

    if flow.shape[0] != N:
        if flow.shape[0] == 1 and N > 1:
            flow = flow.expand(N, *flow.shape[1:])
        else:
            raise ValueError(f"flow batch size {flow.shape[0]} != image batch {N}")

    fH, fW = flow.shape[1], flow.shape[2]
    if (fH != H) or (fW != W):
        sx = W / max(fW, 1)
        sy = H / max(fH, 1)
        flow = _to_bhwc(F.interpolate(_to_nchw(flow), size=(H, W), mode="bilinear", align_corners=True))
        scale = torch.tensor((sx, sy), device=flow.device, dtype=dt).view(1, 1, 1, 2)
        flow = flow * scale

    flow = torch.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)

    # base grid must share dtype & device with input
    base = _get_base_grid(H, W, x_nchw.device, dt)  # [1,H,W,2], same dtype as input
    nx = 2.0 * flow[..., 0] / max(W - 1, 1)
    ny = 2.0 * flow[..., 1] / max(H - 1, 1)
    offs = torch.stack((nx, ny), dim=-1).to(dt)
    grid = (base + offs).clamp(-1.0, 1.0)

    y = F.grid_sample(x_nchw, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return _to_bhwc(y)


@torch.no_grad()
def _lk_optical_flow_prev_to_curr_bhwc(prev_gray_bhwc: torch.Tensor, curr_gray_bhwc: torch.Tensor,
                                       levels: int = 3, window: int = 5, iters: int = 3, reg: float = 1e-3) -> torch.Tensor:
    """
    Pyramidal Lucas–Kanade optical flow (RAM-friendly).
    Precomputes curr gradients per level; avoids recomputing Ix/Iy each inner iteration.
    Inputs: prev_gray_bhwc, curr_gray_bhwc — BHWC, float32 in [0,1] (1 channel preferred).
    Returns: flow BHWC (dx, dy) in pixels, float32.
    """
    if prev_gray_bhwc.shape != curr_gray_bhwc.shape:
        raise ValueError("Flow inputs must have the same shape.")
    device = prev_gray_bhwc.device
    N, H_full, W_full, _ = curr_gray_bhwc.shape

    pixels = H_full * W_full
    batch_chunk = 1 if pixels >= 2_000_000 else (2 if pixels >= 1_000_000 else (4 if pixels >= 500_000 else 8))
    batch_chunk = max(1, min(batch_chunk, N))

    out_flows: List[torch.Tensor] = []
    # Use the dtype of the input frames for all internal computations.  This
    # preserves half precision when the caller provides float16 inputs.  It is
    # assumed that prev_gray_bhwc and curr_gray_bhwc share the same dtype.
    dtype = prev_gray_bhwc.dtype
    k = int(window)

    for b0 in range(0, N, batch_chunk):
        b1 = min(N, b0 + batch_chunk)
        # Preserve the original dtype when converting the clamped batches.
        prev_b = prev_gray_bhwc[b0:b1].clamp(0.0, 1.0).to(dtype)
        curr_b = curr_gray_bhwc[b0:b1].clamp(0.0, 1.0).to(dtype)

        def _down_to_level(x: torch.Tensor, lvl: int) -> torch.Tensor:
            scale = 1.0 / (2 ** (levels - 1 - lvl))
            return _downscale_bhwc(x, scale) if scale < 0.999 else x

        flow = None

        for lvl in range(levels):
            prev_l = _down_to_level(prev_b, lvl)
            curr_l = _down_to_level(curr_b, lvl)
            NB, H, W, _ = curr_l.shape

            if flow is None:
                # Initialise the flow field using the working dtype.  This ensures
                # that subsequent warping and interpolation stay in the same
                # precision as the input.
                flow = torch.zeros((NB, H, W, 2), device=device, dtype=dtype)
            else:
                # Upscale previous level's flow and double vector magnitude
                flow = _to_bhwc(F.interpolate(_to_nchw(flow), size=(H, W), mode="bilinear", align_corners=True)) * 2.0
                flow = torch.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)

            # Precompute gradients of current image ONCE per level
            curr_nchw = _to_nchw(curr_l)
            Ix, Iy = _sobel_nchw(curr_nchw)

            for _ in range(iters):
                prev_w = _flow_warp_bhwc(prev_l, flow)
                It = curr_nchw - _to_nchw(prev_w)  # N,1,H,W

                Ixt = Ix * It
                Iyt = Iy * It

                Ixx = Ix * Ix
                Iyy = Iy * Iy
                Ixy = Ix * Iy

                Sxx = _box_sum_nchw(Ixx, k)
                Syy = _box_sum_nchw(Iyy, k)
                Sxy = _box_sum_nchw(Ixy, k)
                Sxt = _box_sum_nchw(Ixt, k)
                Syt = _box_sum_nchw(Iyt, k)

                det = (Sxx * Syy - Sxy * Sxy) + reg
                du = (-Syy * Sxt + Sxy * Syt) / det
                dv = ( Sxy * Sxt - Sxx * Syt) / det

                dflow = torch.cat((du, dv), dim=1)  # N,2,H,W
                flow = torch.nan_to_num(flow + _to_bhwc(dflow), nan=0.0, posinf=0.0, neginf=0.0)

            # free per-level temporaries explicitly
            del curr_nchw, Ix, Iy

        out_flows.append(flow)

    return torch.cat(out_flows, dim=0)


# ====== Node (optimized parts) ======

class PreserveGradientsBlend:
    """
    GPU-optimized, device-selectable.

    Behavior:
      • 'downscale_by' controls a full low-res execution path for mask computation.
      • Pixel-sized parameters are interpreted in FULL-RES pixels and scaled by 'downscale_by'.
      • 'highpass_radius' and 'mask_blur' are FLOAT (support fractional), max 500.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "foreground_image": ("IMAGE",),
                "background_image": ("IMAGE",),

                "highpass_radius": ("FLOAT", {"default": 15.0, "min": 0.0,  "max": 500.0, "step": 0.1}),
                "downscale_by":   ("FLOAT", {"default": 0.25, "min": 0.01, "max": 1.0,    "step": 0.01}),
                "strength":       ("FLOAT", {"default": 10.0, "min": 0.0,  "max": 32.0,  "step": 0.1}),

                "expand":         ("INT",   {"default": 5,    "min": 0,    "max": 128,   "step": 1}),
                "mask_blur":      ("FLOAT", {"default": 25.0, "min": 0.0,  "max": 500.0, "step": 0.1}),

                "clamp_blacks":   ("FLOAT", {"default": 0.05, "min": 0.0,  "max": 1.0,   "step": 0.01}),
                "clamp_whites":   ("FLOAT", {"default": 0.90, "min": 0.0,  "max": 1.0,   "step": 0.01}),
                "temporal_deflicker": ("BOOLEAN", {"default": True}),
                "device":         (["auto","cuda","cpu"],),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "apply"
    CATEGORY = "image/blending"

    @staticmethod
    def _scaled_int(px_full: float, ds: float) -> int:
        """Convert a full-res pixel parameter into a low-res integer radius for morphological ops."""
        if px_full <= 0.0:
            return 0
        return max(1, int(round(float(px_full) * float(ds))))

    @staticmethod
    def _scaled_float(px_full: float, ds: float) -> float:
        """Scale full-res float radius to low-res float radius."""
        return float(px_full) * float(ds)

    @torch.no_grad()
    def _hp_once_abs_mask_small(self, img_small_bhwc: torch.Tensor, radius_full_px: float, strength: float, ds: float) -> torch.Tensor:
        """
        High-pass magnitude mask at LOW resolution (1-channel, 0..1).
        radius_full_px is in full-res pixels; scaled by 'ds' for the small image.
        """
        radius_small = self._scaled_float(radius_full_px, ds)
        img_small = img_small_bhwc.clamp(0.0, 1.0)
        blur_small = _fast_gauss_approx_bhwc(img_small, float(radius_small), frame_chunk=1, only_first_channel=False) if radius_small > 0.0 else img_small
        hp = 0.5 + 0.5 * strength * (img_small - blur_small)
        return (2.0 * torch.abs(hp - 0.5)).clamp(0.0, 1.0)[..., :1]  # 1-channel

    @torch.no_grad()
    def _base_mask_small(
        self,
        bg_small_bhwc: torch.Tensor,
        highpass_radius_full: float,
        strength: float,
        clamp_blacks: float,
        clamp_whites: float,
        use_amp: bool,
        ds: float,
    ) -> torch.Tensor:
        """Base (pre-deflicker) mask at LOW resolution, 1-channel, normalized by clamp_*."""
        dev_type = bg_small_bhwc.device.type
        with torch.autocast(device_type=dev_type, dtype=torch.float16, enabled=(use_amp and dev_type == "cuda")):
            m = self._hp_once_abs_mask_small(bg_small_bhwc, float(highpass_radius_full), float(strength), float(ds))
            lo = float(clamp_blacks)
            hi = float(clamp_whites)
            if hi <= lo:
                hi = lo + 1e-6
            m = ((m - lo) / (hi - lo)).clamp(0.0, 1.0)
        return m  # BHWC, 1ch, low-res

    @torch.no_grad()
    def apply(
        self,
        foreground_image,
        background_image,
        highpass_radius,
        downscale_by,
        strength,
        expand,
        mask_blur,
        clamp_blacks,
        clamp_whites,
        temporal_deflicker=False,
        device="auto",
        progress_bar=True
    ):
        if background_image.shape != foreground_image.shape:
            raise ValueError("PreserveGradientsBlend: background_image and foreground_image must have the same BHWC shape.")

        tgt = _pick_device(device)
        use_amp = True

        B, H, W, _ = background_image.shape
        ds = float(min(1.0, max(0.01, float(downscale_by))))

        # Low-res parameters
        expand_small    = self._scaled_int(int(expand), ds)        # int for dilation
        mask_blur_small = self._scaled_float(float(mask_blur), ds) # float for blur

        out_imgs: List[torch.Tensor] = []
        out_masks: List[torch.Tensor] = []

        prev_gray_small = None
        prev_mask_base_small = None
        next_gray_small = None
        next_mask_base_small = None

        if B >= 2:
            bg1_full = _move(background_image[1:2], tgt).clamp(0.0, 1.0)
            bg1_small = _downscale_bhwc(bg1_full, ds)
            # Preserve dtype during grayscale conversion
            next_gray_small = _to_gray_bhwc(bg1_small).to(bg1_small.dtype)
            next_mask_base_small = self._base_mask_small(
                bg1_small, float(highpass_radius), float(strength),
                float(clamp_blacks), float(clamp_whites), use_amp=use_amp, ds=ds
            )

        if progress_bar: pb = ProgressBar(B)
        t0 = time.perf_counter()

        for i in range(B):
            # Full-res images for final blending
            bg_cur_full = _move(background_image[i:i+1], tgt).clamp(0.0, 1.0)
            fg_cur_full = _move(foreground_image[i:i+1], tgt).clamp(0.0, 1.0)

            # Low-res background for mask pipeline
            bg_small = _downscale_bhwc(bg_cur_full, ds)
            # Preserve dtype during grayscale conversion
            cur_gray_small = _to_gray_bhwc(bg_small).to(bg_small.dtype)

            # Base mask at low res (already clamped by blacks/whites)
            cur_mask_base_small = self._base_mask_small(
                bg_small, float(highpass_radius), float(strength),
                float(clamp_blacks), float(clamp_whites), use_amp=use_amp, ds=ds
            )

            # Temporal deflicker at LOW resolution
            m_small = cur_mask_base_small
            if temporal_deflicker:
                have_prev = (prev_gray_small is not None) and (prev_mask_base_small is not None)
                have_next = (next_gray_small is not None) and (next_mask_base_small is not None)

                if have_prev or have_next:
                    if have_prev and have_next:
                        w_cur, w_prev, w_next = 0.50, 0.25, 0.25
                    elif have_prev:
                        w_cur, w_prev, w_next = 0.50, 0.50, 0.00
                    else:
                        w_cur, w_prev, w_next = 0.50, 0.00, 0.50

                    acc = m_small * w_cur
                    if have_prev:
                        flow_p = _lk_optical_flow_prev_to_curr_bhwc(prev_gray_small, cur_gray_small, levels=3, window=5, iters=3, reg=1e-3)
                        acc = acc + _sanitize_unit(_flow_warp_bhwc(prev_mask_base_small, flow_p)) * w_prev
                    if have_next:
                        flow_n = _lk_optical_flow_prev_to_curr_bhwc(next_gray_small, cur_gray_small, levels=3, window=5, iters=3, reg=1e-3)
                        acc = acc + _sanitize_unit(_flow_warp_bhwc(next_mask_base_small, flow_n)) * w_next

                    m_small = _sanitize_unit(acc / max(w_cur + (w_prev if have_prev else 0.0) + (w_next if have_next else 0.0), 1e-8))

            # Post-ops at LOW resolution
            if int(expand_small) > 0:
                m_small = _morph_dilate_mask(m_small, int(expand_small)).clamp(0.0, 1.0)
            if float(mask_blur_small) > 0.0:
                m_small = _fast_gauss_approx_bhwc(m_small, float(mask_blur_small), frame_chunk=1, only_first_channel=True).clamp(0.0, 1.0)

            # Upscale final LOW-RES mask back to FULL resolution (1-channel)
            mask_full_1ch = _upscale_to_bhwc(m_small, H, W).clamp(0.0, 1.0)

            # Full-res blending
            out_full = (bg_cur_full * (1.0 - mask_full_1ch) + fg_cur_full * mask_full_1ch).clamp(0.0, 1.0)

            # Collect outputs on original device
            out_imgs.append(_move(out_full, background_image.device))
            out_masks.append(_move(mask_full_1ch.repeat(1, 1, 1, 3), background_image.device))

            # Advance temporal state (LOW-RES)
            prev_gray_small = cur_gray_small
            prev_mask_base_small = cur_mask_base_small

            if i + 2 < B:
                bg_next2_full = _move(background_image[i+2:i+3], tgt).clamp(0.0, 1.0)
                bg_next2_small = _downscale_bhwc(bg_next2_full, ds)
                # Preserve dtype during grayscale conversion
                next_gray_small = _to_gray_bhwc(bg_next2_small).to(bg_next2_small.dtype)
                next_mask_base_small = self._base_mask_small(
                    bg_next2_small, float(highpass_radius), float(strength),
                    float(clamp_blacks), float(clamp_whites), use_amp=use_amp, ds=ds
                )
            else:
                next_gray_small = None
                next_mask_base_small = None

            if progress_bar: pb.update(1)

        gc.collect()
        if tgt.type == 'cuda':
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f"[PreserveGradientsBlend] {B} frame(s), {dt:.3f} s, {dt/max(1,B):.4f} s/frame", flush=True)

        return (torch.cat(out_imgs, dim=0), torch.cat(out_masks, dim=0))


class UpscaleWithModelBatchedTiled:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "upscale_model": ("UPSCALE_MODEL",),
                "images": ("IMAGE",),  # UI по-прежнему ждёт IMAGE; PIL поддерживается для внутреннего вызова
                "per_batch": ("INT",  {"default": 4, "min": 1, "max": 64, "step": 1}),
                "force_tiled": ("BOOLEAN", {"default": True}),
                "max_tile_size": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 32}),
                "tile_overlap": ("INT", {"default": 32, "min": 0, "max": 256, "step": 1}),
                "run_distributed": ("BOOLEAN", {"default": True}),
            },
            "hidden": {
                "multi_job_id": ("STRING", {"default": ""}),
                "is_worker": ("BOOLEAN", {"default": False}),
                "master_url": ("STRING", {"default": ""}),
                "enabled_worker_ids": ("STRING", {"default": "[]"}),
                "worker_id": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "apply"
    CATEGORY = "image/upscaling"

    @torch.no_grad()
    def apply(
        self,
        upscale_model,
        images,                   # Tensor [B,H,W,C] OR PIL.Image OR list[PIL.Image]
        per_batch: int,
        force_tiled: bool,
        max_tile_size: int,
        tile_overlap: int,
        run_distributed: bool = True,
        # hidden
        multi_job_id: str = "",
        is_worker: bool = False,
        master_url: str = "",
        enabled_worker_ids: str = "[]",
        worker_id: str = "",
    ):
        # -------- normalize input: accept Tensor OR PIL OR list[PIL] --------
        pil_in = False
        if isinstance(images, Image.Image):
            pil_in = True
            frames_pil = [images]
        elif isinstance(images, list) and (len(images) == 0 or isinstance(images[0], Image.Image)):
            pil_in = True
            frames_pil = images
        elif isinstance(images, torch.Tensor):
            assert images.ndim == 4, "Expected [B,H,W,C] tensor."
            if images.shape[-1] not in (1, 2, 3, 4):
                raise ValueError("Expected last dimension (channels) ∈ {1,2,3,4}.")
            if images.shape[-1] != 3:
                c = images.shape[-1]
                if c == 1:
                    images = images.repeat(1, 1, 1, 3)
                elif c == 4:
                    images = images[..., :3]
                else:
                    raise ValueError("Unsupported channel count. Expect RGB or convertible to RGB.")
            frames_pil = None
        else:
            raise ValueError("images must be a BHWC torch.Tensor or PIL.Image/list[PIL.Image].")

        # Keep original dtype/device for final tensor assembly (fallback defaults for PIL-only)
        if isinstance(images, torch.Tensor):
            orig_dtype = images.dtype
            orig_device = images.device
            B = int(images.shape[0])
        else:
            orig_dtype = torch.get_default_dtype()
            orig_device = torch.device("cpu")
            B = len(frames_pil)

        # -------- local-only branch (no distribution) --------
        if not bool(run_distributed):
            self._log(
                f"Local-only mode: run_distributed=False; processing all B={B} frames locally "
                f"(per_batch={per_batch}, tiled={force_tiled}, tile={max_tile_size}, overlap={tile_overlap})"
            )
            out_pil = self._run_local_helper_pil(
                upscale_model=upscale_model,
                images=frames_pil if pil_in else images,
                per_batch=per_batch,
                force_tiled=force_tiled,
                max_tile=max_tile_size,
                overlap=tile_overlap,
            )
            out_tensor = self._pil_list_to_bhwc_tensor(out_pil, dtype=orig_dtype, device=orig_device)
            _empty_cache()
            return (out_tensor,)

        # -------- parse workers --------
        enabled_workers = self._parse_enabled_workers(enabled_worker_ids)
        sorted_workers = sorted(enabled_workers)
        total_nodes = 1 + len(sorted_workers)

        # this node index
        if bool(is_worker):
            if not master_url or not multi_job_id or not worker_id:
                raise RuntimeError("Worker mode requires non-empty 'master_url', 'multi_job_id', and 'worker_id'.")
            try:
                idx_in_sorted = sorted_workers.index(str(worker_id))
            except ValueError:
                raise RuntimeError(f"Worker id '{worker_id}' is not in enabled_worker_ids.")
            node_idx = 1 + idx_in_sorted
        else:
            node_idx = 0

        # split by index (equal split; remainder to last)
        base = B // total_nodes
        remainder = B % total_nodes
        if total_nodes == 1:
            start_idx, end_idx = 0, B
        else:
            if node_idx < total_nodes - 1:
                start_idx = node_idx * base
                end_idx = (node_idx + 1) * base
            else:
                start_idx = (total_nodes - 1) * base
                end_idx = B

        # ---------------- Worker branch ----------------
        if bool(is_worker):
            self._log(
                f"Worker compute: worker_id={worker_id}, range=[{start_idx}:{end_idx}) of B={B}, "
                f"per_batch={per_batch}, tiled={force_tiled}, tile={max_tile_size}, overlap={tile_overlap}"
            )
            # slice local
            if pil_in:
                local_in = frames_pil[start_idx:end_idx]
            else:
                local_in = images[start_idx:end_idx].contiguous()

            # run locally -> PIL list
            local_out_pil = self._run_local_helper_pil(
                upscale_model=upscale_model,
                images=local_in,
                per_batch=per_batch,
                force_tiled=force_tiled,
                max_tile=max_tile_size,
                overlap=tile_overlap,
            )

            if end_idx <= start_idx:
                self._log(f"Worker empty slice, waiting for finals")
                final_out = wait_for_final_frames(
                    multi_job_id=multi_job_id,
                    master_url=master_url,
                    worker_id=str(worker_id),
                    dtype=orig_dtype,
                    device=orig_device,
                )
                _empty_cache()
                return (final_out,)

            # submit to master (JPEG)
            self._worker_submit_finals_chunked_sync(
                local_out=local_out_pil,  # now supports PIL list
                global_offset=start_idx,
                multi_job_id=multi_job_id,
                master_url=master_url,
                worker_id=str(worker_id),
            )

            # wait for finals (tensor)
            self._log(f"Worker awaiting finals broadcast: multi_job_id={multi_job_id}")
            final_out = wait_for_final_frames(
                multi_job_id=multi_job_id,
                master_url=master_url,
                worker_id=str(worker_id),
                dtype=orig_dtype,
                device=orig_device,
            )
            self._log(f"Worker received finals: B={final_out.shape[0]}")
            _empty_cache()
            return (final_out,)

        # ---------------- Master branch ----------------
        if len(sorted_workers) == 0 or not multi_job_id:
            if len(sorted_workers) > 0 and not multi_job_id:
                self._log("Enabled workers present but 'multi_job_id' empty — processing locally only.")
            self._log(
                f"Master local compute only: B={B}, per_batch={per_batch}, "
                f"tiled={force_tiled}, tile={max_tile_size}, overlap={tile_overlap}"
            )
            out_pil = self._run_local_helper_pil(
                upscale_model=upscale_model,
                images=frames_pil if pil_in else images,
                per_batch=per_batch,
                force_tiled=force_tiled,
                max_tile=max_tile_size,
                overlap=tile_overlap,
            )
            out_tensor = self._pil_list_to_bhwc_tensor(out_pil, dtype=orig_dtype, device=orig_device)
            _empty_cache()
            return (out_tensor,)

        # init dynamic job
        try:
            run_async_in_server_loop(
                init_dynamic_job(multi_job_id, batch_size=B, enabled_workers=sorted_workers),
                timeout=15.0
            )
            self._log(f"Master initialized dynamic job: B={B}, workers={len(sorted_workers)}")
        except Exception as e:
            self._log(f"init_dynamic_job error: {type(e).__name__}: {e}")
            out_pil = self._run_local_helper_pil(
                upscale_model=upscale_model,
                images=frames_pil if pil_in else images,
                per_batch=per_batch,
                force_tiled=force_tiled,
                max_tile=max_tile_size,
                overlap=tile_overlap,
            )
            out_tensor = self._pil_list_to_bhwc_tensor(out_pil, dtype=orig_dtype, device=orig_device)
            _empty_cache()
            return (out_tensor,)

        # master's slice -> local compute (PIL)
        self._log(
            f"Master local compute: range=[{start_idx}:{end_idx}) of B={B}, "
            f"per_batch={per_batch}, tiled={force_tiled}, tile={max_tile_size}, overlap={tile_overlap}"
        )
        local_in = frames_pil[start_idx:end_idx] if pil_in else images[start_idx:end_idx].contiguous()
        local_out_pil = self._run_local_helper_pil(
            upscale_model=upscale_model,
            images=local_in,
            per_batch=per_batch,
            force_tiled=force_tiled,
            max_tile=max_tile_size,
            overlap=tile_overlap,
        )

        # collect workers + master's local part into dict[int->PIL]
        completed_images = self._master_collect_and_fill(
            multi_job_id=multi_job_id,
            total=B,
            master_slice_out=local_out_pil,  # PIL list
            master_offset=start_idx,
        )

        # fallback for missing indices
        missing = [i for i in range(B) if i not in completed_images]
        if missing:
            self._log(f"Master fallback: computing {len(missing)} missing frame(s) locally {missing[:16]}...")
            miss_in = [frames_pil[i] for i in missing] if pil_in else images[missing].contiguous()
            miss_out_pil = self._run_local_helper_pil(
                upscale_model=upscale_model,
                images=miss_in,
                per_batch=per_batch,
                force_tiled=force_tiled,
                max_tile=max_tile_size,
                overlap=tile_overlap,
            )
            for k, gi in enumerate(missing):
                completed_images[gi] = miss_out_pil[k]

        missing_after = [i for i in range(B) if i not in completed_images]
        if missing_after:
            raise RuntimeError(f"Missing frames after fallback: {missing_after[:16]} (total {len(missing_after)})")

        # order and broadcast
        ordered_pil = [completed_images[i] for i in range(B)]
        arrs = [np.asarray(p, dtype=np.uint8).copy(order="C") for p in ordered_pil]
        try:
            executor = get_jpeg_executor()
            frames_bytes = list(executor.map(_encode_np_to_jpeg, arrs))
        except Exception:
            frames_bytes = []
            for a in arrs:
                pil = Image.fromarray(a, mode="RGB")
                bio = io.BytesIO()
                pil.save(bio, format="JPEG", quality=100, subsampling=0, optimize=False)
                frames_bytes.append(bio.getvalue())

        try:
            dist_send_finals_chunked(
                multi_job_id=multi_job_id,
                frames_bytes=frames_bytes,
                enabled_workers=sorted_workers,
            )
            #self._log(f"Master registered finals for workers: B={len(frames_bytes)}")
        except Exception as e:
            self._log(f"Broadcast finals error: {type(e).__name__}: {e}")

        # return as Comfy IMAGE tensor (preserve dtype/device)
        final_tensor = self._pil_list_to_bhwc_tensor(ordered_pil, dtype=orig_dtype, device=orig_device)
        _empty_cache()
        return (final_tensor,)

    # --------------------------- local compute (PIL out) ---------------------------

    @torch.no_grad()
    def _run_local_helper_pil(self, upscale_model, images, per_batch, force_tiled, max_tile, overlap):
        """
        Call the unified helper; always return list[PIL.Image] to reduce RAM and encode easily.
        """
        out_pil = upscale_with_model_batched(
            upscale_model=upscale_model,
            images=images,                 # tensor OR PIL(list)
            per_batch=int(per_batch),
            show_tqdm=False,
            verbose=False,
            prefer_tiled=bool(force_tiled),
            sticky_tiled_after_oom=True,
            auto_tiled_vram_threshold=0.90,
            direct_clear_cuda_each_batch=True,
            clear_cuda_each_batch=True,
            init_tile=int(max_tile),
            min_tile=128,
            overlap=int(overlap),
            remember_tile_across_batches=True,
            return_pil=True,               # ← всегда PIL
        )
        return out_pil

    # --------------------------- master collecting ---------------------------

    def _master_collect_and_fill(self, multi_job_id: str, total: int, master_slice_out, master_offset: int) -> dict:
        """
        Accept master's local results (list[PIL] or tensor[B,H,W,C]) and drain workers' queue
        into a dict[int -> PIL.Image].
        """
        completed = {}

        # put master's part
        if isinstance(master_slice_out, list) and (len(master_slice_out) == 0 or isinstance(master_slice_out[0], Image.Image)):
            for i, im in enumerate(master_slice_out):
                completed[master_offset + i] = im
        elif isinstance(master_slice_out, torch.Tensor):
            for i in range(master_slice_out.shape[0]):
                completed[master_offset + i] = self._tensor_to_pil(master_slice_out[i])
        else:
            raise ValueError("master_slice_out must be list[PIL.Image] or torch.Tensor[B,H,W,3].")

        timeout = float(get_worker_timeout_seconds())
        start = time.time()

        async def _drain_until_done():
            prompt_server = ensure_tile_jobs_initialized()
            async with prompt_server.distributed_tile_jobs_lock:
                job = prompt_server.distributed_pending_tile_jobs.get(multi_job_id)
                if not job or 'queue' not in job:
                    raise RuntimeError(f"Job queue not initialized for {multi_job_id}")
                q = job['queue']

            while len(completed) < total and (time.time() - start) < timeout:
                try:
                    result = await asyncio.wait_for(q.get(), timeout=1.0)
                    if 'image_idx' in result and 'image' in result:
                        gi = int(result['image_idx'])
                        if gi not in completed:
                            completed[gi] = result['image']  # already PIL
                            #self._log(f"Master registered image {gi} from worker")
                except asyncio.TimeoutError:
                    pass

        run_async_in_server_loop(_drain_until_done(), timeout=timeout + 5.0)
        return completed

    # --------------------------- worker upload ---------------------------

    def _encode_tensor_list_to_jpeg_bytes(self, t: torch.Tensor) -> list:
        """Encode [N,H,W,3] float[0..1] tensor to list of JPEG bytes."""
        arrs = []
        for i in range(int(t.shape[0])):
            arr = (t[i].detach().cpu().numpy().clip(0.0, 1.0) * 255.0).astype(np.uint8)
            arrs.append(arr)
        try:
            executor = get_jpeg_executor()
            frames_bytes = list(executor.map(_encode_np_to_jpeg, arrs))
        except Exception:
            frames_bytes = []
            for a in arrs:
                pil = Image.fromarray(a, mode="RGB")
                bio = io.BytesIO()
                pil.save(bio, format="JPEG", quality=100, subsampling=0, optimize=False)
                frames_bytes.append(bio.getvalue())
        return frames_bytes

    def _encode_pil_list_to_jpeg_bytes(self, frames_pil: list) -> list:
        """Encode list[PIL.Image] to list of JPEG bytes using threadpool."""
        u8_list = [np.asarray(im, dtype=np.uint8).copy(order="C") for im in frames_pil]
        try:
            executor = get_jpeg_executor()
            frames_bytes = list(executor.map(_encode_np_to_jpeg, u8_list))
        except Exception:
            frames_bytes = []
            for a in u8_list:
                pil = Image.fromarray(a, mode="RGB")
                bio = io.BytesIO()
                pil.save(bio, format="JPEG", quality=100, subsampling=0, optimize=False)
                frames_bytes.append(bio.getvalue())
        return frames_bytes

    def _worker_submit_finals_chunked_sync(
        self,
        local_out,             # torch.Tensor[B,H,W,3] OR list[PIL.Image]
        global_offset: int,
        multi_job_id: str,
        master_url: str,
        worker_id: str,
    ):
        """
        Batch-submit worker frames to master using chunked base64 JSON.
        Accepts either tensor or list[PIL.Image].
        """
        max_batch = self._env_max_batch()
        max_payload = self._env_max_payload()
        concurrency = self._env_upload_concurrency()

        if isinstance(local_out, torch.Tensor):
            frames_bytes = self._encode_tensor_list_to_jpeg_bytes(local_out)
        elif isinstance(local_out, list) and (len(local_out) == 0 or isinstance(local_out[0], Image.Image)):
            frames_bytes = self._encode_pil_list_to_jpeg_bytes(local_out)
        else:
            raise ValueError("local_out must be tensor[B,H,W,3] or list[PIL.Image].")

        total = len(frames_bytes)
        self._log(f"Worker upload: prepared {total} JPEG(s), max_batch={max_batch}, max_payload={max_payload}")

        async def _try_post_json(session: aiohttp.ClientSession, payload: dict) -> bool:
            urls = [
                f"{master_url}/distributed/submit_images_indexed",
                f"{master_url}/distributed/submit_images",
                f"{master_url}/distributed/submit_worker_finals",
                f"{master_url}/distributed/submit_finals",
            ]
            for u in urls:
                try:
                    async with session.post(u, json=payload) as resp:
                        if resp.status == 200:
                            return True
                        if resp.status in (404, 405):
                            continue
                except Exception:
                    continue
            return False

        async def _send_single_images_parallel(session, frames, offset, job_id, wid, parallel):
            sem = asyncio.Semaphore(parallel)
            url = f"{master_url}/distributed/submit_image"

            async def _post_one(i: int):
                async with sem:
                    gi = offset + i
                    is_last_flag = (i == len(frames) - 1)
                    data = aiohttp.FormData()
                    data.add_field("multi_job_id", str(job_id))
                    data.add_field("worker_id", str(wid))
                    data.add_field("image_idx", str(gi))
                    data.add_field("is_last", "true" if is_last_flag else "false")
                    data.add_field("full_image", io.BytesIO(frames[i]), filename=f"image_{gi}.jpg", content_type="image/jpeg")

                    max_retries = 5
                    delay = 0.5
                    for attempt in range(max_retries):
                        try:
                            async with session.post(url, data=data) as resp:
                                resp.raise_for_status()
                                return
                        except Exception:
                            if attempt < max_retries - 1:
                                await asyncio.sleep(delay)
                                delay = min(5.0, delay * 2)
                            else:
                                raise

            tasks = [asyncio.create_task(_post_one(i)) for i in range(len(frames))]
            await asyncio.gather(*tasks)
            self._log(f"Worker upload: completed parallel single-image upload, total={len(frames)}")

        async def _send_batched():
            session = await get_client_session()
            sent = 0
            chunk_idx = 0
            for chunk in UltimateSDVideoUpscaleDistributed.iter_finals_payload_chunks(frames_bytes, max_batch, max_payload):
                payload = {
                    "multi_job_id": str(multi_job_id),
                    "worker_id": str(worker_id),
                    "offset": int(global_offset + sent),
                    "is_last": bool(sent + len(chunk) == total),
                    "frames_b64": [base64.b64encode(b).decode("ascii") for b in chunk],
                }
                ok = await _try_post_json(session, payload)
                if not ok:
                    self._log("Worker upload: batch endpoint unavailable, falling back to parallel single-image POSTs")
                    await _send_single_images_parallel(session, frames_bytes, global_offset, multi_job_id, worker_id, concurrency)
                    return
                sent += len(chunk)
                chunk_idx += 1
                self._log(f"Worker upload: sent chunk {chunk_idx} ({len(chunk)} frames), progress {sent}/{total}")
            self._log(f"Worker upload: completed batched upload, total={total}")

        run_async_in_server_loop(
            _send_batched(),
            timeout=max(60.0, 2.0 * float(get_worker_timeout_seconds())),
        )

    # --------------------------- conversions & misc ---------------------------

    def _tensor_to_pil(self, t: torch.Tensor) -> Image.Image:
        arr = (t.detach().cpu().numpy().clip(0.0, 1.0) * 255.0).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

    def _pil_list_to_bhwc_tensor(self, imgs: list, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if len(imgs) == 0:
            return torch.empty((0,), dtype=dtype, device=device)
        w0, h0 = imgs[0].size
        B = len(imgs)
        out = torch.empty((B, h0, w0, 3), dtype=dtype, device=device)
        for i, im in enumerate(imgs):
            a = np.asarray(im, dtype=np.uint8)
            t_cpu = torch.from_numpy(a).to(dtype=dtype)
            if torch.is_floating_point(t_cpu):
                t_cpu = t_cpu / 255.0
            t = t_cpu.to(device=device, dtype=dtype, non_blocking=False)
            out[i].copy_(t)
            del a, t_cpu, t
        return out

    def _parse_enabled_workers(self, s: str) -> list:
        try:
            return [w for w in (json.loads(s) if s else []) if isinstance(w, str) and w]
        except Exception:
            return []

    def _env_max_batch(self) -> int:
        try:
            v = int(os.getenv("COMFYUI_MAX_BATCH", "20"))
            return max(1, min(256, v))
        except Exception:
            return 20

    def _env_max_payload(self) -> int:
        try:
            v = int(os.getenv("COMFYUI_MAX_PAYLOAD_SIZE", str(50 * 1024 * 1024)))
            return max(256 * 1024, min(1024 * 1024 * 1024, v))
        except Exception:
            return 50 * 1024 * 1024

    def _env_upload_concurrency(self) -> int:
        try:
            v = int(os.getenv("COMFYUI_UPLOAD_CONCURRENCY", "8"))
            return max(1, min(32, v))
        except Exception:
            return 8

    def _log(self, msg: str):
        print(f"[UpscaleWithModelBatchedTiled] {msg}", flush=True)


# Register nodes (if needed in this file)
NODE_CLASS_MAPPINGS = {
    #"UltimateSDUpscaleDistributed": UltimateSDUpscaleDistributed,
    "UltimateSDVideoUpscaleDistributed": UltimateSDVideoUpscaleDistributed,
    "HighPassBlend": HighPassBlend,
    "PreserveGradientsBlend": PreserveGradientsBlend,
    "UpscaleWithModelBatchedTiled": UpscaleWithModelBatchedTiled,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    #"UltimateSDUpscaleDistributed": "Ultimate SD Upscale Distributed (No Upscale)",
    "UltimateSDVideoUpscaleDistributed": "Ultimate SD Video Upscale Distributed",
    "HighPassBlend": "High Pass Blend",
    "PreserveGradientsBlend": "Preserve Gradients Blend",
    "UpscaleWithModelBatchedTiled": "Upscale With Model (Batched, Tiled)",
}