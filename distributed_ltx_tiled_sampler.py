"""Distributed LTX Tiled Sampler for ComfyUI.

This module implements a distributed version of the LTX tiled sampler. It allows
splitting an LTX latent tensor into spatial tiles, distributing those tiles to
remote workers for sampling, and blending the results back together. The design
intentionally isolates job management, HTTP endpoints, and tensor packing so as
not to interfere with existing distributed nodes. It uses safetensors to
serialize tile tensors over the network.

When the multi_job_id input is empty this node falls back to the base
LTXTiledSampler implementation and runs on a single GPU. When multi_job_id
is provided the node operates in master or worker mode depending on the
is_worker flag. Master mode will run the sampling locally and dispatch
tiles to any connected workers via the /distributed/ltx API. Worker mode
connects to the master and processes tiles as they arrive.
"""

import asyncio
import base64
import json
import math
import io
from typing import Dict, List, Tuple, Optional

import torch

try:
    from safetensors.torch import save, load
except Exception:
    save = None
    load = None

try:
    from server import PromptServer
    from .utils.async_helpers import run_async_in_server_loop
    from .utils.network import get_client_session
except Exception:
    PromptServer = None
    run_async_in_server_loop = None
    def get_client_session(*args, **kwargs):
        return None

try:
    from latent_tiled_sampler import LTXTiledSampler, _extract_components
except Exception:
    LTXTiledSampler = None
    def _extract_components(samples, debug: bool = False):
        """Fallback extractor. Treats samples as (video, audio)."""
        if isinstance(samples, (list, tuple)) and samples:
            video = samples[0]
            audio = samples[1] if len(samples) > 1 else None
            return video, audio, None
        return samples, None, None


def _compute_tile_starts(total_size: int, n_tiles: int, overlap: int) -> Tuple[List[int], int]:
    """Compute evenly spaced start indices and tile size for tiling."""
    if n_tiles <= 1 or total_size <= 1:
        return [0], total_size
    tile_size = math.ceil((total_size + (n_tiles - 1) * overlap) / n_tiles)
    tile_size = min(tile_size, total_size)
    if n_tiles == 2:
        return [0, max(0, total_size - tile_size)], tile_size
    starts = []
    stride = (total_size - tile_size) / (n_tiles - 1)
    for i in range(n_tiles):
        starts.append(int(round(i * stride)))
    return starts, tile_size


def _make_window_1d(size: int, fade_left: int, fade_right: int, dtype, device):
    """Create a 1D cosine window for blending tile edges."""
    win = torch.ones(size, dtype=dtype, device=device)
    if fade_left > 0:
        fl = min(fade_left, size)
        i = torch.arange(fl, dtype=dtype, device=device)
        win[:fl] = 0.5 * (1.0 - torch.cos(math.pi * i / fl))
    if fade_right > 0:
        fr = min(fade_right, size)
        i = torch.arange(fr, dtype=dtype, device=device)
        win[size - fr:] = 0.5 * (1.0 + torch.cos(math.pi * i / fr))
    return win


def _pack_tensors(tensors: Dict[str, torch.Tensor]) -> bytes:
    """Serialize a dictionary of tensors into bytes using safetensors."""
    if save is None:
        raise RuntimeError("safetensors library not available")
    cpu_tensors = {}
    for name, tensor in tensors.items():
        if tensor is None:
            continue
        cpu_tensors[name] = tensor.detach().contiguous().to("cpu")
    buffer = io.BytesIO()
    save(cpu_tensors, buffer)
    return buffer.getvalue()


def _unpack_tensors(data: bytes, device) -> Dict[str, torch.Tensor]:
    """Deserialize tensors from safetensors bytes."""
    if load is None:
        raise RuntimeError("safetensors library not available")
    buffer = io.BytesIO(data)
    tensors = load(buffer)
    out = {}
    for k, v in tensors.items():
        out[k] = v.to(device)
    return out


def _ensure_ltx_state(server):
    """Ensure the server has the global state containers for LTX jobs."""
    app = server.app
    if not hasattr(app.state, "ltx_jobs"):
        app.state.ltx_jobs = {}   # job_id -> job info
        app.state.ltx_results = {}   # job_id -> {tile_index: payload_bytes}


def _init_job(server, job_id: str, queue: List[dict]):
    """Initialize a job with a task queue."""
    _ensure_ltx_state(server)
    job = {
        "queue": list(queue),
        "in_progress": set(),
        "results": {},
        "completed": 0,
        "total": len(queue),
    }
    server.app.state.ltx_jobs[job_id] = job
    server.app.state.ltx_results[job_id] = {}


def _get_job(server, job_id: str):
    return server.app.state.ltx_jobs.get(job_id)


def _get_next_tile(server, job_id: str) -> Optional[dict]:
    """Pop the next tile from the job queue and mark it in-progress."""
    job = _get_job(server, job_id)
    if not job:
        return None
    while job["queue"]:
        tile = job["queue"].pop(0)
        idx = tile["tile_index"]
        if idx not in job["in_progress"]:
            job["in_progress"].add(idx)
            return tile
    return None


def _store_result(server, job_id: str, tile_index: int, payload: bytes):
    """Store a completed tile payload for later retrieval by the master."""
    server.app.state.ltx_results[job_id][tile_index] = payload
    job = _get_job(server, job_id)
    if job:
        job["completed"] += 1
        job["in_progress"].discard(tile_index)


def _drain_results(server, job_id: str) -> Dict[int, bytes]:
    """Retrieve and clear all available results for a job."""
    results = server.app.state.ltx_results.get(job_id)
    if not results:
        return {}
    out = dict(results)
    results.clear()
    return out


def _cleanup_job(server, job_id: str):
    """Remove the job and its results from server state."""
    if hasattr(server.app.state, "ltx_jobs"):
        server.app.state.ltx_jobs.pop(job_id, None)
        server.app.state.ltx_results.pop(job_id, None)


if PromptServer is not None:
    server = PromptServer.instance
    _ensure_ltx_state(server)

    @server.post("/distributed/ltx/request_tile")
    async def _ltx_request_tile(request):
        data = await request.json()
        job_id = data.get("job_id")
        tile = _get_next_tile(server, job_id)
        if tile is None:
            return {"action": "done"}
        return {"action": "process", **tile}

    @server.post("/distributed/ltx/submit_tile")
    async def _ltx_submit_tile(request):
        data = await request.json()
        job_id = data.get("job_id")
        tile_index = data.get("tile_index")
        payload_b64 = data.get("payload")
        if not job_id or payload_b64 is None or tile_index is None:
            return {"status": "error", "message": "missing fields"}
        try:
            payload = base64.b64decode(payload_b64)
        except Exception:
            return {"status": "error", "message": "invalid payload"}
        _store_result(server, job_id, int(tile_index), payload)
        return {"status": "ok"}


async def _ltx_worker_process(
    node, noise, guider, sampler, sigmas, latent_image,
    denoise_mask, n_h_tiles, n_w_tiles, tile_overlap
):
    """Worker loop: pulls tile assignments from master, processes them, and sends results back."""
    if get_client_session is None:
        raise RuntimeError("No HTTP client available for worker mode")
    session = get_client_session()
    job_id = node.multi_job_id
    master_url = node.master_url
    # Precompute full noise once for consistency across workers
    full_noise = noise.generate_noise({"samples": latent_image})
    # Request and process tiles indefinitely
    while True:
        try:
            async with session.post(
                f"{master_url}/distributed/ltx/request_tile",
                json={"job_id": job_id}
            ) as resp:
                response = await resp.json()
        except Exception:
            break
        if not response:
            await asyncio.sleep(0.1)
            continue
        action = response.get("action")
        if action == "done":
            break
        if action != "process":
            await asyncio.sleep(0.05)
            continue
        tile_index = response["tile_index"]
        y_start = response["h_start"]
        y_end = response["h_end"]
        x_start = response["w_start"]
        x_end = response["w_end"]
        tile_latent = latent_image[:, :, :, y_start:y_end, x_start:x_end].contiguous()
        tile_noise = full_noise[:, :, :, y_start:y_end, x_start:x_end].contiguous()
        tile_mask = None
        if denoise_mask is not None:
            tile_mask = denoise_mask[:, :, :, y_start:y_end, x_start:x_end].contiguous()
        result = guider.sample(
            tile_noise, tile_latent, sampler, sigmas, denoise_mask=tile_mask,
            disable_pbar=True
        )
        if isinstance(result, tuple):
            tile_samples, tile_denoised = result
        else:
            tile_samples, tile_denoised = result, None
        tensors = {"samples": tile_samples}
        if tile_denoised is not None:
            tensors["denoised"] = tile_denoised
        payload_bytes = _pack_tensors(tensors)
        payload_b64 = base64.b64encode(payload_bytes).decode()
        try:
            async with session.post(
                f"{master_url}/distributed/ltx/submit_tile",
                json={
                    "job_id": job_id,
                    "tile_index": tile_index,
                    "payload": payload_b64,
                },
            ) as resp:
                await resp.json()
        except Exception:
            pass
    # worker loop ends


async def _ltx_master_process(
    node, noise, guider, sampler, sigmas, latent_image, denoise_mask,
    n_h_tiles, n_w_tiles, tile_overlap
):
    """Master loop: runs tile sampling locally while handling remote workers."""
    server = PromptServer.instance
    job_id = node.multi_job_id
    # Derive shape
    B, C, F, H, W = latent_image.shape
    starts_y, tile_h = _compute_tile_starts(H, n_h_tiles, tile_overlap)
    starts_x, tile_w = _compute_tile_starts(W, n_w_tiles, tile_overlap)
    queue: List[dict] = []
    tile_idx = 0
    for yi, y_start in enumerate(starts_y):
        y_end = min(y_start + tile_h, H)
        for xi, x_start in enumerate(starts_x):
            x_end = min(x_start + tile_w, W)
            queue.append({
                "tile_index": tile_idx,
                "h_start": y_start,
                "h_end": y_end,
                "w_start": x_start,
                "w_end": x_end,
                "grid_y": yi,
                "grid_x": xi,
            })
            tile_idx += 1
    _init_job(server, job_id, queue)
    full_noise = noise.generate_noise({"samples": latent_image})
    device = latent_image.device
    dtype = latent_image.dtype
    output_sum = torch.zeros_like(latent_image)
    weight_sum = torch.zeros_like(latent_image)
    denoised_sum = torch.zeros_like(latent_image) if denoise_mask is not None else None
    processed = 0
    total_tiles = len(queue)
    while processed < total_tiles:
        results = _drain_results(server, job_id)
        for idx, payload in results.items():
            tens = _unpack_tensors(payload, device)
            tile = next((t for t in queue if t["tile_index"] == idx), None)
            if tile is None:
                continue
            y_start = tile["h_start"]; y_end = tile["h_end"]
            x_start = tile["w_start"]; x_end = tile["w_end"]
            yi = tile["grid_y"]; xi = tile["grid_x"]
            fade_top = tile_overlap if yi > 0 else 0
            fade_bottom = tile_overlap if yi < n_h_tiles - 1 else 0
            fade_left = tile_overlap if xi > 0 else 0
            fade_right = tile_overlap if xi < n_w_tiles - 1 else 0
            wy = _make_window_1d(y_end - y_start, fade_top, fade_bottom, dtype, device)
            wx = _make_window_1d(x_end - x_start, fade_left, fade_right, dtype, device)
            w = wy.view(1, 1, 1, y_end - y_start, 1) * wx.view(1, 1, 1, 1, x_end - x_start)
            out_slice = output_sum[:, :, :, y_start:y_end, x_start:x_end]
            weight_slice = weight_sum[:, :, :, y_start:y_end, x_start:x_end]
            out_slice += tens["samples"] * w
            weight_slice += w
            if denoised_sum is not None and "denoised" in tens:
                denoised_slice = denoised_sum[:, :, :, y_start:y_end, x_start:x_end]
                denoised_slice += tens["denoised"] * w
            processed += 1
        if processed < total_tiles:
            next_tile = None
            for t in queue:
                if t.get("_local_done"):
                    continue
                idx = t["tile_index"]
                if idx in server.app.state.ltx_results.get(job_id, {}):
                    continue
                next_tile = t
                break
            if next_tile is None:
                await asyncio.sleep(0.05)
                continue
            next_tile["_local_done"] = True
            y_start = next_tile["h_start"]; y_end = next_tile["h_end"]
            x_start = next_tile["w_start"]; x_end = next_tile["w_end"]
            yi = next_tile["grid_y"]; xi = next_tile["grid_x"]
            tile_latent = latent_image[:, :, :, y_start:y_end, x_start:x_end].contiguous()
            tile_noise = full_noise[:, :, :, y_start:y_end, x_start:x_end].contiguous()
            tile_mask = None
            if denoise_mask is not None:
                tile_mask = denoise_mask[:, :, :, y_start:y_end, x_start:x_end].contiguous()
            result = guider.sample(
                tile_noise, tile_latent, sampler, sigmas,
                denoise_mask=tile_mask, disable_pbar=True
            )
            if isinstance(result, tuple):
                tile_samples, tile_denoised = result
            else:
                tile_samples, tile_denoised = result, None
            fade_top = tile_overlap if yi > 0 else 0
            fade_bottom = tile_overlap if yi < n_h_tiles - 1 else 0
            fade_left = tile_overlap if xi > 0 else 0
            fade_right = tile_overlap if xi < n_w_tiles - 1 else 0
            wy = _make_window_1d(y_end - y_start, fade_top, fade_bottom, dtype, device)
            wx = _make_window_1d(x_end - x_start, fade_left, fade_right, dtype, device)
            w = wy.view(1, 1, 1, y_end - y_start, 1) * wx.view(1, 1, 1, 1, x_end - x_start)
            out_slice = output_sum[:, :, :, y_start:y_end, x_start:x_end]
            weight_slice = weight_sum[:, :, :, y_start:y_end, x_start:x_end]
            out_slice += tile_samples * w
            weight_slice += w
            if denoised_sum is not None and tile_denoised is not None:
                denoised_slice = denoised_sum[:, :, :, y_start:y_end, x_start:x_end]
                denoised_slice += tile_denoised * w
            processed += 1
    weight_sum = torch.where(weight_sum == 0, torch.ones_like(weight_sum), weight_sum)
    output_sum = output_sum / weight_sum
    if denoised_sum is not None:
        denoised_sum = denoised_sum / weight_sum
    _cleanup_job(server, job_id)
    video = output_sum
    _, input_audio, fmt = _extract_components(latent_image)
    audio = input_audio
    if denoised_sum is not None:
        return video, audio, denoised_sum
    return video, audio, None


class LTXTiledSamplerDistributed:
    """
    Distributed wrapper around the standard LTXTiledSampler. When running without
    a multi_job_id this node behaves exactly like the base sampler. When a
    multi_job_id is provided, the node either runs in master mode (is_worker=False)
    or worker mode (is_worker=True).
    """
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = {}
        if LTXTiledSampler is not None:
            base_inputs = LTXTiledSampler.INPUT_TYPES()
        base_inputs.setdefault("hidden", {})
        base_inputs["hidden"].update({
            "multi_job_id": ("STRING", {"default": ""}),
            "is_worker": ("BOOLEAN", {"default": False}),
            "master_url": ("STRING", {"default": ""}),
            "worker_id": ("STRING", {"default": ""}),
            "enabled_worker_ids": ("STRING", {"default": "[]"}),
        })
        return base_inputs

    RETURN_TYPES = ("LATENT", "LATENT", "LATENT")
    FUNCTION = "sample_tiled_distributed"
    CATEGORY = "latent/diffusion"

    def __init__(self):
        self.base = LTXTiledSampler() if LTXTiledSampler is not None else None

    @staticmethod
    def IS_CHANGED(**kwargs):
        return float("nan")

    def sample_tiled_distributed(
        self, noise, guider, sampler, sigmas, latent_image, denoise_mask=None,
        h_tiles: int = 1, w_tiles: int = 1, overlap: int = 0,
        multi_job_id: str = "", is_worker: bool = False, master_url: str = "",
        worker_id: str = "", enabled_worker_ids: str = "[]"
    ):
        if not multi_job_id or LTXTiledSampler is None:
            return self.base.sample_tiled(
                noise, guider, sampler, sigmas,
                latent_image, denoise_mask, h_tiles, w_tiles, overlap
            )
        self.multi_job_id = multi_job_id
        self.master_url = master_url.rstrip("/")
        self.worker_id = worker_id
        self.enabled_worker_ids = enabled_worker_ids
        if is_worker:
            if run_async_in_server_loop is None:
                raise RuntimeError("async helpers unavailable in worker mode")
            run_async_in_server_loop(
                _ltx_worker_process(
                    self, noise, guider, sampler, sigmas,
                    latent_image, denoise_mask,
                    h_tiles, w_tiles, overlap
                )
            )
            return (latent_image, latent_image, latent_image)
        else:
            if run_async_in_server_loop is None:
                raise RuntimeError("async helpers unavailable in master mode")
            video, audio, denoised = run_async_in_server_loop(
                _ltx_master_process(
                    self, noise, guider, sampler, sigmas,
                    latent_image, denoise_mask,
                    h_tiles, w_tiles, overlap
                )
            )
            if audio is None:
                return (video, denoised if denoised is not None else video, video)
            else:
                wrapped = (video, audio)
                wrapped_denoised = (denoised, audio) if denoised is not None else (video, audio)
                return (wrapped, wrapped_denoised, wrapped)


NODE_CLASS_MAPPINGS = {"LTXTiledSamplerDistributed": LTXTiledSamplerDistributed}
NODE_DISPLAY_NAME_MAPPINGS = {"LTXTiledSamplerDistributed": "LTX Tiled Sampler (Distributed)"}
