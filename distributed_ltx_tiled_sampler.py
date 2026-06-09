"""
Distributed LTX tiled sampler node.

This module is intentionally isolated from distributed_upscale.py and uses its
own /distributed/ltx/* API namespace. It assumes the original LTXTiledSampler
module is importable in the same ComfyUI runtime. Workers must run the same
workflow and have the same model/custom-node stack installed; guider/sampler
objects are not serialized over the network.
"""

import asyncio
import base64
import io
import json
import time
import uuid

import aiohttp
from aiohttp import web
import torch
import server

try:
    from .utils.async_helpers import run_async_in_server_loop
    from .utils.config import get_worker_timeout_seconds
    from .utils.constants import TILE_WAIT_TIMEOUT, TILE_SEND_TIMEOUT, MAX_PAYLOAD_SIZE
    from .utils.logging import debug_log, log
    from .utils.network import get_client_session, handle_api_error
except Exception:
    def debug_log(message): print(message)
    def log(message): print(message)
    def get_worker_timeout_seconds(): return 90
    TILE_WAIT_TIMEOUT = 30
    TILE_SEND_TIMEOUT = 120
    MAX_PAYLOAD_SIZE = 50 * 1024 * 1024
    async def get_client_session(): return aiohttp.ClientSession()
    async def handle_api_error(request, error, status=500):
        return web.json_response({"status": "error", "message": str(error)}, status=status)
    def run_async_in_server_loop(coro, timeout=None):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(asyncio.wait_for(coro, timeout) if timeout else coro)
        finally:
            loop.close()

try:
    from .latent_tiled_sampler import LTXTiledSampler
except Exception:
    try:
        from latent_tiled_sampler import LTXTiledSampler
    except Exception as exc:
        raise ImportError(
            "LTXTiledSamplerDistributed requires latent_tiled_sampler.py to be "
            "importable on master and workers."
        ) from exc

try:
    from safetensors.torch import save as _st_save, load as _st_load
    _HAS_SAFETENSORS = True
except Exception:
    _st_save = None
    _st_load = None
    _HAS_SAFETENSORS = False


def _pack_tensors(tensors):
    clean = {k: v.detach().contiguous().cpu() for k, v in tensors.items() if isinstance(v, torch.Tensor)}
    if _HAS_SAFETENSORS:
        return "safetensors", _st_save(clean)
    buf = io.BytesIO()
    torch.save(clean, buf)
    return "torch", buf.getvalue()


def _unpack_tensors(raw, serializer, device):
    if serializer == "safetensors":
        data = _st_load(raw)
    else:
        buf = io.BytesIO(raw)
        try:
            data = torch.load(buf, map_location="cpu", weights_only=True)
        except TypeError:
            buf.seek(0)
            data = torch.load(buf, map_location="cpu")
    return {k: v.to(device=device) for k, v in data.items()}


def _b64(raw):
    return base64.b64encode(raw).decode("ascii")


def _unb64(text):
    return base64.b64decode(text.encode("ascii"))


def _ensure_ltx_state():
    ps = server.PromptServer.instance
    if not hasattr(ps, "distributed_ltx_jobs"):
        ps.distributed_ltx_jobs = {}
        ps.distributed_ltx_jobs_lock = asyncio.Lock()
    return ps


async def _init_job(job_id, count, workers):
    ps = _ensure_ltx_state()
    async with ps.distributed_ltx_jobs_lock:
        if job_id in ps.distributed_ltx_jobs:
            return
        pending = asyncio.Queue()
        for i in range(int(count)):
            await pending.put(i)
        workers = [str(w) for w in workers or []]
        ps.distributed_ltx_jobs[job_id] = {
            "pending": pending,
            "results": asyncio.Queue(),
            "completed": {},
            "workers": {w: time.time() for w in workers},
            "assigned": {w: [] for w in workers},
        }


async def _cleanup_job(job_id):
    ps = _ensure_ltx_state()
    async with ps.distributed_ltx_jobs_lock:
        ps.distributed_ltx_jobs.pop(job_id, None)


async def _get_next_tile(job_id, worker_id=None):
    ps = _ensure_ltx_state()
    async with ps.distributed_ltx_jobs_lock:
        job = ps.distributed_ltx_jobs.get(job_id)
        if not job:
            return None
        try:
            idx = await asyncio.wait_for(job["pending"].get(), timeout=0.05)
        except asyncio.TimeoutError:
            return None
        if worker_id is not None:
            worker_id = str(worker_id)
            job["workers"][worker_id] = time.time()
            job["assigned"].setdefault(worker_id, []).append(idx)
        return int(idx)


async def _mark_completed(job_id, tile_idx, packet=None):
    ps = _ensure_ltx_state()
    async with ps.distributed_ltx_jobs_lock:
        job = ps.distributed_ltx_jobs.get(job_id)
        if not job:
            return
        job["completed"][int(tile_idx)] = packet or {"local": True}
        for assigned in job["assigned"].values():
            while int(tile_idx) in assigned:
                assigned.remove(int(tile_idx))


async def _drain_results(job_id):
    ps = _ensure_ltx_state()
    async with ps.distributed_ltx_jobs_lock:
        job = ps.distributed_ltx_jobs.get(job_id)
        if not job:
            return []
        out = []
        while True:
            try:
                packet = job["results"].get_nowait()
            except asyncio.QueueEmpty:
                break
            tile_idx = int(packet["tile_idx"])
            if tile_idx not in job["completed"]:
                job["completed"][tile_idx] = packet
                out.append(packet)
            for assigned in job["assigned"].values():
                while tile_idx in assigned:
                    assigned.remove(tile_idx)
        return out


async def _counts(job_id):
    ps = _ensure_ltx_state()
    async with ps.distributed_ltx_jobs_lock:
        job = ps.distributed_ltx_jobs.get(job_id)
        if not job:
            return {"pending": 0, "assigned": 0, "completed": 0, "active_workers": 0}
        return {
            "pending": job["pending"].qsize(),
            "assigned": sum(len(v) for v in job["assigned"].values()),
            "completed": len(job["completed"]),
            "active_workers": len(job["workers"]),
        }


async def _requeue_stale(job_id):
    ps = _ensure_ltx_state()
    timeout = float(get_worker_timeout_seconds())
    now = time.time()
    async with ps.distributed_ltx_jobs_lock:
        job = ps.distributed_ltx_jobs.get(job_id)
        if not job:
            return 0
        total = 0
        for worker_id, seen in list(job["workers"].items()):
            if now - seen <= timeout:
                continue
            assigned = list(job["assigned"].get(worker_id, []))
            job["workers"].pop(worker_id, None)
            job["assigned"][worker_id] = []
            for idx in assigned:
                if idx not in job["completed"]:
                    await job["pending"].put(idx)
                    total += 1
        return total


@server.PromptServer.instance.routes.get("/distributed/ltx/job_status")
async def _ltx_job_status(request):
    job_id = request.query.get("multi_job_id")
    ps = _ensure_ltx_state()
    async with ps.distributed_ltx_jobs_lock:
        return web.json_response({"ready": bool(job_id and job_id in ps.distributed_ltx_jobs)})


@server.PromptServer.instance.routes.post("/distributed/ltx/request_tile")
async def _ltx_request_tile(request):
    try:
        data = await request.json()
        job_id = data.get("multi_job_id")
        worker_id = str(data.get("worker_id", ""))
        if not job_id or not worker_id:
            return await handle_api_error(request, "Missing multi_job_id or worker_id", 400)
        idx = await _get_next_tile(job_id, worker_id)
        return web.json_response({"tile_idx": idx})
    except Exception as exc:
        return await handle_api_error(request, exc, 500)


@server.PromptServer.instance.routes.post("/distributed/ltx/submit_tile")
async def _ltx_submit_tile(request):
    try:
        length = request.headers.get("content-length")
        if length and int(length) > int(MAX_PAYLOAD_SIZE):
            return await handle_api_error(request, f"Payload too large: {length} bytes", 413)
        data = await request.post()
        job_id = data.get("multi_job_id")
        worker_id = str(data.get("worker_id", ""))
        tile_idx = data.get("tile_idx")
        field = data.get("tensor_payload")
        if not job_id or not worker_id or tile_idx is None or field is None or not hasattr(field, "file"):
            return await handle_api_error(request, "Missing tile submission field", 400)
        ps = _ensure_ltx_state()
        async with ps.distributed_ltx_jobs_lock:
            job = ps.distributed_ltx_jobs.get(job_id)
            if not job:
                return await handle_api_error(request, "Job not found", 404)
            await job["results"].put({
                "worker_id": worker_id,
                "tile_idx": int(tile_idx),
                "serializer": data.get("serializer", "torch"),
                "payload": field.file.read(),
            })
        return web.json_response({"status": "success"})
    except Exception as exc:
        return await handle_api_error(request, exc, 500)


class LTXTiledSamplerDistributed(LTXTiledSampler):
    @classmethod
    def INPUT_TYPES(cls):
        base = LTXTiledSampler.INPUT_TYPES()
        optional = dict(base.get("optional", {}))
        optional["master_participates"] = ("BOOLEAN", {"default": False})
        return {
            "required": dict(base.get("required", {})),
            "optional": optional,
            "hidden": {
                "multi_job_id": ("STRING", {"default": ""}),
                "is_worker": ("BOOLEAN", {"default": False}),
                "master_url": ("STRING", {"default": ""}),
                "enabled_worker_ids": ("STRING", {"default": "[]"}),
                "worker_id": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("output", "denoised_output")
    FUNCTION = "sample_tiled_distributed"
    CATEGORY = "10S Nodes/Sampling"

    def sample_tiled_distributed(self, noise, guider, sampler, sigmas, latent_image,
                                 bypass_tiling=False, tile_axis="auto", n_tiles=2,
                                 tile_overlap=8, max_size_for_no_tile=24,
                                 audio_pass="passthrough", audio_carrier_tile="first",
                                 debug=False, grid_tiling=False,
                                 master_participates=False,
                                 multi_job_id="", is_worker=False, master_url="",
                                 enabled_worker_ids="[]", worker_id=""):
        # Safe fallback: with no distributed job id this is exactly the original node.
        if not multi_job_id or bypass_tiling:
            return super().sample_tiled(
                noise, guider, sampler, sigmas, latent_image,
                bypass_tiling=bypass_tiling, tile_axis=tile_axis,
                n_tiles=n_tiles, tile_overlap=tile_overlap,
                max_size_for_no_tile=max_size_for_no_tile,
                audio_pass=audio_pass, audio_carrier_tile=audio_carrier_tile,
                debug=debug, grid_tiling=grid_tiling,
            )

        # This first implementation intentionally delegates actual per-tile math
        # to the verified LTXTiledSampler path on the process that owns the graph.
        # The distributed API namespace is present and isolated; the next patch can
        # wire tile-index execution into LTXTiledSampler internals without touching
        # existing Distributed upscalers.
        if is_worker:
            log(f"LTX Dist Worker[{str(worker_id)[:8]}]: job {multi_job_id} delegated to local passthrough")
            return (latent_image, latent_image)

        try:
            workers = json.loads(enabled_worker_ids) if enabled_worker_ids else []
            if not isinstance(workers, list):
                workers = []
        except Exception:
            workers = []
        if workers:
            log(f"LTX Dist: workers detected ({len(workers)}), using stable local sampler fallback for this revision")

        return super().sample_tiled(
            noise, guider, sampler, sigmas, latent_image,
            bypass_tiling=False, tile_axis=tile_axis,
            n_tiles=n_tiles, tile_overlap=tile_overlap,
            max_size_for_no_tile=max_size_for_no_tile,
            audio_pass=audio_pass, audio_carrier_tile=audio_carrier_tile,
            debug=debug, grid_tiling=grid_tiling,
        )


NODE_CLASS_MAPPINGS = {"LTXTiledSamplerDistributed": LTXTiledSamplerDistributed}
NODE_DISPLAY_NAME_MAPPINGS = {"LTXTiledSamplerDistributed": "🎲 LTX Tiled Sampler Distributed"}
