import asyncio
import time
import json
import copy
import os
import io
from aiohttp import web, ClientTimeout
import base64
import server
from PIL import Image

# Import from other utilities
from .logging import debug_log, log
from .network import handle_api_error, get_client_session
# We avoid converting to tensors on the master for tiles; blending uses PIL

# Configure maximum payload size (50MB default, configurable via environment variable)
MAX_PAYLOAD_SIZE = int(os.environ.get('COMFYUI_MAX_PAYLOAD_SIZE', str(50 * 1024 * 1024)))

# Import MAX_BATCH for final frame batching
from .constants import MAX_BATCH

# Import HEARTBEAT_TIMEOUT from constants
from .constants import HEARTBEAT_TIMEOUT
from .config import load_config


def _parse_tiles_from_form(data):
    """Parse tiles submitted via multipart/form-data into a list of tile dicts.

    Expects the following fields in the aiohttp form data:
    - 'tiles_metadata': JSON list with per-tile metadata items containing at least
      'tile_idx', 'x', 'y', 'extracted_width', 'extracted_height'. Optional
      'batch_idx' and 'global_idx' are included when available.
    - 'tile_{i}': image bytes for each tile described in tiles_metadata (PNG).
    - 'padding': integer padding used during extraction (optional; defaults 0).

    Returns: list of dicts with keys: 'image', 'tile_idx', 'x', 'y',
    'extracted_width', 'extracted_height', and optional 'batch_idx', 'global_idx',
    plus 'padding'.
    """
    try:
        # Parse padding if present
        padding = int(data.get('padding', 0)) if data.get('padding') is not None else 0
    except Exception:
        padding = 0

    # Parse tiles metadata (JSON list)
    meta_raw = data.get('tiles_metadata')
    if meta_raw is None:
        raise ValueError("Missing tiles_metadata")

    try:
        metadata = json.loads(meta_raw)
    except Exception as e:
        raise ValueError(f"Invalid tiles_metadata JSON: {e}")

    if not isinstance(metadata, list):
        raise ValueError("tiles_metadata must be a list")

    tiles = []
    # Iterate over metadata items and corresponding uploaded files tile_0, tile_1, ...
    for i, meta in enumerate(metadata):
        file_field = data.get(f'tile_{i}')
        if file_field is None or not hasattr(file_field, 'file'):
            raise ValueError(f"Missing tile data for index {i}")

        # Read image bytes and decode to PIL
        raw = file_field.file.read()
        try:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Invalid image data for tile {i}: {e}")

        # Build tile dictionary (store PIL only; master blends via PIL)
        try:
            tile_info = {
                'image': img,
                'tile_idx': int(meta.get('tile_idx', i)),
                'x': int(meta.get('x', 0)),
                'y': int(meta.get('y', 0)),
                'extracted_width': int(meta.get('extracted_width', img.width)),
                'extracted_height': int(meta.get('extracted_height', img.height)),
                'padding': int(padding),
            }
        except Exception as e:
            raise ValueError(f"Invalid metadata values for tile {i}: {e}")

        # Optional fields
        if 'batch_idx' in meta:
            try:
                tile_info['batch_idx'] = int(meta['batch_idx'])
            except Exception:
                pass
        if 'global_idx' in meta:
            try:
                tile_info['global_idx'] = int(meta['global_idx'])
            except Exception:
                pass

        tiles.append(tile_info)

    return tiles


# Unified Job Data Structure Keys
JOB_QUEUE = 'queue'
JOB_MODE = 'mode'
JOB_COMPLETED_TASKS = 'completed_tasks'
JOB_WORKER_STATUS = 'worker_status'
JOB_ASSIGNED_TO_WORKERS = 'assigned_to_workers'
JOB_PENDING_TASKS = 'pending_tasks'
JOB_BATCH_SIZE = 'batch_size'  # For dynamic
JOB_NUM_TILES_PER_IMAGE = 'num_tiles_per_image'  # For static

# Task Types
TASK_TYPE_TILE = 'tile'
TASK_TYPE_IMAGE = 'image'

from typing import List, Optional

async def init_dynamic_job(multi_job_id: str, batch_size: int, enabled_workers: List[str], all_indices: Optional[List[int]] = None):
    """Initialize queue for dynamic mode (per-image), with collector fields.

    - Creates JOB_PENDING_TASKS with image indices
    - Adds 'completed_images' dict and 'pending_images' alias used by collectors
    """
    await _init_job_queue(
        multi_job_id,
        'dynamic',
        batch_size=batch_size,
        all_indices=all_indices or list(range(batch_size)),
        enabled_workers=enabled_workers,
    )

    prompt_server = ensure_tile_jobs_initialized()
    async with prompt_server.distributed_tile_jobs_lock:
        job_data = prompt_server.distributed_pending_tile_jobs[multi_job_id]
        job_data['completed_images'] = {}
        job_data['pending_images'] = job_data[JOB_PENDING_TASKS]
    debug_log(f"Job {multi_job_id} initialized with {batch_size} images")


async def init_static_job_batched(multi_job_id: str, batch_size: int, num_tiles_per_image: int, enabled_workers: List[str]):
    """Initialize queue for static mode (batched-per-tile).

    - Populates JOB_PENDING_TASKS with tile ids [0..num_tiles_per_image-1]
    """
    await _init_job_queue(
        multi_job_id,
        'static',
        batch_size=batch_size,
        num_tiles_per_image=num_tiles_per_image,
        enabled_workers=enabled_workers,
        batched_static=True,
    )
    # Initialization handled by master; avoid duplicate init logs here

async def _init_job_queue(multi_job_id, mode, batch_size=None, num_tiles_per_image=None, all_indices=None, enabled_workers=None, task_assignments=None, batched_static: bool = False):
    """Unified initialization for job queues in static and dynamic modes."""
    prompt_server = ensure_tile_jobs_initialized()
    async with prompt_server.distributed_tile_jobs_lock:
        if multi_job_id in prompt_server.distributed_pending_tile_jobs:
            debug_log(f"Queue already exists for {multi_job_id}")
            return

        job_data = {
            JOB_QUEUE: asyncio.Queue(),
            JOB_MODE: mode,
            JOB_COMPLETED_TASKS: {},
            JOB_WORKER_STATUS: {w: time.time() for w in enabled_workers or []},
            JOB_ASSIGNED_TO_WORKERS: {w: [] for w in enabled_workers or []},
            JOB_PENDING_TASKS: asyncio.Queue(),
        }

        if mode == 'dynamic':
            job_data[JOB_BATCH_SIZE] = batch_size
            pending_queue = job_data[JOB_PENDING_TASKS]
            for i in (all_indices or range(batch_size)):
                await pending_queue.put(i)
            debug_log(f"Initialized image queue with {batch_size} pending items")
        elif mode == 'static':
            job_data[JOB_NUM_TILES_PER_IMAGE] = num_tiles_per_image
            job_data[JOB_BATCH_SIZE] = batch_size
            job_data['batched_static'] = bool(batched_static)
            # For batched static distribution, populate only tile ids [0..num_tiles_per_image-1]
            pending_queue = job_data[JOB_PENDING_TASKS]
            if batched_static and num_tiles_per_image is not None:
                for i in range(num_tiles_per_image):
                    await pending_queue.put(i)
            else:
                total_tiles = batch_size * num_tiles_per_image
                for i in range(total_tiles):
                    await pending_queue.put(i)
            
            # Keep backward compatibility - if task assignments provided, still track them
            if task_assignments and enabled_workers:
                # task_assignments[0] is for master, 1+ are for workers
                for i, worker_id in enumerate(enabled_workers):
                    if i + 1 < len(task_assignments):
                        job_data[JOB_ASSIGNED_TO_WORKERS][worker_id] = task_assignments[i + 1]
                        debug_log(f"Worker {worker_id} pre-assigned {len(task_assignments[i + 1])} tasks (legacy mode)")
        else:
            raise ValueError(f"Unknown mode: {mode}")

        prompt_server.distributed_pending_tile_jobs[multi_job_id] = job_data

# Note: legacy task distribution and queue pull helpers removed

async def _drain_results_queue(multi_job_id):
    """Drain pending results from queue and update completed_tasks. Returns count drained.

    Uses non-blocking get_nowait to avoid await timeouts and reduce latency.
    """
    prompt_server = ensure_tile_jobs_initialized()
    async with prompt_server.distributed_tile_jobs_lock:
        job_data = prompt_server.distributed_pending_tile_jobs.get(multi_job_id)
        if not job_data or JOB_QUEUE not in job_data or JOB_COMPLETED_TASKS not in job_data:
            return 0
        q = job_data[JOB_QUEUE]
        completed_tasks = job_data[JOB_COMPLETED_TASKS]

        collected = 0
        while True:
            try:
                result = q.get_nowait()
            except asyncio.QueueEmpty:
                break

            worker_id = result['worker_id']
            is_last = result.get('is_last', False)

            if 'image_idx' in result and 'image' in result:
                task_id = result['image_idx']
                if task_id not in completed_tasks:
                    completed_tasks[task_id] = result['image']
                    collected += 1
            elif 'tiles' in result:
                for tile_data in result['tiles']:
                    task_id = tile_data.get('global_idx', tile_data['tile_idx'])
                    if task_id not in completed_tasks:
                        completed_tasks[task_id] = tile_data
                        collected += 1
            elif 'tensor' in result and 'tile_idx' in result:  # Single tile backward compat
                task_id = result.get('global_idx', result['tile_idx'])
                if task_id not in completed_tasks:
                    completed_tasks[task_id] = {
                        'tensor': result['tensor'],
                        'tile_idx': result['tile_idx'],
                        'x': result['x'],
                        'y': result['y'],
                        'extracted_width': result['extracted_width'],
                        'extracted_height': result['extracted_height'],
                        'padding': result['padding'],
                        'batch_idx': result.get('batch_idx', 0),
                        'global_idx': task_id
                    }
                    collected += 1

            if is_last:
                # Track worker completion
                if worker_id in job_data[JOB_WORKER_STATUS]:
                    del job_data[JOB_WORKER_STATUS][worker_id]

        return collected

async def _check_and_requeue_timed_out_workers(multi_job_id, total_tasks):
    """Check timed out workers and requeue their tasks. Returns requeued count."""
    prompt_server = ensure_tile_jobs_initialized()
    async with prompt_server.distributed_tile_jobs_lock:
        job_data = prompt_server.distributed_pending_tile_jobs.get(multi_job_id)
        if not job_data:
            return 0

        current_time = time.time()
        requeued_count = 0
        completed_tasks = job_data.get(JOB_COMPLETED_TASKS, {})

        # Allow override via config setting 'worker_timeout_seconds'
        cfg = load_config()
        hb_timeout = int(cfg.get('settings', {}).get('worker_timeout_seconds', HEARTBEAT_TIMEOUT))

        for worker, last_heartbeat in list(job_data.get(JOB_WORKER_STATUS, {}).items()):
            age = current_time - last_heartbeat
            debug_log(f"Timeout check: worker={worker} age={age:.1f}s threshold={hb_timeout}s")
            if age > hb_timeout:
                # Busy-only grace policy: require positive signal from worker (/prompt)
                # We also log assignment state for diagnostics but do not grace on it alone.
                assigned = job_data.get(JOB_ASSIGNED_TO_WORKERS, {}).get(worker, [])
                incomplete_assigned = 0
                try:
                    if assigned:
                        batched_static = bool(job_data.get('batched_static', False))
                        if batched_static:
                            num_tiles_per_image = job_data.get(JOB_NUM_TILES_PER_IMAGE, 1)
                            batch_size = job_data.get(JOB_BATCH_SIZE, 1)
                            for task_id in assigned:
                                for b in range(batch_size):
                                    gidx = b * num_tiles_per_image + task_id
                                    if gidx not in completed_tasks:
                                        incomplete_assigned += 1
                                        break
                        else:
                            for task_id in assigned:
                                if task_id not in completed_tasks:
                                    incomplete_assigned += 1
                    debug_log(f"Assigned diagnostics: total_assigned={len(assigned)} incomplete_assigned={incomplete_assigned}")
                except Exception as e:
                    debug_log(f"Assigned diagnostics failed for worker {worker}: {e}")

                busy = False
                probe_status = None
                probe_queue = None
                try:
                    cfg_workers = load_config().get('workers', [])
                    wrec = next((w for w in cfg_workers if str(w.get('id')) == str(worker)), None)
                    if wrec:
                        host = wrec.get('host') or 'localhost'
                        port = int(wrec.get('port', 8188))
                        url = f"http://{host}:{port}/prompt"
                        debug_log(f"Probing worker {worker} at {url}")
                        session = await get_client_session()
                        async with session.get(url, timeout=ClientTimeout(total=2.0)) as resp:
                            probe_status = resp.status
                            if resp.status == 200:
                                try:
                                    payload = await resp.json()
                                    probe_queue = int(payload.get('exec_info', {}).get('queue_remaining', 0))
                                except Exception:
                                    probe_queue = 0
                                busy = probe_queue is not None and probe_queue > 0
                except Exception as e:
                    debug_log(f"Probe failed for worker {worker}: {e}")
                finally:
                    debug_log(f"Probe diagnostics: http_status={probe_status} queue_remaining={probe_queue}")

                if busy:
                    job_data[JOB_WORKER_STATUS][worker] = current_time
                    debug_log(f"Heartbeat grace: worker {worker} busy via probe; skipping requeue")
                    continue

                log(f"Worker {worker} timed out")
                for task_id in job_data.get(JOB_ASSIGNED_TO_WORKERS, {}).get(worker, []):
                    # If batched_static, task_id is a tile_idx; consider it complete only if
                    # all corresponding global_idx entries are present in completed_tasks.
                    batched_static = bool(job_data.get('batched_static', False))
                    if batched_static:
                        num_tiles_per_image = job_data.get(JOB_NUM_TILES_PER_IMAGE, 1)
                        batch_size = job_data.get(JOB_BATCH_SIZE, 1)
                        # Check all global indices for this tile across the batch
                        all_done = True
                        for b in range(batch_size):
                            gidx = b * num_tiles_per_image + task_id
                            if gidx not in completed_tasks:
                                all_done = False
                                break
                        if not all_done:
                            await job_data[JOB_PENDING_TASKS].put(task_id)
                            requeued_count += 1
                    else:
                        # Legacy/global-idx mode: task_id is a global index key
                        if task_id not in completed_tasks:
                            await job_data[JOB_PENDING_TASKS].put(task_id)
                            requeued_count += 1
                if JOB_WORKER_STATUS in job_data:
                    del job_data[JOB_WORKER_STATUS][worker]
                if JOB_ASSIGNED_TO_WORKERS in job_data:
                    job_data[JOB_ASSIGNED_TO_WORKERS][worker] = []

        return requeued_count

async def _get_completed_count(multi_job_id):
    """Get count of completed tasks."""
    prompt_server = ensure_tile_jobs_initialized()
    async with prompt_server.distributed_tile_jobs_lock:
        job_data = prompt_server.distributed_pending_tile_jobs.get(multi_job_id)
        if job_data and JOB_COMPLETED_TASKS in job_data:
            return len(job_data[JOB_COMPLETED_TASKS])
        return 0

async def _mark_task_completed(multi_job_id, task_id, result):
    """Mark a task as completed."""
    prompt_server = ensure_tile_jobs_initialized()
    async with prompt_server.distributed_tile_jobs_lock:
        job_data = prompt_server.distributed_pending_tile_jobs.get(multi_job_id)
        if job_data and JOB_COMPLETED_TASKS in job_data:
            job_data[JOB_COMPLETED_TASKS][task_id] = result

async def _send_heartbeat_to_master(multi_job_id, master_url, worker_id):
    """Send heartbeat to master."""
    try:
        data = {'multi_job_id': multi_job_id, 'worker_id': str(worker_id)}
        session = await get_client_session()
        url = f"{master_url}/distributed/heartbeat"
        async with session.post(url, json=data) as response:
            response.raise_for_status()
    except Exception as e:
        debug_log(f"Heartbeat failed: {e}")

async def _cleanup_job(multi_job_id):
    """Cleanup the job data."""
    prompt_server = ensure_tile_jobs_initialized()
    async with prompt_server.distributed_tile_jobs_lock:
        if multi_job_id in prompt_server.distributed_pending_tile_jobs:
            del prompt_server.distributed_pending_tile_jobs[multi_job_id]
            debug_log(f"Cleaned up job {multi_job_id}")

# API Endpoints (generalized)

@server.PromptServer.instance.routes.post("/distributed/heartbeat")
async def heartbeat_endpoint(request):
    try:
        data = await request.json()
        worker_id = data.get('worker_id')
        multi_job_id = data.get('multi_job_id')
        
        if not worker_id or not multi_job_id:
            return await handle_api_error(request, "Missing worker_id or multi_job_id", 400)
        
        prompt_server = ensure_tile_jobs_initialized()
        async with prompt_server.distributed_tile_jobs_lock:
            if multi_job_id in prompt_server.distributed_pending_tile_jobs:
                job_data = prompt_server.distributed_pending_tile_jobs[multi_job_id]
                if JOB_WORKER_STATUS in job_data:
                    job_data[JOB_WORKER_STATUS][worker_id] = time.time()
                    debug_log(f"Heartbeat from worker {worker_id}")
                    return web.json_response({"status": "success"})
                else:
                    return await handle_api_error(request, "Worker status tracking not available", 400)
            else:
                return await handle_api_error(request, "Job not found", 404)
    except Exception as e:
        return await handle_api_error(request, e, 500)

@server.PromptServer.instance.routes.post("/distributed/submit_tiles")
async def submit_tiles_endpoint(request):
    """Endpoint for workers to submit processed tiles in static mode."""
    try:
        content_length = request.headers.get('content-length')
        if content_length and int(content_length) > MAX_PAYLOAD_SIZE:
            return await handle_api_error(request, f"Payload too large: {content_length} bytes", 413)
        
        data = await request.post()
        multi_job_id = data.get('multi_job_id')
        worker_id = data.get('worker_id')
        is_last = data.get('is_last', 'False').lower() == 'true'
        
        if multi_job_id is None or worker_id is None:
            return await handle_api_error(request, "Missing multi_job_id or worker_id", 400)

        prompt_server = ensure_tile_jobs_initialized()
        
        batch_size = int(data.get('batch_size', 0))
        tiles = []
        
        # Handle completion signal
        if batch_size == 0 and is_last:
            async with prompt_server.distributed_tile_jobs_lock:
                if multi_job_id in prompt_server.distributed_pending_tile_jobs:
                    job_data = prompt_server.distributed_pending_tile_jobs[multi_job_id]
                    if JOB_MODE in job_data and job_data[JOB_MODE] != 'static':
                        return await handle_api_error(request, "Job not configured for tile submissions", 400)
                    if JOB_QUEUE in job_data:
                        await job_data[JOB_QUEUE].put({
                            'worker_id': worker_id,
                            'is_last': True,
                            'tiles': []
                        })
                        debug_log(f"Received completion signal from worker {worker_id}")
                        return web.json_response({"status": "success"})
        
        try:
            tiles = _parse_tiles_from_form(data)
        except ValueError as e:
            return await handle_api_error(request, str(e), 400)

        # Submit tiles to queue
        async with prompt_server.distributed_tile_jobs_lock:
            if multi_job_id in prompt_server.distributed_pending_tile_jobs:
                job_data = prompt_server.distributed_pending_tile_jobs[multi_job_id]
                if JOB_MODE in job_data and job_data[JOB_MODE] != 'static':
                    return await handle_api_error(request, "Job not configured for tile submissions", 400)
                
                q = job_data[JOB_QUEUE]
                if batch_size > 0 or len(tiles) > 0:
                    await q.put({
                        'worker_id': worker_id,
                        'tiles': tiles,
                        'is_last': is_last
                    })
                    debug_log(f"Received {len(tiles)} tiles from worker {worker_id} (is_last={is_last})")
                else:
                    await q.put({
                        'worker_id': worker_id,
                        'is_last': True,
                        'tiles': []
                    })
                    
                return web.json_response({"status": "success"})
            else:
                return await handle_api_error(request, "Job not found", 404)
    except Exception as e:
        return await handle_api_error(request, e, 500)

@server.PromptServer.instance.routes.post("/distributed/submit_image")
async def submit_image_endpoint(request):
    """Endpoint for workers to submit processed images in dynamic mode."""
    try:
        content_length = request.headers.get('content-length')
        if content_length and int(content_length) > MAX_PAYLOAD_SIZE:
            return await handle_api_error(request, f"Payload too large: {content_length} bytes", 413)
        
        data = await request.post()
        multi_job_id = data.get('multi_job_id')
        worker_id = data.get('worker_id')
        is_last = data.get('is_last', 'False').lower() == 'true'
        
        if multi_job_id is None or worker_id is None:
            return await handle_api_error(request, "Missing multi_job_id or worker_id", 400)

        prompt_server = ensure_tile_jobs_initialized()
        
        # Handle image submission
        if 'full_image' in data and 'image_idx' in data:
            image_idx = int(data.get('image_idx'))
            img_data = data['full_image'].file.read()
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
            
            debug_log(f"Received full image {image_idx} from worker {worker_id}")
            
            async with prompt_server.distributed_tile_jobs_lock:
                if multi_job_id in prompt_server.distributed_pending_tile_jobs:
                    job_data = prompt_server.distributed_pending_tile_jobs[multi_job_id]
                    if JOB_MODE in job_data and job_data[JOB_MODE] != 'dynamic':
                        return await handle_api_error(request, "Job not configured for image submissions", 400)
                    if JOB_QUEUE in job_data:
                        await job_data[JOB_QUEUE].put({
                            'worker_id': worker_id,
                            'image_idx': image_idx,
                            'image': img,
                            'is_last': is_last
                        })
                        return web.json_response({"status": "success"})
        
        # Handle completion signal (no image data)
        elif is_last:
            async with prompt_server.distributed_tile_jobs_lock:
                if multi_job_id in prompt_server.distributed_pending_tile_jobs:
                    job_data = prompt_server.distributed_pending_tile_jobs[multi_job_id]
                    if JOB_MODE in job_data and job_data[JOB_MODE] != 'dynamic':
                        return await handle_api_error(request, "Job not configured for image submissions", 400)
                    if JOB_QUEUE in job_data:
                        await job_data[JOB_QUEUE].put({
                            'worker_id': worker_id,
                            'is_last': True,
                            'tiles': []  # For compatibility
                        })
                        debug_log(f"Received completion signal from worker {worker_id}")
                        return web.json_response({"status": "success"})
        else:
            return await handle_api_error(request, "Missing image data or invalid request", 400)
            
        return await handle_api_error(request, "Job not found", 404)
    except Exception as e:
        return await handle_api_error(request, e, 500)

# Note: Removed legacy /distributed/tile_complete endpoint. Use /distributed/submit_tiles.



# Helper functions for shallow copying conditioning without duplicating models
def clone_control_chain(control, clone_hint=True):
    """Shallow copy the ControlNet chain, optionally cloning hints but sharing models."""
    if control is None:
        return None
    new_control = copy.copy(control)  # Shallow copy (shares model)
    if clone_hint and hasattr(control, 'cond_hint_original'):
        hint = getattr(control, 'cond_hint_original', None)
        new_control.cond_hint_original = hint.clone() if hint is not None else None
    if hasattr(control, 'previous_controlnet'):
        new_control.previous_controlnet = clone_control_chain(control.previous_controlnet, clone_hint)
    return new_control

def clone_conditioning(cond_list, clone_hints=True):
    """Clone conditioning without duplicating ControlNet models."""
    new_cond = []
    for emb, cond_dict in cond_list:
        new_emb = emb.clone() if emb is not None else None
        new_dict = cond_dict.copy()
        if 'control' in new_dict:
            new_dict['control'] = clone_control_chain(new_dict['control'], clone_hints)
        if 'mask' in new_dict:
            if new_dict['mask'] is not None:
                new_dict['mask'] = new_dict['mask'].clone()
        # Handle other potential fields if needed
        if 'pooled_output' in new_dict:
            if new_dict['pooled_output'] is not None:
                new_dict['pooled_output'] = new_dict['pooled_output'].clone()
        if 'area' in new_dict:
            new_dict['area'] = new_dict['area'][:]  # Shallow copy list/tuple
        new_cond.append([new_emb, new_dict])
    return new_cond

# Helper function to ensure persistent state is initialized
def ensure_tile_jobs_initialized():
    """Ensure tile job storage is initialized on the server instance."""
    prompt_server = server.PromptServer.instance
    if not hasattr(prompt_server, 'distributed_pending_tile_jobs'):
        debug_log("Initializing persistent tile job queue on server instance.")
        prompt_server.distributed_pending_tile_jobs = {}
        prompt_server.distributed_tile_jobs_lock = asyncio.Lock()
        # Final frames storage
        prompt_server.distributed_final_frames = {}
    else:
        # Clean up any legacy queue structures that don't have the 'mode' field
        # (Should be rare after fixes, but keep for safety)
        to_remove = [job_id for job_id, job_data in prompt_server.distributed_pending_tile_jobs.items()
                     if not isinstance(job_data, dict) or 'mode' not in job_data]
        for job_id in to_remove:
            debug_log(f"Removing legacy queue structure for job {job_id}")
            del prompt_server.distributed_pending_tile_jobs[job_id]
        # Ensure final frames storage exists
        if not hasattr(prompt_server, 'distributed_final_frames'):
            prompt_server.distributed_final_frames = {}
    return prompt_server

# Register final frames for distribution to workers
async def register_final_frames(
    multi_job_id: str,
    frames: list,
    enabled_workers: list,
    expected_total: Optional[int] = None,
    append: bool = True,
    seal: Optional[bool] = None,
):
    """
    Accumulate JPEG-encoded final frames for workers.

    Notes:
    - Frames must be bytes. This function APPENDS new chunks by default.
    - Per-worker progress is preserved; new workers are initialized with 0.
    - 'expected_total' can be provided to advertise the final count.
    - If 'seal' is True, the set is marked as complete; if False, unsealed.
      If 'seal' is None, we auto-seal when len(frames) >= expected_total.
    """
    prompt_server = ensure_tile_jobs_initialized()
    async with prompt_server.distributed_tile_jobs_lock:
        if not hasattr(prompt_server, 'distributed_final_frames'):
            prompt_server.distributed_final_frames = {}

        final_jobs = prompt_server.distributed_final_frames

        # Normalize workers to strings
        new_workers = [str(w) for w in (enabled_workers or [])]
        add_count = len(frames) if frames else 0

        job = final_jobs.get(multi_job_id)
        if job is None:
            # Create a new job entry
            job = {
                'frames': list(frames) if frames else [],
                'progress': {w: 0 for w in new_workers},
                'enabled_workers': new_workers,
                'expected_total': int(expected_total) if expected_total is not None else None,
                'sealed': bool(seal) if seal is not None else False,
            }
            final_jobs[multi_job_id] = job
        else:
            # Append or replace frames
            if append:
                if frames:
                    job_frames = job.get('frames')
                    if job_frames is None or not isinstance(job_frames, list):
                        job['frames'] = []
                        job_frames = job['frames']
                    job_frames.extend(frames)
            else:
                job['frames'] = list(frames) if frames else []

            # Merge workers without losing progress
            if 'enabled_workers' not in job or not isinstance(job['enabled_workers'], list):
                job['enabled_workers'] = []
            for w in new_workers:
                if w not in job['enabled_workers']:
                    job['enabled_workers'].append(w)

            prog = job.get('progress')
            if prog is None or not isinstance(prog, dict):
                prog = {}
                job['progress'] = prog
            for w in new_workers:
                if w not in prog:
                    prog[w] = 0

            # Update expected_total if provided
            if expected_total is not None:
                try:
                    job['expected_total'] = int(expected_total)
                except Exception:
                    pass

            # Handle sealing state
            if seal is True:
                job['sealed'] = True
            elif seal is False:
                job['sealed'] = False
            else:
                # Auto-seal if we reached the advertised total
                et = job.get('expected_total')
                if et is not None and isinstance(et, int):
                    if len(job.get('frames', [])) >= et:
                        job['sealed'] = True
                else:
                    # Keep previous sealed state as-is when no total is known
                    job.setdefault('sealed', False)

        total_now = len(final_jobs[multi_job_id].get('frames', []))
        workers_tracked = len(final_jobs[multi_job_id].get('progress', {}))
        debug_log(
            f"Registered +{add_count} finals (total={total_now}) for job {multi_job_id}; "
            f"workers={workers_tracked}, sealed={final_jobs[multi_job_id].get('sealed', False)}, "
            f"expected_total={final_jobs[multi_job_id].get('expected_total')}"
        )


# API Endpoint for tile completion
@server.PromptServer.instance.routes.post("/distributed/request_image")
async def request_image_endpoint(request):
    """Endpoint for workers to request tasks (images in dynamic mode, tiles in static mode)."""
    try:
        data = await request.json()
        worker_id = data.get('worker_id')
        multi_job_id = data.get('multi_job_id')
        
        if not worker_id or not multi_job_id:
            return await handle_api_error(request, "Missing worker_id or multi_job_id", 400)
        
        prompt_server = ensure_tile_jobs_initialized()
        async with prompt_server.distributed_tile_jobs_lock:
            if multi_job_id in prompt_server.distributed_pending_tile_jobs:
                job_data = prompt_server.distributed_pending_tile_jobs[multi_job_id]
                if not isinstance(job_data, dict) or 'mode' not in job_data:
                    return await handle_api_error(request, "Invalid job data structure", 500)
                
                mode = job_data['mode']
                
                # Handle both dynamic and static modes
                if mode == 'dynamic' and 'pending_images' in job_data:
                    pending_queue = job_data['pending_images']
                elif mode == 'static' and JOB_PENDING_TASKS in job_data:
                    pending_queue = job_data[JOB_PENDING_TASKS]
                else:
                    return await handle_api_error(request, "Invalid job configuration", 400)
                
                try:
                    task_idx = await asyncio.wait_for(pending_queue.get(), timeout=0.1)
                    # Track assigned task
                    if 'assigned_to_workers' in job_data and worker_id in job_data['assigned_to_workers']:
                        job_data['assigned_to_workers'][worker_id].append(task_idx)
                    # Update worker heartbeat
                    if 'worker_status' in job_data:
                        job_data['worker_status'][worker_id] = time.time()
                    # Get estimated remaining count
                    remaining = pending_queue.qsize()  # Approximate
                    
                    # Return appropriate response based on mode
                    if mode == 'dynamic':
                        debug_log(f"UltimateSDUpscale API - Assigned image {task_idx} to worker {worker_id}")
                        return web.json_response({"image_idx": task_idx, "estimated_remaining": remaining})
                    else:  # static
                        debug_log(f"UltimateSDUpscale API - Assigned tile {task_idx} to worker {worker_id}")
                        return web.json_response({"tile_idx": task_idx, "estimated_remaining": remaining, "batched_static": job_data.get('batched_static', False)})
                except asyncio.TimeoutError:
                    if mode == 'dynamic':
                        return web.json_response({"image_idx": None})
                    else:
                        return web.json_response({"tile_idx": None})
            else:
                return await handle_api_error(request, "Job not found", 404)
    except Exception as e:
        return await handle_api_error(request, e, 500)

@server.PromptServer.instance.routes.get("/distributed/job_status")
async def job_status_endpoint(request):
    """Endpoint to check if a job is ready."""
    multi_job_id = request.query.get('multi_job_id')
    if not multi_job_id:
        return web.json_response({"ready": False})
    prompt_server = ensure_tile_jobs_initialized()
    async with prompt_server.distributed_tile_jobs_lock:
        job_data = prompt_server.distributed_pending_tile_jobs.get(multi_job_id)
        ready = bool(job_data and isinstance(job_data, dict) and 'queue' in job_data)
        return web.json_response({"ready": ready})

# Endpoint for workers to request final frames from master
# Returns a batch of final frames encoded in base64 JPEG strings.
# Enforces MAX_BATCH frames per response and MAX_PAYLOAD_SIZE limit.
@server.PromptServer.instance.routes.post("/distributed/request_finals")
async def request_finals_endpoint(request):
    """Workers pull final frames. Honors MAX_BATCH/MAX_PAYLOAD_SIZE with base64
    estimation and only returns is_last=True when the job is sealed and the
    worker has consumed all produced frames."""
    try:
        data = await request.json()
        multi_job_id = data.get('multi_job_id')
        worker_id = str(data.get('worker_id', ''))
        if not multi_job_id or not worker_id:
            return await handle_api_error(request, "Missing multi_job_id or worker_id", 400)

        prompt_server = ensure_tile_jobs_initialized()
        async with prompt_server.distributed_tile_jobs_lock:
            final_jobs = getattr(prompt_server, 'distributed_final_frames', None)
            if not final_jobs or multi_job_id not in final_jobs:
                return await handle_api_error(request, "Final frames not ready", 404)

            job = final_jobs[multi_job_id]
            frames = job.get('frames', []) or []
            progress = job.get('progress', {}) or {}
            expected_total = job.get('expected_total')
            sealed = bool(job.get('sealed', False))

            if worker_id not in progress:
                progress[worker_id] = 0

            start_idx = int(progress[worker_id])
            total_frames = len(frames)

            # Base64 size estimator (~4/3 expansion)
            def _b64_size(n: int) -> int:
                return ((int(n) + 2) // 3) * 4

            max_json_bytes = int(MAX_PAYLOAD_SIZE) - (1024 * 1024)  # headroom
            overhead_fixed = 256
            overhead_per_item = 64

            batch = []
            used = 0
            count = 0
            i = start_idx
            while i < total_frames and count < int(MAX_BATCH):
                raw = frames[i]
                est = _b64_size(len(raw)) + overhead_per_item
                if batch and used + est + overhead_fixed > max_json_bytes:
                    break
                if not batch and est + overhead_fixed > max_json_bytes:
                    # single huge item: still send alone
                    batch.append(raw)
                    used += est
                    count += 1
                    i += 1
                    break
                batch.append(raw)
                used += est
                count += 1
                i += 1

            progress[worker_id] = i

            consumed_all = (i >= total_frames)
            is_last = bool(sealed and consumed_all)

            frames_b64 = [base64.b64encode(b).decode('ascii') for b in batch]
            debug_log(
                f"Sending {len(frames_b64)} final frames to worker {worker_id} "
                f"(is_last={is_last}, sealed={sealed}, have={total_frames}, expected_total={expected_total})"
            )
            return web.json_response({"frames": frames_b64, "is_last": is_last})
    except Exception as e:
        return await handle_api_error(request, e, 500)
async def dist_async_wait_for_final_frames(multi_job_id: str,
                                           master_url: str,
                                           worker_id: str,
                                           dtype,
                                           device):
    """
    Poll the master for final frames and assemble them.
    Adds backoff when receiving empty batches with is_last=False to avoid busy loops.
    """
    final_frames = []
    start_time = time.time()
    batch_idx = 0
    total_received = 0
    total_decoded = 0
    empty_batches = 0

    while True:
        # Heartbeat is best-effort
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
                    frames_b64 = data.get('frames', []) or []
                    is_last = bool(data.get('is_last', False))

                    # Track inbound limits
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
                        empty_batches = 0  # reset on progress
                        # Parallel JPEG decode
                        try:
                            executor = get_jpeg_executor()
                            arrs = list(executor.map(_decode_base64_to_np, frames_b64))
                        except Exception:
                            arrs = [_decode_base64_to_np(fb64) for fb64 in frames_b64]
                        # To tensors
                        for j, arr in enumerate(arrs):
                            try:
                                tensor = torch.from_numpy(arr).to(dtype=dtype).to(device)
                                final_frames.append(tensor)
                                total_decoded += 1
                            except Exception as e:
                                _vd_log(f"Decode failed at batch={batch_idx}, idx={j}: {type(e).__name__}: {e}")
                    else:
                        # No new frames; backoff to avoid tight loop
                        empty_batches += 1
                        await asyncio.sleep(min(1.0, 0.05 * empty_batches))

                    if is_last:
                        _vd_log(f"Completed: batches={batch_idx+1}, received={total_received}, decoded={total_decoded}")
                        break

                elif resp.status == 404:
                    _vd_log("Master not ready (404), retrying…")
                    await asyncio.sleep(1.0)
                else:
                    _vd_log(f"Unexpected status={resp.status}, retrying…")
                    await asyncio.sleep(1.0)

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
