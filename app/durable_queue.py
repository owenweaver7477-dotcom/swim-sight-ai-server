"""Optional Redis Streams execution for crash-recoverable video jobs.

The existing in-process background task remains the default. When
ENABLE_DURABLE_QUEUE=true and REDIS_URL is configured, requests are written to
a Redis Stream and consumed through a consumer group. Unacknowledged messages
are reclaimed after a lease timeout when a worker restarts.

The signed URL is present only in the private queue payload required to process
the video. It is never copied into job status, logs, callbacks, or reports.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import uuid
from typing import Any, Awaitable, Callable, Dict, Optional

from app.models import VideoProcessingRequest


logger = logging.getLogger(__name__)
_TRUTHY = {"1", "true", "yes", "on"}
TERMINAL_STATUSES = {"completed", "manual_review_recommended", "error"}


def durable_queue_requested(env: Optional[Dict[str, str]] = None) -> bool:
    source = os.environ if env is None else env
    return str(source.get("ENABLE_DURABLE_QUEUE", "false")).strip().lower() in _TRUTHY


class DurableJobQueue:
    def __init__(self) -> None:
        self.requested = durable_queue_requested()
        self.redis_url = os.getenv("REDIS_URL", "").strip()
        self.stream_name = os.getenv("AI_JOB_STREAM", "swim-sight:ai-jobs")
        self.group_name = os.getenv("AI_JOB_GROUP", "swim-sight-workers")
        self.job_ttl_seconds = int(os.getenv("AI_JOB_STATUS_TTL_SECONDS", "86400"))
        self.lease_ms = int(os.getenv("AI_JOB_LEASE_MS", "300000"))
        self.stream_max_length = int(os.getenv("AI_JOB_STREAM_MAX_LENGTH", "1000"))
        self.consumer_name = (
            f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        )
        self._client: Any = None
        self._consumer_task: Optional[asyncio.Task] = None
        self._handler: Optional[Callable[[VideoProcessingRequest, str], Awaitable[None]]] = None

    @property
    def ready(self) -> bool:
        return self._client is not None and self._consumer_task is not None

    async def start(
        self,
        handler: Callable[[VideoProcessingRequest, str], Awaitable[None]],
    ) -> None:
        if not self.requested:
            return
        if not self.redis_url:
            raise RuntimeError("ENABLE_DURABLE_QUEUE requires REDIS_URL")

        try:
            import redis.asyncio as redis
            from redis.exceptions import ResponseError
        except ImportError as error:
            raise RuntimeError(
                "ENABLE_DURABLE_QUEUE requires the redis package"
            ) from error

        self._client = redis.from_url(self.redis_url, decode_responses=True)
        await self._client.ping()
        try:
            await self._client.xgroup_create(
                self.stream_name,
                self.group_name,
                id="0-0",
                mkstream=True,
            )
        except ResponseError as error:
            if "BUSYGROUP" not in str(error):
                raise

        self._handler = handler
        self._consumer_task = asyncio.create_task(self._consume_loop())
        logger.info(
            "Durable AI queue ready: stream=%s group=%s consumer=%s",
            self.stream_name,
            self.group_name,
            self.consumer_name,
        )

    async def stop(self) -> None:
        if self._consumer_task:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass
        self._consumer_task = None
        if self._client:
            await self._client.aclose()
        self._client = None

    async def enqueue(self, request: VideoProcessingRequest, job_id: str) -> bool:
        if not self.ready:
            raise RuntimeError("Durable AI queue is not ready")

        dedupe_key = f"swim-sight:ai-job-dedupe:{job_id}"
        inserted = await self._client.set(
            dedupe_key,
            "queued",
            nx=True,
            ex=self.job_ttl_seconds,
        )
        if not inserted:
            logger.info("[job=%s] Durable queue duplicate suppressed", job_id)
            return False

        try:
            await self._client.xadd(
                self.stream_name,
                {
                    "job_id": job_id,
                    "request": request.model_dump_json(),
                },
                maxlen=self.stream_max_length,
                approximate=True,
            )
        except Exception:
            await self._client.delete(dedupe_key)
            raise
        logger.info("[job=%s] Enqueued for durable processing", job_id)
        return True

    async def persist_job(self, job: Dict[str, Any]) -> None:
        if not self._client or not job.get("job_id"):
            return
        key = f"swim-sight:ai-job-status:{job['job_id']}"
        try:
            await self._client.set(
                key,
                json.dumps(job),
                ex=self.job_ttl_seconds,
            )
        except Exception as error:
            logger.warning("[job=%s] Durable status persistence failed: %s", job["job_id"], error)

    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        if not self._client:
            return None
        try:
            raw = await self._client.get(f"swim-sight:ai-job-status:{job_id}")
        except Exception as error:
            logger.warning("[job=%s] Durable status lookup failed: %s", job_id, error)
            return None
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("[job=%s] Invalid durable job snapshot ignored", job_id)
            return None

    async def _consume_loop(self) -> None:
        assert self._client is not None
        while True:
            try:
                messages = await self._claim_stale_messages()
                if not messages:
                    batches = await self._client.xreadgroup(
                        self.group_name,
                        self.consumer_name,
                        {self.stream_name: ">"},
                        count=1,
                        block=5000,
                    )
                    messages = batches[0][1] if batches else []

                for message_id, fields in messages:
                    await self._handle_message(message_id, fields)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.exception("Durable queue consumer error: %s", error)
                await asyncio.sleep(2)

    async def _claim_stale_messages(self):
        result = await self._client.xautoclaim(
            self.stream_name,
            self.group_name,
            self.consumer_name,
            min_idle_time=self.lease_ms,
            start_id="0-0",
            count=1,
        )
        return result[1] if result and len(result) > 1 else []

    async def _handle_message(self, message_id: str, fields: Dict[str, str]) -> None:
        job_id = fields.get("job_id", "")
        try:
            existing = await self.get_job(job_id)
            if existing and existing.get("status") in TERMINAL_STATUSES:
                await self._acknowledge(message_id)
                return

            request = VideoProcessingRequest.model_validate_json(fields["request"])
            if not self._handler:
                raise RuntimeError("Durable queue handler is not configured")
            await self._handler(request, job_id)
            await self._acknowledge(message_id)
        except Exception as error:
            logger.exception("[job=%s] Durable job attempt failed: %s", job_id, error)

    async def _acknowledge(self, message_id: str) -> None:
        await self._client.xack(self.stream_name, self.group_name, message_id)
        await self._client.xdel(self.stream_name, message_id)
