"""
queue_manager.py
Async in-process job queue and dispatcher for Company-Server OCR service:
- Accepts jobs and returns unique job_id
- Processes jobs asynchronously in background
- Tracks job status (pending -> processing -> completed | failed)
- Delivers completed results to optional webhook_url via HTTP POST
"""

import asyncio
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Dict, Optional
import httpx
from audit_logger import log_audit_event


@dataclass
class JobRecord:
    job_id: str
    doc_type: str
    status: str  # 'pending', 'processing', 'completed', 'failed'
    created_at: str
    updated_at: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    webhook_url: Optional[str] = None
    customer_id: Optional[str] = None


class AsyncJobQueue:
    """In-memory asyncio-based job queue and result store."""

    def __init__(self, max_queue_size: int = 1000):
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self._jobs: Dict[str, JobRecord] = {}
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self):
        """Start the background worker."""
        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())

    async def stop(self):
        """Stop background worker gracefully."""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    def create_job(
        self,
        doc_type: str,
        webhook_url: Optional[str] = None,
        customer_id: Optional[str] = None,
    ) -> JobRecord:
        """Create and register a new job."""
        job_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        record = JobRecord(
            job_id=job_id,
            doc_type=doc_type,
            status="pending",
            created_at=now,
            updated_at=now,
            webhook_url=webhook_url,
            customer_id=customer_id,
        )
        # Bounded in-memory job retention (evict oldest finished jobs when threshold exceeded)
        if len(self._jobs) > 2000:
            terminal_keys = [k for k, j in self._jobs.items() if j.status in ("completed", "failed", "error", "low_confidence")]
            for k in terminal_keys[:200]:
                self._jobs.pop(k, None)

        self._jobs[job_id] = record
        return record

    async def enqueue(self, job_id: str, work_coro_fn: Callable[[], Coroutine[Any, Any, Dict[str, Any]]]):
        """Enqueue task execution for the job."""
        await self._queue.put((job_id, work_coro_fn))

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        """Retrieve job record by ID."""
        return self._jobs.get(job_id)

    async def _worker_loop(self):
        """Continuous background worker loop processing queued OCR tasks."""
        while self._running:
            try:
                job_id, work_coro_fn = await self._queue.get()
                record = self._jobs.get(job_id)
                if not record:
                    self._queue.task_done()
                    continue

                record.status = "processing"
                record.updated_at = datetime.now(timezone.utc).isoformat()

                try:
                    res = await work_coro_fn()
                    record.status = res.get("status", "completed") if res.get("status") in ("completed", "low_confidence", "error") else "completed"
                    record.result = res
                    record.updated_at = datetime.now(timezone.utc).isoformat()

                    # Trigger webhook if specified
                    if record.webhook_url:
                        asyncio.create_task(self._deliver_webhook(record.webhook_url, record))

                except Exception as ex:
                    record.status = "failed"
                    record.error = str(ex)
                    record.updated_at = datetime.now(timezone.utc).isoformat()
                    # Audit failure (without PII)
                    log_audit_event(
                        doc_type=record.doc_type,
                        status="error",
                        confidence=0.0,
                        job_id=job_id,
                        customer_id=record.customer_id,
                        reason="worker_exception",
                    )
                finally:
                    self._queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(0.1)

    async def _deliver_webhook(self, url: str, record: JobRecord):
        """Deliver job outcome to webhook URL."""
        payload = {
            "job_id": record.job_id,
            "doc_type": record.doc_type,
            "status": record.status,
            "created_at": record.created_at,
            "completed_at": record.updated_at,
            "result": record.result,
            "error": record.error,
        }
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(url, json=payload)
        except Exception:
            pass


# Global singleton instance
job_queue = AsyncJobQueue()
