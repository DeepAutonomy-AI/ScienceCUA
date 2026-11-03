"""Bounded background transport validation; the Agent still reviews one result at a time."""
from __future__ import annotations

import queue
import threading
import time
from typing import Any, Callable

from .source_discovery_contracts import canonical_candidate_url


class SourceValidationQueue:
    def __init__(self, inspect: Callable[[str], dict[str, Any]], *, concurrency: int = 3,
                 deadline: float | None = None):
        self.inspect = inspect
        self.concurrency = min(3, max(1, concurrency))
        self.deadline = deadline
        self._condition = threading.Condition()
        self._pending: queue.Queue[str] = queue.Queue()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._workers: list[threading.Thread] = []
        self.closed = threading.Event()

    def start(self, urls: list[str]) -> int:
        # Only search results enter automatically: <=12 per search, <=3 searches.
        # Links found inside HTML are inspected explicitly, preventing recursive crawling.
        with self._condition:
            if self.closed.is_set():
                return 0
            added = 0
            for url in urls[:12]:
                key = canonical_candidate_url(url)
                if key in self._jobs or len(self._jobs) >= 36:
                    continue
                self._jobs[key] = {'url': url, 'state': 'queued', 'reviewed': False}
                self._pending.put(key)
                added += 1
            if added and not self._workers:
                for index in range(self.concurrency):
                    worker = threading.Thread(target=self._work, name=f'source-validation-{index}', daemon=True)
                    self._workers.append(worker)
                    worker.start()
            return added

    def _work(self) -> None:
        while not self.closed.is_set():
            try:
                key = self._pending.get(timeout=.1)
            except queue.Empty:
                continue
            with self._condition:
                job = self._jobs[key]
                if self.closed.is_set() or job['state'] != 'queued':
                    continue
                job['state'] = 'running'
            try:
                result = self.inspect(job['url'])
            except Exception:
                # No exception object, headers or credentials enter the model's result.
                result = {'status': 'failed', 'url': job['url'], 'error': 'validation_interrupted'}
            with self._condition:
                job.update(state='ready', result=result, completed_at=time.monotonic())
                self._condition.notify_all()

    def contains(self, url: str) -> bool:
        with self._condition:
            return canonical_candidate_url(url) in self._jobs

    def review(self, url: str, *, skip: bool = False) -> None:
        with self._condition:
            job = self._jobs.get(canonical_candidate_url(url))
            if job:
                job['reviewed'] = True
                if skip and job['state'] == 'queued':
                    job['state'] = 'skipped'
                self._condition.notify_all()

    def counts(self) -> dict[str, int]:
        with self._condition:
            return {'total': len(self._jobs),
                    'active': sum(job['state'] in {'queued', 'running'} for job in self._jobs.values()),
                    'pending': sum(job['state'] in {'queued', 'running'} or not job['reviewed'] for job in self._jobs.values()),
                    'unreviewed': sum(not job['reviewed'] for job in self._jobs.values())}

    def take(self, *, url: str | None = None, wait_seconds: float = 30) -> dict[str, Any] | None:
        until = time.monotonic() + wait_seconds
        if self.deadline is not None:
            until = min(until, self.deadline)
        key = canonical_candidate_url(url) if url else None
        with self._condition:
            while not self.closed.is_set():
                ready = [job for candidate, job in self._jobs.items()
                         if job['state'] == 'ready' and (candidate == key if key else not job['reviewed'])]
                if ready:
                    job = min(ready, key=lambda item: item['completed_at'])
                    job['reviewed'] = True
                    return job['result']
                remaining = until - time.monotonic()
                if remaining <= 0 or not any(job['state'] in {'queued', 'running'} for job in self._jobs.values()):
                    return None
                self._condition.wait(timeout=min(.1, remaining))
        return None

    def close(self) -> None:
        self.closed.set()
        with self._condition:
            self._condition.notify_all()
