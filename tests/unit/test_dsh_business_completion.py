from __future__ import annotations

import asyncio
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from app.runtime import AgentInvocation, AgentRuntimeBusinessCompleted
from app.runtime.dsh_client import DshAgentRuntime, DshBridgeConfig, _SessionWorker


class _InputPipe:
    def __init__(self) -> None:
        self.written = asyncio.Event()

    def write(self, data: bytes) -> None:
        self.written.set()

    async def drain(self) -> None:
        return None


class _WorkerProcess:
    def __init__(self) -> None:
        self.stdin = _InputPipe()
        self.stdout = asyncio.StreamReader()
        self.stderr = asyncio.StreamReader()
        self.returncode: int | None = None
        self.ended = asyncio.Event()

    def terminate(self) -> None:
        self.returncode = 0
        self.stdout.feed_eof()
        self.stderr.feed_eof()
        self.ended.set()

    async def wait(self) -> int:
        await self.ended.wait()
        return 0


class DshBusinessCompletionTests(unittest.IsolatedAsyncioTestCase):
    async def test_business_stop_releases_runtime_slot_with_distinct_reason(self):
        runtime = DshAgentRuntime(DshBridgeConfig(workspace=Path.cwd()))
        process = _WorkerProcess()
        session_id = "child_business_complete"
        worker = _SessionWorker(process=process, route_signature=())
        runtime._workers[session_id] = worker

        async def get_worker(_session_id: str, _payload: dict):
            return worker

        runtime._get_or_start_worker = get_worker
        invocation = AgentInvocation(
            run_id="run_business_complete", session_id=session_id, role="root",
            prompt="test", model={"model": "fake"},
        )
        turn = asyncio.create_task(runtime.run(invocation))
        await asyncio.wait_for(process.stdin.written.wait(), timeout=1.0)
        self.assertTrue(await runtime.stop_completed_session(session_id))
        with self.assertRaises(AgentRuntimeBusinessCompleted):
            await asyncio.wait_for(turn, timeout=1.0)
        self.assertFalse(runtime._run_semaphore.locked())
        self.assertNotIn(session_id, runtime._completed_sessions)

    @pytest.mark.integration
    async def test_real_sdk_business_stop_closes_owned_workers_and_releases_slot(self):
        import tempfile
        from app.runtime import AgentRuntimeBusinessCompleted

        requested = threading.Event()
        release = threading.Event()
        received = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                received.append(self.path)
                requested.set()
                release.wait(timeout=15)
                # Model work is deliberately held while the host completes its stage.

            def log_message(self, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        with tempfile.TemporaryDirectory() as directory:
            runtime = DshAgentRuntime(DshBridgeConfig(workspace=Path.cwd(), dsh_home=str(Path(directory) / "dsh")))
            session = "child_native_business_stop"
            turn = asyncio.create_task(runtime.run(AgentInvocation(
                run_id="run_test", session_id=session, role="root", prompt="Test request.",
                model={"model": "model-test", "base_url": f"http://127.0.0.1:{server.server_port}/v1",
                       "api_key": "fake-test-key", "api_protocol": "chat_completions"},
            )))
            try:
                self.assertTrue(await asyncio.to_thread(requested.wait, 10))
                self.assertTrue(await asyncio.wait_for(runtime.stop_completed_session(session), 5))
                with self.assertRaises(AgentRuntimeBusinessCompleted):
                    await asyncio.wait_for(turn, 3)
                self.assertFalse(runtime._run_semaphore.locked())
                self.assertNotIn(session, runtime._workers)
                self.assertEqual(received, ["/v1/chat/completions"])
            finally:
                turn.cancel()
                await asyncio.gather(turn, return_exceptions=True)
                await runtime.close_session(session)
                release.set()
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)
