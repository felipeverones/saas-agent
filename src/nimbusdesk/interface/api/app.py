"""The HTTP surface: /health, /chat (SSE), /approvals/{thread}.

STREAMING DESIGN (SSE)
Agent runs take seconds; users tolerate latency they can SEE. /chat streams
Server-Sent Events as the graph progresses:

    event: node               {"node": "triage"}          per node entered
    event: approval_required  {payload..., "thread_id"}   run paused (HITL)
    event: answer             {answer, resolved_by, ...}  turn finished

Node-level (not token-level) streaming is a deliberate scope cut: token
streaming needs a streaming method on the LLM port end-to-end; node progress
already gives the "it's alive and working" UX at a fraction of the surface.

HITL OVER HTTP: when a run interrupts for approval, the stream ends with
`approval_required` and the state sits checkpointed. A separate client (an
operator tool — or our CLI playing both roles) later POSTs the decision to
/approvals/{thread_id}, which resumes the SAME thread and returns the final
answer. Two independent HTTP calls, possibly days apart: the checkpoint is
what connects them.

TESTABILITY: create_app() accepts a prebuilt AppRuntime so tests inject a
graph wired with fakes; production builds the real one in the lifespan.
"""

import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel, Field

from nimbusdesk.agents.graph import build_turn_input
from nimbusdesk.agents.state import SupportState
from nimbusdesk.infrastructure.settings import get_settings
from nimbusdesk.interface.wiring import AppRuntime
from nimbusdesk.observability.cost import estimate_usd

logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    email: str | None = None
    thread_id: str = "default"


class ApprovalRequest(BaseModel):
    approved: bool
    note: str = ""


def create_app(runtime: AppRuntime | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Heavy composition (models, DB connections) belongs in the lifespan,
        # not at import time — uvicorn workers import the module before forking.
        app.state.runtime, app.state.startup_error = _build_production_runtime()
        yield

    if runtime is None:
        app = FastAPI(title="NimbusDesk Support API", lifespan=lifespan)
    else:
        # Injected runtime (tests): no lifespan needed, state is ready now.
        app = FastAPI(title="NimbusDesk Support API")
        app.state.runtime = runtime
        app.state.startup_error = None

    @app.get("/health")
    def health() -> dict:
        if app.state.startup_error:
            return {"status": "degraded", "detail": app.state.startup_error}
        return {"status": "ok"}

    @app.post("/chat")
    def chat(request: ChatRequest) -> StreamingResponse:
        rt = _require_runtime(app)
        return StreamingResponse(
            _stream_turn(rt, request), media_type="text/event-stream"
        )

    @app.post("/approvals/{thread_id}")
    def approve(thread_id: str, decision: ApprovalRequest) -> dict:
        """Resume a paused run with the human's verdict. Fail-closed shape:
        only an explicit approved=true executes the pending action."""
        rt = _require_runtime(app)
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = rt.graph.get_state(config)
        if not snapshot.next:  # nothing paused on this thread
            raise HTTPException(status_code=409, detail="no pending approval on this thread")
        result = rt.graph.invoke(
            Command(resume={"approved": decision.approved, "note": decision.note}),
            config=config,
        )
        return _answer_payload(rt, SupportState.model_validate(result))

    return app


def _build_production_runtime() -> tuple[AppRuntime | None, str | None]:
    """Startup is allowed to fail SOFT (health reports degraded) so the
    container comes up and explains itself instead of crash-looping when the
    API key is missing — friendlier first-run experience, same information."""
    from nimbusdesk.interface.wiring import (
        build_runtime,
        ensure_knowledge_base,
        maybe_enable_tracing,
    )

    settings = get_settings()
    try:
        maybe_enable_tracing(settings)
        ensure_knowledge_base(settings)
        return build_runtime(settings), None
    except Exception as error:
        logger.exception("startup failed")
        return None, f"{type(error).__name__}: {error}"


def _require_runtime(app: FastAPI) -> AppRuntime:
    if app.state.runtime is None:
        raise HTTPException(status_code=503, detail=app.state.startup_error)
    return app.state.runtime


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _stream_turn(rt: AppRuntime, request: ChatRequest):
    """Sync generator (FastAPI iterates it off the event loop): stream node
    progress, then either the final answer or an approval_required pause."""
    config = {"configurable": {"thread_id": request.thread_id}}
    inputs = build_turn_input(request.message, request.thread_id, request.email)
    try:
        for update in rt.graph.stream(inputs, config=config, stream_mode="updates"):
            for node_name, _ in update.items():
                if node_name == "__interrupt__":
                    continue
                yield _sse("node", {"node": node_name})

        snapshot = rt.graph.get_state(config)
        if snapshot.next:  # paused mid-run: the interrupt is waiting
            interrupt = snapshot.interrupts[0] if snapshot.interrupts else None
            payload = dict(interrupt.value) if interrupt else {}
            payload["thread_id"] = request.thread_id
            yield _sse("approval_required", payload)
            return

        state = SupportState.model_validate(snapshot.values)
        yield _sse("answer", _answer_payload(rt, state))
    except Exception as error:  # stream errors must be events, not half-closed sockets
        logger.exception("chat turn failed")
        yield _sse("error", {"detail": f"{type(error).__name__}: {error}"})


def _answer_payload(rt: AppRuntime, state: SupportState) -> dict:
    settings = rt.settings
    cost = estimate_usd(
        settings.nimbus_model_fast, rt.fast.input_tokens, rt.fast.output_tokens
    ) + estimate_usd(
        settings.nimbus_model_strong, rt.strong.input_tokens, rt.strong.output_tokens
    )
    return {
        "answer": state.final_answer,
        "resolved_by": state.resolved_by,
        "escalated": state.escalated,
        "grounded_flags": state.input_flags,
        "triage": state.triage.model_dump(mode="json") if state.triage else None,
        "turn_index": state.turn_index,
        # Process-lifetime totals (not per-turn): good enough for a dev
        # dashboard; per-request attribution arrives with per-request trackers.
        "session_est_cost_usd": round(cost, 4),
    }
