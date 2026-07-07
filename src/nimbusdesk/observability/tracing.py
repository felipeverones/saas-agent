"""OpenTelemetry setup + the one span helper the whole codebase uses.

WHY OTel AND NOT A VENDOR SDK (ADR-10)
Instrumentation outlives backends: spans emitted here are standard OTLP, so
the viewer is swappable (Arize Phoenix locally today; Datadog/Langfuse/
whatever tomorrow) with zero changes to instrumented code.

THE NO-OP TRICK (why instrumentation can be unconditional)
Until `setup_tracing()` installs a real TracerProvider, `trace.get_tracer()`
returns a no-op implementation — spans cost near-zero and export nowhere.
So application code is ALWAYS instrumented; whether telemetry flows is purely
a deployment decision (TRACING_ENABLED=1). No `if tracing:` litter anywhere.

WHAT WE TRACE (the tree one request produces)
    graph.<node>              one span per LangGraph node
      agent.loop              a specialist's inner ReAct run
        llm.complete          every model call (model, token usage)
        tool.<name>           every tool execution (args, error flag)
      rag.retrieve / rerank   the retrieval funnel
The point: when an answer is wrong, the failing STEP is visible in the tree —
logs interleave and lie; traces keep causality.
"""

from contextlib import contextmanager
from typing import Iterator

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span

_TRACER_NAME = "nimbusdesk"


def setup_tracing(endpoint: str, service_name: str = "nimbusdesk") -> None:
    """Install a real provider exporting OTLP/gRPC (Phoenix listens on 4317).
    Call at most once, from the composition root, only when tracing is on."""
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)


@contextmanager
def span(name: str, **attributes: object) -> Iterator[Span]:
    """One import, one line per instrumented step:
        with span("rag.retrieve", query=q, k=k) as s: ...
    None attributes are skipped; values are stringified conservatively."""
    tracer = trace.get_tracer(_TRACER_NAME)
    with tracer.start_as_current_span(name) as current:
        for key, value in attributes.items():
            if value is None:
                continue
            if isinstance(value, (bool, int, float, str)):
                current.set_attribute(key, value)
            else:
                current.set_attribute(key, str(value)[:500])
        yield current
