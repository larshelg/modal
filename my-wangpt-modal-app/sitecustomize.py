"""Runtime compatibility patches for the pinned WanGP dependencies.

Python imports ``sitecustomize`` automatically during interpreter startup when
it is present on ``PYTHONPATH``.  MCP 1.10.1 accepts a permanent GET/SSE stream
even when FastMCP is configured for stateless JSON responses.  That open HTTP
request prevents Modal's web-server function from becoming idle.
"""

from http import HTTPStatus

from mcp.server.streamable_http import StreamableHTTPServerTransport
from starlette.responses import Response


async def _reject_standalone_sse_get(self, request, send) -> None:
    """Advertise that this stateless MCP endpoint supports POST only."""
    response = Response(
        status_code=HTTPStatus.METHOD_NOT_ALLOWED,
        headers={"Allow": "POST"},
    )
    await response(request.scope, request.receive, send)


StreamableHTTPServerTransport._handle_get_request = _reject_standalone_sse_get
