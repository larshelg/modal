# Wan2GP MCP server on Modal

This app runs Wan2GP's streamable-HTTP MCP server on a Modal GPU. Its endpoint
requires Modal proxy authentication.

The image also installs a small `sitecustomize.py` compatibility patch. WanGP
already configures MCP for stateless JSON responses, but its pinned MCP SDK
still accepts an optional permanent GET/SSE stream. The patch returns HTTP 405
for that GET while preserving MCP initialization, tool discovery, and tool
calls over POST, allowing Modal's idle scale-down timer to start.

## Persistent cache

The `wangp-data` Volume is mounted at `/data` and persists:

- Wan2GP checkpoints, outputs, settings, configuration, and LoRAs
- Hugging Face Hub and Transformers caches
- Triton kernels and Torch extensions
- The generic XDG cache

The app uses the named Modal Secret `huggingface-secret`, which must contain an
uppercase `HF_TOKEN` key.

See [LORA_WAN_GUIDE.md](LORA_WAN_GUIDE.md) for instructions on checking LoRA
compatibility, uploading one to the Volume, and activating it in Wan2GP.

## Deploy

The standalone `modal` executable is not currently on this machine's PATH, so
invoke the installed client as a Python module:

```bash
cd my-wangpt-modal-app
python3 -m modal deploy app.py
```

Modal prints the protected endpoint URL after deployment. Requests must include
the Modal proxy-auth headers described in the Modal dashboard for the endpoint.

## Connect from Codex

The project-scoped `.codex/config.toml` connects to the `/mcp` route and reads
the proxy headers from `MODAL_KEY` and `MODAL_SECRET`. Start Codex from a shell
where the existing environment file has been loaded:

```bash
cd my-wangpt-modal-app
set -a
source ../my-comfy-modal-app/.env
set +a
codex
```

In the new Codex session, use `/mcp` to verify that `wangp` is connected.

## Connect from Cursor

The project-scoped `.cursor/mcp.json` connects Cursor to the same protected MCP
endpoint. Use the trailing-slash URL (`/mcp/`) — Modal redirects bare `/mcp` to
`http://.../mcp/`, which breaks HTTP MCP clients that follow redirects. Start
Cursor from a shell where the proxy credentials are available:

```bash
cd my-wangpt-modal-app
set -a
source ../my-comfy-modal-app/.env
set +a
cursor .
```

Cursor must inherit `MODAL_KEY` and `MODAL_SECRET`; the MCP configuration reads
them without storing either secret in the repository. In Cursor, open
**Settings > Tools & MCP** to confirm that `wangp` is enabled and connected.

## Configuration

- App name: `wangp-mcp`
- GPU: `L40S` (override locally with `WANGP_GPU` before deployment)
- Maximum containers: 1
- Idle scale-down: 20 minutes after the last HTTP request finishes
- Server startup timeout: 30 minutes
- MCP transport: streamable HTTP on port 8000
