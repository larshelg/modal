Exactly — that’s the catch. If we build a hand-written REST API like:

POST /generate-video
POST /generate-image
POST /upscale

with strongly typed parameters, then yes: we’d end up duplicating a lot of WanGP’s API surface, and every new WanGP model/parameter could require us to update our wrapper.

The better approach is to not port the generate API at all.

Wan2AI already solves this locally by wrapping WanGP’s internal API rather than reimplementing it; the repo explicitly describes wangp.py as a wrapper with “no code duplication.” WanGP itself also now exposes an internal API intended for apps/agents.

So on Modal I’d expose something much more generic:

POST /execute
{
  "command": "generate",
  "args": {
      "model": "...",
      "prompt": "...",
      "whatever_wangp_accepts": ...
  }
}

or even better:

POST /generate
{
    "model": "minimax_h3",
    "params": { ... }
}

where params is just an opaque dictionary.

Then the Modal side effectively does:

def generate(model, params):
    return wangp_api.generate(
        model=model,
        **params
    )

So Modal doesn't need to understand:

steps
guidance
frames
resolution
loras
sliding_window
control_video
reference_images
denoise
seed
etc...

It merely transports them.

Conceptually:

Wan2AI skill
      │
      │ knows WanGP params
      ▼
{
  model: "minimax_h3",
  params: {
     prompt: "...",
     frames: 121,
     ...
  }
}
      │
      ▼
Modal thin transport layer
      │
      ▼
WanGP internal API

That is much better.

And there is another interesting option: WanGP already supports processing exported settings/queue JSON headlessly. Its CLI accepts a WanGP settings JSON or queue ZIP and processes it without the GUI.

So your endpoint could potentially become ridiculously generic:

POST /jobs
{
   "settings": {
       ...native WanGP settings JSON...
   }
}

Modal then hands that directly to WanGP.

That gives us almost zero translation layer:

Eve
 ↓
Wan2AI skill
 ↓
native WanGP config
 ↓
Modal transport
 ↓
WanGP
What I would choose

I think Wan2AI's wangp.py should actually live inside the Modal GPU image.

Then externally we'd expose only something like:

POST /wangp
{
   "action": "generate",
   "args": {...}
}

The Modal function calls Wan2AI's wrapper locally:

wangp.generate(**args)

Now all the model-specific compatibility logic remains in Wan2AI/WanGP, next to WanGP itself.

When WanGP adds:

MiniMax H4
new guidance mode
new ControlNet
new LoRA parameter
new sliding-window option

our Modal API doesn't change.

Only the payload changes.

So the architecture I'd now recommend is:

                EVE / CODEX
                     │
              Wan2AI Agent Skill
                     │
              generic JSON request
                     │
                     ▼
┌──────────────────── MODAL ─────────────────────┐
│                                               │
│       tiny API                                │
│           │                                   │
│           ▼                                   │
│     Wan2AI wangp.py                           │
│           │                                   │
│           ▼                                   │
│     WanGP internal API                        │
│           │                                   │
│           ▼                                   │
│          GPU                                  │
│                                               │
└───────────────────────────────────────────────┘

That completely changes my earlier recommendation: we should not design our own WanGP REST schema. Make Modal a thin remote transport around the existing Wan2AI/WanGP interface.

And I think this is probably the key reason Wan2AI is so interesting for your project — it lets us avoid maintaining a second WanGP API ourselves.

Kilder
yes you nailed it, and another benefit her is scaling, with this approach we can scale the api on modal 

Exactly — that’s a major benefit.

With this approach, you can scale the API layer and the GPU execution layer separately, which is much cleaner than baking everything into one monolithic Wan2GP server.

Why this is nice on Modal

Think of it as two layers:

1. Thin API / control layer

This is the small Modal endpoint that receives requests like:

generate
get job status
cancel
list models
download lora

This layer is lightweight and can scale easily.

2. GPU worker layer

This is where Wan2GP / Wan2AI wrapper actually runs the generation.

That means:

the HTTP/API endpoint can stay responsive
the GPU jobs can queue or scale independently
you don’t need one always-on giant Wan2GP MCP server just to accept requests
The big architecture win

Instead of:

client
  ↓
Wan2GP server
  ↓
same process handles api + gpu + job state

you get:

client
  ↓
Modal API endpoint
  ↓
spawn job
  ↓
GPU worker runs Wan2GP
  ↓
result stored in volume / S3

That gives you much better scaling behavior.

Why this scales better
A. API can scale independently

If Eve, Codex, Slack, Discord, Buzz, etc. all send requests at once, the API layer can accept them quickly and return job IDs.

It does not need to block while generation is running.

So 20 users can hit the API at once without requiring 20 live Wan2GP server instances.

B. GPU workers can scale based on demand

Modal is really strong here.

You can have:

0 workers when idle
1 worker for light use
N workers when many jobs come in

So your scaling unit becomes the job execution worker, not the API server.

That is much more cloud-native.

C. Easier queueing

Because generation is async, you naturally get:

submit job
queue job
run when GPU available
poll status or send callback/webhook

This is much cleaner for video generation, especially with jobs that may take several minutes.

Even better: split CPU orchestration and GPU execution

You can go one step further and do:

POST /generate
   ↓
CPU function validates request
   ↓
enqueue / spawn GPU job
   ↓
GPU worker runs Wan2GP

So the cheap CPU side handles:

auth
validation
job bookkeeping
LoRA lookup
model selection
rate limiting
status endpoints

And the expensive GPU side only does:

load model
run generation
save output

This is ideal.

Nice practical benefit for you

This also means your agent-facing API can remain stable, even while your GPU backend evolves.

For example, today your backend may use:

MiniMax H3
Krea2
Qwen Image
Wan video models

Later you might add:

new LoRA logic
ControlNet support
a different postprocess path
new models
better fallback logic

None of that has to change the public API much.

The outer API can stay like:

{
  "action": "generate",
  "model": "minimax_h3",
  "params": { ... }
}

and internally you decide how to route it.

Another huge advantage: you avoid MCP as the scaling bottleneck

If Wan2GP itself is the MCP server, then:

MCP server availability becomes critical
GPU container may need to stay warm just to answer metadata requests
scaling gets awkward because the “tool server” and “GPU execution” are tightly coupled

With your new idea:

the agent talks to a simple API
Modal handles fan-out to workers
Wan2GP stays a backend engine

That is a much better separation of concerns.

The ideal mental model

Treat Wan2GP like this:

Wan2GP is not your public server. Wan2GP is your internal rendering engine.

And treat Wan2AI-style wrapper like this:

Wan2AI / wangp.py is your execution adapter.

And treat Modal like this:

Modal is your elastic control plane + compute plane.

That’s the clean setup.

In practice, you’d likely want these endpoints

Something like:

POST /jobs → submit generation
GET /jobs/{id} → status
POST /jobs/{id}/cancel
GET /models
GET /loras
POST /loras/download
DELETE /loras/{name}

But the important one is that /jobs just forwards a mostly opaque payload into Wan2AI/Wan2GP logic.

So you avoid reimplementing Wan’s full param surface.

Best part for your use case

For your Eve / Codex / Buzz setup, this means:

Eve can submit jobs fast
multiple jobs can run in parallel
you can start on a smaller GPU for prototyping
later increase max_containers for production
the outer integration stays mostly unchanged

So yes — this architecture is not only simpler, it is also much more scalable on Modal.

If you want, I can sketch the exact Modal architecture next, with:

API function
GPU worker class
job status model
LoRA download flow
how Eve would call it

in a very concrete way.

I will start adding the missing pieces from

            EVE / CODEX
                 │
          Wan2AI Agent Skill
                 │
          generic JSON request
                 │
                 ▼


┌──────────────────── MODAL ─────────────────────┐
│ │
│ tiny API │
│ │ │
│ ▼ │
│ Wan2AI wangp.py │
│ │ │
│ ▼ │
│ WanGP internal API │
│ │ │
│ ▼ │
│ GPU │
│ │
└───────────────────────────────────────────────┘

Yes — that’s the right place to start.

I’d add the missing pieces outside-in, so you keep the architecture clean:

            EVE / CODEX
                 │
          Wan2AI Agent Skill
                 │
          generic JSON request
                 │
                 ▼
┌──────────────────── MODAL ─────────────────────┐
│                                               │
│   tiny HTTP API                               │
│       │                                       │
│       ├── auth                                │
│       ├── submit job                          │
│       ├── job status                          │
│       └── cancel                              │
│                                               │
│       ▼                                       │
│   job/router layer                            │
│       │                                       │
│       ├── choose GPU worker                   │
│       ├── spawn async generation              │
│       └── persist job metadata                │
│                                               │
│       ▼                                       │
│   Wan2AI wangp.py                             │
│       │                                       │
│       ├── model logic                         │
│       ├── defaults                            │
│       ├── LoRA handling                       │
│       └── WanGP parameter mapping             │
│                                               │
│       ▼                                       │
│   WanGP internal API                          │
│       │                                       │
│       ▼                                       │
│      GPU                                      │
│                                               │
│   shared storage                              │
│       ├── models                              │
│       ├── LoRAs                               │
│       └── outputs → S3                        │
│                                               │
└───────────────────────────────────────────────┘

The first version can be very small. I would initially implement only:

POST /jobs
GET  /jobs/{job_id}
POST /jobs/{job_id}/cancel
GET  /models
POST /loras

And crucially, /jobs should accept something generic like:

{
  "action": "generate",
  "model": "minimax_h3",
  "args": {
    "prompt": "...",
    "duration": 5,
    "seed": 1234,
    "loras": [...]
  }
}

Don't start validating every WanGP argument in the API. Let the Wan2AI/WanGP layer own those semantics.

I’d also keep job state outside the GPU process from day one. Even a simple record such as:

{
  "id": "job_123",
  "status": "running",
  "created_at": "...",
  "model": "minimax_h3",
  "output": null,
  "error": null
}

makes scaling much easier later because no individual GPU container becomes the source of truth.

For your first milestone, I’d aim for exactly this:

curl POST /jobs
        ↓
Modal returns job_id immediately
        ↓
GPU worker starts
        ↓
wangp.py calls WanGP
        ↓
MP4 written
        ↓
uploaded to S3
        ↓
GET /jobs/id returns completed + URL

Once that works, adding Eve/Codex on top becomes almost trivial because the agent skill only has to understand the generic job API rather than Modal internals.

And I would not touch the current WanGP generation code more than necessary. Treat it as the engine; build the remote/scaling layer around it.