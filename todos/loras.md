Yes — most likely, yes.
You’d probably build that tiny MCP yourself.

But the good news is: it can be extremely small. It does not need to be a big project.

What it really is

It’s basically just a thin helper layer that exposes a few safe tools like:

download_lora(url, model_family)
list_loras(model_family)
delete_lora(name, model_family)
maybe refresh_loras() if needed

And internally it just:

maps model_family to the correct Wan2GP LoRA folder
downloads the file there
optionally validates extension / filename
returns success metadata

So yes, it’s “your own MCP”, but in practice it’s more like a small utility server than a whole app.

Why I would build it myself

Because the built-in Wan2GP MCP:

already handles generation
already knows about jobs
already exposes its own tool set

But it does not expose the infrastructure action you need:

managing LoRA files in the right folder on your shared volume

That makes this a perfect case for a custom sidecar/helper MCP.

You do not want to fork Wan2GP just to add one file-management command unless absolutely necessary.

Best architecture

I’d keep it like this:

Codex / Eve
   │
   ├── Wan2GP MCP
   │      generate / list / jobs
   │
   └── Your LoRA MCP
          download / list / delete

Both share:

/wangp/loras/

So your helper MCP writes files where Wan2GP can use them.

Why this is better than hacking Wan2GP
Option A — build tiny helper MCP

Pros

very small
no Wan2GP fork needed
easy to maintain
safe and focused
can evolve independently

Cons

one extra small service
Option B — fork Wan2GP MCP and add commands

Pros

everything in one MCP

Cons

harder to maintain
upstream updates become annoying
you now own a modified Wan2GP MCP forever

For your use case, Option A is clearly better.

How much code are we talking about?

Honestly, probably not much.

If you keep it minimal, it could be roughly:

one MCP server file
one path resolver
one downloader helper
maybe one config file

So this is not a “big build”.

It’s more like:

lora_mcp/
  server.py
  config.py
  downloader.py

Very manageable.

Even simpler alternative

You could also skip MCP entirely and expose a tiny internal HTTP endpoint or Modal function like:

POST /download-lora
POST /delete-lora
GET /list-loras

Then let Eve/Codex call that through whatever tool layer you prefer.

But since you’re already thinking MCP-first, a tiny MCP is cleaner because then it behaves like the rest of your tool ecosystem.

What I would include in v1

For version 1, I would only build:

Tools
wangp_download_lora(url, family, filename?)
wangp_list_loras(family?)
wangp_delete_lora(name, family)
Safety
only allow known model families
only allow writes inside approved LoRA directories
only allow .safetensors (and maybe a few approved types)
optionally restrict domains if you want
Nice-to-have
return whether Wan2GP likely needs a refresh
file size
saved path
timestamp