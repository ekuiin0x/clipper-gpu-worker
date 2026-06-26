"""Cl1pper render engine (GPU-worker subset).

Vendored rendering closure only: 9:16 composition (v11_render) + libass caption
stamp (captions_ass) + style packs. NO VLM, planning prompts, bot, web, or
payment code — those stay in the private app repo. This __init__ is intentionally
empty so importing a submodule (e.g. ``engine.v11_render``) does not drag in the
full pipeline.
"""
__version__ = "0.1.0-gpuworker"
