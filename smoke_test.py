"""Local CPU smoke test — proves the import closure + full render pipeline.

Runs the handler self-test with ``RENDER_GPU`` unset, so the render falls back
to libx264. Validates that every vendored module imports, the compose+stamp
pipeline produces real mp4s, and the response shape is correct. NVENC will show
as unavailable here (no GPU locally) — that's expected; this only proves the
code path is sound before deploying to RunPod.
"""
from __future__ import annotations

import json
import sys

from handler import handler


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    result = handler({"input": {
        "selftest": True,
        "variants_count": n,
        "seconds": 8,
        "return_video": False,
    }})

    print(json.dumps(result, indent=2))

    if not result.get("ok"):
        print("\nSMOKE TEST FAILED", file=sys.stderr)
        return 1

    variants = result.get("variants", [])
    if len(variants) != n:
        print(f"\nFAILED: expected {n} variants, got {len(variants)}", file=sys.stderr)
        return 1
    if any(v.get("bytes", 0) <= 0 for v in variants):
        print("\nFAILED: a variant produced an empty file", file=sys.stderr)
        return 1

    print(f"\nSMOKE TEST PASSED — {len(variants)} variants rendered (libx264).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
