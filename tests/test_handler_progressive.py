"""handler(): with output_prefix set, each variant lands on the volume the
moment it's stamped (visible mid-render), atomically (.part never returned
or left behind); anything the callback missed is published post-loop."""
from pathlib import Path

import handler as h


def _base_input():
    return {
        "source_key": "clipper/j/source.mp4",
        "output_prefix": "clipper/j/out",
        "transcript": {"phrases": []},
        "segments": [{"plan": {}, "t_start": 0.0, "t_end": 15.0}],
        "variants": [["clean", "word_only"], ["bold", "progressive"]],
        "format": "9x16",
    }


def _wire(tmp_path, monkeypatch):
    vol = tmp_path / "vol"
    (vol / "clipper/j").mkdir(parents=True)
    (vol / "clipper/j/source.mp4").write_bytes(b"src")
    monkeypatch.setattr(h, "VOLUME_ROOT", vol)
    monkeypatch.setattr(h, "_probe_duration", lambda p: 15.0)
    monkeypatch.setattr(h, "_ffmpeg_caps", lambda: {})
    return vol


def test_progressive_publish_visible_mid_render(tmp_path, monkeypatch):
    vol = _wire(tmp_path, monkeypatch)
    visible_mid_render = []

    def fake_run_render_job(*, work_dir, on_variant=None, **kw):
        results = []
        for i, (pack, mode) in enumerate(
                [("clean", "word_only"), ("bold", "progressive")]):
            p = Path(work_dir) / f"variant_{i:02d}.mp4"
            p.write_bytes(b"vid%d" % i)
            r = {"index": i, "pack": pack, "subtitle_mode": mode,
                 "path": str(p), "compose_s": 1.0, "stamp_s": 1.0,
                 "reused_base": i > 0}
            results.append(r)
            if i == 0 and on_variant is not None:
                on_variant(r)          # only the FIRST publishes mid-render
                visible_mid_render.append(
                    (vol / "clipper/j/out/variant_00_clean.mp4").exists())
        return results

    monkeypatch.setattr(h, "run_render_job", fake_run_render_job)
    out = h.handler({"input": _base_input()})

    assert out["ok"], out
    assert visible_mid_render == [True]
    # variant 1 (never given to on_variant) got published by the post-loop
    keys = [v["key"] for v in out["variants"]]
    assert keys == ["clipper/j/out/variant_00_clean.mp4",
                    "clipper/j/out/variant_01_bold.mp4"]
    assert all("path" not in v for v in out["variants"])
    assert all(v["bytes"] == 4 for v in out["variants"])
    # atomic writes: no .part leftovers on the volume
    names = sorted(p.name for p in (vol / "clipper/j/out").iterdir())
    assert names == ["variant_00_clean.mp4", "variant_01_bold.mp4"]


def test_return_video_opts_out_of_progressive(tmp_path, monkeypatch):
    _wire(tmp_path, monkeypatch)
    seen = {}

    def fake_run_render_job(*, work_dir, on_variant=None, **kw):
        seen["on_variant"] = on_variant
        p = Path(work_dir) / "variant_00.mp4"
        p.write_bytes(b"vid")
        return [{"index": 0, "pack": "clean", "subtitle_mode": "word_only",
                 "path": str(p), "compose_s": 1.0, "stamp_s": 1.0,
                 "reused_base": False}]

    monkeypatch.setattr(h, "run_render_job", fake_run_render_job)
    out = h.handler({"input": {**_base_input(), "return_video": True,
                               "variants": [["clean", "word_only"]]}})
    assert out["ok"], out
    assert seen["on_variant"] is None        # base64 path needs local files
    assert out["variants"][0]["mp4_base64"]
