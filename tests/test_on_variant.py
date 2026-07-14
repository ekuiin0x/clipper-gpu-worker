"""run_render_job(on_variant=...) fires once per finished variant, in
order, and a raising callback never breaks the render."""
from pathlib import Path

import render_job as rj


def _wire(monkeypatch):
    monkeypatch.setattr(rj, "load_subtitles_json", lambda p: [])
    monkeypatch.setattr(rj, "render_variant_base",
                        lambda **kw: kw["dst"].write_bytes(b"base"))
    monkeypatch.setattr(rj, "preset_for_pack",
                        lambda pack, plan, hook_enabled: ({}, object()))
    monkeypatch.setattr(rj, "permute_panels", lambda plan, idx: plan)
    monkeypatch.setattr(rj, "burn_styled_ffmpeg",
                        lambda **kw: kw["dst"].write_bytes(b"vid"))


def _run(tmp_path, on_variant):
    tp = tmp_path / "transcript.json"
    tp.write_text("{}")
    return rj.run_render_job(
        src=tmp_path / "src.mp4",
        segments=[rj.Segment(plan={}, t_start=0.0, t_end=10.0)],
        transcript_path=tp,
        variants=[("clean", "word_only"), ("bold", "progressive")],
        out_w=1080, out_h=1920, fps=30.0,
        work_dir=tmp_path / "work",
        on_variant=on_variant,
    )


def test_on_variant_fires_in_order(tmp_path, monkeypatch):
    _wire(monkeypatch)
    seen = []
    results = _run(tmp_path, lambda r: seen.append(r["index"]))
    assert seen == [0, 1]
    assert [r["index"] for r in results] == [0, 1]
    # callback got the SAME dict that lands in results (handler mutates it)
    assert all("path" in r for r in results)


def test_on_variant_errors_do_not_break_render(tmp_path, monkeypatch):
    _wire(monkeypatch)

    def boom(r):
        raise RuntimeError("volume hiccup")

    results = _run(tmp_path, boom)
    assert len(results) == 2                      # render completed anyway
    assert all("path" in r for r in results)      # post-loop can still publish


def test_no_callback_is_the_default(tmp_path, monkeypatch):
    _wire(monkeypatch)
    tp = tmp_path / "transcript.json"
    tp.write_text("{}")
    results = rj.run_render_job(
        src=tmp_path / "src.mp4",
        segments=[rj.Segment(plan={}, t_start=0.0, t_end=10.0)],
        transcript_path=tp,
        variants=[("clean", "word_only")],
        out_w=1080, out_h=1920, fps=30.0,
        work_dir=tmp_path / "work",
    )
    assert len(results) == 1
