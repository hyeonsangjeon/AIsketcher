from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from PIL import Image

from aisketcher.studio_app.runtime import AppController


@pytest.mark.skipif(
    importlib.util.find_spec("aisketcher") is None,
    reason="AIsketcher source package is not on this test environment's path",
)
def test_controller_matches_real_studio_api_with_fake_backend(tmp_path: Path) -> None:
    from aisketcher import FakeBackend, Studio

    controller = AppController(
        studio_factory=lambda preset: Studio.from_preset(
            preset,
            backend=FakeBackend(),
        ),
        workspace_root=tmp_path / "work",
    )
    source = tmp_path / "sketch.png"
    Image.new("RGB", (80, 64), "white").save(source)

    response = controller.explore(
        controller.initial_state(),
        source,
        "Simple paper-cut forms",
        "graphic_design",
        "balanced",
        "sdxl-canny-lite@1",
        1,
        "scout",
        "",
        True,
        10,
        4.0,
        ("structure",),
    )

    assert len(response.gallery) == 1
    assert response.state["selected_index"] == 0
    record = controller.registry.get(
        str(response.state["run_id"]), str(response.state["session_id"])
    )
    assert record.study.recipe.steps == 10
    assert record.study.recipe.guidance_scale == 4.0
    assert record.candidates[0].seed is not None
    assert Path(record.candidates[0].path).is_file()

    archive, _ = controller.export(response.state)
    assert Path(archive).is_file()

    replayed = controller.replay_manifest(
        response.state,
        archive,
        "sdxl-canny-lite@1",
    )

    assert replayed.status == "Strict replay completed."
    assert replayed.state["run_id"] != response.state["run_id"]
    assert len(replayed.gallery) == 1
    assert Path(replayed.source or "").is_file()
    assert Path(replayed.selected or "").is_file()
