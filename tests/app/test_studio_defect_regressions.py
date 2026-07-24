from __future__ import annotations

import importlib.util
import json
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

import aisketcher.studio_app.runtime as runtime_module
from aisketcher.prompt_normalization import TranslatorMetadata
from aisketcher.studio_app import AppController, build_app
from aisketcher.studio_app.app import STUDIO_JS
from aisketcher.studio_app.i18n import text
from aisketcher.studio_app.runtime import (
    StudioAppError,
    StudioJobCancelled,
    _CrossProcessFileLease,
    prepare_replay_input,
)

PRESET = "sdxl-canny-lite@1"


@dataclass
class _Candidate:
    image: Image.Image
    seed: int
    technical_badge: str = "Closest structure"
    reason: str = "Recorded structure evidence."
    recipe: Any = None


@dataclass(frozen=True)
class _BoundedRecipe:
    prompt: str
    model_prompt: str | None
    prompt_metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if len(self.prompt) > 10_000:
            raise ValueError("prompt exceeds native recipe limit")
        if self.model_prompt is not None and len(self.model_prompt) > 10_000:
            raise ValueError("model prompt exceeds native recipe limit")
        encoded = json.dumps(
            self.prompt_metadata,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(encoded) > 20_000:
            raise ValueError("prompt metadata exceeds native recipe limit")


class _Study:
    def __init__(self, count: int, *, recipe: Any = None) -> None:
        self.candidates = [
            _Candidate(
                Image.new("RGB", (32, 32), (30 + index * 20, 80, 140)),
                100 + index,
                recipe=recipe,
            )
            for index in range(count)
        ]

    def pick(self, index: int) -> _Candidate:
        return self.candidates[index]

    def export(self, destination: str) -> Path:
        root = Path(destination)
        root.mkdir(parents=True, exist_ok=True)
        manifest = root / "manifest.json"
        manifest.write_text(
            json.dumps({"schema": "aisketcher.manifest/v1", "backend": "test"}),
            encoding="utf-8",
        )
        return manifest


class _Studio:
    def __init__(
        self,
        preset: str,
        *,
        entered: threading.Event | None = None,
        release: threading.Event | None = None,
        fail_refine_with_oom: bool = False,
    ) -> None:
        self.preset = preset
        self.entered = entered
        self.release = release
        self.fail_refine_with_oom = fail_refine_with_oom
        self.closed = False
        self.cancel_requested = False

    def prepare(self, source: str, **_: Any) -> str:
        return source

    def explore(self, prepared: str, *, outputs: int, **_: Any) -> _Study:
        del prepared
        if self.entered is not None:
            self.entered.set()
        if self.release is not None:
            assert self.release.wait(timeout=5)
        return _Study(outputs)

    def vary(self, selected: _Candidate, *, outputs: int, **_: Any) -> _Study:
        del selected
        if self.fail_refine_with_oom:
            raise RuntimeError("CUDA error: out of memory")
        return _Study(outputs)

    def request_cancel(self) -> None:
        self.cancel_requested = True

    def close(self) -> None:
        self.closed = True


class _NativeLikeStudio(_Studio):
    """Test double whose ``vary`` signature matches the public Studio API."""

    def __init__(
        self,
        preset: str,
        *,
        prompt_metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(preset)
        self.prompt_metadata = prompt_metadata
        self.vary_calls: list[_Candidate] = []

    def explore(
        self,
        prepared: str,
        *,
        intent: Any,
        outputs: int,
        **_: Any,
    ) -> _Study:
        del prepared
        recipe = _BoundedRecipe(
            prompt=intent.prompt,
            model_prompt=intent.model_prompt,
            prompt_metadata=(
                dict(self.prompt_metadata)
                if self.prompt_metadata is not None
                else dict(intent.prompt_metadata)
            ),
        )
        return _Study(outputs, recipe=recipe)

    def vary(
        self,
        selected: _Candidate,
        *,
        outputs: int = 4,
        strength: str = "subtle",
        locks: tuple[str, ...] = ("structure",),
        seed_plan: Any = None,
    ) -> _Study:
        del strength, locks, seed_plan
        self.vary_calls.append(selected)
        return _Study(outputs, recipe=selected.recipe)


def _source(path: Path) -> Path:
    Image.new("RGB", (64, 48), "white").save(path)
    return path


def _explore(
    controller: AppController,
    source: Path,
    state: dict[str, Any] | None = None,
    *,
    brief: str = "Layered paper shapes in navy, coral, and gold",
) -> Any:
    return controller.explore(
        state or controller.initial_state(),
        source,
        brief,
        "graphic_design",
        "balanced",
        PRESET,
        4,
        "scout",
        "",
        True,
        30,
        5.0,
        ("structure",),
    )


def _wait_until(predicate: Any, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not reached before timeout")


@pytest.mark.skipif(
    importlib.util.find_spec("fcntl") is None,
    reason="advisory host leases require fcntl",
)
def test_host_file_lease_recovers_stale_metadata_and_records_the_new_owner(
    tmp_path: Path,
) -> None:
    path = tmp_path / "shared" / "accelerator.lock"
    path.parent.mkdir(parents=True)
    path.write_text('{"pid": 999999, "session_id": "stale"}', encoding="utf-8")
    lease = _CrossProcessFileLease(path)
    cancellation_event = threading.Event()

    with lease.acquire(
        cancellation_event=cancellation_event,
        timeout_seconds=1,
        session_id="fresh-session",
    ):
        owner = json.loads(path.read_text(encoding="utf-8"))
        assert owner["schema"] == "aisketcher.host-lease/v1"
        assert owner["session_id"] == "fresh-session"

    # The file is intentionally persistent; a second acquisition proves stale
    # owner JSON is never treated as a lock.
    with lease.acquire(
        cancellation_event=cancellation_event,
        timeout_seconds=1,
        session_id="next-session",
    ):
        owner = json.loads(path.read_text(encoding="utf-8"))
        assert owner["session_id"] == "next-session"


@pytest.mark.skipif(
    importlib.util.find_spec("fcntl") is None,
    reason="advisory host leases require fcntl",
)
def test_host_file_lease_times_out_and_stop_interrupts_a_waiter(
    tmp_path: Path,
) -> None:
    path = tmp_path / "accelerator.lock"
    owner = _CrossProcessFileLease(path)
    waiter = _CrossProcessFileLease(path)
    owner_cancel = threading.Event()

    with owner.acquire(
        cancellation_event=owner_cancel,
        timeout_seconds=1,
        session_id="owner",
    ):
        with (
            pytest.raises(TimeoutError, match="another AIsketcher process"),
            waiter.acquire(
                cancellation_event=threading.Event(),
                timeout_seconds=0.05,
                session_id="timeout",
            ),
        ):
            raise AssertionError("contended lease must not be acquired")

        waiter_cancel = threading.Event()
        with ThreadPoolExecutor(max_workers=1) as executor:
            blocked = executor.submit(
                _acquire_cancelled_host_lease,
                waiter,
                waiter_cancel,
            )
            time.sleep(0.05)
            waiter_cancel.set()
            with pytest.raises(StudioJobCancelled, match="Stopped"):
                blocked.result(timeout=1)


def test_host_file_lease_fails_closed_without_a_platform_lock_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MissingPlatformLocks:
        @staticmethod
        def import_module(_name: str) -> Any:
            raise ImportError("no platform lock backend")

    monkeypatch.setattr(runtime_module, "importlib", MissingPlatformLocks())
    lease = _CrossProcessFileLease(tmp_path / "accelerator.lock")

    with (
        pytest.raises(StudioAppError, match="no supported cross-process"),
        lease.acquire(
            cancellation_event=threading.Event(),
            timeout_seconds=1,
            session_id="unsupported-platform",
        ),
    ):
        raise AssertionError("unsupported platforms must fail closed")


def _acquire_cancelled_host_lease(
    lease: _CrossProcessFileLease,
    cancellation_event: threading.Event,
) -> None:
    with lease.acquire(
        cancellation_event=cancellation_event,
        timeout_seconds=5,
        session_id="cancelled-waiter",
    ):
        raise AssertionError("cancelled waiter must not acquire the host lease")


def test_stop_during_blocking_translation_never_starts_generation_and_recovers(
    tmp_path: Path,
) -> None:
    korean_brief = "귀여운 종이 왕국"
    translated_brief = "A cute layered paper kingdom"

    class BlockingTranslator:
        def __init__(self) -> None:
            self.entered = threading.Event()
            self.release = threading.Event()
            self.calls: list[str] = []
            self.blocked_once = False

        @property
        def metadata(self) -> TranslatorMetadata:
            return TranslatorMetadata(
                provider="test",
                model_id="example/ko-en",
                revision="test-revision",
                local_files_only=True,
            )

        def translate(self, value: str) -> str:
            self.calls.append(value)
            if value == korean_brief and not self.blocked_once:
                self.blocked_once = True
                self.entered.set()
                assert self.release.wait(timeout=5)
            return translated_brief

    class CountingStudio(_Studio):
        def __init__(self, preset: str) -> None:
            super().__init__(preset)
            self.explore_prompts: list[str] = []

        def explore(
            self,
            prepared: str,
            *,
            intent: Any,
            outputs: int,
            **kwargs: Any,
        ) -> _Study:
            self.explore_prompts.append(str(getattr(intent, "model_prompt", None) or intent.prompt))
            return super().explore(prepared, outputs=outputs, **kwargs)

    class CountingController(AppController):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.finished_sessions: list[str] = []

        def finish_operation(
            self,
            state_value: Any,
            operation_event: threading.Event,
        ) -> None:
            session_id = getattr(state_value, "session_id", None)
            if session_id is None:
                session_id = state_value["session_id"]
            self.finished_sessions.append(str(session_id))
            super().finish_operation(state_value, operation_event)

    translator = BlockingTranslator()
    studio = CountingStudio(PRESET)
    controller = CountingController(
        studio_factory=lambda _preset: studio,
        prompt_translator=translator,
        workspace_root=tmp_path / "work",
    )
    source = _source(tmp_path / "source.png")
    blocked_state = controller.initial_state("ko")
    other_state = controller.initial_state("en")

    with ThreadPoolExecutor(max_workers=2) as executor:
        blocked = executor.submit(
            _explore,
            controller,
            source,
            blocked_state,
            brief=korean_brief,
        )
        assert translator.entered.wait(timeout=2)
        assert controller.operation_state(blocked_state) == "running"

        unaffected = _explore(
            controller,
            source,
            other_state,
            brief="A crisp English paper kingdom",
        )
        assert len(unaffected.gallery) == 4
        assert studio.explore_prompts == ["A crisp English paper kingdom"]

        controller.cancel_operation(blocked_state)
        assert controller.operation_state(blocked_state) == "stopping"
        controller.clear_operation(blocked_state)
        assert controller.operation_state(blocked_state) == "stopping"
        translator.release.set()
        with pytest.raises(StudioJobCancelled):
            blocked.result(timeout=2)

    assert controller.operation_state(blocked_state) == "idle"
    assert controller.registry.latest(str(blocked_state["session_id"])) is None
    assert controller.finished_sessions.count(str(blocked_state["session_id"])) == 1
    assert studio.explore_prompts == ["A crisp English paper kingdom"]

    recovered = _explore(
        controller,
        source,
        blocked_state,
        brief=korean_brief,
    )
    assert len(recovered.gallery) == 4
    assert studio.explore_prompts[-1] == translated_brief
    assert controller.operation_state(blocked_state) == "idle"
    assert controller.finished_sessions.count(str(blocked_state["session_id"])) == 2


def test_stop_during_blocking_refinement_translation_never_calls_vary(
    tmp_path: Path,
) -> None:
    korean_instruction = "종이 가장자리를 더 선명하게"

    class BlockingTranslator:
        def __init__(self) -> None:
            self.entered = threading.Event()
            self.release = threading.Event()
            self.blocked_once = False

        @property
        def metadata(self) -> TranslatorMetadata:
            return TranslatorMetadata(
                provider="test",
                model_id="example/ko-en",
                revision="test-revision",
                local_files_only=True,
            )

        def translate(self, value: str) -> str:
            if value == korean_instruction and not self.blocked_once:
                self.blocked_once = True
                self.entered.set()
                assert self.release.wait(timeout=5)
            return "Use crisper paper edges"

    class CountingStudio(_Studio):
        def __init__(self, preset: str) -> None:
            super().__init__(preset)
            self.vary_calls = 0

        def vary(self, selected: _Candidate, *, outputs: int, **kwargs: Any) -> _Study:
            self.vary_calls += 1
            return super().vary(selected, outputs=outputs, **kwargs)

    class CountingController(AppController):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.finished_sessions: list[str] = []

        def finish_operation(
            self,
            state_value: Any,
            operation_event: threading.Event,
        ) -> None:
            session_id = getattr(state_value, "session_id", None)
            if session_id is None:
                session_id = state_value["session_id"]
            self.finished_sessions.append(str(session_id))
            super().finish_operation(state_value, operation_event)

    translator = BlockingTranslator()
    studio = CountingStudio(PRESET)
    controller = CountingController(
        studio_factory=lambda _preset: studio,
        prompt_translator=translator,
        workspace_root=tmp_path / "work",
    )
    source = _source(tmp_path / "source.png")
    explored = _explore(controller, source)
    session_id = str(explored.state["session_id"])
    completed_before = controller.finished_sessions.count(session_id)

    with ThreadPoolExecutor(max_workers=1) as executor:
        blocked = executor.submit(
            controller.refine,
            explored.state,
            "subtle",
            ("structure",),
            korean_instruction,
        )
        assert translator.entered.wait(timeout=2)
        assert controller.operation_state(explored.state) == "running"
        controller.cancel_operation(explored.state)
        assert controller.operation_state(explored.state) == "stopping"
        translator.release.set()
        with pytest.raises(StudioJobCancelled):
            blocked.result(timeout=2)

    assert controller.operation_state(explored.state) == "idle"
    assert studio.vary_calls == 0
    assert controller.finished_sessions.count(session_id) == completed_before + 1

    recovered = controller.refine(
        explored.state,
        "subtle",
        ("structure",),
        korean_instruction,
    )
    assert len(recovered.gallery) == 4
    assert studio.vary_calls == 1


def test_preclaim_stop_tombstones_only_its_ticket_and_retry_can_claim(
    tmp_path: Path,
) -> None:
    controller = AppController(
        studio_factory=_Studio,
        workspace_root=tmp_path / "work",
    )
    state = controller.initial_state("ko")

    stopped_ticket = controller.start_operation(state)
    controller.cancel_operation(state)
    assert controller.operation_state(state) == "idle"

    retry_ticket = controller.start_operation(state)
    assert retry_ticket != stopped_ticket
    with pytest.raises(StudioJobCancelled, match="중지"):
        controller.claim_operation(state, stopped_ticket)

    # A delayed cleanup belonging to the cancelled callback cannot remove the
    # retry's newer token.
    controller.clear_operation(state, stopped_ticket)
    assert controller.operation_state(state) == "running"
    retry_event = controller.claim_operation(state, retry_ticket)
    controller.finish_operation(state, retry_event)
    assert controller.operation_state(state) == "idle"

    # When the cancelled callback arrives before a retry, its exact ticket is
    # consumed without leaving an unbounded pending token behind.
    another_state = controller.initial_state("en")
    abandoned_ticket = controller.start_operation(another_state)
    controller.cancel_operation(another_state)
    with pytest.raises(StudioJobCancelled, match="Stopped"):
        controller.claim_operation(another_state, abandoned_ticket)
    controller.clear_operation(another_state, abandoned_ticket)
    assert another_state["session_id"] not in controller._operation_tokens
    assert another_state["session_id"] not in controller._operation_ids
    assert another_state["session_id"] not in controller._operation_kinds


def test_quick_retry_cannot_overlap_a_claimed_job_that_is_still_stopping(
    tmp_path: Path,
) -> None:
    controller = AppController(
        studio_factory=_Studio,
        workspace_root=tmp_path / "work",
    )
    state = controller.initial_state("en")

    running_ticket = controller.start_operation(state)
    running_event = controller.claim_operation(state, running_ticket)
    controller.cancel_operation(state)
    assert controller.operation_state(state) == "stopping"

    # A second click while the cancelled backend is still unwinding must bind
    # to the existing operation instead of creating an orphan token that could
    # later overlap the first GPU job.
    retry_while_stopping = controller.start_operation(state)
    assert retry_while_stopping == running_ticket
    with pytest.raises(StudioAppError, match="already active"):
        controller.claim_operation(state, retry_while_stopping)
    controller.clear_operation(state, retry_while_stopping)
    assert controller.operation_state(state) == "stopping"

    controller.finish_operation(state, running_event)
    assert controller.operation_state(state) == "idle"

    retry_ticket = controller.start_operation(state)
    assert retry_ticket != running_ticket
    retry_event = controller.claim_operation(state, retry_ticket)
    controller.finish_operation(state, retry_event)
    assert controller.operation_state(state) == "idle"


def test_cancelled_shared_runtime_does_not_poison_another_sessions_completed_run(
    tmp_path: Path,
) -> None:
    class ClosedAwareStudio(_Studio):
        def vary(self, selected: _Candidate, *, outputs: int, **kwargs: Any) -> _Study:
            if self.closed:
                raise RuntimeError("closed shared backend")
            return super().vary(selected, outputs=outputs, **kwargs)

    created: list[ClosedAwareStudio] = []

    def factory(preset: str) -> ClosedAwareStudio:
        studio = ClosedAwareStudio(preset)
        created.append(studio)
        return studio

    controller = AppController(
        studio_factory=factory,
        workspace_root=tmp_path / "work",
    )
    source = _source(tmp_path / "source.png")
    completed = _explore(controller, source, controller.initial_state("en"))
    shared = created[0]
    shared.entered = threading.Event()
    shared.release = threading.Event()
    cancelled_state = controller.initial_state("ko")

    with ThreadPoolExecutor(max_workers=1) as executor:
        cancelled = executor.submit(
            _explore,
            controller,
            source,
            cancelled_state,
        )
        assert shared.entered.wait(timeout=2)
        controller.cancel_operation(cancelled_state)
        shared.release.set()
        with pytest.raises(StudioJobCancelled):
            cancelled.result(timeout=2)

    assert shared.closed is True
    refined = controller.refine(
        completed.state,
        "subtle",
        ("structure",),
        "Use crisper paper edges",
    )
    assert len(refined.gallery) == 4
    assert len(created) == 2
    assert created[1].closed is False
    assert (
        controller.registry.get(
            str(refined.state["run_id"]),
            str(refined.state["session_id"]),
        ).studio
        is created[1]
    )


def test_cancelled_job_waiting_for_gpu_never_loads_its_backend_and_can_retry(
    tmp_path: Path,
) -> None:
    first_entered = threading.Event()
    release_first = threading.Event()
    second_created: list[_Studio] = []
    first = AppController(
        studio_factory=lambda preset: _Studio(
            preset,
            entered=first_entered,
            release=release_first,
        ),
        workspace_root=tmp_path / "first",
    )

    def second_factory(preset: str) -> _Studio:
        studio = _Studio(preset)
        second_created.append(studio)
        return studio

    second = AppController(
        studio_factory=second_factory,
        workspace_root=tmp_path / "second",
    )
    source = _source(tmp_path / "source.png")
    second_state = second.initial_state("ko")

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(_explore, first, source)
        assert first_entered.wait(timeout=2)
        second_future = executor.submit(_explore, second, source, second_state)
        _wait_until(lambda: second_state["session_id"] in second._claimed_operation_tokens)

        second.cancel_operation(second_state)
        second.clear_operation(second_state)
        with pytest.raises(StudioJobCancelled):
            second_future.result(timeout=2)
        assert second_created == []

        release_first.set()
        assert len(first_future.result(timeout=2).gallery) == 4

    recovered = _explore(second, source, second_state)
    assert len(recovered.gallery) == 4
    assert len(second_created) == 1


def test_waiting_refine_stop_does_not_cancel_another_sessions_shared_backend(
    tmp_path: Path,
) -> None:
    shared = _Studio(PRESET)
    controller = AppController(
        studio_factory=lambda _preset: shared,
        workspace_root=tmp_path / "work",
    )
    source = _source(tmp_path / "source.png")
    waiting_state = controller.initial_state("ko")
    past_run = _explore(controller, source, waiting_state)

    active_entered = threading.Event()
    release_active = threading.Event()
    shared.entered = active_entered
    shared.release = release_active
    active_state = controller.initial_state("en")

    with ThreadPoolExecutor(max_workers=2) as executor:
        active_future = executor.submit(_explore, controller, source, active_state)
        assert active_entered.wait(timeout=2)
        waiting_future = executor.submit(
            controller.refine,
            past_run.state,
            "subtle",
            ("structure",),
            "Sharper paper edges",
        )
        _wait_until(lambda: past_run.state["session_id"] in controller._claimed_operation_tokens)

        controller.cancel_operation(past_run.state)
        controller.clear_operation(past_run.state)
        assert shared.cancel_requested is False

        release_active.set()
        assert len(active_future.result(timeout=2).gallery) == 4
        with pytest.raises(StudioJobCancelled):
            waiting_future.result(timeout=2)


def test_replay_zip_rejects_files_not_declared_by_its_manifest(
    tmp_path: Path,
) -> None:
    source_bytes = b"canonical-source"
    manifest = {
        "schema": "aisketcher.manifest/v1",
        "files": {
            "source": {
                "path": "source.png",
                "sha256": sha256(source_bytes).hexdigest(),
            }
        },
    }
    bundle = tmp_path / "study.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("study/manifest.json", json.dumps(manifest))
        archive.writestr("study/source.png", source_bytes)
        archive.writestr("study/debug.log", "must not be replayed")

    stage = tmp_path / "stage"
    with pytest.raises(StudioAppError, match="files not declared by its manifest"):
        prepare_replay_input(bundle, stage)

    assert not stage.exists()


def test_refresh_state_recovers_the_latest_completed_run_for_its_session(
    tmp_path: Path,
) -> None:
    controller = AppController(
        studio_factory=_Studio,
        workspace_root=tmp_path / "work",
    )
    source = _source(tmp_path / "source.png")
    browser_state = controller.initial_state("ko")
    completed = _explore(controller, source, browser_state)

    recovered = controller.recover_latest_run(browser_state, "ko")

    assert recovered is not None
    assert recovered.state["run_id"] == completed.state["run_id"]
    assert recovered.state["session_id"] == browser_state["session_id"]
    assert len(recovered.gallery) == 4
    assert controller.recover_latest_run(controller.initial_state("ko"), "ko") is None


def test_cancelled_model_install_releases_session_claim_and_allows_retry(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path / "source.png")
    install_entered = threading.Event()
    release_install = threading.Event()

    class Installer:
        def __init__(self) -> None:
            self.attempts = 0
            self.cancel_calls = 0

        def install(
            self,
            preset: str,
            *,
            should_cancel: Any,
            **_: Any,
        ) -> None:
            assert preset == PRESET
            self.attempts += 1
            if self.attempts == 1:
                install_entered.set()
                assert release_install.wait(timeout=5)
                if should_cancel():
                    raise RuntimeError("provider stopped at a safe boundary")

        def request_cancel(self) -> None:
            self.cancel_calls += 1
            release_install.set()

    class Translator:
        def __init__(self) -> None:
            self.prepare_calls = 0

        @property
        def metadata(self) -> TranslatorMetadata:
            return TranslatorMetadata(
                provider="test",
                model_id="example/ko-en",
                revision="test-revision",
                local_files_only=True,
            )

        def prepare(self, **_: Any) -> None:
            self.prepare_calls += 1

        def translate(self, value: str) -> str:
            return value

    installer = Installer()
    translator = Translator()
    controller = AppController(
        studio_factory=_Studio,
        model_installer=installer,
        prompt_translator=translator,
        workspace_root=tmp_path / "work",
    )
    state = controller.initial_state("ko")

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            controller.install_model,
            PRESET,
            True,
            "ko",
            state,
        )
        assert install_entered.wait(timeout=2)
        with pytest.raises(StudioAppError, match="이미 생성 작업"):
            _explore(controller, source, state)

        controller.cancel_operation(state)
        controller.clear_operation(state)
        with pytest.raises(StudioJobCancelled):
            future.result(timeout=2)

    assert installer.cancel_calls >= 1
    assert translator.prepare_calls == 0
    assert controller.install_model(PRESET, True, "ko", state).startswith("로컬 모델")
    assert translator.prepare_calls == 1
    assert len(_explore(controller, source, state).gallery) == 4


def test_waiting_model_setup_stop_does_not_cancel_another_session(
    tmp_path: Path,
) -> None:
    install_entered = threading.Event()
    release_install = threading.Event()

    class Installer:
        def __init__(self) -> None:
            self.attempts = 0
            self.cancel_calls = 0

        def install(self, _preset: str, **_: Any) -> None:
            self.attempts += 1
            install_entered.set()
            assert release_install.wait(timeout=5)

        def request_cancel(self) -> None:
            self.cancel_calls += 1
            release_install.set()

    class Translator:
        def prepare(self, **_: Any) -> None:
            return None

        def translate(self, value: str) -> str:
            return value

    installer = Installer()
    controller = AppController(
        studio_factory=_Studio,
        model_installer=installer,
        prompt_translator=Translator(),
        workspace_root=tmp_path / "work",
    )
    active_state = controller.initial_state("en")
    waiting_state = controller.initial_state("ko")

    with ThreadPoolExecutor(max_workers=2) as executor:
        active = executor.submit(
            controller.install_model,
            PRESET,
            True,
            "en",
            active_state,
        )
        assert install_entered.wait(timeout=2)
        waiting = executor.submit(
            controller.install_model,
            PRESET,
            True,
            "ko",
            waiting_state,
        )
        _wait_until(lambda: waiting_state["session_id"] in controller._claimed_operation_tokens)

        controller.cancel_operation(waiting_state)
        controller.clear_operation(waiting_state)
        assert installer.cancel_calls == 0
        with pytest.raises(StudioJobCancelled):
            waiting.result(timeout=2)

        release_install.set()
        assert active.result(timeout=2) == "Local model is ready."

    assert installer.attempts == 1


def test_generation_progress_copy_does_not_promise_four_outputs() -> None:
    assert "four directions" not in text("en", "status_generating").lower()
    assert "4개" not in text("ko", "status_generating")


def test_refine_oom_evicts_the_failed_runtime_and_next_generation_recovers(
    tmp_path: Path,
) -> None:
    created: list[_Studio] = []

    def factory(preset: str) -> _Studio:
        studio = _Studio(preset, fail_refine_with_oom=not created)
        created.append(studio)
        return studio

    controller = AppController(
        studio_factory=factory,
        workspace_root=tmp_path / "work",
    )
    source = _source(tmp_path / "source.png")
    explored = _explore(controller, source)
    session_root = next((tmp_path / "work").iterdir())
    original_runs = {path.name for path in session_root.iterdir() if path.is_dir()}

    with pytest.raises(StudioAppError, match="GPU ran out of memory"):
        controller.refine(
            explored.state,
            "subtle",
            ("structure",),
            "Sharper paper edges",
        )

    assert created[0].cancel_requested is True
    assert created[0].closed is True
    assert {path.name for path in session_root.iterdir() if path.is_dir()} == original_runs

    recovered = _explore(controller, source)
    assert len(recovered.gallery) == 4
    assert len(created) == 2


def test_native_like_refine_embeds_instruction_in_the_candidate_recipe(
    tmp_path: Path,
) -> None:
    studio = _NativeLikeStudio(PRESET)
    controller = AppController(
        studio_factory=lambda _preset: studio,
        workspace_root=tmp_path / "work",
    )
    explored = _explore(controller, _source(tmp_path / "source.png"))

    refined = controller.refine(
        explored.state,
        "subtle",
        ("structure",),
        "Use sharper paper edges",
    )

    assert len(refined.gallery) == 4
    assert len(studio.vary_calls) == 1
    recipe = studio.vary_calls[0].recipe
    assert recipe.prompt.endswith("Refinement instruction: Use sharper paper edges")
    assert recipe.model_prompt.endswith("Refinement instruction: Use sharper paper edges")


def test_native_like_refine_rejects_a_composed_prompt_over_the_recipe_limit(
    tmp_path: Path,
) -> None:
    studio = _NativeLikeStudio(PRESET)
    controller = AppController(
        studio_factory=lambda _preset: studio,
        workspace_root=tmp_path / "work",
    )
    source = _source(tmp_path / "source.png")
    explored = _explore(controller, source, brief="a" * 9_990)

    with pytest.raises(
        StudioAppError,
        match="10,000-character prompt limit",
    ):
        controller.refine(
            explored.state,
            "subtle",
            ("structure",),
            "Use sharper paper edges",
        )

    assert studio.vary_calls == []
    latest = controller.registry.latest(str(explored.state["session_id"]))
    assert latest is not None
    assert latest.run_id == explored.state["run_id"]


def test_refine_rejects_prompt_metadata_over_the_recipe_limit(
    tmp_path: Path,
) -> None:
    studio = _NativeLikeStudio(
        PRESET,
        prompt_metadata={"padding": "x" * 19_850},
    )
    controller = AppController(
        studio_factory=lambda _preset: studio,
        workspace_root=tmp_path / "work",
    )
    explored = _explore(controller, _source(tmp_path / "source.png"))

    with pytest.raises(
        StudioAppError,
        match="20,000-byte prompt metadata limit",
    ):
        controller.refine(
            explored.state,
            "subtle",
            ("structure",),
            "Use sharper paper edges",
        )

    assert studio.vary_calls == []


def test_refine_rejects_nonreplaceable_recipe_when_backend_drops_instructions(
    tmp_path: Path,
) -> None:
    class NonreplaceableRecipeStudio(_NativeLikeStudio):
        def explore(
            self,
            prepared: str,
            *,
            outputs: int,
            **_: Any,
        ) -> _Study:
            del prepared
            return _Study(outputs, recipe={"opaque": True})

    studio = NonreplaceableRecipeStudio(PRESET)
    controller = AppController(
        studio_factory=lambda _preset: studio,
        workspace_root=tmp_path / "work",
    )
    explored = _explore(controller, _source(tmp_path / "source.png"))

    with pytest.raises(
        StudioAppError,
        match="cannot apply refinement instructions safely",
    ):
        controller.refine(
            explored.state,
            "subtle",
            ("structure",),
            "Use sharper paper edges",
        )

    assert studio.vary_calls == []


@pytest.mark.skipif(
    importlib.util.find_spec("gradio") is None,
    reason="Gradio demo extra is absent",
)
def test_every_long_running_studio_action_has_a_wired_cancel_dependency(
    tmp_path: Path,
) -> None:
    app = build_app(AppController(workspace_root=tmp_path))
    components = {component["id"]: component for component in app.config["components"]}

    def elem_id(component_id: int) -> str | None:
        value = components.get(component_id, {}).get("props", {}).get("elem_id")
        return value if isinstance(value, str) else None

    dependencies = app.config["dependencies"]
    expected_cancel_counts = {
        "stop-action": 4,
        "model-stop-action": 1,
        "simple-model-stop-action": 1,
    }
    for action, expected_count in expected_cancel_counts.items():
        cancel_dependencies = [
            dependency
            for dependency in dependencies
            if (
                next(
                    (elem_id(component_id) for component_id, _event_name in dependency["targets"]),
                    None,
                )
                == action
            )
            and dependency.get("cancels")
        ]
        assert len(cancel_dependencies) == 1
        cancelled = cancel_dependencies[0]["cancels"]
        assert len(cancelled) == expected_count
        assert all(0 <= index < len(dependencies) for index in cancelled)
        assert cancel_dependencies[0]["queue"] is False


@pytest.mark.skipif(
    importlib.util.find_spec("gradio") is None,
    reason="Gradio demo extra is absent",
)
def test_connection_recovery_layer_handles_html_responses_without_raw_error_ui(
    tmp_path: Path,
) -> None:
    app = build_app(AppController(workspace_root=tmp_path))
    recovery = next(
        component["props"]["value"]
        for component in app.config["components"]
        if component.get("props", {}).get("elem_id") == "connection-recovery-host"
    )

    assert 'role="alertdialog"' in recovery
    assert "hidden" in recovery
    assert 'data-connection-action="reload"' in recovery
    assert 'data-connection-action="dismiss"' in recovery
    for signal in (
        "Could not parse server response",
        "Unexpected token",
        "Failed to fetch",
        "Connection to the server was lost",
        "unhandledrejection",
        "MutationObserver",
    ):
        assert signal in STUDIO_JS
    assert "window.location.reload()" in STUDIO_JS
    assert "layer.hidden = true" in STUDIO_JS
    assert STUDIO_JS.lstrip().startswith("(() => {")
    assert STUDIO_JS.rstrip().endswith("})();")
    assert "js" not in app._studio_launch_kwargs
    assert "head" not in app._studio_launch_kwargs
    component = next(
        component
        for component in app.config["components"]
        if component.get("props", {}).get("elem_id") == "connection-recovery-host"
    )
    assert component["props"]["js_on_load"] == STUDIO_JS
    assert app._studio_launch_kwargs["show_error"] is False
    assert (
        "body:has(#connection-recovery-layer:not([hidden])) .toast-wrap"
        in app._studio_launch_kwargs["css"]
    )
