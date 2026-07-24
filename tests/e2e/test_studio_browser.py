from __future__ import annotations

import os
import socket
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from playwright.sync_api import Browser, Page, Playwright, expect, sync_playwright

from aisketcher.studio_app import AppController, build_app

pytestmark = pytest.mark.skipif(
    os.environ.get("AISKETCHER_BROWSER_E2E") != "1",
    reason="set AISKETCHER_BROWSER_E2E=1 after installing Playwright Chromium",
)


def _available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _wait_until_ready(url: str, *, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as response:  # noqa: S310
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.05)
    raise AssertionError(f"Studio did not become ready at {url}")


def _launch_studio(controller: AppController, port: int) -> Any:
    app = build_app(controller, language="en")
    launch_kwargs = dict(app._studio_launch_kwargs)
    launch_kwargs.update(
        server_port=port,
        prevent_thread_lock=True,
        quiet=True,
    )
    app.launch(**launch_kwargs)
    _wait_until_ready(f"http://127.0.0.1:{port}/")
    return app


@pytest.fixture(scope="module")
def studio_url(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    workspace = tmp_path_factory.mktemp("studio-browser")
    controller = AppController(workspace_root=workspace)
    port = _available_port()
    app = _launch_studio(controller, port)
    url = f"http://127.0.0.1:{port}/"
    try:
        yield url
    finally:
        app.close()
        controller.close()


@pytest.fixture(scope="module")
def browser_runtime() -> Iterator[tuple[Playwright, Browser]]:
    runtime = sync_playwright().start()
    browser = runtime.chromium.launch(headless=True)
    try:
        yield runtime, browser
    finally:
        browser.close()
        runtime.stop()


@pytest.fixture
def page(
    browser_runtime: tuple[Playwright, Browser],
    studio_url: str,
) -> Iterator[Page]:
    _runtime, browser = browser_runtime
    context = browser.new_context(viewport={"width": 1920, "height": 1200})
    current = context.new_page()
    current.goto(studio_url, wait_until="domcontentloaded")
    expect(current.get_by_role("button", name="Try the guided sample")).to_be_visible(
        timeout=15_000
    )
    try:
        yield current
    finally:
        context.close()


def _open_guided_sample(page: Page) -> Any:
    page.get_by_role("button", name="Try the guided sample").click()
    cards = page.locator("#result-gallery .gallery-item")
    expect(cards).to_have_count(4, timeout=15_000)
    expect(page.locator("#selected-preview img")).to_be_visible()
    return cards


def test_guided_gallery_is_large_horizontal_and_has_no_inner_scroll(page: Page) -> None:
    cards = _open_guided_sample(page)
    expect(page.locator("#simple-model-plan")).to_contain_text("1.9 GB")
    expect(page.locator("#simple-model-plan")).not_to_contain_text("315 MB")
    boxes = [cards.nth(index).bounding_box() for index in range(4)]

    assert all(box is not None for box in boxes)
    resolved = [box for box in boxes if box is not None]
    assert max(box["y"] for box in resolved) - min(box["y"] for box in resolved) <= 4
    assert all(box["width"] >= 240 for box in resolved)
    assert all(box["height"] >= 190 for box in resolved)

    scroll_state = page.locator("#result-gallery").evaluate(
        """element => {
          const wrap = element.querySelector(".grid-wrap");
          return [element, wrap].filter(Boolean).map((node) => ({
            overflowX: getComputedStyle(node).overflowX,
            overflowY: getComputedStyle(node).overflowY,
            clientHeight: node.clientHeight,
            scrollHeight: node.scrollHeight,
            clientWidth: node.clientWidth,
            scrollWidth: node.scrollWidth,
          }));
        }"""
    )
    assert scroll_state
    for state in scroll_state:
        assert state["overflowX"] not in {"auto", "scroll"}
        assert state["overflowY"] not in {"auto", "scroll"}
        assert state["scrollHeight"] <= state["clientHeight"] + 2
        assert state["scrollWidth"] <= state["clientWidth"] + 2


def test_gallery_preview_close_restores_cards_and_document_layout(page: Page) -> None:
    cards = _open_guided_sample(page)
    cards.first.click()

    preview = page.locator("#result-gallery .preview")
    expect(preview).to_be_visible(timeout=10_000)
    close = preview.get_by_role("button", name="Close")
    expect(close).to_be_visible()
    close.click()

    expect(preview).to_be_hidden(timeout=10_000)
    restored = page.locator("#result-gallery .gallery-item")
    expect(restored).to_have_count(4)
    assert page.evaluate("document.documentElement.style.overflow") in {"", "auto"}
    assert page.evaluate("document.body.style.overflow") in {"", "auto"}
    assert page.locator("#result-gallery > .thumbnails").count() == 0
    restored_boxes = [restored.nth(index).bounding_box() for index in range(4)]
    assert all(box is not None for box in restored_boxes)
    resolved = [box for box in restored_boxes if box is not None]
    assert max(box["y"] for box in resolved) - min(box["y"] for box in resolved) <= 4
    assert all(box["height"] >= 190 for box in resolved)


def test_guided_refine_uses_model_layer_and_language_toggle(page: Page) -> None:
    _open_guided_sample(page)
    page.get_by_role("button", name="Refine this direction").click()

    overlay = page.locator("#guided-refine-overlay")
    expect(overlay).to_be_visible()
    expect(overlay).to_contain_text("Ready to make this direction yours?")
    expect(page.locator(".toast-wrap")).not_to_contain_text("Error")

    overlay.get_by_role("button", name="Keep exploring the sample").click()
    expect(overlay).to_be_hidden()
    page.locator("#language-nav label", has_text="한국어").click()
    expect(page.get_by_role("button", name="가이드 샘플 체험")).to_be_visible()
    expect(page.get_by_role("button", name="이 방향 발전시키기")).to_be_visible()
    expect(page.locator("#simple-model-plan")).to_contain_text("1.9 GB")
    expect(page.locator("#simple-model-plan")).not_to_contain_text("315 MB")
    page.get_by_role("button", name="이 방향 발전시키기").click()
    expect(overlay).to_be_visible()
    expect(overlay).to_contain_text("이 방향을 내 작업으로 발전시킬까요?")


def test_html_response_failure_opens_recovery_layer_without_destroying_results(
    page: Page,
) -> None:
    _open_guided_sample(page)
    page.evaluate(
        """() => {
          window.dispatchEvent(new PromiseRejectionEvent("unhandledrejection", {
            promise: Promise.resolve(),
            reason: new Error(
              "Could not parse server response: SyntaxError: Unexpected token '<'"
            ),
          }));
        }"""
    )

    recovery = page.locator("#connection-recovery-layer")
    expect(recovery).to_be_visible()
    expect(recovery).to_contain_text("This Studio session has ended")
    expect(page.locator("#result-gallery .gallery-item")).to_have_count(4)
    expect(page.locator(".toast-wrap")).to_be_hidden()
    recovery.get_by_role("button", name="Keep this screen").click()
    expect(recovery).to_be_hidden()
    expect(page.locator("#result-gallery .gallery-item")).to_have_count(4)


def test_browser_tabs_are_isolated_while_one_tab_refresh_recovers_its_run(
    browser_runtime: tuple[Playwright, Browser],
    tmp_path: Path,
) -> None:
    _runtime, browser = browser_runtime
    controller = AppController(workspace_root=tmp_path)
    port = _available_port()
    url = f"http://127.0.0.1:{port}/"
    app = _launch_studio(controller, port)
    context = browser.new_context(viewport={"width": 1920, "height": 1200})
    first = context.new_page()
    second = context.new_page()
    try:
        first.goto(url, wait_until="domcontentloaded")
        second.goto(url, wait_until="domcontentloaded")
        expect(first.get_by_role("button", name="Try the guided sample")).to_be_visible(
            timeout=15_000
        )
        expect(second.get_by_role("button", name="Try the guided sample")).to_be_visible(
            timeout=15_000
        )
        _open_guided_sample(first)
        _open_guided_sample(second)

        session_ids = {record.session_id for record in controller.registry._records.values()}
        assert len(session_ids) == 2

        first.reload(wait_until="domcontentloaded")
        expect(first.locator("#result-gallery .gallery-item")).to_have_count(
            4,
            timeout=15_000,
        )
        assert {
            record.session_id for record in controller.registry._records.values()
        } == session_ids
    finally:
        context.close()
        app.close()
        controller.close()


def test_server_restart_recovers_same_address_without_losing_old_results(
    browser_runtime: tuple[Playwright, Browser],
    tmp_path: Path,
) -> None:
    _runtime, browser = browser_runtime
    controller = AppController(workspace_root=tmp_path)
    port = _available_port()
    url = f"http://127.0.0.1:{port}/"
    app = _launch_studio(controller, port)
    restarted_app = None
    context = browser.new_context(viewport={"width": 1920, "height": 1200})
    page = context.new_page()
    try:
        page.goto(url, wait_until="domcontentloaded")
        expect(page.get_by_role("button", name="Try the guided sample")).to_be_visible(
            timeout=15_000
        )
        _open_guided_sample(page)

        app.close()
        app = None
        page.get_by_role("button", name="Refine this direction").click()

        recovery = page.locator("#connection-recovery-layer")
        expect(recovery).to_be_visible(timeout=10_000)
        expect(recovery).to_contain_text("This Studio session has ended")
        expect(page.locator("#result-gallery .gallery-item")).to_have_count(4)

        restarted_app = _launch_studio(controller, port)
        recovery.get_by_role("button", name="Reload this address").click()
        expect(page.get_by_role("button", name="Try the guided sample")).to_be_visible(
            timeout=15_000
        )
        expect(page.locator("#connection-recovery-layer")).to_be_hidden()
    finally:
        context.close()
        if restarted_app is not None:
            restarted_app.close()
        if app is not None:
            app.close()
        controller.close()
