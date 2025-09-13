"""Test fixtures and helpers for the hass-opnsense integration.

This module provides pytest fixtures, fake clients, and monkeypatch helpers
used across the integration's test suite to avoid network IO, neutralize
background tasks, and simplify Home Assistant testing.
"""

import asyncio
from collections.abc import Callable
import contextlib
import importlib
import inspect
import logging
import sys
import types
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry
from pytest_homeassistant_custom_component.plugins import get_scheduled_timer_handles
from yarl import URL

import custom_components.opnsense as _init_mod
from custom_components.opnsense import pyopnsense as _pyopnsense_mod
from custom_components.opnsense.const import CONF_DEVICE_UNIQUE_ID
import homeassistant.core as ha_core

logger = logging.getLogger(__name__)


@pytest.fixture
async def hass_with_running_loop(ph_hass) -> MagicMock:
    """Provide a hass-like object wired to the running event loop.

    Prefer the real `ph_hass` fixture when available so tests interact
    with a real HomeAssistant instance provided by pytest-homeassistant.
    When `ph_hass` is not present, fall back to a `MagicMock` with a
    running loop and minimal `data` mapping.
    """
    # Prefer the provided `ph_hass` fixture when available. If tests or
    # environments do not provide ph_hass, the fixture will be None and we
    # fallback below.
    hass_local = ph_hass if ph_hass is not None else MagicMock(spec=ha_core.HomeAssistant)

    # Ensure hass has a running loop attribute for tests that expect it
    # Set loop if a running loop is available - suppress RuntimeError when not
    with contextlib.suppress(RuntimeError):
        hass_local.loop = asyncio.get_running_loop()

    # Ensure a minimal data mapping and config_entries compatibility
    hass_local.data = getattr(hass_local, "data", {}) or {}
    _ensure_hass_compat(hass_local)
    return hass_local


@pytest.fixture
def hass_without_loop(ph_hass) -> MagicMock:
    """Provide a hass-like object with no event loop (loop is None).

    Prefer the `ph_hass` fixture when available and simply set its
    `loop` attribute to `None` for tests that expect synchronous
    behaviour. If `ph_hass` is not provided, fall back to a MagicMock
    with the same minimal shape.
    """
    if ph_hass is not None:
        # Create a lightweight proxy MagicMock that exposes the minimal
        # attributes tests expect but does not mutate the real ph_hass
        # (avoids breaking teardown of the real hass fixture).
        proxy = MagicMock(spec=ha_core.HomeAssistant)
        proxy.config_entries = getattr(ph_hass, "config_entries", MagicMock())
        proxy.services = getattr(ph_hass, "services", MagicMock())
        proxy.async_create_task = getattr(ph_hass, "async_create_task", MagicMock())
        # Copy a shallow copy of data so tests can manipulate it safely
        proxy.data = dict(getattr(ph_hass, "data", {}) or {})
        proxy.data.setdefault("integrations", {})
        proxy.loop = None
        with contextlib.suppress(Exception):
            _ensure_hass_compat(proxy)
        return proxy

    # Fallback for environments without ph_hass: return a standalone MagicMock
    hass_local = MagicMock(spec=ha_core.HomeAssistant)
    hass_local.loop = None
    hass_local.data = {"integrations": {}}
    _ensure_hass_compat(hass_local)
    return hass_local


# expose the pyopnsense module under the plain name for tests that
# import the fixture and expect `pyopnsense` to be available.
pyopnsense = _pyopnsense_mod


def map_hass_components_to_custom_components(
    domain: str = "opnsense", modules: tuple | None = None
) -> object:
    """Map `homeassistant.components.<domain>.*` to local `custom_components.<domain>.*`.

    Returns the created homeassistant components package module for optional use by tests.
    This is a best-effort helper: missing modules are ignored.
    """
    modules = modules or (
        "binary_sensor",
        "device_tracker",
        "sensor",
        "switch",
        "update",
        "services",
    )
    pkg_name = f"homeassistant.components.{domain}"

    # Record prior values so cleanup can restore them
    prev_pkg = sys.modules.get(pkg_name)
    injected_module_keys: dict[str, object | None] = {}
    injected_attrs: dict[str, object | None] = {}

    hc_pkg = types.ModuleType(pkg_name)

    # Install the package module, recording any previous value
    injected_module_keys[pkg_name] = prev_pkg
    sys.modules[pkg_name] = hc_pkg

    # Track whether we raised while importing so we can still cleanup
    raised: Exception | None = None
    try:
        for m in modules:
            key = f"{pkg_name}.{m}"
            try:
                loaded = importlib.import_module(f"custom_components.{domain}.{m}")
                # Record previous value for this key (or None)
                injected_module_keys[key] = sys.modules.get(key)
                sys.modules[key] = loaded
                # Record previous attribute on the package (or None)
                injected_attrs[m] = getattr(hc_pkg, m, None)
                setattr(hc_pkg, m, loaded)
            except Exception:  # noqa: BLE001 - best-effort mapping
                # Ignore missing modules; continue mapping others
                continue
    except Exception as exc:  # noqa: BLE001
        raised = exc
        # Fall through to finally to ensure cleanup runs if desired by caller
    finally:

        def _cleanup() -> None:
            """Restore sys.modules and package attributes to their prior state.

            This is safe to call multiple times and guards against missing keys.
            """
            # Restore or remove module entries we injected
            for k, prev in list(injected_module_keys.items()):
                if prev is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = prev  # type: ignore[assignment]

            # Restore or remove attributes on the package module
            pkg = sys.modules.get(pkg_name)
            if pkg is not None:
                for attr, prev in list(injected_attrs.items()):
                    if prev is None:
                        with contextlib.suppress(Exception):
                            delattr(pkg, attr)
                    else:
                        setattr(pkg, attr, prev)

        # Attach the cleanup and a context-manager provider to the returned
        # package module so callers can opt-in to automatic cleanup.
        setattr(hc_pkg, "_cleanup", _cleanup)

        @contextlib.contextmanager
        def _as_context():
            try:
                yield hc_pkg
            finally:
                _cleanup()

        setattr(hc_pkg, "_as_context", _as_context)

        # If an exception occurred during mapping, propagate it after
        # attaching the cleanup so callers can still perform explicit cleanup.
        if raised:
            raise raised

    return hc_pkg


def _completed_future_result() -> Any:
    """Return an already-completed awaitable yielding None.

    If a running loop is available, return a Future set to result=None.
    Otherwise return a minimal awaitable object that behaves like a done task.
    """
    try:
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        fut.set_result(None)
    except RuntimeError:

        class _DoneTask:
            def done(self) -> bool:  # pragma: no cover - trivial
                return True

            def cancel(self) -> None:  # pragma: no cover - trivial
                return None

            def cancelled(self) -> bool:  # pragma: no cover - trivial
                return False

            def result(self) -> None:  # pragma: no cover - trivial
                return None

            def exception(self) -> None:  # pragma: no cover - trivial
                return None

            def add_done_callback(self, cb):  # pragma: no cover - trivial
                with contextlib.suppress(Exception):
                    cb(self)

            def __await__(self):  # pragma: no cover - trivial
                if False:
                    yield None

        return _DoneTask()
    else:
        return fut


def _is_pyopnsense_background_coro(coro: Any) -> bool:
    """Detect if a coroutine is a pyopnsense background worker.

    This intentionally uses a narrow heuristic to avoid false positives.
    Return True only when the coroutine's function name matches a known
    background worker (for example `_monitor_queue` or `_process_queue`)
    or when the coroutine is a bound method whose `self` is an
    :class:`pyopnsense.OPNsenseClient` instance.

    Avoid matching based on the coroutine's module name or other globals
    since those can produce false positives in tests that import or wrap
    pyopnsense helpers.
    """
    # Fast-path: match known background worker names on the code object
    with contextlib.suppress(AttributeError):
        name = getattr(getattr(coro, "cr_code", None), "co_name", "")
        if name in {"_monitor_queue", "_process_queue"}:
            return True

    # Bound-method detection via frame locals: prefer explicit type check
    frame = getattr(coro, "cr_frame", None)
    if frame:
        with contextlib.suppress(AttributeError, TypeError):
            locs = getattr(frame, "f_locals", {}) or {}
            self_obj = locs.get("self")
            if (
                self_obj is not None
                and getattr(pyopnsense, "OPNsenseClient", None) is not None
                and isinstance(self_obj, pyopnsense.OPNsenseClient)
            ):
                return True

    return False


def _ensure_async_create_task_mock(real, side_effect):
    """Ensure `real.async_create_task` is a MagicMock with the given side_effect.

    Attempt three strategies in order (matching the original logic):
    1. Direct assignment: real.async_create_task = MagicMock(side_effect=...)
    2. Use object.__setattr__ to bypass attribute protections.
    3. If an existing callable exists, wrap it with MagicMock(side_effect=lambda coro: orig(coro)).
    """
    with contextlib.suppress(AttributeError, TypeError):
        real.async_create_task = MagicMock(side_effect=side_effect)
    if not hasattr(real, "async_create_task") or not isinstance(
        getattr(real, "async_create_task", None), MagicMock
    ):
        # Try object.__setattr__ in case of attribute protections.
        with contextlib.suppress(AttributeError, TypeError):
            object.__setattr__(real, "async_create_task", MagicMock(side_effect=side_effect))
    if not hasattr(real, "async_create_task") or not isinstance(
        getattr(real, "async_create_task", None), MagicMock
    ):
        # As a last resort, wrap an existing callable if present.
        orig = getattr(real, "async_create_task", None)
        if callable(orig):
            with contextlib.suppress(AttributeError, TypeError):
                object.__setattr__(
                    real,
                    "async_create_task",
                    MagicMock(side_effect=lambda coro, *a, **k: orig(coro, *a, **k)),
                )


def _ensure_hass_compat(hass_obj: Any) -> None:
    """Mutate a hass-like object to provide lightweight config_entries.

    Behaviour used across these tests.

    The function ensures hass.data["integrations"] exists and attaches
    safe no-op async methods and an update_entry shim that mutates
    MockConfigEntry objects in-place to avoid hitting HA internals.
    """
    hass_obj.data = getattr(hass_obj, "data", {}) or {}
    hass_obj.data.setdefault("integrations", {})

    async def cfg_async_forward(entry, platforms):
        return True

    async def cfg_async_unload(entry, platforms):
        return True

    async def cfg_async_reload(entry_id):
        return None

    def _async_update_entry(entry, data=None, options=None, version=None, unique_id=None, **kwargs):
        if data is not None:
            object.__setattr__(entry, "data", data)
        if options is not None:
            object.__setattr__(entry, "options", options)
        if unique_id is not None:
            object.__setattr__(entry, "unique_id", unique_id)
        if version is not None:
            object.__setattr__(entry, "version", version)
        return True

    cfg = getattr(hass_obj, "config_entries", None)
    if cfg is None:
        hass_obj.config_entries = MagicMock()
        cfg = hass_obj.config_entries

    cfg.async_forward_entry_setups = cfg_async_forward
    cfg.async_unload_platforms = cfg_async_unload
    cfg.async_reload = cfg_async_reload
    cfg.async_update_entry = _async_update_entry
    # Ensure a simple backing store for known entries so tests can
    # register MockConfigEntry instances without touching HA internals.
    if not hasattr(cfg, "_entries") or getattr(cfg, "_entries") is None:
        # Attempt normal assignment, but do not raise if the object disallows it.
        with contextlib.suppress(Exception):
            cfg._entries = {}
        # If the direct assignment failed (for example on some MagicMock
        # objects), attempt to set the attribute via object.__setattr__ as a
        # best-effort fallback.
        with contextlib.suppress(Exception):
            object.__setattr__(cfg, "_entries", {})

    # Provide a lightweight async_get_known_entry for compatibility with
    if not hasattr(cfg, "async_get_known_entry"):

        def _async_get_known_entry(entry_id: str):
            return cfg._entries.get(entry_id)

        with contextlib.suppress(Exception):
            cfg.async_get_known_entry = _async_get_known_entry
        if getattr(cfg, "async_get_known_entry", None) is not _async_get_known_entry:
            with contextlib.suppress(Exception):
                object.__setattr__(cfg, "async_get_known_entry", _async_get_known_entry)


@pytest.fixture
def coordinator_capture(coordinator_factory: Callable[..., Any]):
    """Provide a reusable capture for created coordinator instances.

    Exposes:
      - instances: list of created instances
      - factory(coord_cls=None): returns a creator that appends instances to the list
    """

    class _C:
        def __init__(self) -> None:
            self.instances: list[Any] = []

        def factory(self, coord_cls=None):
            def _create(**kwargs: Any):
                return coordinator_factory(
                    cls=coord_cls,
                    capture_list=self.instances,
                    stub_async=True,
                    **kwargs,
                )

            return _create

    return _C()


@pytest.fixture
async def make_client():
    """Return a factory that constructs an OPNsenseClient for tests."""

    clients: list[pyopnsense.OPNsenseClient] = []

    def _make(
        session: aiohttp.ClientSession | None = None,
        username: str = "u",
        password: str = "p",
        url: str = "http://localhost",
    ) -> pyopnsense.OPNsenseClient:
        # Tests should not pass a real aiohttp.ClientSession. If session is
        # omitted, substitute the test FakeClientSession to avoid passing None
        # into the production client which expects a session-like object.
        if session is None:
            session = cast("aiohttp.ClientSession", FakeClientSession())
        client = pyopnsense.OPNsenseClient(
            url=url, username=username, password=password, session=session
        )
        clients.append(client)
        return client

    try:
        yield _make
    finally:
        # Ensure all created clients are closed to avoid leaking background tasks.
        for c in clients:
            with contextlib.suppress(Exception):
                await c.async_close()


@pytest.fixture
def coordinator(coordinator_factory: Callable[..., Any]):
    """Provide a lightweight coordinator mock for tests (MagicMock)."""
    return coordinator_factory(stub_async=True)


@pytest.fixture
def fake_client(fake_client_factory):
    """Backwards-compatible fixture returning the general FakeClient class."""

    def _make(
        device_id: object = "dev1",
        firmware_version: str = "99.0",
        telemetry: dict | None = None,
        close_result: bool = True,
    ):
        return fake_client_factory(
            flow=False,
            device_id=device_id,
            firmware_version=firmware_version,
            telemetry=telemetry,
            close_result=close_result,
        )

    return _make


@pytest.fixture
def fake_flow_client(fake_client_factory):
    """Fixture returning either the FakeFlowClient class or a runtime instance.

    Usage:
      client_cls = fake_flow_client()           # returns class (legacy behavior)
      inst = fake_flow_client(runtime=True)     # returns a prepared instance
    """

    def _make(
        device_id: str = "unique-id",
        firmware: str = "25.1",
        plugin_installed: bool = False,
        runtime: bool = False,
    ):
        cls = fake_client_factory(
            flow=True,
            device_id=device_id,
            firmware=firmware,
            plugin_installed=plugin_installed,
        )
        if not runtime:
            return cls

        inst = cls()
        inst._closed = False

        async def _aclose():
            inst._closed = True
            return True

        inst.async_close = _aclose
        return inst

    return _make


@pytest.fixture
def patch_asyncio_sleep(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Patch asyncio.sleep with an AsyncMock and return the mock.

    Tests that need to avoid real sleeps or assert sleep calls can request
    this fixture to get a ready-to-use spy.
    """
    spy = AsyncMock(return_value=None)
    monkeypatch.setattr(asyncio, "sleep", spy)
    return spy


@pytest.fixture
def patch_async_create_clientsession(monkeypatch: pytest.MonkeyPatch):
    """Return a helper that patches config_flow.async_create_clientsession.

    Usage:
        patch_async_create_clientsession(lambda *a, **k: MagicMock())
    """

    def _patch(factory=lambda *a, **k: MagicMock()):
        monkeypatch.setattr(
            "custom_components.opnsense.config_flow.async_create_clientsession",
            factory,
            raising=False,
        )

        return factory

    return _patch


@pytest.fixture
def patch_registry_async_get(monkeypatch: pytest.MonkeyPatch):
    """Return a helper to patch device/entity registry async_get functions.

    Usage:
        patch_registry_async_get(device_reg=my_dev_reg, entity_reg=my_ent_reg)
    """

    def _patch(device_reg=None, entity_reg=None):
        if device_reg is not None:
            with contextlib.suppress(Exception):
                monkeypatch.setattr(
                    _init_mod.dr, "async_get", lambda hass: device_reg, raising=False
                )
            # Fallback: also try patching by import path in case the direct
            # attribute access is not available in some test environments.
            with contextlib.suppress(Exception):
                monkeypatch.setattr(
                    "custom_components.opnsense.dr.async_get",
                    lambda hass: device_reg,
                    raising=False,
                )
        if entity_reg is not None:
            with contextlib.suppress(Exception):
                monkeypatch.setattr(
                    _init_mod.er, "async_get", lambda hass: entity_reg, raising=False
                )
            with contextlib.suppress(Exception):
                monkeypatch.setattr(
                    "custom_components.opnsense.er.async_get",
                    lambda hass: entity_reg,
                    raising=False,
                )

    return _patch


@pytest.fixture
def patch_registry_entries(monkeypatch: pytest.MonkeyPatch):
    """Helper to patch async_entries_for_config_entry for entity/device registries.

    Usage:
        patch_registry_entries(entity_entries=lambda reg, entry_id: [...], device_entries=lambda reg, entry_id: [...])
    """

    def _patch(entity_entries=None, device_entries=None):
        # Patch the imported module attributes when available, fall back to string paths.
        if entity_entries is not None:
            with contextlib.suppress(Exception):
                monkeypatch.setattr(
                    _init_mod.er, "async_entries_for_config_entry", entity_entries, raising=False
                )
            with contextlib.suppress(Exception):
                monkeypatch.setattr(
                    "custom_components.opnsense.er.async_entries_for_config_entry",
                    entity_entries,
                    raising=False,
                )
        if device_entries is not None:
            with contextlib.suppress(Exception):
                monkeypatch.setattr(
                    _init_mod.dr, "async_entries_for_config_entry", device_entries, raising=False
                )
            with contextlib.suppress(Exception):
                monkeypatch.setattr(
                    "custom_components.opnsense.dr.async_entries_for_config_entry",
                    device_entries,
                    raising=False,
                )

    return _patch


@pytest.fixture
def patch_opnsense_client(monkeypatch: pytest.MonkeyPatch):
    """Return a helper to patch one or more module-level OPNsenseClient targets.

    Usage:
        patch_opnsense_client([
            "custom_components.opnsense.config_flow.OPNsenseClient",
            "custom_components.opnsense.__init__.OPNsenseClient",
        ], client_cls)

    """

    def _patch(targets: list[str], client_factory: object):
        for t in targets:
            monkeypatch.setattr(t, client_factory, raising=False)
        return client_factory

    return _patch


@pytest.fixture
def patch_init_opnsense_client(monkeypatch: pytest.MonkeyPatch):
    """Helper to patch the integration module's OPNsenseClient target directly.

    Usage:
        patch_init_opnsense_client(lambda **k: fake_client())
    """

    def _patch(client_factory: object):
        # Wrap the provided client_factory so that calling OPNsenseClient(...)
        # always returns an instance. Some tests pass a factory that returns
        # a client class (not an instance), so we detect that case and
        # instantiate it with the same args.
        def _wrapper(*args, **kwargs):
            res = client_factory(*args, **kwargs)
            # If the factory returned a class, instantiate it.
            try:
                if inspect.isclass(res):
                    return res(*args, **kwargs)
            except (TypeError, AttributeError):
                # If something unexpected happens inspecting/instantiating,
                # fall back to returning the result as-is.
                pass
            return res

        # Patch the imported integration module object for consistency across tests
        monkeypatch.setattr(_init_mod, "OPNsenseClient", _wrapper, raising=False)
        return client_factory

    return _patch


@pytest.fixture
def patch_cf_opnsense_client(monkeypatch: pytest.MonkeyPatch):
    """Convenience helper to patch the config_flow OPNsenseClient target.

    Usage: patch_cf_opnsense_client(client_cls_or_factory)
    """

    def _patch(client_factory: object):
        # Patch both the module object and the import path string for resilience.
        with contextlib.suppress(Exception):
            monkeypatch.setattr(
                "custom_components.opnsense.config_flow.OPNsenseClient",
                client_factory,
                raising=False,
            )
        with contextlib.suppress(Exception):
            monkeypatch.setattr(
                importlib.import_module("custom_components.opnsense.config_flow"),
                "OPNsenseClient",
                client_factory,
                raising=False,
            )
        return client_factory

    return _patch


@pytest.fixture
def patch_issue_registry(monkeypatch: pytest.MonkeyPatch):
    """Helper to patch async_create_issue and async_delete_issue across modules.

    Usage: patch_issue_registry(create_fn=..., delete_fn=...)
    """

    def _patch(create_fn=None, delete_fn=None):
        if create_fn is not None:
            with contextlib.suppress(Exception):
                monkeypatch.setattr(_init_mod.ir, "async_create_issue", create_fn, raising=False)
            with contextlib.suppress(Exception):
                monkeypatch.setattr(
                    "custom_components.opnsense.ir.async_create_issue", create_fn, raising=False
                )
        if delete_fn is not None:
            with contextlib.suppress(Exception):
                monkeypatch.setattr(_init_mod.ir, "async_delete_issue", delete_fn, raising=False)
            with contextlib.suppress(Exception):
                monkeypatch.setattr(
                    "custom_components.opnsense.ir.async_delete_issue", delete_fn, raising=False
                )

    return _patch


@pytest.fixture
def patch_services_get_clients(monkeypatch: pytest.MonkeyPatch):
    """Helper to patch services._get_clients across module/import paths.

    Usage: patch_services_get_clients(fake_get_coroutine)
    """

    def _patch(fake_get):
        with contextlib.suppress(Exception):
            monkeypatch.setattr(
                "custom_components.opnsense.services._get_clients", fake_get, raising=False
            )
        with contextlib.suppress(Exception):
            monkeypatch.setattr(
                importlib.import_module("custom_components.opnsense.services"),
                "_get_clients",
                fake_get,
                raising=False,
            )
        return fake_get

    return _patch


@pytest.fixture
def make_config_entry():
    """Return a factory for creating MockConfigEntry instances for tests.

    Usage:
        entry = make_config_entry()
        entry2 = make_config_entry(data={...}, title="MyTitle", unique_id="id", entry_id="eid", version=2, options={})

    Keyword args supported:
      - data: dict for entry.data (defaults to {CONF_DEVICE_UNIQUE_ID: 'test-device-123'})
      - title: entry title
      - unique_id: entry.unique_id
      - entry_id: entry.entry_id
      - version: entry.version
      - options: entry.options
      - runtime_data: value to assign to entry.runtime_data (default: MagicMock())
    """

    def _make(
        data: dict | None = None,
        *,
        title: str | None = None,
        unique_id: str | None = None,
        entry_id: str | None = None,
        version: int | None = None,
        options: dict | None = None,
        runtime_data: Any | None = None,
    ) -> MockConfigEntry:
        data = data or {CONF_DEVICE_UNIQUE_ID: "test-device-123"}
        entry = MockConfigEntry(
            domain="opnsense", data=data, title=(title if title is not None else "OPNSense Test")
        )

        # Apply optional attributes using object.__setattr__ to bypass property protections.
        if unique_id is not None:
            object.__setattr__(entry, "unique_id", unique_id)
        if entry_id is not None:
            object.__setattr__(entry, "entry_id", entry_id)
        if version is not None:
            object.__setattr__(entry, "version", version)
        if options is not None:
            object.__setattr__(entry, "options", options)
        # runtime_data default is a MagicMock to support attribute-style access in tests
        entry.runtime_data = runtime_data if runtime_data is not None else MagicMock()
        return entry

    return _make


@pytest.fixture
def ph_hass(request, hass=None):
    """Safe hass-like fixture: prefer real PHCC `hass` when available.

    Prefer the pytest-injected `hass` fixture when the pytest-homeassistant-
    custom-component plugin is present. To support environments where the
    plugin is absent (or where fixture injection order yields an async
    generator), fall back to using `request.getfixturevalue("hass")` only
    as a last resort; if that still isn't available, return a MagicMock
    that provides the minimal attributes tests expect.
    """

    # Helper used to schedule coroutines on the running loop when possible.
    def _schedule_or_return(coro):
        try:
            loop = asyncio.get_running_loop()
            return loop.create_task(coro)
        except RuntimeError:
            # No running loop available (unlikely in async tests); fall
            # back to returning the coroutine so callers can decide.
            return coro

    # If pytest injected a `hass` fixture, prefer it (but avoid advancing
    # async-generator fixtures here). This lets pytest supply the real
    # PHCC hass instance when available without calling getfixturevalue.
    real = hass
    if real is not None:
        # If the injected fixture is an async-generator object, we must not
        # advance it here because its lifecycle is managed by the plugin
        # (treat as unavailable and fall back below).
        # Reuse helper to ensure async_create_task is a MagicMock so tests
        # can assert `.called` etc.
        _ensure_async_create_task_mock(real, _schedule_or_return)
        # Apply compatibility shims so the real `hass` fixture exposes a
        # minimal `_entries` backing store and helper methods used by
        # tests (this avoids per-test boilerplate).
        with contextlib.suppress(Exception):
            # Best-effort: if we cannot patch the real hass, continue
            # and let tests that need more control set things up.
            _ensure_hass_compat(real)
        return real

    # No injected hass or injected hass unusable; try the legacy fallback
    # of requesting the fixture by name. Only call getfixturevalue as a
    # safety net when injection did not occur.
    try:
        real = request.getfixturevalue("hass")
        if inspect.isasyncgen(real):
            real = None
        if real is not None:
            # Mirror the same robust assignment logic for the plugin-provided
            # hass fixture path using the helper.
            _ensure_async_create_task_mock(real, _schedule_or_return)
            with contextlib.suppress(Exception):
                _ensure_hass_compat(real)
            return real
    except pytest.FixtureLookupError:
        # No PHCC hass available; will return MagicMock fallback below.
        pass

    # No real hass fixture available; return a MagicMock fallback.
    m = MagicMock()
    m.config_entries = MagicMock()
    m.data = {}

    class _Services:
        def __init__(self):
            self._reg: dict[str, dict[str, Any]] = {}

        def async_register(self, domain: str, service: str, service_func, schema=None, **_):
            self._reg.setdefault(domain, {})[service] = service_func

        def async_services(self) -> dict[str, dict[str, Any]]:
            return {d: dict(svcs) for d, svcs in self._reg.items()}

        def async_clear(self) -> None:
            self._reg.clear()

    m.services = _Services()
    # Mirror HomeAssistant API used by the integration/tests.
    m.async_create_task = MagicMock(side_effect=_schedule_or_return)
    # provide a loop wrapper that cancels scheduled timer handles immediately
    # so the pytest-homeassistant-custom-component plugin does not report
    # lingering timers during test teardown.
    try:
        real_loop = asyncio.get_running_loop()
    except RuntimeError:
        real_loop = asyncio.new_event_loop()

    class FakeLoop:
        def __init__(self, loop):
            self._loop = loop

        def call_later(self, delay, callback, *args):
            handle = self._loop.call_later(delay, callback, *args)
            with contextlib.suppress(Exception):
                handle.cancel()
            return handle

        def __getattr__(self, name):
            return getattr(self._loop, name)

    m.loop = FakeLoop(real_loop)

    # Best-effort: don't fail fixture creation if shim cannot be applied
    with contextlib.suppress(Exception):
        _ensure_hass_compat(m)

    return m


@pytest.fixture
def patch_async_call_later(monkeypatch: pytest.MonkeyPatch):
    """Patch async_call_later to return a remover that immediately clears the delay.

    Returns the fake function for optional assertions.
    """

    def fake_async_call_later(hass, delay, action):
        def remover():
            action(None)

        return remover

    monkeypatch.setattr("custom_components.opnsense.switch.async_call_later", fake_async_call_later)
    return fake_async_call_later


@pytest.fixture
def coordinator_factory() -> Callable[..., Any]:
    """Factory for creating lightweight coordinator instances for tests.

    Usage:
        create = coordinator_factory
        coord = create()  # MagicMock
    coord = create(kind="dummy")  # FakeCoordinator()
        coord = create(cls=MyCoordinatorClass, device_tracker_coordinator=True)

        Keyword args supported:
    - cls: custom class to instantiate (default: MagicMock or FakeCoordinator if kind="dummy")
      - kind: "mock" (default) or "dummy"; ignored when cls is provided
      - capture_list: optional list to which the created instance will be appended
      - device_tracker_coordinator: if True, set the _is_device_tracker flag on the instance
      - stub_async: when True and using MagicMock, auto-stub async methods
        (async_config_entry_first_refresh, async_shutdown) with AsyncMock
      - other kwargs are forwarded to the class constructor
    """

    def _make(
        *,
        cls: type | None = None,
        kind: str = "mock",
        capture_list: list | None = None,
        stub_async: bool = False,
        **kwargs: Any,
    ) -> Any:
        # Resolve class to instantiate
        if cls is None:
            if kind == "dummy":
                # Define a lightweight FakeCoordinator class inline so tests can
                # request `kind="dummy"` without relying on a module-level
                # class. This mirrors the previous module-level implementation
                # but keeps the definition local to the factory.
                class FakeCoordinator(MagicMock):
                    def __init__(self, **kwargs):
                        super().__init__()
                        self._is_device_tracker = kwargs.get("device_tracker_coordinator", False)

                    async def async_config_entry_first_refresh(self):
                        self.refreshed = True
                        return True

                    async def async_shutdown(self):
                        self.shut = True
                        return True

                cls = FakeCoordinator
            else:
                cls = MagicMock

        # Peek without consuming so the kw is still forwarded to the class
        _is_dt = bool(kwargs.get("device_tracker_coordinator", False))
        inst = cls(**kwargs)

        # Mirror existing tests which sometimes rely on this flag
        if _is_dt:
            with contextlib.suppress(Exception):
                setattr(inst, "_is_device_tracker", True)

        if capture_list is not None:
            capture_list.append(inst)

        # Optionally stub common async methods on MagicMock instances
        if stub_async and isinstance(inst, MagicMock):
            for name, default in (
                ("async_config_entry_first_refresh", True),
                ("async_shutdown", True),
            ):
                meth = getattr(inst, name, None)
                if not inspect.iscoroutinefunction(meth) and not isinstance(meth, AsyncMock):
                    setattr(inst, name, AsyncMock(return_value=default))

        return inst

    return _make


@pytest.fixture
def fake_client_factory():
    """Unified factory for creating FakeClient classes used in tests.

    Supports two modes:
      - flow=True: returns a FakeFlowClient class used by config/option flow tests.
      - flow=False: returns a general FakeClient class for other tests.
    """

    def _make(
        *,
        flow: bool = False,
        device_id: object = "dev1",
        firmware: str | None = None,
        firmware_version: str | None = None,
        telemetry: dict | None = None,
        close_result: bool = True,
        plugin_installed: bool = False,
    ):
        # Back-compat: allow either firmware or firmware_version kw
        fw = (
            firmware if firmware is not None else (firmware_version if firmware_version else "99.0")
        )

        if flow:

            class FakeFlowClient:
                """Configurable fake client for flow tests (with last_instance tracking)."""

                last_instance: "FakeFlowClient | None" = None

                def __init__(self, *args, **kwargs):
                    FakeFlowClient.last_instance = self
                    self._is_plugin_called = 0
                    self._device_id = device_id
                    self._firmware = fw
                    self._plugin_installed = plugin_installed

                async def get_host_firmware_version(self) -> str:
                    return self._firmware

                async def set_use_snake_case(self, initial: bool = False) -> None:
                    return None

                async def get_system_info(self) -> dict:
                    return {"name": "OPNsense"}

                async def get_device_unique_id(self) -> str:
                    return cast("str", self._device_id)

                async def is_plugin_installed(self) -> bool:
                    self._is_plugin_called += 1
                    return self._plugin_installed

            return FakeFlowClient

        class FakeClient:
            def __init__(self, **kwargs):
                # prefer explicit args passed to the fixture factory above
                self._device_id = device_id
                self._firmware = fw
                self._telemetry = telemetry or {}
                self._close_result = close_result

                # state for query counts used by coordinator tests
                self._query_counts_reset = False
                self._query_counts = (1, 1)

            async def get_device_unique_id(self):
                return self._device_id

            async def get_host_firmware_version(self):
                return self._firmware

            async def async_close(self):
                return self._close_result

            async def get_telemetry(self):
                return self._telemetry

            async def set_use_snake_case(self, initial: bool = False):
                return True

            async def reset_query_counts(self):
                # mark reset and return None (used by coordinator)
                self._query_counts_reset = True

            async def get_query_counts(self):
                return self._query_counts

            async def get_interfaces(self):
                return {"eth0": {"inbytes": 200, "outbytes": 100}}

            async def get_openvpn(self):
                return {"servers": {}}

            async def get_wireguard(self):
                return {"servers": {}}

        return FakeClient

    return _make


@pytest.fixture
def fake_stream_response_factory():
    r"""Return a factory that constructs a fake streaming response.

    Usage:
        resp = fake_stream_response_factory([b'data: {...}\n\n', b'data: {...}\n\n'])
        session.get = lambda *a, **k: resp

    The returned object implements:
      - .status / .reason / .ok
      - async context manager __aenter__/__aexit__
      - .content.iter_chunked(n) async generator yielding provided chunks
    """

    def _make(chunks: list[bytes], status: int = 200, reason: str = "OK", ok: bool = True):
        class _Resp:
            def __init__(self):
                self.status = status
                self.reason = reason
                self.ok = ok

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            @property
            def content(self):
                class C:
                    def __init__(self, chunks):
                        self._chunks = chunks

                    async def iter_chunked(self, _n):
                        for c in self._chunks:
                            yield c

                return C(list(chunks))

        return _Resp()

    return _make


@pytest.fixture
def fake_http_error_response_factory():
    """Return a factory that constructs a pre-populated HTTP response for error cases.

    The returned object implements:
      - .status / .reason / .ok
      - .request_info.real_url, .history, .headers
      - async context manager __aenter__/__aexit__
      - .json(content_type=None) async coroutine
      - .content.iter_chunked(n) async generator
    """

    def _make(
        status: int = 500,
        ok: bool = False,
        json_ret: Any | None = None,
        chunks: list[bytes] | None = None,
        reason: str = "Err",
    ):
        class _Resp:
            def __init__(self):
                self.status = status
                self.reason = reason
                self.ok = ok

                class RI:
                    real_url = URL("http://localhost")

                self.request_info = RI()
                self.history = []
                self.headers = {}
                self._json_ret = json_ret if json_ret is not None else {"x": 1}
                self._chunks = list(chunks or [b"data:{}\n\n"])

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def json(self, content_type=None):
                return self._json_ret

            @property
            def content(self):
                class C:
                    def __init__(self, chunks):
                        self._chunks = chunks

                    async def iter_chunked(self, _n):
                        for c in self._chunks:
                            yield c

                return C(self._chunks)

        return _Resp()

    return _make


@pytest.fixture
def fake_reg_factory():
    """Return a factory that constructs a configurable fake device registry.

    Usage:
        # registry where device does not exist
        fake = fake_reg_factory(device_exists=False)

        # registry where device exists and has id
        fake = fake_reg_factory(device_exists=True, device_id="removed-device-id")

    The returned object exposes:
      - async_get_device(self, *args, **kwargs) -> object | None
      - async_remove_device(self, *args, **kwargs) -> any
      - removed: boolean flag set to True when async_remove_device is called
    """

    def _make(
        device_exists: bool = False, device_id: str = "dev", remove_result: object | None = None
    ):
        class _FakeReg:
            def __init__(self):
                self.removed = False
                self._device_exists = device_exists
                self._device_id = device_id
                self._remove_result = remove_result

            def async_get_device(self, *args, **kwargs):
                if self._device_exists:

                    class _D:
                        id = self._device_id

                    return _D()
                return None

            def async_remove_device(self, *args, **kwargs):
                # mirror previous tests which sometimes inspect a `removed` flag
                self.removed = True
                return self._remove_result

        return _FakeReg()

    return _make


@pytest.fixture(autouse=True)
def _patch_async_create_clientsession(monkeypatch):
    """Ensure the integration's async_create_clientsession does not create real sessions.

    This prevents tests from opening real network resources and leaking connectors.
    """
    monkeypatch.setattr(
        _init_mod,
        "async_create_clientsession",
        lambda *a, **k: FakeClientSession(),
        raising=False,
    )


@pytest.fixture(autouse=True)
def _patch_homeassistant_stop(monkeypatch):
    """Wrap HomeAssistant.stop to ignore 'Event loop is closed' runtime errors.

    Some tests or integrations can close the event loop unexpectedly. During
    test teardown the pytest-homeassistant-custom-component plugin attempts to
    stop HomeAssistant instances which may call into a closed loop; this
    wrapper silently swallows that specific RuntimeError to allow teardown to
    continue in a best-effort manner.
    """

    original_stop = getattr(ha_core.HomeAssistant, "stop", None)

    if original_stop is None:
        return

    def _safe_stop(self, *args, **kwargs):
        try:
            return original_stop(self, *args, **kwargs)
        except RuntimeError as err:
            if "Event loop is closed" in str(err):
                # Log for diagnostics then swallow this specific error during tests.
                logger.debug(
                    "HomeAssistant.stop suppressed during test teardown: Event loop is closed",
                    exc_info=True,
                )
                return None
            raise

    monkeypatch.setattr(ha_core.HomeAssistant, "stop", _safe_stop, raising=False)


@pytest.fixture(autouse=True)
def _patch_asyncio_create_task(monkeypatch, request):
    """Patch asyncio.create_task to avoid creating background workers for pyopnsense during tests.

    For coroutines created by pyopnsense, close the coroutine object and return a dummy task-like
    object to prevent "coroutine was never awaited" warnings while avoiding scheduling real
    background work during tests.
    """

    # Allow tests to opt-out of this global patch when they exercise pyopnsense internals.
    # Tests can either add the marker `@pytest.mark.no_patch_create_task` or run a test file
    # that targets pyopnsense directly (for example `tests/test_pyopnsense.py`). In those
    # cases we should not replace `pyopnsense.asyncio.create_task` so the real behavior is
    # preserved for the test.
    try:
        # Marker-based opt-out (existing behavior)
        if getattr(request, "node", None) and request.node.get_closest_marker(
            "no_patch_create_task"
        ):
            return

        # File-path based opt-out: skip patching for tests that exercise
        # pyopnsense internals directly (for example `tests/test_pyopnsense.py`)
        # since those tests may rely on real task scheduling. Only skip
        # when a running event loop is available; if there is no running
        # loop at the time the fixture runs, leave our patch in place so
        # object construction that calls create_task doesn't raise.
        fspath = getattr(request, "fspath", None)
        # request.node may provide a fspath attribute in some pytest versions
        if fspath is None and getattr(request, "node", None) is not None:
            fspath = getattr(request.node, "fspath", None)
        if fspath and "test_pyopnsense.py" in str(fspath):
            try:
                # If there is a running loop, tests can rely on real scheduling
                asyncio.get_running_loop()
            except RuntimeError:
                # No running loop right now; keep our patch to avoid RuntimeError
                pass
            else:
                # Running loop present: skip patching to allow real scheduling
                return

    except (AttributeError, TypeError):
        # If we cannot determine the requesting test, continue with patching.
        pass

    # keep a reference to the original so we can delegate for non-target coroutines
    # Prefer the one from the pyopnsense module if present, otherwise fall back
    # to the global asyncio.create_task.
    # Prefer the pyopnsense module's asyncio.create_task when available; fall
    # back to the global asyncio.create_task otherwise. Avoid relying on an
    # ImportError here since the module import already occurred at module
    # load time. Instead, detect presence safely using globals() and
    # getattr.
    if "_pyopnsense_mod" in globals() and getattr(_pyopnsense_mod, "asyncio", None) is not None:
        # Prefer the module-scoped asyncio.create_task when available so we can
        # delegate for non-target coroutines. Fall back to the global
        # asyncio.create_task if the module doesn't expose one.
        _original_create_task = getattr(_pyopnsense_mod.asyncio, "create_task", asyncio.create_task)
    else:
        # pyopnsense.asyncio is not present; delegate to the global
        # asyncio.create_task. We intentionally avoid patching the global
        # asyncio module below unless necessary.
        _original_create_task = asyncio.create_task
        logger.debug(
            "pyopnsense.asyncio not present; attaching minimal namespace with fake create_task; delegating others to global asyncio"
        )

    def _fake_create_task(coro, *args, **kwargs):
        # If the coroutine is a pyopnsense background worker, close it and
        # return a completed awaitable to avoid scheduling background work.
        if _is_pyopnsense_background_coro(coro):
            with contextlib.suppress(Exception):
                coro.close()
            return _completed_future_result()
        # Delegate to the original create_task for all other coroutines.
        return _original_create_task(coro, *args, **kwargs)

    # Patch create_task only on the pyopnsense module to avoid interfering
    # with the rest of the test environment (Home Assistant / pytest-asyncio).
    # If the pyopnsense module does not expose an `asyncio` attribute, attach
    # a minimal namespace with our patched create_task so tests that construct
    # OPNsenseClient outside a running loop do not attempt to schedule real
    # background work. This avoids touching the global asyncio module.
    try:
        # Construct a proxy object that delegates all attributes to the real
        # asyncio module except `create_task`, which we override with our
        # test-local `_fake_create_task`. This avoids mutating the global
        # asyncio module and confines behavior to the pyopnsense module.
        real_asyncio = getattr(_pyopnsense_mod, "asyncio", asyncio)

        class _AsyncioProxy:
            """Proxy delegating attribute access to the real asyncio module.

            Only `create_task` is implemented on the proxy to forward to the
            provided fake implementation; all other attributes are looked up
            on the underlying real asyncio module via __getattr__.
            """

            def __init__(self, real, create_task_impl):
                self._real = real
                # store the impl as a bound attribute so monkeypatch can
                # replace it later if needed
                self.create_task = create_task_impl

            def __getattr__(self, name):
                return getattr(self._real, name)

        proxy = _AsyncioProxy(real_asyncio, _fake_create_task)

        # Replace whatever the pyopnsense module exposes with our proxy so
        # calls like `pyopnsense.asyncio.create_task(...)` hit the proxy and
        # use the fake implementation while all other asyncio behavior
        # delegates to the real module.
        monkeypatch.setattr(_pyopnsense_mod, "asyncio", proxy, raising=False)
    except Exception:  # noqa: BLE001
        logger.debug(
            "Failed to attach asyncio proxy on pyopnsense; falling back to direct patching"
        )


@pytest.fixture(autouse=True)
def _neutralize_pyopnsense_background_tasks(monkeypatch, request):
    """Autouse fixture to replace pyopnsense background queue workers with no-ops.

    This prevents the integration from scheduling background coroutines during
    tests which could interact with the event loop, create network IO, or
    produce 'coroutine was never awaited' warnings.
    """

    async def _noop_async(self, *args, **kwargs):
        return None

    # Try patching via the module/class object when available; fall back to
    # import-path based monkeypatching for resilience in different test envs.
    # Do not neutralize when running tests that exercise pyopnsense internals
    # directly (they need the real implementations). Skip patching for those
    # test modules (e.g., tests/test_pyopnsense.py).
    try:
        test_path = getattr(request, "fspath", None)
        if test_path and "test_pyopnsense.py" in str(test_path):
            return
    except (AttributeError, TypeError):
        # If we cannot determine the requesting test, continue with patching.
        pass

    try:
        if getattr(pyopnsense, "OPNsenseClient", None) is not None:
            monkeypatch.setattr(
                pyopnsense.OPNsenseClient, "_monitor_queue", _noop_async, raising=False
            )
            monkeypatch.setattr(
                pyopnsense.OPNsenseClient, "_process_queue", _noop_async, raising=False
            )
    except (AttributeError, TypeError):
        # best-effort; continue to fallback below
        pass

    # Fallback to import path strings in case direct attribute access failed.
    with contextlib.suppress(Exception):
        monkeypatch.setattr(
            "custom_components.opnsense.pyopnsense.OPNsenseClient._monitor_queue",
            _noop_async,
            raising=False,
        )
    with contextlib.suppress(Exception):
        monkeypatch.setattr(
            "custom_components.opnsense.pyopnsense.OPNsenseClient._process_queue",
            _noop_async,
            raising=False,
        )

    # Also make our patched asyncio.create_task (defined earlier in this file)
    # recognize coroutines that are bound methods of OPNsenseClient even when
    # the coroutine object originates from the test module (for example when
    # tests replace the methods with test-local no-ops). Inspect the
    # coroutine frame locals and treat coroutines with a `self` that is an
    # OPNsenseClient as pyopnsense background workers.
    try:
        # If the module-level fake create_task exists, decorate it using the
        # shared helpers so the logic is centralized and consistent.
        target_asyncio = getattr(_pyopnsense_mod, "asyncio", None)
        if target_asyncio is not None and hasattr(target_asyncio, "create_task"):
            orig = target_asyncio.create_task

            def _wrap_create_task(coro, *args, **kwargs):
                if _is_pyopnsense_background_coro(coro):
                    with contextlib.suppress(Exception):
                        coro.close()
                    return _completed_future_result()
                return orig(coro, *args, **kwargs)

            monkeypatch.setattr(target_asyncio, "create_task", _wrap_create_task, raising=False)
    except (AttributeError, TypeError):
        # Best-effort; do not fail tests if this decoration cannot be applied.
        pass


def pytest_runtest_teardown(item: Any, nextitem: Any) -> None:
    """Pytest hook: cancel any scheduled timer handles after each test.

    Prevent the pytest-homeassistant-custom-component plugin from failing tests
    due to lingering timer handles created by the integration (for example via
    hass.loop.call_later / async_call_later).
    """
    try:
        # Prefer the running loop when called from a running async context.
        event_loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop; fall back to the event loop from the current
        # policy. This mirrors the recommended replacement for
        # asyncio.get_event_loop() in synchronous code.
        event_loop = asyncio.get_event_loop_policy().get_event_loop()
    # If some integration code created and closed the global loop, we may
    # need to replace it with a fresh loop to allow the PHCC plugin to
    # perform teardown. However, this repository opts in to that behavior
    # via the `expected_lingering_timers` fixture. Only perform loop
    # replacement when the current test requested it; otherwise skip the
    # surgery but still attempt to cancel any scheduled timer handles in a
    # best-effort manner.
    if getattr(event_loop, "is_closed", lambda: False)():
        replace_loop = False
        try:
            # Prefer an explicit per-test fixture value when present.
            replace_loop = bool(item.funcargs.get("expected_lingering_timers", False))
        except (AttributeError, KeyError):
            # If funcargs isn't available, fall back to checking for a pytest
            # marker named `expected_lingering_timers` so tests can opt-in
            # without the removed fixture.
            replace_loop = False
            mk = getattr(item, "get_closest_marker", None)
            if callable(mk) and mk("expected_lingering_timers") is not None:
                replace_loop = True

        if replace_loop:
            try:
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                event_loop = new_loop
            except (OSError, RuntimeError):
                # Best-effort: if we cannot recreate the loop, continue and
                # let teardown attempt to proceed (it may still error).
                pass

    # Collect scheduled timer handles from the (possibly replaced) loop;
    # if the loop is closed and handle collection fails, skip cancellation
    # gracefully.
    try:
        handles = get_scheduled_timer_handles(event_loop)
    except (RuntimeError, OSError):
        handles = []

    for handle in handles:
        # Best-effort cancellation; don't raise from teardown hook.
        with contextlib.suppress(Exception):
            if not handle.cancelled():
                handle.cancel()


# Provide a shared FakeClientSession for tests to avoid creating real aiohttp sessions
class FakeClientSession:
    """Minimal fake client session used by tests in lieu of aiohttp.ClientSession."""

    def __init__(self, *args, **kwargs):
        """Initialize the fake client session (no-op)."""

    async def __aenter__(self):
        """Enter async context and return the session-like object."""
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """Exit async context, close the session and propagate exceptions."""
        await self.close()
        return False

    async def close(self):
        """Close the fake session (no-op)."""
        return True
