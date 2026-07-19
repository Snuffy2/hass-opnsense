"""Tests for runtime entity reconciliation scheduling and dedupe behavior."""

import asyncio
from collections.abc import Callable, Coroutine, Iterable
from typing import Any
from unittest.mock import MagicMock

from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import Entity, EntityDescription
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.opnsense.runtime_entity_reconciliation import (
    attach_runtime_entity_reconciler,
)


class _RuntimeCoordinator:
    """Minimal coordinator stub that captures runtime reconciliation listeners."""

    def __init__(self) -> None:
        """Initialize listener state."""
        self._listeners: list[Callable[..., None]] = []
        self.remove_calls: int = 0

    def async_add_listener(self, listener: Callable[..., None]) -> Callable[[], None]:
        """Register a listener and return a remover callback."""
        self._listeners.append(listener)

        def _remove() -> None:
            """Remove a listener once if it is still registered."""
            if listener in self._listeners:
                self._listeners.remove(listener)
            self.remove_calls += 1

        return _remove

    def fire_update(self, *args: object, **kwargs: object) -> None:
        """Invoke every registered listener immediately."""
        for listener in list(self._listeners):
            listener(*args, **kwargs)


class _SimpleEntity(Entity):
    """Entity with deterministic unique ID for reconciler tests."""

    def __init__(
        self,
        unique_id: str | None,
        description_key: str | None = None,
    ) -> None:
        """Initialize an entity with a given unique ID value."""
        self._attr_unique_id = unique_id
        if description_key is not None:
            self.entity_description = EntityDescription(key=description_key)


def _ignore_entities(
    new_entities: Iterable[Entity],
    update_before_add: bool = False,
) -> None:
    """Ignore entities added by a reconciler test."""
    del new_entities, update_before_add


@pytest.mark.asyncio
async def test_reconciler_registers_single_coordinator_listener(
    ph_hass: HomeAssistant,
    make_config_entry: Callable[..., MockConfigEntry],
) -> None:
    """Only one coordinator listener should be registered by the reconciler."""
    entry = make_config_entry(entry_id="entry-runtime-listener")
    coordinator = _RuntimeCoordinator()
    entry.async_on_unload = MagicMock()

    async def _compile() -> list[_SimpleEntity]:
        return []

    attach_runtime_entity_reconciler(
        ph_hass,
        entry,
        coordinator,
        _ignore_entities,
        [],
        _compile,
    )

    assert len(coordinator._listeners) == 1
    entry.async_on_unload.assert_called_once()


@pytest.mark.asyncio
async def test_reconciler_adds_only_unseen_ids_from_runtime_compile(
    ph_hass: HomeAssistant,
    make_config_entry: Callable[..., MockConfigEntry],
) -> None:
    """Already-known IDs should be reserved and excluded from add-only reconciles."""
    entry = make_config_entry(entry_id="entry-runtime-seed")
    coordinator = _RuntimeCoordinator()
    entry.async_on_unload = MagicMock()

    added_batches: list[list[Entity]] = []

    async def _compile() -> list[_SimpleEntity]:
        return [_SimpleEntity("existing"), _SimpleEntity("new")]

    def _add_entities(
        new_entities: Iterable[Entity],
        update_before_add: bool = False,
    ) -> None:
        del update_before_add
        added_batches.append(list(new_entities))

    attach_runtime_entity_reconciler(
        ph_hass,
        entry,
        coordinator,
        _add_entities,
        [_SimpleEntity("existing")],
        _compile,
    )

    coordinator.fire_update()
    await asyncio.sleep(0)

    assert len(added_batches) == 1
    assert {entity.unique_id for entity in added_batches[0]} == {"new"}


@pytest.mark.asyncio
async def test_reconciler_deduplicates_by_stable_description_key(
    ph_hass: HomeAssistant,
    make_config_entry: Callable[..., MockConfigEntry],
) -> None:
    """Description keys should deduplicate entities with mutable unique IDs."""
    entry = make_config_entry(entry_id="entry-runtime-stable-key")
    coordinator = _RuntimeCoordinator()
    entry.async_on_unload = MagicMock()
    added_batches: list[list[Entity]] = []

    async def _compile() -> list[_SimpleEntity]:
        return [_SimpleEntity("gateway-new-name", "gateway-status")]

    def _add_entities(
        new_entities: Iterable[Entity],
        update_before_add: bool = False,
    ) -> None:
        del update_before_add
        added_batches.append(list(new_entities))

    attach_runtime_entity_reconciler(
        ph_hass,
        entry,
        coordinator,
        _add_entities,
        [_SimpleEntity("gateway-old-name", "gateway-status")],
        _compile,
    )

    coordinator.fire_update()
    await asyncio.sleep(0)

    assert added_batches == []


@pytest.mark.asyncio
async def test_reconciler_ignores_empty_or_none_unique_ids(
    ph_hass: HomeAssistant,
    make_config_entry: Callable[..., MockConfigEntry],
) -> None:
    """Entities without usable unique IDs are ignored for dedupe and adds."""
    entry = make_config_entry(entry_id="entry-runtime-empty")
    coordinator = _RuntimeCoordinator()
    entry.async_on_unload = MagicMock()

    added_batches: list[list[Entity]] = []

    async def _compile() -> list[_SimpleEntity]:
        return [
            _SimpleEntity(""),
            _SimpleEntity("   "),
            _SimpleEntity(None),
            _SimpleEntity("usable"),
        ]

    def _add_entities(
        new_entities: Iterable[Entity],
        update_before_add: bool = False,
    ) -> None:
        del update_before_add
        added_batches.append(list(new_entities))

    attach_runtime_entity_reconciler(
        ph_hass,
        entry,
        coordinator,
        _add_entities,
        [],
        _compile,
    )

    coordinator.fire_update()
    await asyncio.sleep(0)

    assert len(added_batches) == 1
    assert {entity.unique_id for entity in added_batches[0]} == {"usable"}


@pytest.mark.asyncio
async def test_reconciler_skips_updates_while_reconciliation_active(
    ph_hass: HomeAssistant,
    make_config_entry: Callable[..., MockConfigEntry],
) -> None:
    """No compile should run while active repair reconciliation blocks runtime updates."""
    entry = make_config_entry(entry_id="entry-runtime-repair")
    coordinator = _RuntimeCoordinator()
    entry.async_on_unload = MagicMock()

    calls = 0

    async def _compile() -> list[_SimpleEntity]:
        nonlocal calls
        calls += 1
        return [_SimpleEntity(f"id-{calls}")]

    active = True
    attach_runtime_entity_reconciler(
        ph_hass,
        entry,
        coordinator,
        _ignore_entities,
        [],
        _compile,
        is_reconciliation_active_fn=lambda _config_entry: active,
    )

    coordinator.fire_update()
    await asyncio.sleep(0)
    assert calls == 0

    active = False
    coordinator.fire_update()
    await asyncio.sleep(0)
    assert calls == 1


@pytest.mark.asyncio
async def test_reconciler_coalesces_overlapping_updates(
    ph_hass: HomeAssistant,
    make_config_entry: Callable[..., MockConfigEntry],
) -> None:
    """Multiple rapid updates should coalesce into sequential compile passes."""
    entry = make_config_entry(entry_id="entry-runtime-coalesce")
    coordinator = _RuntimeCoordinator()
    entry.async_on_unload = MagicMock()

    added_batches: list[list[Entity]] = []
    calls = 0
    first_pass_started = asyncio.Event()
    first_pass_continue = asyncio.Event()

    async def _compile() -> list[_SimpleEntity]:
        nonlocal calls
        calls += 1
        if calls == 1:
            first_pass_started.set()
            await first_pass_continue.wait()
            return [_SimpleEntity("first")]
        return [_SimpleEntity("second")]

    def _add_entities(
        new_entities: Iterable[Entity],
        update_before_add: bool = False,
    ) -> None:
        del update_before_add
        added_batches.append(list(new_entities))

    attach_runtime_entity_reconciler(
        ph_hass,
        entry,
        coordinator,
        _add_entities,
        [],
        _compile,
    )

    coordinator.fire_update()
    await first_pass_started.wait()

    coordinator.fire_update()
    coordinator.fire_update()

    first_pass_continue.set()
    await asyncio.sleep(0)

    assert calls == 2
    assert len(added_batches) == 2
    assert [batch[0].unique_id for batch in added_batches] == ["first", "second"]


@pytest.mark.asyncio
async def test_reconciler_reserves_task_before_synchronous_updates(
    make_config_entry: Callable[..., MockConfigEntry],
) -> None:
    """Synchronous updates should share one serialized compile task."""
    entry = make_config_entry(entry_id="entry-runtime-task-reservation")
    coordinator = _RuntimeCoordinator()
    entry.async_on_unload = MagicMock()

    scheduled_tasks: list[asyncio.Task[None]] = []
    hass = MagicMock()

    def _create_task(coro: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
        """Schedule and record one reconciler task."""
        task = asyncio.create_task(coro)
        scheduled_tasks.append(task)
        return task

    hass.async_create_task.side_effect = _create_task
    first_pass_started = asyncio.Event()
    first_pass_continue = asyncio.Event()
    active_compiles = 0
    max_active_compiles = 0
    compile_calls = 0
    added_batches: list[list[Entity]] = []

    async def _compile() -> list[_SimpleEntity]:
        nonlocal active_compiles, compile_calls, max_active_compiles
        active_compiles += 1
        max_active_compiles = max(max_active_compiles, active_compiles)
        compile_calls += 1
        try:
            if compile_calls == 1:
                first_pass_started.set()
                await first_pass_continue.wait()
            return [_SimpleEntity("shared")]
        finally:
            active_compiles -= 1

    def _add_entities(
        new_entities: Iterable[Entity],
        update_before_add: bool = False,
    ) -> None:
        """Capture runtime additions."""
        del update_before_add
        added_batches.append(list(new_entities))

    attach_runtime_entity_reconciler(
        hass,
        entry,
        coordinator,
        _add_entities,
        [],
        _compile,
    )

    coordinator.fire_update()
    coordinator.fire_update()
    coordinator.fire_update()

    assert len(scheduled_tasks) == 1
    await first_pass_started.wait()

    coordinator.fire_update()
    coordinator.fire_update()
    first_pass_continue.set()
    await scheduled_tasks[0]

    assert compile_calls == 2
    assert max_active_compiles == 1
    assert len(added_batches) == 1
    assert [entity.unique_id for entity in added_batches[0]] == ["shared"]


@pytest.mark.asyncio
async def test_reconciler_no_duplicate_adds_across_compilations_and_no_deletes(
    ph_hass: HomeAssistant,
    make_config_entry: Callable[..., MockConfigEntry],
) -> None:
    """Repeated runtime runs preserve seen IDs and never perform deletions."""
    entry = make_config_entry(entry_id="entry-runtime-idempotent")
    coordinator = _RuntimeCoordinator()
    entry.async_on_unload = MagicMock()

    added_batches: list[list[Entity]] = []

    async def _compile() -> list[_SimpleEntity]:
        return [_SimpleEntity("dup"), _SimpleEntity("dup"), _SimpleEntity("new")]

    def _add_entities(
        new_entities: Iterable[Entity],
        update_before_add: bool = False,
    ) -> None:
        del update_before_add
        added_batches.append(list(new_entities))

    attach_runtime_entity_reconciler(
        ph_hass,
        entry,
        coordinator,
        _add_entities,
        [_SimpleEntity("already")],
        _compile,
    )

    coordinator.fire_update()
    await asyncio.sleep(0)

    assert len(added_batches) == 1
    assert [e.unique_id for e in added_batches[0]] == ["dup", "new"]


@pytest.mark.asyncio
async def test_reconciler_retries_batch_after_submission_failure(
    make_config_entry: Callable[..., MockConfigEntry],
) -> None:
    """A failed entity submission should release identities for a later retry."""
    entry = make_config_entry(entry_id="entry-runtime-submit-retry")
    coordinator = _RuntimeCoordinator()
    entry.async_on_unload = MagicMock()
    scheduled_tasks: list[asyncio.Task[None]] = []
    hass = MagicMock()

    def _create_task(coro: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
        """Schedule and record one reconciler task."""
        task = asyncio.create_task(coro)
        scheduled_tasks.append(task)
        return task

    hass.async_create_task.side_effect = _create_task
    submission_attempts = 0
    added_batches: list[list[Entity]] = []

    async def _compile() -> list[_SimpleEntity]:
        return [_SimpleEntity("mutable-name", "stable-key")]

    def _add_entities(
        new_entities: Iterable[Entity],
        update_before_add: bool = False,
    ) -> None:
        nonlocal submission_attempts
        del update_before_add
        submission_attempts += 1
        if submission_attempts == 1:
            raise RuntimeError("submission failed")
        added_batches.append(list(new_entities))

    attach_runtime_entity_reconciler(
        hass,
        entry,
        coordinator,
        _add_entities,
        [],
        _compile,
    )

    coordinator.fire_update()
    with pytest.raises(RuntimeError, match="submission failed"):
        await scheduled_tasks[0]

    coordinator.fire_update()
    await scheduled_tasks[1]

    assert submission_attempts == 2
    assert len(added_batches) == 1
    assert [entity.unique_id for entity in added_batches[0]] == ["mutable-name"]


@pytest.mark.asyncio
async def test_reconciler_cleanup_stops_reconciliation_and_listener(
    ph_hass: HomeAssistant,
    make_config_entry: Callable[..., MockConfigEntry],
) -> None:
    """Unloading the config entry should prevent further coordinator-driven updates."""
    entry = make_config_entry(entry_id="entry-runtime-cleanup")
    coordinator = _RuntimeCoordinator()
    unload_callbacks: list[Callable[[], None]] = []

    entry.async_on_unload = MagicMock(side_effect=unload_callbacks.append)
    first_pass_started = asyncio.Event()
    first_pass_continue = asyncio.Event()
    calls = 0
    added: list[list[Entity]] = []

    async def _compile() -> list[_SimpleEntity]:
        nonlocal calls
        calls += 1
        first_pass_started.set()
        await first_pass_continue.wait()
        return [_SimpleEntity("late")]

    def _add_entities(
        new_entities: Iterable[Entity],
        update_before_add: bool = False,
    ) -> None:
        del update_before_add
        added.append(list(new_entities))

    attach_runtime_entity_reconciler(
        ph_hass,
        entry,
        coordinator,
        _add_entities,
        [],
        _compile,
    )
    assert unload_callbacks
    remove_callback = unload_callbacks[0]

    coordinator.fire_update()
    await first_pass_started.wait()

    remove_callback()

    first_pass_continue.set()
    await asyncio.sleep(0)

    coordinator.fire_update()
    await asyncio.sleep(0)

    assert calls == 1
    assert added == []
    assert coordinator.remove_calls == 1
