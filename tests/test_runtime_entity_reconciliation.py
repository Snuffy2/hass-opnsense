"""Tests for runtime entity reconciliation scheduling and de-duplication behavior."""

import asyncio
from collections.abc import Callable, Coroutine, Iterable
from typing import Any
from unittest.mock import MagicMock

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
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
async def test_reconciler_skips_submission_when_inventory_unchanged(
    ph_hass: HomeAssistant,
    make_config_entry: Callable[..., MockConfigEntry],
) -> None:
    """A stable fingerprint prevents resubmission when inventory is unchanged."""
    entry = make_config_entry(entry_id="entry-runtime-unchanged")
    coordinator = _RuntimeCoordinator()
    entry.async_on_unload = MagicMock()
    compile_calls = 0

    async def _compile() -> list[_SimpleEntity]:
        nonlocal compile_calls
        compile_calls += 1
        return [_SimpleEntity("static")]

    def _add_entities(
        new_entities: Iterable[Entity],
        update_before_add: bool = False,
    ) -> None:
        del update_before_add
        del new_entities

    attach_runtime_entity_reconciler(
        ph_hass,
        entry,
        coordinator,
        _add_entities,
        [_SimpleEntity("static")],
        _compile,
        inventory_fingerprint=lambda: ("static",),
    )

    coordinator.fire_update()
    await asyncio.sleep(0)

    assert compile_calls == 0


@pytest.mark.asyncio
async def test_reconciler_reconciles_changed_inventory_fingerprint_and_no_redundant_readds(
    ph_hass: HomeAssistant,
    make_config_entry: Callable[..., MockConfigEntry],
) -> None:
    """Changed fingerprints should trigger adds while duplicate fingerprints should not."""
    entry = make_config_entry(entry_id="entry-runtime-fingerprint")
    coordinator = _RuntimeCoordinator()
    entry.async_on_unload = MagicMock()
    added_batches: list[list[Entity]] = []
    calls = 0
    inventory: list[str] = []

    async def _compile() -> list[_SimpleEntity]:
        nonlocal calls
        calls += 1
        return [_SimpleEntity(inventory[0])]

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
        inventory_fingerprint=lambda: tuple(inventory),
    )

    inventory.append("same")
    coordinator.fire_update()
    await asyncio.sleep(0)

    coordinator.fire_update()
    await asyncio.sleep(0)

    inventory[0] = "next"
    coordinator.fire_update()
    await asyncio.sleep(0)

    assert calls == 2
    assert len(added_batches) == 2
    assert [batch[0].unique_id for batch in added_batches] == ["same", "next"]


@pytest.mark.asyncio
async def test_reconciler_fingerprint_not_advanced_after_submission_failure(
    make_config_entry: Callable[..., MockConfigEntry],
) -> None:
    """A failed submission should keep prior fingerprint so later events still retry."""
    coordinator = _RuntimeCoordinator()
    entry = make_config_entry(entry_id="entry-runtime-fingerprint-fail")
    entry.async_on_unload = MagicMock()
    scheduled_tasks: list[asyncio.Task[None]] = []
    hass = MagicMock()

    submission_attempts = 0
    added_batches: list[list[Entity]] = []
    inventory: list[str] = []

    def _create_task(coro: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
        """Record reconciler tasks spawned by the mocked HA loop."""
        task = asyncio.create_task(coro)
        scheduled_tasks.append(task)
        return task

    hass.async_create_task.side_effect = _create_task

    async def _compile() -> list[_SimpleEntity]:
        return [_SimpleEntity("retry-token")]

    def _add_entities(
        new_entities: Iterable[Entity],
        update_before_add: bool = False,
    ) -> None:
        nonlocal submission_attempts
        del update_before_add
        submission_attempts += 1
        if submission_attempts == 1:
            raise HomeAssistantError("submission failed")
        added_batches.append(list(new_entities))

    attach_runtime_entity_reconciler(
        hass,
        entry,
        coordinator,
        _add_entities,
        [],
        _compile,
        inventory_fingerprint=lambda: tuple(inventory),
    )

    inventory.append("retry-token")
    coordinator.fire_update()
    await scheduled_tasks[0]

    coordinator.fire_update()
    await scheduled_tasks[1]

    assert submission_attempts == 2
    assert len(added_batches) == 1
    assert [entity.unique_id for entity in added_batches[0]] == ["retry-token"]


@pytest.mark.asyncio
async def test_reconciler_advances_fingerprint_only_after_successful_compile(
    ph_hass: HomeAssistant,
    make_config_entry: Callable[..., MockConfigEntry],
) -> None:
    """A failed compile retries later, while a successful empty pass advances."""
    entry = make_config_entry(entry_id="entry-runtime-compile-fingerprint")
    coordinator = _RuntimeCoordinator()
    entry.async_on_unload = MagicMock()
    inventory: list[str] = []
    compile_calls = 0

    async def _compile() -> list[_SimpleEntity]:
        nonlocal compile_calls
        compile_calls += 1
        if compile_calls == 1:
            raise RuntimeError("compile failed")
        return []

    attach_runtime_entity_reconciler(
        ph_hass,
        entry,
        coordinator,
        _ignore_entities,
        [],
        _compile,
        inventory_fingerprint=lambda: tuple(inventory),
    )

    inventory.append("new")
    coordinator.fire_update()
    await asyncio.sleep(0)

    coordinator.fire_update()
    await asyncio.sleep(0)

    coordinator.fire_update()
    await asyncio.sleep(0)

    assert compile_calls == 2


@pytest.mark.asyncio
async def test_reconciler_rejects_normalized_description_key_collisions(
    ph_hass: HomeAssistant,
    make_config_entry: Callable[..., MockConfigEntry],
) -> None:
    """Description-key normalization collisions should resolve to a single runtime add."""
    entry = make_config_entry(entry_id="entry-runtime-desc-collision")
    coordinator = _RuntimeCoordinator()
    entry.async_on_unload = MagicMock()
    added_batches: list[list[Entity]] = []

    async def _compile() -> list[_SimpleEntity]:
        return [
            _SimpleEntity("first", "Gateway Status"),
            _SimpleEntity("second", "gateway-status"),
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
    assert {entity.unique_id for entity in added_batches[0]} == {"first"}


@pytest.mark.asyncio
async def test_reconciler_retains_stable_gateway_identity_across_rename(
    ph_hass: HomeAssistant,
    make_config_entry: Callable[..., MockConfigEntry],
) -> None:
    """A renamed gateway should retain its stable description-key identity."""
    entry = make_config_entry(entry_id="entry-runtime-gateway-rename")
    coordinator = _RuntimeCoordinator()
    entry.async_on_unload = MagicMock()
    added_batches: list[list[Entity]] = []

    async def _compile() -> list[_SimpleEntity]:
        return [_SimpleEntity("gateway.new-name.status", "gateway.wan.status")]

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
        [_SimpleEntity("gateway.old-name.status", "gateway.wan.status")],
        _compile,
    )

    coordinator.fire_update()
    await asyncio.sleep(0)

    assert added_batches == []


@pytest.mark.asyncio
async def test_reconciler_rejects_normalized_unique_id_collisions(
    ph_hass: HomeAssistant,
    make_config_entry: Callable[..., MockConfigEntry],
) -> None:
    """Punctuation/case unique-id collisions should not create duplicates."""
    entry = make_config_entry(entry_id="entry-runtime-uid-collision")
    coordinator = _RuntimeCoordinator()
    entry.async_on_unload = MagicMock()
    added_batches: list[list[Entity]] = []

    async def _compile() -> list[_SimpleEntity]:
        return [_SimpleEntity("gateway-A"), _SimpleEntity("Gateway A")]

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
        [_SimpleEntity("gateway_a")],
        _compile,
    )

    coordinator.fire_update()
    await asyncio.sleep(0)

    assert added_batches == []


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
async def test_reconciler_reconciliation_continuity_reserves_task_during_sync_updates(
    make_config_entry: Callable[..., MockConfigEntry],
) -> None:
    """Concurrent updates while async work runs should serialize on one task."""
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
    compile_calls = 0
    added_batches: list[list[Entity]] = []

    async def _compile() -> list[_SimpleEntity]:
        nonlocal compile_calls
        compile_calls += 1
        if compile_calls == 1:
            first_pass_started.set()
            await first_pass_continue.wait()
        return [_SimpleEntity("shared")]

    def _add_entities(
        new_entities: Iterable[Entity],
        update_before_add: bool = False,
    ) -> None:
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
    assert len(added_batches) == 1
    assert [entity.unique_id for entity in added_batches[0]] == ["shared"]


@pytest.mark.asyncio
async def test_reconciler_cleanup_stops_reconciliation_and_listener(
    ph_hass: HomeAssistant,
    make_config_entry: Callable[..., MockConfigEntry],
) -> None:
    """Unloading the config entry should stop reconciliations and listener removal."""
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
