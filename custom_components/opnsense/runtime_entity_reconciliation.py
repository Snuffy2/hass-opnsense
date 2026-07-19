"""Shared runtime add-only coordinator-driven entity reconciler helpers."""

import asyncio
from collections.abc import Awaitable, Callable, Iterable
import logging
from typing import Protocol

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .repair_reconciliation import is_reconciliation_active

_LOGGER = logging.getLogger(__name__)

TypeCompileCallback = Callable[[], Awaitable[Iterable[Entity]]]
TypeIsReconciliationActive = Callable[[ConfigEntry], bool]


class TypeRuntimeCoordinator(Protocol):
    """Coordinator interface required by runtime entity reconciliation."""

    def async_add_listener(
        self,
        update_callback: Callable[..., None],
    ) -> Callable[[], None]:
        """Register an update listener and return its removal callback."""


def _entity_identity(entity: Entity) -> str | None:
    """Extract a stable per-platform identity for entity de-duplication.

    Args:
        entity: Entity candidate returned by a runtime compile callback.

    Returns:
        str | None: A non-empty entity description key, falling back to the
            entity unique ID, or ``None`` when neither is available.
    """
    description = getattr(entity, "entity_description", None)
    description_key = getattr(description, "key", None)
    if isinstance(description_key, str) and description_key.strip():
        return description_key

    unique_id = entity.unique_id
    if isinstance(unique_id, str) and unique_id.strip():
        return unique_id
    return None


class _RuntimeEntityReconciler:
    """Coordinate one coordinator listener with add-only reconciliations."""

    def __init__(
        self,
        hass: HomeAssistant,
        config_entry: ConfigEntry,
        async_add_entities: AddEntitiesCallback,
        compile_entities: TypeCompileCallback,
        is_reconciliation_active_fn: TypeIsReconciliationActive,
        initial_entities: Iterable[Entity],
    ) -> None:
        """Create the per-platform runtime reconciler state.

        Args:
            hass: Home Assistant instance used to schedule async work.
            config_entry: Config entry owning the listener and cleanup lifecycle.
            async_add_entities: Callback used to add newly discovered entities.
            compile_entities: Async compiler for runtime entities.
            is_reconciliation_active_fn: Predicate for repair-era skip windows.
            initial_entities: Entities already added during setup.
        """
        self._hass = hass
        self._config_entry = config_entry
        self._async_add_entities = async_add_entities
        self._compile_entities = compile_entities
        self._is_reconciliation_active_fn = is_reconciliation_active_fn
        self._seen_entity_identities: set[str] = {
            identity
            for identity in (_entity_identity(entity) for entity in initial_entities)
            if identity is not None
        }
        self._compile_requested = False
        self._compile_task: asyncio.Task[None] | None = None
        self._unloaded = False
        self._remove_coordinator_listener: Callable[[], None] | None = None
        self._cleanup_registered = False

    @property
    def remove_coordinator_listener(self) -> Callable[[], None] | None:
        """Return the coordinator listener unregister callback if available."""
        return self._remove_coordinator_listener

    @callback
    def schedule_reconcile(self, *_args: object, **_kwargs: object) -> None:
        """Schedule a coordinator-driven reconciliation pass.

        Args:
            *_args: Unused positional args from coordinator notifications.
            **_kwargs: Unused keyword args from coordinator notifications.
        """
        if self._unloaded:
            return
        self._compile_requested = True

        if self._compile_task is None:
            self._compile_task = self._hass.async_create_task(self._run_compilation())

    async def _run_compilation(self) -> None:
        """Run serialized and coalesced compile passes until up-to-date.

        The routine loops only while updates are pending and no unload is pending.
        Each pass reserves stable entity identities before calling
        ``async_add_entities``.
        """
        try:
            while self._compile_requested and not self._unloaded:
                self._compile_requested = False
                to_add: list[Entity] = []
                reserved_identities: list[str] = []

                if self._is_reconciliation_active_fn(self._config_entry):
                    self._compile_requested = True
                    _LOGGER.debug(
                        "Skipping runtime reconciler for %s while reconciliation is active",
                        self._config_entry.entry_id,
                    )
                    return

                try:
                    new_entities = await self._compile_entities()
                    for entity in new_entities:
                        identity = _entity_identity(entity)
                        if identity is None or identity in self._seen_entity_identities:
                            continue
                        self._seen_entity_identities.add(identity)
                        reserved_identities.append(identity)
                        to_add.append(entity)
                except asyncio.CancelledError:
                    raise
                except AttributeError, KeyError, RuntimeError, TypeError, ValueError:
                    _LOGGER.exception(
                        "Runtime entity compile callback failed for entry %s",
                        self._config_entry.entry_id,
                    )
                    return
                if to_add and not self._unloaded:
                    submitted = False
                    try:
                        self._async_add_entities(to_add)
                        submitted = True
                    finally:
                        if not submitted:
                            self._seen_entity_identities.difference_update(reserved_identities)
        finally:
            self._compile_task = None

    def attach_cleanup(self) -> None:
        """Register unload cleanup for listener registration and in-flight tasks."""

        def _cleanup() -> None:
            """Stop future reconciler work and release listener/task resources."""
            if self._unloaded:
                return
            self._unloaded = True
            if self._remove_coordinator_listener is not None:
                self._remove_coordinator_listener()
                self._remove_coordinator_listener = None
            if self._compile_task is not None and not self._compile_task.done():
                self._compile_task.cancel()

        if not self._cleanup_registered:
            self._config_entry.async_on_unload(_cleanup)
            self._cleanup_registered = True

    def attach_coordinator_listener(
        self,
        coordinator: TypeRuntimeCoordinator,
    ) -> None:
        """Attach one coordinator listener used for compile scheduling.

        Args:
            coordinator: Coordinator emitting runtime update callbacks.
        """
        self._remove_coordinator_listener = coordinator.async_add_listener(self.schedule_reconcile)


def attach_runtime_entity_reconciler(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    coordinator: TypeRuntimeCoordinator,
    async_add_entities: AddEntitiesCallback,
    initial_entities: Iterable[Entity],
    async_compile_entities: TypeCompileCallback,
    *,
    is_reconciliation_active_fn: TypeIsReconciliationActive = is_reconciliation_active,
) -> None:
    """Attach a runtime reconciler that only adds unseen coordinator-derived entities.

    Args:
        hass: Home Assistant instance used to schedule async compilation.
        config_entry: Config entry owning the reconciler lifecycle.
        coordinator: Coordinator whose updates trigger reconciliation.
        async_add_entities: Callback used to register newly discovered entities.
        initial_entities: Entities whose stable identities were already added
            during platform setup.
        async_compile_entities: Async callback that compiles candidate runtime entities.
        is_reconciliation_active_fn: Optional override used for unit tests.

    Notes:
        The function registers exactly one coordinator listener and one unload hook.
        It never removes existing entities.
    """
    reconciler = _RuntimeEntityReconciler(
        hass=hass,
        config_entry=config_entry,
        async_add_entities=async_add_entities,
        compile_entities=async_compile_entities,
        is_reconciliation_active_fn=is_reconciliation_active_fn,
        initial_entities=initial_entities,
    )
    reconciler.attach_coordinator_listener(coordinator)
    reconciler.attach_cleanup()
