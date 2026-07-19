"""Shared runtime add-only coordinator-driven entity reconciler helpers."""

import asyncio
from collections.abc import Awaitable, Callable, Hashable, Iterable
import logging
from typing import Protocol

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.util import slugify

from .repair_reconciliation import is_reconciliation_active

_LOGGER = logging.getLogger(__name__)

TypeCompileCallback = Callable[[], Awaitable[Iterable[Entity]]]
TypeIsReconciliationActive = Callable[[ConfigEntry], bool]
TypeInventoryFingerprint = Callable[[], Hashable]

_FINGERPRINT_UNSET = object()


class TypeRuntimeCoordinator(Protocol):
    """Coordinator interface required by runtime entity reconciliation."""

    def async_add_listener(
        self,
        update_callback: Callable[..., None],
    ) -> Callable[[], None]:
        """Register an update listener and return its removal callback."""


def _normalize_identity_token(value: str | None) -> str | None:
    """Normalize a raw identity candidate into a stable comparison token.

    Args:
        value: A raw value from description key or unique ID.

    Returns:
        str | None: Slugified value or ``None`` when value is unusable.
    """
    if not isinstance(value, str):
        return None

    normalized = value.strip()
    if not normalized:
        return None

    normalized_token = slugify(normalized)
    return normalized_token or None


def _entity_identity_tokens(entity: Entity) -> tuple[tuple[str, str], str | None]:
    """Extract normalized identity and unique-id collision tokens for an entity.

    Args:
        entity: Candidate entity to normalize.

    Returns:
        tuple[tuple[str, str], str | None]: Identity token and optional normalized
            unique-id token.

    Raises:
        ValueError: When no stable identity can be derived.
    """
    description = getattr(entity, "entity_description", None)
    description_key = getattr(description, "key", None)
    description_token = _normalize_identity_token(description_key)

    if description_token is not None:
        unique_token = _normalize_identity_token(entity.unique_id)
        return ("description", description_token), unique_token

    unique_token = _normalize_identity_token(entity.unique_id)
    if unique_token is None:
        raise ValueError("Missing entity identity for runtime reconciliation")

    return ("unique", unique_token), unique_token


class _RuntimeEntityReconciler:
    """Coordinate one coordinator listener with add-only reconciliations."""

    def __init__(
        self,
        hass: HomeAssistant,
        config_entry: ConfigEntry,
        async_add_entities: AddEntitiesCallback,
        compile_entities: TypeCompileCallback,
        is_reconciliation_active_fn: TypeIsReconciliationActive,
        inventory_fingerprint: TypeInventoryFingerprint | None,
        initial_entities: Iterable[Entity],
    ) -> None:
        """Create the per-platform runtime reconciler state.

        Args:
            hass: Home Assistant instance used to schedule async work.
            config_entry: Config entry owning the listener and cleanup lifecycle.
            async_add_entities: Callback used to add runtime entities.
            compile_entities: Async compiler for runtime entities.
            is_reconciliation_active_fn: Predicate for repair-era skip windows.
            inventory_fingerprint: Optional coordinator-inventory fingerprint callback.
            initial_entities: Entities already added during setup.
        """
        self._hass = hass
        self._config_entry = config_entry
        self._async_add_entities = async_add_entities
        self._compile_entities = compile_entities
        self._is_reconciliation_active_fn = is_reconciliation_active_fn
        self._inventory_fingerprint_fn = inventory_fingerprint
        initial_entities_tuple = tuple(initial_entities)
        self._seen_entity_identities: set[str] = set()
        self._seen_unique_tokens: set[str] = set()
        for entity in initial_entities_tuple:
            identity = self._identity_or_none(entity)
            if identity is None:
                continue
            (identity_prefix, identity_token), unique_token = identity
            identity_key = f"{identity_prefix}:{identity_token}"
            self._seen_entity_identities.add(identity_key)
            if unique_token is not None:
                self._seen_unique_tokens.add(unique_token)
        self._inventory_fingerprint: Hashable | object = _FINGERPRINT_UNSET
        self._pending_inventory_fingerprint: Hashable | object = _FINGERPRINT_UNSET
        if self._inventory_fingerprint_fn is not None:
            try:
                self._inventory_fingerprint = self._inventory_fingerprint_fn()
            except RuntimeError, TypeError, ValueError:
                _LOGGER.exception(
                    "Runtime inventory fingerprint callback failed for entry %s",
                    self._config_entry.entry_id,
                )

        self._compile_requested = False
        self._compile_task: asyncio.Task[None] | None = None
        self._unloaded = False
        self._remove_coordinator_listener: Callable[[], None] | None = None
        self._cleanup_registered = False

    @staticmethod
    def _identity_or_none(entity: Entity) -> tuple[tuple[str, str], str | None] | None:
        """Resolve stable identity tokens or return ``None`` when unavailable."""
        try:
            return _entity_identity_tokens(entity)
        except ValueError:
            return None

    @callback
    def schedule_reconcile(self, *_args: object, **_kwargs: object) -> None:
        """Schedule a coordinator-driven reconciliation pass.

        Args:
            *_args: Unused positional args from coordinator notifications.
            **_kwargs: Unused keyword args from coordinator notifications.
        """
        if self._unloaded:
            return

        if self._inventory_fingerprint_fn is not None:
            try:
                next_fingerprint = self._inventory_fingerprint_fn()
            except RuntimeError, TypeError, ValueError:
                _LOGGER.exception(
                    "Runtime inventory fingerprint callback failed for entry %s",
                    self._config_entry.entry_id,
                )
                return
            if next_fingerprint == self._inventory_fingerprint:
                _LOGGER.debug(
                    "Skipping runtime reconciler for %s because inventory is unchanged",
                    self._config_entry.entry_id,
                )
                return
            self._pending_inventory_fingerprint = next_fingerprint

        self._compile_requested = True

        if self._compile_task is None:
            self._compile_task = self._hass.async_create_task(self._run_compilation())

    async def _run_compilation(self) -> None:
        """Run serialized and coalesced compile passes until up-to-date.

        The routine loops while updates remain and no unload is pending. It reserves
        identities before submission and rolls them back on submission failure.
        """
        try:
            while self._compile_requested and not self._unloaded:
                self._compile_requested = False
                pass_fingerprint = self._pending_inventory_fingerprint
                to_add: list[Entity] = []
                reserved_identity_keys: set[str] = set()
                reserved_unique_tokens: set[str] = set()

                if self._is_reconciliation_active_fn(self._config_entry):
                    self._compile_requested = True
                    _LOGGER.debug(
                        "Skipping runtime reconciler for %s while reconciliation is active",
                        self._config_entry.entry_id,
                    )
                    return

                try:
                    new_entities = tuple(await self._compile_entities())
                except asyncio.CancelledError:
                    raise
                except AttributeError, KeyError, RuntimeError, TypeError, ValueError:
                    _LOGGER.exception(
                        "Runtime entity compile callback failed for entry %s",
                        self._config_entry.entry_id,
                    )
                    return

                for entity in new_entities:
                    identity = self._identity_or_none(entity)
                    if identity is None:
                        continue

                    (identity_prefix, identity_token), unique_token = identity
                    identity_key = f"{identity_prefix}:{identity_token}"
                    if (
                        identity_key in self._seen_entity_identities
                        or identity_key in reserved_identity_keys
                    ):
                        _LOGGER.debug(
                            "Skipping runtime entity %s for %s due duplicate identity key",
                            identity_key,
                            self._config_entry.entry_id,
                        )
                        continue

                    if unique_token is not None and (
                        unique_token in self._seen_unique_tokens
                        or unique_token in reserved_unique_tokens
                    ):
                        _LOGGER.debug(
                            "Skipping runtime entity %s for %s due normalized unique-id collision",
                            identity_key,
                            self._config_entry.entry_id,
                        )
                        continue

                    reserved_identity_keys.add(identity_key)
                    self._seen_entity_identities.add(identity_key)
                    if unique_token is not None:
                        reserved_unique_tokens.add(unique_token)
                        self._seen_unique_tokens.add(unique_token)
                    to_add.append(entity)

                if to_add:
                    try:
                        self._async_add_entities(to_add)
                    except HomeAssistantError, RuntimeError, ValueError:
                        self._seen_entity_identities.difference_update(reserved_identity_keys)
                        self._seen_unique_tokens.difference_update(reserved_unique_tokens)
                        _LOGGER.exception(
                            "Runtime reconciler submit failed for entry %s",
                            self._config_entry.entry_id,
                        )
                        return
                if pass_fingerprint is not _FINGERPRINT_UNSET:
                    self._inventory_fingerprint = pass_fingerprint
        finally:
            self._compile_task = None

    def attach_cleanup(self) -> None:
        """Register unload cleanup for listener and in-flight tasks."""

        def _cleanup() -> None:
            """Stop reconciler work and release listener/task resources."""
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
    inventory_fingerprint: TypeInventoryFingerprint | None = None,
) -> None:
    """Attach a runtime reconciler that only adds unseen coordinator-derived entities.

    Args:
        hass: Home Assistant instance used to schedule async compilation.
        config_entry: Config entry owning the reconciler lifecycle.
        coordinator: Coordinator whose updates trigger reconciliation.
        async_add_entities: Callback used to register newly discovered entities.
        initial_entities: Entities whose stable identities were already added
            during platform setup.
        async_compile_entities: Async callback that compiles candidate runtime
            entities.
        is_reconciliation_active_fn: Optional override used for unit tests.
        inventory_fingerprint: Optional callback used to fingerprint coordinator
            inventory before compiling runtime entities.

    Notes:
        The function registers exactly one coordinator listener and one unload
        hook. It never removes existing entities.
    """
    reconciler = _RuntimeEntityReconciler(
        hass=hass,
        config_entry=config_entry,
        async_add_entities=async_add_entities,
        compile_entities=async_compile_entities,
        is_reconciliation_active_fn=is_reconciliation_active_fn,
        inventory_fingerprint=inventory_fingerprint,
        initial_entities=initial_entities,
    )
    reconciler.attach_coordinator_listener(coordinator)
    reconciler.attach_cleanup()
