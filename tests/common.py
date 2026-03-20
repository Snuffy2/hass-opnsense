"""Local test helpers for the hass-opnsense test suite."""

from __future__ import annotations

from typing import Any

from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from homeassistant.util import ulid as ulid_util


class MockConfigEntry(config_entries.ConfigEntry):
    """Helper for creating config entries with sensible test defaults."""

    def __init__(
        self,
        *,
        data: dict[str, Any] | None = None,
        disabled_by: config_entries.ConfigEntryDisabler | None = None,
        discovery_keys: dict[str, Any] | None = None,
        domain: str = "test",
        entry_id: str | None = None,
        minor_version: int = 1,
        options: dict[str, Any] | None = None,
        pref_disable_new_entities: bool | None = None,
        pref_disable_polling: bool | None = None,
        reason: str | None = None,
        source: str | None = config_entries.SOURCE_USER,
        state: config_entries.ConfigEntryState | None = None,
        subentries_data: tuple[Any, ...] | None = None,
        title: str = "Mock Title",
        unique_id: str | None = None,
        version: int = 1,
    ) -> None:
        """Initialize a mock config entry."""
        kwargs = {
            "data": data or {},
            "disabled_by": disabled_by,
            "discovery_keys": discovery_keys or {},
            "domain": domain,
            "entry_id": entry_id or ulid_util.ulid_now(),
            "minor_version": minor_version,
            "options": options or {},
            "pref_disable_new_entities": pref_disable_new_entities,
            "pref_disable_polling": pref_disable_polling,
            "subentries_data": subentries_data or (),
            "title": title,
            "unique_id": unique_id,
            "version": version,
        }
        if source is not None:
            kwargs["source"] = source
        if state is not None:
            kwargs["state"] = state
        super().__init__(**kwargs)
        if reason is not None:
            object.__setattr__(self, "reason", reason)

    def add_to_hass(self, hass: HomeAssistant) -> None:
        """Test helper to add the entry to Home Assistant."""
        hass.config_entries._entries[self.entry_id] = self

    def add_to_manager(self, manager: config_entries.ConfigEntries) -> None:
        """Test helper to add the entry to a config entry manager."""
        manager._entries[self.entry_id] = self

    def mock_state(
        self,
        hass: HomeAssistant,
        state: config_entries.ConfigEntryState,
        reason: str | None = None,
    ) -> None:
        """Mock the state of a config entry."""
        self._async_set_state(hass, state, reason)
