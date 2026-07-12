"""Tests for the OPNsense device-ID replacement repair flow."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import aiohttp
from homeassistant.components.repairs import ConfirmRepairFlow
from homeassistant.config_entries import ConfigEntryState
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.exceptions import HomeAssistantError
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.opnsense import repairs
from custom_components.opnsense.const import (
    CONF_DEVICE_UNIQUE_ID,
    CONF_ENTRY_TYPE,
    DOMAIN,
    ENTRY_TYPE_CARP,
)


def _make_entry(
    *,
    entry_id: str = "entry-1",
    device_id: str = "dev1",
    unique_id: str | None = "dev1",
    state: ConfigEntryState = ConfigEntryState.NOT_LOADED,
    carp: bool = False,
    options: dict[str, Any] | None = None,
) -> MockConfigEntry:
    """Build a config entry with connection data used by the repair tests."""
    data: dict[str, Any] = {
        "url": "https://router.example",
        "username": "api-user",
        "password": "api-password",
        CONF_DEVICE_UNIQUE_ID: device_id,
    }
    if carp:
        data[CONF_ENTRY_TYPE] = ENTRY_TYPE_CARP
    entry = MockConfigEntry(domain=DOMAIN, data=data, title="OPNsense Test")
    object.__setattr__(entry, "entry_id", entry_id)
    object.__setattr__(entry, "unique_id", unique_id)
    object.__setattr__(entry, "state", state)
    object.__setattr__(entry, "options", options or {"scan_interval": 30})
    return entry


def _make_flow(hass: Any, entry: MockConfigEntry) -> repairs.DeviceIDMismatchRepairFlow:
    """Create a configured repair flow for direct step testing."""
    flow = repairs.DeviceIDMismatchRepairFlow(
        entry_id=entry.entry_id,
        old_device_id=entry.data[CONF_DEVICE_UNIQUE_ID],
        new_device_id="other",
    )
    flow.hass = hass
    flow.handler = DOMAIN
    flow.issue_id = f"{entry.entry_id}_device_id_mismatched"
    flow.flow_id = "flow-1"
    return flow


def _configure_hass(hass: Any, entry: MockConfigEntry) -> None:
    """Configure the config-entry manager methods shared by flow tests."""
    hass.config_entries = MagicMock()
    hass.config_entries.async_get_entry.return_value = entry
    hass.config_entries.async_entries.return_value = [entry]
    hass.config_entries.async_unload = AsyncMock(return_value=True)
    hass.config_entries.async_update_entry = MagicMock(return_value=True)
    hass.config_entries.async_schedule_reload = MagicMock()


def _patch_registries(
    monkeypatch: pytest.MonkeyPatch,
    entities: list[Any] | None = None,
    devices: list[Any] | None = None,
) -> tuple[MagicMock, MagicMock]:
    """Patch entity/device registry lookups and return the fake registries."""
    entity_registry = MagicMock()
    device_registry = MagicMock()
    monkeypatch.setattr(repairs.er, "async_get", lambda hass: entity_registry)
    monkeypatch.setattr(
        repairs.er,
        "async_entries_for_config_entry",
        lambda registry, config_entry_id: entities or [],
    )
    monkeypatch.setattr(repairs.dr, "async_get", lambda hass: device_registry)
    monkeypatch.setattr(
        repairs.dr,
        "async_entries_for_config_entry",
        lambda registry, config_entry_id: devices or [],
    )
    return entity_registry, device_registry


def _patch_issue_registry(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch issue lookups used to render confirmation placeholders."""
    issue_registry = MagicMock()
    issue_registry.async_get_issue.return_value = SimpleNamespace(
        translation_placeholders={
            "entry_title": "OPNsense Test",
            "old_device_id": "dev1",
            "new_device_id": "other",
        }
    )
    monkeypatch.setattr(repairs.ir, "async_get", lambda hass: issue_registry)
    return issue_registry


def _patch_probe_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    observed_device_id: Any = "other",
    probe_error: BaseException | None = None,
    events: list[str] | None = None,
) -> MagicMock:
    """Patch strict client construction and return the client mock."""
    client = MagicMock()

    async def _probe() -> Any:
        """Return the configured replacement identifier."""
        if events is not None:
            events.append("probe")
        return observed_device_id

    async def _close() -> None:
        """Record strict probe client closure."""
        if events is not None:
            events.append("close")

    client.get_device_unique_id = AsyncMock(
        side_effect=probe_error if probe_error is not None else _probe
    )
    client.async_close = AsyncMock(side_effect=_close)
    factory = MagicMock(return_value=client)
    client.factory = factory
    monkeypatch.setattr(repairs, "create_opnsense_client_from_config_entry", factory)
    return client


@pytest.mark.asyncio
async def test_initial_flow_renders_replacement_ids_and_confirmation_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Initial step should show old/new IDs and a destructive-rebuild confirmation."""
    hass = MagicMock()
    entry = _make_entry()
    _patch_issue_registry(monkeypatch)
    flow = _make_flow(hass, entry)

    result = await flow.async_step_init()

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "confirm"
    assert result["description_placeholders"] == {
        "entry_title": entry.title,
        "old_device_id": "dev1",
        "new_device_id": "other",
    }
    data_schema = result["data_schema"]
    assert data_schema is not None
    assert data_schema({}) == {}


@pytest.mark.asyncio
async def test_confirmation_reprobes_with_strict_client_and_closes_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Confirmation should re-probe the current ID with throw_errors enabled."""
    hass = MagicMock()
    entry = _make_entry()
    _configure_hass(hass, entry)
    _patch_registries(monkeypatch)
    client = _patch_probe_client(monkeypatch)
    monkeypatch.setattr(repairs.ir, "async_delete_issue", MagicMock())
    flow = _make_flow(hass, entry)

    result = await flow.async_step_confirm({})

    assert result["type"] is FlowResultType.CREATE_ENTRY
    client.factory.assert_called_once_with(hass=hass, config_entry=entry, throw_errors=True)
    client.get_device_unique_id.assert_awaited_once_with()
    client.async_close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_duplicate_entry_aborts_before_unload_or_registry_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Duplicate replacement IDs must abort before any destructive mutation."""
    hass = MagicMock()
    entry = _make_entry(state=ConfigEntryState.LOADED)
    duplicate = _make_entry(entry_id="entry-2", device_id="other", unique_id="other")
    _configure_hass(hass, entry)
    hass.config_entries.async_entries.return_value = [entry, duplicate]
    entity_registry, device_registry = _patch_registries(
        monkeypatch,
        entities=[SimpleNamespace(entity_id="sensor.old")],
        devices=[SimpleNamespace(id="device")],
    )
    _patch_probe_client(monkeypatch)
    issue_delete = MagicMock()
    monkeypatch.setattr(repairs.ir, "async_delete_issue", issue_delete)
    flow = _make_flow(hass, entry)

    result = await flow.async_step_confirm({})

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "already_configured"
    hass.config_entries.async_unload.assert_not_awaited()
    entity_registry.async_remove.assert_not_called()
    device_registry.async_update_device.assert_not_called()
    hass.config_entries.async_update_entry.assert_not_called()
    hass.config_entries.async_schedule_reload.assert_not_called()
    issue_delete.assert_not_called()


@pytest.mark.asyncio
async def test_loaded_entry_unloads_before_registry_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A loaded entry must unload before entities or devices are removed."""
    hass = MagicMock()
    entry = _make_entry(state=ConfigEntryState.LOADED)
    events: list[str] = []
    _configure_hass(hass, entry)

    def _unload(entry_id: str) -> bool:
        """Record unload before returning success."""
        events.append("unload")
        return True

    class _OtherEntry:
        """Duplicate-scan candidate whose ID access is observable."""

        entry_id = "entry-2"

        @property
        def unique_id(self) -> str:
            """Record the duplicate candidate comparison."""
            events.append("duplicate_check")
            return "different"

    def _entries(domain: str) -> list[Any]:
        """Record the duplicate scan and return only non-conflicting entries."""
        events.append("duplicate_scan")
        return [entry, _OtherEntry()]

    hass.config_entries.async_entries.side_effect = _entries
    hass.config_entries.async_unload.side_effect = _unload
    entity = SimpleNamespace(entity_id="sensor.old", disabled_by=None)
    device = SimpleNamespace(id="device")
    entity_registry, device_registry = _patch_registries(
        monkeypatch, entities=[entity], devices=[device]
    )
    entity_registry.async_remove.side_effect = lambda entity_id: events.append("entity")
    device_registry.async_update_device.side_effect = lambda device_id, **kwargs: events.append(
        "device"
    )
    hass.config_entries.async_update_entry.side_effect = lambda *args, **kwargs: events.append(
        "config_update"
    )
    hass.config_entries.async_schedule_reload.side_effect = lambda entry_id: events.append("reload")
    _patch_probe_client(monkeypatch, events=events)
    monkeypatch.setattr(repairs.ir, "async_delete_issue", lambda *args: events.append("delete"))
    flow = _make_flow(hass, entry)

    def _create_entry(**_: Any) -> dict[str, str]:
        """Record creation after the reload was scheduled."""
        events.append("create")
        return {"type": "create_entry"}

    object.__setattr__(
        flow,
        "async_create_entry",
        _create_entry,
    )

    await flow.async_step_confirm({})

    assert events == [
        "probe",
        "close",
        "duplicate_scan",
        "duplicate_check",
        "unload",
        "entity",
        "device",
        "config_update",
        "reload",
        "create",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("probe_error", [aiohttp.ClientError("transport"), TimeoutError("timeout")])
async def test_transport_probe_error_aborts_without_mutations(
    monkeypatch: pytest.MonkeyPatch,
    probe_error: BaseException,
) -> None:
    """Raw transport errors from strict probing should abort and close the client."""
    hass = MagicMock()
    entry = _make_entry()
    _configure_hass(hass, entry)
    entity_registry, device_registry = _patch_registries(
        monkeypatch,
        entities=[SimpleNamespace(entity_id="sensor.old")],
        devices=[SimpleNamespace(id="device")],
    )
    client = _patch_probe_client(monkeypatch, probe_error=probe_error)
    flow = _make_flow(hass, entry)

    result = await flow.async_step_confirm({})

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "cannot_connect"
    client.async_close.assert_awaited_once_with()
    entity_registry.async_remove.assert_not_called()
    device_registry.async_update_device.assert_not_called()
    hass.config_entries.async_unload.assert_not_awaited()
    hass.config_entries.async_update_entry.assert_not_called()
    hass.config_entries.async_schedule_reload.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("observed_device_id", [None, ""])
async def test_invalid_observed_device_id_aborts_without_mutations(
    monkeypatch: pytest.MonkeyPatch,
    observed_device_id: str | None,
) -> None:
    """Empty replacement IDs should abort after closing the probe client."""
    hass = MagicMock()
    entry = _make_entry()
    _configure_hass(hass, entry)
    entity_registry, device_registry = _patch_registries(
        monkeypatch,
        entities=[SimpleNamespace(entity_id="sensor.old")],
        devices=[SimpleNamespace(id="device")],
    )
    client = _patch_probe_client(monkeypatch, observed_device_id=observed_device_id)
    flow = _make_flow(hass, entry)

    result = await flow.async_step_confirm({})

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "cannot_connect"
    client.async_close.assert_awaited_once_with()
    entity_registry.async_remove.assert_not_called()
    device_registry.async_update_device.assert_not_called()
    hass.config_entries.async_unload.assert_not_awaited()
    hass.config_entries.async_update_entry.assert_not_called()
    hass.config_entries.async_schedule_reload.assert_not_called()


@pytest.mark.asyncio
async def test_unload_failure_aborts_before_registry_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed unload must leave entity/device registries and entry data intact."""
    hass = MagicMock()
    entry = _make_entry(state=ConfigEntryState.LOADED)
    _configure_hass(hass, entry)
    hass.config_entries.async_unload.return_value = False
    entity_registry, device_registry = _patch_registries(
        monkeypatch,
        entities=[SimpleNamespace(entity_id="sensor.old")],
        devices=[SimpleNamespace(id="device")],
    )
    client = _patch_probe_client(monkeypatch)
    flow = _make_flow(hass, entry)

    result = await flow.async_step_confirm({})

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "cannot_unload"
    client.async_close.assert_awaited_once_with()
    hass.config_entries.async_unload.assert_awaited_once_with(entry.entry_id)
    entity_registry.async_remove.assert_not_called()
    device_registry.async_update_device.assert_not_called()
    hass.config_entries.async_update_entry.assert_not_called()
    hass.config_entries.async_schedule_reload.assert_not_called()


@pytest.mark.asyncio
async def test_entry_removed_during_unload_aborts_before_registry_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A removed entry after unload must not allow stale registry mutations."""
    hass = MagicMock()
    entry = _make_entry(state=ConfigEntryState.LOADED)
    _configure_hass(hass, entry)
    hass.config_entries.async_get_entry.side_effect = [entry, entry, None]
    entity_registry, device_registry = _patch_registries(
        monkeypatch,
        entities=[SimpleNamespace(entity_id="sensor.old")],
        devices=[SimpleNamespace(id="device")],
    )
    _patch_probe_client(monkeypatch)
    flow = _make_flow(hass, entry)

    result = await flow.async_step_confirm({})

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "entry_changed"
    entity_registry.async_remove.assert_not_called()
    device_registry.async_update_device.assert_not_called()
    hass.config_entries.async_update_entry.assert_not_called()
    hass.config_entries.async_schedule_reload.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation_stage", ["probe", "unload"])
@pytest.mark.parametrize("field", ["url", "username", "password", "options", "unique_id"])
async def test_entry_changed_during_probe_or_unload_aborts_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
    mutation_stage: str,
    field: str,
) -> None:
    """Config-entry changes during awaited work must stop the destructive repair."""
    hass = MagicMock()
    entry = _make_entry(
        state=ConfigEntryState.LOADED if mutation_stage == "unload" else ConfigEntryState.NOT_LOADED
    )
    _configure_hass(hass, entry)
    entity_registry, device_registry = _patch_registries(
        monkeypatch,
        entities=[SimpleNamespace(entity_id="sensor.old")],
        devices=[SimpleNamespace(id="device")],
    )
    client = _patch_probe_client(monkeypatch)

    def _mutate_entry() -> None:
        """Apply one persisted-entry mutation while the repair is awaiting."""
        if field in {"url", "username", "password"}:
            object.__setattr__(entry, "data", {**entry.data, field: f"changed-{field}"})
        elif field == "options":
            entry.options["scan_interval"] = 99
        else:
            object.__setattr__(entry, "unique_id", "changed-unique-id")

    if mutation_stage == "probe":

        async def _probe_and_mutate() -> str:
            """Mutate the entry before the probe completes."""
            _mutate_entry()
            return "other"

        client.get_device_unique_id.side_effect = _probe_and_mutate
    else:

        def _unload_and_mutate(entry_id: str) -> bool:
            """Mutate the entry while unloading it."""
            _mutate_entry()
            return True

        hass.config_entries.async_unload.side_effect = _unload_and_mutate

    flow = _make_flow(hass, entry)

    result = await flow.async_step_confirm({})

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "entry_changed"
    entity_registry.async_remove.assert_not_called()
    device_registry.async_update_device.assert_not_called()
    hass.config_entries.async_update_entry.assert_not_called()
    hass.config_entries.async_schedule_reload.assert_not_called()
    if mutation_stage == "probe":
        hass.config_entries.async_unload.assert_not_awaited()
    else:
        hass.config_entries.async_unload.assert_awaited_once_with(entry.entry_id)


@pytest.mark.asyncio
async def test_cleanup_removes_disabled_entities_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Entity cleanup should remove all entries, including integration/user-disabled ones."""
    hass = MagicMock()
    entry = _make_entry()
    _configure_hass(hass, entry)
    entities = [
        SimpleNamespace(entity_id="sensor.enabled", disabled_by=None),
        SimpleNamespace(entity_id="sensor.integration_disabled", disabled_by="integration"),
        SimpleNamespace(entity_id="sensor.user_disabled", disabled_by="user"),
    ]
    entity_registry, _ = _patch_registries(monkeypatch, entities=entities)
    _patch_probe_client(monkeypatch)
    monkeypatch.setattr(repairs.ir, "async_delete_issue", MagicMock())
    flow = _make_flow(hass, entry)

    await flow.async_step_confirm({})

    assert [call.args[0] for call in entity_registry.async_remove.call_args_list] == [
        "sensor.enabled",
        "sensor.integration_disabled",
        "sensor.user_disabled",
    ]


@pytest.mark.asyncio
async def test_cleanup_removes_only_this_config_entry_from_shared_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shared devices should retain associations with other config entries."""
    hass = MagicMock()
    entry = _make_entry()
    _configure_hass(hass, entry)
    devices = [
        SimpleNamespace(id="device-only", config_entries={entry.entry_id}),
        SimpleNamespace(id="device-shared", config_entries={entry.entry_id, "entry-2"}),
    ]
    _, device_registry = _patch_registries(monkeypatch, devices=devices)
    _patch_probe_client(monkeypatch)
    monkeypatch.setattr(repairs.ir, "async_delete_issue", MagicMock())
    flow = _make_flow(hass, entry)

    await flow.async_step_confirm({})

    assert device_registry.async_update_device.call_args_list == [
        (("device-only",), {"remove_config_entry_id": entry.entry_id}),
        (("device-shared",), {"remove_config_entry_id": entry.entry_id}),
    ]


@pytest.mark.asyncio
async def test_update_preserves_connection_and_options_while_replacing_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Entry update should replace only the device ID and unique ID."""
    hass = MagicMock()
    entry = _make_entry(options={"scan_interval": 45})
    _configure_hass(hass, entry)
    _patch_registries(monkeypatch)
    _patch_probe_client(monkeypatch)
    monkeypatch.setattr(repairs.ir, "async_delete_issue", MagicMock())
    flow = _make_flow(hass, entry)

    await flow.async_step_confirm({})

    hass.config_entries.async_update_entry.assert_called_once_with(
        entry,
        data={**entry.data, CONF_DEVICE_UNIQUE_ID: "other"},
        unique_id="other",
    )
    assert entry.data["url"] == "https://router.example"
    assert entry.data["username"] == "api-user"
    assert entry.data["password"] == "api-password"
    assert entry.options == {"scan_interval": 45}


@pytest.mark.asyncio
async def test_success_deletes_issue_and_schedules_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Successful rebuild should schedule reload and let the manager delete the issue."""
    hass = MagicMock()
    entry = _make_entry()
    _configure_hass(hass, entry)
    _patch_registries(monkeypatch)
    _patch_probe_client(monkeypatch)
    issue_delete = MagicMock()
    monkeypatch.setattr(repairs.ir, "async_delete_issue", issue_delete)
    flow = _make_flow(hass, entry)

    result = await flow.async_step_confirm({})

    assert result["type"] is FlowResultType.CREATE_ENTRY
    issue_delete.assert_not_called()
    hass.config_entries.async_schedule_reload.assert_called_once_with(entry.entry_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_point", ["device", "entry"])
async def test_cleanup_failure_aborts_without_issue_delete_or_reload(
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    """Registry/config failures retain the repair issue after prior mutations."""
    hass = MagicMock()
    entry = _make_entry()
    _configure_hass(hass, entry)
    entity_registry, device_registry = _patch_registries(
        monkeypatch,
        entities=[SimpleNamespace(entity_id="sensor.old")],
        devices=[SimpleNamespace(id="device")],
    )
    _patch_probe_client(monkeypatch)
    issue_delete = MagicMock()
    monkeypatch.setattr(repairs.ir, "async_delete_issue", issue_delete)
    if failure_point == "device":
        device_registry.async_update_device.side_effect = HomeAssistantError("device update")
    else:
        hass.config_entries.async_update_entry.side_effect = HomeAssistantError("entry update")
    flow = _make_flow(hass, entry)

    result = await flow.async_step_confirm({})

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "repair_failed"
    entity_registry.async_remove.assert_called_once_with("sensor.old")
    if failure_point == "device":
        device_registry.async_update_device.assert_called_once()
        hass.config_entries.async_update_entry.assert_not_called()
    else:
        device_registry.async_update_device.assert_called_once()
        hass.config_entries.async_update_entry.assert_called_once()
    issue_delete.assert_not_called()
    hass.config_entries.async_schedule_reload.assert_not_called()


@pytest.mark.asyncio
async def test_removed_entry_aborts_without_mutations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deleted config entry should abort before creating a strict client."""
    hass = MagicMock()
    entry = _make_entry()
    _configure_hass(hass, entry)
    hass.config_entries.async_get_entry.return_value = None
    client_factory = MagicMock()
    monkeypatch.setattr(repairs, "create_opnsense_client_from_config_entry", client_factory)
    flow = _make_flow(hass, entry)

    result = await flow.async_step_confirm({})

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "entry_not_found"
    client_factory.assert_not_called()
    hass.config_entries.async_unload.assert_not_awaited()
    hass.config_entries.async_update_entry.assert_not_called()
    hass.config_entries.async_schedule_reload.assert_not_called()


@pytest.mark.asyncio
async def test_carp_entry_is_rejected_without_mutations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CARP entries never create device-ID mismatch repairs."""
    hass = MagicMock()
    entry = _make_entry(carp=True)
    _configure_hass(hass, entry)
    client_factory = MagicMock()
    monkeypatch.setattr(repairs, "create_opnsense_client_from_config_entry", client_factory)
    flow = _make_flow(hass, entry)

    result = await flow.async_step_confirm({})

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "entry_not_found"
    client_factory.assert_not_called()
    hass.config_entries.async_unload.assert_not_awaited()
    hass.config_entries.async_update_entry.assert_not_called()
    hass.config_entries.async_schedule_reload.assert_not_called()


@pytest.mark.asyncio
async def test_fix_flow_factory_validates_issue_suffix_and_payload() -> None:
    """Only well-formed device-ID issues should construct the destructive flow."""
    hass = MagicMock()
    data: dict[str, str | int | float | None] = {
        "entry_id": "entry-1",
        "old_device_id": "dev1",
        "new_device_id": "other",
    }

    flow = await repairs.async_create_fix_flow(hass, "entry-1_device_id_mismatched", data)

    assert isinstance(flow, repairs.DeviceIDMismatchRepairFlow)
    unknown_flow = await repairs.async_create_fix_flow(hass, "unrelated", None)
    assert isinstance(unknown_flow, ConfirmRepairFlow)
