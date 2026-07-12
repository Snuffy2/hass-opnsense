"""Repair flows for the OPNsense integration."""

from aiopnsense.exceptions import OPNsenseError
from homeassistant.components.repairs import ConfirmRepairFlow, RepairsFlow, RepairsFlowResult
from homeassistant.config_entries import ConfigEntry, ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr, entity_registry as er, issue_registry as ir
import voluptuous as vol

from .const import CONF_DEVICE_UNIQUE_ID, DOMAIN
from .helpers import create_opnsense_client_from_config_entry, is_carp_entry

_ISSUE_SUFFIX = "_device_id_mismatched"


def async_create_device_id_mismatch_issue(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    observed_device_id: str,
) -> None:
    """Create a fixable hardware-replacement issue for a normal device entry."""
    old_device_id = config_entry.data[CONF_DEVICE_UNIQUE_ID]
    ir.async_create_issue(
        hass=hass,
        domain=DOMAIN,
        issue_id=f"{config_entry.entry_id}{_ISSUE_SUFFIX}",
        is_fixable=True,
        is_persistent=False,
        severity=ir.IssueSeverity.ERROR,
        translation_key="device_id_mismatched",
        translation_placeholders={
            "entry_title": config_entry.title,
            "old_device_id": old_device_id,
            "new_device_id": observed_device_id,
        },
        data={
            "entry_id": config_entry.entry_id,
            "old_device_id": old_device_id,
            "new_device_id": observed_device_id,
        },
    )


class DeviceIDMismatchRepairFlow(RepairsFlow):
    """Rebuild one OPNsense config entry after confirmed hardware replacement."""

    def __init__(self, entry_id: str, old_device_id: str, new_device_id: str) -> None:
        """Initialize a repair flow from issue data."""
        self._entry_id = entry_id
        self._old_device_id = old_device_id
        self._new_device_id = new_device_id
        self._description_placeholders: dict[str, str] = {
            "entry_title": "",
            "old_device_id": old_device_id,
            "new_device_id": new_device_id,
        }

    async def async_step_init(self, user_input: dict[str, str] | None = None) -> RepairsFlowResult:
        """Load issue placeholders and display the confirmation step."""
        del user_input
        issue_registry = ir.async_get(self.hass)
        issue = issue_registry.async_get_issue(self.handler, self.issue_id)
        if issue is not None and issue.translation_placeholders is not None:
            self._description_placeholders = dict(issue.translation_placeholders)
        return await self.async_step_confirm()

    async def async_step_confirm(
        self, user_input: dict[str, str] | None = None
    ) -> RepairsFlowResult:
        """Confirm and perform the ordered registry rebuild."""
        if user_input is None:
            return self.async_show_form(
                step_id="confirm",
                data_schema=vol.Schema({}),
                description_placeholders=self._description_placeholders,
            )

        entry = self.hass.config_entries.async_get_entry(self._entry_id)
        if entry is None or is_carp_entry(entry):
            return self.async_abort(reason="entry_not_found")

        try:
            client = create_opnsense_client_from_config_entry(
                hass=self.hass,
                config_entry=entry,
                throw_errors=True,
            )
            try:
                observed_device_id = await client.get_device_unique_id()
            finally:
                await client.async_close()
        except OPNsenseError:
            return self.async_abort(reason="cannot_connect")

        if not isinstance(observed_device_id, str) or not observed_device_id:
            return self.async_abort(reason="cannot_connect")

        duplicate = next(
            (
                candidate
                for candidate in self.hass.config_entries.async_entries(DOMAIN)
                if candidate.entry_id != entry.entry_id
                and candidate.unique_id == observed_device_id
            ),
            None,
        )
        if duplicate is not None:
            return self.async_abort(reason="already_configured")

        if (
            entry.state is ConfigEntryState.LOADED
            and not await self.hass.config_entries.async_unload(entry.entry_id)
        ):
            return self.async_abort(reason="cannot_unload")

        entity_registry = er.async_get(self.hass)
        for entity in er.async_entries_for_config_entry(entity_registry, entry.entry_id):
            entity_registry.async_remove(entity.entity_id)

        device_registry = dr.async_get(self.hass)
        for device in dr.async_entries_for_config_entry(device_registry, entry.entry_id):
            device_registry.async_update_device(
                device.id,
                remove_config_entry_id=entry.entry_id,
            )

        new_data = {**entry.data, CONF_DEVICE_UNIQUE_ID: observed_device_id}
        self.hass.config_entries.async_update_entry(
            entry,
            data=new_data,
            unique_id=observed_device_id,
        )
        ir.async_delete_issue(
            self.hass,
            DOMAIN,
            f"{entry.entry_id}{_ISSUE_SUFFIX}",
        )
        self.hass.config_entries.async_schedule_reload(entry.entry_id)
        return self.async_create_entry(data={})


async def async_create_fix_flow(
    hass: HomeAssistant,
    issue_id: str,
    data: dict[str, str | int | float | None] | None,
) -> RepairsFlow:
    """Create a device-ID replacement flow for a well-formed issue."""
    del hass
    if not issue_id.endswith(_ISSUE_SUFFIX) or data is None:
        return ConfirmRepairFlow()

    entry_id = data.get("entry_id")
    old_device_id = data.get("old_device_id")
    new_device_id = data.get("new_device_id")
    if not (
        isinstance(entry_id, str)
        and entry_id
        and isinstance(old_device_id, str)
        and old_device_id
        and isinstance(new_device_id, str)
        and new_device_id
        and issue_id == f"{entry_id}{_ISSUE_SUFFIX}"
    ):
        return ConfirmRepairFlow()

    return DeviceIDMismatchRepairFlow(
        entry_id=entry_id,
        old_device_id=old_device_id,
        new_device_id=new_device_id,
    )
