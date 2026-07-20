"""Support for tracking for OPNsense devices."""

from collections.abc import Iterable, Mapping, MutableMapping
import contextlib
from datetime import datetime, timedelta, timezone
import logging
from typing import Any

from homeassistant.components.device_tracker import ScannerEntity, SourceType
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, entity_registry as er
from homeassistant.helpers.device_registry import (
    CONNECTION_NETWORK_MAC,
    DeviceRegistry,
    async_get as async_get_dev_reg,
)
from homeassistant.helpers.entity import DeviceInfo, Entity
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.util import slugify

from .config_flow import normalize_mac_address
from .const import (
    CONF_DEVICE_TRACKER_CONSIDER_HOME,
    CONF_DEVICE_TRACKER_ENABLED,
    CONF_DEVICE_UNIQUE_ID,
    CONF_DEVICES,
    DEFAULT_DEVICE_TRACKER_CONSIDER_HOME,
    DEFAULT_DEVICE_TRACKER_ENABLED,
    DEVICE_TRACKER_COORDINATOR,
    DOMAIN,
    SHOULD_RELOAD,
    TRACKED_MACS,
)
from .coordinator import OPNsenseDataUpdateCoordinator
from .entity import OPNsenseBaseEntity
from .helpers import (
    detach_shared_router_parent,
    dict_get,
    get_arp_ip,
    get_arp_mac,
    normalize_arp_mac,
)
from .repair_reconciliation import is_reconciliation_active, record_desired_entities
from .runtime_entity_reconciliation import attach_runtime_entity_reconciler

_LOGGER: logging.Logger = logging.getLogger(__name__)


def _arp_inventory_fingerprint(state: object) -> tuple[str, ...]:
    """Return stable normalized MAC identities from valid ARP rows."""
    if not isinstance(state, Mapping):
        return ()
    arp_entries = state.get("arp_table")
    if not isinstance(arp_entries, list):
        return ()
    return tuple(
        sorted(
            {
                _normalize_mac_for_device_tracker(mac)
                for row in arp_entries
                if isinstance(row, MutableMapping) and (mac := get_arp_mac(row))
            }
        )
    )


def _normalize_mac_for_device_tracker(mac_address: str) -> str:
    """Normalize a user-facing or payload MAC with canonical fallback.

    Args:
        mac_address: A raw MAC-like input.

    Returns:
        Canonical MAC when possible, otherwise permissive lower-case normalized value.
    """
    normalized_mac = normalize_mac_address(mac_address)
    if normalized_mac is not None:
        return normalized_mac
    return normalize_arp_mac(mac_address)


def _device_data_from_arp_entry(
    mac_address: str,
    arp_entry: MutableMapping[str, Any],
) -> dict[str, Any]:
    """Build tracked device data from an ARP table entry.

    Args:
        mac_address: MAC address used as the tracked-device identity.
        arp_entry: ARP entry containing optional metadata.

    Returns:
        A device dictionary populated with the MAC address and metadata.
    """
    device: dict[str, Any] = {"mac": mac_address}

    hostname = _hostname_from_arp_entry(arp_entry)
    if hostname is not None:
        device["hostname"] = hostname

    manufacturer = arp_entry.get("manufacturer")
    if isinstance(manufacturer, str) and manufacturer:
        device["manufacturer"] = manufacturer

    return device


def _device_from_arp_entry(mac_address: str, arp_entries: list[Any]) -> dict[str, Any]:
    """Build tracked device data from a configured MAC and matching ARP entry.

    Args:
        mac_address: Configured MAC address for the tracker entity.
        arp_entries: Raw ARP entries returned by OPNsense.

    Returns:
        A device dictionary for the matching ARP entry, or a MAC-only fallback.
    """
    device: dict[str, Any] = {"mac": mac_address}
    normalized_mac = _normalize_mac_for_device_tracker(mac_address)
    for arp_entry in arp_entries:
        if not isinstance(arp_entry, MutableMapping):
            continue
        arp_mac = get_arp_mac(arp_entry)
        if not arp_mac or _normalize_mac_for_device_tracker(arp_mac) != normalized_mac:
            continue
        device.update(_device_data_from_arp_entry(mac_address, arp_entry))
        break
    return device


def _devices_from_arp_entries(arp_entries: list[Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """Build tracked device data from unique ARP table MAC addresses.

    Args:
        arp_entries: Raw ARP entries returned by OPNsense.

    Returns:
        A tuple of device dictionaries and the unique MAC addresses found.
    """
    devices: list[dict[str, Any]] = []
    mac_addresses: list[str] = []

    for arp_entry in arp_entries:
        if not isinstance(arp_entry, MutableMapping):
            continue
        mac_address = get_arp_mac(arp_entry)
        if not mac_address:
            continue
        normalized_mac = _normalize_mac_for_device_tracker(mac_address)
        if not normalized_mac or normalized_mac in mac_addresses:
            continue
        mac_addresses.append(normalized_mac)
        devices.append(_device_data_from_arp_entry(normalized_mac, arp_entry))

    return devices, mac_addresses


def _track_all_arp_entries_are_complete(arp_entries: list[Any]) -> bool:
    """Return whether every non-entity ARP row is skippable in track-all mode.

    Args:
        arp_entries: Raw ARP entries returned by OPNsense.

    Returns:
        ``True`` when every row is a mapping and any row with a normalizable MAC
        contributes a unique MAC address.
    """
    seen_macs: set[str] = set()
    for arp_entry in arp_entries:
        if not isinstance(arp_entry, MutableMapping):
            return False
        mac_address = get_arp_mac(arp_entry)
        if not mac_address:
            continue
        normalized_mac = _normalize_mac_for_device_tracker(mac_address)
        if not normalized_mac:
            continue
        if normalized_mac in seen_macs:
            continue
        seen_macs.add(normalized_mac)
    return True


def _hostname_from_arp_entry(entry: MutableMapping[str, Any]) -> str | None:
    """Return the normalized hostname from an ARP entry.

    Args:
        entry: ARP entry to normalize.

    Returns:
        The stripped hostname, or ``None`` when no usable hostname exists.
    """
    hostname = entry.get("hostname")
    if not isinstance(hostname, str):
        return None
    hostname = hostname.strip("?")
    return hostname or None


def _normalize_mac_values(values: Iterable[Any]) -> list[str]:
    """Normalize and dedupe MAC strings preserving discovery order."""
    normalized_macs: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        normalized_mac = _normalize_mac_for_device_tracker(value)
        if not normalized_mac or normalized_mac in seen:
            continue
        normalized_macs.append(normalized_mac)
        seen.add(normalized_mac)
    return normalized_macs


def _arp_expires_attribute(value: object) -> str | datetime | None:
    """Return the Home Assistant attribute value for an ARP expiry.

    Args:
        value: Raw expiry value from OPNsense.

    Returns:
        ``"Never"`` for permanent entries, a datetime for relative expiry, or ``None``.
    """
    if value == -1:
        return "Never"
    if isinstance(value, int | float):
        return datetime.now().astimezone() + timedelta(seconds=value)
    return None


def _update_arp_extra_state_attributes(
    attributes: dict[str, Any],
    entry: MutableMapping[str, Any],
) -> None:
    """Update optional ARP extra state attributes from a coordinator entry.

    Args:
        attributes: Entity attributes to mutate in place.
        entry: ARP entry providing optional metadata.
    """
    for attr in ("interface", "expires", "type"):
        attributes.pop(attr, None)

    interface = entry.get("intf_description")
    if interface:
        attributes["interface"] = interface

    expires = _arp_expires_attribute(entry.get("expires"))
    if expires is not None:
        attributes["expires"] = expires

    arp_type = entry.get("type")
    if arp_type:
        attributes["type"] = arp_type


def _compile_tracked_devices(
    config_entry: ConfigEntry,
    arp_entries: list[Any],
) -> tuple[list[dict[str, Any]], list[str], bool]:
    """Compile device tracker source data from options and ARP entries.

    Args:
        config_entry: Config entry containing device-tracker options.
        arp_entries: Raw ARP entries returned by OPNsense.

    Returns:
        A tuple of devices, MAC addresses, and the default enabled flag.
    """
    if not config_entry.options.get(CONF_DEVICE_TRACKER_ENABLED, DEFAULT_DEVICE_TRACKER_ENABLED):
        return [], [], False

    configured_mac_addresses: list[str] = []
    for mac_address in config_entry.options.get(CONF_DEVICES, []):
        if not isinstance(mac_address, str):
            continue
        normalized_mac = _normalize_mac_for_device_tracker(mac_address)
        if not normalized_mac or normalized_mac in configured_mac_addresses:
            continue
        configured_mac_addresses.append(normalized_mac)

    if configured_mac_addresses:
        _LOGGER.debug(
            "[device_tracker async_setup_entry] configured_mac_addresses: %s",
            configured_mac_addresses,
        )
        devices = [
            _device_from_arp_entry(mac_address, arp_entries)
            for mac_address in configured_mac_addresses
        ]
        return devices, list(configured_mac_addresses), True

    devices, mac_addresses = _devices_from_arp_entries(arp_entries)
    return devices, mac_addresses, False


def _build_scanner_entities(
    config_entry: ConfigEntry,
    coordinator: OPNsenseDataUpdateCoordinator,
    devices: list[dict[str, Any]],
    *,
    enabled_default: bool,
) -> list[OPNsenseScannerEntity]:
    """Build scanner entities from compiled tracker device data.

    Args:
        config_entry: Config entry owning the tracker entities.
        coordinator: Coordinator providing ARP updates.
        devices: Compiled device dictionaries containing stable MAC identities.
        enabled_default: Whether new entities should be enabled by default.

    Returns:
        Scanner entities for device rows with string MAC identities.
    """
    entities: list[OPNsenseScannerEntity] = []
    for device in devices:
        mac = device.get("mac")
        if not isinstance(mac, str):
            continue
        entities.append(
            OPNsenseScannerEntity(
                config_entry=config_entry,
                coordinator=coordinator,
                enabled_default=enabled_default,
                mac=mac,
                mac_vendor=device.get("manufacturer"),
                hostname=device.get("hostname"),
            )
        )
    return entities


async def _compile_runtime_track_all_entities(
    config_entry: ConfigEntry,
    coordinator: OPNsenseDataUpdateCoordinator,
) -> list[OPNsenseScannerEntity]:
    """Compile current ARP devices for add-only track-all reconciliation.

    Args:
        config_entry: Config entry owning the tracker entities.
        coordinator: Coordinator providing the current ARP inventory.

    Returns:
        Scanner entity candidates from valid rows in the current ARP table.
    """
    state = coordinator.data
    if not isinstance(state, MutableMapping):
        return []
    arp_entries = dict_get(state, "arp_table")
    if not isinstance(arp_entries, list):
        return []
    devices, _mac_addresses = _devices_from_arp_entries(arp_entries)
    return _build_scanner_entities(
        config_entry,
        coordinator,
        devices,
        enabled_default=False,
    )


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up device tracker entities for the OPNsense component.

    Args:
        hass: Home Assistant instance.
        config_entry: Config entry being set up.
        async_add_entities: Callback used to register new entities.
    """
    dev_reg = async_get_dev_reg(hass)

    previous_mac_addresses: list = config_entry.data.get(TRACKED_MACS, [])
    coordinator: OPNsenseDataUpdateCoordinator = getattr(
        config_entry.runtime_data, DEVICE_TRACKER_COORDINATOR
    )
    state: dict[str, Any] = coordinator.data
    if not isinstance(state, MutableMapping):
        _LOGGER.error("Missing state data in device tracker async_setup_entry")
        return

    arp_inventory = dict_get(state, "arp_table")
    arp_entries = arp_inventory
    configured_macs = config_entry.options.get(CONF_DEVICES, [])
    has_configured_macs = bool(
        isinstance(configured_macs, list)
        and any(
            isinstance(mac_address, str) and mac_address.strip() for mac_address in configured_macs
        )
    )
    track_all_inventory = isinstance(arp_entries, list)
    if not isinstance(arp_entries, list):
        arp_entries = []
    track_all_inventory_is_authoritative = (
        coordinator.category_is_authoritative("arp_table")
        and track_all_inventory
        and _track_all_arp_entries_are_complete(arp_entries)
    )
    inventory_authoritative = has_configured_macs or track_all_inventory_is_authoritative
    devices, mac_addresses, enabled_default = _compile_tracked_devices(config_entry, arp_entries)

    entities = _build_scanner_entities(
        config_entry,
        coordinator,
        devices,
        enabled_default=enabled_default,
    )

    persisted_mac_addresses = _normalize_mac_values(previous_mac_addresses)
    pending_mac_addresses: list[str] = []
    persistence_retry_generation = 0

    def _persist_discovered_macs(entities_to_add: list[OPNsenseScannerEntity]) -> None:
        """Persist discovered MAC identities after successful entity registration."""
        nonlocal pending_mac_addresses, persisted_mac_addresses, persistence_retry_generation
        discovered_mac_addresses = _normalize_mac_values(
            entity.mac_address for entity in entities_to_add
        )
        pending_mac_addresses = _normalize_mac_values(
            pending_mac_addresses + discovered_mac_addresses
        )
        merged = _normalize_mac_values(persisted_mac_addresses + pending_mac_addresses)
        if merged == persisted_mac_addresses:
            pending_mac_addresses = []
            return

        setattr(config_entry.runtime_data, SHOULD_RELOAD, False)
        new_data = config_entry.data.copy()
        new_data[TRACKED_MACS] = merged
        try:
            hass.config_entries.async_update_entry(config_entry, data=new_data)
        except HomeAssistantError, RuntimeError, ValueError:
            persistence_retry_generation += 1
            _LOGGER.exception(
                "Could not persist runtime device tracker MACs for entry %s; "
                "will retry after a coordinator update",
                config_entry.entry_id,
            )
            return
        persisted_mac_addresses = merged
        pending_mac_addresses = []

    def _add_entities_with_tracking(
        new_entities: Iterable[Entity], update_before_add: bool = False
    ) -> None:
        """Add entities and persist successfully registered track-all MACs."""
        submitted_entities = list(new_entities)
        if update_before_add:
            async_add_entities(submitted_entities, True)
        else:
            async_add_entities(submitted_entities)
        _persist_discovered_macs(
            [entity for entity in submitted_entities if isinstance(entity, OPNsenseScannerEntity)]
        )

    if not is_reconciliation_active(config_entry):
        _cleanup_stale_tracked_devices(
            hass=hass,
            config_entry=config_entry,
            device_registry=dev_reg,
            previous_mac_addresses=previous_mac_addresses,
            current_mac_addresses=mac_addresses,
            inventory_authoritative=inventory_authoritative,
            expand_owned_mac_addresses=has_configured_macs,
        )

    if inventory_authoritative and set(mac_addresses) != set(previous_mac_addresses):
        setattr(config_entry.runtime_data, SHOULD_RELOAD, False)
        new_data = config_entry.data.copy()
        new_data[TRACKED_MACS] = mac_addresses.copy()
        hass.config_entries.async_update_entry(config_entry, data=new_data)
        persisted_mac_addresses = _normalize_mac_values(mac_addresses)

    _LOGGER.debug("[device_tracker async_setup_entry] entities: %s", len(entities))
    record_desired_entities(
        config_entry,
        "device_tracker",
        entities,
        {
            "arp": inventory_authoritative,
        },
    )
    async_add_entities(entities)

    tracker_enabled = config_entry.options.get(
        CONF_DEVICE_TRACKER_ENABLED, DEFAULT_DEVICE_TRACKER_ENABLED
    )
    if tracker_enabled and not has_configured_macs:

        async def async_compile_runtime_entities() -> list[OPNsenseScannerEntity]:
            """Compile track-all ARP inventory for runtime additions."""
            _persist_discovered_macs([])
            return await _compile_runtime_track_all_entities(config_entry, coordinator)

        attach_runtime_entity_reconciler(
            hass,
            config_entry,
            coordinator,
            _add_entities_with_tracking,
            entities,
            async_compile_runtime_entities,
            inventory_fingerprint=lambda: (
                _arp_inventory_fingerprint(coordinator.data),
                persistence_retry_generation,
            ),
        )


def _cleanup_stale_tracked_devices(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    device_registry: DeviceRegistry,
    previous_mac_addresses: list[Any],
    current_mac_addresses: list[str],
    *,
    inventory_authoritative: bool,
    expand_owned_mac_addresses: bool,
) -> None:
    """Remove stale tracker entities and reparent shared tracker devices.

    Args:
        hass: Home Assistant runtime object.
        config_entry: Active integration config entry for this setup run.
        device_registry: Device registry used to query and mutate tracked devices.
        previous_mac_addresses: Previously persisted MAC addresses from config entry data.
        current_mac_addresses: MAC addresses currently discovered during setup.
        inventory_authoritative: Whether the current inventory can safely drive deletion.
        expand_owned_mac_addresses: Whether config-entry-owned tracker devices should
            supplement persisted MAC bookkeeping for a configured-mode transition.
    """
    if not inventory_authoritative:
        return

    owned_mac_addresses: set[str] = set()
    if expand_owned_mac_addresses:
        router_identifier = (DOMAIN, config_entry.data[CONF_DEVICE_UNIQUE_ID])
        for device_entry in dr.async_entries_for_config_entry(
            device_registry, config_entry.entry_id
        ):
            if router_identifier in device_entry.identifiers:
                continue
            owned_mac_addresses.update(
                connection_value
                for connection_type, connection_value in device_entry.connections
                if connection_type == CONNECTION_NETWORK_MAC and connection_value
            )

    stale_mac_addresses = (set(previous_mac_addresses) | owned_mac_addresses) - set(
        current_mac_addresses
    )
    if not stale_mac_addresses:
        return

    entity_registry = er.async_get(hass)
    router_device = device_registry.async_get_device(
        identifiers={(DOMAIN, config_entry.data[CONF_DEVICE_UNIQUE_ID])}
    )
    router_device_id = router_device.id if router_device else None

    for mac_address in stale_mac_addresses:
        rem_device = device_registry.async_get_device(
            connections={(CONNECTION_NETWORK_MAC, mac_address)}
        )
        expected_unique_id = slugify(
            f"{config_entry.data[CONF_DEVICE_UNIQUE_ID]}_mac_{mac_address}"
        )
        if entity_id := entity_registry.async_get_entity_id(
            Platform.DEVICE_TRACKER, DOMAIN, expected_unique_id
        ):
            _LOGGER.debug(
                "[device_tracker async_setup_entry] removing tracker entity_id %s for stale MAC %s",
                entity_id,
                mac_address,
            )
            entity_registry.async_remove(entity_id)

        if rem_device:
            effective_router_device_id: str | None = router_device_id
            if (
                effective_router_device_id is None
                and isinstance(rem_device.via_device_id, str)
                and device_registry.async_get(rem_device.via_device_id) is None
            ):
                effective_router_device_id = rem_device.via_device_id
            _from_current_router, replacement_router_id = detach_shared_router_parent(
                shared_config_entry_id=config_entry.entry_id,
                shared_device_entry=rem_device,
                router_device_id=effective_router_device_id,
                config_entries=hass.config_entries,
                device_registry=device_registry,
            )
            if replacement_router_id is not None:
                _LOGGER.debug(
                    "[device_tracker async_setup_entry] reparenting shared tracker "
                    "device %s from %s to %s",
                    rem_device.id,
                    router_device_id,
                    replacement_router_id,
                )


class OPNsenseScannerEntity(OPNsenseBaseEntity, ScannerEntity, RestoreEntity):
    """Represent a scanned device."""

    def __init__(
        self,
        config_entry: ConfigEntry,
        coordinator: OPNsenseDataUpdateCoordinator,
        enabled_default: bool,
        mac: str,
        mac_vendor: str | None,
        hostname: str | None,
    ) -> None:
        """Set up the OPNsense scanner entity.

        Args:
            config_entry: Config entry owning the entity.
            coordinator: Shared OPNsense data coordinator.
            enabled_default: Whether the entity is enabled by default.
            mac: MAC address tracked by the entity.
            mac_vendor: Vendor name reported for the MAC address.
            hostname: Hostname reported by OPNsense.
        """
        super().__init__(config_entry, coordinator, unique_id_suffix=f"mac_{mac}")
        self._mac_vendor: str | None = mac_vendor
        self._attr_name: str | None = None
        self._last_known_ip: str | None = None
        self._last_known_hostname: str | None = None
        self._is_connected: bool = False
        self._last_known_connected_time: datetime | None = None
        self._attr_entity_registry_enabled_default: bool = enabled_default
        self._attr_hostname: str | None = hostname
        self._attr_ip_address: str | None = None
        self._attr_mac_address: str | None = mac
        self._attr_source_type: SourceType = SourceType.ROUTER
        self._attr_icon: str | None = None
        self._fallback_device_info_consumed: bool = False

    def _has_matching_enabled_mac_device(self) -> bool:
        """Return whether a matching MAC device exists and is not disabled."""
        if self.mac_address is None:
            return False

        hass = getattr(self, "hass", None)
        if hass is None:
            return False

        device_registry = async_get_dev_reg(hass)
        existing_device = device_registry.async_get_device(
            connections={(CONNECTION_NETWORK_MAC, self.mac_address)}
        )
        if existing_device is None:
            return False

        return getattr(existing_device, "disabled_by", None) is None

    @property
    def is_connected(self) -> bool:
        """Return if the tracker is connected.

        Returns:
            ``True`` when the device is currently considered connected.
        """
        return self._is_connected

    @property
    def unique_id(self) -> str | None:
        """Return a stable object-id hint when linking to an existing device.

        Returns:
            The Home Assistant entity unique ID.
        """
        return self._attr_unique_id

    @property
    def suggested_object_id(self) -> str | None:
        """Return a stable object-id hint when linking to an existing device.

        Returns:
            Hostname when available, otherwise MAC, but only for the
            auto-link path (enabled matching MAC + new-entity preference disabled).
        """
        if (
            self._has_matching_enabled_mac_device()
            and not self.config_entry.pref_disable_new_entities
        ):
            if self._attr_hostname is not None:
                return self._attr_hostname
            return self._attr_mac_address
        return None

    @property
    def entity_registry_enabled_default(self) -> bool:
        """Return if the entity registry is enabled by default.

        Returns:
            ``True`` when the entity should be enabled by default.
        """
        if self._attr_entity_registry_enabled_default:
            return True
        if self._fallback_device_info_consumed:
            return False
        return self._has_matching_enabled_mac_device()

    @callback
    def _handle_coordinator_update(self) -> None:
        """Refresh tracker state from the latest ARP table."""
        state: dict[str, Any] = self.coordinator.data
        arp_table = dict_get(state, "arp_table")
        if not isinstance(arp_table, list) or not isinstance(state, MutableMapping):
            self._mark_unavailable()
            return
        self._available = True
        entry: MutableMapping[str, Any] | None = None
        for arp_entry in arp_table:
            if not isinstance(arp_entry, MutableMapping):
                continue
            arp_mac = get_arp_mac(arp_entry)
            if (
                isinstance(self._attr_mac_address, str)
                and self._attr_mac_address.lower() == arp_mac
            ):
                entry = arp_entry
                break
        if not entry:
            entry = {}
        ip_address = get_arp_ip(entry)
        self._attr_ip_address = ip_address if isinstance(ip_address, str) and ip_address else None

        if self._attr_ip_address:
            self._last_known_ip = self._attr_ip_address

        self._attr_hostname = _hostname_from_arp_entry(entry)

        if self._attr_hostname:
            self._last_known_hostname = self._attr_hostname

        if not isinstance(entry, MutableMapping) or not entry or entry.get("expired", False):
            self._is_connected = False
            device_tracker_consider_home = self.config_entry.options.get(
                CONF_DEVICE_TRACKER_CONSIDER_HOME, DEFAULT_DEVICE_TRACKER_CONSIDER_HOME
            )
            if device_tracker_consider_home > 0 and isinstance(
                self._last_known_connected_time, datetime
            ):
                elapsed: timedelta = datetime.now().astimezone() - self._last_known_connected_time
                if elapsed.total_seconds() < device_tracker_consider_home:
                    self._is_connected = True

        else:
            update_time = state.get("update_time")
            if isinstance(update_time, float):
                self._last_known_connected_time = datetime.fromtimestamp(
                    int(update_time),
                    tz=timezone(datetime.now().astimezone().utcoffset() or timedelta()),
                )
            self._is_connected = True

        _update_arp_extra_state_attributes(self._attr_extra_state_attributes, entry)

        if self._attr_hostname is None and self._last_known_hostname:
            self._attr_extra_state_attributes["last_known_hostname"] = self._last_known_hostname
        else:
            self._attr_extra_state_attributes.pop("last_known_hostname", None)

        if self._attr_ip_address is None and self._last_known_ip:
            self._attr_extra_state_attributes["last_known_ip"] = self._last_known_ip
        else:
            self._attr_extra_state_attributes.pop("last_known_ip", None)

        if self._last_known_connected_time is not None:
            self._attr_extra_state_attributes["last_known_connected_time"] = (
                self._last_known_connected_time
            )

        self._attr_icon = "mdi:lan-connect" if self.is_connected else "mdi:lan-disconnect"

        self.async_write_ha_state()

    @property  # type: ignore[misc] # overriding final from ScannerEntity
    def device_info(self) -> DeviceInfo | None:
        """Return device registry metadata for the tracker.

        Home Assistant's ``ScannerEntity`` can automatically attach a scanner
        entity to an existing device when the device registry already has a
        matching network MAC connection. That auto-link path only runs when
        the scanner entity does not provide its own device info.

        When a matching enabled device exists, return ``None`` so the base
        scanner implementation links this tracker to that device, unless
        ``pref_disable_new_entities`` is enabled on the config entry.
        In that preference-enabled case, return device info so the entity is
        linked to an existing device during registry creation while the entity
        remains disabled by preference.

        When no matching enabled device exists, return the historical
        hass-opnsense device info so Home Assistant still creates the manually
        associated tracker device or preserves disabled-device behavior.

        Returns:
            ``None`` for an existing enabled MAC-matched device, except when
            ``pref_disable_new_entities`` is enabled, otherwise fallback
            device registry metadata for the tracked MAC address.
        """
        # Returning None here opts into ScannerEntity's existing-device linking
        # path. Returning DeviceInfo below preserves the previous fallback
        # behavior for trackers whose MAC is not already in the device registry.
        has_matching_enabled_mac_device = self._has_matching_enabled_mac_device()
        if has_matching_enabled_mac_device and not self.config_entry.pref_disable_new_entities:
            return None

        if not has_matching_enabled_mac_device:
            self._fallback_device_info_consumed = True

        connections: set[tuple[str, str]] = set()
        if self.mac_address is not None:
            connections.add((CONNECTION_NETWORK_MAC, self.mac_address))

        return DeviceInfo(
            connections=connections,
            default_manufacturer=self._mac_vendor or "",
            default_name=self.hostname or self.mac_address or "",
            via_device=(DOMAIN, self._device_unique_id),
        )

    async def _restore_last_state(self) -> None:
        """Restore tracker state from Home Assistant's last saved snapshot."""
        last_state = await self.async_get_last_state()
        if last_state is None or last_state.attributes is None:
            return

        state = last_state.attributes
        if not isinstance(state, Mapping):
            return

        self._last_known_hostname = state.get("last_known_hostname", None)
        self._last_known_ip = state.get("last_known_ip", None)

        for attr in ("interface", "expires", "type"):
            value = state.get(attr, None)
            if value:
                self._attr_extra_state_attributes[attr] = value

        lkct = state.get("last_known_connected_time", None)
        parsed_last_known_connected_time: datetime | None = None
        if isinstance(lkct, datetime):
            parsed_last_known_connected_time = lkct
        elif isinstance(lkct, str):
            with contextlib.suppress(ValueError):
                parsed_last_known_connected_time = datetime.fromisoformat(lkct)

        if (
            parsed_last_known_connected_time is not None
            and parsed_last_known_connected_time.tzinfo is not None
            and parsed_last_known_connected_time.utcoffset() is not None
        ):
            self._last_known_connected_time = parsed_last_known_connected_time
            self._attr_extra_state_attributes["last_known_connected_time"] = (
                parsed_last_known_connected_time
            )

    async def async_added_to_hass(self) -> None:
        """Commands to run after entity is created."""
        await self._restore_last_state()
        await super().async_added_to_hass()
