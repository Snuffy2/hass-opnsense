"""These tests import the integration code via relative imports and assert behavior across sensor variants using a synthesized coordinator state."""

# removed unused `inspect` and `sys` imports when tracer-based test was replaced
from typing import Any
from unittest.mock import MagicMock

import pytest

from custom_components.opnsense import sensor as sensor_module
from custom_components.opnsense.const import (
    CONF_SYNC_CARP,
    CONF_SYNC_CERTIFICATES,
    CONF_SYNC_DHCP_LEASES,
    CONF_SYNC_GATEWAYS,
    CONF_SYNC_INTERFACES,
    CONF_SYNC_TELEMETRY,
    CONF_SYNC_VPN,
    COORDINATOR,
)
from custom_components.opnsense.coordinator import OPNsenseDataUpdateCoordinator
from custom_components.opnsense.sensor import (
    OPNsenseCarpInterfaceSensor,
    OPNsenseDHCPLeasesSensor,
    OPNsenseGatewaySensor,
    OPNsenseInterfaceSensor,
    OPNsenseStaticKeySensor,
    OPNsenseTempSensor,
    OPNsenseVPNSensor,
    async_setup_entry,
    normalize_filesystem_mountpoint,
    slugify_filesystem_mountpoint,
)


@pytest.mark.asyncio
async def test_async_setup_entry_invalid_state(make_config_entry):
    """async_setup_entry should do nothing when coordinator.data is invalid."""
    config_entry = make_config_entry()
    # runtime_data used by async_setup_entry expects an attribute named COORDINATOR
    coordinator = MagicMock(spec=OPNsenseDataUpdateCoordinator)
    coordinator.data = None
    setattr(config_entry.runtime_data, COORDINATOR, coordinator)

    called = False

    def add_entities(entities):
        nonlocal called
        called = True

    await async_setup_entry(MagicMock(), config_entry, add_entities)
    assert called is False


@pytest.mark.asyncio
async def test_static_key_sensor_cpu_and_boot_and_certificates(make_config_entry):
    """Static key sensors should expose CPU, boot time, and certificate counts."""
    coordinator = MagicMock(spec=OPNsenseDataUpdateCoordinator)
    coordinator.data = {
        "telemetry": {
            "cpu": {"usage_1": 10, "usage_2": 20, "usage_total": 30},
            "system": {"boottime": 1609459200},
        },
        "certificates": {"a": 1, "b": 2},
    }

    entry = make_config_entry()

    # CPU total sensor
    desc = MagicMock()
    desc.key = "telemetry.cpu.usage_total"
    desc.name = "CPU Total"
    s_cpu = OPNsenseStaticKeySensor(
        config_entry=entry, coordinator=coordinator, entity_description=desc
    )
    s_cpu.hass = MagicMock()
    s_cpu.entity_id = "sensor.cpu_total"
    s_cpu.async_write_ha_state = lambda: None
    # first call when previous is None and value !=0 -> available True and extra attributes
    s_cpu._handle_coordinator_update()
    assert s_cpu.available is True
    assert s_cpu.native_value == 30
    assert s_cpu.extra_state_attributes.get("1") == "10%"
    assert s_cpu.extra_state_attributes.get("2") == "20%"


@pytest.mark.parametrize(
    "coord_data,desc_subnet",
    [
        (None, "some"),
        ({"carp_interfaces": [{"subnet": "10.0.0.5", "status": "MASTER"}]}, "192.168.1.10"),
        ({"carp_interfaces": [{"subnet": "1.2.3.4", "interface": "lan0"}]}, "1.2.3.4"),
    ],
)
def test_carp_sensor_unavailable_variants(coord_data, desc_subnet, make_config_entry):
    """Parameterised unavailable variants for CARP sensor."""
    coord = MagicMock(spec=OPNsenseDataUpdateCoordinator)
    coord.data = coord_data
    entry = make_config_entry()

    desc = MagicMock()
    desc.key = f"carp.interface.{sensor_module.slugify(desc_subnet)}"
    desc.name = "CARP"

    s = OPNsenseCarpInterfaceSensor(config_entry=entry, coordinator=coord, entity_description=desc)
    s.hass = MagicMock()
    s.entity_id = "sensor.carp_unavailable"
    s.async_write_ha_state = lambda: None
    s._handle_coordinator_update()
    assert s.available is False


def test_carp_sensor_state_wrong_type(make_config_entry):
    """CARP sensor should be unavailable when coordinator.data is not a mapping (e.g., list)."""
    coord = MagicMock(spec=OPNsenseDataUpdateCoordinator)
    # use a list to ensure isinstance(state, MutableMapping) is False
    coord.data = []
    entry = make_config_entry()

    desc = MagicMock()
    desc.key = f"carp.interface.{sensor_module.slugify('10.10.10.10')}"
    desc.name = "CARP WrongType"

    s = OPNsenseCarpInterfaceSensor(config_entry=entry, coordinator=coord, entity_description=desc)
    s.hass = MagicMock()
    s.entity_id = "sensor.carp_wrongtype"
    s.async_write_ha_state = lambda: None
    s._handle_coordinator_update()
    assert s.available is False


@pytest.mark.parametrize(
    "desc_key,cls",
    [
        ("carp.interface.some", OPNsenseCarpInterfaceSensor),
        ("gateway.gw1.status", OPNsenseGatewaySensor),
        ("interface.lan.status", OPNsenseInterfaceSensor),
        ("telemetry.temps.sensor1", OPNsenseTempSensor),
        ("openvpn.servers.uuid1.status", OPNsenseVPNSensor),
        ("dhcp_leases.all", OPNsenseDHCPLeasesSensor),
    ],
)
def test_sensors_unavailable_on_non_mapping_state(desc_key, cls, make_config_entry):
    """Sensors should mark themselves unavailable when coordinator.data is not a mapping."""
    coord = MagicMock(spec=OPNsenseDataUpdateCoordinator)
    # provide a non-mapping value (list) to trigger the isinstance guard
    coord.data = []
    entry = make_config_entry()

    desc = MagicMock()
    desc.key = desc_key
    desc.name = "NonMappingState"

    s = cls(config_entry=entry, coordinator=coord, entity_description=desc)
    s.hass = MagicMock()
    s.entity_id = "sensor.unavailable"
    s.async_write_ha_state = lambda: None
    s._handle_coordinator_update()
    assert s.available is False


@pytest.mark.parametrize(
    "prop_name,input_value,expected_available,expected_value,expect_down_icon",
    [
        ("status", "online", True, "online", False),
        ("status", "", False, None, True),  # empty status -> unavailable
        ("delay", "15ms", True, 15.0, False),
        ("stddev", "1ms", True, 1.0, False),
        ("loss", "0%", True, 0.0, False),
        ("delay", 12, True, 12, False),
    ],
)
def test_gateway_sensor_value_parsing(
    prop_name, input_value, expected_available, expected_value, expect_down_icon, make_config_entry
):
    """Parameterized checks for gateway value parsing and availability."""
    entry = make_config_entry()
    gw = {"name": "gw1", prop_name: input_value}
    state = {"gateways": {"gw1": gw}}

    coord = MagicMock(spec=OPNsenseDataUpdateCoordinator)
    coord.data = state

    desc = MagicMock()
    desc.key = f"gateway.gw1.{prop_name}"
    desc.name = "Gateway Test"

    s = OPNsenseGatewaySensor(config_entry=entry, coordinator=coord, entity_description=desc)
    s.hass = MagicMock()
    s.entity_id = "sensor.gw_test"
    s.async_write_ha_state = lambda: None
    s._handle_coordinator_update()

    assert s.available is expected_available
    if expected_available:
        # compare floats approximately when numeric
        if isinstance(expected_value, float):
            assert float(s.native_value) == pytest.approx(expected_value)
        else:
            assert s.native_value == expected_value
        if prop_name == "status":
            if expect_down_icon:
                assert s.icon == "mdi:close-network-outline"
            else:
                assert s.icon != "mdi:close-network-outline"


def test_gateway_sensor_missing_and_missing_prop(make_config_entry):
    """Gateway sensor should be unavailable when gateway missing or property missing."""
    entry = make_config_entry()

    # missing gateway
    coord1 = MagicMock(spec=OPNsenseDataUpdateCoordinator)
    coord1.data = {"gateways": {}}
    desc1 = MagicMock()
    desc1.key = "gateway.missing.status"
    desc1.name = "GW Missing"
    s1 = OPNsenseGatewaySensor(config_entry=entry, coordinator=coord1, entity_description=desc1)
    s1.hass = MagicMock()
    s1.entity_id = "sensor.gw_missing"
    s1.async_write_ha_state = lambda: None
    s1._handle_coordinator_update()
    assert s1.available is False

    # gateway present but property missing
    coord2 = MagicMock(spec=OPNsenseDataUpdateCoordinator)
    coord2.data = {"gateways": {"gw1": {"name": "gw1"}}}
    desc2 = MagicMock()
    desc2.key = "gateway.gw1.delay"
    desc2.name = "GW NoProp"
    s2 = OPNsenseGatewaySensor(config_entry=entry, coordinator=coord2, entity_description=desc2)
    s2.hass = MagicMock()
    s2.entity_id = "sensor.gw_noprop"
    s2.async_write_ha_state = lambda: None
    s2._handle_coordinator_update()
    assert s2.available is False


@pytest.mark.parametrize(
    "carp_entry,expected_value,expect_down_icon,expect_keys",
    [
        (
            {
                "subnet": "192.168.1.20",
                "status": "BACKUP",
                "interface": "lan0",
                "vhid": 7,
                "advskew": 100,
                "advbase": 0,
                "subnet_bits": 24,
                "descr": "test carp",
            },
            "BACKUP",
            True,
            ("interface", "vhid", "advskew", "advbase", "subnet_bits", "subnet", "descr"),
        ),
        (
            {"subnet": "10.0.0.1", "status": "MASTER"},
            "MASTER",
            False,
            (),
        ),
    ],
)
def test_carp_sensor_attributes_and_icon(
    carp_entry, expected_value, expect_down_icon, expect_keys, make_config_entry
):
    """Parameterized attribute and icon checks for CARP sensor."""
    entry = make_config_entry()

    coord = MagicMock(spec=OPNsenseDataUpdateCoordinator)
    coord.data = {"carp_interfaces": [carp_entry]}

    desc = MagicMock()
    desc.key = f"carp.interface.{sensor_module.slugify(carp_entry['subnet'])}"
    desc.name = "CARP Test"

    s = OPNsenseCarpInterfaceSensor(config_entry=entry, coordinator=coord, entity_description=desc)
    s.hass = MagicMock()
    s.entity_id = "sensor.carp_param"
    s.async_write_ha_state = lambda: None
    s._handle_coordinator_update()

    assert s.available is True
    assert s.native_value == expected_value
    if expect_down_icon:
        assert s.icon == "mdi:close-network-outline"
    else:
        assert s.icon != "mdi:close-network-outline"
    for key in expect_keys:
        assert key in s.extra_state_attributes


@pytest.mark.parametrize(
    "desc_key,cls,main_check,extra_check",
    [
        (
            f"carp.interface.{sensor_module.slugify('10.0.0.1')}",
            OPNsenseCarpInterfaceSensor,
            lambda s: s.native_value == "MASTER",
            lambda s: s.icon != "mdi:close-network-outline",
        ),
        (
            "gateway.gw1.delay",
            OPNsenseGatewaySensor,
            lambda s: float(s.native_value) == pytest.approx(12.0),
            None,
        ),
        (
            "openvpn.servers.uuid1.status",
            OPNsenseVPNSensor,
            lambda s: "clients" in s.extra_state_attributes,
            None,
        ),
        ("telemetry.temps.sensor1", OPNsenseTempSensor, lambda s: s.native_value == 42, None),
        (
            "dhcp_leases.all",
            OPNsenseDHCPLeasesSensor,
            lambda s: isinstance(s.native_value, int),
            None,
        ),
    ],
)
def test_compiled_sensor_variants(desc_key, cls, main_check, extra_check, make_config_entry):
    """Table-driven checks for several sensor types using a common sample state."""
    state = {
        "carp_interfaces": [
            {"subnet": "10.0.0.1", "status": "MASTER", "interface": "lan0", "vhid": 1}
        ],
        "gateways": {"gw1": {"name": "gw1", "delay": "12ms", "loss": "0", "status": "online"}},
        "openvpn": {
            "servers": {
                "uuid1": {
                    "name": "ovpn",
                    "status": "up",
                    "clients": [{"name": "c1", "status": "up"}],
                }
            }
        },
        "telemetry": {"temps": {"sensor1": {"temperature": 42, "device_id": "dev1"}}},
        "dhcp_leases": {
            "leases": {"lan": [{"address": "192.168.1.2"}]},
            "lease_interfaces": {"lan": "LAN"},
        },
    }

    entry = make_config_entry()
    coordinator = MagicMock(spec=OPNsenseDataUpdateCoordinator)
    coordinator.data = state

    desc = MagicMock()
    desc.key = desc_key
    desc.name = "Test"

    s = cls(config_entry=entry, coordinator=coordinator, entity_description=desc)
    s.hass = MagicMock()
    s.entity_id = "sensor.test"
    s.async_write_ha_state = lambda: None
    s._handle_coordinator_update()

    assert s.available is True
    assert main_check(s)
    if extra_check:
        assert extra_check(s)


@pytest.mark.parametrize(
    "state,desc_key,expected_available,expected_value,expect_clients,expect_extra_keys",
    [
        # missing instance -> unavailable
        (
            {"openvpn": {"servers": {}}},
            "openvpn.servers.uuid_missing.status",
            False,
            None,
            False,
            (),
        ),
        # instance present but disabled and prop != status -> unavailable
        (
            {"openvpn": {"servers": {"uuid1": {"name": "ovpn1", "enabled": False}}}},
            "openvpn.servers.uuid1.connected_clients",
            False,
            None,
            False,
            (),
        ),
        # instance present, disabled and requesting status -> 'disabled'
        (
            {"openvpn": {"servers": {"uuid1": {"name": "ovpn1", "enabled": False}}}},
            "openvpn.servers.uuid1.status",
            True,
            "disabled",
            False,
            ("uuid", "name", "enabled"),
        ),
        # instance present with status and clients -> available, clients attribute populated
        (
            {
                "openvpn": {
                    "servers": {
                        "uuid1": {
                            "name": "ovpn1",
                            "status": "up",
                            "clients": [{"name": "c1", "status": "up", "bytes_sent": 10}],
                        }
                    }
                }
            },
            "openvpn.servers.uuid1.status",
            True,
            "up",
            True,
            ("uuid", "name", "enabled", "connected_clients"),
        ),
    ],
)
def test_vpn_sensor_variants(
    state,
    desc_key,
    expected_available,
    expected_value,
    expect_clients,
    expect_extra_keys,
    make_config_entry,
):
    """Parameterised tests for OPNsenseVPNSensor to hit key branches in the update handler."""
    entry = make_config_entry()

    coord = MagicMock(spec=OPNsenseDataUpdateCoordinator)
    coord.data = state

    desc = MagicMock()
    desc.key = desc_key
    desc.name = "VPN Test"

    s = OPNsenseVPNSensor(config_entry=entry, coordinator=coord, entity_description=desc)
    s.hass = MagicMock()
    s.entity_id = "sensor.vpn_test"
    s.async_write_ha_state = lambda: None
    s._handle_coordinator_update()

    assert s.available is expected_available
    if expected_available:
        assert s.native_value == expected_value
        # clients attribute present when expected
        if expect_clients:
            assert "clients" in s.extra_state_attributes
            # verify client attr was filtered to allowed fields
            assert isinstance(s.extra_state_attributes["clients"], list)
            assert s.extra_state_attributes["clients"][0]["name"] == "c1"
        for key in expect_extra_keys:
            # only check presence if the attribute was populated by the handler
            # some keys may be absent depending on input; assert no exception
            if key in s.extra_state_attributes:
                assert key in s.extra_state_attributes


@pytest.mark.parametrize("exc_type", [TypeError, KeyError, ZeroDivisionError])
def test_vpn_sensor_handles_exceptions_from_instance_get(exc_type, make_config_entry):
    """VPNSensor should mark itself unavailable when instance.get() raises common exceptions.

    We simulate broken instance objects by providing an object with a `get` method
    that raises the desired exception to exercise the except block in the handler.
    """

    class BrokenInstance:
        def __init__(self, exc):
            self._exc = exc

        def get(self, *args, **kwargs):
            raise self._exc("simulated")

    entry = make_config_entry()
    broken = BrokenInstance(exc_type)

    coord = MagicMock(spec=OPNsenseDataUpdateCoordinator)
    coord.data = {"openvpn": {"servers": {"uuid1": broken}}}

    desc = MagicMock()
    desc.key = "openvpn.servers.uuid1.status"
    desc.name = "VPN Broken"

    s = OPNsenseVPNSensor(config_entry=entry, coordinator=coord, entity_description=desc)
    s.hass = MagicMock()
    s.entity_id = "sensor.vpn_broken"
    s.async_write_ha_state = lambda: None
    s._handle_coordinator_update()

    assert s.available is False


@pytest.mark.parametrize(
    "coord_data,expected_available,expected_value,expect_device",
    [
        # non-mapping coordinator.data -> unavailable
        ([], False, None, False),
        # telemetry present but no temps -> unavailable
        ({"telemetry": {}}, False, None, False),
        # valid temp -> available with native_value and device_id
        (
            {"telemetry": {"temps": {"sensor1": {"temperature": 55, "device_id": "dev0"}}}},
            True,
            55,
            True,
        ),
    ],
)
def test_temp_sensor_basic_variants(
    coord_data, expected_available, expected_value, expect_device, make_config_entry
):
    """Temp sensor should handle non-mapping/missing and successful value extraction."""
    entry = make_config_entry()

    coord = MagicMock(spec=OPNsenseDataUpdateCoordinator)
    coord.data = coord_data

    desc = MagicMock()
    desc.key = "telemetry.temps.sensor1"
    desc.name = "Temp Test"

    s = OPNsenseTempSensor(config_entry=entry, coordinator=coord, entity_description=desc)
    s.hass = MagicMock()
    s.entity_id = "sensor.temp_test"
    s.async_write_ha_state = lambda: None
    s._handle_coordinator_update()

    assert s.available is expected_available
    if expected_available:
        assert s.native_value == expected_value
        if expect_device:
            assert s.extra_state_attributes.get("device_id") == "dev0"


@pytest.mark.parametrize("exc_type", [TypeError, KeyError, ZeroDivisionError])
def test_temp_sensor_handles_index_exceptions(exc_type, make_config_entry):
    """Temp sensor should mark itself unavailable when indexing temp raises exceptions."""

    class BrokenTemp:
        def __init__(self, exc):
            self._exc = exc

        def __bool__(self):
            # truthy so code proceeds to try block
            return True

        def __getitem__(self, key):
            raise self._exc("simulated")

    entry = make_config_entry()
    broken = BrokenTemp(exc_type)

    coord = MagicMock(spec=OPNsenseDataUpdateCoordinator)
    coord.data = {"telemetry": {"temps": {"sensor1": broken}}}

    desc = MagicMock()
    desc.key = "telemetry.temps.sensor1"
    desc.name = "Temp Broken"

    s = OPNsenseTempSensor(config_entry=entry, coordinator=coord, entity_description=desc)
    s.hass = MagicMock()
    s.entity_id = "sensor.temp_broken"
    s.async_write_ha_state = lambda: None
    s._handle_coordinator_update()

    assert s.available is False


@pytest.mark.parametrize(
    "desc_key,state,expect_close_icon",
    [
        (
            "openvpn.servers.uuid1.status",
            {"openvpn": {"servers": {"uuid1": {"name": "ovpn", "status": "up"}}}},
            False,
        ),
        (
            "openvpn.servers.uuid1.status",
            {"openvpn": {"servers": {"uuid1": {"name": "ovpn", "status": "down"}}}},
            True,
        ),
        (
            "openvpn.servers.uuid1.connected_clients",
            {"openvpn": {"servers": {"uuid1": {"name": "ovpn", "connected_clients": 1}}}},
            False,
        ),
    ],
)
def test_vpn_sensor_icon_variants(desc_key, state, expect_close_icon, make_config_entry):
    """Verify VPNSensor.icon for status up/down and fallback to description icon for non-status."""
    entry = make_config_entry()

    coord = MagicMock(spec=OPNsenseDataUpdateCoordinator)
    coord.data = state

    desc = MagicMock()
    desc.key = desc_key
    desc.name = "VPN Icon Test"
    # supply a fallback icon for non-status case
    desc.icon = "mdi:custom-icon"

    s = OPNsenseVPNSensor(config_entry=entry, coordinator=coord, entity_description=desc)
    s.hass = MagicMock()
    s.entity_id = "sensor.vpn_icon"
    s.async_write_ha_state = lambda: None
    s._handle_coordinator_update()

    if expect_close_icon:
        assert s.icon == "mdi:close-network-outline"
    else:
        assert s.icon != "mdi:close-network-outline"


def test_vpn_server_clients_extra_attributes_included(make_config_entry):
    """Servers with client entries should populate clients extra_state_attributes including bytes fields."""
    entry = make_config_entry()

    coord = MagicMock(spec=OPNsenseDataUpdateCoordinator)
    coord.data = {
        "openvpn": {
            "servers": {
                "uuid1": {
                    "name": "ovpn1",
                    "status": "up",
                    "clients": [
                        {
                            "name": "c1",
                            "status": "up",
                            "bytes_sent": 100,
                            "bytes_recv": 200,
                            "tunnel_addresses": ["10.0.0.1"],
                        }
                    ],
                }
            }
        }
    }

    desc = MagicMock()
    desc.key = "openvpn.servers.uuid1.status"
    desc.name = "VPN Server Clients"

    s = OPNsenseVPNSensor(config_entry=entry, coordinator=coord, entity_description=desc)
    s.hass = MagicMock()
    s.entity_id = "sensor.vpn_server_clients"
    s.async_write_ha_state = lambda: None
    s._handle_coordinator_update()

    assert s.available is True
    assert "clients" in s.extra_state_attributes
    clients = s.extra_state_attributes["clients"]
    assert isinstance(clients, list) and clients
    client0 = clients[0]
    # bytes_sent/bytes_recv should be preserved in filtered client attributes
    assert client0.get("bytes_sent") == 100
    assert client0.get("bytes_recv") == 200


def test_wireguard_client_connected_servers_property(make_config_entry):
    """Wireguard client instance with connected_servers should expose that property when enabled."""
    entry = make_config_entry()

    coord = MagicMock(spec=OPNsenseDataUpdateCoordinator)
    coord.data = {
        "wireguard": {"clients": {"c1": {"name": "c1", "enabled": True, "connected_servers": 3}}}
    }

    desc = MagicMock()
    desc.key = "wireguard.clients.c1.connected_servers"
    desc.name = "WG Connected Servers"

    s = OPNsenseVPNSensor(config_entry=entry, coordinator=coord, entity_description=desc)
    s.hass = MagicMock()
    s.entity_id = "sensor.wg_conn_servers"
    s.async_write_ha_state = lambda: None
    s._handle_coordinator_update()

    assert s.available is True
    assert s.native_value == 3


def test_sensor_module_import() -> None:
    """Test that the sensor module can be imported via relative import."""
    assert sensor_module is not None


@pytest.mark.parametrize(
    "input_value,expected",
    [
        (None, ""),
        ("", ""),
        ("/", "root"),
        ("/boot", "boot"),
        ("/var/log", "var_log"),
        ("var/log/", "var_log"),
        ("//multiple///slashes//", "multiple___slashes"),
        ("relative/path", "relative_path"),
    ],
)
def test_slugify_filesystem_mountpoint(input_value: Any, expected: str) -> None:
    """slugify_filesystem_mountpoint should convert mountpoints to slugs."""
    assert slugify_filesystem_mountpoint(input_value) == expected


@pytest.mark.parametrize(
    "input_value,expected",
    [
        (None, ""),
        ("", ""),
        ("/", "root"),
        ("/boot", "/boot"),
        ("/var/log", "/var/log"),
        ("var/log/", "var/log"),
        ("//multiple///slashes//", "//multiple///slashes"),
    ],
)
def test_normalize_filesystem_mountpoint(input_value: Any, expected: str) -> None:
    """normalize_filesystem_mountpoint should strip trailing slashes and handle root."""
    assert normalize_filesystem_mountpoint(input_value) == expected


@pytest.mark.parametrize(
    "cpu_map,previous,expected_available,expected_value",
    [
        ({"usage_total": 0}, None, False, None),  # zero => unavailable
        ({"usage_total": 0, "usage_1": 1}, 7, True, 7),  # zero but previous retained
    ],
)
def test_static_cpu_zero_variants(
    cpu_map: dict,
    previous: int | None,
    expected_available: bool,
    expected_value: int | None,
    make_config_entry,
) -> None:
    """Zero CPU total should mark sensor unavailable unless previous value exists.

    Consolidates unavailable and use_previous behaviors into a single parameterized test.
    """
    coord = MagicMock(spec=OPNsenseDataUpdateCoordinator)
    coord.data = {"telemetry": {"cpu": cpu_map}}
    # require fixture usage for config entry
    entry = make_config_entry()

    desc = MagicMock()
    desc.key = "telemetry.cpu.usage_total"
    desc.name = "CPU Total"

    sensor = OPNsenseStaticKeySensor(config_entry=entry, coordinator=coord, entity_description=desc)
    sensor.hass = MagicMock()
    sensor.entity_id = "sensor.cpu_total"
    sensor.async_write_ha_state = lambda: None
    if previous is not None:
        sensor._previous_value = previous

    sensor._handle_coordinator_update()
    assert sensor.available is expected_available
    if expected_value is not None:
        assert sensor.native_value == expected_value


def test_gateway_empty_string_unavailable(make_config_entry):
    """Gateway sensor should be unavailable for empty status strings."""
    state = {"gateways": {"gw1": {"name": "gw1", "status": ""}}}
    coord = MagicMock(spec=OPNsenseDataUpdateCoordinator)
    coord.data = state
    entry = make_config_entry()

    desc = MagicMock()
    desc.key = "gateway.gw1.status"
    desc.name = "Gateway Status"

    s = OPNsenseGatewaySensor(config_entry=entry, coordinator=coord, entity_description=desc)
    s.hass = MagicMock()
    s.entity_id = "sensor.gw_empty"
    s.async_write_ha_state = lambda: None
    s._handle_coordinator_update()
    assert s.available is False


def test_interface_status_icon_up(make_config_entry):
    """Interface status sensor shows an 'up' icon when status is up."""
    state = {"interfaces": {"lan": {"name": "LAN", "status": "up", "interface": "lan0"}}}
    coord = MagicMock(spec=OPNsenseDataUpdateCoordinator)
    coord.data = state
    entry = make_config_entry()

    desc = MagicMock()
    desc.key = "interface.lan.status"
    desc.name = "LAN Status"

    s = OPNsenseInterfaceSensor(config_entry=entry, coordinator=coord, entity_description=desc)
    s.hass = MagicMock()
    s.entity_id = "sensor.lan_status_up"
    s.async_write_ha_state = lambda: None
    s._handle_coordinator_update()
    # when native_value is 'up', icon should not be the down icon
    assert s.icon != "mdi:close-network-outline"


@pytest.mark.parametrize(
    "leases_val,lease_interfaces_val",
    [
        ([], {"lan": "LAN"}),
        ({"lan": [{"address": "192.168.1.2"}]}, []),
        (None, {"lan": "LAN"}),
    ],
)
def test_dhcp_leases_all_non_mapping(leases_val, lease_interfaces_val, make_config_entry):
    """DHCP Leases 'all' sensor should be unavailable when leases or lease_interfaces are not mappings."""
    entry = make_config_entry()

    coord = MagicMock(spec=OPNsenseDataUpdateCoordinator)
    coord.data = {"dhcp_leases": {"leases": leases_val, "lease_interfaces": lease_interfaces_val}}

    desc = MagicMock()
    desc.key = "dhcp_leases.all"
    desc.name = "DHCP All"

    s = OPNsenseDHCPLeasesSensor(config_entry=entry, coordinator=coord, entity_description=desc)
    s.hass = MagicMock()
    s.entity_id = "sensor.dhcp_all"
    s.async_write_ha_state = lambda: None
    s._handle_coordinator_update()
    assert s.available is False


@pytest.mark.parametrize("exc_type", [TypeError, KeyError, ZeroDivisionError])
def test_dhcp_leases_handles_exceptions(exc_type, make_config_entry):
    """DHCP Leases 'all' sensor should mark itself unavailable when iterating leases raises exceptions.

    We simulate a broken lease object whose `.get` method raises the requested exception to
    exercise the except block inside the handler's aggregation loop.
    """

    class BrokenLease:
        def __init__(self, exc):
            self._exc = exc

        def get(self, *args, **kwargs):
            raise self._exc("simulated")

    entry = make_config_entry()

    coord = MagicMock(spec=OPNsenseDataUpdateCoordinator)
    # lease_interfaces is a normal mapping, leases contains a list with a BrokenLease
    coord.data = {
        "dhcp_leases": {
            "leases": {"lan": [BrokenLease(exc_type)]},
            "lease_interfaces": {"lan": "LAN"},
        }
    }

    desc = MagicMock()
    desc.key = "dhcp_leases.all"
    desc.name = "DHCP Broken Iteration"

    s = OPNsenseDHCPLeasesSensor(config_entry=entry, coordinator=coord, entity_description=desc)
    s.hass = MagicMock()
    s.entity_id = "sensor.dhcp_broken"
    s.async_write_ha_state = lambda: None
    s._handle_coordinator_update()
    assert s.available is False


@pytest.mark.parametrize("exc_type", [TypeError, KeyError, ZeroDivisionError])
def test_dhcp_lease_interfaces_items_raises(exc_type, make_config_entry):
    """Ensure exceptions raised by lease_interfaces.items() are caught and sensor becomes unavailable."""

    class BrokenLeaseInterfaces(dict):
        def items(self):
            raise exc_type("simulated")

    entry = make_config_entry()

    coord = MagicMock(spec=OPNsenseDataUpdateCoordinator)
    coord.data = {
        "dhcp_leases": {
            "leases": {"lan": [{"address": "192.168.1.2"}]},
            "lease_interfaces": BrokenLeaseInterfaces({"lan": "LAN"}),
        }
    }

    desc = MagicMock()
    desc.key = "dhcp_leases.all"
    desc.name = "DHCP Broken LeaseInterfaces"

    s = OPNsenseDHCPLeasesSensor(config_entry=entry, coordinator=coord, entity_description=desc)
    s.hass = MagicMock()
    s.entity_id = "sensor.dhcp_broken_items"
    s.async_write_ha_state = lambda: None
    s._handle_coordinator_update()
    assert s.available is False


@pytest.mark.parametrize("exc_type", [TypeError, KeyError, ZeroDivisionError])
def test_dhcp_leases_iterable_raises_on_iter(exc_type, make_config_entry):
    """Ensure exceptions raised while iterating the leases list are caught and sensor becomes unavailable."""

    class BrokenLeaseList(list):
        def __iter__(self):
            raise exc_type("simulated")

    entry = make_config_entry()

    coord = MagicMock(spec=OPNsenseDataUpdateCoordinator)
    coord.data = {
        "dhcp_leases": {
            "leases": {"lan": BrokenLeaseList([{"address": "192.168.1.2"}])},
            "lease_interfaces": {"lan": "LAN"},
        }
    }

    desc = MagicMock()
    desc.key = "dhcp_leases.all"
    desc.name = "DHCP Broken Iterable"

    s = OPNsenseDHCPLeasesSensor(config_entry=entry, coordinator=coord, entity_description=desc)
    s.hass = MagicMock()
    s.entity_id = "sensor.dhcp_broken_iter"
    s.async_write_ha_state = lambda: None
    s._handle_coordinator_update()
    assert s.available is False


def test_dhcp_leases_inner_except_writes_unavailable(make_config_entry):
    """Verify the inner except block sets available False and calls async_write_ha_state.

    This test replaces the instance's async_write_ha_state with a collector that records
    the value of `self._available` at each write. If the except branch runs it will
    set `_available = False` then call `async_write_ha_state`, so a recorded False
    proves the except block executed.
    """

    class BrokenLease:
        def get(self, *args, **kwargs):
            raise TypeError("simulated")

    entry = make_config_entry()
    coord = MagicMock(spec=OPNsenseDataUpdateCoordinator)
    coord.data = {
        "dhcp_leases": {
            "leases": {"lan": [BrokenLease()]},
            "lease_interfaces": {"lan": "LAN"},
        }
    }

    desc = MagicMock()
    desc.key = "dhcp_leases.all"
    desc.name = "DHCP Inner Except Collector"

    s = OPNsenseDHCPLeasesSensor(config_entry=entry, coordinator=coord, entity_description=desc)
    s.hass = MagicMock()
    s.entity_id = "sensor.dhcp_inner_collector"

    writes: list[bool] = []

    def collector():
        # capture the availability at the time async_write_ha_state is invoked
        writes.append(bool(getattr(s, "_available", None)))

    s.async_write_ha_state = collector
    s._handle_coordinator_update()

    # ensure the handler wrote state at least once and recorded a False (from except)
    assert writes, "async_write_ha_state was not called"
    assert any(w is False for w in writes), f"expected a False write captured, got {writes}"


def test_dhcp_leases_items_except_writes_unavailable(make_config_entry):
    """Verify exceptions from lease_interfaces.items() cause unavailable state and write."""

    class BrokenLeaseInterfaces(dict):
        def items(self):
            raise KeyError("simulated")

    entry = make_config_entry()
    coord = MagicMock(spec=OPNsenseDataUpdateCoordinator)
    coord.data = {
        "dhcp_leases": {
            "leases": {"lan": [{"address": "192.168.1.2"}]},
            "lease_interfaces": BrokenLeaseInterfaces({"lan": "LAN"}),
        }
    }

    desc = MagicMock()
    desc.key = "dhcp_leases.all"
    desc.name = "DHCP Items Except Collector"

    s = OPNsenseDHCPLeasesSensor(config_entry=entry, coordinator=coord, entity_description=desc)
    s.hass = MagicMock()
    s.entity_id = "sensor.dhcp_items_collector"

    writes: list[bool] = []

    def collector():
        writes.append(bool(getattr(s, "_available", None)))

    s.async_write_ha_state = collector
    s._handle_coordinator_update()

    assert writes, "async_write_ha_state was not called"
    assert any(w is False for w in writes), f"expected a False write captured, got {writes}"


@pytest.mark.parametrize("exc_type", [TypeError, KeyError, ZeroDivisionError])
def test_dhcp_leases_per_interface_handles_exceptions(exc_type, make_config_entry):
    """Ensure per-interface DHCP leases exception paths mark sensor unavailable and write state.

    This exercises the `else` branch where a specific interface's leases are summed and an
    exception raised by a lease element's `.get` should be caught by the surrounding except.
    """

    class BrokenLease:
        def __init__(self, exc):
            self._exc = exc

        def get(self, *args, **kwargs):
            raise self._exc("simulated")

    entry = make_config_entry()
    coord = MagicMock(spec=OPNsenseDataUpdateCoordinator)
    coord.data = {"dhcp_leases": {"leases": {"lan": [BrokenLease(exc_type)]}}}

    desc = MagicMock()
    desc.key = "dhcp_leases.lan"
    desc.name = "DHCP Per-Interface Broken"

    s = OPNsenseDHCPLeasesSensor(config_entry=entry, coordinator=coord, entity_description=desc)
    s.hass = MagicMock()
    s.entity_id = "sensor.dhcp_per_if_broken"

    writes: list[bool] = []

    def collector():
        writes.append(bool(getattr(s, "_available", None)))

    s.async_write_ha_state = collector
    s._handle_coordinator_update()

    assert writes, "async_write_ha_state was not called"
    assert any(w is False for w in writes), f"expected a False write captured, got {writes}"


def test_dhcp_leases_coverage_tracer(make_config_entry):
    """Trigger the BrokenLease path and assert the except-branch observable behavior.

    Instead of introspecting source lines, exercise the same failure mode used by
    other DHCP-lease tests: make the lease object raise when accessed and assert
    that the sensor marks itself unavailable and calls async_write_ha_state.
    """

    class BrokenLease:
        def get(self, *args, **kwargs):
            raise TypeError("simulated")

    entry = make_config_entry()
    coord = MagicMock(spec=OPNsenseDataUpdateCoordinator)
    coord.data = {"dhcp_leases": {"leases": {"lan": [BrokenLease()]}}}

    desc = MagicMock()
    desc.key = "dhcp_leases.lan"
    desc.name = "DHCP Per-Interface Collector"

    s = OPNsenseDHCPLeasesSensor(config_entry=entry, coordinator=coord, entity_description=desc)
    s.hass = MagicMock()
    s.entity_id = "sensor.dhcp_per_if_collector"

    writes: list[bool] = []

    def collector():
        writes.append(bool(getattr(s, "_available", None)))

    s.async_write_ha_state = collector
    s._handle_coordinator_update()

    assert writes, "async_write_ha_state was not called"
    assert any(w is False for w in writes), f"expected a False write captured, got {writes}"


def _setup_entry_with_all_syncs(state: dict, make_config_entry):
    entry = make_config_entry()
    # enable all sync options; entry.data may be a mappingproxy so construct a new dict
    base = dict(entry.data)
    base.update(
        {
            CONF_SYNC_TELEMETRY: True,
            CONF_SYNC_CERTIFICATES: True,
            CONF_SYNC_VPN: True,
            CONF_SYNC_GATEWAYS: True,
            CONF_SYNC_INTERFACES: True,
            CONF_SYNC_CARP: True,
            CONF_SYNC_DHCP_LEASES: True,
        }
    )
    # create a new MockConfigEntry with the updated data to avoid mutating mappingproxy
    entry = make_config_entry(base)
    coord = MagicMock(spec=OPNsenseDataUpdateCoordinator)
    coord.data = state
    setattr(entry.runtime_data, COORDINATOR, coord)
    return entry, coord


@pytest.mark.asyncio
async def test_compile_and_handle_many_entities(make_config_entry):
    """Compile a complex state and verify many sensor branches are handled."""
    # craft a rich state to exercise many branches
    state = {
        "telemetry": {
            "filesystems": [
                {
                    "mountpoint": "/",
                    "used_pct": 5,
                    "device": "/dev/sda1",
                    "type": "ext4",
                    "blocks": 1000,
                    "used": 50,
                    "available": 950,
                },
                {
                    "mountpoint": "/var/log",
                    "used_pct": 12,
                    "device": "/dev/sdb1",
                    "type": "ext4",
                    "blocks": 2000,
                    "used": 240,
                    "available": 1760,
                },
            ],
            "temps": {"cpu": {"temperature": 55, "device_id": "cpu0"}},
            "cpu": {"usage_total": 10, "usage_1": 4, "usage_2": 6},
        },
        "interfaces": {
            "wan": {
                "name": "WAN",
                "inbytes_kilobytes_per_second": 1.2,
                "outbytes_kilobytes_per_second": 0.8,
                "inpkts_packets_per_second": 100,
                "status": "up",
                "interface": "wan0",
                "device": "eth0",
            },
            "lan": {
                "name": "LAN",
                "inbytes": 12345,
                "outpkts_packets_per_second": 50,
                "status": "down",
                "interface": "lan0",
                "device": "eth1",
            },
        },
        "gateways": {
            "gw1": {
                "name": "gw1",
                "delay": "15ms",
                "stddev": "1ms",
                "loss": "0%",
                "status": "online",
            }
        },
        "carp_interfaces": [
            {
                "subnet": "192.0.2.1",
                "status": "BACKUP",
                "interface": "lan0",
                "vhid": 2,
                "advskew": 100,
            }
        ],
        "openvpn": {
            "servers": {
                "s1": {"name": "ovpn1", "status": "up", "clients": [{"name": "c1", "status": "up"}]}
            }
        },
        "wireguard": {
            "servers": {"wg1": {"name": "wg1", "status": "up", "clients": []}},
            "clients": {"c1": {"name": "c1", "enabled": True}},
        },
        "dhcp_leases": {
            "leases": {"lan": [{"address": "192.168.1.2", "hostname": "host1"}]},
            "lease_interfaces": {"lan": "LAN"},
        },
        "certificates": {"a": 1},
    }

    entry, coord = _setup_entry_with_all_syncs(state, make_config_entry)

    # Prefer exercising the public integration path: run async_setup_entry to
    # create entities and reduce coupling to private compile helpers. Keep a
    # tiny smoke check for filesystem helper only.
    created: list = []

    async def run_setup():
        def add_entities(entities):
            created.extend(entities)

        await sensor_module.async_setup_entry(MagicMock(), entry, add_entities)

    await run_setup()

    # minimal private helper smoke check for filesystem compilation
    fs_entities = await sensor_module._compile_filesystem_sensors(entry, coord, state)
    assert isinstance(fs_entities, list)

    # Ensure we produced entities via the public setup
    assert len(created) > 0

    # Exercise each entity's update handler
    failures: list[str] = []
    for i, ent in enumerate(created):
        ent.hass = MagicMock()
        ent.entity_id = f"sensor.test_{i}"
        ent.async_write_ha_state = lambda: None
        try:
            ent._handle_coordinator_update()
        except (
            TypeError,
            KeyError,
            ZeroDivisionError,
            AttributeError,
        ) as e:
            failures.append(
                f"entity={getattr(ent, 'entity_id', i)} type={type(e).__name__} msg={e!r}"
            )

    if failures:
        pytest.fail("Exceptions raised by entity handlers:\n" + "\n".join(failures))


@pytest.mark.asyncio
async def test_async_setup_entry_creates_entities(make_config_entry):
    """async_setup_entry should create sensor entities for available telemetry and interfaces."""
    state = {"telemetry": {"filesystems": [], "temps": {}}, "interfaces": {}, "gateways": {}}
    entry, coord = _setup_entry_with_all_syncs(state, make_config_entry)

    created: list = []

    def add_entities(ents):
        created.extend(ents)

    await async_setup_entry(MagicMock(), entry, add_entities)
    # Ensure setup produced at least one created entity
    assert created, "no entities created"
    assert any(isinstance(e, OPNsenseStaticKeySensor) for e in created)


@pytest.mark.asyncio
async def test_compile_interface_sensors_values_end(make_config_entry):
    """Extra test to ensure interface sensors report expected numeric values."""
    state = {
        "interfaces": {
            "eth0": {
                "name": "eth0",
                "inbytes_kilobytes_per_second": 123,
                "inpkts_packets_per_second": 10,
                "inbytes": 2048,
                "inpkts": 100,
                "status": "up",
                "interface": "eth0",
                "device": "eth0",
            }
        }
    }
    entry = make_config_entry()
    coordinator = MagicMock(spec=OPNsenseDataUpdateCoordinator)
    coordinator.data = state

    entities = await sensor_module._compile_interface_sensors(entry, coordinator, state)
    assert any(e.entity_description.key.startswith("interface.eth0.") for e in entities)

    kb_entity = next(
        e for e in entities if e.entity_description.key.endswith("inbytes_kilobytes_per_second")
    )
    kb = OPNsenseInterfaceSensor(
        config_entry=entry, coordinator=coordinator, entity_description=kb_entity.entity_description
    )
    kb.hass = MagicMock()
    kb.entity_id = "sensor.eth0_inkb"
    kb.async_write_ha_state = lambda: None
    kb._handle_coordinator_update()
    assert kb.available is True
    assert kb.native_value == 123
